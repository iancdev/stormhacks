"""Epoch-boundary resumable training, evaluation, and portable weight export."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import warnings
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from forza_ai.data.dataset import SteeringDataset, estimate_cache_bytes
from forza_ai.data.cache import PreprocessingCache
from forza_ai.data.sessions import Alignment, load_sessions, split_sessions
from forza_ai.policies.steering_model import Preprocessing, SteeringModel

ARCHITECTURE = 'pilotnet_speed_v1'
FORMAT_VERSION = 1


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    validation_fraction: float = 0.25
    seed: int = 7
    workers: int = 0
    cache_mib: int = 512

    def __post_init__(self):
        if type(self.cache_mib) is not int or self.cache_mib < 0:
            raise ValueError('cache_mib must be a nonnegative integer')
        if self.batch_size < 1 or self.workers < 0:
            raise ValueError('batch_size must be positive and workers nonnegative')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if not 0 < self.validation_fraction < 1:
            raise ValueError('validation_fraction must be between 0 and 1')


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type not in {'cuda', 'cpu'}:
        raise ValueError('supported training devices: cpu, cuda, auto')
    return device


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('format_version') != FORMAT_VERSION or value.get('architecture') != ARCHITECTURE:
        raise ValueError('unsupported checkpoint format or architecture')
    Preprocessing(**value['preprocessing'])
    return value


def _recover_best(saved, checkpoint_path):
    """The last checkpoint is authoritative; best.pt is a repairable projection."""
    snapshot = saved.get('best_checkpoint')
    if snapshot is None:
        # Compatibility with initial v1 files, and with resuming best.pt itself.
        if saved['history'][-1]['validation']['model']['rmse_deg'] == saved['best_rmse_deg']:
            snapshot = {key: value for key, value in saved.items() if key != 'best_checkpoint'}
        else:
            best_path = Path(checkpoint_path).parent / 'best.pt'
            if not best_path.is_file():
                raise ValueError('legacy checkpoint has no recoverable best.pt; restore the original best checkpoint')
            snapshot = load_checkpoint(best_path)
    if (snapshot['history'][-1]['validation']['model']['rmse_deg'] != saved['best_rmse_deg']
            or snapshot['dataset_fingerprints'] != saved['dataset_fingerprints']
            or snapshot['train_sessions'] != saved['train_sessions']
            or snapshot['validation_sessions'] != saved['validation_sessions']
            or snapshot['preprocessing'] != saved['preprocessing']
            or snapshot['epoch'] > saved['epoch']):
        raise ValueError('best checkpoint does not match resumed run')
    return deepcopy({key: value for key, value in snapshot.items() if key != 'best_checkpoint'})


def _metrics(predictions, targets, mean):
    p, y = np.asarray(predictions), np.asarray(targets)
    if not len(y):
        raise ValueError('cannot evaluate zero accepted samples')
    def errors(values):
        error = values - y
        return {'mae_deg': float(np.abs(error).mean()), 'rmse_deg': float(np.sqrt((error ** 2).mean()))}
    return {'samples': len(y), 'model': errors(p), 'zero_baseline': errors(np.zeros_like(y)),
            'train_mean_baseline': errors(np.full_like(y, mean))}


def _admit_cache(sessions, preprocessing, alignment, cache_mib, workers=0):
    budget = cache_mib * 1024 * 1024
    required = None
    reason = 'disabled_by_request'
    if budget and workers:
        reason = 'loader_workers'
        warnings.warn('preprocessing cache disabled with workers > 0 to avoid worker copies', RuntimeWarning)
    elif budget:
        try:
            required = estimate_cache_bytes(sessions, preprocessing, alignment)
        except (MemoryError, RuntimeError) as error:
            if not PreprocessingCache._is_oom(error):
                raise
            reason = 'estimate_allocation_failed'
        else:
            if type(required) is not int or required < 0:
                reason = 'unknown_working_set'
            elif required > budget:
                reason = 'working_set_exceeds_budget'
            else:
                reason = 'fits_budget'
    enabled = reason == 'fits_budget'
    cache = None
    if enabled:
        try:
            cache = PreprocessingCache(budget)
        except (MemoryError, RuntimeError) as error:
            if not PreprocessingCache._is_oom(error):
                raise
            enabled, reason = False, 'cache_initialization_failed'
    print(json.dumps({'preprocessing_cache': {
        'enabled': enabled, 'reason': reason, 'budget_bytes': budget,
        'required_bytes': required,
    }}), file=sys.stderr)
    return cache


def evaluate_model(model, sessions, preprocessing, mean, device, batch_size=64, cache=None, alignment=None):
    model.eval()
    all_predictions, all_targets, by_session = [], [], {}
    with torch.inference_mode():
        for session in sessions:
            predictions, targets = [], []
            loader = DataLoader(SteeringDataset([session], preprocessing, cache, alignment), batch_size=batch_size)
            for image, speed, label in loader:
                prediction = model(image.to(device), speed.to(device))
                predictions.extend((prediction.cpu() * preprocessing.angle_scale_deg).tolist())
                targets.extend((label * preprocessing.angle_scale_deg).tolist())
            by_session[session.session_id] = _metrics(predictions, targets, mean)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    result = _metrics(all_predictions, all_targets, mean)
    result['sessions'] = by_session
    return result


def train(data, output, epochs=1, config=None, alignment=None, preprocessing=None,
          device='auto', resume=None):
    if epochs < 1:
        raise ValueError('epochs must be >=1 (total target epoch count)')
    output = Path(output)
    if resume is None and output.exists() and any(output.iterdir()):
        raise ValueError('new training output directory must be empty; use resume to continue')
    if resume and output.resolve() != Path(resume).resolve().parent:
        raise ValueError('resume must write into its checkpoint directory; copy the whole run to relocate it')
    saved = load_checkpoint(resume) if resume else None
    if saved:
        config = TrainConfig(**saved['train_config'])
        alignment = Alignment(**saved['alignment'])
        preprocessing = Preprocessing(**saved['preprocessing'])
    else:
        config = config or TrainConfig()
        alignment = alignment or Alignment()
        preprocessing = preprocessing or Preprocessing()
    sessions = load_sessions(data, alignment)
    fingerprints = {s.session_id: s.fingerprint for s in sessions}
    if saved:
        if fingerprints != saved['dataset_fingerprints']:
            raise ValueError('resume dataset changed; restore the exact sessions or start a new run')
        train_sessions = [s for s in sessions if s.session_id in saved['train_sessions']]
        val_sessions = [s for s in sessions if s.session_id in saved['validation_sessions']]
    else:
        train_sessions, val_sessions = split_sessions(sessions, config.validation_fraction, config.seed)
    # Stable order independent of where an archived dataset was extracted.
    train_sessions.sort(key=lambda s: s.session_id)
    val_sessions.sort(key=lambda s: s.session_id)
    device = choose_device(device)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = SteeringModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start, history, best = 0, [], float('inf')
    best_snapshot = None
    if saved:
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        start, history, best = saved['epoch'], saved['history'], saved['best_rmse_deg']
        torch.set_rng_state(saved['torch_rng_state'])
        if device.type == 'cuda' and saved.get('cuda_rng_state'):
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    if epochs <= start:
        raise ValueError(f'checkpoint already completed {start} epochs; target epochs must be greater')
    if saved:
        best_snapshot = _recover_best(saved, resume)
        atomic_save(best_snapshot, output / 'best.pt')
    # Never pickle/fork cache contents into per-worker copies. One budget serves
    # both training and validation when loading in the parent process.
    cache = _admit_cache(train_sessions + val_sessions, preprocessing, alignment,
                         config.cache_mib, config.workers)
    dataset = SteeringDataset(train_sessions, preprocessing, cache, alignment)
    mean = float(np.mean([s.angle_deg for session in train_sessions for s in session.samples]))
    for epoch in range(start, epochs):
        # Epoch-derived shuffling gives repeatable continuation, without skipping batches.
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.workers, generator=generator)
        model.train()
        total, count = 0.0, 0
        for image, speed, label in loader:
            image, speed, label = image.to(device), speed.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image, speed)
            loss = torch.nn.functional.mse_loss(prediction, label)
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(label)
            count += len(label)
        metrics = evaluate_model(model, val_sessions, preprocessing, mean, device, config.batch_size, cache, alignment)
        rmse = metrics['model']['rmse_deg']
        improved = rmse < best
        best = min(best, rmse)
        history.append({'epoch': epoch + 1, 'train_mse_normalized': total / count, 'validation': metrics})
        checkpoint = {
            'format_version': FORMAT_VERSION, 'architecture': ARCHITECTURE,
            'epoch': epoch + 1, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'preprocessing': preprocessing.to_dict(),
            'alignment': asdict(alignment), 'train_config': asdict(config),
            'train_sessions': [s.session_id for s in train_sessions],
            'validation_sessions': [s.session_id for s in val_sessions],
            'dataset_fingerprints': fingerprints, 'train_mean_angle_deg': mean,
            'dataset_groups': {s.session_id: s.group for s in sessions},
            'dataset_provenance': {s.session_id: s.provenance for s in sessions},
            'best_rmse_deg': best, 'history': history,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
        }
        if improved:
            # state_dict tensors alias live weights; freeze them before the next epoch.
            best_snapshot = deepcopy(checkpoint)
        checkpoint['best_checkpoint'] = best_snapshot
        # Publish recoverable state first. A crash before best.pt is published is
        # repaired from this snapshot at resume, even when later epochs worsen.
        atomic_save(checkpoint, output / 'last.pt')
        atomic_save(best_snapshot, output / 'best.pt')
        print(json.dumps(history[-1]), flush=True)
    return checkpoint


def evaluate(checkpoint, data, device='cpu', unseen=False, cache_mib=0):
    if type(cache_mib) is not int or cache_mib < 0:
        raise ValueError('cache_mib must be a nonnegative integer')
    saved = load_checkpoint(checkpoint)
    sessions = load_sessions(data, Alignment(**saved['alignment']))
    if unseen:
        used_groups = set(saved.get('dataset_groups', {key: key for key in saved['dataset_fingerprints']}).values())
        if any(s.session_id in saved['dataset_fingerprints'] or s.group in used_groups for s in sessions):
            raise ValueError('--unseen requires entirely new session IDs and recording groups')
    else:
        selected = {s.session_id: s for s in sessions}
        if not set(saved['validation_sessions']) <= selected.keys():
            raise ValueError('missing held-out validation sessions; use --unseen for a new dataset')
        sessions = [selected[key] for key in saved['validation_sessions']]
        if any(s.fingerprint != saved['dataset_fingerprints'][s.session_id] for s in sessions):
            raise ValueError('held-out validation data changed')
    device = choose_device(device)
    model = SteeringModel().to(device)
    model.load_state_dict(saved['model_state'])
    return evaluate_model(model, sessions, Preprocessing(**saved['preprocessing']),
                          saved['train_mean_angle_deg'], device,
                          cache=_admit_cache(sessions, Preprocessing(**saved['preprocessing']),
                                             Alignment(**saved['alignment']), cache_mib),
                          alignment=Alignment(**saved['alignment']))


def export(checkpoint, destination):
    saved = load_checkpoint(checkpoint)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('export destination must be empty')
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {
        'format_version': FORMAT_VERSION, 'architecture': ARCHITECTURE,
        'preprocessing': saved['preprocessing'], 'image_stage': 'road_crop',
        'input': {'image': 'uint8 HWC RGB', 'resize': 'Pillow bilinear',
                  'pixels': 'pixel / 127.5 - 1', 'speed': 'mps / speed_scale_mps'},
        'output': {'units': 'physical wheel angle degrees', 'positive': 'right',
                   'scaling': 'tanh network output * angle_scale_deg'},
        'training_alignment': saved['alignment'], 'epoch': saved['epoch'],
        'validation': saved['history'][-1]['validation'],
        'training_session_ids': saved['train_sessions'],
        'validation_session_ids': saved['validation_sessions'],
        'dataset_groups': saved.get('dataset_groups', {}),
        'dataset_provenance': saved.get('dataset_provenance', {}),
    }
    atomic_save({key: value.cpu() for key, value in saved['model_state'].items()}, destination / 'model.pt')
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
