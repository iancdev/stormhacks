from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from forza_ai.data.cache import PreprocessingCache
from forza_ai.data.dataset import SteeringDataset
from forza_ai.data.sessions import Alignment, load_sessions, split_sessions
from forza_ai.data.synthetic import generate
from forza_ai.policies.steering_model import Preprocessing, SteeringModel
from forza_ai.training import engine


@pytest.fixture
def sessions(tmp_path):
    return load_sessions(generate(tmp_path / 'data', sessions=3, frames=8))


def equal_batch(a, b):
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_exact_tensors_shuffle_loss_predictions_and_no_second_decode(sessions):
    cache = PreprocessingCache(8 * 1024**2)
    cached = SteeringDataset(sessions, Preprocessing(), cache, Alignment())
    plain = SteeringDataset(sessions, Preprocessing())
    for i in range(len(plain)):
        equal_batch(plain[i], cached[i])
    # Mutating a caller's tensors cannot poison resident values.
    cached[0][0].zero_()
    equal_batch(cached[0], plain[0])
    torch.manual_seed(7)
    model = SteeringModel()
    loaders = [DataLoader(d, batch_size=4, shuffle=True,
                          generator=torch.Generator().manual_seed(8)) for d in (cached, plain)]
    for a, b in zip(*loaders):
        equal_batch(a, b)
        with torch.no_grad():
            x, y = model(a[0], a[1]), model(b[0], b[1])
            assert torch.equal(x, y)
            assert torch.equal(torch.nn.functional.mse_loss(x, a[2]), torch.nn.functional.mse_loss(y, b[2]))
    with patch('forza_ai.data.dataset.Image.open', side_effect=AssertionError('unexpected decode')):
        for i in range(len(cached)):
            cached[i]
    assert cache.hits >= len(cached)


def test_lru_budget_and_too_small_fallback(sessions):
    plain = SteeringDataset(sessions, Preprocessing())
    size = sum(x.numel() * x.element_size() for x in plain[0])
    cache = PreprocessingCache(2 * size)
    data = SteeringDataset(sessions, Preprocessing(), cache)
    data[0]; data[1]; data[0]; data[2]
    assert cache.resident_bytes == 2 * size
    assert cache.evictions == 1
    with patch('forza_ai.data.dataset.Image.open', side_effect=AssertionError('evicted item decoded')):
        data[0]
        with pytest.raises(AssertionError):
            data[1]
    for budget in [0, size - 1]:
        small = PreprocessingCache(budget)
        fallback = SteeringDataset(sessions, Preprocessing(), small)
        equal_batch(fallback[0], plain[0]); equal_batch(fallback[0], plain[0])
        assert small.resident_bytes == 0


def test_cache_oom_releases_storage_and_falls_back(sessions, monkeypatch):
    cache = PreprocessingCache(8 * 1024**2)
    data = SteeringDataset(sessions, Preprocessing(), cache)
    plain = SteeringDataset(sessions, Preprocessing())
    data[0]
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, 'clone', lambda _: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError('test')))
        with pytest.warns(RuntimeWarning, match='continuing uncached'):
            equal_batch(data[0], plain[0])
    assert cache.max_bytes == cache.resident_bytes == 0
    equal_batch(data[1], plain[1])


@pytest.mark.parametrize('change', ['fingerprint', 'accepted', 'labels', 'preprocess', 'alignment'])
def test_cache_identity_invalidates(sessions, change):
    cache = PreprocessingCache(8 * 1024**2)
    original = sessions[0]
    base = SteeringDataset([original], Preprocessing(), cache, Alignment())
    base[0]
    session, preprocessing, alignment = original, Preprocessing(), Alignment()
    if change == 'fingerprint': session = replace(original, fingerprint='changed')
    if change == 'accepted': session = replace(original, samples=original.samples[1:])
    if change == 'labels': session = replace(original, samples=[replace(original.samples[0], angle_deg=50)] + original.samples[1:])
    if change == 'preprocess': preprocessing = Preprocessing(speed_scale_mps=100)
    if change == 'alignment': alignment = Alignment(label_offset_ns=1)
    changed = SteeringDataset([session], preprocessing, cache, alignment)
    assert changed.cache_keys[0] != base.cache_keys[0]
    hits = cache.hits
    equal_batch(changed[0], SteeringDataset([session], preprocessing)[0])
    assert cache.hits == hits


def test_validation_reuses_shared_cache_and_preserves_session_split(sessions):
    torch.set_num_threads(1)
    training, validation = split_sessions(sessions, .25, 7)
    assert {s.group for s in training}.isdisjoint(s.group for s in validation)
    cache = PreprocessingCache(8 * 1024**2)
    model = SteeringModel()
    args = (model, validation, Preprocessing(), 0, torch.device('cpu'))
    expected = engine.evaluate_model(*args)
    assert engine.evaluate_model(*args, cache=cache, alignment=Alignment()) == expected
    with patch('forza_ai.data.dataset.Image.open', side_effect=AssertionError('validation decoded twice')):
        assert engine.evaluate_model(*args, cache=cache, alignment=Alignment()) == expected
    with pytest.raises(ValueError, match='two independent'):
        split_sessions([sessions[0]], .25, 7)


def assert_state(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_state(x, y)
    else: assert a == b


def test_full_training_resume_and_legacy_checkpoint_parity(tmp_path):
    torch.set_num_threads(1)
    data = generate(tmp_path / 'data', sessions=3, frames=8)
    baseline = engine.train(data, tmp_path / 'plain', epochs=2,
                            config=engine.TrainConfig(batch_size=4, cache_mib=0), device='cpu')
    cached = engine.train(data, tmp_path / 'cached', epochs=2,
                          config=engine.TrainConfig(batch_size=4, cache_mib=8), device='cpu')
    engine.train(data, tmp_path / 'resume', epochs=1,
                 config=engine.TrainConfig(batch_size=4, cache_mib=8), device='cpu')
    resumed = engine.train(data, tmp_path / 'resume', epochs=2,
                           resume=tmp_path / 'resume/last.pt', device='cpu')
    evicting = engine.train(data, tmp_path / 'evicting', epochs=2,
                            config=engine.TrainConfig(batch_size=4, cache_mib=1), device='cpu')
    for result in (cached, resumed, evicting):
        for key in ['model_state', 'optimizer_state', 'history', 'train_sessions', 'validation_sessions', 'torch_rng_state']:
            assert_state(baseline[key], result[key])
    # Old format-1 checkpoints predate cache_mib; they still resume identically.
    legacy = engine.train(data, tmp_path / 'legacy', epochs=1,
                          config=engine.TrainConfig(batch_size=4, cache_mib=0), device='cpu')
    del legacy['train_config']['cache_mib']
    engine.atomic_save(legacy, tmp_path / 'legacy/last.pt')
    result = engine.train(data, tmp_path / 'legacy', epochs=2,
                          resume=tmp_path / 'legacy/last.pt', device='cpu')
    assert_state(baseline['model_state'], result['model_state'])
    assert engine.evaluate(tmp_path / 'cached/last.pt', data, cache_mib=8) == engine.evaluate(tmp_path / 'cached/last.pt', data, cache_mib=0)


def test_workers_disable_cache_without_changing_worker_count(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = generate(tmp_path / 'data', sessions=3, frames=4)
    real_loader = engine.DataLoader
    seen = []
    def loader(dataset, *args, **kwargs):
        seen.append(kwargs.get('num_workers', 0))
        assert dataset.cache is None
        # Avoid platform process startup in this test; verify requested value.
        kwargs['num_workers'] = 0
        return real_loader(dataset, *args, **kwargs)
    monkeypatch.setattr(engine, 'DataLoader', loader)
    with pytest.warns(RuntimeWarning, match='disabled with workers'):
        engine.train(data, tmp_path / 'run', config=engine.TrainConfig(workers=2), device='cpu')
    assert 2 in seen


def test_insertion_oom_and_unrelated_errors(sessions, monkeypatch):
    cache = PreprocessingCache(8 * 1024**2)
    data = SteeringDataset(sessions, Preprocessing(), cache)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, 'clone', lambda _: (_ for _ in ()).throw(RuntimeError("DefaultCPUAllocator: can't allocate memory")))
        with pytest.warns(RuntimeWarning, match='continuing uncached'):
            result = data[0]
    equal_batch(result, SteeringDataset(sessions, Preprocessing())[0])
    assert cache.max_bytes == cache.resident_bytes == 0
    cache = PreprocessingCache(8 * 1024**2)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, 'clone', lambda _: (_ for _ in ()).throw(RuntimeError('unrelated bug')))
        with pytest.raises(RuntimeError, match='unrelated bug'):
            cache.put('a', result)


@pytest.mark.parametrize('budget', [-1, 0.1, True])
def test_invalid_budget(budget):
    with pytest.raises(ValueError): engine.TrainConfig(cache_mib=budget)
    with pytest.raises(ValueError): PreprocessingCache(budget)


@pytest.mark.parametrize('operation', ['get', 'put'])
@pytest.mark.parametrize('kind', ['python', 'torch', 'allocator'])
def test_oom_fallback_without_top_level_torch_alias(monkeypatch, operation, kind):
    values = (torch.ones(3),)
    cache = PreprocessingCache(1024)
    cache.put('existing', values)
    error = {'python': MemoryError('allocation'),
             'torch': torch.cuda.OutOfMemoryError('allocation'),
             'allocator': RuntimeError("DefaultCPUAllocator: can't allocate memory")}[kind]
    monkeypatch.delattr(torch, 'OutOfMemoryError', raising=False)
    def fail_clone(_):
        raise error
    monkeypatch.setattr(torch.Tensor, 'clone', fail_clone)
    with pytest.warns(RuntimeWarning, match='continuing uncached'):
        if operation == 'get': assert cache.get('existing') is None
        else: cache.put('new', values)
    assert cache.resident_bytes == cache.max_bytes == 0
    assert not cache._entries
    assert not cache._is_oom(RuntimeError('unrelated error'))


def test_estimate_matches_actual_tensor_bytes_and_key_uniqueness(sessions):
    from forza_ai.data.dataset import estimate_cache_bytes
    config = Preprocessing()
    data = SteeringDataset(sessions, config)
    actual = sum(t.numel() * t.element_size() for i in range(len(data)) for t in data[i])
    assert estimate_cache_bytes(sessions, config, Alignment()) == actual
    assert estimate_cache_bytes(sessions + [sessions[0]], config, Alignment()) == actual
    assert actual == len(data) * 158408


def test_admission_includes_validation_and_reports_budget(sessions, capsys):
    training, validation = split_sessions(sessions, .25, 7)
    config, alignment = Preprocessing(), Alignment()
    # Training alone fits 3 MiB; all 24 accepted samples do not.
    assert engine._admit_cache(training, config, alignment, 3) is not None
    assert engine._admit_cache(training + validation, config, alignment, 3) is None
    report = __import__('json').loads(capsys.readouterr().err.splitlines()[-1])['preprocessing_cache']
    assert report == {'enabled': False, 'reason': 'working_set_exceeds_budget',
                      'budget_bytes': 3 * 1024**2, 'required_bytes': 24 * 158408}
    assert engine._admit_cache(training + validation, config, alignment, 4) is not None
    assert engine._admit_cache(training + validation, config, alignment, 0) is None


@pytest.mark.parametrize('result', [None, MemoryError('budget estimation allocation failed')])
def test_unknown_or_failed_estimate_disables_cache(sessions, monkeypatch, result, capsys):
    def estimate(*args):
        if isinstance(result, Exception): raise result
        return result
    monkeypatch.setattr(engine, 'estimate_cache_bytes', estimate)
    assert engine._admit_cache(sessions, Preprocessing(), Alignment(), 4) is None
    report = __import__('json').loads(capsys.readouterr().err)['preprocessing_cache']
    assert report['reason'] in {'unknown_working_set', 'estimate_allocation_failed'}
    assert not report['enabled']


def test_over_budget_cli_never_creates_or_churns_cache(tmp_path, monkeypatch, capsys):
    from forza_ai.training.cli import main
    torch.set_num_threads(1)
    data = generate(tmp_path / 'data', sessions=3, frames=8)
    def forbidden(*args): raise AssertionError('over-budget cache allocated')
    monkeypatch.setattr(engine, 'PreprocessingCache', forbidden)
    main(['train', str(data), str(tmp_path / 'run'), '--epochs', '2', '--device', 'cpu', '--cache-mib', '1'])
    report = __import__('json').loads(capsys.readouterr().err)['preprocessing_cache']
    assert report['reason'] == 'working_set_exceeds_budget'
    assert report['required_bytes'] > report['budget_bytes']


def test_cache_initialization_failure_disables_once(sessions, monkeypatch, capsys):
    class FailingCache(PreprocessingCache):
        def __init__(self, *args):
            raise MemoryError('cache metadata')
    monkeypatch.setattr(engine, 'PreprocessingCache', FailingCache)
    assert engine._admit_cache(sessions, Preprocessing(), Alignment(), 4) is None
    assert 'cache_initialization_failed' in capsys.readouterr().err


def test_default_512mib_admits_1988_frames_but_rejects_larger(sessions, capsys):
    from forza_ai.data.dataset import estimate_cache_bytes
    config = engine.TrainConfig()
    assert config.cache_mib == 512
    sample = sessions[0].samples[0]
    def sized(count):
        return [replace(sessions[0], samples=[replace(sample, capture_time_ns=i) for i in range(count)])]
    real_sized = sized(1988)  # Size-only unit fixture; no train/validation claim.
    assert estimate_cache_bytes(real_sized, Preprocessing()) == 314915104
    cache = engine._admit_cache(real_sized, Preprocessing(), Alignment(), config.cache_mib)
    assert cache is not None and cache.max_bytes == 512 * 1024**2
    assert engine._admit_cache(sized(4000), Preprocessing(), Alignment(), config.cache_mib) is None
    assert engine._admit_cache(real_sized, Preprocessing(), Alignment(), 256) is None


def test_cli_defaults_to_512_and_preserves_explicit_zero(tmp_path, monkeypatch):
    from forza_ai.training.cli import main
    configs = []
    monkeypatch.setattr(engine, 'train', lambda data, output, epochs, config, *args, **kwargs: configs.append(config))
    main(['train', str(tmp_path / 'data'), str(tmp_path / 'out')])
    main(['train', str(tmp_path / 'data'), str(tmp_path / 'out'), '--cache-mib', '0'])
    assert [c.cache_mib for c in configs] == [512, 0]
