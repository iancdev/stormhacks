import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from forza_ai.data.sessions import Alignment, load_session, load_sessions, split_sessions
from forza_ai.data.synthetic import generate
from forza_ai.policies.predictor import SteeringPredictor
from forza_ai.training.engine import TrainConfig, evaluate, export, load_checkpoint, train


@pytest.fixture
def data(tmp_path):
    return generate(tmp_path / 'data', sessions=3, frames=8)


def edit_csv(path, mutate):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    mutate(rows)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_whole_session_split(data):
    sessions = load_sessions(data)
    a, b = split_sessions(sessions, .25, 7)
    assert {s.session_id for s in a}.isdisjoint({s.session_id for s in b})
    assert sum(len(s.samples) for s in a + b) == 24
    assert [s.session_id for s in a] == [s.session_id for s in split_sessions(sessions[::-1], .25, 7)[0]]


def test_interpolation_speed_is_causal(data):
    path = data / 'synthetic-000'
    edit_csv(path / 'frames.csv', lambda rows: rows[0].update(capture_time_ns='1016666666'))
    edit_csv(path / 'telemetry.csv', lambda rows: rows[1].update(speed_mps='99'))
    samples = load_session(path).samples
    expected = (0 + 75 * np.sin(1 / 5)) * (16_666_666 / 33_333_333)
    assert samples[0].angle_deg == pytest.approx(expected)
    assert samples[0].speed_mps == 15  # Never use the future speed=99 row.


def test_offset_rejects_mode_boundary_and_assist(data):
    path = data / 'synthetic-000'
    edit_csv(path / 'wheel.csv', lambda rows: rows[1].update(control_mode='takeover'))
    result = load_session(path, Alignment(label_offset_ns=33_333_333))
    assert result.rejected['nonexpert_or_mode_boundary'] == 2
    edit_csv(path / 'wheel.csv', lambda rows: [r.update(control_mode='assist') for r in rows])
    assert not load_session(path).samples


def test_stale_and_paused_telemetry(data):
    path = data / 'synthetic-000'
    edit_csv(path / 'telemetry.csv', lambda rows: rows.__delitem__(slice(1, None)))
    result = load_session(path, Alignment(max_telemetry_age_ns=20_000_000))
    assert len(result.samples) == 1
    assert result.rejected['stale_telemetry'] == 7
    edit_csv(path / 'telemetry.csv', lambda rows: rows[0].update(is_race_on='0'))
    assert not load_session(path).samples


def test_wheel_gap_and_pause_inside_offset(data):
    path = data / 'synthetic-000'
    edit_csv(path / 'wheel.csv', lambda rows: rows.pop(1))
    assert load_session(path).rejected['wheel_gap'] == 1
    edit_csv(path / 'telemetry.csv', lambda rows: rows[1].update(is_race_on='0'))
    result = load_session(path, Alignment(label_offset_ns=66_666_666, max_wheel_gap_ns=100_000_000))
    assert result.rejected['race_off'] >= 1


@pytest.mark.parametrize('case', ['path', 'nan', 'timestamp', 'incomplete', 'pedal', 'angle'])
def test_invalid_session_rejected(data, case):
    path = data / 'synthetic-000'
    if case == 'path':
        edit_csv(path / 'frames.csv', lambda rows: rows[0].update(image_path='../escape.png'))
    elif case == 'nan':
        edit_csv(path / 'telemetry.csv', lambda rows: rows[0].update(speed_mps='nan'))
    elif case == 'timestamp':
        edit_csv(path / 'wheel.csv', lambda rows: rows[1].update(timestamp_ns=rows[0]['timestamp_ns']))
    elif case == 'incomplete':
        meta = json.loads((path / 'metadata.json').read_text())
        meta['completed'] = False
        (path / 'metadata.json').write_text(json.dumps(meta))
    elif case == 'pedal':
        edit_csv(path / 'wheel.csv', lambda rows: rows[0].update(throttle='2'))
    elif case == 'angle':
        edit_csv(path / 'wheel.csv', lambda rows: rows[0].update(angle_deg='451'))
    with pytest.raises(ValueError):
        load_session(path)


def test_train_resume_evaluate_export(data, tmp_path):
    torch.set_num_threads(1)
    config = TrainConfig(batch_size=4)
    resumed, continuous = tmp_path / 'resumed', tmp_path / 'continuous'
    first = train(data, resumed, epochs=1, config=config, device='cpu')
    assert first['epoch'] == 1
    train(data, resumed, epochs=2, resume=resumed / 'last.pt', device='cpu')
    train(data, continuous, epochs=2, config=config, device='cpu')
    a, b = load_checkpoint(resumed / 'last.pt'), load_checkpoint(continuous / 'last.pt')
    for key in a['model_state']:
        torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
    metrics = evaluate(resumed / 'last.pt', data)
    assert metrics['samples'] == 8
    assert np.isfinite(metrics['model']['rmse_deg'])
    assert metrics['zero_baseline']['mae_deg'] > 0
    artifact = tmp_path / 'export'
    export(resumed / 'last.pt', artifact)
    predictor = SteeringPredictor(artifact)
    sample = load_sessions(data)[0].samples[0]
    pixels = np.asarray(Image.open(sample.image_path).convert('RGB'))
    angle = predictor.predict(pixels, sample.speed_mps)
    assert -450 <= angle <= 450
    with pytest.raises(ValueError):
        predictor.predict(pixels, float('nan'))
    with pytest.raises(ValueError):
        predictor.predict(pixels.astype(float), 10)
    with pytest.raises(ValueError, match='new session'):
        evaluate(resumed / 'last.pt', data, unseen=True)
    edit_csv(data / 'synthetic-000' / 'wheel.csv', lambda rows: rows[0].update(angle_deg='1'))
    with pytest.raises(ValueError, match='dataset changed'):
        train(data, resumed, epochs=3, resume=resumed / 'last.pt')


def test_negative_offset_and_stale_speed_inside_interval(data):
    path = data / 'synthetic-000'
    samples = load_session(path, Alignment(label_offset_ns=-33_333_333))
    assert samples.rejected['wheel_coverage'] == 1
    assert samples.samples[0].angle_deg == 0
    edit_csv(path / 'telemetry.csv', lambda rows: rows.__delitem__(slice(1, 5)))
    result = load_session(path, Alignment(label_offset_ns=166_666_665))
    assert result.rejected['stale_telemetry'] >= 1


def test_image_symlink_and_corrupt_image_rejected(data, tmp_path):
    path = data / 'synthetic-000'
    image = path / 'images/000000.png'
    image.unlink()
    outside = tmp_path / 'outside.png'
    Image.new('RGB', (10, 10)).save(outside)
    image.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        load_session(path)
    image.unlink()
    image.write_text('not an image')
    with pytest.raises(OSError):
        load_session(path)


def test_no_eligible_samples_and_duplicate_ids(data):
    path = data / 'synthetic-000'
    edit_csv(path / 'wheel.csv', lambda rows: [r.update(control_mode='assist') for r in rows])
    with pytest.raises(ValueError, match='accepted samples'):
        split_sessions(load_sessions(data), .25, 7)
    meta_path = path / 'metadata.json'
    metadata = json.loads(meta_path.read_text())
    metadata['session_id'] = 'synthetic-001'
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='unique'):
        load_sessions(data)


def test_export_parity_and_validation_integrity(data, tmp_path):
    from forza_ai.policies.steering_model import Preprocessing, SteeringModel, preprocess_rgb
    torch.set_num_threads(1)
    output = tmp_path / 'run'
    saved = train(data, output, config=TrainConfig(batch_size=8), device='cpu')
    export(output / 'last.pt', tmp_path / 'artifact')
    predictor = SteeringPredictor(tmp_path / 'artifact')
    sample = load_sessions(data)[0].samples[0]
    pixels = np.asarray(Image.open(sample.image_path).convert('RGB'))
    model = SteeringModel().eval()
    model.load_state_dict(saved['model_state'])
    with torch.inference_mode():
        expected = model(preprocess_rgb(pixels, Preprocessing()).unsqueeze(0),
                         torch.tensor([sample.speed_mps / 50])).item() * 450
    assert predictor.predict(pixels, sample.speed_mps) == expected
    with pytest.raises(ValueError, match='checkpoint directory'):
        train(data, tmp_path / 'different-run', epochs=2, resume=output / 'last.pt')
    held_out = data / saved['validation_sessions'][0]
    edit_csv(held_out / 'wheel.csv', lambda rows: rows[0].update(angle_deg='1'))
    with pytest.raises(ValueError, match='validation data changed'):
        evaluate(output / 'last.pt', data)


@pytest.mark.parametrize('frame_ms,pause_ms,accepted', [
    (30, 10, False),  # Pause before capture, inside interpolation's left support.
    (10, 20, False),  # Pause after capture, inside interpolation's right support.
    (40, 10, True),   # Exact wheel label needs no earlier interpolation support.
    (0, 20, True),    # Exact wheel label needs no future interpolation support.
])
def test_pause_across_interpolation_support(tmp_path, frame_ms, pause_ms, accepted):
    path = tmp_path / 'session'
    (path / 'images').mkdir(parents=True)
    (path / 'metadata.json').write_text(json.dumps({
        'schema_version': 1, 'session_id': 'pause', 'clock': 'monotonic_ns',
        'wheel_rotation_deg': 900, 'image_stage': 'road_crop', 'completed': True,
    }))
    Image.new('RGB', (200, 66)).save(path / 'images/frame.png')
    (path / 'frames.csv').write_text(f'frame_id,image_path,capture_time_ns\n0,images/frame.png,{frame_ms * 1_000_000}\n')
    (path / 'wheel.csv').write_text('timestamp_ns,angle_deg,throttle,brake,control_mode\n0,0,0,0,manual\n40000000,90,0,0,manual\n')
    (path / 'telemetry.csv').write_text(
        f'timestamp_ns,speed_mps,is_race_on\n0,10,1\n{pause_ms * 1_000_000},20,0\n{(pause_ms + 5) * 1_000_000},30,1\n')
    result = load_session(path)
    assert bool(result.samples) == accepted
    if not accepted:
        assert result.rejected == {'race_off': 1}


def test_stale_telemetry_at_future_interpolation_support(tmp_path):
    data = generate(tmp_path / 'data', sessions=2, frames=4)
    path = data / 'synthetic-000'
    edit_csv(path / 'frames.csv', lambda rows: (rows.__delitem__(slice(1, None)), rows[0].update(capture_time_ns='1010000000')))
    edit_csv(path / 'telemetry.csv', lambda rows: rows.__delitem__(slice(1, None)))
    result = load_session(path, Alignment(max_telemetry_age_ns=20_000_000))
    assert result.rejected == {'stale_telemetry': 1}


@pytest.mark.parametrize('failure', ['before_last', 'before_best'])
def test_interrupted_checkpoint_pair_recovers_best(data, tmp_path, monkeypatch, failure):
    from forza_ai.training import engine
    torch.set_num_threads(1)
    run = tmp_path / 'run'
    # Seed a completed epoch, then force the next epoch to improve.
    metric_values = iter([3.0, 1.0, 2.0])
    monkeypatch.setattr(engine, 'evaluate_model', lambda *args, **kwargs: {
        'model': {'rmse_deg': next(metric_values)}})
    train(data, run, epochs=1, config=TrainConfig(batch_size=8), device='cpu')
    original_save = engine.atomic_save
    old_best = load_checkpoint(run / 'best.pt')
    def interrupt(value, path):
        path = Path(path)
        if value['epoch'] == 2 and path.name == ('last.pt' if failure == 'before_last' else 'best.pt'):
            raise InterruptedError('simulated process interruption')
        original_save(value, path)
    monkeypatch.setattr(engine, 'atomic_save', interrupt)
    with pytest.raises(InterruptedError):
        train(data, run, epochs=2, resume=run / 'last.pt', device='cpu')
    monkeypatch.setattr(engine, 'atomic_save', original_save)
    last = load_checkpoint(run / 'last.pt')
    assert last['epoch'] == (1 if failure == 'before_last' else 2)
    # Stale best is repaired from last, even if resumed epoch is worse.
    assert load_checkpoint(run / 'best.pt')['epoch'] == 1
    train(data, run, epochs=last['epoch'] + 1, resume=run / 'last.pt', device='cpu')
    best = load_checkpoint(run / 'best.pt')
    if failure == 'before_best':
        assert best['epoch'] == 2
        assert best['best_rmse_deg'] == 1
        for key, value in last['best_checkpoint']['model_state'].items():
            torch.testing.assert_close(best['model_state'][key], value, rtol=0, atol=0)
    else:
        # Interrupted epoch 2 was not durable; repeating it can legitimately improve.
        assert best['best_rmse_deg'] == 2
    assert best['best_rmse_deg'] <= old_best['best_rmse_deg']


def test_first_epoch_pair_interruption_repaired(data, tmp_path, monkeypatch):
    from forza_ai.training import engine
    torch.set_num_threads(1)
    run = tmp_path / 'run'
    metric_values = iter([1.0, 2.0])
    monkeypatch.setattr(engine, 'evaluate_model', lambda *args, **kwargs: {
        'model': {'rmse_deg': next(metric_values)}})
    original_save = engine.atomic_save
    def interrupt(value, path):
        if Path(path).name == 'best.pt':
            raise InterruptedError('first best publication interrupted')
        original_save(value, path)
    monkeypatch.setattr(engine, 'atomic_save', interrupt)
    with pytest.raises(InterruptedError):
        train(data, run, config=TrainConfig(batch_size=8), device='cpu')
    assert not (run / 'best.pt').exists()
    monkeypatch.setattr(engine, 'atomic_save', original_save)
    train(data, run, epochs=2, resume=run / 'last.pt', device='cpu')
    assert load_checkpoint(run / 'best.pt')['epoch'] == 1
    assert load_checkpoint(run / 'last.pt')['epoch'] == 2
