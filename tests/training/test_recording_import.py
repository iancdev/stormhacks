import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from forza_ai.data.recording import COLUMNS, import_recording
from forza_ai.data.sessions import Alignment, load_session, load_sessions, split_sessions
from forza_ai.training.cli import main
from forza_ai.training.engine import TrainConfig, evaluate, export, train


def recording(path, name='drive-001', frames=8):
    (path / 'frames').mkdir(parents=True)
    metadata = {
        'session': name, 'config': {'monitor': 0, 'crop': [0, 0, 240, 80], 'masks': [], 'save_width': 240},
        'saved_size': [240, 80], 'fps_target': 30, 'frames': frames, 'segments': 2,
        'dropped': 0, 'vjoy': True, 'telemetry': True, 'steer_units_per_deg': 73.0,
        'pedals': '0 = released, 1 = floored', 'gaps': None,
    }
    (path / 'meta.json').write_text(json.dumps(metadata))
    with (path / 'labels.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for index in range(frames):
            Image.fromarray(np.full((80, 240, 3), index * 20, dtype=np.uint8)).save(path / f'frames/{index:06d}.jpg')
            writer.writerow([index, int(index >= frames // 2), f'{index / 30:.4f}', index * 730,
                             f'{index * 10:.2f}', '0.0000', '0.4000', '1.0', '15.000', 1, 0, '2.0'])
    return path


def edit_rows(path, operation):
    with (path / 'labels.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    operation(rows)
    with (path / 'labels.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def edit_meta(path, operation):
    metadata = json.loads((path / 'meta.json').read_text())
    operation(metadata)
    (path / 'meta.json').write_text(json.dumps(metadata))


def test_import_preserves_original_labels_images_and_provenance(tmp_path):
    source = recording(tmp_path / 'source')
    dest = tmp_path / 'imported'
    result = import_recording(source, dest, expert_mode='manual')
    assert result['accepted'] == 8
    assert result['split_group'] == 'record.py:drive-001'
    assert result['provenance']['timing']['image_age_bound_ns'] is None
    assert result['provenance']['timing']['time_resolution_ns'] == 100_000
    assert (dest / 'labels.csv').read_bytes() == (source / 'labels.csv').read_bytes()
    assert (dest / 'frames/000000.jpg').read_bytes() == (source / 'frames/000000.jpg').read_bytes()
    assert not (dest / 'wheel.csv').exists()  # Never fabricate source sample times.
    samples = load_session(dest).samples
    assert [s.angle_deg for s in samples] == list(range(0, 80, 10))
    assert samples[1].capture_time_ns == 33_300_000  # Quantized retrieval, not capture.
    with pytest.raises(ValueError, match='zero label offset'):
        load_session(dest, Alignment(label_offset_ns=1))
    with pytest.raises(ValueError, match='destination'):
        import_recording(source, dest, expert_mode='manual')


@pytest.mark.parametrize('case', ['telemetry', 'count', 'missing', 'extra', 'segments', 'no-meta', 'image-size', 'symlink', 'nan', 'calibration'])
def test_invalid_or_incomplete_recording_fails_transactionally(tmp_path, case):
    source = recording(tmp_path / 'source')
    if case == 'telemetry':
        edit_meta(source, lambda m: m.update(telemetry=False))
    elif case == 'count':
        edit_meta(source, lambda m: m.update(frames=9))
    elif case == 'missing':
        (source / 'frames/000001.jpg').unlink()
    elif case == 'extra':
        (source / 'frames/extra.jpg').write_bytes(b'extra')
    elif case == 'segments':
        edit_meta(source, lambda m: m.update(segments=3))
    elif case == 'no-meta':
        (source / 'meta.json').unlink()
    elif case == 'image-size':
        Image.new('RGB', (2, 2)).save(source / 'frames/000000.jpg')
    elif case == 'symlink':
        image = source / 'frames/000000.jpg'
        image.unlink()
        image.symlink_to(source / 'frames/000001.jpg')
    elif case == 'nan':
        edit_rows(source, lambda rows: rows[0].update(speed_mps='nan'))
    elif case == 'calibration':
        edit_rows(source, lambda rows: rows[0].update(steer_deg='20.00'))
    with pytest.raises((ValueError, OSError)):
        import_recording(source, tmp_path / 'dest', expert_mode='manual')
    assert not (tmp_path / 'dest').exists()


def test_conservative_age_rounding_and_duplicate_rows(tmp_path):
    source = recording(tmp_path / 'source')
    def changes(rows):
        rows[0]['wheel_age_ms'] = '0.0'  # Rounded interval straddles future time.
        rows[1]['wheel_age_ms'] = '50.0'  # +0.05ms exceeds default maximum.
        rows[2]['tele_age_ms'] = '100.0'
        rows[3]['tele_age_ms'] = '-0.1'
        rows[5]['t'] = rows[4]['t']
        rows[7]['race_on'] = '0'
    edit_rows(source, changes)
    report = import_recording(source, tmp_path / 'dest', expert_mode='manual')
    assert report['accepted'] == 2
    assert report['rejected'] == {'ambiguous_or_future_sample_age': 2, 'stale_wheel': 1,
                                  'stale_telemetry': 1, 'duplicate_rounded_time_or_image': 1, 'race_off': 1}


def test_repeated_source_sample_ages_do_not_create_duplicate_stream_timestamps(tmp_path):
    source = recording(tmp_path / 'source')
    # The same received telemetry can label multiple retrieved images. Do not try
    # to reconstruct unique stream samples from independently rounded t and ages.
    edit_rows(source, lambda rows: (rows[0].update(tele_age_ms='2.0'),
                                   rows[1].update(tele_age_ms='35.3'),
                                   rows[2].update(tele_age_ms='68.7')))
    report = import_recording(source, tmp_path / 'dest', expert_mode='manual')
    assert report['accepted'] == 8


def test_duplicate_cached_jpeg_excluded(tmp_path):
    source = recording(tmp_path / 'source')
    (source / 'frames/000001.jpg').write_bytes((source / 'frames/000000.jpg').read_bytes())
    report = import_recording(source, tmp_path / 'dest', expert_mode='manual')
    assert report['accepted'] == 7
    assert report['rejected']['duplicate_rounded_time_or_image'] == 1


def test_parent_recordings_are_independent_split_groups(tmp_path):
    dest = tmp_path / 'data'
    import_recording(recording(tmp_path / 'one', 'one'), dest / 'one', expert_mode='manual')
    with pytest.raises(ValueError, match='two independent'):
        split_sessions(load_sessions(dest), .25, 7)  # Two segments still one recording.
    import_recording(recording(tmp_path / 'two', 'two'), dest / 'two', expert_mode='manual')
    sessions = load_sessions(dest)
    a, b = split_sessions(sessions, .25, 7)
    assert {s.group for s in a}.isdisjoint({s.group for s in b})
    assert len(a[0].samples) == 8 and len(b[0].samples) == 8
    # Also honor grouping when a future format exposes separate segment sessions.
    from dataclasses import replace
    sibling = replace(sessions[0], session_id='separate-segment-same-recording')
    a, b = split_sessions(sessions + [sibling], .25, 7)
    assert {s.group for s in a}.isdisjoint({s.group for s in b})


def test_import_cli_train_resume_export(tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / 'data'
    for index in range(2):
        source = recording(tmp_path / f'source-{index}', f'drive-{index}')
        main(['import-recording', str(source), str(data / f'drive-{index}'), '--expert-mode', 'manual'])
    main(['validate', str(data)])
    run = tmp_path / 'run'
    train(data, run, config=TrainConfig(batch_size=4), device='cpu')
    saved = train(data, run, epochs=2, resume=run / 'last.pt', device='cpu')
    metrics = evaluate(run / 'best.pt', data)
    assert metrics['samples'] == 8
    artifact = export(run / 'best.pt', tmp_path / 'artifact')
    assert len(artifact['dataset_provenance']) == 2
    assert all(p['timing']['image_age_bound_ns'] is None for p in artifact['dataset_provenance'].values())
    assert set(saved['dataset_groups'].values()) == {'record.py:drive-0', 'record.py:drive-1'}


def test_manual_mode_is_required(tmp_path):
    source = recording(tmp_path / 'source')
    with pytest.raises(ValueError, match='expert-mode manual'):
        import_recording(source, tmp_path / 'dest')
    with pytest.raises(SystemExit) as error:
        main(['import-recording', str(source), str(tmp_path / 'dest')])
    assert error.value.code == 2
