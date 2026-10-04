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
    manifest_bytes = (dest / 'metadata.json').read_bytes()
    assert b'\r\n' not in manifest_bytes
    assert manifest_bytes == (json.dumps(json.loads(manifest_bytes), indent=2) + '\n').encode('utf-8')
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
        edit_meta(source, lambda m: m.update(segments=4))
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


@pytest.mark.parametrize('prefix', ['', 'imported-session/'])
def test_colab_archive_accepts_imported_recording_and_validates(tmp_path, prefix):
    import runpy
    import zipfile

    helpers = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/colab_archives.py'))
    source = recording(tmp_path / 'source')
    imported = tmp_path / 'imported'
    original = import_recording(source, imported, expert_mode='manual')

    def archive_import():
        archive = tmp_path / 'recording.zip'
        with zipfile.ZipFile(archive, 'w') as output:
            for path in imported.rglob('*'):
                if path.is_file():
                    output.write(path, prefix + path.relative_to(imported).as_posix())
        return archive

    summaries = helpers['extract_sessions']([archive_import()], tmp_path / 'extracted')
    assert summaries == [original]
    # A valid directory shape is insufficient: canonical loader still checks provenance.
    manifest = imported / 'metadata.json'
    metadata = json.loads(manifest.read_text())
    metadata['expert_mode'] = 'assist'
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='provenance'):
        helpers['extract_sessions']([archive_import()], tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
    metadata['schema_version'] = 'unknown_format'
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='unsupported session schema'):
        helpers['extract_sessions']([archive_import()], tmp_path / 'unsupported')
    assert not (tmp_path / 'unsupported').exists()


@pytest.mark.parametrize('case,empty_segments', [('trailing', 1), ('leading', 1), ('middle', 1), ('multiple', 3)])
def test_empty_segments_require_dropped_frame_evidence(tmp_path, case, empty_segments):
    source = recording(tmp_path / 'source')
    if case == 'leading':
        edit_rows(source, lambda rows: [r.update(segment=str(int(r['segment']) + 1)) for r in rows])
    elif case == 'middle':
        edit_rows(source, lambda rows: [r.update(segment='2') for r in rows if r['segment'] == '1'])
    edit_meta(source, lambda m: m.update(segments=2 + empty_segments, dropped=empty_segments, completed=True))
    report = import_recording(source, tmp_path / 'valid', expert_mode='manual')
    assert report['accepted'] == 8
    assert report['provenance']['segments'] == 2 + empty_segments
    edit_meta(source, lambda m: m.update(dropped=empty_segments - 1))
    with pytest.raises(ValueError, match='empty segments exceed'):
        import_recording(source, tmp_path / 'invalid', expert_mode='manual')
    assert not (tmp_path / 'invalid').exists()


def test_original_ctrl_c_can_leave_one_trailing_unqueued_segment(tmp_path):
    source = recording(tmp_path / 'source')
    edit_meta(source, lambda m: m.update(segments=3, dropped=0))
    report = import_recording(source, tmp_path / 'accepted', expert_mode='manual')
    assert report['accepted'] == 8
    assert 'KeyboardInterrupt' in report['provenance']['empty_segment_evidence']
    # The rule is tied to the tracked original metadata shape, not later variants.
    edit_meta(source, lambda m: m.update(discarded_at_stop=5))
    with pytest.raises(ValueError, match='empty segments exceed'):
        import_recording(source, tmp_path / 'unverified-variant', expert_mode='manual')


@pytest.mark.parametrize('case', ['two-trailing', 'leading', 'middle', 'modern'])
def test_interrupt_compatibility_does_not_explain_other_empty_segments(tmp_path, case):
    source = recording(tmp_path / 'source')
    edit_meta(source, lambda m: m.update(segments=3, dropped=0))
    if case == 'two-trailing':
        edit_meta(source, lambda m: m.update(segments=4))
    elif case == 'leading':
        edit_rows(source, lambda rows: [r.update(segment=str(int(r['segment']) + 1)) for r in rows])
    elif case == 'middle':
        edit_rows(source, lambda rows: [r.update(segment='2') for r in rows if r['segment'] == '1'])
    else:
        edit_meta(source, lambda m: m.update(completed=True))
    with pytest.raises(ValueError, match='empty segments exceed'):
        import_recording(source, tmp_path / 'invalid', expert_mode='manual')


def test_diagnostic_extended_variant_keeps_failures_and_original_bytes(tmp_path):
    import hashlib
    from forza_ai.data.recording import inspect_recording_for_diagnostics
    source = recording(tmp_path / 'source')
    edit_meta(source, lambda m: m.update(segments=3, dropped=0, discarded_at_stop=5))
    labels = source / 'labels.csv'
    rows = list(csv.DictReader(labels.open()))
    extras = ['race_time', 'distance', 'yaw_rate', 'game_ms', 'gear']
    with labels.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS + extras)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, race_time='1.000', distance='-37.4', yaw_rate='0.0000', game_ms='100', gear='1'))
    before = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='columns'):
        import_recording(source, tmp_path / 'strict', expert_mode='manual')
    session = inspect_recording_for_diagnostics(source, expert_mode='manual')
    assert len(session.samples) == 8
    assert session.provenance['diagnostic_only'] is True
    assert session.provenance['production_validation_passed'] is False
    assert len(session.provenance['strict_validation_issues']) == 2
    assert session.provenance['empty_segment_evidence'] == 'unresolved'
    assert not (source / 'metadata.json').exists()
    after = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in source.rglob('*') if p.is_file()}
    assert before == after
    with pytest.raises(ValueError, match='diagnostic-only'):
        split_sessions([session, session], .25, 7)
    # Diagnostic scope does not bypass missing files or malformed base data.
    (source / 'frames/000003.jpg').unlink()
    with pytest.raises(ValueError, match='missing or extra'):
        inspect_recording_for_diagnostics(source, expert_mode='manual')


def extended_recording(path, *, cars=False, gap=False):
    source = recording(path)
    extras = ['race_time', 'distance', 'yaw_rate', 'game_ms', 'gear']
    if cars:
        extras += ['car_ordinal', 'car_class', 'car_pi']
    with (source / 'labels.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if gap:
        # Pending frames 3 and 4 were discarded before later rows were saved.
        for row in reversed(rows[3:]):
            old = int(row['frame'])
            row['frame'] = str(old + 2)
            (source / f'frames/{old:06d}.jpg').rename(source / f'frames/{old + 2:06d}.jpg')
    with (source / 'labels.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS + extras)
        writer.writeheader()
        for row in rows:
            row.update(race_time='1.000', distance='-37.4', yaw_rate='0.0000', game_ms='100', gear='1')
            if cars:
                row.update(car_ordinal='1', car_class='4', car_pi='800')
            writer.writerow(row)
    def meta(m):
        m.update(segments=3, discarded_at_stop=5, drop_seconds=5.0, rewind_button=None,
                 rewinds=int(gap))
        if cars:
            m.update(takeovers=0, discarded_by_rewind_or_takeover=2 if gap else 0)
        else:
            m.update(discarded_by_rewind=2 if gap else 0)
    edit_meta(source, meta)
    return source


@pytest.mark.parametrize('cars', [False, True])
@pytest.mark.parametrize('gap', [False, True])
def test_buffered_producer_variants_and_ids(tmp_path, cars, gap):
    source = extended_recording(tmp_path / 'source', cars=cars, gap=gap)
    report = import_recording(source, tmp_path / 'out', expert_mode='manual')
    assert report['accepted'] == 8
    assert report['provenance']['buffering']['stopped'] == 5
    assert len(report['provenance']['additional_columns_not_model_inputs']) == (8 if cars else 5)
    assert (source / 'labels.csv').read_bytes() == (tmp_path / 'out/labels.csv').read_bytes()
    assert sorted(p.name for p in (source / 'frames').iterdir()) == sorted(p.name for p in (tmp_path / 'out/frames').iterdir())
    with pytest.raises(ValueError, match='two independent'):
        split_sessions([load_session(tmp_path / 'out')], .25, 7)


@pytest.mark.parametrize('fault', ['missing-image', 'unexplained-gap', 'negative-discard', 'unexplained-segment',
                                  'double-count', 'no-event', 'bad-extra-value', 'false-completion', 'missing-timing'])
def test_extended_producer_rejects_unexplained_or_corrupt_data(tmp_path, fault):
    source = extended_recording(tmp_path / 'source', cars=True, gap=True)
    if fault == 'missing-image':
        (source / 'frames/000006.jpg').unlink()
    elif fault == 'unexplained-gap':
        edit_meta(source, lambda m: m.update(discarded_by_rewind_or_takeover=0))
    elif fault == 'negative-discard':
        edit_meta(source, lambda m: m.update(discarded_at_stop=-1))
    elif fault == 'unexplained-segment':
        edit_meta(source, lambda m: m.update(segments=20))
    elif fault == 'double-count':
        edit_meta(source, lambda m: m.update(discarded_at_stop=0))
    elif fault == 'no-event':
        edit_meta(source, lambda m: m.update(rewinds=0))
    elif fault == 'bad-extra-value':
        p = source / 'labels.csv'
        p.write_text(p.read_text().replace('-37.4', 'nan'))
    elif fault == 'false-completion':
        edit_meta(source, lambda m: m.update(completed=False))
    elif fault == 'missing-timing':
        edit_meta(source, lambda m: m.update(capture_provenance={'timing_file': 'capture_timing.csv'}))
    with pytest.raises((ValueError, OSError)):
        import_recording(source, tmp_path / 'out', expert_mode='manual')
    assert not (tmp_path / 'out').exists()


def test_preserves_optional_hud_and_timing_without_using_as_inputs(tmp_path):
    source = extended_recording(tmp_path / 'source', cars=True)
    (source / 'hud').mkdir()
    Image.new('RGB', (8, 10)).save(source / 'hud/000000.png')
    timing = 'frame,retrieved_ns\n' + ''.join(f'{i},{i * 1000000}\n' for i in range(8))
    (source / 'capture_timing.csv').write_text(timing)
    edit_meta(source, lambda m: m.update(completed=True, jpeg_quality=90, measured_capture={},
                                        capture_provenance={'timing_file': 'capture_timing.csv'}))
    report = import_recording(source, tmp_path / 'out', expert_mode='manual')
    assert report['accepted'] == 8
    assert report['provenance']['auxiliary_files_preserved'] == ['capture_timing.csv', 'hud/000000.png']
    for name in report['provenance']['auxiliary_files_preserved']:
        assert (source / name).read_bytes() == (tmp_path / 'out' / name).read_bytes()
    assert report['provenance']['timing']['image_age_bound_ns'] is None


def test_optional_acquired_counter_reconciles_but_is_not_required(tmp_path):
    source = extended_recording(tmp_path / 'source', cars=True, gap=True)
    # 8 saved + 2 rewound + 5 pending at stop.
    edit_meta(source, lambda m: m.update(accepted_frame_count=15))
    assert import_recording(source, tmp_path / 'valid', expert_mode='manual')['accepted'] == 8
    edit_meta(source, lambda m: m.update(accepted_frame_count=14))
    with pytest.raises(ValueError, match='accepted_frame_count'):
        import_recording(source, tmp_path / 'invalid', expert_mode='manual')


def test_explicit_exclusion_file_uses_original_session_identity(tmp_path):
    source = extended_recording(tmp_path / 'renamed-folder')
    rules = tmp_path / 'exclude.txt'
    rules.write_text('# operator exclusions\ndrive- # excluded parent\n')
    with pytest.raises(ValueError, match='exclusion prefix'):
        import_recording(source, tmp_path / 'excluded', expert_mode='manual', exclude_sessions=rules)
    rules.write_text('different- # unrelated\n')
    result = import_recording(source, tmp_path / 'accepted', expert_mode='manual', exclude_sessions=rules)
    assert result['accepted'] == 8
    assert 'different-' in json.loads((tmp_path / 'accepted/metadata.json').read_text())['exclusion_prefixes_checked']


def test_optional_producer_identity_preserved_without_claiming_verification(tmp_path):
    source = extended_recording(tmp_path / 'source', cars=True)
    edit_meta(source, lambda m: m.update(producer_schema='record_py_buffered_20_v1', producer_sha256='ab' * 32))
    result = import_recording(source, tmp_path / 'out', expert_mode='manual')
    assert result['provenance']['declared_producer_schema'] == 'record_py_buffered_20_v1'
    assert result['provenance']['declared_producer_sha256'] == 'ab' * 32


@pytest.mark.parametrize('fields,cars', [
    ({'producer_schema': 'unknown'}, True),
    ({'producer_schema': 'record_py_buffered_20_v1'}, False),
    ({'producer_sha256': 'A' * 64}, True),
    ({'producer_sha256': 'a' * 63}, True),
    ({'producer_sha256': 123}, True),
])
def test_invalid_declared_producer_identity_fails(tmp_path, fields, cars):
    source = extended_recording(tmp_path / 'source', cars=cars)
    edit_meta(source, lambda m: m.update(fields))
    with pytest.raises(ValueError, match='producer_'):
        import_recording(source, tmp_path / 'out', expert_mode='manual')


def test_source_directory_named_hud_preserves_timing_sidecar(tmp_path):
    source = extended_recording(tmp_path / 'hud', cars=True)
    (source / 'capture_timing.csv').write_text('frame,retrieved_ns\n' + ''.join(f'{i},{i}\n' for i in range(8)))
    first = import_recording(source, tmp_path / 'first', expert_mode='manual')
    source.rename(tmp_path / 'renamed')
    second = import_recording(tmp_path / 'renamed', tmp_path / 'second', expert_mode='manual')
    assert first == second


@pytest.mark.parametrize('identity', ['20261003_150225', '20261003_152123-extra'])
def test_checked_in_exclusions_apply_without_cli_flag_and_cannot_be_replaced(tmp_path, identity):
    source = extended_recording(tmp_path / 'unrelated-folder')
    edit_meta(source, lambda m: m.update(session=identity))
    empty_rules = tmp_path / 'empty-rules.txt'
    empty_rules.write_text('# cannot replace mandatory defaults\n')
    for rules in [None, empty_rules]:
        with pytest.raises(ValueError, match='exclusion prefix'):
            import_recording(source, tmp_path / 'out', expert_mode='manual', exclude_sessions=rules)
    from forza_ai.data.recording import inspect_recording_for_diagnostics
    assert inspect_recording_for_diagnostics(source, expert_mode='manual').provenance['diagnostic_only']


@pytest.mark.parametrize('identity', ['20261003_150225', '20261003_152123'])
def test_already_imported_excluded_data_rejected_before_training_or_evaluation(tmp_path, identity):
    from forza_ai.training.engine import load_checkpoint
    source = extended_recording(tmp_path / 'source')
    imported = tmp_path / 'data/session'
    import_recording(source, imported, expert_mode='manual')
    # Simulate an old import that passed before today's repository policy.
    metadata = json.loads((imported / 'metadata.json').read_text())
    metadata['source_metadata']['session'] = identity
    (imported / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='exclusion prefix'):
        load_sessions(imported.parent)
    with pytest.raises(ValueError, match='exclusion prefix'):
        train(imported.parent, tmp_path / 'run', device='cpu')
    # Evaluation loads the policy before touching any images/model. Supply only
    # the fields its loader needs; no compute or fabricated validation metrics.
    from forza_ai.training import engine
    from unittest.mock import patch
    with patch.object(engine, 'load_checkpoint', return_value={'alignment': {
        'label_offset_ns': 0, 'max_wheel_gap_ns': 50_000_000, 'max_telemetry_age_ns': 100_000_000}}):
        with pytest.raises(ValueError, match='exclusion prefix'):
            evaluate('unused', imported.parent)
    assert not (tmp_path / 'run').exists()


def test_packaged_policy_matches_repository_and_missing_policy_fails_closed(tmp_path, monkeypatch):
    from forza_ai.data import exclusions
    root = Path(__file__).resolve().parents[2]
    assert (root / 'config/exclude_sessions.txt').read_bytes() == Path(exclusions.__file__).with_name('exclude_sessions.txt').read_bytes()
    monkeypatch.setattr(exclusions, 'default_policy_path', lambda: tmp_path / 'missing.txt')
    with pytest.raises(FileNotFoundError):
        exclusions.enforce_exclusions('20261003_152944')
