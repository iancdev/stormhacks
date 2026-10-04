"""Explicit import of record.py's rounded, per-frame labels (not v1 streams)."""
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from PIL import Image

FORMAT = 'record_py_aligned_v1'
COLUMNS = ['frame', 'segment', 't', 'steer_raw', 'steer_deg', 'brake', 'gas',
           'wheel_age_ms', 'speed_mps', 'race_on', 'tele_steer', 'tele_age_ms']
LEGACY_METADATA_KEYS = {'session', 'config', 'saved_size', 'fps_target', 'frames',
                        'segments', 'dropped', 'vjoy', 'telemetry', 'steer_units_per_deg',
                        'pedals', 'gaps'}
DIAGNOSTIC_EXTRA_COLUMNS = {'race_time', 'distance', 'yaw_rate', 'game_ms', 'gear'}

EXTENDED_SUFFIXES = [
    ['race_time', 'distance'],
    ['race_time', 'distance', 'yaw_rate', 'game_ms'],
    ['race_time', 'distance', 'yaw_rate', 'game_ms', 'gear'],
    ['race_time', 'distance', 'yaw_rate', 'game_ms', 'gear', 'car_ordinal', 'car_class', 'car_pi'],
]


def _buffer_profile(metadata, columns):
    """Recognize source-backed buffering variants, never infer counters from gaps."""
    if columns not in [COLUMNS + suffix for suffix in EXTENDED_SUFFIXES]:
        return None
    common = {'rewinds', 'discarded_at_stop', 'drop_seconds', 'rewind_button'}
    old = 'discarded_by_rewind' in metadata
    new = {'discarded_by_rewind_or_takeover', 'takeovers'} <= metadata.keys()
    if (not common <= metadata.keys() or old == new
            or ('car_pi' in columns and not new)):
        return None
    rewinds = _integer(metadata['rewinds'], 'rewinds')
    takeovers = _integer(metadata['takeovers'], 'takeovers') if new else 0
    key = 'discarded_by_rewind_or_takeover' if new else 'discarded_by_rewind'
    discarded = _integer(metadata[key], key)
    stopped = _integer(metadata['discarded_at_stop'], 'discarded_at_stop')
    if discarded and not rewinds + takeovers:
        raise ValueError('discarded frames require a rewind/takeover event')
    if _decimal(metadata['drop_seconds'], 'drop_seconds') < 0:
        raise ValueError('drop_seconds must be nonnegative')
    if metadata['rewind_button'] is not None:
        _integer(metadata['rewind_button'], 'rewind_button')
    return {'discarded': discarded, 'stopped': stopped,
            'family': 'buffered_takeover' if new else 'buffered_rewind',
            'source_reference': '0117f25' if 'car_pi' in columns else '82c2e46 / 0bb45a1 / 2c4c81e'}


def _auxiliary_files(path):
    files = []
    timing = path / 'capture_timing.csv'
    if timing.is_symlink():
        raise ValueError('timing sidecar symlinks are forbidden')
    if timing.exists():
        if not timing.is_file():
            raise ValueError('timing sidecar must be a file')
        files.append(timing)
    hud = path / 'hud'
    if hud.is_symlink():
        raise ValueError('HUD directory symlinks are forbidden')
    if hud.exists():
        if not hud.is_dir():
            raise ValueError('HUD must be a directory')
        files.extend(sorted(hud.iterdir()))
    return files

TIMING = {
    'timestamp_meaning': 'relative perf_counter at get_latest_frame return, not capture',
    'time_resolution_ns': 100_000,
    'time_rounding_error_bound_ns': 50_000,
    'age_resolution_ns': 100_000,
    'age_rounding_error_bound_ns': 50_000,
    'image_age_bound_ns': None,
    'cached_frames_possible': True,
    'label_alignment': 'recorded per-frame values only; no interpolation or shift',
}


def _decimal(value, key, places=None):
    if places is not None and not re.fullmatch(rf'-?\d+\.\d{{{places}}}', value or ''):
        raise ValueError(f'{key}: expected recorder decimal precision ({places} places)')
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f'invalid {key}') from error
    if not number.is_finite():
        raise ValueError(f'{key} must be finite')
    return number


def _integer(value, key):
    if isinstance(value, bool) or not re.fullmatch(r'\d+', str(value)):
        raise ValueError(f'{key} must be a nonnegative integer')
    return int(value)


def _read_source(path, *, diagnostic_issues=None):
    """Require artifacts written after normal writer closure; no recovery inference."""
    path = Path(path).resolve()
    for name in ['meta.json', 'labels.csv', 'frames']:
        entry = path / name
        if entry.is_symlink() or not entry.exists():
            raise ValueError(f'{name} missing or symlinked; recording must be normally closed')
    metadata = json.loads((path / 'meta.json').read_text())
    if 'producer_schema' in metadata and metadata['producer_schema'] != 'record_py_buffered_20_v1':
        raise ValueError('unsupported producer_schema')
    if 'producer_sha256' in metadata and (not isinstance(metadata['producer_sha256'], str)
            or re.fullmatch(r'[0-9a-f]{64}', metadata['producer_sha256']) is None):
        raise ValueError('producer_sha256 must be 64 lowercase hexadecimal characters')
    if metadata.get('completed') is False:
        raise ValueError('source explicitly marks recording incomplete')
    if metadata.get('telemetry') is not True:
        raise ValueError('--no-telemetry recordings cannot train an image+speed model')
    session = metadata.get('session')
    if not isinstance(session, str) or not session.strip():
        raise ValueError('meta.json needs a nonempty session')
    if _decimal(metadata.get('steer_units_per_deg'), 'steer_units_per_deg') != 73:
        raise ValueError('unsupported steering calibration; expected 73 units/degree')
    count = _integer(metadata.get('frames'), 'frames')
    segments = _integer(metadata.get('segments'), 'segments')
    if count == 0 or segments == 0:
        raise ValueError('recording must contain frames and segments')
    dropped = _integer(metadata.get('dropped'), 'dropped')
    if _decimal(metadata.get('fps_target'), 'fps_target') <= 0:
        raise ValueError('fps_target must be positive')
    config = metadata.get('config', {})
    crop = config.get('crop')
    if not isinstance(crop, list) or len(crop) != 4 or any(type(x) is not int for x in crop):
        raise ValueError('config.crop must be four integer coordinates')
    l, t, r, b = crop
    if not (0 <= l < r and 0 <= t < b):
        raise ValueError('invalid crop rectangle')
    width = _integer(config.get('save_width'), 'save_width')
    _integer(config.get('monitor'), 'monitor')
    masks = config.get('masks')
    if not isinstance(masks, list) or any(not isinstance(m, list) or len(m) != 4
            or any(type(x) is not int for x in m) or not (0 <= m[0] < m[2] and 0 <= m[1] < m[3]) for m in masks):
        raise ValueError('config.masks must contain integer rectangles')
    size = metadata.get('saved_size')
    expected_size = [width, int(round(width * (b - t) / (r - l) / 2)) * 2]
    if (not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size)
            or size != expected_size):
        raise ValueError('saved_size does not match recorder crop/save_width calculation')
    with (path / 'labels.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        profile = _buffer_profile(metadata, columns)
        if metadata.get('producer_schema') == 'record_py_buffered_20_v1' and (
                columns != COLUMNS + EXTENDED_SUFFIXES[-1] or profile is None
                or profile['family'] != 'buffered_takeover'):
            raise ValueError('producer_schema does not match columns/buffering metadata')
        if columns != COLUMNS and profile is None:
            extras = columns[len(COLUMNS):]
            if (diagnostic_issues is None or columns[:len(COLUMNS)] != COLUMNS
                    or not extras or len(set(columns)) != len(columns)
                    or not set(extras) <= DIAGNOSTIC_EXTRA_COLUMNS):
                raise ValueError('labels.csv columns do not match record.py')
            diagnostic_issues.append('unsupported producer extension columns: ' + ', '.join(extras))
        rows = list(reader)
    if len(rows) != count:
        raise ValueError('meta.json frames count does not match labels.csv; incomplete recording')
    actual_images = {p.name for p in (path / 'frames').iterdir()}
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('malformed labels.csv row')
    frame_ids = [_integer(row['frame'], 'frame') for row in rows]
    if any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
        raise ValueError('frame indexes must be unique and strictly increasing')
    missing_ids = frame_ids[-1] + 1 - count
    if profile is None and frame_ids != list(range(count)):
        raise ValueError('frame indexes must be contiguous from zero')
    if 'accepted_frame_count' in metadata:
        acquired = _integer(metadata['accepted_frame_count'], 'accepted_frame_count')
        accounted = count + dropped + (profile['discarded'] + profile['stopped'] if profile else 0)
        if acquired != accounted or frame_ids[-1] >= acquired:
            raise ValueError('accepted_frame_count does not reconcile with saved/discarded frames')
    if profile is not None and missing_ids > dropped + profile['discarded']:
        raise ValueError('frame index gaps exceed queue-drop/rewind/takeover counts')
    expected_images = {f'{index:06d}.jpg' for index in frame_ids}
    if actual_images != expected_images:
        raise ValueError('missing or extra frame files; cannot infer completed recording')
    previous_t, previous_segment = Decimal('-1'), -1
    observed_segments = set()
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError('malformed labels.csv row')
        frame_id = frame_ids[index]
        segment = _integer(row['segment'], 'segment')
        time = _decimal(row['t'], 't', 4)
        if time < 0 or time < previous_t or segment < previous_segment or segment >= segments:
            raise ValueError('time/segment ordering inconsistent with recorder')
        previous_t, previous_segment = time, segment
        observed_segments.add(segment)
        # Equal rounded times are legitimate: retain their precision and exclude
        # duplicate times later, never synthesize strictly increasing nanoseconds.
        raw = _decimal(row['steer_raw'], 'steer_raw')
        angle = _decimal(row['steer_deg'], 'steer_deg', 2)
        if raw != raw.to_integral_value() or not -32768 <= raw <= 32767 or abs(angle - raw / 73) > Decimal('.005001'):
            raise ValueError('steer_raw/steer_deg calibration mismatch')
        for key in ['brake', 'gas']:
            value = _decimal(row[key], key, 4)
            if not 0 <= value <= 1:
                raise ValueError(f'{key} outside [0, 1]')
        for key in ['wheel_age_ms', 'tele_age_ms']:
            _decimal(row[key], key, 1)
        if _decimal(row['speed_mps'], 'speed_mps', 3) < 0:
            raise ValueError('negative speed')
        if row['race_on'] not in {'0', '1'}:
            raise ValueError('race_on must be 0 or 1')
        steer = _decimal(row['tele_steer'], 'tele_steer')
        if steer != steer.to_integral_value() or not -128 <= steer <= 127:
            raise ValueError('invalid telemetry steer')
        for key, places in [('race_time', 3), ('distance', 1), ('yaw_rate', 4)]:
            if key in row:
                _decimal(row[key], key, places)
        for key, upper in [('game_ms', 2**32 - 1), ('gear', 255),
                           ('car_ordinal', 2**31 - 1), ('car_class', 7), ('car_pi', 2**31 - 1)]:
            if key in row and _integer(row[key], key) > upper:
                raise ValueError(f'invalid {key}')
        image_path = path / 'frames' / f'{frame_id:06d}.jpg'
        if image_path.is_symlink() or not image_path.is_file():
            raise ValueError('frame symlinks and non-files are forbidden')
        with Image.open(image_path) as image:
            if image.format != 'JPEG' or list(image.size) != size:
                raise ValueError('frame format/dimensions differ from meta.json')
            image.convert('RGB').load()
    # record.py opens a segment before queue.put_nowait. A segment with only
    # dropped frames can be absent anywhere, including after the last saved row.
    # Every such segment requires at least one recorded queue.Full drop.
    empty_segments = segments - len(observed_segments)
    legacy_interrupt = (set(metadata) == LEGACY_METADATA_KEYS
                        and previous_segment == segments - 2
                        and empty_segments == dropped + 1)
    budget = dropped + (profile['discarded'] + profile['stopped'] if profile else 0)
    if profile and previous_segment + 1 - len(observed_segments) > missing_ids:
        raise ValueError('empty earlier segments lack missing frame-index evidence')
    trailing_empty = segments - previous_segment - 1
    if profile and missing_ids + trailing_empty > budget:
        raise ValueError('missing frame IDs and trailing segments exceed total discard evidence')
    if empty_segments > budget and not legacy_interrupt:
        issue = 'meta.json empty segments exceed dropped-frame evidence'
        if diagnostic_issues is None:
            raise ValueError(issue)
        diagnostic_issues.append(issue)
    advertised_timing = metadata.get('capture_provenance', {}).get('timing_file')
    if advertised_timing is not None and (advertised_timing != 'capture_timing.csv'
            or not (path / advertised_timing).is_file()):
        raise ValueError('advertised timing sidecar missing or unsupported')
    expected_hud = {f'{i:06d}' for i in frame_ids}
    for artifact in _auxiliary_files(path):
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError('auxiliary symlinks/non-files are forbidden')
        if artifact.name == 'capture_timing.csv':
            with artifact.open(newline='') as handle:
                timing_rows = list(csv.DictReader(handle))
            if (len(timing_rows) != count or any('frame' not in row or None in row
                    or any(v is None for v in row.values()) for row in timing_rows)
                    or [_integer(row['frame'], 'timing frame') for row in timing_rows] != frame_ids):
                raise ValueError('timing sidecar frame coverage differs from labels')
        if artifact.parent == path / 'hud':
            if artifact.suffix != '.png' or artifact.stem not in expected_hud:
                raise ValueError('unexpected HUD patch filename')
            with Image.open(artifact) as image:
                if image.format != 'PNG':
                    raise ValueError('HUD patch must be PNG')
                image.load()
    return metadata, rows


def import_recording(source, destination, *, expert_mode=None, exclude_sessions=None):
    """Copy one complete recording, retaining original CSV/JPEG bytes and metadata."""
    from forza_ai.data.sessions import load_session

    if expert_mode != 'manual':
        raise ValueError('import requires --expert-mode manual; assist labels are not expert demonstrations')
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink() or destination.resolve().is_relative_to(source):
        raise ValueError('destination must be new and outside the source recording')
    metadata, _ = _read_source(source)
    from forza_ai.data.exclusions import enforce_exclusions
    exclusions = enforce_exclusions(metadata['session'], additional=exclude_sessions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='forza-import-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'recording'
        staging.mkdir()
        shutil.copyfile(source / 'meta.json', staging / 'meta.json')
        shutil.copyfile(source / 'labels.csv', staging / 'labels.csv')
        shutil.copytree(source / 'frames', staging / 'frames')
        for artifact in _auxiliary_files(source):
            target = staging / artifact.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact, target)
        group = 'record.py:' + metadata['session']
        manifest = {
            'schema_version': FORMAT, 'session_id': group, 'split_group': group,
            'image_stage': 'road_crop', 'completed': True,
            'completion_evidence': 'meta.json after normal closure plus matching rows, image count and dimensions',
            'expert_mode': 'manual', 'exclusion_prefixes_checked': exclusions,
            'timing': TIMING, 'source_metadata': metadata,
        }
        (staging / 'metadata.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
        validated = load_session(staging)
        if not validated.samples:
            raise ValueError('no eligible recorded labels after conservative age/duplicate filtering')
        staging.rename(destination)
    return validated.summary()


def load_recording(path, metadata, alignment):
    from forza_ai.data.sessions import Sample, Session

    if alignment.label_offset_ns != 0:
        raise ValueError('record.py imports support only zero label offset; rounded rows cannot be realigned')
    source_meta, rows = _read_source(path)
    group = 'record.py:' + source_meta['session']
    if (metadata.get('expert_mode') != 'manual'
            or metadata.get('completed') is not True
            or metadata.get('session_id') != group or metadata.get('split_group') != group
            or metadata.get('timing') != TIMING or metadata.get('source_metadata') != source_meta):
        raise ValueError('invalid imported-recording provenance')
    return _aligned_session(path, source_meta, rows, alignment, ['metadata.json', 'meta.json', 'labels.csv'])


def _aligned_session(path, source_meta, rows, alignment, fingerprint_files):
    from forza_ai.data.sessions import Sample, Session

    group = 'record.py:' + source_meta['session']
    accepted, rejected = [], Counter()
    digest = hashlib.sha256()
    for name in list(fingerprint_files) + [p.relative_to(path).as_posix() for p in _auxiliary_files(path)]:
        digest.update(name.encode())
        digest.update((path / name).read_bytes())
    previous_time, previous_image = None, None
    for row in rows:
        image = path / 'frames' / f"{int(row['frame']):06d}.jpg"
        content = image.read_bytes()
        image_hash = hashlib.sha256(content).digest()
        digest.update(image.name.encode())
        digest.update(image_hash)
        time_ns = int(Decimal(row['t']) * 1_000_000_000)  # quantized elapsed time, NOT certified capture
        duplicate_time, duplicate_image = time_ns == previous_time, image_hash == previous_image
        previous_time, previous_image = time_ns, image_hash
        if row['race_on'] != '1':
            rejected['race_off'] += 1
            continue
        if duplicate_time or duplicate_image:
            rejected['duplicate_rounded_time_or_image'] += 1
            continue
        wheel_age = Decimal(row['wheel_age_ms']) * 1_000_000
        telemetry_age = Decimal(row['tele_age_ms']) * 1_000_000
        # Ages rounded to .1ms: include the full +/- .05ms uncertainty interval.
        # Zero/negative rounded age cannot certify the input existed at retrieval.
        if min(wheel_age, telemetry_age) - 50_000 < 0:
            rejected['ambiguous_or_future_sample_age'] += 1
            continue
        if wheel_age + 50_000 > alignment.max_wheel_gap_ns:
            rejected['stale_wheel'] += 1
            continue
        if telemetry_age + 50_000 > alignment.max_telemetry_age_ns:
            rejected['stale_telemetry'] += 1
            continue
        accepted.append(Sample(image, time_ns, float(row['steer_deg']), float(row['speed_mps']), 'manual',
                               float(row['gas']), float(row['brake'])))
    profile = _buffer_profile(source_meta, list(rows[0]))
    provenance = {
        'format': FORMAT, 'timing': TIMING, 'source_session': source_meta['session'],
        'segments': source_meta['segments'], 'capture_config': source_meta['config'],
        'saved_size': source_meta['saved_size'], 'expert_basis': 'explicit --expert-mode manual',
        'buffering': profile,
        'auxiliary_files_preserved': [p.relative_to(path).as_posix() for p in _auxiliary_files(path)],
        'additional_columns_not_model_inputs': list(rows[0])[len(COLUMNS):],
        'source_identity': 'schema-compatible; exact executed commit not recorded',
        'declared_producer_schema': source_meta.get('producer_schema'),
        'declared_producer_sha256': source_meta.get('producer_sha256'),
        'empty_segment_evidence': ('recorded queue drops plus rewind/takeover/stop discards' if profile else ('legacy KeyboardInterrupt may leave one unqueued trailing segment'
            if source_meta['segments'] - len({r['segment'] for r in rows}) > source_meta['dropped']
            and set(source_meta) == LEGACY_METADATA_KEYS else 'recorded dropped-frame counts')),
    }
    return Session(path, group, accepted, dict(rejected), digest.hexdigest(), group, provenance)


def inspect_recording_for_diagnostics(source, *, expert_mode=None, alignment=None):
    """Read unchanged rows for an isolated offline smoke; never certify a session.

    Only two known diagnostic incompatibilities are tolerated: the named additive
    telemetry columns and unexplained empty segment counts. Image integrity,
    row count/order/ranges, freshness filtering, and manual provenance still apply.
    No files are written. The returned object is barred from production splitting.
    """
    from forza_ai.data.sessions import Alignment

    if expert_mode != 'manual':
        raise ValueError('diagnostics require explicit expert_mode="manual"')
    alignment = alignment or Alignment()
    if alignment.label_offset_ns != 0:
        raise ValueError('diagnostic recorded labels support only zero label offset')
    path = Path(source).resolve()
    issues = []
    source_meta, rows = _read_source(path, diagnostic_issues=issues)
    result = _aligned_session(path, source_meta, rows, alignment, ['meta.json', 'labels.csv'])
    result.provenance.update({
        'format': 'record_py_diagnostic_only', 'diagnostic_only': True,
        'production_validation_passed': False, 'strict_validation_issues': issues,
        'extra_columns_ignored': sorted(set(rows[0]) - set(COLUMNS)),
        'completion': 'not certified by this diagnostic path',
        'empty_segment_evidence': 'unresolved' if any('segments' in issue for issue in issues)
                                  else result.provenance['empty_segment_evidence'],
    })
    if not result.samples:
        raise ValueError('diagnostic recording has no eligible samples')
    return result
