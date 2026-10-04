"""Fresh capture, timing compatibility, and failure closure without native devices."""

import importlib.util
import csv
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def record():
    path = Path(__file__).resolve().parents[1] / "record.py"
    spec = importlib.util.spec_from_file_location("standalone_recorder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.now = 1_000_000_000
        self.sleeps = []

    def ns(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += round(seconds * 1e9)


class Camera:
    is_capturing = False

    def __init__(self, clock, frames):
        self.clock, self.frames = clock, iter(frames)
        self.calls = []
        self.released = False

    def grab(self, **kwargs):
        self.calls.append(kwargs)
        self.clock.now += 1_000_000
        try:
            return next(self.frames)
        except StopIteration:
            raise KeyboardInterrupt

    def release(self):
        self.released = True


def image(value):
    return np.full((4, 8, 3), value, dtype=np.uint8)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def install_hardware_mocks(record, monkeypatch, tmp_path, frames, *, active=True,
                           buttons=None, backjump=None, config=None, telemetry_steer=None):
    clock = Clock()
    camera = Camera(clock, frames)
    instances = []

    class Reader:
        error = None
        live = True

        def __init__(self, arg):
            self.stop = threading.Event()
            self.ready = threading.Event()
            self.joined = False
            instances.append(self)

        def start(self):
            self.ready.set()

        def join(self, timeout):
            assert self.stop.is_set()
            self.joined = True

        def is_alive(self):
            return False

    class Wheel(Reader):
        @property
        def buttons(self):
            return frozenset(buttons(clock.now, len(camera.calls)) if buttons else ())

        @property
        def latest_ns(self):
            return (clock.now - 2_000_000, 730, 32767, 6553)

    class Telemetry(Reader):
        @property
        def last_backjump(self):
            return backjump(clock.now, len(camera.calls)) if backjump else -1e9

        @property
        def latest_ns(self):
            race_on = active(clock.now) if callable(active) else active
            steer = telemetry_steer(len(camera.calls)) if telemetry_steer else 0
            return (clock.now - 3_000_000, int(race_on), 15.0, steer,
                    12.5, 100.0, .125, 12000, 3, 234, 5, 800)

    monkeypatch.setitem(sys.modules, "dxcam", SimpleNamespace(create=lambda **kwargs: camera))
    monkeypatch.setattr(record, "WheelReader", Wheel)
    monkeypatch.setattr(record, "TelemetryReader", Telemetry)
    monkeypatch.setattr(record, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(record, "time", SimpleNamespace(perf_counter_ns=clock.ns, perf_counter=lambda: clock.now / 1e9, sleep=clock.sleep))
    monkeypatch.setattr(record, "load_config", lambda: {
        "monitor": 0, "crop": [0, 0, 8, 4], "masks": [[0, 0, 2, 2]],
        "save_width": 8, **(config or {})})
    return clock, camera, instances


def test_default_thirty_and_fps_validation(record):
    assert record.argument_parser().parse_args(["record"]).fps == 30
    assert record.argument_parser().parse_args(["record", "--fps", "60"]).fps == 60
    assert record.argument_parser().parse_args(["record", "--fps", "59.94"]).fps == 59.94
    for value in ["nan", "inf", "-1", "0", "241", "bad"]:
        with pytest.raises(SystemExit):
            record.argument_parser().parse_args(["record", "--fps", value])
    assert record.capture_fps(240) == 240


def test_fresh_one_shot_capture_never_reuses_none_and_paces_every_poll(record):
    clock = Clock()
    first, third = image(10), image(30)
    camera = Camera(clock, [first, None, third, None])
    capture = record.FreshCapture(camera, [0, 0, 8, 4], 60,
                                  clock_ns=clock.ns, sleep=clock.sleep)
    results = [capture.grab() for _ in range(4)]
    assert results[0][0] is first and results[1][0] is None
    assert results[2][0] is third and results[3][0] is None
    assert len(capture.times) == 2
    assert capture.attempts == 4 and capture.no_frame_polls == 2
    assert len(clock.sleeps) == 3
    assert all(call == {"region": (0, 0, 8, 4), "copy": True, "new_frame_only": True}
               for call in camera.calls)
    assert [result[2] - result[1] for result in results] == [1_000_000] * 4
    assert all(b[1] - a[1] >= 16_666_666 for a, b in zip(results, results[1:]))


def test_active_camera_is_rejected_and_not_released_by_recorder(record, monkeypatch, tmp_path):
    _, camera, readers = install_hardware_mocks(record, monkeypatch, tmp_path, [])
    camera.is_capturing = True
    with pytest.raises(RuntimeError, match="already capturing"):
        record.main(["record", "--drop-seconds", "0"])
    assert not camera.released
    assert all(reader.joined for reader in readers)
    assert not list(tmp_path.glob("recordings/*/meta.json"))


def test_source_changes_to_ring_buffer_are_rejected(record):
    clock = Clock()
    camera = Camera(clock, [image(10)])
    capture = record.FreshCapture(camera, [0, 0, 8, 4], 60, clock_ns=clock.ns, sleep=clock.sleep)
    camera.is_capturing = True
    with pytest.raises(RuntimeError, match="remain stopped"):
        capture.grab()
    assert not camera.calls


def test_measured_fps_and_gaps_use_fresh_and_saved_frames(record):
    capture = SimpleNamespace(attempts=6, no_frame_polls=3,
                              times=[(0, 1_000_000_000), (0, 1_020_000_000), (0, 1_080_000_000)])
    writer = SimpleNamespace(saved=2, capture_times=[(0, 1.0), (0, 1.08)])
    report = record.measured_capture_report(capture, writer, 100_000_000, 60, 1, 0)
    assert report["captured_fps"] == 30
    assert report["saved_fps"] == 20
    assert report["fresh_frames"] == 3 and report["saved_frames"] == 2
    assert report["no_new_frame_polls"] == 3
    assert report["fresh_capture_gaps"]["max_ms"] == 60
    assert report["fresh_capture_gaps"]["over_2x"] == 1
    assert report["saved_capture_gaps"]["max_ms"] == 80
    assert report["dropped_queue_frames"] == 1


@pytest.mark.parametrize("requested_fps", [None, 60])
def test_recording_keeps_legacy_columns_and_imports_without_cached_frames(record, monkeypatch, tmp_path, requested_fps):
    clock, camera, readers = install_hardware_mocks(
        record, monkeypatch, tmp_path, [image(40), None, image(80), None, image(120)])
    target_fps = 30 if requested_fps is None else requested_fps
    record.main(["record", "--drop-seconds", "0"] + ([] if requested_fps is None else ["--fps", str(requested_fps)]))
    session = next((tmp_path / "recordings").iterdir())
    labels = read_csv(session / "labels.csv")
    timing = read_csv(session / "capture_timing.csv")
    metadata = json.loads((session / "meta.json").read_text())
    assert list(labels[0]) == record.COLUMNS
    assert len(labels) == len(timing) == metadata["frames"] == 3
    assert [int(r["capture_index"]) for r in timing] == [0, 1, 2]
    assert [r["frame"] for r in labels] == ["0", "1", "2"]
    assert all(float(r["t"]) == pytest.approx((int(t["retrieved_ns"]) - metadata["capture_provenance"]["origin_ns"]) / 1e9,
                                              abs=.00005) for r, t in zip(labels, timing))
    assert all(int(t["retrieved_ns"]) - int(t["capture_start_ns"]) == 1_000_000 for t in timing)
    assert labels[0]["t"] == "0.0010"  # Retrieval, not capture-start t=0.
    assert labels[0]["wheel_age_ms"] == "2.0"
    assert labels[0]["tele_age_ms"] == "3.0"
    assert metadata["fps_target"] == target_fps
    assert metadata["completed"] is True
    assert metadata["capture_provenance"]["cached_frames_reused"] is False
    assert metadata["measured_capture"]["no_new_frame_polls"] == 2
    assert metadata["measured_capture"]["fresh_frames"] == 3
    assert metadata["measured_capture"]["captured_fps"] < target_fps
    assert metadata["measured_capture"]["fresh_capture_gaps"]["median_ms"] == round(2000 / target_fps, 1)
    assert camera.released and all(reader.joined for reader in readers)
    assert len(clock.sleeps) == 5
    from forza_ai.data.recording import import_recording
    imported = import_recording(session, tmp_path / "imported", expert_mode="manual")
    assert imported["accepted"] == 3
    assert (tmp_path / "imported" / "labels.csv").read_bytes() == (session / "labels.csv").read_bytes()


def test_same_crop_masks_and_label_scaling(record):
    cfg = {"crop": [2, 4, 10, 8], "save_width": 8, "masks": [[2, 4, 4, 6]]}
    pixels = image(100)
    actual = record.process(pixels, record.masks_in_crop(cfg), record.save_size(cfg))
    assert not actual[:2, :2].any()
    assert np.all(actual[2:, 2:] == 100)
    assert np.all(pixels == 100)
    row = record.make_record_row(0, 0, 1_010_000_000, 1_000_000_000,
                                 (1_008_000_000, 730, 32767, -32768),
                                 (1_007_000_000, 1, 15, -1, 12.5, 100, .125, 12000, 3, 234, 5, 800))
    assert dict(zip(record.COLUMNS, row)) == {
        "frame": 0, "segment": 0, "t": "0.0100", "steer_raw": 730,
        "steer_deg": "10.00", "brake": "0.0000", "gas": "1.0000",
        "wheel_age_ms": "2.0", "speed_mps": "15.000", "race_on": 1,
        "tele_steer": -1, "tele_age_ms": "3.0", "race_time": "12.500",
        "distance": "100.0", "yaw_rate": "0.1250", "game_ms": 12000,
        "gear": 3, "car_ordinal": 234, "car_class": 5, "car_pi": 800}


def test_pause_during_no_frame_poll_still_starts_a_new_segment(record, monkeypatch, tmp_path):
    install_hardware_mocks(record, monkeypatch, tmp_path, [image(40), None, image(80)],
                           active=lambda now: not 1_020_000_000 < now < 1_050_000_000)
    record.main(["record", "--drop-seconds", "0"])
    session = next((tmp_path / "recordings").iterdir())
    labels = read_csv(session / "labels.csv")
    assert [row["segment"] for row in labels] == ["0", "1"]
    assert json.loads((session / "meta.json").read_text())["segments"] == 2


def test_queue_drop_keeps_original_ids_and_does_not_open_empty_segment(record, monkeypatch, tmp_path):
    import queue
    original_writer = record.Writer
    class DropFirstQueue(queue.Queue):
        dropped = False
        def put_nowait(self, item):
            if not self.dropped:
                self.dropped = True
                raise queue.Full
            return super().put_nowait(item)
    def writer(*args):
        result = original_writer(*args)
        result.q = DropFirstQueue(maxsize=300)
        return result
    monkeypatch.setattr(record, "Writer", writer)
    install_hardware_mocks(record, monkeypatch, tmp_path, [image(40), image(80), image(120)])
    record.main(["record", "--drop-seconds", "0"])
    session = next((tmp_path / "recordings").iterdir())
    labels = read_csv(session / "labels.csv")
    metadata = json.loads((session / "meta.json").read_text())
    assert [row["frame"] for row in labels] == ["1", "2"]
    assert [row["segment"] for row in labels] == ["0", "0"]
    assert metadata["dropped"] == metadata["measured_capture"]["dropped_queue_frames"] == 1
    assert metadata["frames"] == 2 and metadata["segments"] == 1
    from forza_ai.data.recording import import_recording
    assert import_recording(session, tmp_path / "imported", expert_mode="manual")["accepted"] == 2


def test_no_frames_produces_zero_achieved_rate_and_no_labels(record, monkeypatch, tmp_path):
    _, camera, _ = install_hardware_mocks(record, monkeypatch, tmp_path, [None] * 4)
    record.main(["record", "--drop-seconds", "0"])
    session = next((tmp_path / "recordings").iterdir())
    metadata = json.loads((session / "meta.json").read_text())
    assert metadata["frames"] == metadata["segments"] == 0
    assert metadata["measured_capture"]["captured_fps"] == 0
    assert metadata["measured_capture"]["saved_fps"] == 0
    assert metadata["measured_capture"]["no_new_frame_polls"] == 4
    assert read_csv(session / "labels.csv") == []
    assert camera.released


def test_writer_failure_leaves_no_completion_metadata_and_releases_hardware(record, monkeypatch, tmp_path):
    _, camera, readers = install_hardware_mocks(record, monkeypatch, tmp_path, [image(40)])
    monkeypatch.setattr(record.cv2, "imwrite", lambda *args: False)
    with pytest.raises(RuntimeError, match="JPEG write failed"):
        record.main(["record", "--drop-seconds", "0"])
    assert not list(tmp_path.glob("recordings/*/meta.json"))
    assert camera.released and all(reader.joined for reader in readers)


def test_capture_error_leaves_no_completion_metadata(record, monkeypatch, tmp_path):
    _, camera, readers = install_hardware_mocks(record, monkeypatch, tmp_path, [])
    def fail(**kwargs):
        raise RuntimeError("capture disconnected")
    camera.grab = fail
    with pytest.raises(RuntimeError, match="capture disconnected"):
        record.main(["record", "--drop-seconds", "0"])
    assert not list(tmp_path.glob("recordings/*/meta.json"))
    assert camera.released and all(reader.joined for reader in readers)


def test_writer_shutdown_timeout_does_not_block_forever(record, tmp_path):
    writer = record.Writer(str(tmp_path), 90)
    entered, release = threading.Event(), threading.Event()
    def hung():
        entered.set()
        release.wait(2)
    writer.run = hung
    writer.start()
    assert entered.wait(1)
    try:
        with pytest.raises(RuntimeError, match="shutdown deadline"):
            writer.finish(timeout=.01)
    finally:
        release.set()
        writer.join(1)
    assert not (tmp_path / "meta.json").exists()


def assert_frame_accounting(metadata):
    assert metadata["producer_schema"] == "record_py_buffered_20_v1"
    producer_path = Path(__file__).resolve().parents[1] / "record.py"
    assert metadata["producer_sha256"] == hashlib.sha256(producer_path.read_bytes()).hexdigest()
    assert metadata["accepted_frame_count"] == (
        metadata["frames"] + metadata["dropped"]
        + metadata["discarded_by_rewind_or_takeover"] + metadata["discarded_at_stop"])
    assert metadata["measured_capture"]["saved_frames"] == metadata["frames"]
    assert metadata["measured_capture"]["fresh_not_saved"] == (
        metadata["measured_capture"]["fresh_frames"] - metadata["frames"])


def test_default_holdback_discards_final_frames_without_false_saved_count(record, monkeypatch, tmp_path):
    # 10 s: real sessions showed mistakes building up to ~12 s before a reset.
    assert record.argument_parser().parse_args(["record"]).drop_seconds == 10
    _, camera, readers = install_hardware_mocks(
        record, monkeypatch, tmp_path, [image(40), image(80), image(120)])
    record.main(["record"])
    session = next((tmp_path / "recordings").iterdir())
    metadata = json.loads((session / "meta.json").read_text())
    assert metadata["completed"] is True
    assert metadata["frames"] == metadata["dropped"] == 0
    assert metadata["segments"] == 1  # Producer intervals include this discarded segment.
    assert metadata["accepted_frame_count"] == metadata["discarded_at_stop"] == 3
    assert metadata["discarded_by_rewind_or_takeover"] == 0
    assert read_csv(session / "labels.csv") == read_csv(session / "capture_timing.csv") == []
    assert list((session / "frames").glob("*.jpg")) == []
    assert metadata["measured_capture"]["saved_capture_gaps"] is None
    assert_frame_accounting(metadata)
    assert camera.released and all(reader.joined for reader in readers)


@pytest.mark.parametrize("rewind_source, expected_ids", [
    ("button", [2, 3, 4]), ("telemetry", [2, 3]), ("disabled", list(range(6)))])
def test_rewind_clears_pending_preserves_ids_and_excludes_playback(
        record, monkeypatch, tmp_path, rewind_source, expected_ids):
    # At 5 FPS the third grab occurs at t=1.401 s. The auto-rewind watchdog
    # suppresses grabs three and four; a button press suppresses grab three only.
    frames = [image(value) for value in range(40, 120, 10)]
    buttons = (lambda now, calls: {7} if calls == 3 else set()) if rewind_source == "button" else None
    backjump = (lambda now, calls: 1.401 if calls >= 3 else -1e9) if rewind_source != "button" else None
    install_hardware_mocks(record, monkeypatch, tmp_path, frames, buttons=buttons, backjump=backjump)
    args = ["record", "--fps", "5", "--drop-seconds", ".3"]
    args += ["--rewind-button", "7"] if rewind_source == "button" else []
    args += ["--no-auto-rewind"] if rewind_source == "disabled" else []
    record.main(args)
    session = next((tmp_path / "recordings").iterdir())
    metadata = json.loads((session / "meta.json").read_text())
    labels, timing = read_csv(session / "labels.csv"), read_csv(session / "capture_timing.csv")
    assert [int(row["frame"]) for row in labels] == expected_ids
    assert [row["frame"] for row in timing] == [row["frame"] for row in labels]
    assert sorted(path.name for path in (session / "frames").glob("*.jpg")) == [
        f"{frame_id:06d}.jpg" for frame_id in expected_ids]
    assert metadata["frames"] == len(expected_ids)
    assert metadata["discarded_at_stop"] == 2
    assert metadata["rewinds"] == (0 if rewind_source == "disabled" else 1)
    assert metadata["discarded_by_rewind_or_takeover"] == (0 if rewind_source == "disabled" else 2)
    assert metadata["dropped"] == 0
    assert metadata["segments"] == (1 if rewind_source == "disabled" else 2)
    assert metadata["completed"] is True
    if rewind_source != "disabled":
        # Segment IDs retain the discarded original interval, without renumbering.
        assert {row["segment"] for row in labels} == {"1"}
        assert float(labels[0]["t"]) >= (.6 if rewind_source == "button" else .8)
    assert_frame_accounting(metadata)


def test_hud_and_road_are_captured_once_from_union_region(record, monkeypatch, tmp_path):
    config = {"crop": [2, 4, 10, 8], "hud_box": [11, 2, 13, 5],
              "masks": [[2, 4, 4, 6]], "save_width": 8}
    union = np.full((6, 11, 3), 100, dtype=np.uint8)
    union[:3, 9:11] = [10, 20, 30]
    _, camera, _ = install_hardware_mocks(record, monkeypatch, tmp_path, [union], config=config)
    encoded = {}
    original_imwrite = record.cv2.imwrite
    def remember_write(path, pixels, *args):
        encoded[Path(path).suffix] = pixels.copy()
        return original_imwrite(path, pixels, *args)
    monkeypatch.setattr(record.cv2, "imwrite", remember_write)
    record.main(["record", "--drop-seconds", "0"])
    session = next((tmp_path / "recordings").iterdir())
    assert camera.calls == [{"region": (2, 2, 13, 8), "copy": True, "new_frame_only": True}] * 2
    assert encoded[".jpg"].shape == (4, 8, 3)
    assert not encoded[".jpg"][:2, :2].any()
    assert np.all(encoded[".jpg"][2:, :] == 100)
    assert encoded[".png"].shape == (3, 2, 3)
    assert np.all(encoded[".png"] == [10, 20, 30])
    assert np.array_equal(record.cv2.imread(str(session / "hud" / "000000.png")), union[:3, 9:11])
    assert read_csv(session / "labels.csv")[0]["gear"] == "3"
    assert_frame_accounting(json.loads((session / "meta.json").read_text()))


def test_hud_write_failure_prevents_completion_and_label_commit(record, monkeypatch, tmp_path):
    _, camera, readers = install_hardware_mocks(
        record, monkeypatch, tmp_path, [image(40)], config={"hud_box": [4, 0, 6, 2]})
    original_imwrite = record.cv2.imwrite
    monkeypatch.setattr(record.cv2, "imwrite", lambda path, pixels, *args:
                        False if str(path).endswith(".png") else original_imwrite(path, pixels, *args))
    with pytest.raises(RuntimeError, match="(?i)(HUD|PNG).*write failed"):
        record.main(["record", "--drop-seconds", "0"])
    session = next((tmp_path / "recordings").iterdir())
    assert not (session / "meta.json").exists()
    assert read_csv(session / "labels.csv") == []
    assert read_csv(session / "capture_timing.csv") == []
    assert camera.released and all(reader.joined for reader in readers)


def test_without_telemetry_preserves_twenty_columns_and_empty_vehicle_fields(record):
    row = record.make_record_row(9, 2, 1_010_000_000, 1_000_000_000,
                                 (1_008_000_000, 730, 32767, -32768), None)
    assert len(row) == len(record.COLUMNS) == 20
    assert row[:8] == [9, 2, "0.0100", 730, "10.00", "0.0000", "1.0000", "2.0"]
    assert row[8:] == [""] * 12


def test_telemetry_retains_extended_fields_precise_clock_and_rewind_watchdog(record, monkeypatch):
    import struct
    def packet(race_time, speed=15.0):
        data = bytearray(324)
        for fmt, offset, value in [
                ("<i", 0, 1), ("<I", 4, 12000), ("<f", 48, .125),
                ("<i", 212, 234), ("<i", 216, 5), ("<i", 220, 800),
                ("<f", 256, speed), ("<f", 292, 100), ("<f", 308, race_time),
                ("<B", 319, 3), ("<b", 320, -7)]:
            struct.pack_into(fmt, data, offset, value)
        return bytes(data)
    packets = iter([packet(12.5), packet(11, speed=float("nan")), packet(11)])
    stamps = iter([1_000_000_000, 2_000_000_000, 3_000_000_000])
    state, snapshots = {}, []
    class Socket:
        closed = False
        def bind(self, address):
            assert address == ("127.0.0.1", 9999)
        def settimeout(self, timeout):
            assert timeout == .2
        def recvfrom(self, limit):
            assert limit == 1024
            reader = state["reader"]
            snapshots.append((reader.latest_ns, reader.last_backjump))
            try:
                return next(packets), ("127.0.0.1", 40000)
            except StopIteration:
                reader.stop.set()
                raise TimeoutError
        def close(self):
            self.closed = True
    sock = Socket()
    monkeypatch.setattr(record, "socket", SimpleNamespace(
        AF_INET=1, SOCK_DGRAM=2, socket=lambda *args: sock, timeout=TimeoutError))
    monkeypatch.setattr(record, "time", SimpleNamespace(perf_counter_ns=lambda: next(stamps)))
    reader = state["reader"] = record.TelemetryReader(9999)
    reader.run()
    assert reader.latest_ns == (3_000_000_000, 1, 15., -7, 11., 100., .125, 12000, 3, 234, 5, 800)
    assert reader.latest == (3., *reader.latest_ns[1:])
    assert reader.last_backjump == 3.0
    assert snapshots[1][0][0] == 1_000_000_000
    assert snapshots[1] == snapshots[2]  # Invalid data never replaces a sample or triggers rewind.
    assert reader.error is None and sock.closed


def test_game_steering_takeover_discards_runup_and_resumes_new_segment(record, monkeypatch, tmp_path):
    # Human wheel is 10 degrees throughout. Learn game-steer / wheel-angle = 1,
    # then expose six fresh frames controlled by the game, followed by recovery.
    install_hardware_mocks(
        record, monkeypatch, tmp_path, [image(80) for _ in range(200)],
        telemetry_steer=lambda call: 120 if 111 <= call <= 116 else 10)
    record.main(["record", "--fps", "30", "--drop-seconds", "1"])
    session = next((tmp_path / "recordings").iterdir())
    metadata = json.loads((session / "meta.json").read_text())
    labels, timing = read_csv(session / "labels.csv"), read_csv(session / "capture_timing.csv")
    assert metadata["takeovers"] == 1 and metadata["rewinds"] == 0
    assert metadata["segments"] == 2
    assert metadata["discarded_by_rewind_or_takeover"] == 15
    assert metadata["discarded_at_stop"] > 0
    assert metadata["dropped"] == 0
    assert_frame_accounting(metadata)
    original = [t for t in timing if t["segment"] == "0"]
    resumed = [t for t in timing if t["segment"] == "1"]
    assert original and resumed
    # The buffered run-up and all model-independent game-control frames are gone.
    assert max(int(t["capture_index"]) for t in original) < 110
    assert int(resumed[0]["capture_index"]) == 145  # 30 matching frames after detection.
    assert int(resumed[0]["frame"]) == 115  # Original allocated IDs survive discarded rows.
    assert int(resumed[0]["frame"]) > int(original[-1]["frame"]) + 1
    assert all(row["tele_steer"] == "10" for row in labels)
    assert [row["frame"] for row in labels] == [row["frame"] for row in timing]
