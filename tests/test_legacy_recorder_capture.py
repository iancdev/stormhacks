"""Fresh capture, timing compatibility, and failure closure without native devices."""

import importlib.util
import csv
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


def install_hardware_mocks(record, monkeypatch, tmp_path, frames, *, active=True):
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
        def latest_ns(self):
            return (clock.now - 2_000_000, 730, 32767, 6553)

    class Telemetry(Reader):
        @property
        def latest_ns(self):
            race_on = active(clock.now) if callable(active) else active
            return (clock.now - 3_000_000, int(race_on), 15.0, 0)

    monkeypatch.setitem(sys.modules, "dxcam", SimpleNamespace(create=lambda **kwargs: camera))
    monkeypatch.setattr(record, "WheelReader", Wheel)
    monkeypatch.setattr(record, "TelemetryReader", Telemetry)
    monkeypatch.setattr(record, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(record, "time", SimpleNamespace(perf_counter_ns=clock.ns, sleep=clock.sleep))
    monkeypatch.setattr(record, "load_config", lambda: {
        "monitor": 0, "crop": [0, 0, 8, 4], "masks": [[0, 0, 2, 2]], "save_width": 8})
    return clock, camera, instances


def test_default_sixty_and_fps_validation(record):
    assert record.argument_parser().parse_args(["record"]).fps == 60
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
        record.main(["record"])
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


def test_recording_keeps_legacy_columns_and_imports_without_cached_frames(record, monkeypatch, tmp_path):
    clock, camera, readers = install_hardware_mocks(
        record, monkeypatch, tmp_path, [image(40), None, image(80), None, image(120)])
    record.main(["record"])
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
    assert metadata["fps_target"] == 60
    assert metadata["completed"] is True
    assert metadata["capture_provenance"]["cached_frames_reused"] is False
    assert metadata["measured_capture"]["no_new_frame_polls"] == 2
    assert metadata["measured_capture"]["fresh_frames"] == 3
    assert metadata["measured_capture"]["captured_fps"] < 60
    assert metadata["measured_capture"]["fresh_capture_gaps"]["median_ms"] == 33.3
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
                                 (1_007_000_000, 1, 15, -1))
    assert dict(zip(record.COLUMNS, row)) == {
        "frame": 0, "segment": 0, "t": "0.0100", "steer_raw": 730,
        "steer_deg": "10.00", "brake": "0.0000", "gas": "1.0000",
        "wheel_age_ms": "2.0", "speed_mps": "15.000", "race_on": 1,
        "tele_steer": -1, "tele_age_ms": "3.0"}


def test_pause_during_no_frame_poll_still_starts_a_new_segment(record, monkeypatch, tmp_path):
    install_hardware_mocks(record, monkeypatch, tmp_path, [image(40), None, image(80)],
                           active=lambda now: not 1_010_000_000 < now < 1_030_000_000)
    record.main(["record"])
    session = next((tmp_path / "recordings").iterdir())
    labels = read_csv(session / "labels.csv")
    assert [row["segment"] for row in labels] == ["0", "1"]
    assert json.loads((session / "meta.json").read_text())["segments"] == 2


def test_queue_drop_keeps_contiguous_saved_ids_and_does_not_open_empty_segment(record, monkeypatch, tmp_path):
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
    record.main(["record"])
    session = next((tmp_path / "recordings").iterdir())
    labels = read_csv(session / "labels.csv")
    metadata = json.loads((session / "meta.json").read_text())
    assert [row["frame"] for row in labels] == ["0", "1"]
    assert [row["segment"] for row in labels] == ["0", "0"]
    assert metadata["dropped"] == metadata["measured_capture"]["dropped_queue_frames"] == 1
    assert metadata["frames"] == 2 and metadata["segments"] == 1
    from forza_ai.data.recording import import_recording
    assert import_recording(session, tmp_path / "imported", expert_mode="manual")["accepted"] == 2


def test_no_frames_produces_zero_achieved_rate_and_no_labels(record, monkeypatch, tmp_path):
    _, camera, _ = install_hardware_mocks(record, monkeypatch, tmp_path, [None] * 4)
    record.main(["record"])
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
        record.main(["record"])
    assert not list(tmp_path.glob("recordings/*/meta.json"))
    assert camera.released and all(reader.joined for reader in readers)


def test_capture_error_leaves_no_completion_metadata(record, monkeypatch, tmp_path):
    _, camera, readers = install_hardware_mocks(record, monkeypatch, tmp_path, [])
    def fail(**kwargs):
        raise RuntimeError("capture disconnected")
    camera.grab = fail
    with pytest.raises(RuntimeError, match="capture disconnected"):
        record.main(["record"])
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
