import csv
import json
import threading
import time

import numpy as np
import pytest

from forza_ai.contracts import CapturedFrame, ControlMode, VehicleState, WheelState
from forza_ai.data.sessions import Alignment, load_session, load_sessions, split_sessions
from forza_ai.recording import SessionRecorder


def wheel(ms, angle=10):
    return WheelState(ms * 1_000_000, angle, .4, 0)


def vehicle(ms, speed=15):
    return VehicleState(ms * 1_000_000, speed, True, ms, 2000, 0)


def frame(index, ms):
    pixels = np.full((66, 200, 3), index, dtype=np.uint8)
    pixels.flags.writeable = False
    return CapturedFrame(index, ms * 1_000_000, pixels)


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metadata(path):
    return json.loads((path / "metadata.json").read_text())


def record_demonstration(path, name):
    recorder = SessionRecorder(path, session_id=name, metadata={"synthetic": True}).start()
    for index in range(8):
        ms = 100 + index * 20
        assert recorder.submit(wheel(ms, index * 2), vehicle(ms), frame(index, ms),
                               ControlMode.MANUAL, expert=True, reason="demonstration")
    recorder.close()
    return recorder


def test_mixed_modes_keep_only_explicit_expert_labels(tmp_path):
    path = tmp_path / "mixed"
    recorder = SessionRecorder(path).start()
    modes = [("manual", True), ("assist", True), ("takeover", True),
             ("takeover", False), ("fault", True), ("manual", False)]
    for index, (mode, expert) in enumerate(modes):
        ms = 100 + index * 20
        assert recorder.submit(wheel(ms), vehicle(ms), frame(index, ms), mode,
                               expert=expert, reason=f"case {index}")
    recorder.close()
    session = load_session(path)
    assert [sample.control_mode for sample in session.samples] == ["manual", "takeover"]
    assert session.rejected == {"nonexpert_or_mode_boundary": 4}
    assert [r["control_mode"] for r in rows(path / "wheel.csv")] == [
        "manual", "assist", "takeover", "assist", "assist", "assist"]
    events = rows(path / "events.csv")
    assert events[4]["control_mode"] == "fault"
    assert events[4]["training_mode"] == "assist"
    assert events[4]["reason"] == "case 4"
    assert metadata(path)["completed"] is True


def test_cached_pre_takeover_frame_is_not_a_human_label(tmp_path):
    path = tmp_path / "session"
    recorder = SessionRecorder(path).start()
    recorder.submit(wheel(100), vehicle(100), frame(0, 100), "assist")
    # Newly delivered to the recorder, but captured before human takeover.
    cached = frame(1, 110)
    recorder.submit(wheel(120), vehicle(110), cached, "takeover", expert=True)
    recorder.submit(wheel(130), vehicle(120), cached, "takeover", expert=True)
    recorder.submit(wheel(140), vehicle(130), frame(2, 135), "takeover", expert=True)
    recorder.close()
    written = rows(path / "frames.csv")
    assert [r["frame_id"] for r in written] == ["0", "2"]
    assert [s.capture_time_ns for s in load_session(path).samples] == [135_000_000]
    assert recorder.stats["frames_skipped"] == 2


def test_complete_archive_can_have_no_expert_samples(tmp_path):
    path = tmp_path / "assist-only"
    recorder = SessionRecorder(path).start()
    recorder.submit(wheel(100), vehicle(100), frame(0, 100), "assist")
    recorder.submit(wheel(120), vehicle(100), frame(1, 110), "takeover",
                    expert=False, reason="command timeout")
    recorder.close()
    result = load_session(path)
    assert metadata(path)["completed"] is True
    assert not result.samples
    assert result.rejected == {"nonexpert_or_mode_boundary": 2}


def test_frame_after_latest_wheel_keeps_time_and_rejects_unbracketed_edge(tmp_path):
    path = tmp_path / "trailing-frame"
    recorder = SessionRecorder(path).start()
    recorder.submit(wheel(100), vehicle(100), frame(0, 100), "manual", expert=True)
    recorder.submit(wheel(120), vehicle(120), frame(1, 125), "manual", expert=True)
    recorder.close()
    result = load_session(path)
    assert [s.capture_time_ns for s in result.samples] == [100_000_000]
    assert result.rejected == {"wheel_coverage": 1}
    assert rows(path / "frames.csv")[-1]["capture_time_ns"] == "125000000"


def test_streams_keep_original_times_and_interpolation_boundaries(tmp_path):
    path = tmp_path / "session"
    recorder = SessionRecorder(path).start()
    recorder.submit(wheel(100, 0), vehicle(100), None, "manual", expert=True)
    recorder.submit(wheel(120, 20), vehicle(100), frame(0, 110), "manual", expert=True)
    recorder.submit(wheel(140, 40), vehicle(140), frame(1, 130), "assist")
    recorder.submit(wheel(160, 60), vehicle(140), frame(2, 160), "takeover", expert=True)
    recorder.close()
    result = load_session(path)
    assert [s.angle_deg for s in result.samples] == [10, 60]
    assert result.rejected == {"nonexpert_or_mode_boundary": 1}
    assert [r["timestamp_ns"] for r in rows(path / "telemetry.csv")] == ["100000000", "140000000"]
    assert rows(path / "frames.csv")[0]["capture_time_ns"] == "110000000"
    offset = load_session(path, Alignment(label_offset_ns=20_000_000))
    assert offset.rejected["nonexpert_or_mode_boundary"] == 2


def test_two_recorded_sessions_load_split_and_train(tmp_path):
    root = tmp_path / "sessions"
    for name in ("first", "second"):
        record_demonstration(root / name, name)
    sessions = load_sessions(root)
    train_sessions, validation_sessions = split_sessions(sessions, .5, 7)
    assert len(train_sessions) == len(validation_sessions) == 1
    assert len(train_sessions[0].samples) == len(validation_sessions[0].samples) == 8
    torch = pytest.importorskip("torch")
    from forza_ai.training.engine import TrainConfig, train
    torch.set_num_threads(1)
    result = train(root, tmp_path / "run", epochs=1,
                   config=TrainConfig(batch_size=4), device="cpu")
    assert result["epoch"] == 1
    assert (tmp_path / "run" / "last.pt").is_file()


def test_queue_overload_fails_closed_without_blocking_control(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = SessionRecorder._write_item
    def slow(self, item, writers):
        entered.set()
        assert release.wait(2)
        return original(self, item, writers)
    monkeypatch.setattr(SessionRecorder, "_write_item", slow)
    path = tmp_path / "session"
    recorder = SessionRecorder(path, queue_size=1).start()
    assert recorder.submit(wheel(100), vehicle(100), frame(0, 100), "manual", expert=True)
    assert entered.wait(1)
    assert recorder.submit(wheel(120), vehicle(120), frame(1, 120), "manual", expert=True)
    before = time.monotonic()
    assert not recorder.submit(wheel(140), vehicle(140), frame(2, 140), "manual", expert=True)
    assert time.monotonic() - before < .1
    with pytest.raises(RuntimeError, match="queue full"):
        recorder.check()
    release.set()
    with pytest.raises(RuntimeError, match="queue full"):
        recorder.close()
    assert recorder.stats["dropped"] == 1
    assert metadata(path)["completed"] is False
    with pytest.raises(ValueError, match="completed"):
        load_session(path)


def test_image_write_error_exposed_and_never_completed(tmp_path, monkeypatch):
    from PIL import Image
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(Image.Image, "save", fail)
    path = tmp_path / "session"
    recorder = SessionRecorder(path).start()
    recorder.submit(wheel(100), vehicle(100), frame(0, 100), "manual", expert=True)
    with pytest.raises(RuntimeError, match="disk full"):
        recorder.close()
    assert recorder.stats["error"] == "disk full"
    assert metadata(path)["completed"] is False


def test_shutdown_deadline_never_publishes_completed_later(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = SessionRecorder._write_item
    def hung(self, item, writers):
        entered.set()
        release.wait(2)
        return original(self, item, writers)
    monkeypatch.setattr(SessionRecorder, "_write_item", hung)
    monkeypatch.setattr(SessionRecorder, "CLOSE_TIMEOUT_S", .01)
    path = tmp_path / "session"
    recorder = SessionRecorder(path).start()
    recorder.submit(wheel(100), vehicle(100), frame(0, 100), "manual", expert=True)
    assert entered.wait(1)
    try:
        with pytest.raises(RuntimeError, match="shutdown deadline"):
            recorder.close()
        assert metadata(path)["completed"] is False
    finally:
        release.set()
        recorder._thread.join(1)
    assert metadata(path)["completed"] is False


@pytest.mark.parametrize("stream", ["wheel", "telemetry", "frame"])
def test_temporal_reversal_is_rejected(tmp_path, stream):
    path = tmp_path / "session"
    recorder = SessionRecorder(path).start()
    recorder.submit(wheel(100), vehicle(100), frame(0, 100), "assist")
    recorder.submit(wheel(90 if stream == "wheel" else 120),
                    vehicle(90 if stream == "telemetry" else 120),
                    frame(1, 90 if stream == "frame" else 120), "assist")
    with pytest.raises(RuntimeError, match="reversed"):
        recorder.close()
    assert metadata(path)["completed"] is False


def test_existing_destination_and_reserved_metadata_are_not_overwritten(tmp_path):
    path = tmp_path / "existing"
    path.mkdir()
    (path / "keep.txt").write_text("user data")
    with pytest.raises(ValueError, match="must not exist"):
        SessionRecorder(path).start()
    assert (path / "keep.txt").read_text() == "user data"
    with pytest.raises(ValueError, match="completed"):
        SessionRecorder(tmp_path / "new", metadata={"completed": True})
    assert not (tmp_path / "new").exists()


def test_failure_closure_and_missing_streams_remain_incomplete(tmp_path):
    recorder = SessionRecorder(tmp_path / "aborted").start()
    recorder.close(completed=False)
    assert metadata(recorder.destination)["completed"] is False
    empty = SessionRecorder(tmp_path / "empty").start()
    with pytest.raises(RuntimeError, match="missing wheel"):
        empty.close()
    assert metadata(empty.destination)["completed"] is False


def test_metadata_replace_failure_keeps_original_incomplete(tmp_path, monkeypatch):
    from pathlib import Path
    path = tmp_path / "session"
    recorder = SessionRecorder(path).start()
    recorder.submit(wheel(100), vehicle(100), frame(0, 100), "manual", expert=True)
    def fail(*args, **kwargs):
        raise OSError("atomic replacement failed")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(RuntimeError, match="atomic replacement failed"):
        recorder.close()
    assert metadata(path)["completed"] is False
    assert recorder.stats["completed"] is False
