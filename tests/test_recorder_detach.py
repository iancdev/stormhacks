"""Disconnected TMX handles must never produce fresh recording labels."""

import csv
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def record():
    spec = importlib.util.spec_from_file_location(
        "recorder_detach_test", Path(__file__).resolve().parents[1] / "record.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("when", ["before_poll", "during_poll"])
def test_detach_invalidates_warmed_reader_before_cleanup(record, monkeypatch, when):
    reader = record.WheelReader(vjoy=True)
    state = {"attached": True, "polls": 0, "clocks": 0, "waits": 0,
             "closed": False, "quit": False}
    outputs = []
    invalidated_at_cleanup = []

    def update():
        state["polls"] += 1
        assert state["polls"] <= 3, "detached reader kept polling"

    def axis(handle, index):
        if when == "during_poll" and state["polls"] == 3 and index == 1:
            state["attached"] = False
        # Real disconnected handles may still expose their last axis values.
        return (0, 0, 0)[index] if state["polls"] == 1 else (730, 32767, -32768)[index]

    def clock():
        state["clocks"] += 1
        return state["clocks"] * 1_000_000

    def wait(seconds):
        state["waits"] += 1
        if state["waits"] == 2:
            assert reader.live
            assert reader.latest_ns == (2_000_000, 730, 32767, -32768)
            if when == "before_poll":
                state["attached"] = False

    def set_axis(axis_name, value):
        outputs.append((axis_name, value))
        if not state["attached"]:
            invalidated_at_cleanup.append((reader.live, reader.latest, reader.latest_ns, reader.error))

    reader.stop = SimpleNamespace(is_set=lambda: False, wait=wait)
    sdl = SimpleNamespace(
        SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS=b"background", SDL_INIT_JOYSTICK=1,
        SDL_SetHint=lambda *args: None, SDL_Init=lambda *args: 0,
        SDL_NumJoysticks=lambda: 1, SDL_JoystickNameForIndex=lambda index: b"Thrustmaster TMX",
        SDL_JoystickOpen=lambda index: object(),
        SDL_JoystickGetAttached=lambda handle: state["attached"],
        SDL_JoystickUpdate=update, SDL_JoystickGetAxis=axis,
        SDL_JoystickClose=lambda handle: state.update(closed=True),
        SDL_Quit=lambda: state.update(quit=True),
    )
    monkeypatch.setitem(sys.modules, "sdl2", sdl)
    monkeypatch.setitem(sys.modules, "pyvjoy", SimpleNamespace(
        VJoyDevice=lambda device_id: SimpleNamespace(set_axis=set_axis),
        HID_USAGE_X="X", HID_USAGE_Y="Y", HID_USAGE_Z="Z"))
    monkeypatch.setattr(record, "time", SimpleNamespace(perf_counter_ns=clock))

    reader.run()

    assert "TMX disconnected" in reader.error
    assert not reader.live and reader.latest is None and reader.latest_ns is None
    assert reader.ready.is_set()
    assert state["polls"] == (2 if when == "before_poll" else 3)
    assert state["clocks"] == 2  # No fresh timestamp for the detached poll.
    assert state["closed"] and state["quit"]
    rest = list(record.VJOY_REST.items())
    assert outputs[:3] == rest and outputs[-3:] == rest
    assert outputs[3:6] == [("X", 1 + (730 + 32768) * 0x7FFF // 65535),
                            ("Y", 32768), ("Z", 1)]
    assert len(outputs) == 9  # Startup, one live poll, then cleanup only.
    assert invalidated_at_cleanup == [(False, None, None, "TMX disconnected")] * 3


def test_detach_during_capture_never_saves_new_label_or_completion(record, monkeypatch, tmp_path):
    clock = SimpleNamespace(now=1_000_000_000)
    readers = []

    class Wheel(record.WheelReader):
        def __init__(self, vjoy):
            super().__init__(vjoy)
            self.live = True
            self.latest_ns = (clock.now - 2_000_000, 730, 32767, -32768)
            readers.append(self)

        def start(self):
            self.ready.set()

        def join(self, timeout):
            assert self.stop.is_set()

        def is_alive(self):
            return False

    class Telemetry:
        error = None

        def __init__(self, port):
            self.stop = threading.Event()

        @property
        def latest_ns(self):
            return (clock.now - 3_000_000, 1, 15.0, 0)

        def start(self):
            pass

        def join(self, timeout):
            assert self.stop.is_set()

        def is_alive(self):
            return False

    class Camera:
        is_capturing = False
        calls = 0
        released = False

        def grab(self, **kwargs):
            self.calls += 1
            assert self.calls <= 2, "capture continued after wheel detach"
            clock.now += 1_000_000
            if self.calls == 2:
                # Reproduce detach after cmd_record's top-of-loop error check,
                # before it reads latest_ns. Unpacking None used to hide the cause.
                readers[0]._fail("TMX disconnected")
            return np.full((4, 8, 3), 40 * self.calls, dtype=np.uint8)

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(record, "WheelReader", Wheel)
    monkeypatch.setattr(record, "TelemetryReader", Telemetry)
    monkeypatch.setattr(record, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(record, "load_config", lambda: {
        "monitor": 0, "crop": [0, 0, 8, 4], "masks": [], "save_width": 8})
    monkeypatch.setattr(record, "time", SimpleNamespace(
        perf_counter_ns=lambda: clock.now,
        sleep=lambda seconds: setattr(clock, "now", clock.now + round(seconds * 1e9))))
    monkeypatch.setitem(sys.modules, "dxcam", SimpleNamespace(create=lambda **kwargs: camera))

    with pytest.raises(RuntimeError, match="Recording incomplete.*TMX disconnected") as error:
        record.main(["record"])

    assert "TypeError" not in str(error.value)
    assert camera.released and camera.calls == 2
    assert readers[0].latest_ns is None and not readers[0].live
    assert not list(tmp_path.glob("recordings/*/meta.json"))
    with next(tmp_path.glob("recordings/*/labels.csv")).open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    assert len(labels) == 1 and labels[0]["frame"] == "0"
    assert len(list(tmp_path.glob("recordings/*/frames/*.jpg"))) == 1
