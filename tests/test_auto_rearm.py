"""Auto re-engagement after a transient inference blip, but never after a human takeover."""

from forza_ai.contracts import ControlMode, ObservationUnavailable
from forza_ai.runtime import run
from test_direct_vjoy import RecordingAdapter


class Blippy:
    """Fixed 5 deg policy whose 20th call times out once."""
    def __init__(self): self.calls = 0
    def predict(self, observation):
        self.calls += 1
        if self.calls == 20:
            raise ObservationUnavailable("remote inference timed out or disconnected")
        return 5.0


def test_blip_latches_off_by_default():
    summary = run(RecordingAdapter(), Blippy(), duration=1.5, assist=True, direct_vjoy=True)
    assert summary["mode"] == ControlMode.TAKEOVER.value and summary["reason"] == "inference_failure"


def test_blip_auto_rearms_when_enabled():
    summary = run(RecordingAdapter(), Blippy(), duration=1.5, assist=True, direct_vjoy=True, auto_rearm_s=3)
    assert summary["mode"] == ControlMode.ASSIST.value
    assert summary.get("auto_rearms") == 1


def test_no_auto_rearm_after_human_takeover():
    adapter = RecordingAdapter(forced_angle_after=(60, 30.0))     # human turns the wheel past 20 deg
    summary = run(adapter, Blippy(), duration=1.5, assist=True, direct_vjoy=True, auto_rearm_s=3)
    assert summary["mode"] == ControlMode.TAKEOVER.value       # still off: no automatic re-engagement
    assert "auto_rearms" not in summary
    assert adapter.writes[-1] == 30.0                           # the human keeps steering
