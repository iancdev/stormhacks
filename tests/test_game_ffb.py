"""Game force feedback forwarding: packet tracking and replay on the wheel while the human drives."""

from forza_ai.contracts import ControlMode
from forza_ai.hardware.game_ffb import (CTRL_STOPALL, EFF_START, ET_CONST, PT_CONSTREP, PT_CTRLREP,
                                        PT_EFFREP, PT_EFOPREP, PT_GAINREP, GameForceTracker)
from forza_ai.policies.placeholder import FixedAnglePolicy
from forza_ai.runtime import run
from test_direct_vjoy import RecordingAdapter

MS = 1_000_000


def forza_update(tracker, magnitude, now, periodic=True):
    """One FH4-style update: constant force + effect report + start, plus an ignorable periodic one."""
    tracker.handle({"EffectBlockIndex": 1, "Magnitude": magnitude}, PT_CONSTREP, now)
    tracker.handle({"EffectBlockIndex": 1, "EffectType": ET_CONST, "Duration": 50, "Gain": 255}, PT_EFFREP, now)
    tracker.handle({"EffectBlockIndex": 1, "EffectOp": EFF_START, "LoopCount": 1}, PT_EFOPREP, now)
    if periodic:
        tracker.handle({"EffectBlockIndex": 2, "EffectType": 4, "Duration": 10, "Gain": 0}, PT_EFFREP, now)
        tracker.handle({"EffectBlockIndex": 2, "EffectOp": EFF_START, "LoopCount": 1}, PT_EFOPREP, now)


def test_tracker_follows_constant_force_and_ignores_periodic():
    t = GameForceTracker(clock_ns=lambda: 0)
    forza_update(t, 4000, 1000 * MS)
    assert abs(t.force(1010 * MS) - 0.4) < 1e-9
    forza_update(t, -2500, 1016 * MS)
    assert abs(t.force(1020 * MS) + 0.25) < 1e-9


def test_tracker_zero_when_stale_expired_stopped_or_gain_zero():
    t = GameForceTracker(clock_ns=lambda: 0)
    forza_update(t, 4000, 1000 * MS)
    assert t.force(1060 * MS) == 0.0                     # 50 ms effect over
    forza_update(t, 4000, 2000 * MS)
    assert t.force(2200 * MS) == 0.0                     # no packets for 200 ms
    forza_update(t, 4000, 3000 * MS)
    t.handle(CTRL_STOPALL, PT_CTRLREP, 3005 * MS)
    assert t.force(3010 * MS) == 0.0
    forza_update(t, 4000, 4000 * MS)
    t.handle(128, PT_GAINREP, 4001 * MS)
    assert abs(t.force(4010 * MS) - 0.4 * 128 / 255) < 1e-9


class GameForceAdapter(RecordingAdapter):
    def __init__(self, force, **kw):
        super().__init__(**kw); self.force = force
    def game_force(self, now_ns, max_age_ns=150_000_000):
        return self.force


def test_runtime_replays_game_force_only_while_human_drives():
    from forza_ai.control import SteeringConfig
    manual = GameForceAdapter(0.4)
    run(manual, FixedAnglePolicy(5.0), duration=.3, assist=False, ffb_scale=0.5,
        config=SteeringConfig(torque_limit=0.3))
    assert any(abs(t - 0.2) < 1e-9 for t in manual.torques)          # 0.5 x 0.4 replayed while human drives
    engaged = GameForceAdapter(0.4)
    s = run(engaged, FixedAnglePolicy(5.0), duration=.3, assist=True, direct_vjoy=True, mirror_wheel=True,
            ffb_scale=0.5)
    assert s["mode"] == ControlMode.ASSIST.value and s["game_ffb_torque"] == 0     # AI driving: mirror only


def test_runtime_caps_and_flips_game_force():
    a = GameForceAdapter(1.0)
    run(a, FixedAnglePolicy(5.0), duration=.2, assist=False, ffb_scale=1.0, ffb_sign=-1.0)
    assert min(a.torques) == -0.15                                    # default torque limit


def test_adapter_registers_and_removes_vjoy_ffb_callback():
    from unittest.mock import Mock
    from test_hardware import Rig
    rig = Rig(); rig.sdk.FfbRegisterGenCB = Mock(); rig.sdk.FfbRemoveCB = Mock()
    adapter = rig.adapter(forward_ffb=True)
    callback, device = rig.sdk.FfbRegisterGenCB.call_args.args
    forza_update(adapter.game_ffb, 3000, 0, periodic=False)
    adapter.game_ffb._clock = lambda: 5 * MS
    assert abs(adapter.game_force(5 * MS) - 0.3) < 1e-9 and device == 1
    adapter.close()
    rig.sdk.FfbRemoveCB.assert_called_once_with(1)
    plain = Rig(); plain.sdk.FfbRegisterGenCB = Mock()
    plain.adapter().close()
    plain.sdk.FfbRegisterGenCB.assert_not_called()                     # off by default


class JitteryGameForce(RecordingAdapter):
    """Game force alternating +0.3 / -0.1 every read (mean +0.1): pure jitter around a small pull."""
    def game_force(self, now_ns, max_age_ns=150_000_000):
        return 0.3 if self.reads % 2 else -0.1


def test_smoothing_removes_jitter_and_keeps_the_average():
    from forza_ai.control import SteeringConfig
    raw = JitteryGameForce(); run(raw, FixedAnglePolicy(0), duration=.6, assist=False, ffb_scale=1.0,
                                  config=SteeringConfig(torque_limit=0.3))
    smooth = JitteryGameForce(); run(smooth, FixedAnglePolicy(0), duration=.6, assist=False, ffb_scale=1.0,
                                     ffb_smooth_ms=50, config=SteeringConfig(torque_limit=0.3))
    tail_raw, tail_smooth = raw.torques[-21:-1], smooth.torques[-21:-1]      # last entry = zero at shutdown
    assert max(tail_raw) - min(tail_raw) > 0.3                     # raw swings +0.3 / -0.1
    assert max(tail_smooth) - min(tail_smooth) < 0.06               # smoothed: small ripple
    assert abs(sum(tail_smooth) / len(tail_smooth) - 0.1) < 0.03    # around the true average pull
