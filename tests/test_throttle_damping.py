"""Damped AI throttle with auto pedals: cap, ramp-up limit, instant drop; human-pedals mode."""

from forza_ai.contracts import DrivingPrediction
from forza_ai.runtime import run
from test_driving_control import Adapter


class FullThrottle:
    driving = True
    def predict(self, observation):
        return DrivingPrediction(0, 1.0, 0)


def throttles(adapter):
    return [s.throttle for _, s in adapter.writes]


def test_default_passes_model_throttle():
    a = Adapter(); run(a, FullThrottle(), duration=.4, assist=True, auto_pedals=True, direct_vjoy=True)
    assert max(throttles(a)) == 1.0


def test_cap_limits_ai_throttle():
    a = Adapter(); run(a, FullThrottle(), duration=.4, assist=True, auto_pedals=True, direct_vjoy=True,
                       throttle_cap=0.6)
    assert max(throttles(a)) == 0.6


def test_ramp_rises_gradually_from_zero():
    a = Adapter(); run(a, FullThrottle(), duration=.6, assist=True, auto_pedals=True, direct_vjoy=True,
                       throttle_rate=0.5)
    t = throttles(a)
    assert max(t) <= 0.5 * 0.6 + 0.05           # at most ~0.3 after 0.6 s at 0.5/s
    assert all(b >= a_ - 1e-9 for a_, b in zip(t, t[1:]) if b > 0)   # never jumps down while ramping


def test_speed_cap_cuts_throttle():
    from test_live_runtime import FakeTelemetry
    a = Adapter()
    run(a, FullThrottle(), duration=.3, assist=True, auto_pedals=True, direct_vjoy=True,
        receiver=FakeTelemetry(), max_speed_kmh=10)     # FakeTelemetry reports 10 m/s = 36 km/h
    assert max(throttles(a)) == 0.0
