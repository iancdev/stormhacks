"""Deterministic stationary test policy. This does not follow a road."""

from dataclasses import dataclass
import math
from forza_ai.contracts import VehicleState, WheelState


@dataclass(frozen=True)
class TestObservation:
    wheel: WheelState
    vehicle: VehicleState | None = None

    @property
    def timestamp_ns(self):
        return self.wheel.timestamp_ns


class FixedAnglePolicy:
    name = "fixed-angle test policy (NOT a driving model)"

    def __init__(self, target_angle_deg: float = 5.0):
        if not math.isfinite(target_angle_deg):
            raise ValueError("target angle must be finite")
        self.target_angle_deg = target_angle_deg

    def predict(self, observation: TestObservation) -> float:
        return self.target_angle_deg


class SweepPolicy:
    """Finite 0/right/0/left/0 sequence, then hold centre until stopped."""

    name = "stationary sweep test (NOT a driving model)"

    def __init__(self, amplitude_deg: float = 5.0, hold_seconds: float = 2.0):
        if not math.isfinite(amplitude_deg) or not 0 < amplitude_deg <= 15:
            raise ValueError("stationary sweep amplitude must be in (0, 15] degrees")
        if not math.isfinite(hold_seconds) or hold_seconds < 0.1:
            raise ValueError("hold_seconds must be finite and at least 0.1")
        self.targets = (0.0, amplitude_deg, 0.0, -amplitude_deg, 0.0)
        self.hold_ns = int(hold_seconds * 1e9)
        self._started_ns = None

    def predict(self, observation: TestObservation) -> float:
        if self._started_ns is None:
            self._started_ns = observation.timestamp_ns
        elapsed = max(0, observation.timestamp_ns - self._started_ns)
        index = min(len(self.targets) - 1, elapsed // self.hold_ns)
        return self.targets[index]
