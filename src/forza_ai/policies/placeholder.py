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
