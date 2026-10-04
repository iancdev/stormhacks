"""Hardware-independent runtime types. Angles are physical degrees, right positive."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class ControlMode(str, Enum):
    MANUAL = "manual"
    ASSIST = "assist"
    TAKEOVER = "takeover"
    FAULT = "fault"


@dataclass(frozen=True)
class WheelState:
    timestamp_ns: int
    angle_deg: float
    throttle: float
    brake: float
    buttons: tuple[int, ...] = ()  # SDL's zero-based button indices
    connected: bool = True


@dataclass(frozen=True)
class VehicleState:
    timestamp_ns: int  # host monotonic receive time, not game simulation time
    speed_mps: float
    is_race_on: bool
    game_timestamp_ms: int
    rpm: float
    steering_input: int  # Forza's signed input byte, not physical wheel degrees


@dataclass(frozen=True)
class SteeringCommand:
    target_angle_deg: float
    generated_time_ns: int
    observation_time_ns: int
    valid_until_ns: int
    inference_ms: float | None = None
    throttle: float | None = None
    brake: float | None = None


@dataclass(frozen=True)
class ControlStatus:
    mode: ControlMode
    reason: str
    target_angle_deg: float
    actual_angle_deg: float
    torque: float  # normalized [-1, 1], positive physically right


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: int
    timestamp_ns: int  # host capture start, conservatively includes grab latency
    rgb: Any  # uint8 HWC NumPy array; driver imports stay outside contracts


@dataclass(frozen=True)
class ModelObservation:
    frame: CapturedFrame
    vehicle: VehicleState  # sample at or before frame time, as during training

    @property
    def timestamp_ns(self) -> int:
        return self.frame.timestamp_ns


class ObservationUnavailable(RuntimeError):
    """Expected temporary missing/stale/paused input; inhibit assistance."""


class ActuationExpired(RuntimeError):
    """Native preparation outlived the command's absolute actuation deadline."""


@dataclass(frozen=True)
class DrivingPrediction:
    angle_deg: float
    throttle: float
    brake: float

    def __post_init__(self):
        values = (self.angle_deg, self.throttle, self.brake)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError('driving outputs must be finite numbers')
        if abs(self.angle_deg) > 450 or not 0 <= self.throttle <= 1 or not 0 <= self.brake <= 1:
            raise ValueError('driving outputs exceed angle/pedal range')
        if self.throttle > 0 and self.brake > 0:
            raise ValueError('simultaneous throttle and brake is not a driving command')
