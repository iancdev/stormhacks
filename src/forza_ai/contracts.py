"""Hardware-independent runtime types. Angles are physical degrees, right positive."""

from dataclasses import dataclass
from enum import Enum
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
