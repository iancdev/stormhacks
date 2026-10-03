"""Hardware-independent runtime types. Angles are physical degrees, right positive."""

from dataclasses import dataclass
from enum import Enum


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


@dataclass(frozen=True)
class ControlStatus:
    mode: ControlMode
    reason: str
    target_angle_deg: float
    actual_angle_deg: float
    torque: float  # normalized [-1, 1], positive physically right
