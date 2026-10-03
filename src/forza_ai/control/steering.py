"""Rate-limited PD steering with explicit, latched disengagement."""

from dataclasses import dataclass
import math

from forza_ai.contracts import ControlMode, ControlStatus, SteeringCommand, WheelState


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class SteeringConfig:
    # Conservative test defaults, NOT hardware-tuned gains.
    kp: float = 0.008
    kd: float = 0.001
    torque_limit: float = 0.15
    target_limit_deg: float = 90.0
    target_rate_deg_s: float = 60.0
    physical_limit_deg: float = 450.0
    velocity_filter_s: float = 0.04
    max_command_age_ns: int = 250_000_000
    max_observation_age_ns: int = 250_000_000
    max_wheel_age_ns: int = 50_000_000
    max_tick_gap_ns: int = 100_000_000

    def __post_init__(self):
        for name in ("kp", "kd", "torque_limit", "target_limit_deg",
                     "target_rate_deg_s", "physical_limit_deg", "velocity_filter_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.torque_limit <= 1:
            raise ValueError("torque_limit must be in (0, 1]")
        if not 0 < self.target_limit_deg <= self.physical_limit_deg:
            raise ValueError("target_limit_deg must be within physical travel")
        if self.target_rate_deg_s <= 0:
            raise ValueError("target_rate_deg_s must be positive")
        for name in ("max_command_age_ns", "max_observation_age_ns", "max_wheel_age_ns", "max_tick_gap_ns"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


class SteeringController:
    def __init__(self, config: SteeringConfig | None = None):
        self.config = config or SteeringConfig()
        self.mode = ControlMode.MANUAL
        self.reason = "not_engaged"
        self.target = 0.0
        self._last_tick_ns = None
        self._last_wheel_ns = None
        self._last_angle = None
        self._velocity = 0.0
        self._last_command_ns = None

    def wheel_error(self, wheel: WheelState, now_ns: int) -> str | None:
        if not wheel.connected:
            return "wheel_disconnected"
        if not all(math.isfinite(x) for x in (wheel.angle_deg, wheel.throttle, wheel.brake)):
            return "invalid_wheel_value"
        if abs(wheel.angle_deg) > self.config.physical_limit_deg + 0.1:
            return "wheel_outside_calibration"
        if not (0 <= wheel.throttle <= 1 and 0 <= wheel.brake <= 1):
            return "invalid_pedal_value"
        if not 0 <= now_ns - wheel.timestamp_ns <= self.config.max_wheel_age_ns:
            return "stale_wheel"
        if self._last_wheel_ns is not None and wheel.timestamp_ns < self._last_wheel_ns:
            return "wheel_time_reversed"
        return None

    def command_error(self, command: SteeringCommand | None, now_ns: int) -> str | None:
        if command is None:
            return "no_command"
        if not math.isfinite(command.target_angle_deg):
            return "invalid_target"
        if not 0 <= now_ns - command.generated_time_ns <= self.config.max_command_age_ns:
            return "stale_command"
        if command.observation_time_ns > command.generated_time_ns:
            return "future_observation"
        if not 0 <= now_ns - command.observation_time_ns <= self.config.max_observation_age_ns:
            return "stale_observation"
        if command.valid_until_ns <= now_ns:
            return "command_expired"
        if self._last_command_ns is not None and command.generated_time_ns < self._last_command_ns:
            return "command_time_reversed"
        return None

    def disengage(self, reason: str = "manual_takeover", fault: bool = False):
        self.mode = ControlMode.FAULT if fault else ControlMode.TAKEOVER
        self.reason = reason

    def step(self, wheel: WheelState, command: SteeringCommand | None, now_ns: int,
             *, engage: bool = False, takeover: bool = False) -> ControlStatus:
        """One motor update. Commands alone never engage/re-engage assistance."""
        wheel_error = self.wheel_error(wheel, now_ns)
        gap = None if self._last_tick_ns is None else now_ns - self._last_tick_ns
        self._last_tick_ns = now_ns
        if wheel_error:
            self.disengage(wheel_error, fault=True)
            return self._status(wheel, 0.0)
        if gap is not None and (gap <= 0 or gap > self.config.max_tick_gap_ns):
            if self.mode == ControlMode.ASSIST:
                self.disengage("control_loop_gap", fault=True)
                self._velocity = 0.0
                self._last_angle = wheel.angle_deg
                self._last_wheel_ns = wheel.timestamp_ns
                return self._status(wheel, 0.0)
            self._velocity = 0.0

        # Derivative uses sample time, not repeated reads of the same sample.
        if self._last_wheel_ns is not None and wheel.timestamp_ns > self._last_wheel_ns:
            sample_dt = (wheel.timestamp_ns - self._last_wheel_ns) / 1e9
            measured_velocity = (wheel.angle_deg - self._last_angle) / sample_dt
            alpha = sample_dt / (self.config.velocity_filter_s + sample_dt)
            self._velocity += alpha * (measured_velocity - self._velocity)
        self._last_wheel_ns = wheel.timestamp_ns
        self._last_angle = wheel.angle_deg

        if takeover:
            self.disengage()
            return self._status(wheel, 0.0)

        command_error = self.command_error(command, now_ns)
        if engage and self.mode != ControlMode.ASSIST:
            if command_error:
                self.disengage(command_error)
            else:
                self.mode = ControlMode.ASSIST
                self.reason = "engaged"
                self.target = wheel.angle_deg  # no target jump on engagement
                self._velocity = 0.0
                self._last_command_ns = command.generated_time_ns
                return self._status(wheel, 0.0)
        if self.mode != ControlMode.ASSIST:
            return self._status(wheel, 0.0)
        if command_error:
            self.disengage(command_error)
            return self._status(wheel, 0.0)

        self._last_command_ns = command.generated_time_ns
        dt = 0.0 if gap is None else gap / 1e9
        requested = clamp(command.target_angle_deg, self.config.target_limit_deg)
        self.target += clamp(requested - self.target, self.config.target_rate_deg_s * dt)
        torque = self.config.kp * (self.target - wheel.angle_deg) - self.config.kd * self._velocity
        return self._status(wheel, clamp(torque, self.config.torque_limit))

    def _status(self, wheel: WheelState, torque: float) -> ControlStatus:
        return ControlStatus(self.mode, self.reason, self.target, wheel.angle_deg, torque)
