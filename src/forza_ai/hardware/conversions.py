"""Calibration conversions for the tested TMX and vJoy axis arrangement."""

import math


def finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def rotation(value: float) -> float:
    value = finite(value, "rotation_deg")
    if value <= 0:
        raise ValueError("rotation_deg must be positive")
    return value


def steering_degrees(raw: int, rotation_deg: float = 900) -> float:
    """SDL has asymmetric signed endpoints: -32768, 0, +32767."""
    extent = rotation(rotation_deg) / 2
    raw = max(-32768, min(32767, finite(raw, "raw steering")))
    return raw / (32768 if raw < 0 else 32767) * extent


def pedal_fraction(raw: int) -> float:
    """TMX brake a1 and throttle a2: +32767 released, -32768 pressed."""
    raw = max(-32768, min(32767, finite(raw, "raw pedal")))
    return (32767 - raw) / 65535


def vjoy_steering(angle_deg: float, rotation_deg: float = 900) -> int:
    """Map calibrated physical angle to vJoy X, preserving center 0x4000."""
    value = finite(angle_deg, "angle_deg") / (rotation(rotation_deg) / 2)
    value = max(-1, min(1, value))
    return round(16384 + value * (16383 if value < 0 else 16384))


def vjoy_pedal(fraction: float) -> int:
    """vJoy Y/Z retain the game's inverted binding: 32768 released, 1 pressed."""
    value = max(0, min(1, finite(fraction, "pedal")))
    return round(32768 - value * 32767)


def sdl_force_level(torque: float, torque_limit: float) -> int:
    limit = finite(torque_limit, "torque_limit")
    if not 0 <= limit <= 1:
        raise ValueError("torque_limit must be between 0 and 1")
    value = max(-limit, min(limit, finite(torque, "torque")))
    # Tested physical sign: a negative SDL constant force pushes the TMX right.
    # Truncate toward zero so quantization never exceeds the configured limit.
    return -int(value * 32767)
