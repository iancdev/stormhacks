"""Native deadline regressions: real adapter, fake SDL, no physical devices."""

from unittest.mock import patch

import pytest

from forza_ai.contracts import ActuationExpired
from forza_ai.hardware import HardwareError
from test_hardware import Rig


class Clock:
    def __init__(self):
        self.now_ns = 1_000_000_000

    def monotonic_ns(self):
        return self.now_ns

    def advance_ms(self, amount):
        self.now_ns += amount * 1_000_000


@pytest.fixture
def native():
    rig, clock = Rig(), Clock()
    adapter = rig.adapter()
    with patch("forza_ai.hardware.windows.time.monotonic_ns", clock.monotonic_ns):
        try:
            yield rig, clock, adapter
        finally:
            adapter.close()


def delay(rig, clock, operation, milliseconds):
    native_call = getattr(rig.sdl, operation)
    original = native_call.side_effect

    def delayed(*args):
        clock.advance_ms(milliseconds)
        return original(*args) if original is not None else native_call.return_value

    native_call.side_effect = delayed


def test_finite_effect_fits_remaining_sample_validity(native):
    rig, clock, adapter = native
    deadline = clock.now_ns + 50_000_000
    delay(rig, clock, "SDL_HapticUpdateEffect", 2)
    delay(rig, clock, "SDL_HapticRunEffect", 1)

    adapter.set_torque_before(.1, deadline)

    assert rig.active
    assert 1 <= rig.effect_length < 50
    assert rig.effect_length <= adapter.effect_ttl_ms
    assert clock.now_ns + rig.effect_length * 1_000_000 <= deadline
    assert rig.events == ["stop", "update", "run"]


def test_long_deadline_cannot_extend_configured_effect_ttl(native):
    rig, clock, adapter = native
    adapter.set_torque_before(.1, clock.now_ns + 1_000_000_000)
    assert rig.effect_length == adapter.effect_ttl_ms == 100
    assert rig.sdl.SDL_HapticRunEffect.call_args.args[-1] == 1


def test_expired_command_stops_existing_force_without_starting_another(native):
    rig, clock, adapter = native
    adapter.set_torque(.1)
    previous_starts = rig.sdl.SDL_HapticRunEffect.call_count

    with pytest.raises(ActuationExpired, match="before native output"):
        adapter.set_torque_before(.1, clock.now_ns)

    assert not rig.active
    assert rig.sdl.SDL_HapticRunEffect.call_count == previous_starts
    rig.sdl.SDL_HapticStopAll.assert_called_once()


@pytest.mark.parametrize("operation", ["SDL_JoystickGetAttached", "SDL_HapticStopEffect"])
def test_expiry_during_attachment_or_stop_never_updates_or_starts(native, operation):
    rig, clock, adapter = native
    delay(rig, clock, operation, 60)
    with pytest.raises(ActuationExpired, match="after native stop"):
        adapter.set_torque_before(.1, clock.now_ns + 50_000_000)
    rig.sdl.SDL_HapticUpdateEffect.assert_not_called()
    rig.sdl.SDL_HapticRunEffect.assert_not_called()
    rig.sdl.SDL_HapticStopAll.assert_called_once()
    assert not rig.active


@pytest.mark.parametrize("delay_ms", [6, 300])
def test_slow_update_cannot_start_expired_or_overlong_force(native, delay_ms):
    # 300 ms reproduces the review defect. Six ms spends the reserved headroom
    # while the command is still fresh: its preconfigured effect no longer fits.
    rig, clock, adapter = native
    delay(rig, clock, "SDL_HapticUpdateEffect", delay_ms)
    with pytest.raises(ActuationExpired, match="native update"):
        adapter.set_torque_before(.1, clock.now_ns + 50_000_000)
    rig.sdl.SDL_HapticRunEffect.assert_not_called()
    rig.sdl.SDL_HapticStopAll.assert_called_once()
    assert not rig.active


@pytest.mark.parametrize("delay_ms", [6, 300])
def test_late_native_start_is_stopped_when_it_returns(native, delay_ms):
    # No Python guard can interrupt synchronous native RunEffect. This asserts
    # only the enforceable post-return stop, not a hard real-time guarantee.
    rig, clock, adapter = native
    delay(rig, clock, "SDL_HapticRunEffect", delay_ms)
    with pytest.raises(ActuationExpired, match="native start"):
        adapter.set_torque_before(.1, clock.now_ns + 50_000_000)
    rig.sdl.SDL_HapticRunEffect.assert_called_once()
    rig.sdl.SDL_HapticStopAll.assert_called_once()
    assert not rig.active


@pytest.mark.parametrize("remaining_ns", [1, 999_999, 1_000_000, 1_999_999])
def test_insufficient_whole_millisecond_budget_does_not_round_up(native, remaining_ns):
    rig, clock, adapter = native
    with pytest.raises(ActuationExpired):
        adapter.set_torque_before(.1, clock.now_ns + remaining_ns)
    rig.sdl.SDL_HapticUpdateEffect.assert_not_called()
    rig.sdl.SDL_HapticRunEffect.assert_not_called()


def test_zero_always_releases_even_with_expired_deadline(native):
    rig, clock, adapter = native
    adapter.set_torque(.1)
    adapter.set_torque_before(0, clock.now_ns - 1)
    assert not rig.active
    assert rig.sdl.SDL_HapticRunEffect.call_count == 1


@pytest.mark.parametrize("deadline", [None, True, -1, 1.5])
def test_invalid_deadline_stops_existing_effect(native, deadline):
    rig, clock, adapter = native
    adapter.set_torque(.1)
    with pytest.raises(ValueError, match="deadline_ns"):
        adapter.set_torque_before(.1, deadline)
    assert not rig.active
    assert rig.sdl.SDL_HapticRunEffect.call_count == 1


def test_native_update_failure_still_stops_and_preserves_driver_error(native):
    rig, clock, adapter = native
    rig.sdl.SDL_HapticUpdateEffect.side_effect = None
    rig.sdl.SDL_HapticUpdateEffect.return_value = -1
    with pytest.raises(HardwareError, match="SDL_HapticUpdateEffect"):
        adapter.set_torque_before(.1, clock.now_ns + 50_000_000)
    rig.sdl.SDL_HapticRunEffect.assert_not_called()
    rig.sdl.SDL_HapticStopAll.assert_called_once()


def test_legacy_calls_restore_the_configured_finite_duration(native):
    rig, clock, adapter = native
    adapter.set_torque_before(.1, clock.now_ns + 50_000_000)
    assert rig.effect_length < adapter.effect_ttl_ms
    adapter.set_torque(.1)
    assert rig.effect_length == adapter.effect_ttl_ms
