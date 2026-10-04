"""Windows TMX/vJoy adapter, owned and called by one local control thread.

Only this adapter writes our commanded motor force. No game FFB callback is registered.
Short effects bound a host stall only if the wheel driver honors their duration;
verify expiry and the calibrated force direction on the real Windows device.
Opening SDL may reset feedback or engage native autocenter before our settings
are applied. Stopping our effects does not certify the absence of driver forces.
"""

import importlib
import sys
import time
from collections.abc import Mapping
from ctypes import c_uint

from ..contracts import ActuationExpired, WheelState
from .conversions import (
    finite,
    pedal_fraction,
    rotation,
    sdl_force_level,
    steering_degrees,
    vjoy_pedal,
    vjoy_steering,
)


_NO_DEADLINE = object()


class HardwareError(RuntimeError):
    """A hardware operation failed; the caller should leave assistance disabled."""


class WindowsAdapter:
    """Read TMX, forward measured inputs, and renew bounded motor commands.

    ``button_map`` maps SDL zero-based buttons to explicitly configured vJoy
    one-based buttons. An empty map forwards no buttons. Construction never runs
    our command effect, but SDL device opening can engage native centering.
    ``autocenter_disabled_confirmed`` reports whether startup explicitly disabled
    it; a false value means unconfirmed, not necessarily active. Zero torque
    stops our command effect and does not establish mechanical isolation.
    Call ``set_torque`` more frequently than ``effect_ttl_ms`` while engaged.
    """

    def __init__(
        self,
        rotation_deg: float = 900,
        torque_limit: float = 0.15,
        effect_ttl_ms: int = 100,
        button_map: Mapping[int, int] | None = None,
        vjoy_device_id: int = 1,
        use_motor: bool = True,
        use_vjoy: bool = True,
    ):
        # use_vjoy=False (observe-only shadow tests while Forza reads the TMX
        # directly): never acquire or write vJoy, so Forza doesn't get a second,
        # duplicated controller that it keeps switching to.
        self.use_vjoy = bool(use_vjoy)
        # use_motor=False (direct-vJoy fallback): never open SDL haptics, so the
        # fallback works even when the motor/FFB path is what failed. Only zero
        # torque is then accepted.
        self.use_motor = bool(use_motor)
        self.rotation_deg = rotation(rotation_deg)
        self.torque_limit = finite(torque_limit, "torque_limit")
        sdl_force_level(0, self.torque_limit)
        if type(effect_ttl_ms) is not int or not 1 <= effect_ttl_ms <= 250:
            raise ValueError("effect_ttl_ms must be an integer from 1 to 250")
        if type(vjoy_device_id) is not int or not 1 <= vjoy_device_id <= 16:
            raise ValueError("vjoy_device_id must be an integer from 1 to 16")
        self.effect_ttl_ms = effect_ttl_ms
        self.button_map = dict(button_map or {})
        for source, target in self.button_map.items():
            if type(source) is not int or source < 0:
                raise ValueError("SDL button indices must be nonnegative integers")
            if type(target) is not int or not 1 <= target <= 128:
                raise ValueError("vJoy button indices must be integers from 1 to 128")
        if len(set(self.button_map.values())) != len(self.button_map):
            raise ValueError("Each vJoy button must have one physical source")
        if sys.platform != "win32":
            raise OSError("WindowsAdapter requires Windows and installed TMX/vJoy drivers")

        self._sdl = None
        self._vjoy = None
        self._sdk = None
        self._joystick = None
        self._haptic = None
        self._effect_id = None
        self._sdl_initialized = False
        self._vjoy_acquired = False
        self._autocenter_disabled_confirmed = False
        self._closed = False
        # SDL reports 0 on every axis until the wheel sends its first report, and a
        # raw 0 pedal means half pressed. Treat each pedal as released until it moves.
        self._first_pedals = None
        self._pedal_seen = [False, False]   # brake a1, throttle a2
        self._vjoy_id = vjoy_device_id
        self.cleanup_errors: tuple[str, ...] = ()
        try:
            self._sdl = importlib.import_module("sdl2")
            self._vjoy = importlib.import_module("pyvjoy")
            self._sdk = importlib.import_module("pyvjoy._sdk")
            self._open_sdl()
            if self.use_vjoy:
                self._open_vjoy()
        except BaseException:
            # Includes KeyboardInterrupt and pyvjoy's SystemExit on DLL failure.
            try:
                self.close()
            except Exception:
                pass
            raise

    def _sdl_error(self, operation: str) -> HardwareError:
        detail = (self._sdl.SDL_GetError() or b"unknown SDL error").decode(errors="replace")
        return HardwareError(f"{operation}: {detail}")

    def _check_sdl(self, result: int, operation: str) -> None:
        if result < 0:
            raise self._sdl_error(operation)

    @staticmethod
    def _check_vjoy(result, operation: str) -> None:
        if not result:
            raise HardwareError(f"vJoy {operation} failed")

    def _open_sdl(self) -> None:
        sdl = self._sdl
        if not sdl.SDL_SetHint(sdl.SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, b"1"):
            raise self._sdl_error("Enable background wheel events")
        if not self.use_motor:
            # Without the motor, read the TMX through RawInput, never DirectInput:
            # SDL's DirectInput backend acquires force-feedback wheels exclusively,
            # which kills Forza's own FFB when Forza reads the TMX directly
            # (same fix as record.py). RawInput is read-only and can't drive haptics.
            sdl.SDL_SetHint(sdl.SDL_HINT_DIRECTINPUT_ENABLED, b"0")
            sdl.SDL_SetHint(sdl.SDL_HINT_JOYSTICK_RAWINPUT, b"1")
        # Initialize separately so a partial initialization can always be undone.
        self._sdl_flags = []
        flags = (sdl.SDL_INIT_JOYSTICK, sdl.SDL_INIT_HAPTIC) if self.use_motor else (sdl.SDL_INIT_JOYSTICK,)
        for flag in flags:
            self._check_sdl(sdl.SDL_InitSubSystem(flag), "SDL_InitSubSystem")
            self._sdl_flags.append(flag)
            self._sdl_initialized = True
        count = sdl.SDL_NumJoysticks()
        self._check_sdl(count, "SDL_NumJoysticks")
        if not self.use_motor:
            # RawInput lists devices a moment after initialization.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not any(
                    b"tmx" in (sdl.SDL_JoystickNameForIndex(i) or b"").lower() for i in range(count)):
                sdl.SDL_JoystickUpdate()
                time.sleep(0.05)
                count = sdl.SDL_NumJoysticks()
                self._check_sdl(count, "SDL_NumJoysticks")
        for index in range(count):
            name = sdl.SDL_JoystickNameForIndex(index)
            if name is None:
                raise self._sdl_error("SDL_JoystickNameForIndex")
            name = name.decode(errors="replace").lower()
            # The tested calibration is TMX-specific, not shared by every
            # Thrustmaster product discovered by the diagnostic script.
            if "vjoy" not in name and "tmx" in name:
                self._joystick = sdl.SDL_JoystickOpen(index)
                break
        else:
            raise HardwareError("No TMX wheel found; other wheel models require their own calibration")
        if not self._joystick:
            raise self._sdl_error("Open TMX (plug in and power the wheel)")
        axes = sdl.SDL_JoystickNumAxes(self._joystick)
        self._check_sdl(axes, "SDL_JoystickNumAxes")
        if axes < 3:
            raise HardwareError("TMX must expose steering a0, brake a1, throttle a2")
        self._button_count = sdl.SDL_JoystickNumButtons(self._joystick)
        self._check_sdl(self._button_count, "SDL_JoystickNumButtons")
        if any(button >= self._button_count for button in self.button_map):
            raise HardwareError("button_map references a missing TMX button")
        if not self.use_motor:
            return
        self._haptic = sdl.SDL_HapticOpenFromJoystick(self._joystick)
        if not self._haptic:
            raise self._sdl_error("SDL_HapticOpenFromJoystick")
        supported = sdl.SDL_HapticQuery(self._haptic)
        if not supported & sdl.SDL_HAPTIC_CONSTANT:
            raise HardwareError("TMX does not report SDL constant-force support")
        if supported & sdl.SDL_HAPTIC_GAIN:
            self._check_sdl(sdl.SDL_HapticSetGain(self._haptic, 100), "SDL_HapticSetGain")
        if supported & sdl.SDL_HAPTIC_AUTOCENTER:
            self._check_sdl(sdl.SDL_HapticSetAutocenter(self._haptic, 0), "SDL_HapticSetAutocenter")
            self._autocenter_disabled_confirmed = True
        self._effect = sdl.SDL_HapticEffect()
        self._effect.type = sdl.SDL_HAPTIC_CONSTANT
        self._effect.constant.type = sdl.SDL_HAPTIC_CONSTANT
        self._effect.constant.direction.type = sdl.SDL_HAPTIC_CARTESIAN
        self._effect.constant.direction.dir[0] = 1
        self._effect.constant.length = self.effect_ttl_ms
        self._effect.constant.level = 0
        effect_id = sdl.SDL_HapticNewEffect(self._haptic, self._effect)
        self._check_sdl(effect_id, "SDL_HapticNewEffect")
        self._effect_id = effect_id

    def _open_vjoy(self) -> None:
        # pyvjoyffb 0.4's VJoyDevice destructor relinquishes even when acquisition
        # failed. Own the SDK lifecycle explicitly instead of using that object.
        self._check_vjoy(self._sdk.vJoyEnabled(), "enabled")
        if self._sdk.GetVJDStatus(self._vjoy_id) != self._vjoy.VJD_STAT_FREE:
            raise HardwareError(f"vJoy device {self._vjoy_id} is not free")
        self._check_vjoy(self._sdk.AcquireVJD(self._vjoy_id), "acquire")
        self._vjoy_acquired = True
        self._reset_virtual()

    def _ensure_open(self) -> None:
        if self._closed:
            raise HardwareError("WindowsAdapter is closed")

    @property
    def button_count(self) -> int:
        """Number of physical SDL buttons, for validating takeover bindings."""
        return self._button_count

    @property
    def autocenter_disabled_confirmed(self) -> bool:
        """Whether SDL explicitly disabled native autocenter at startup."""
        return self._autocenter_disabled_confirmed

    def _attached(self) -> bool:
        return bool(self._sdl.SDL_JoystickGetAttached(self._joystick))

    def read_state(self, now_ns: int) -> WheelState:
        """Poll the device; timestamp denotes host poll time, not a device clock.

        SDL's DirectInput backend can silently keep cached values after a failed
        native poll. Attachment/error checks do not certify a fresh USB report.
        """
        self._ensure_open()
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be nonnegative monotonic nanoseconds")
        sdl = self._sdl
        sdl.SDL_JoystickUpdate()  # void API
        if not self._attached():
            if self._haptic:
                self._check_sdl(sdl.SDL_HapticStopAll(self._haptic), "SDL_HapticStopAll")
            return WheelState(now_ns, 0, 0, 0, connected=False)
        # Axis/button APIs use zero for both a valid value and failure. Clear and
        # inspect SDL's error around the whole group, after validating indices.
        sdl.SDL_ClearError()
        axes = [sdl.SDL_JoystickGetAxis(self._joystick, index) for index in range(3)]
        buttons = tuple(index for index in range(self._button_count)
                        if sdl.SDL_JoystickGetButton(self._joystick, index))
        if sdl.SDL_GetError():
            raise self._sdl_error("Read TMX axes/buttons")
        if not self._attached():
            if self._haptic:
                self._check_sdl(sdl.SDL_HapticStopAll(self._haptic), "SDL_HapticStopAll")
            return WheelState(now_ns, 0, 0, 0, connected=False)
        pedals = (axes[1], axes[2])
        if self._first_pedals is None:
            self._first_pedals = pedals
        # Exactly 0 is SDL's "no report yet" value (a released TMX pedal reads +32767);
        # any nonzero or changed value is a real report.
        self._pedal_seen = [seen or raw != 0 or raw != first
                            for seen, raw, first in zip(self._pedal_seen, pedals, self._first_pedals)]
        brake = pedal_fraction(axes[1]) if self._pedal_seen[0] else 0.0
        throttle = pedal_fraction(axes[2]) if self._pedal_seen[1] else 0.0
        return WheelState(now_ns, steering_degrees(axes[0], self.rotation_deg), throttle, brake, buttons)

    def _axis(self, usage: int, value: int) -> None:
        self._check_vjoy(self._sdk.SetAxis(value, self._vjoy_id, usage), "set axis")

    def write_virtual_state(self, state: WheelState) -> None:
        self._ensure_open()
        if not self.use_vjoy:
            return
        if not state.connected:
            self._reset_virtual()
            return
        # Convert and validate all values before sending any part of the sample.
        x = vjoy_steering(state.angle_deg, self.rotation_deg)
        y, z = vjoy_pedal(state.brake), vjoy_pedal(state.throttle)
        self._axis(self._vjoy.HID_USAGE_X, x)
        self._axis(self._vjoy.HID_USAGE_Y, y)
        self._axis(self._vjoy.HID_USAGE_Z, z)
        for physical, virtual in self.button_map.items():
            self._check_vjoy(self._sdk.SetBtn(physical in state.buttons, self._vjoy_id, virtual),
                             "set button")

    def _reset_virtual(self) -> None:
        # Continue every neutralization step if one fails. A missing X axis or a
        # failed ResetVJD must not prevent an otherwise writable pedal release.
        # ResetVJD centers axes; our inverted pedals must instead be released.
        operations = (
            lambda: self._check_vjoy(self._sdk.ResetVJD(self._vjoy_id), "reset"),
            lambda: self._axis(self._vjoy.HID_USAGE_X, 16384),
            lambda: self._axis(self._vjoy.HID_USAGE_Y, 32768),
            lambda: self._axis(self._vjoy.HID_USAGE_Z, 32768),
            lambda: self._check_vjoy(self._sdk.ResetButtons(self._vjoy_id), "reset buttons"),
        )
        errors = []
        for operation in operations:
            try:
                operation()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise HardwareError("vJoy neutralization failed: " + "; ".join(errors))

    def set_torque(self, torque: float) -> None:
        """Legacy finite effect, or unconditional zero-torque cleanup.

        Live control must use ``set_torque_before`` for nonzero output so driver
        work cannot silently renew a command whose input samples have expired.
        """
        self._set_torque(torque)

    def set_torque_before(self, torque: float, deadline_ns: int) -> None:
        """Start a finite effect only while its absolute host deadline permits.

        Reserve time for native update/start calls and recheck their returns.
        Expiry stops effects and raises ``ActuationExpired`` for the runtime to
        latch disengagement. Zero output always stops even after the deadline.

        SDL calls are synchronous and cannot provide a hard real-time guarantee:
        a blocked RunEffect may start force before returning. A late return is
        stopped immediately; the finite driver effect is the fallback while the
        call is blocked. Actual duration enforcement still needs hardware tests.
        """
        self._set_torque(torque, deadline_ns=deadline_ns)

    def _set_torque(self, torque: float, *, deadline_ns=_NO_DEADLINE) -> None:
        self._ensure_open()
        if not self.use_motor:
            if sdl_force_level(torque, self.torque_limit):
                raise HardwareError("motor output disabled (direct-vJoy mode)")
            return
        sdl = self._sdl
        try:
            level = sdl_force_level(torque, self.torque_limit)
            timed = deadline_ns is not _NO_DEADLINE
            if timed and (type(deadline_ns) is not int or deadline_ns < 0):
                raise ValueError("deadline_ns must be nonnegative monotonic nanoseconds")
            if timed and level and time.monotonic_ns() >= deadline_ns:
                raise ActuationExpired("actuation deadline expired before native output")
            if not self._attached():
                raise HardwareError("TMX disconnected")
            self._check_sdl(sdl.SDL_HapticStopEffect(self._haptic, self._effect_id),
                            "SDL_HapticStopEffect")
            if level == 0:
                return
            self._effect.constant.level = level
            effect_ms = self.effect_ttl_ms
            if timed:
                # Whole milliseconds only: rounding up would exceed validity.
                available_ms = (deadline_ns - time.monotonic_ns()) // 1_000_000
                # Leave 1--5 ms for UpdateEffect/RunEffect. If native work spends
                # that allowance, fail closed instead of starting a too-long
                # effect or entering an unbounded update/retry loop.
                reserve_ms = max(1, min(5, available_ms // 4))
                effect_ms = min(effect_ms, available_ms - reserve_ms)
                if effect_ms < 1:
                    raise ActuationExpired("actuation deadline exhausted after native stop")
            self._effect.constant.length = effect_ms
            # SDL 2.32.10's DirectInput UpdateEffect only calls SetParameters.
            # Do not assume it renews duration. Stop/update/run exactly once:
            # https://github.com/libsdl-org/SDL/blob/release-2.32.10/src/haptic/windows/SDL_dinputhaptic.c
            # https://wiki.libsdl.org/SDL2/SDL_HapticRunEffect
            self._check_sdl(sdl.SDL_HapticUpdateEffect(self._haptic, self._effect_id, self._effect),
                            "SDL_HapticUpdateEffect")
            if timed and time.monotonic_ns() + effect_ms * 1_000_000 > deadline_ns:
                raise ActuationExpired("actuation deadline exhausted by native update")
            self._check_sdl(sdl.SDL_HapticRunEffect(self._haptic, self._effect_id, 1),
                            "SDL_HapticRunEffect")
            if timed and time.monotonic_ns() + effect_ms * 1_000_000 > deadline_ns:
                raise ActuationExpired("actuation deadline exhausted by native start")
        except BaseException:
            try:
                self._check_sdl(sdl.SDL_HapticStopAll(self._haptic), "SDL_HapticStopAll")
            except Exception:
                pass
            raise

    def _release_vjoy(self) -> None:
        # Official SDK declares VOID, but pyvjoyffb 0.4 checks an undefined int
        # return and can raise after successful release. Use the typed void API.
        # https://github.com/jshafer817/vJoy/blob/master/SDK/inc/vjoyinterface.h
        release = self._sdk._vj.RelinquishVJD
        release.argtypes = [c_uint]
        release.restype = None
        release(self._vjoy_id)
        if self._sdk.GetVJDStatus(self._vjoy_id) == self._vjoy.VJD_STAT_OWN:
            raise HardwareError("vJoy device remains owned after relinquish")

    def close(self) -> None:
        """Best-effort cleanup of every acquired resource; report any failures."""
        if self._closed:
            return
        self._closed = True
        errors = []

        def attempt(operation):
            try:
                operation()
            except Exception as error:
                errors.append(str(error))

        if self._haptic:
            attempt(lambda: self._check_sdl(self._sdl.SDL_HapticStopAll(self._haptic),
                                           "SDL_HapticStopAll"))
        if self._vjoy_acquired:
            attempt(self._reset_virtual)
            attempt(self._release_vjoy)
            self._vjoy_acquired = False
        if self._haptic:
            if self._effect_id is not None:
                attempt(lambda: self._sdl.SDL_HapticDestroyEffect(self._haptic, self._effect_id))
            attempt(lambda: self._sdl.SDL_HapticClose(self._haptic))
            self._haptic = None
        if self._joystick:
            attempt(lambda: self._sdl.SDL_JoystickClose(self._joystick))
            self._joystick = None
        if self._sdl_initialized:
            for flag in reversed(self._sdl_flags):
                attempt(lambda flag=flag: self._sdl.SDL_QuitSubSystem(flag))
            self._sdl_initialized = False
        self.cleanup_errors = tuple(errors)
        if errors:
            raise HardwareError("Cleanup failed: " + "; ".join(errors))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
