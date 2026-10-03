"""Hardware protocol tests using fakes: no Windows, SDL DLL, vJoy, or motor."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from forza_ai.contracts import WheelState
from forza_ai.hardware import HardwareError, WindowsAdapter
from forza_ai.hardware.conversions import (
    pedal_fraction, sdl_force_level, steering_degrees, vjoy_pedal, vjoy_steering,
)


class Rig:
    """Model finite effect expiry and explicit device acquisition/ownership."""

    def __init__(self):
        self.now_ms = 0
        self.deadline_ms = 0
        self.effect_length = None
        self.level = None
        self.events = []
        self.error = b""
        self.axes = [0, 32767, 32767]
        self.buttons = set()
        self.virtual_axes = {}
        self.virtual_buttons = {}
        self.status = 1
        self.sdl = SimpleNamespace(
            SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS=b"background",
            SDL_INIT_JOYSTICK=1, SDL_INIT_HAPTIC=2,
            SDL_HAPTIC_CONSTANT=1, SDL_HAPTIC_GAIN=2, SDL_HAPTIC_AUTOCENTER=4,
            SDL_HAPTIC_CARTESIAN=0,
        )
        self.sdl.SDL_GetError = Mock(side_effect=lambda: self.error)
        self.sdl.SDL_ClearError = Mock(side_effect=lambda: setattr(self, "error", b""))
        self.sdl.SDL_HapticEffect = Mock(side_effect=lambda: SimpleNamespace(
            type=0, constant=SimpleNamespace(type=0, length=0, level=0,
                                            direction=SimpleNamespace(type=0, dir=[0, 0, 0]))))
        for name, result in {
            "SDL_SetHint": 1, "SDL_InitSubSystem": 0, "SDL_NumJoysticks": 2,
            "SDL_JoystickOpen": "wheel", "SDL_JoystickNumAxes": 3,
            "SDL_JoystickNumButtons": 4, "SDL_HapticOpenFromJoystick": "haptic",
            "SDL_HapticQuery": 7, "SDL_HapticSetGain": 0,
            "SDL_HapticSetAutocenter": 0, "SDL_HapticNewEffect": 0,
            "SDL_JoystickGetAttached": 1,
        }.items():
            setattr(self.sdl, name, Mock(return_value=result))
        for name in ("SDL_JoystickUpdate", "SDL_HapticDestroyEffect", "SDL_HapticClose",
                     "SDL_JoystickClose", "SDL_QuitSubSystem"):
            setattr(self.sdl, name, Mock())
        self.sdl.SDL_JoystickNameForIndex = Mock(side_effect=lambda index: [b"vJoy", b"Thrustmaster TMX"][index])
        self.sdl.SDL_JoystickGetAxis = Mock(side_effect=lambda joystick, index: self.axes[index])
        self.sdl.SDL_JoystickGetButton = Mock(side_effect=lambda joystick, index: index in self.buttons)
        self.sdl.SDL_HapticStopEffect = Mock(side_effect=self.stop)
        self.sdl.SDL_HapticUpdateEffect = Mock(side_effect=self.update)
        self.sdl.SDL_HapticRunEffect = Mock(side_effect=self.run)
        self.sdl.SDL_HapticStopAll = Mock(side_effect=self.stop_all)
        self.vjoy = SimpleNamespace(HID_USAGE_X=0x30, HID_USAGE_Y=0x31, HID_USAGE_Z=0x32,
                                    VJD_STAT_FREE=1, VJD_STAT_OWN=0)
        self.sdk = SimpleNamespace(
            vJoyEnabled=Mock(return_value=True),
            GetVJDStatus=Mock(side_effect=lambda device: self.status),
            AcquireVJD=Mock(side_effect=self.acquire),
            ResetVJD=Mock(side_effect=self.reset),
            ResetButtons=Mock(side_effect=self.reset_buttons),
            SetAxis=Mock(side_effect=self.axis),
            SetBtn=Mock(side_effect=self.button),
            _vj=SimpleNamespace(RelinquishVJD=Mock(side_effect=self.release)),
        )

    @property
    def active(self):
        return self.now_ms < self.deadline_ms

    def stop(self, *args):
        self.events.append("stop")
        self.deadline_ms = 0
        return 0

    def stop_all(self, *args):
        self.events.append("stop_all")
        self.deadline_ms = 0
        return 0

    def update(self, haptic, effect_id, effect):
        self.events.append("update")
        self.level = effect.constant.level
        self.effect_length = effect.constant.length
        return 0  # Deliberately does not renew expiry.

    def run(self, haptic, effect_id, iterations):
        self.events.append("run")
        assert iterations == 1
        self.deadline_ms = self.now_ms + self.effect_length
        return 0

    def acquire(self, device):
        self.status = 0
        return True

    def release(self, device):
        self.status = 1
        # VOID, deliberately returns None.

    def reset(self, device):
        self.virtual_axes = dict.fromkeys((0x30, 0x31, 0x32), 16384)
        self.virtual_buttons = {}
        return True

    def axis(self, value, device, usage):
        self.virtual_axes[usage] = value
        return True

    def reset_buttons(self, device):
        self.virtual_buttons = {}
        return True

    def button(self, pressed, device, button):
        self.virtual_buttons[button] = pressed
        return True

    def adapter(self, **kwargs):
        modules = {"sdl2": self.sdl, "pyvjoy": self.vjoy, "pyvjoy._sdk": self.sdk}
        with patch("forza_ai.hardware.windows.sys.platform", "win32"), \
             patch("forza_ai.hardware.windows.importlib.import_module", side_effect=modules.__getitem__):
            return WindowsAdapter(**kwargs)


class ConversionTests(unittest.TestCase):
    def test_asymmetric_signed_steering_endpoints_and_center(self):
        self.assertEqual([steering_degrees(v) for v in (-32768, 0, 32767)], [-450, 0, 450])
        self.assertEqual(steering_degrees(-16384), -225)
        self.assertAlmostEqual(steering_degrees(16384), 225.0068666646321)
        self.assertEqual([vjoy_steering(v) for v in (-450, 0, 450)], [1, 16384, 32768])
        self.assertEqual(vjoy_steering(-900), 1)
        self.assertEqual(vjoy_steering(900), 32768)

    def test_pedal_inversion_preserves_endpoints(self):
        self.assertEqual([pedal_fraction(v) for v in (32767, -32768)], [0, 1])
        self.assertEqual([vjoy_pedal(v) for v in (0, 1)], [32768, 1])
        for raw in (-32768, -12000, 0, 12000, 32767):
            self.assertAlmostEqual(vjoy_pedal(pedal_fraction(raw)),
                                   1 + (raw + 32768) * 32767 / 65535, delta=0.5)

    def test_force_sign_and_limit(self):
        self.assertEqual(sdl_force_level(10, 0.15), -4915)
        self.assertEqual(sdl_force_level(-10, 0.15), 4915)
        self.assertEqual(sdl_force_level(0, 0.15), 0)
        self.assertEqual(sdl_force_level(1, 0), 0)
        self.assertLessEqual(abs(sdl_force_level(1, 0.1)) / 32767, 0.1)

    def test_nonfinite_values_and_invalid_rotation_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for function in (steering_degrees, pedal_fraction, vjoy_steering, vjoy_pedal):
                with self.subTest(function=function.__name__, value=value), self.assertRaises(ValueError):
                    function(value)
            with self.assertRaises(ValueError):
                sdl_force_level(value, 0.15)
        with self.assertRaises(ValueError):
            vjoy_steering(0, 0)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()

    def open(self, **kwargs):
        adapter = self.rig.adapter(**kwargs)
        self.addCleanup(adapter.close)
        return adapter

    def test_non_windows_rejected_before_importing_drivers(self):
        with patch("forza_ai.hardware.windows.sys.platform", "darwin"), \
             patch("forza_ai.hardware.windows.importlib.import_module") as importer:
            with self.assertRaises(OSError):
                WindowsAdapter()
            importer.assert_not_called()

    def test_open_does_not_actuate_and_resets_inverted_pedals(self):
        self.open()
        self.rig.sdl.SDL_HapticRunEffect.assert_not_called()
        self.rig.sdl.SDL_JoystickOpen.assert_called_once_with(1)
        self.assertEqual(self.rig.virtual_axes, {0x30: 16384, 0x31: 32768, 0x32: 32768})

    def test_reads_measured_angle_and_inverted_brake_and_throttle(self):
        adapter = self.open()
        self.rig.axes = [-32768, -32768, 32767]
        self.rig.buttons = {0, 3}
        self.assertEqual(adapter.read_state(123), WheelState(123, -450, 0, 1, (0, 3)))

    def test_button_count_exposes_physical_count_read_only(self):
        adapter = self.open()
        self.assertEqual(adapter.button_count, 4)
        with self.assertRaises(AttributeError):
            adapter.button_count = 99

    def test_uncalibrated_thrustmaster_models_are_not_opened(self):
        self.rig.sdl.SDL_JoystickNameForIndex.side_effect = lambda index: [
            b"vJoy", b"Thrustmaster T150"][index]
        with self.assertRaisesRegex(HardwareError, "No TMX wheel found"):
            self.rig.adapter()
        self.rig.sdl.SDL_JoystickOpen.assert_not_called()
        self.rig.sdl.SDL_HapticOpenFromJoystick.assert_not_called()
        self.assertEqual(self.rig.sdl.SDL_QuitSubSystem.call_count, 2)

    def test_write_only_measured_state_and_explicit_button_map(self):
        adapter = self.open(button_map={0: 7, 3: 2})
        adapter.write_virtual_state(WheelState(123, 225, 0.25, 1, (0, 1)))
        self.assertEqual(self.rig.virtual_axes, {0x30: 24576, 0x31: 1, 0x32: 24576})
        self.assertEqual(self.rig.virtual_buttons, {7: True, 2: False})
        adapter.write_virtual_state(WheelState(124, 0, 0, 0, (3,)))
        self.assertEqual(self.rig.virtual_buttons, {7: False, 2: True})

    def test_no_default_game_button_mapping(self):
        adapter = self.open()
        adapter.write_virtual_state(WheelState(1, 0, 0, 0, (0, 1, 2)))
        self.rig.sdk.SetBtn.assert_not_called()

    def test_each_refresh_restarts_finite_effect_even_for_same_torque(self):
        adapter = self.open()
        adapter.set_torque(0.1)
        self.assertTrue(self.rig.active)
        self.assertLess(self.rig.level, 0)
        self.rig.now_ms = 75
        adapter.set_torque(0.1)
        self.assertEqual(self.rig.events, ["stop", "update", "run"] * 2)
        self.rig.now_ms = 101
        self.assertTrue(self.rig.active)  # Actually restarted; update alone cannot pass.
        self.rig.now_ms = 175
        self.assertFalse(self.rig.active)

    def test_zero_stops_without_new_run_and_invalid_torque_releases(self):
        adapter = self.open()
        adapter.set_torque(1)
        self.assertEqual(self.rig.level, -4915)
        adapter.set_torque(0)
        self.assertFalse(self.rig.active)
        self.assertEqual(self.rig.sdl.SDL_HapticRunEffect.call_count, 1)
        adapter.set_torque(0.1)
        with self.assertRaises(ValueError):
            adapter.set_torque(float("nan"))
        self.assertFalse(self.rig.active)

    def test_disconnection_stops_motor_and_yields_invalid_state(self):
        adapter = self.open()
        adapter.set_torque(0.1)
        self.rig.sdl.SDL_JoystickGetAttached.return_value = 0
        state = adapter.read_state(200)
        self.assertFalse(state.connected)
        self.assertFalse(self.rig.active)
        with self.assertRaisesRegex(HardwareError, "disconnected"):
            adapter.set_torque(0.1)
        adapter.write_virtual_state(state)
        self.assertEqual(self.rig.virtual_axes, {0x30: 16384, 0x31: 32768, 0x32: 32768})

    def test_read_error_is_not_mistaken_for_valid_zero(self):
        adapter = self.open()
        def fail_read(joystick, index):
            self.rig.error = b"input failed"
            return 0
        self.rig.sdl.SDL_JoystickGetAxis.side_effect = fail_read
        with self.assertRaisesRegex(HardwareError, "input failed"):
            adapter.read_state(1)

    def test_update_failure_stops_all_and_propagates(self):
        adapter = self.open()
        adapter.set_torque(0.1)
        self.rig.sdl.SDL_HapticUpdateEffect.side_effect = None
        self.rig.sdl.SDL_HapticUpdateEffect.return_value = -1
        with self.assertRaisesRegex(HardwareError, "SDL_HapticUpdateEffect"):
            adapter.set_torque(0.1)
        self.assertFalse(self.rig.active)
        self.rig.sdl.SDL_HapticStopAll.assert_called()

    def test_failed_haptic_acquisition_releases_joystick_and_sdl(self):
        self.rig.sdl.SDL_HapticOpenFromJoystick.return_value = None
        with self.assertRaises(HardwareError):
            self.rig.adapter()
        self.rig.sdl.SDL_JoystickClose.assert_called_once_with("wheel")
        self.assertEqual(self.rig.sdl.SDL_QuitSubSystem.call_count, 2)
        self.rig.sdk.AcquireVJD.assert_not_called()
        self.rig.sdk._vj.RelinquishVJD.assert_not_called()

    def test_partial_sdl_init_balances_only_successful_init(self):
        self.rig.sdl.SDL_InitSubSystem.side_effect = [0, -1]
        with self.assertRaises(HardwareError):
            self.rig.adapter()
        self.rig.sdl.SDL_QuitSubSystem.assert_called_once_with(1)

    def test_failed_vjoy_acquisition_does_not_release_unowned_device(self):
        self.rig.sdk.AcquireVJD.side_effect = RuntimeError("acquisition failed")
        with self.assertRaisesRegex(RuntimeError, "acquisition failed"):
            self.rig.adapter()
        self.rig.sdk._vj.RelinquishVJD.assert_not_called()
        self.rig.sdl.SDL_HapticClose.assert_called_once()
        self.rig.sdl.SDL_JoystickClose.assert_called_once()

    def test_vjoy_reset_failure_after_acquisition_still_releases(self):
        self.rig.sdk.ResetVJD.side_effect = None
        self.rig.sdk.ResetVJD.return_value = False
        with self.assertRaisesRegex(HardwareError, "reset"):
            self.rig.adapter()
        self.rig.sdk._vj.RelinquishVJD.assert_called_once_with(1)
        self.assertEqual(self.rig.status, 1)
        self.rig.sdl.SDL_HapticClose.assert_called_once()

    def test_close_is_idempotent_stops_resets_and_relinquishes(self):
        adapter = self.open(button_map={0: 1})
        adapter.write_virtual_state(WheelState(1, 450, 1, 1, (0,)))
        adapter.set_torque(0.1)
        adapter.close()
        adapter.close()
        self.assertFalse(self.rig.active)
        self.assertEqual(self.rig.virtual_axes, {0x30: 16384, 0x31: 32768, 0x32: 32768})
        self.assertEqual(self.rig.virtual_buttons, {})
        self.rig.sdk._vj.RelinquishVJD.assert_called_once_with(1)
        self.assertIsNone(self.rig.sdk._vj.RelinquishVJD.restype)
        self.rig.sdl.SDL_HapticDestroyEffect.assert_called_once_with("haptic", 0)
        self.rig.sdl.SDL_JoystickClose.assert_called_once()
        with self.assertRaisesRegex(HardwareError, "closed"):
            adapter.set_torque(0.1)

    def test_close_continues_after_stop_failure(self):
        adapter = self.open()
        self.rig.sdl.SDL_HapticStopAll.side_effect = None
        self.rig.sdl.SDL_HapticStopAll.return_value = -1
        with self.assertRaisesRegex(HardwareError, "Cleanup failed"):
            adapter.close()
        self.rig.sdk._vj.RelinquishVJD.assert_called_once()
        self.rig.sdl.SDL_HapticClose.assert_called_once()
        self.rig.sdl.SDL_JoystickClose.assert_called_once()
        self.assertTrue(adapter.cleanup_errors)

    def test_close_releases_pedals_and_buttons_despite_reset_and_steering_failure(self):
        adapter = self.open(button_map={0: 1})
        adapter.write_virtual_state(WheelState(1, 450, 1, 1, (0,)))
        self.rig.sdk.ResetVJD.side_effect = None
        self.rig.sdk.ResetVJD.return_value = False
        def axis_with_broken_steering(value, device, usage):
            if usage == 0x30:
                raise RuntimeError("X failed")
            return self.rig.axis(value, device, usage)
        self.rig.sdk.SetAxis.side_effect = axis_with_broken_steering
        with self.assertRaisesRegex(HardwareError, "Cleanup failed"):
            adapter.close()
        self.assertEqual(self.rig.virtual_axes[0x31], 32768)
        self.assertEqual(self.rig.virtual_axes[0x32], 32768)
        self.assertEqual(self.rig.virtual_buttons, {})
        self.rig.sdk._vj.RelinquishVJD.assert_called_once()

    def test_invalid_configuration_rejected(self):
        for configuration in ({"effect_ttl_ms": 0}, {"effect_ttl_ms": 1000},
                              {"torque_limit": 1.01}, {"button_map": {0: 0}},
                              {"button_map": {0: 1, 1: 1}}, {"rotation_deg": -1}):
            with self.subTest(configuration=configuration), self.assertRaises(ValueError):
                self.rig.adapter(**configuration)
        with self.assertRaisesRegex(HardwareError, "missing TMX button"):
            self.rig.adapter(button_map={10: 1})


if __name__ == "__main__":
    unittest.main()
