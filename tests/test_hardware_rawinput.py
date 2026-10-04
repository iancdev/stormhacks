"""Motorless (direct-vJoy) adapter: RawInput, no haptics, and no phantom half-pressed pedals."""

import unittest
from unittest.mock import call

from forza_ai.contracts import WheelState
from forza_ai.hardware import HardwareError

from test_hardware import Rig


class MotorlessAdapterTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()
        self.rig.sdl.SDL_HINT_DIRECTINPUT_ENABLED = b"dinput"
        self.rig.sdl.SDL_HINT_JOYSTICK_RAWINPUT = b"rawinput"

    def open(self, **kwargs):
        adapter = self.rig.adapter(**kwargs)
        self.addCleanup(adapter.close)
        return adapter

    def test_motorless_uses_rawinput_and_never_opens_haptics(self):
        adapter = self.open(use_motor=False)
        hints = self.rig.sdl.SDL_SetHint.call_args_list
        self.assertIn(call(b"dinput", b"0"), hints)
        self.assertIn(call(b"rawinput", b"1"), hints)
        self.rig.sdl.SDL_HapticOpenFromJoystick.assert_not_called()
        self.assertNotIn(call(2), self.rig.sdl.SDL_InitSubSystem.call_args_list)  # no SDL_INIT_HAPTIC
        adapter.set_torque(0.0)                                # zero is always allowed
        with self.assertRaises(HardwareError):
            adapter.set_torque(0.1)

    def test_motor_mode_keeps_directinput_and_haptics(self):
        self.open()
        self.assertNotIn(call(b"dinput", b"0"), self.rig.sdl.SDL_SetHint.call_args_list)
        self.rig.sdl.SDL_HapticOpenFromJoystick.assert_called_once()

    def test_pedals_read_released_until_the_wheel_reports(self):
        adapter = self.open(use_motor=False)
        self.rig.axes = [0, 0, 0]                              # SDL before the first device report
        self.assertEqual(adapter.read_state(1), WheelState(1, 0, 0.0, 0.0, ()))
        self.rig.axes = [0, 0, -32768]                         # throttle floored: real report
        state = adapter.read_state(2)
        self.assertEqual((state.throttle, state.brake), (1.0, 0.0))
        self.rig.axes = [0, 32767, 32767]                      # brake reports released
        state = adapter.read_state(3)
        self.assertEqual((state.throttle, state.brake), (0.0, 0.0))
        self.rig.axes = [0, 0, 32767]                          # later raw 0 is a real half press
        self.assertAlmostEqual(adapter.read_state(4).brake, 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
