"""Opt-in motor command throttling: fewer USB commands, never longer force than already allowed."""

import time

from test_hardware import Rig


def counts(rig):
    return (rig.sdl.SDL_HapticStopEffect.call_count, rig.sdl.SDL_HapticRunEffect.call_count)


def test_default_resends_every_call():
    rig = Rig(); adapter = rig.adapter()
    try:
        for _ in range(5):
            adapter.set_torque(0.1)
        assert counts(rig)[1] == 5
    finally:
        adapter.close()


def test_unchanged_force_not_resent_within_window_and_repeated_stop_skipped():
    rig = Rig(); adapter = rig.adapter(min_update_ms=30)
    try:
        for _ in range(5):
            adapter.set_torque(0.1)
        assert counts(rig)[1] == 1                       # same force: one start only
        adapter.set_torque(0.12)
        assert counts(rig)[1] == 2                       # changed force: sent at once
        stops = rig.sdl.SDL_HapticStopEffect.call_count
        adapter.set_torque(0.0); adapter.set_torque(0.0); adapter.set_torque(0.0)
        assert rig.sdl.SDL_HapticStopEffect.call_count == stops + 1   # stop sent once
        time.sleep(0.035)
        adapter.set_torque(0.12)
        assert counts(rig)[1] == 3
    finally:
        adapter.close()


def test_renewal_after_window_and_quantization():
    rig = Rig(); adapter = rig.adapter(min_update_ms=30, torque_step=0.02)
    try:
        adapter.set_torque(0.101); adapter.set_torque(0.104)     # both round to the same 0.10 step
        assert counts(rig)[1] == 1
        time.sleep(0.035)
        adapter.set_torque(0.101)
        assert counts(rig)[1] == 2                                # renewed after the window
    finally:
        adapter.close()
