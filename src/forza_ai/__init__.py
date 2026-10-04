"""Forza steering assistance. Hardware and training dependencies are optional."""

import sys as _sys
import time as _time

# Before Python 3.13, Windows time.monotonic() is GetTickCount64 with ~15.6 ms
# resolution. The 100 Hz control loop then sees repeated timestamps (a zero tick
# gap faults as control_loop_gap) and 50 ms freshness checks are quantized.
# Python 3.13 made monotonic() use QueryPerformanceCounter, like perf_counter().
# Do the same here so every forza_ai clock read (and tests that build timestamps
# with time.monotonic_ns) shares one precise, monotonic clock.
if _sys.platform == "win32" and _sys.version_info < (3, 13):
    _time.monotonic = _time.perf_counter
    _time.monotonic_ns = _time.perf_counter_ns
