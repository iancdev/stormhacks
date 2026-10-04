"""Small deterministic wheel plant for integration tests, not a TMX physics model."""

import math
import time
from forza_ai.contracts import ActuationExpired, WheelState


class SimulatedAdapter:
    def __init__(self, torque_limit: float = 0.15):
        self.angle = 0.0
        self.velocity = 0.0
        self.torque = 0.0
        self.torque_limit = torque_limit
        self.buttons = ()
        self.closed = False
        self.virtual_state = None
        self._last_ns = None

    def read_state(self, now_ns: int) -> WheelState:
        if self.closed:
            raise RuntimeError("adapter is closed")
        if self._last_ns is not None:
            dt = max(0.0, min(0.1, (now_ns - self._last_ns) / 1e9))
            self.velocity += (4000 * self.torque - 8 * self.velocity) * dt
            self.angle = max(-450.0, min(450.0, self.angle + self.velocity * dt))
        self._last_ns = now_ns
        return WheelState(now_ns, self.angle, 0.0, 0.0, self.buttons)

    def write_virtual_state(self, state: WheelState):
        self.virtual_state = state

    def set_torque(self, torque: float):
        if not math.isfinite(torque):
            raise ValueError("torque must be finite")
        self.torque = max(-self.torque_limit, min(self.torque_limit, torque))

    def close(self):
        self.torque = 0.0
        self.closed = True
        self.virtual_state = None

    def set_torque_before(self, torque: float, deadline_ns: int):
        if type(deadline_ns) is not int or deadline_ns < 0:
            raise ValueError("deadline_ns must be nonnegative integer nanoseconds")
        if time.monotonic_ns() >= deadline_ns:
            self.set_torque(0.0)
            raise ActuationExpired("simulation command expired before output")
        self.set_torque(torque)
        if time.monotonic_ns() >= deadline_ns:
            self.set_torque(0.0)
            raise ActuationExpired("simulation output returned after its deadline")
