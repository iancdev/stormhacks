"""Track the force-feedback effects a game sends to the virtual vJoy wheel.

Forza Horizon 4 drives one constant-force effect (~60 updates/s, 50 ms duration, magnitude up to
+-10000, gain 255) and periodic effects at gain 0. With HidHide hiding the TMX, those forces land on
vJoy; this tracker turns the packets into one normalized force in [-1, 1] that the runtime can
replay on the physical wheel while the human drives. Packets arrive on vJoy's callback thread; the
public state is replaced atomically, never mutated in place.
"""

import time

PT_EFFREP, PT_CONSTREP, PT_EFOPREP, PT_BLKFRREP, PT_CTRLREP, PT_GAINREP = 0x01, 0x05, 0x0A, 0x0B, 0x0C, 0x0D
ET_CONST = 1
EFF_START, EFF_SOLO, EFF_STOP = 1, 2, 3
CTRL_ENACT, CTRL_DISACT, CTRL_STOPALL, CTRL_DEVRST, CTRL_DEVPAUSE, CTRL_DEVCONT = 1, 2, 3, 4, 5, 6


def _field(data, name, default=0):
    if isinstance(data, dict):
        return data.get(name, default)
    return getattr(data, name, default)


class GameForceTracker:
    def __init__(self, clock_ns=time.monotonic_ns):
        self._clock = clock_ns
        self._effects = {}           # ebi -> dict(type, gain, magnitude, duration_ms, running, started_ns)
        self._device_gain = 255
        self._enabled = True
        self._paused = False
        self.last_packet_ns = None
        self.packets = 0

    def handle(self, data, reptype, now_ns=None):
        """vJoy FFB callback: (parsed packet, report type). Never raises into the driver thread."""
        try:
            now = self._clock() if now_ns is None else now_ns
            self.last_packet_ns, self.packets = now, self.packets + 1
            if reptype == PT_CTRLREP:
                if data in (CTRL_DISACT, CTRL_STOPALL, CTRL_DEVRST):
                    for effect in self._effects.values():
                        effect["running"] = False
                    if data == CTRL_DEVRST:
                        self._effects = {}
                self._enabled = data != CTRL_DISACT if data in (CTRL_ENACT, CTRL_DISACT, CTRL_DEVRST) else self._enabled
                self._paused = data == CTRL_DEVPAUSE if data in (CTRL_DEVPAUSE, CTRL_DEVCONT, CTRL_DEVRST) else self._paused
                return
            if reptype == PT_GAINREP:
                self._device_gain = max(0, min(255, int(data)))
                return
            ebi = int(_field(data, "EffectBlockIndex", 0)) if not isinstance(data, int) else int(data)
            if reptype == PT_BLKFRREP:
                self._effects.pop(ebi, None)
                return
            effect = self._effects.setdefault(ebi, {"type": None, "gain": 255, "magnitude": 0,
                                                    "duration_ms": 0, "running": False, "started_ns": now})
            if reptype == PT_EFFREP:
                effect.update(type=int(_field(data, "EffectType")), gain=int(_field(data, "Gain", 255)),
                              duration_ms=int(_field(data, "Duration", 0)))
            elif reptype == PT_CONSTREP:
                effect["magnitude"] = max(-10000, min(10000, int(_field(data, "Magnitude"))))
            elif reptype == PT_EFOPREP:
                op = int(_field(data, "EffectOp"))
                if op in (EFF_START, EFF_SOLO):
                    if op == EFF_SOLO:
                        for other in self._effects.values():
                            other["running"] = False
                    effect.update(running=True, started_ns=now)
                elif op == EFF_STOP:
                    effect["running"] = False
        except Exception:
            pass

    def force(self, now_ns=None, max_age_ns=150_000_000):
        """Normalized constant force in [-1, 1]; 0 when stale, paused, disabled or nothing runs."""
        now = self._clock() if now_ns is None else now_ns
        if (not self._enabled or self._paused or self.last_packet_ns is None
                or now - self.last_packet_ns > max_age_ns):
            return 0.0
        total = 0.0
        for effect in list(self._effects.values()):
            if effect["type"] != ET_CONST or not effect["running"]:
                continue
            duration = effect["duration_ms"]
            if duration and duration != 0xFFFF and now - effect["started_ns"] > duration * 1_000_000:
                continue                                    # finite effect already over
            total += effect["magnitude"] / 10000 * effect["gain"] / 255
        return max(-1.0, min(1.0, total * self._device_gain / 255))
