"""Live driving-style tuning: bounded, step-limited changes while the AI drives.

The voice assistant (or any other operator tool) never touches the controller directly. It sends
("tune", adjustments, reply_queue) on the runtime's event queue and the control loop applies them
here, on its own thread, between ticks. Every value stays inside the same ranges as run-ai.ps1, and
one command can only move a setting by a limited step, so a misheard "a LOT faster" can't jump from
a gentle throttle to flat out.
"""

import math

# name: (minimum, maximum, largest change per command, description for the LLM)
PARAMETERS = {
    "throttle_cap": (0.1, 1.0, 0.2, "maximum AI throttle, 0-1 (more = faster top speed and acceleration)"),
    "throttle_rate": (0.1, 5.0, 1.0, "how fast the AI throttle may rise, per second (more = harder acceleration)"),
    "max_speed_kmh": (0.0, 400.0, None, "no AI throttle above this speed; 0 = no speed limit"),
    "brake_gain": (0.5, 4.0, 0.5, "multiplies the model's brake (more = brakes harder)"),
    "corner_speed_kmh": (0.0, 400.0, None, "brake when cornering above this speed; 0 = off (lower = slower in corners)"),
    "steer_gain": (0.5, 3.0, 0.4, "multiplies the model's steering angle (more = turns in harder)"),
}
# A speed limit below this is almost certainly a mishearing ("fifteen" for "fifty"), not a request.
MIN_SPEED_LIMIT_KMH = 40.0


class LiveTuning:
    def __init__(self, **values):
        unknown = set(values) - set(PARAMETERS)
        if unknown:
            raise ValueError(f"unknown tuning parameters: {sorted(unknown)}")
        self.values = dict(values)

    def __getattr__(self, name):
        try:
            return self.__dict__["values"][name]
        except KeyError:
            raise AttributeError(name) from None

    def snapshot(self):
        return dict(self.values)

    def apply(self, adjustments):
        """Apply [{"parameter", "mode": "set"|"change", "value"}]; returns one result per adjustment.

        Unknown parameters, settings this run doesn't use and non-numbers are skipped, never raised:
        the control loop must keep running whatever a voice command contained.
        """
        results = []
        for item in adjustments or ():
            name = item.get("parameter") if isinstance(item, dict) else None
            mode = item.get("mode", "change") if isinstance(item, dict) else None
            value = item.get("value") if isinstance(item, dict) else None
            if name not in PARAMETERS or name not in self.values:
                results.append({"parameter": name, "applied": False, "note": "not adjustable in this run"})
                continue
            if mode not in ("set", "change") or isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                results.append({"parameter": name, "applied": False, "note": "invalid value"})
                continue
            low, high, step, _ = PARAMETERS[name]
            old = self.values[name]
            wanted = float(value) if mode == "set" else old + float(value)
            note = ""
            if name in ("max_speed_kmh", "corner_speed_kmh"):
                # Limits: 0 means off; "change" on an off limit has nothing to change from.
                if mode == "change" and old == 0:
                    results.append({"parameter": name, "applied": False, "note": "limit is off; say a speed"})
                    continue
                if 0 < wanted < MIN_SPEED_LIMIT_KMH:
                    wanted, note = MIN_SPEED_LIMIT_KMH, f"raised to the {MIN_SPEED_LIMIT_KMH:g} km/h minimum"
            if step is not None and abs(wanted - old) > step + 1e-9:
                wanted, note = old + math.copysign(step, wanted - old), f"limited to a {step:g} step"
            new = round(min(high, max(low, wanted)), 3)
            if new != round(wanted, 3) and not note:
                note = f"clamped to {low:g}-{high:g}"
            self.values[name] = new
            results.append({"parameter": name, "applied": new != old, "old": old, "new": new, "note": note})
        return results
