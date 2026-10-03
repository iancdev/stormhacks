"""Bounded-memory driving-run summaries and comparisons; no hardware imports.

Route outcomes are explicit operator markers, not inferred driving ground truth.
Latency histograms sample control ticks (including repeated prediction ages).
Percentiles are approximate upper bin bounds, with at most 4% bin width above
0.01 ms. Counts, means, maxima, and assisted tracking RMSE are exact aggregates.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


class _Histogram:
    """Fixed storage, even after millions of samples."""

    edges = (0.0,) + tuple(0.01 * 1.04 ** i for i in range(640))

    def __init__(self):
        self.bins = [0] * (len(self.edges) + 1)
        self.count = 0
        self.mean = 0.0
        self.maximum = 0.0

    def add(self, value):
        value = _number(value)
        if value is None or value < 0:
            return
        self.bins[bisect_left(self.edges, value)] += 1
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.maximum = max(self.maximum, value)

    def percentile(self, fraction):
        if not self.count:
            return None
        rank, seen = math.ceil(self.count * fraction), 0
        for index, count in enumerate(self.bins):
            seen += count
            if seen >= rank:
                return min(self.edges[index], self.maximum) if index < len(self.edges) else self.maximum
        return self.maximum

    def summary(self):
        return {
            "count": self.count,
            "mean_ms": self.mean if self.count else None,
            "max_ms": self.maximum if self.count else None,
            "p50_ms": self.percentile(0.50),
            "p95_ms": self.percentile(0.95),
            "p99_ms": self.percentile(0.99),
        }


class RunMetrics:
    """Single-writer aggregate, called by the runtime after each control tick.

    Call ``event('human_takeover', timestamp_ns)`` once per explicit intervention;
    a fault or timeout transition alone deliberately does not count. Timestamps
    use the runtime's monotonic clock. ``summary`` does not advance that clock.
    """

    MODES = ("manual", "assist", "takeover", "fault", "unknown")
    EVENTS = ("route_start", "route_complete", "route_abort", "human_takeover", "arm")

    def __init__(self):
        self.ticks = 0
        self._start_ns = None
        self._end_ns = None
        self._last_tick_ns = None
        self._last_mode = "unknown"
        self._mode_ns = dict.fromkeys(self.MODES, 0)
        self._events = dict.fromkeys(self.EVENTS, 0)
        self._histograms = {name: _Histogram() for name in
                            ("control_tick_gap", "observation_age", "inference")}
        self._tracking_count = 0
        self._tracking_mean_squared_error = 0.0
        self._max_abs_torque = 0.0
        self._out_of_order_ticks = 0
        self._fault_entries = 0
        self._ready_ticks = 0
        self._last_prediction_id = None
        self._inference_deduplicated = False
        self._route_start_ns = None
        self._route_attempts = 0
        self._route_completed = 0
        self._route_aborted = 0
        self._ignored_route_markers = 0
        self._route_duration = _Histogram()
        self._last_route_seconds = None

    @staticmethod
    def _timestamp(value):
        # Preserve integer ns precision; no float round-trip for real timestamps.
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2 ** 63 - 1:
            return value
        raise ValueError("timestamp_ns must be a nonnegative 64-bit integer")

    def _observe_time(self, timestamp_ns):
        self._start_ns = timestamp_ns if self._start_ns is None else min(self._start_ns, timestamp_ns)
        self._end_ns = timestamp_ns if self._end_ns is None else max(self._end_ns, timestamp_ns)

    def update(self, status):
        timestamp = self._timestamp(status["timestamp_ns"])
        mode = status.get("mode", "unknown")
        mode = mode if mode in self.MODES else "unknown"
        if self._last_tick_ns is not None and timestamp < self._last_tick_ns:
            self._out_of_order_ticks += 1
            return
        self._observe_time(timestamp)
        if self._last_tick_ns is not None:
            elapsed = timestamp - self._last_tick_ns
            self._mode_ns[self._last_mode] += elapsed
            self._histograms["control_tick_gap"].add(elapsed / 1e6)
        if mode == "fault" and self._last_mode != "fault":
            self._fault_entries += 1
        self._last_tick_ns, self._last_mode = timestamp, mode
        self.ticks += 1
        self._ready_ticks += status.get("input_status") == "ready"
        self._histograms["observation_age"].add(status.get("observation_age_ms"))
        prediction_id = status.get("prediction_id")
        if prediction_id is None or prediction_id != self._last_prediction_id:
            self._histograms["inference"].add(status.get("inference_ms"))
        if prediction_id is not None:
            self._inference_deduplicated = True
            self._last_prediction_id = prediction_id
        torque = _number(status.get("torque"))
        if torque is not None:
            self._max_abs_torque = max(self._max_abs_torque, abs(torque))
        actual, target = _number(status.get("actual_angle_deg")), _number(status.get("target_angle_deg"))
        if mode == "assist" and actual is not None and target is not None:
            difference = actual - target
            squared = difference * difference
            if math.isfinite(squared):
                self._tracking_count += 1
                self._tracking_mean_squared_error += (squared - self._tracking_mean_squared_error) / self._tracking_count

    def event(self, kind, timestamp_ns):
        if kind not in self.EVENTS:
            raise ValueError(f"unsupported run event: {kind}")
        timestamp = self._timestamp(timestamp_ns)
        self._observe_time(timestamp)
        self._events[kind] += 1
        if kind == "route_start":
            if self._route_start_ns is not None:
                self._ignored_route_markers += 1
                return
            self._route_start_ns = timestamp
            self._route_attempts += 1
        elif kind in ("route_complete", "route_abort"):
            if self._route_start_ns is None or timestamp < self._route_start_ns:
                self._ignored_route_markers += 1
                return
            elapsed_ms = (timestamp - self._route_start_ns) / 1e6
            self._last_route_seconds = elapsed_ms / 1000
            if kind == "route_complete":
                self._route_completed += 1
                self._route_duration.add(elapsed_ms)
            else:
                self._route_aborted += 1
            self._route_start_ns = None

    def summary(self):
        duration = 0.0 if self._start_ns is None else (self._end_ns - self._start_ns) / 1e9
        assist_seconds = self._mode_ns["assist"] / 1e9
        interventions = self._events["human_takeover"]
        return {
            "schema_version": 1,
            "duration_seconds": duration,
            "ticks": self.ticks,
            "mode_seconds": {key: value / 1e9 for key, value in self._mode_ns.items()},
            "human_interventions": interventions,
            "interventions_per_assist_minute": interventions * 60 / assist_seconds if assist_seconds else None,
            "events": dict(self._events),
            "fault_entries": self._fault_entries,
            "out_of_order_ticks_ignored": self._out_of_order_ticks,
            "ready_input_ticks": self._ready_ticks,
            "unavailable_input_ticks": self.ticks - self._ready_ticks,
            "tracking_rmse_deg": math.sqrt(self._tracking_mean_squared_error)
            if self._tracking_count else None,
            "tracking_samples": self._tracking_count,
            "max_abs_torque": self._max_abs_torque,
            "latency": {key: value.summary() for key, value in self._histograms.items()},
            "percentile_method": "fixed logarithmic histogram; approximate upper bounds (4% bins above 0.01 ms)",
            "latency_sampling": "observation age sampled each control tick; inference deduplicated by prediction_id when provided",
            "inference_ids_available": self._inference_deduplicated,
            "routes": {
                "source": "manual operator markers, not inferred ground truth",
                "attempts": self._route_attempts,
                "completed": self._route_completed,
                "aborted": self._route_aborted,
                "active": self._route_start_ns is not None,
                "active_seconds": (self._end_ns - self._route_start_ns) / 1e9
                if self._route_start_ns is not None else None,
                "last_duration_seconds": self._last_route_seconds,
                "completed_duration": self._route_duration.summary(),
                "ignored_markers": self._ignored_route_markers,
            },
        }


def load_report(path):
    """Read a standalone metrics summary or the runtime's nested run report."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("run report must be a JSON object")
    metrics = report.get("metrics", report)
    if not isinstance(metrics, dict) or "mode_seconds" not in metrics or "latency" not in metrics:
        raise ValueError("report has no supported metrics summary")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare recorded run summaries; route outcomes are manual markers.")
    parser.add_argument("command", choices=("report", "reports", "compare"))
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="emit structured summaries")
    args = parser.parse_args(argv)
    try:
        reports = [{"file": str(path), "metrics": load_report(path)} for path in args.reports]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(reports, indent=2, allow_nan=False))
    else:
        print("Run                          Time(s)  Assist(s)  Human  RMSE(deg)  Age p95(ms)  Routes")
        for report in reports:
            value = report["metrics"]
            rmse = value.get("tracking_rmse_deg")
            age = value["latency"].get("observation_age", {}).get("p95_ms")
            route = value.get("routes", {})
            rmse_text = "--" if rmse is None else f"{rmse:.2f}"
            age_text = "--" if age is None else f"{age:.1f}"
            print(f"{Path(report['file']).name[:28]:28} {value.get('duration_seconds', 0):7.1f} "
                  f"{value['mode_seconds'].get('assist', 0):10.1f} {value.get('human_interventions', 0):6} "
                  f"{rmse_text:>10} {age_text:>12}  {route.get('completed', 0)}/{route.get('attempts', 0)}")
        print("RMSE uses assisted control ticks only. Latency percentiles are approximate; routes are manual markers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
