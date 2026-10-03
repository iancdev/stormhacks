import json
import math

import pytest

from forza_ai.metrics import RunMetrics, load_report, main


def status(seconds, mode="assist", **fields):
    return dict(timestamp_ns=int(seconds * 1e9), mode=mode, actual_angle_deg=3,
                target_angle_deg=5, torque=0.1, input_status="ready", **fields)


def test_assisted_error_and_durations_exclude_manual_and_fault():
    metrics = RunMetrics()
    metrics.update(status(0, "manual"))
    metrics.update(status(1))
    metrics.update(dict(status(3), actual_angle_deg=1))
    metrics.update(status(4, "takeover", reason="command_expired"))
    metrics.update(status(5, "fault"))
    metrics.update(status(6, "fault"))
    result = metrics.summary()
    assert result["duration_seconds"] == 6
    assert result["mode_seconds"] == dict(manual=1, assist=3, takeover=1, fault=1, unknown=0)
    assert result["tracking_rmse_deg"] == pytest.approx(math.sqrt(10))
    assert result["tracking_samples"] == 2
    assert result["human_interventions"] == 0
    assert result["fault_entries"] == 1
    metrics.event("human_takeover", 6_000_000_000)
    assert metrics.summary()["human_interventions"] == 1
    assert metrics.summary()["interventions_per_assist_minute"] == 20


def test_route_outcomes_require_explicit_well_formed_markers():
    metrics = RunMetrics()
    metrics.event("route_complete", 0)
    metrics.event("route_start", 1_000_000_000)
    metrics.event("route_start", 2_000_000_000)
    metrics.update(status(3))
    assert metrics.summary()["routes"]["active_seconds"] == 2
    metrics.event("route_complete", 5_000_000_000)
    metrics.event("route_start", 6_000_000_000)
    metrics.event("route_abort", 7_000_000_000)
    route = metrics.summary()["routes"]
    assert route["attempts"] == 2
    assert route["completed"] == 1
    assert route["aborted"] == 1
    assert route["ignored_markers"] == 2
    assert route["completed_duration"]["mean_ms"] == 4000
    assert route["last_duration_seconds"] == 1
    assert route["active"] is False
    assert "manual" in route["source"]
    with pytest.raises(ValueError, match="unsupported"):
        metrics.event("guessed_completion", 8_000_000_000)


def test_latency_percentiles_inference_deduplication_and_invalid_samples():
    metrics = RunMetrics()
    for i in range(1, 101):
        metrics.update(status(i / 100, observation_age_ms=i, inference_ms=8, prediction_id=i // 10))
    result = metrics.summary()
    age = result["latency"]["observation_age"]
    assert age["count"] == 100
    assert age["mean_ms"] == 50.5
    assert 50 <= age["p50_ms"] <= 52
    assert 95 <= age["p95_ms"] <= 98.8
    assert 99 <= age["p99_ms"] <= 100
    assert result["latency"]["inference"]["count"] == 11
    assert result["latency"]["control_tick_gap"]["mean_ms"] == 10
    metrics.update(dict(status(2), actual_angle_deg=float("nan"), observation_age_ms=-1,
                        inference_ms=float("inf"), prediction_id=100))
    assert metrics.summary()["latency"]["observation_age"]["count"] == 100
    json.dumps(metrics.summary(), allow_nan=False)


def test_storage_stays_bounded_and_counts_remain_exact():
    metrics = RunMetrics()
    lengths = {name: len(hist.bins) for name, hist in metrics._histograms.items()}
    for i in range(30_000):
        metrics.update(status(i / 100, observation_age_ms=i % 123, reason=f"reason-{i}"))
    assert metrics.summary()["ticks"] == 30_000
    assert {name: len(hist.bins) for name, hist in metrics._histograms.items()} == lengths
    assert all(not isinstance(value, list) for value in vars(metrics).values())
    assert len(json.dumps(metrics.summary())) < 5000


def test_out_of_order_ticks_cannot_corrupt_timing():
    metrics = RunMetrics()
    metrics.update(status(2, "manual"))
    metrics.update(status(1, "assist"))
    metrics.update(status(3, "manual"))
    assert metrics.summary()["out_of_order_ticks_ignored"] == 1
    assert metrics.summary()["ticks"] == 2
    assert metrics.summary()["mode_seconds"]["manual"] == 1
    with pytest.raises(ValueError, match="timestamp"):
        metrics.update(dict(status(4), timestamp_ns=float("nan")))


def test_empty_summary_is_json_safe_and_has_no_fabricated_scores():
    result = RunMetrics().summary()
    assert result["tracking_rmse_deg"] is None
    assert result["interventions_per_assist_minute"] is None
    assert result["latency"]["observation_age"]["p95_ms"] is None
    json.dumps(result, allow_nan=False)


def test_report_cli_accepts_standalone_and_nested_summaries(tmp_path, capsys):
    metrics = RunMetrics()
    metrics.update(status(0))
    metrics.update(status(2, observation_age_ms=10))
    flat, nested = tmp_path / "flat.json", tmp_path / "nested.json"
    flat.write_text(json.dumps(metrics.summary()))
    nested.write_text(json.dumps({"hardware_verified": False, "metrics": metrics.summary()}))
    assert load_report(flat) == load_report(nested)
    assert main(["report", str(flat), str(nested)]) == 0
    output = capsys.readouterr().out
    assert "flat.json" in output and "nested.json" in output
    assert "manual markers" in output
    assert main(["compare", str(nested), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["metrics"]["ticks"] == 2
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    with pytest.raises(SystemExit):
        main(["report", str(bad)])
