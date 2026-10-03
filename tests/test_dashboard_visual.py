"""Dashboard presentation contracts with real status payloads and pure client logic.

No browser dependency is needed here; the coordinator also verifies real browser
layout. Node exercises the same inline view/plot functions served to operators.
"""
import json
import queue
import re
import shutil
import subprocess
import time

import pytest

from forza_ai.dashboard import Dashboard, _PAGE, _safe
from forza_ai.metrics import RunMetrics


def client(expression):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to exercise dashboard client behavior")
    script = re.search(r"<script>(.*?)</script>", _PAGE, re.S)[1]
    runner = "const fs=require('fs'),vm=require('vm');const p=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(vm.runInNewContext(p.script+'\\n;JSON.stringify('+p.expression+')',{}));"
    result = subprocess.run([node, "-e", runner], input=json.dumps({"script": script, "expression": expression}),
                            capture_output=True, text=True, check=True, timeout=5)
    return json.loads(result.stdout)


def view(status=None, *, stale=False, readonly=False, offline=False):
    payload = dict(status=status or {}, stale=stale, controls=dict(allow_arm=True, read_only=readonly))
    return client(f"viewState({json.dumps(payload)}, {str(offline).lower()})")


def test_nested_metrics_rates_and_limits_survive_without_secrets():
    metrics = RunMetrics()
    for index in range(3):
        metrics.update(dict(timestamp_ns=1_000_000_000 + index * 10_000_000, mode="assist",
                            actual_angle_deg=2, target_angle_deg=4, observation_age_ms=20,
                            inference_ms=7, prediction_id=index, input_status="ready", torque=.05))
    summary = metrics.summary()
    summary["latency"]["observation_age"]["access_token"] = "never expose"
    dashboard = Dashboard(queue.Queue())
    dashboard.publish(dict(mode="assist", metrics=summary,
                           rates=dict(capture_fps=None, policy_fps=29.3, control_hz=99.7, secret="hidden"),
                           limits=dict(observation_age_ms=250, target_angle_deg=90, torque=.15),
                           policy_name="Remote policy", expert_recording=False))
    result = dashboard.snapshot()["status"]
    assert result["metrics"]["latency"]["observation_age"]["count"] == 3
    assert result["metrics"]["latency"]["observation_age"]["p95_ms"] > 0
    assert result["metrics"]["routes"]["completed_duration"]["count"] == 0
    assert result["metrics"]["tracking_rmse_deg"] == 2
    assert result["rates"] == dict(capture_fps=None, policy_fps=29.3, control_hz=99.7)
    assert result["limits"]["observation_age_ms"] == 250
    assert "never expose" not in json.dumps(result)
    assert "hidden" not in json.dumps(result)


def test_nested_normalization_is_bounded_and_finite():
    tree = {str(i): {str(j): {str(k): {str(n): "x" * 1024 for n in range(8)}
                                     for k in range(8)} for j in range(8)} for i in range(2)}
    result = _safe(tree)
    # A single field has a shared 256-node budget, not an exponential depth cap.
    assert len(json.dumps(result)) < 75_000
    assert _safe({"a": float("nan"), "b": float("inf"), "values": [1, 2]}) == {"a": None, "b": None, "values": None}
    assert _safe({"a": {"b": {"c": {"d": {"password": "hidden"}}}}}) == {"a": {"b": {"c": {"d": None}}}}


@pytest.mark.parametrize("status,stale,readonly,allowed", [
    ({"mode": "manual"}, False, False, True),
    ({"mode": "takeover"}, False, False, True),
    ({"mode": "fault"}, False, False, True),
    ({"mode": "assist"}, False, False, False),
    ({"mode": "manual", "shadow": True}, False, False, False),
    ({"mode": "manual"}, True, False, False),
    ({"mode": "manual"}, False, True, False),
    ({}, False, False, False),
])
def test_server_engagement_availability_matches_operator_state(status, stale, readonly, allowed):
    dashboard = Dashboard(queue.Queue(), read_only=readonly)
    dashboard.publish(status)
    if stale:
        dashboard._published_ns = time.monotonic_ns() - 1_000_000_000
    assert dashboard.snapshot()["controls"]["allow_arm"] is allowed


@pytest.mark.parametrize("status,options,kind,arm", [
    ({}, {}, "waiting", False),
    ({"mode": "manual"}, {}, "manual", True),
    ({"mode": "assist"}, {}, "assist", False),
    ({"mode": "fault", "reason": "stale_wheel"}, {}, "fault", True),
    ({"mode": "manual", "shadow": True}, {}, "shadow", False),
    ({"mode": "assist"}, {"stale": True}, "stale", False),
    ({"mode": "assist"}, {"offline": True}, "offline", False),
    ({"mode": "manual"}, {"readonly": True}, "manual", False),
])
def test_client_state_does_not_declare_health_from_connection_alone(status, options, kind, arm):
    result = view(status, **options)
    assert result["kind"] == kind
    assert result["arm"] is arm
    if kind in ("offline", "stale"):
        assert "unknown" in (result["title"] + result["detail"]).lower()
    if kind == "fault":
        assert "fault" in result["title"].lower()


def test_route_controls_require_a_known_fresh_marker_state():
    idle = view({"mode": "manual", "route_active": False})
    active = view({"mode": "assist", "route_active": True})
    assert idle["startRoute"] and not idle["finishRoute"]
    assert active["finishRoute"] and not active["startRoute"]
    for options in ({"stale": True}, {"offline": True}, {"readonly": True}):
        result = view({"mode": "manual", "route_active": True}, **options)
        assert not result["finishRoute"] and not result["startRoute"]
    # Keep a manual takeover request available while status is stale but HTTP works.
    assert view({"mode": "assist"}, stale=True)["takeover"]
    assert not view({"mode": "assist"}, offline=True)["takeover"]


def test_empty_chart_and_null_values_never_become_zero_measurements():
    result = client("chartData([], ['actual_angle_deg'])")
    assert result["paths"] == [""] and result["hasData"] is False
    rows = [{"timestamp_ns": 1_000_000_000, "actual_angle_deg": None},
            {"timestamp_ns": 1_100_000_000, "actual_angle_deg": None}]
    result = client(f"chartData({json.dumps(rows)}, ['actual_angle_deg'])")
    assert result["paths"] == [""] and result["hasData"] is False
    rows[0]["actual_angle_deg"] = 0
    rows[1]["actual_angle_deg"] = 0
    result = client(f"chartData({json.dumps(rows)}, ['actual_angle_deg'])")
    assert result["hasData"] is True and " L" in result["paths"][0]


def test_chart_uses_timestamps_and_breaks_across_capture_gaps():
    rows = [{"timestamp_ns": 1_000_000_000, "actual_angle_deg": 1},
            {"timestamp_ns": 1_100_000_000, "actual_angle_deg": 2},
            {"timestamp_ns": 2_000_000_000, "actual_angle_deg": 3}]
    result = client(f"chartData({json.dumps(rows)}, ['actual_angle_deg'])")
    points = re.findall(r"[ML]([\d.]+),([\d.]+)", result["paths"][0])
    deltas = [float(points[i + 1][0]) - float(points[i][0]) for i in range(2)]
    assert deltas[1] / deltas[0] == pytest.approx(9, abs=.1)
    assert result["paths"][0].count(" M") == 2
    assert result["hasData"] is True


def test_plot_history_contains_real_source_age_and_torque():
    dashboard = Dashboard(queue.Queue())
    dashboard.publish(dict(timestamp_ns=42, actual_angle_deg=-3, torque=-.08,
                           observation_age_ms=14, inference_ms=9))
    sample = dashboard.snapshot()["history"][0]
    assert sample["torque"] == -.08 and sample["observation_age_ms"] == 14
    assert sample["inference_ms"] == 9
    assert sample["predicted_angle_deg"] is None
