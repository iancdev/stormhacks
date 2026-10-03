import json
import queue
import re
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from forza_ai.dashboard import Dashboard


@pytest.fixture
def dashboard():
    instance = Dashboard(queue.Queue(maxsize=4), port=0)
    instance.start()
    try:
        yield instance
    finally:
        instance.close()


def read(instance, path="/api/status", headers=None):
    return urlopen(Request(instance.url + path, headers=headers or {}), timeout=2)


def token(instance):
    html = read(instance, "/").read().decode()
    return re.search(r"const token='([^']+)'", html)[1]


def post(instance, event="manual", **overrides):
    headers = {"Origin": instance.url, "Content-Type": "application/json", "X-Forza-Token": token(instance)}
    headers.update(overrides)
    return urlopen(Request(instance.url + "/api/events", data=json.dumps({"event": event}).encode(),
                           headers=headers, method="POST"), timeout=2)


def test_http_snapshot_page_history_and_sensitive_fields(dashboard):
    dashboard.publish(dict(timestamp_ns=1, mode="assist", actual_angle_deg=7, target_angle_deg=8,
                           observation_age_ms=float("nan"), secret="must not leak",
                           recording={"frames": 5, "dropped": 1, "access_token": "hidden"}))
    result = json.loads(read(dashboard).read())
    assert result["status"]["actual_angle_deg"] == 7
    assert result["status"]["recording"] == {"frames": 5, "dropped": 1}
    assert "secret" not in result["status"]
    assert result["status"]["observation_age_ms"] is None
    assert result["stale"] is False
    assert result["history"][0]["target_angle_deg"] == 8
    html = read(dashboard, "/").read().decode()
    assert "Physical wheel" in html and "STALE" in html and "Observation age" in html
    assert "__TOKEN__" not in html
    with pytest.raises(HTTPError) as error:
        read(dashboard, "/../../etc/passwd")
    assert error.value.code == 404


def test_stale_status_disables_arm_and_stays_visible(dashboard):
    assert json.loads(read(dashboard).read())["stale"] is True
    dashboard.publish({"actual_angle_deg": 12})
    dashboard._published_ns = time.monotonic_ns() - 1_000_000_000
    value = json.loads(read(dashboard).read())
    assert value["stale"] is True
    assert value["status"]["actual_angle_deg"] == 12
    assert value["controls"]["allow_arm"] is False
    with pytest.raises(HTTPError) as error:
        post(dashboard, "arm")
    assert error.value.code == 409


def test_valid_events_only_queue_and_never_execute_hardware(dashboard):
    dashboard.publish({"mode": "manual"})
    for event in ("arm", "manual", "route_start", "route_complete"):
        response = post(dashboard, event)
        assert response.status == 202
        assert json.loads(response.read())["queued"] == event
    assert [dashboard._queue.get_nowait() for _ in range(4)] == ["arm", "manual", "route_start", "route_complete"]
    assert dashboard._queue.empty()
    with pytest.raises(HTTPError) as error:
        post(dashboard, "run_shell")
    assert error.value.code == 400
    assert dashboard._queue.empty()


@pytest.mark.parametrize("headers", [
    {"Origin": "http://evil.example"}, {"Origin": "null"}, {"Host": "evil.example"},
    {"X-Forza-Token": "wrong"}, {"Sec-Fetch-Site": "cross-site"},
])
def test_cross_origin_csrf_and_host_are_rejected(dashboard, headers):
    with pytest.raises(HTTPError) as error:
        post(dashboard, **headers)
    assert error.value.code == 403
    assert dashboard._queue.empty()


def test_plain_form_post_and_missing_origin_are_rejected(dashboard):
    with pytest.raises(HTTPError) as error:
        post(dashboard, **{"Content-Type": "text/plain"})
    assert error.value.code == 415
    with pytest.raises(HTTPError) as error:
        urlopen(Request(dashboard.url + "/api/events", data=b'{"event":"manual"}',
                        headers={"Content-Type": "application/json", "X-Forza-Token": token(dashboard)}), timeout=2)
    assert error.value.code == 403


def test_shadow_and_read_only_disable_arm(dashboard):
    dashboard.publish({"shadow": True})
    assert dashboard.snapshot()["controls"]["allow_arm"] is False
    with pytest.raises(HTTPError) as error:
        post(dashboard, "arm")
    assert error.value.code == 409
    dashboard._read_only = True
    with pytest.raises(HTTPError) as error:
        post(dashboard, "manual")
    assert error.value.code == 403


def test_queue_full_reports_failure(dashboard):
    for _ in range(4):
        dashboard._queue.put("existing")
    with pytest.raises(HTTPError) as error:
        post(dashboard)
    assert error.value.code == 503


def test_publish_never_waits_for_snapshot_lock(dashboard):
    with dashboard._lock:
        before = time.monotonic()
        assert dashboard.publish({"mode": "assist"}) is False
        assert time.monotonic() - before < 0.1
    for i in range(1000):
        dashboard._last_plot_ns = 0
        dashboard.publish({"timestamp_ns": i, "actual_angle_deg": i})
    assert len(dashboard.snapshot()["history"]) == 180


def test_loopback_only_and_finite_idempotent_cleanup():
    for host in ("0.0.0.0", "192.168.1.3", "::", "example.com"):
        with pytest.raises(ValueError, match="loopback"):
            Dashboard(queue.Queue(), host=host)
    instance = Dashboard(queue.Queue(), host="localhost", port=0)
    instance.start()
    instance.start()
    assert instance.url.startswith("http://127.0.0.1:")
    before = time.monotonic()
    instance.close()
    instance.close()
    assert time.monotonic() - before < 1
    assert not instance._thread.is_alive()
    assert instance.publish({}) is False
    with pytest.raises(RuntimeError, match="restarted"):
        instance.start()
