"""Live game frame for the dashboard HUD: /api/frame.jpg serves the latest captured frame."""

import http.client
import queue

import numpy as np

from forza_ai.contracts import CapturedFrame
from forza_ai.dashboard import Dashboard


def get(dashboard, path):
    host, port = dashboard.address
    conn = http.client.HTTPConnection(host, port, timeout=2)
    conn.request("GET", path, headers={"Host": f"{host}:{port}"})
    response = conn.getresponse()
    return response.status, response.getheader("Content-Type"), response.read()


def test_frame_endpoint_serves_latest_jpeg_and_rate_limits():
    dashboard = Dashboard(queue.Queue(), host="127.0.0.1", port=0)
    dashboard.start()
    try:
        assert get(dashboard, "/api/frame.jpg")[0] == 204          # nothing captured yet
        rgb = np.zeros((66, 320, 3), np.uint8); rgb[:, :160, 0] = 255
        assert dashboard.publish_frame(CapturedFrame(1, 0, rgb), min_interval_ns=0)
        assert not dashboard.publish_frame(CapturedFrame(1, 0, rgb), min_interval_ns=0)   # same frame id
        status, kind, body = get(dashboard, "/api/frame.jpg?t=123")
        assert status == 200 and kind == "image/jpeg" and body[:2] == b"\xff\xd8"
        assert not dashboard.publish_frame(CapturedFrame(2, 0, rgb))                       # within 80 ms
        page = get(dashboard, "/")[2].decode()
        assert 'id="liveFrame"' in page and "pollFrame" in page
    finally:
        dashboard.close()
