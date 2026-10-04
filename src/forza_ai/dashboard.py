"""Loopback-only status dashboard. HTTP handlers enqueue; they never drive hardware."""

from __future__ import annotations

from collections import deque
import ipaddress
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import queue
import secrets
import socket
import threading
import time


_EVENTS = frozenset(("manual", "arm", "route_start", "route_complete", "route_abort"))
_FIELDS = frozenset((
    "timestamp_ns", "ticks", "mode", "reason", "actual_angle_deg", "target_angle_deg",
    "predicted_angle_deg", "torque", "max_abs_torque", "speed_mps", "observation_age_ms",
    "input_status", "inference_ms", "prediction_id", "shadow", "allow_arm", "recording",
    "network", "transport", "route_active", "human_interventions", "error", "closed", "hardware_mode",
    "metrics", "rates", "policy_name", "expert_recording", "target_limit_deg", "torque_limit",
    "max_observation_age_ms", "target_rate_deg_s", "limits",
    "auto_pedals", "predicted_throttle", "predicted_brake", "output_throttle", "output_brake",
))
_SENSITIVE = ("token", "secret", "password", "authorization", "credential", "key")


def _safe(value, depth=0, budget=None):
    """Bounded JSON normalization for diagnostic values; omit credential fields."""
    if budget is None:
        budget = [256]
    budget[0] -= 1
    if budget[0] < 0:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, int):
        return value if abs(value) <= 2 ** 64 - 1 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict) and depth < 4:
        output = {}
        for index, (key, entry) in enumerate(value.items()):
            if index >= 32 or budget[0] <= 0:
                break
            if isinstance(key, str) and not any(part in key.lower() for part in _SENSITIVE):
                output[key[:64]] = _safe(entry, depth + 1, budget)
        return output
    return None


class Dashboard:
    """Publish best-effort snapshots without blocking the control loop.

    ``publish`` returns False when the short snapshot lock is busy; skipping a UI
    update must never delay steering. All queued events are plain strings. The
    runtime remains responsible for engagement rules, including shadow mode.
    """

    def __init__(self, command_queue, host="127.0.0.1", port=8766, *, read_only=False,
                 stale_after_seconds=0.5):
        host = "127.0.0.1" if host == "localhost" else host
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("dashboard host must be a literal loopback address") from exc
        if not address.is_loopback:
            raise ValueError("dashboard is loopback-only; do not expose wheel controls on the LAN")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("invalid dashboard port")
        if not math.isfinite(stale_after_seconds) or stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be finite and positive")
        self._queue = command_queue
        self._host, self._port = str(address), port
        self._read_only = read_only
        self._stale_after_ns = int(stale_after_seconds * 1e9)
        self._token = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._snapshot = {}
        self._published_ns = None
        self._history = deque(maxlen=180)
        self._last_plot_ns = 0
        self._server = None
        self._thread = None
        self._closed = False

    @property
    def address(self):
        if self._server is None:
            raise RuntimeError("start the dashboard before reading its address")
        return self._server.server_address[:2]

    @property
    def url(self):
        host, port = self.address
        authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        return f"http://{authority}"

    def start(self):
        if self._closed:
            raise RuntimeError("a closed dashboard cannot be restarted")
        if self._server is not None:
            return
        owner = self

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False
            address_family = socket.AF_INET6 if ":" in owner._host else socket.AF_INET

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(1.0)

            def log_message(self, *args):
                pass

            def respond(self, code, body, content_type="application/json; charset=utf-8"):
                encoded = body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; "
                                 "style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def reject(self, code, message):
                self.respond(code, json.dumps({"error": message}))

            def valid_host(self):
                # Exact authority prevents DNS-rebinding requests even on loopback.
                return self.headers.get_all("Host") == [owner.url.removeprefix("http://")]

            def do_GET(self):
                if not self.valid_host():
                    return self.reject(403, "unexpected host")
                if self.path == "/":
                    return self.respond(200, _PAGE.replace("__TOKEN__", owner._token), "text/html; charset=utf-8")
                if self.path == "/api/status":
                    return self.respond(200, json.dumps(owner.snapshot(), allow_nan=False))
                self.reject(404, "not found")

            def do_POST(self):
                if self.path != "/api/events":
                    return self.reject(404, "not found")
                if (not self.valid_host() or self.headers.get_all("Origin") != [owner.url]
                        or self.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin"):
                    return self.reject(403, "same-origin requests required")
                if owner._read_only:
                    return self.reject(403, "dashboard is read-only")
                token = self.headers.get("X-Forza-Token", "")
                if not secrets.compare_digest(token.encode("utf-8"), owner._token.encode("ascii")):
                    return self.reject(403, "invalid dashboard token")
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                    return self.reject(415, "application/json required")
                if self.headers.get("Transfer-Encoding") is not None:
                    return self.reject(400, "transfer encoding is unsupported")
                try:
                    if len(self.headers.get_all("Content-Length", [])) != 1:
                        raise ValueError("missing or duplicate content length")
                    length = int(self.headers["Content-Length"])
                    if not 0 < length <= 512:
                        raise ValueError("event body must be 1-512 bytes")
                    payload = json.loads(self.rfile.read(length))
                    event = payload.get("event") if isinstance(payload, dict) else None
                    if not isinstance(event, str) or event not in _EVENTS:
                        raise ValueError("unsupported event")
                    if set(payload) != {"event"}:
                        raise ValueError("unexpected event fields")
                except (ValueError, UnicodeError, TimeoutError):
                    return self.reject(400, "invalid event body")
                snapshot = owner.snapshot()
                if event == "arm" and not snapshot["controls"]["allow_arm"]:
                    return self.reject(409, "arming is unavailable or status is stale")
                try:
                    owner._queue.put_nowait(event)
                except queue.Full:
                    return self.reject(503, "command queue is full")
                self.respond(202, json.dumps({"queued": event}))

        server = Server((self._host, self._port), Handler)
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                                        name="wheel-dashboard", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            server.server_close()
            self._server = None
            raise

    def publish(self, status):
        if self._closed:
            return False
        clean = {key: _safe(status[key]) for key in _FIELDS if key in status}
        now = time.monotonic_ns()
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self._snapshot = clean
            self._published_ns = now
            if now - self._last_plot_ns >= 100_000_000:
                self._history.append({key: clean.get(key) for key in
                                      ("timestamp_ns", "actual_angle_deg", "target_angle_deg", "predicted_angle_deg",
                                       "observation_age_ms", "torque", "inference_ms")})
                self._last_plot_ns = now
        finally:
            self._lock.release()
        return True

    def snapshot(self):
        with self._lock:
            status = dict(self._snapshot)
            history = list(self._history)
            published = self._published_ns
        age = None if published is None else max(0, time.monotonic_ns() - published)
        stale = age is None or age > self._stale_after_ns or self._closed or bool(status.get("closed"))
        return {"status": status, "snapshot_age_ms": age / 1e6 if age is not None else None,
                "stale": stale, "history": history,
                "controls": {"read_only": self._read_only,
                             "allow_arm": not stale and not self._read_only and not status.get("shadow", False)
                             and status.get("mode") in ("manual", "takeover", "fault")
                             and status.get("allow_arm", True)}}

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stormhacks / Steering lab</title><style>
:root{color-scheme:dark;--bg:#0b1014;--panel:#12191e;--raised:#182128;--line:#263239;--ink:#edf3f3;--muted:#899bA4;--mint:#96e4c8;--amber:#eec284;--blue:#a3bce8;--red:#f1a3a0;font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-synthesis:none;background:var(--bg);color:var(--ink)}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 65% -40%,#223a3b66,transparent 60%);min-height:100vh}button{font:inherit}button:focus-visible{outline:2px solid var(--mint);outline-offset:4px}button:disabled{opacity:.35;cursor:not-allowed!important}button:not(:disabled):hover{filter:brightness(1.13)}.shell{max-width:1440px;margin:auto;padding:0 38px 34px}.topbar{position:sticky;top:0;z-index:10;background:#0b1014f5;backdrop-filter:blur(12px);display:flex;align-items:center;justify-content:space-between;gap:20px;min-height:85px;border-bottom:1px solid var(--line)}.brand{display:flex;align-items:center;gap:12px;font-size:14px;font-weight:660;letter-spacing:.02em}.brand-mark{width:28px;height:28px;color:var(--mint)}.brand-divider{height:16px;width:1px;background:#3b474f;margin:0 4px}.brand-sub{color:var(--muted);font-size:12px;font-weight:450}.top-meta{display:flex;align-items:center;gap:22px;font:11px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted)}.quick-action{position:relative}.quick-disengage{border:1px solid #645644;border-radius:5px;background:#29251e;color:#f0c790;padding:10px 13px;font-size:11px;white-space:nowrap;cursor:pointer}.quick-feedback{position:absolute;right:0;top:calc(100% + 8px);min-width:150px;max-width:270px;padding:7px 10px;background:#1b272d;border:1px solid #3b4c52;border-radius:4px;color:#d0dedc;font:10px/1.5 system-ui,sans-serif;box-shadow:0 8px 20px #0004}.quick-feedback[hidden]{display:none}.top-meta strong{color:#c8d4d9;font-weight:500}.connection{display:flex;align-items:center;gap:7px}.dot{display:inline-block;width:6px;height:6px;background:currentColor;border-radius:50%;flex-shrink:0}.connection[data-state="live"]{color:var(--mint)}.connection[data-state="stale"],.connection[data-state="offline"]{color:var(--amber)}
.hero{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;padding:30px 0 25px}.eyebrow,.section-kicker{font-size:10px;letter-spacing:.15em;font-weight:600;text-transform:uppercase;color:var(--muted)}h1{font-size:30px;font-weight:520;letter-spacing:-1.2px;margin:7px 0 7px;line-height:1.15}.subtitle{font-size:13px;line-height:1.55;color:var(--muted);margin:0}.badges{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:3px;justify-content:flex-end}.badge{font:10px ui-monospace,SFMono-Regular,Consolas,monospace;padding:7px 10px;border:1px solid var(--line);border-radius:5px;text-transform:uppercase;letter-spacing:.07em;color:#bdccd2;background:#121a20}.badge.simulation{color:var(--amber);border-color:#554734;background:#241f18}.badge.shadow{color:var(--blue);border-color:#37485f}.state-strip{padding:17px 20px;display:flex;align-items:center;gap:15px;background:#151e23;border:1px solid var(--line);border-left:3px solid var(--muted);border-radius:7px;margin-bottom:18px}.state-strip[data-state="assist"]{border-left-color:var(--mint);background:linear-gradient(100deg,#182923,#151e23 60%)}.state-strip[data-state="takeover"],.state-strip[data-state="stale"],.state-strip[data-state="offline"]{border-left-color:var(--amber)}.state-strip[data-state="fault"]{border-left-color:var(--red);background:linear-gradient(100deg,#2a1e21,#151e23 70%)}.state-strip[data-state="shadow"]{border-left-color:var(--blue)}.state-symbol{width:32px;height:32px;border:1px solid var(--line);border-radius:50%;display:grid;place-items:center;color:var(--muted);flex:none}.state-strip[data-state="assist"] .state-symbol{color:var(--mint)}.state-strip[data-state="fault"] .state-symbol{color:var(--red)}.state-copy{min-width:0;flex:1}.state-title{font-size:13px;font-weight:620;line-height:1.5}.state-detail{font-size:12px;color:#a1b0b7;line-height:1.5;margin-top:2px;overflow-wrap:anywhere}.state-code{font:10px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);text-align:right;line-height:1.7;max-width:240px;overflow-wrap:anywhere}.signal-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px}.signal{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:17px 20px 16px;min-width:0}.signal-label{font-size:11px;color:#a5b4bb;display:flex;align-items:center;justify-content:space-between;gap:8px}.signal-label .index{font:10px ui-monospace,SFMono-Regular,Consolas,monospace;color:#5e747e}.value-line{display:flex;align-items:baseline;gap:7px;margin-top:13px}.value{font:500 35px/1.1 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:-1.5px;font-variant-numeric:tabular-nums;white-space:nowrap}.unit{font-size:12px;color:var(--muted)}.signal-foot{font-size:10px;color:var(--muted);margin-top:11px;min-height:14px}.signal:first-child .value{color:var(--mint)}.signal:nth-child(2) .value{color:var(--blue)}
.workspace{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(300px,1fr);gap:18px;align-items:start}.left-stack,.right-stack{display:grid;gap:18px;min-width:0}.panel{border:1px solid var(--line);border-radius:8px;background:var(--panel);min-width:0;overflow:hidden}.panel-head{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;padding:20px 20px 0}.panel-title{font-size:13px;font-weight:600;margin:0;letter-spacing:-.15px}.panel-caption{font-size:10px;color:var(--muted);line-height:1.55;margin-top:5px}.micro-tag{font:9px ui-monospace,SFMono-Regular,Consolas,monospace;border:1px solid var(--line);border-radius:4px;color:var(--muted);padding:4px 6px;white-space:nowrap}.legend{display:flex;gap:15px;align-items:center;flex-wrap:wrap;padding:16px 20px 0}.legend button{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:10px;border:0;padding:0;background:none;cursor:pointer}.legend button[aria-pressed="false"]{opacity:.4}.swatch{width:13px;height:2px;background:var(--mint)}.target .swatch{background:var(--blue)}.predicted .swatch{background:var(--amber)}.plot-wrap{position:relative;margin:7px 10px 0}.chart{width:100%;height:264px;display:block;overflow:visible}.chart text{font:10px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#82959f}.chart .grid-line{stroke:#263238;stroke-width:1}.chart .zero-line{stroke:#45535b;stroke-dasharray:3 5}.chart path{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;vector-effect:non-scaling-stroke}.chart .trace-target{stroke:var(--blue);stroke-width:1.6}.chart .trace-actual{stroke:var(--mint);stroke-width:2.2}.chart .trace-predicted{stroke:var(--amber);stroke-width:1.4;stroke-dasharray:4 3}.empty-chart{position:absolute;inset:45px 20px 43px 49px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:7px;font-size:11px;color:var(--muted);background:#12191ec4;text-align:center;padding:12px;pointer-events:none}.empty-chart strong{font-size:12px;font-weight:500;color:#b3c2c9}.empty-chart[hidden]{display:none}.chart-footer{display:flex;justify-content:space-between;gap:10px;font-size:10px;color:var(--muted);padding:0 20px 18px}.tooltip{position:absolute;z-index:2;left:65px;top:7px;background:#1c292f;border:1px solid #46595f;border-radius:4px;font:10px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;padding:7px 10px;pointer-events:none;color:#c8d8dc;white-space:pre;max-width:calc(100% - 80px)}.tooltip[hidden]{display:none}.mini-plots{display:grid;grid-template-columns:1fr 1fr;gap:0}.mini-plot+.mini-plot{border-left:1px solid var(--line)}.mini-plot .panel-head{padding:17px 18px 0}.mini-plot .chart{height:132px}.mini-plot .plot-wrap{margin:5px 8px 3px}.mini-plot .empty-chart{inset:22px 10px 28px 43px;font-size:10px}.mini-plot .empty-chart strong{font-size:10px}.mini-plot .panel-title{font-size:11px}.mini-value{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;color:#d3dfdf}
.pipeline{padding:18px 20px 6px}.pipeline-row{display:grid;grid-template-columns:25px minmax(0,1fr) auto;gap:12px;align-items:center;padding:0 0 17px;position:relative}.pipeline-row:not(:last-child):before{content:"";position:absolute;left:12px;top:25px;bottom:0;width:1px;background:#314047}.step{display:grid;place-items:center;width:25px;height:25px;border:1px solid #3b4c52;border-radius:50%;font:10px ui-monospace,SFMono-Regular,Consolas,monospace;color:#94aab2;background:var(--panel);z-index:1}.step-label{font-size:11px;color:#c4d1d6}.step-description{font-size:10px;color:var(--muted);margin-top:4px;overflow-wrap:anywhere}.rate{font:500 16px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.rate small{font:9px system-ui,sans-serif;color:var(--muted);margin-left:3px}.divider{height:1px;background:var(--line);margin:0 20px}.rows{padding:9px 20px 15px}.row{display:flex;align-items:baseline;justify-content:space-between;gap:15px;font-size:10px;padding:7px 0;color:var(--muted)}.row span:last-child{color:#cad7dc;text-align:right;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.stat-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));padding:18px 20px 0;gap:12px}.stat-number{font:20px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:-.6px}.stat-label{font-size:9px;line-height:1.5;color:var(--muted);margin-top:5px}.record-note{font-size:10px;color:var(--muted);padding:15px 20px 17px;line-height:1.65;overflow-wrap:anywhere}.record-note[data-error="true"]{color:var(--red)}.record-pill{font-size:9px;color:var(--muted);display:flex;align-items:center;gap:6px;text-transform:uppercase;letter-spacing:.05em}.record-pill[data-state="recording"]{color:var(--mint)}.record-pill[data-state="error"]{color:var(--red)}
.controls-panel{margin-top:18px}.controls-body{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;padding:19px 20px 0}.controls-block{min-width:0;flex:1}.controls-label{font-size:10px;font-weight:600;color:#b3c4cb;margin-bottom:10px}.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.controls button{font-size:11px;border:1px solid #3a4a52;border-radius:5px;background:#1b272d;color:#cbd8dd;cursor:pointer;padding:11px 15px;line-height:1.1;transition:filter .15s}.controls button.primary{background:var(--mint);border-color:var(--mint);color:#142c23;font-weight:650}.controls button.stop{border-color:#645644;background:#29251e;color:#f0c790}.controls button.route{padding:11px 12px;background:transparent}.controls-note{font-size:10px;line-height:1.65;color:var(--muted);padding:15px 20px 0;max-width:1000px}.controls-note strong{color:#bdcdd3;font-weight:500}#message{font-size:11px;line-height:1.6;color:var(--mint);padding:8px 20px 16px;min-height:41px;margin:0}#message[data-error="true"]{color:var(--red)}.route-panel{margin-top:18px}.route-body{display:grid;grid-template-columns:1.25fr repeat(4,1fr);gap:20px;padding:19px 20px 20px}.route-body>div:not(:first-child){border-left:1px solid var(--line);padding-left:20px}.route-title{font-size:13px;color:#cdd9dc;line-height:1.5}.route-hint{font-size:10px;color:var(--muted);line-height:1.6;margin-top:5px}.footer{display:flex;justify-content:space-between;gap:14px;padding-top:22px;color:#697e88;font:9px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}.footer span:last-child{text-align:right}
@media(min-width:1550px){.shell{padding-top:8px}.chart{height:290px}}@media(max-width:1050px){.shell{padding:0 24px 28px}.workspace{grid-template-columns:minmax(0,1.6fr) minmax(280px,1fr);gap:14px}.signal{padding:16px}.value{font-size:30px}.controls-body{flex-direction:column}.controls-block{width:100%}.route-body{gap:14px}.route-body>div:not(:first-child){padding-left:14px}.top-meta{gap:12px}.brand-sub{display:none}}@media(max-width:820px){.workspace{grid-template-columns:1fr}.right-stack{grid-template-columns:1fr 1fr;align-items:start}.signal-grid{gap:9px}.signal{padding:14px}.value{font-size:25px}.signal-label{font-size:10px}.signal-label .index{display:none}.signal-foot{font-size:9px}.state-code{max-width:170px}.route-body{grid-template-columns:repeat(4,1fr)}.route-body>div:first-child{grid-column:1/-1}.route-body>div:nth-child(2){border:0;padding:0}.hero{padding-top:25px}.chart{height:250px}}@media(max-width:560px){.shell{padding:0 16px 24px}.topbar{min-height:65px;gap:12px}.brand{font-size:12px;gap:8px}.brand-mark{width:22px;height:22px}.brand-divider,.top-meta .elapsed-wrap{display:none}.top-meta{font-size:9px;gap:9px}.top-meta #connectionLabel{display:none}.quick-disengage{font-size:10px;padding:10px 11px}.hero{display:block;padding:24px 0 19px}h1{font-size:28px}.badges{justify-content:flex-start;margin-top:16px}.subtitle{font-size:11px}.state-strip{padding:14px 13px;gap:10px;flex-wrap:wrap}.state-code{display:none}.state-title{font-size:12px}.state-detail{font-size:11px}.signal-grid{grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}.signal{padding:15px}.value{font-size:29px}.signal-foot{font-size:9px}.signal-label{font-size:10px}.right-stack{grid-template-columns:1fr}.left-stack,.right-stack{gap:14px}.panel-head{padding:17px 16px 0}.legend{gap:11px;padding:14px 16px 0}.legend button{font-size:9px}.chart{height:215px}.mini-plots{grid-template-columns:1fr}.mini-plot+.mini-plot{border-left:0;border-top:1px solid var(--line)}.mini-plot .chart{height:128px}.chart-footer{font-size:9px;padding:0 16px 16px}.controls-body{padding:17px 16px 0}.controls button{padding:12px 14px}.controls-note{padding:15px 16px 0}#message{padding-left:16px;padding-right:16px}.route-body{grid-template-columns:1fr 1fr;gap:20px}.route-body>div:nth-child(4){border:0;padding:0}.route-body>div:not(:first-child){padding-left:0;border-left:0}.route-body .stat-number{font-size:23px}.footer{flex-direction:column;gap:5px;font-size:8px}.footer span:last-child{text-align:left}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body><div class="shell">
<header class="topbar"><div class="brand"><svg class="brand-mark" viewBox="0 0 28 28" fill="none" aria-hidden="true"><path d="M4 20 10 7h5l-4 8h6L14 22M18 6l6 8-6 8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg><span>STORMHACKS</span><span class="brand-divider"></span><span class="brand-sub">STEERING LAB</span></div><div class="top-meta"><span class="elapsed-wrap">SESSION <strong id="elapsed">—</strong></span><span id="connection" class="connection" data-state="waiting"><i class="dot"></i><span id="connectionLabel">AWAITING RUNTIME</span></span><div class="quick-action"><button class="quick-disengage" data-event="manual" data-quick="true" title="Queue a request to release AI steering torque" disabled>Disengage AI</button><span id="quickFeedback" class="quick-feedback" role="status" hidden></span></div></div></header>
<section class="hero"><div><div class="eyebrow">Operator console / 01</div><h1>Steering, observed.</h1><p class="subtitle">Physical wheel control. Human pedals. Every signal in view.</p></div><div class="badges"><span id="hardwareBadge" class="badge">Backend unknown</span><span id="policyBadge" class="badge">Policy unavailable</span><span class="badge">Local access</span></div></section>
<section id="stateStrip" class="state-strip" data-state="waiting" aria-live="polite"><div class="state-symbol" aria-hidden="true"><svg width="17" height="17" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="1.2"/><path d="M4 8h12M10 9v8M7 5h6" stroke="currentColor" stroke-width="1.2"/></svg></div><div class="state-copy"><div id="health" class="state-title">Waiting for runtime</div><div id="stateDetail" class="state-detail">Live signals will appear after the first runtime snapshot.</div></div><div class="state-code"><span id="reason">No status received</span><br><span id="freshness">Snapshot —</span></div></section>
<section class="signal-grid" aria-label="Current signals"><article class="signal"><div class="signal-label">Physical wheel <span class="index">01 / ACTUAL</span></div><div class="value-line"><span id="actual" class="value">—</span><span class="unit">deg</span></div><div class="signal-foot">Measured angle · right positive</div></article><article class="signal"><div class="signal-label">Controller target <span class="index">02 / TARGET</span></div><div class="value-line"><span id="target" class="value">—</span><span class="unit">deg</span></div><div class="signal-foot" id="predictionCaption">Policy prediction —</div></article><article class="signal"><div class="signal-label">Game speed <span class="index">03 / VEHICLE</span></div><div class="value-line"><span id="speed" class="value">—</span><span class="unit">km/h</span></div><div class="signal-foot" id="speedCaption">Forza telemetry unavailable</div></article><article class="signal"><div class="signal-label">Observation age <span class="index">04 / SOURCE</span></div><div class="value-line"><span id="age" class="value">—</span><span class="unit">ms</span></div><div class="signal-foot">Capture to control · includes pipeline delay</div></article></section>
<main class="workspace"><div class="left-stack"><section class="panel"><div class="panel-head"><div><h2 class="panel-title">Steering trace</h2><div class="panel-caption">Requested, controlled, and physically measured angles</div></div><span class="micro-tag">ROLLING / 18 s</span></div><div class="legend" aria-label="Visible steering traces"><button class="actual" data-series="actual_angle_deg" aria-pressed="true"><i class="swatch"></i>Physical wheel</button><button class="target" data-series="target_angle_deg" aria-pressed="true"><i class="swatch"></i>Controller target</button><button class="predicted" data-series="predicted_angle_deg" aria-pressed="true"><i class="swatch"></i>Policy prediction</button></div><div class="plot-wrap"><svg id="steeringChart" class="chart" viewBox="0 0 720 260" preserveAspectRatio="none" role="img" aria-label="Recent physical wheel, target and prediction angles over real elapsed time"><g id="steeringGrid"></g><path id="actualLine" class="trace-actual"/><path id="targetLine" class="trace-target"/><path id="predictionLine" class="trace-predicted"/><line id="chartCursor" x1="0" x2="0" y1="20" y2="224" stroke="#5b707a" stroke-dasharray="3 4" visibility="hidden"/></svg><div id="steeringEmpty" class="empty-chart"><strong>Waiting for steering samples</strong><span>No trace is drawn until measured snapshots arrive.</span></div><div id="chartTooltip" class="tooltip" hidden></div></div><div class="chart-footer"><span id="plotInfo">0 samples · no recorded trace</span><span>Degrees · auto scale</span></div></section>
<section class="panel mini-plots" aria-label="Timing and motor traces"><div class="mini-plot"><div class="panel-head"><h2 class="panel-title">Source age</h2><span id="sourceMiniValue" class="mini-value">— ms</span></div><div class="plot-wrap"><svg id="ageChart" class="chart" viewBox="0 0 350 130" preserveAspectRatio="none" role="img" aria-label="Recent observation age in milliseconds"><g id="ageGrid"></g><path id="ageLine" class="trace-predicted"/></svg><div id="ageEmpty" class="empty-chart"><strong>No observation timing yet</strong></div></div></div><div class="mini-plot"><div class="panel-head"><h2 class="panel-title">Applied torque</h2><span id="torque" class="mini-value">—</span></div><div class="plot-wrap"><svg id="torqueChart" class="chart" viewBox="0 0 350 130" preserveAspectRatio="none" role="img" aria-label="Recent normalized motor torque"><g id="torqueGrid"></g><path id="torqueLine" class="trace-actual"/></svg><div id="torqueEmpty" class="empty-chart"><strong>No motor output samples yet</strong></div></div></div></section></div>
<div class="right-stack"><section class="panel"><div class="panel-head"><div><h2 class="panel-title">Signal pipeline</h2><div class="panel-caption">Measured rates over the recent window</div></div><span id="transportTag" class="micro-tag">—</span></div><div class="pipeline"><div class="pipeline-row"><span class="step">1</span><div><div class="step-label">Road capture</div><div class="step-description" id="captureDescription">Fresh frames observed by runtime</div></div><span class="rate"><span id="captureRate">—</span><small>fps</small></span></div><div class="pipeline-row"><span class="step">2</span><div><div class="step-label">Policy inference</div><div class="step-description" id="policyName">Policy not reported</div></div><span class="rate"><span id="policyRate">—</span><small>fps</small></span></div><div class="pipeline-row"><span class="step">3</span><div><div class="step-label">Wheel control</div><div class="step-description" id="hardwareName">Backend not reported</div></div><span class="rate"><span id="controlRate">—</span><small>Hz</small></span></div></div><div class="divider"></div><div class="rows"><div class="row"><span id="inferenceLabel">Policy time</span><span id="inference">—</span></div><div class="row"><span>Source age / p95</span><span id="sourceP95">—</span></div><div class="row"><span>Control interval / p95</span><span id="controlP95">—</span></div><div class="row"><span>Source age limit</span><span id="sourceLimit">—</span></div><div class="row"><span>Target / torque bounds</span><span id="controlLimits">—</span></div><div class="row"><span>Input availability</span><span id="input">Not reported</span></div></div></section>
<section class="panel"><div class="panel-head"><div><h2 class="panel-title">Session recording</h2><div class="panel-caption" id="recordingCaption">Recorder status not reported</div></div><span id="recordingState" class="record-pill" data-state="unknown"><i class="dot"></i><span id="recordingLabel">Unknown</span></span></div><div class="stat-grid"><div><div class="stat-number" id="recordFrames">—</div><div class="stat-label">Frames saved</div></div><div><div class="stat-number" id="recordWheel">—</div><div class="stat-label">Wheel samples</div></div><div><div class="stat-number" id="recordDrops">—</div><div class="stat-label">Dropped samples</div></div></div><div class="rows"><div class="row"><span>Telemetry samples</span><span id="recordTelemetry">—</span></div><div class="row"><span>Writer queue</span><span id="recordQueue">—</span></div><div class="row"><span>Expert label eligibility</span><span id="expertLabel">Not reported</span></div></div><div class="divider"></div><div id="recordNote" class="record-note">Recording counters appear when a session recorder is attached.</div></section></div></main>
<section class="panel route-panel"><div class="panel-head"><div><h2 class="panel-title">Run evaluation</h2><div class="panel-caption">Route outcomes are manual operator markers, not automatic evaluation.</div></div><span id="mode" class="micro-tag">MODE / —</span></div><div class="route-body"><div><div id="route" class="route-title">Route status unavailable</div><div id="routeHint" class="route-hint">Start a marker to track a route attempt.</div></div><div><div id="routesCompleted" class="stat-number">—</div><div class="stat-label">Completed / attempts</div></div><div><div id="interventions" class="stat-number">—</div><div class="stat-label">Human takeovers from assistance</div></div><div><div id="trackingRmse" class="stat-number">—</div><div class="stat-label">Tracking RMSE · degrees</div></div><div><div id="faults" class="stat-number">—</div><div class="stat-label">Control fault entries</div></div></div></section>
<section class="panel controls-panel"><div class="controls-body"><div class="controls-block"><div class="controls-label">Steering authority</div><div class="controls"><button class="primary" id="arm" data-event="arm" disabled>Engage steering</button><button class="stop" id="takeover" data-event="manual" disabled>Take over</button></div></div><div class="controls-block"><div class="controls-label">Route markers</div><div class="controls"><button class="route" data-event="route_start" disabled>Start route</button><button class="route" data-event="route_complete" disabled>Mark complete</button><button class="route" data-event="route_abort" disabled>Abort route</button></div></div></div><div class="controls-note"><strong>Prefer the wheel's arm and takeover buttons while driving.</strong> Forza must stay in the foreground for live steering. Browser commands are queued requests; the runtime snapshot confirms the resulting state.</div><p id="message" role="status" aria-live="polite">Controls become available when runtime status arrives.</p></section>
<footer class="footer"><span>STORMHACKS / HUMAN PEDALS · AI STEERING</span><span id="footerStatus">LOCAL DASHBOARD · AWAITING SNAPSHOT</span></footer></div>
<script>
const token='__TOKEN__';
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const fmt=(v,n=1)=>finite(v)?v.toFixed(n):'—';
const count=v=>finite(v)?Math.round(v).toLocaleString('en-US'):'—';
const duration=v=>finite(v)?Math.floor(v/60).toString().padStart(2,'0')+':'+Math.floor(v%60).toString().padStart(2,'0'):'—';
const humanize=v=>typeof v==='string'?v.replaceAll('_',' '):'Not reported';
const reasons={inference_failure:'The inference link failed. Fresh replies cannot restart steering; explicitly re-engage after recovery.',actuation_deadline_expired:'Motor preparation exceeded its command deadline. Inspect native timing before re-engaging.',not_engaged:'Choose Engage steering or use the wheel arm button when ready.',engaged:'The controller is following policy targets. Human pedals remain in control.',manual_takeover:'AI torque is released. Explicitly re-engage when you are ready.',race_inactive:'Resume the game, then explicitly re-engage steering.',frame_race_inactive:'Waiting for an active-race image and telemetry before engagement.',game_not_foreground:'Return focus to Forza, then use the wheel arm button to re-engage.',telemetry_unavailable:'Check Forza Data Out and wait for fresh telemetry before re-engaging.',telemetry_unavailable_or_paused:'Resume Forza or restore telemetry, then explicitly re-engage.',frame_unavailable:'Check screen capture and game focus before re-engaging.',no_causal_telemetry:'No matching telemetry at capture time. Check the telemetry stream.',no_command:'No policy command is available. Check the inference connection before re-engaging.',command_expired:'The steering command expired. Restore fresh predictions before re-engaging.',stale_command:'The steering command is stale. Check the policy connection before re-engaging.',stale_observation:'The source image is too old. Check capture and inference latency.',stale_wheel:'The wheel input sample is too old. Check control-loop timing and the device connection before restarting.',wheel_disconnected:'The wheel disconnected. Check its connection before restarting.',control_loop_gap:'The control loop missed its timing limit. Inspect the run before re-engaging.'};
function viewState(data,offline=false){const s=data?.status||{},controls=data?.controls||{},waiting=!Object.keys(s).length;let kind=offline?'offline':waiting?'waiting':data.stale?'stale':s.mode==='fault'?'fault':s.shadow?'shadow':s.mode||'waiting';const titles={waiting:'Waiting for runtime',offline:'Connection lost · status unknown',stale:'STALE · last known values',assist:'Assisted steering engaged',manual:'Manual steering',takeover:'Assistance disengaged',fault:'Controller fault · AI torque released',shadow:'Shadow mode · motor control disabled'};let detail=offline?'The dashboard cannot reach the runtime. Displayed values are retained for diagnosis.':waiting?'Live signals will appear after the first runtime snapshot.':kind==='stale'?'Snapshots have stopped updating. Steering state is unknown; use the physical controls.':s.shadow?'Predictions are visible for evaluation. This run cannot engage the motor.':reasons[s.reason]||'Review the controller reason and input availability before changing steering authority.';const unavailable=offline||waiting||data.stale||controls.read_only;return{kind,title:titles[kind]||humanize(kind),detail,arm:!unavailable&&!s.shadow&&s.mode!=='assist'&&controls.allow_arm===true,takeover:!offline&&!waiting&&!controls.read_only,disengage:!offline&&!waiting&&!controls.read_only&&!s.shadow&&(s.mode==='assist'||data.stale),startRoute:!unavailable&&s.route_active===false,finishRoute:!unavailable&&s.route_active===true,readOnly:controls.read_only===true};}
function chartData(rows,keys,{signed=true,minScale=5,width=720,height=260}={}){const left=46,right=width-15,top=20,bottom=height-36;const valid=(Array.isArray(rows)?rows:[]).filter(r=>r&&finite(r.timestamp_ns));const last=valid.length?valid[valid.length-1].timestamp_ns:0;const windowNs=18e9;const samples=valid.filter(r=>r.timestamp_ns>=last-windowNs&&r.timestamp_ns<=last);let scale=minScale;for(const r of samples)for(const k of keys)if(finite(r[k]))scale=Math.max(scale,Math.abs(r[k]));scale=scale<=1?Math.ceil(scale*10)/10:Math.ceil(scale/5)*5;const x=t=>left+(t-(last-windowNs))/windowNs*(right-left);const y=v=>signed?(top+bottom)/2-v/scale*(bottom-top)/2:bottom-v/scale*(bottom-top);let hasData=false;const paths=keys.map(k=>{let d='',open=false,previous=null,n=0;for(const r of samples){if(!finite(r[k])){open=false;continue;}if(previous!==null&&r.timestamp_ns-previous>600e6)open=false;d+=(open?' L':' M')+x(r.timestamp_ns).toFixed(2)+','+y(r[k]).toFixed(2);open=true;previous=r.timestamp_ns;n++;}hasData=hasData||n>=2;return d;});return{paths,scale,hasData,samples,last,left,right,top,bottom,width,height,signed};}
const actionLabels={arm:'engage steering',manual:'disengage AI',route_start:'start route',route_complete:'mark route complete',route_abort:'abort route'};
function makePendingAction(event,data,nowMs){
    const s=data?.status||{};
    return {event,phase:'sending',queuedAt:nowMs,beforeTimestamp:s.timestamp_ns??null,
            beforeTicks:s.ticks??null,beforeCounter:s.metrics?.events?.[event]??null};
}
function actionOutcome(action,data,nowMs,isOffline=false){
    const s=data?.status||{},counter=s.metrics?.events?.[action.event];
    const counterAdvanced=finite(counter)&&finite(action.beforeCounter)&&counter>action.beforeCounter;
    const newSnapshot=(finite(s.timestamp_ns)&&(action.beforeTimestamp===null||s.timestamp_ns>action.beforeTimestamp))
        ||(finite(s.ticks)&&finite(action.beforeTicks)&&s.ticks>action.beforeTicks)||counterAdvanced;
    let confirmed=false;
    if(action.phase==='queued'&&newSnapshot&&!isOffline&&!data?.stale){
        if(action.event==='arm')confirmed=s.mode==='assist'&&!s.shadow;
        else if(action.event==='manual')confirmed=s.mode==='takeover'||s.mode==='manual';
        else if(action.event==='route_start')confirmed=counterAdvanced
            ||(action.beforeCounter===null&&s.route_active===true);
        else if(action.event==='route_complete'||action.event==='route_abort')confirmed=counterAdvanced;
    }
    const confirmations={arm:'assisted steering engaged',manual:'manual control / AI assistance disengaged',
        route_start:'route-start marker recorded',route_complete:'route-complete marker recorded',route_abort:'route-abort marker recorded'};
    if(confirmed)return {done:true,error:false,message:'Confirmed by runtime: '+confirmations[action.event]+'.'};
    if(action.phase==='queued'&&nowMs-action.queuedAt>=8000){
        const reason=isOffline||data?.stale?'fresh runtime status is unavailable'
            :'runtime reports '+humanize(s.mode||'unknown')+' ('+humanize(s.reason)+')';
        return {done:true,error:true,message:'Not confirmed: '+actionLabels[action.event]+' — '+reason+'.'};
    }
    return {done:false,error:false,message:'Queued: '+actionLabels[action.event]+'. Awaiting runtime confirmation.'};
}
function timeTicks(width){
    const fractions=width<450?[0,.5,1]:[0,.25,.5,.75,1];
    return fractions.map(fraction=>({fraction,label:fraction===1?'latest':'-'+Number(((1-fraction)*18).toFixed(1))+'s'}));
}

let lastData=null,offline=false,pendingAction=null,lastReceivedAt=null;const visibleSeries=new Set(['actual_angle_deg','target_angle_deg','predicted_angle_deg']);
function renderGrid(id,chart){const ticks=chart.signed?[chart.scale,chart.scale/2,0,-chart.scale/2,-chart.scale]:[chart.scale,chart.scale/2,0];let content='';ticks.forEach((v,i)=>{const y=chart.top+i*(chart.bottom-chart.top)/(ticks.length-1);content+='<line class="'+(v===0?'grid-line zero-line':'grid-line')+'" x1="'+chart.left+'" x2="'+chart.right+'" y1="'+y+'" y2="'+y+'"/><text x="'+(chart.left-8)+'" y="'+(y+3)+'" text-anchor="end">'+fmt(v,chart.scale<=1?1:0)+'</text>';});timeTicks(chart.width).forEach(tick=>{const v=tick.fraction,x=chart.left+v*(chart.right-chart.left);content+='<line class="grid-line" x1="'+x+'" x2="'+x+'" y1="'+chart.top+'" y2="'+chart.bottom+'" opacity=".5"/><text x="'+x+'" y="'+(chart.height-12)+'" text-anchor="'+(v===1?'end':v===0?'start':'middle')+'">'+tick.label+'</text>';});document.getElementById(id).innerHTML=content;}
function chartSize(id,width,height){const node=document.getElementById(id);const size={width:node.clientWidth||width,height:node.clientHeight||height};node.setAttribute('viewBox','0 0 '+size.width+' '+size.height);return size;}
function plot(rows){const keys=['actual_angle_deg','target_angle_deg','predicted_angle_deg'],chart=chartData(rows,keys.filter(k=>visibleSeries.has(k)),chartSize('steeringChart',720,260));renderGrid('steeringGrid',chart);keys.forEach((k,i)=>{const index=[...keys.filter(key=>visibleSeries.has(key))].indexOf(k);document.getElementById(['actualLine','targetLine','predictionLine'][i]).setAttribute('d',index<0?'':chart.paths[index]);});document.getElementById('steeringEmpty').hidden=chart.hasData;document.getElementById('steeringEmpty').querySelector('strong').textContent=visibleSeries.size?'Waiting for steering samples':'Select a trace above';document.getElementById('plotInfo').textContent=chart.samples.length?chart.samples.length+' snapshots · '+fmt((chart.last-chart.samples[0].timestamp_ns)/1e9)+' s observed':'0 samples · no recorded trace';for(const spec of [{key:'observation_age_ms',grid:'ageGrid',chart:'ageChart',line:'ageLine',empty:'ageEmpty',signed:false,minScale:50},{key:'torque',grid:'torqueGrid',chart:'torqueChart',line:'torqueLine',empty:'torqueEmpty',signed:true,minScale:.2}]){const mini=chartData(rows,[spec.key],{signed:spec.signed,minScale:spec.minScale,...chartSize(spec.chart,350,130)});renderGrid(spec.grid,mini);document.getElementById(spec.line).setAttribute('d',mini.paths[0]);document.getElementById(spec.empty).hidden=mini.hasData;}return chart;}
function render(data,isOffline=false){lastData=data;offline=isOffline;const s=data?.status||{},m=s.metrics||{},rates=s.rates||{},routes=m.routes||{},record=s.recording,v=viewState(data,isOffline);const el=id=>document.getElementById(id),text=(id,value)=>{el(id).textContent=value;};el('stateStrip').dataset.state=v.kind;text('health',v.title);text('stateDetail',v.detail);text('reason',s.reason||'No status received');text('freshness','Snapshot '+fmt(data?.snapshot_age_ms,0)+' ms ago');el('connection').dataset.state=isOffline?'offline':data?.stale?'stale':v.kind==='waiting'?'waiting':'live';text('connectionLabel',isOffline?'OFFLINE':v.kind==='waiting'?'AWAITING RUNTIME':data?.stale?'SNAPSHOT STALE':'RUNTIME CONNECTED');text('elapsed',duration(m.duration_seconds));text('hardwareBadge',s.hardware_mode==='simulation'?'Simulation / no hardware':s.hardware_mode||'Backend unknown');el('hardwareBadge').className='badge'+(s.hardware_mode==='simulation'?' simulation':'');text('policyBadge',s.shadow?'Shadow / observe only':s.policy_name||'Policy unavailable');el('policyBadge').className='badge'+(s.shadow?' shadow':'');text('actual',fmt(s.actual_angle_deg));text('target',fmt(s.target_angle_deg));text('speed',fmt(finite(s.speed_mps)?s.speed_mps*3.6:null));text('age',fmt(s.observation_age_ms));text('predictionCaption','Policy prediction '+fmt(s.predicted_angle_deg)+'°'+(s.auto_pedals?' · Gas '+fmt(s.output_throttle*100,0)+'% · Brake '+fmt(s.output_brake*100,0)+'%':''));text('speedCaption',finite(s.speed_mps)?fmt(s.speed_mps)+' m/s · Forza telemetry':'Forza telemetry unavailable');text('sourceMiniValue',fmt(s.observation_age_ms)+' ms');text('torque',fmt(s.torque,3)+' normalized');text('captureRate',fmt(rates.capture_fps));text('captureDescription',s.hardware_mode==='simulation'&&!finite(rates.capture_fps)?'No camera · simulated wheel test':'Fresh frames observed by runtime');text('policyRate',fmt(rates.policy_fps));text('controlRate',fmt(rates.control_hz));text('policyName',s.policy_name||'Policy not reported');text('hardwareName',s.hardware_mode==='simulation'?'Simulated wheel · no physical motor':s.hardware_mode||'Backend not reported');text('transportTag',s.transport||s.network||'Not reported');text('inferenceLabel',s.network==='LAN inference'?'Prediction round trip':'Policy time');text('inference',fmt(s.inference_ms)+' ms');text('sourceLimit',fmt(s.limits?.observation_age_ms,0)+' ms');text('controlLimits','±'+fmt(s.limits?.target_angle_deg,0)+'° / '+fmt(finite(s.limits?.torque)?s.limits.torque*100:null,0)+'%');text('sourceP95',fmt(m.latency?.observation_age?.p95_ms)+' ms');text('controlP95',fmt(m.latency?.control_tick_gap?.p95_ms)+' ms');text('input',humanize(s.input_status));let recordState='unknown',recordLabel='Unknown',recordNote='Recording counters appear when a session recorder is attached.';if(record===null){recordState='off';recordLabel='Not recording';recordNote='No recording session is attached to this run.';}else if(record){recordState=record.error?'error':record.completed?'complete':record.running?'recording':'stopped';recordLabel=record.error?'Write error':record.completed?'Complete':record.running?'Recording':'Stopped';recordNote=record.error?String(record.error):record.completed?'Session writer completed. Validate the saved session before training.':record.running?'Samples are being saved. Completion is confirmed only after the writer drains.':'The writer is stopped. Session completion is not confirmed.';}el('recordingState').dataset.state=recordState;text('recordingLabel',recordLabel);text('recordingCaption',record?'Timestamped frames, wheel, and telemetry':record===null?'Recording is not enabled':'Recorder status not reported');text('recordFrames',count(record?.frames_written));text('recordWheel',count(record?.wheel_samples));text('recordDrops',count(record?.dropped));text('recordTelemetry',count(record?.telemetry_samples));text('recordQueue',count(record?.queue_depth));text('expertLabel',s.expert_recording===true?'Eligible human samples':s.expert_recording===false?'Not currently eligible':'Not reported');text('recordNote',recordNote);el('recordNote').dataset.error=recordState==='error';text('mode','MODE / '+(s.shadow?'SHADOW':(s.mode||'—').toUpperCase()));text('route',s.route_active===true?'Route in progress':s.route_active===false?'No active route':'Route status unavailable');text('routeHint',routes.active&&finite(routes.active_seconds)?duration(routes.active_seconds)+' elapsed · operator marked':finite(routes.last_duration_seconds)?'Last attempt '+duration(routes.last_duration_seconds)+' · operator marked':'Start a marker to track a route attempt.');text('routesCompleted',count(routes.completed)+' / '+count(routes.attempts));text('interventions',count(m.human_interventions??s.human_interventions));text('trackingRmse',fmt(m.tracking_rmse_deg,2));text('faults',count(m.fault_entries));text('footerStatus',v.readOnly?'READ-ONLY VIEW · CONTROLS DISABLED':isOffline?'CONNECTION LOST · LAST KNOWN VALUES':data?.stale?'STALE SNAPSHOT · LAST KNOWN VALUES':'LOCAL DASHBOARD · P95 VALUES ARE APPROXIMATE');document.querySelectorAll('[data-event]').forEach(b=>{const e=b.dataset.event;b.disabled=e==='arm'?!v.arm:e==='manual'?(b.dataset.quick?!v.disengage:!v.takeover):e==='route_start'?!v.startRoute:!v.finishRoute;});text('arm',s.mode==='assist'&&!s.shadow?(s.auto_pedals?'Driving engaged':'Steering engaged'):(s.auto_pedals?'Engage driving':'Engage steering'));updatePending(data,isOffline);if(!el('message').dataset.hasFeedback)text('message',v.readOnly?'Read-only dashboard. Operator controls are disabled.':isOffline?'Dashboard offline. Use the physical takeover control.':data?.stale?'Engagement and route markers are disabled while snapshots are stale.':s.shadow?'Shadow run: predictions only. Motor engagement is disabled.':'Requests are queued; watch the mode above for confirmation.');plot(data?.history||[]);}
function commandFeedback(button,text,isError=false){const message=document.getElementById('message');message.dataset.hasFeedback='true';message.dataset.error=String(isError);message.textContent=text;if(button.dataset.quick){const quick=document.getElementById('quickFeedback');quick.hidden=false;quick.textContent=text;}}
function updatePending(data,isOffline=false){if(!pendingAction||pendingAction.phase!=='queued')return;const outcome=actionOutcome(pendingAction,data,performance.now(),isOffline);commandFeedback(pendingAction.button,outcome.message,outcome.error);if(outcome.done)pendingAction=null;}
async function requestJSON(url,options={},timeoutMs=1000){
    const controller=new AbortController();
    let timer;
    const deadline=new Promise((resolve,reject)=>{timer=setTimeout(()=>{
        reject(Error('Request timed out'));
        controller.abort();
    },timeoutMs);});
    try{
        return await Promise.race([deadline,(async()=>{
            const response=await fetch(url,{...options,signal:controller.signal});
            return {response,result:await response.json()};
        })()]);
    }finally{clearTimeout(timer);}
}
function checkFreshness(){
    if(!lastData||offline||lastData.stale||lastReceivedAt===null)return;
    const age=(finite(lastData.snapshot_age_ms)?lastData.snapshot_age_ms:0)+performance.now()-lastReceivedAt;
    if(age>=500)render({...lastData,stale:true,snapshot_age_ms:age});
}
async function refresh(){try{const {response,result}=await requestJSON('/api/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);lastReceivedAt=performance.now();render(result);}catch(error){render(lastData||{status:{},history:[],controls:{},stale:true},true);}finally{setTimeout(refresh,250);}}
if(typeof document!=='undefined'){document.querySelectorAll('[data-event]').forEach(button=>button.addEventListener('click',async()=>{
    button.disabled=true;
    const action=makePendingAction(button.dataset.event,lastData,performance.now());
    action.button=button;
    pendingAction=action;
    if(!button.dataset.quick)document.getElementById('quickFeedback').hidden=true;
    commandFeedback(button,'Sending request…');
    try{
        const {response,result}=await requestJSON('/api/events',{method:'POST',headers:{'Content-Type':'application/json','X-Forza-Token':token},body:JSON.stringify({event:action.event})});
        if(pendingAction!==action)return;
        if(response.ok){
            action.phase='queued';
            action.queuedAt=performance.now();
            updatePending(lastData,offline);
        }else{
            pendingAction=null;
            commandFeedback(button,'Request rejected: '+(result.error||'unknown error'),true);
        }
    }catch(error){
        if(pendingAction!==action)return;
        pendingAction=null;
        commandFeedback(button,'Delivery not confirmed. Check runtime state or use the wheel controls.',true);
    }
}));document.querySelectorAll('[data-series]').forEach(button=>button.addEventListener('click',()=>{const key=button.dataset.series;if(visibleSeries.has(key))visibleSeries.delete(key);else visibleSeries.add(key);button.setAttribute('aria-pressed',String(visibleSeries.has(key)));plot(lastData?.history||[]);}));const chart=document.getElementById('steeringChart');chart.addEventListener('pointermove',event=>{const model=chartData(lastData?.history||[],['actual_angle_deg','target_angle_deg','predicted_angle_deg'],chartSize('steeringChart',720,260));if(!model.hasData)return;const box=chart.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,((event.clientX-box.left)/box.width*model.width-model.left)/(model.right-model.left)));const desired=model.last-(1-fraction)*18e9;let sample=model.samples[0];for(const candidate of model.samples)if(Math.abs(candidate.timestamp_ns-desired)<Math.abs(sample.timestamp_ns-desired))sample=candidate;const x=model.left+(sample.timestamp_ns-(model.last-18e9))/18e9*(model.right-model.left),cursor=document.getElementById('chartCursor');cursor.setAttribute('x1',x);cursor.setAttribute('x2',x);cursor.setAttribute('visibility','visible');const tooltip=document.getElementById('chartTooltip');tooltip.textContent=fmt((model.last-sample.timestamp_ns)/1e9)+' s ago\nActual '+fmt(sample.actual_angle_deg)+'°  Target '+fmt(sample.target_angle_deg)+'°\nPrediction '+fmt(sample.predicted_angle_deg)+'°';tooltip.hidden=false;});chart.addEventListener('pointerleave',()=>{document.getElementById('chartTooltip').hidden=true;document.getElementById('chartCursor').setAttribute('visibility','hidden');});window.addEventListener('resize',()=>plot(lastData?.history||[]));setInterval(checkFreshness,100);plot([]);refresh();}
</script></body></html>
"""
