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
    "network", "transport", "route_active", "human_interventions", "error", "closed",
))
_SENSITIVE = ("token", "secret", "password", "authorization", "credential", "key")


def _safe(value, depth=0):
    """Bounded JSON normalization for diagnostic values; omit credential fields."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, int):
        return value if abs(value) <= 2 ** 64 - 1 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict) and depth < 2:
        output = {}
        for index, (key, entry) in enumerate(value.items()):
            if index >= 32:
                break
            if isinstance(key, str) and not any(part in key.lower() for part in _SENSITIVE):
                output[key[:64]] = _safe(entry, depth + 1)
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
                                      ("timestamp_ns", "actual_angle_deg", "target_angle_deg", "predicted_angle_deg")})
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


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stormhacks · Wheel control</title><style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#0a101a;color:#e6eef8}
*{box-sizing:border-box}body{margin:0;padding:34px 5vw;max-width:1280px;margin:auto}header{display:flex;justify-content:space-between;align-items:center;gap:18px}
h1{font-size:30px;letter-spacing:-1px;margin:4px 0 10px}.eyebrow{font:12px monospace;letter-spacing:2px;color:#74ddbd;text-transform:uppercase}
.muted{color:#94a8bf;font-size:14px}.badge{padding:10px 16px;border-radius:30px;background:#18283e;font-weight:650}.good{color:#74ddbd}.bad{color:#ffb68d}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:26px}.card,.panel{background:#121d2c;border:1px solid #27374c;border-radius:14px;padding:20px}
.label{color:#a6b8cc;font-size:13px}.value{font-variant-numeric:tabular-nums;font-size:31px;margin:10px 0 3px}.unit{font-size:13px;color:#94a8bf}
.panel{margin-top:16px}svg{width:100%;height:220px;display:block;margin-top:16px}.legend{display:flex;gap:22px;font-size:13px}.target{color:#87adff}.actual{color:#74ddbd}.predicted{color:#ffcc80}
.controls{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}button{font:inherit;font-size:14px;padding:11px 17px;border-radius:9px;border:1px solid #40536d;background:#24344c;color:white;cursor:pointer}
.controls-note{margin-top:12px;line-height:1.5}
button.primary{background:#74ddbd;color:#08231b;border-color:#74ddbd;font-weight:700}button.stop{border-color:#edaa87;color:#ffd1b7}button:disabled{opacity:.35;cursor:not-allowed}
.details{display:grid;grid-template-columns:1fr 1fr;gap:14px}.row{display:flex;justify-content:space-between;gap:15px;padding:8px 0;font-size:14px;border-bottom:1px solid #263549}.row span:last-child{text-align:right;overflow-wrap:anywhere}
#message{min-height:20px;margin:12px 0;color:#b6c9e0}pre{white-space:pre-wrap;font-size:12px;color:#9eb4cc;max-height:160px;overflow:auto}
@media(max-width:750px){body{padding:22px}.grid{grid-template-columns:1fr 1fr}.details{grid-template-columns:1fr}h1{font-size:24px}.value{font-size:25px}}
</style></head><body><header><div><div class="eyebrow">Stormhacks / live control</div><h1>The wheel, in real time.</h1><div class="muted">Local operator panel · human pedals, AI steering</div></div><span id="health" class="badge bad">Waiting for runtime</span></header>
<div class="grid"><div class="card"><div class="label">Physical wheel</div><div class="value" id="actual">—</div><div class="unit">degrees · right positive</div></div>
<div class="card"><div class="label">Controller target</div><div class="value" id="target">—</div><div class="unit">degrees</div></div>
<div class="card"><div class="label">Game speed</div><div class="value" id="speed">—</div><div class="unit">km/h</div></div>
<div class="card"><div class="label">Observation age</div><div class="value" id="age">—</div><div class="unit">milliseconds · includes pipeline delay</div></div></div>
<div class="panel"><div class="legend"><span class="actual">● Measured wheel</span><span class="target">● Controller target</span><span class="predicted">● Model prediction</span></div>
<svg viewBox="0 0 1000 220" role="img" aria-label="Recent physical wheel, target and prediction angles"><line x1="0" x2="1000" y1="110" y2="110" stroke="#32465f" stroke-dasharray="5 5"/><path id="actualLine" fill="none" stroke="#74ddbd" stroke-width="2.5"/><path id="targetLine" fill="none" stroke="#87adff" stroke-width="2"/><path id="predictionLine" fill="none" stroke="#ffcc80" stroke-width="1.5"/></svg>
<div class="muted" id="plotInfo">Latest 18 seconds · auto-scaled degrees</div></div>
<div class="controls"><button class="primary" id="arm" data-event="arm" disabled>Engage steering</button><button class="stop" data-event="manual">Take over</button><button data-event="route_start">Start route</button><button data-event="route_complete">Mark complete</button><button data-event="route_abort">Abort route</button></div>
<div class="muted controls-note">Commands are requests to the runtime. Route outcomes are manual markers, not automatic evaluation.</div><div id="message" role="status"></div>
<div class="details"><div class="panel"><div class="label">Control & timing</div><div class="row"><span>Mode</span><span id="mode">—</span></div><div class="row"><span>Reason</span><span id="reason">—</span></div><div class="row"><span>Input status</span><span id="input">—</span></div><div class="row"><span>Route marker</span><span id="route">—</span></div><div class="row"><span>Human takeovers from assistance</span><span id="interventions">—</span></div><div class="row"><span>Applied torque</span><span id="torque">—</span></div><div class="row"><span>Inference</span><span id="inference">—</span></div><div class="row"><span>Dashboard snapshot age</span><span id="freshness">—</span></div></div>
<div class="panel"><div class="label">Recording & transport</div><pre id="pipeline">Waiting for runtime…</pre><div class="muted">Values are current runtime counters. Stale values are retained for diagnosis.</div></div></div>
<script>
const token='__TOKEN__';const el=id=>document.getElementById(id);const fmt=(v,n=1)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'—';
function plot(rows){const keys=['actual_angle_deg','target_angle_deg','predicted_angle_deg'];const ids=['actualLine','targetLine','predictionLine'];let scale=5;for(const r of rows)for(const k of keys)if(Number.isFinite(r[k]))scale=Math.max(scale,Math.abs(r[k]));
 keys.forEach((key,i)=>{let path='',open=false;rows.forEach((r,j)=>{const v=r[key];if(!Number.isFinite(v)){open=false;return;}const x=j*1000/Math.max(1,rows.length-1),y=110-v/scale*100;path+=(open?' L':' M')+x.toFixed(1)+','+y.toFixed(1);open=true;});el(ids[i]).setAttribute('d',path);});el('plotInfo').textContent='Latest snapshots · scale ±'+fmt(scale)+'° · up to 18 seconds';}
async function refresh(){try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const data=await response.json(),s=data.status;
 el('health').textContent=data.stale?'STALE · steering status unknown':s.shadow?'SHADOW · no motor control':'LIVE · '+(s.mode||'waiting').toUpperCase();el('health').className='badge '+(data.stale?'bad':'good');
 el('actual').textContent=fmt(s.actual_angle_deg);el('target').textContent=fmt(s.target_angle_deg);el('speed').textContent=fmt(typeof s.speed_mps==='number'?s.speed_mps*3.6:null);el('age').textContent=fmt(s.observation_age_ms);
 el('mode').textContent=s.shadow?'shadow (motor disabled)':s.mode||'—';el('reason').textContent=s.reason||'—';el('input').textContent=s.input_status||'—';el('torque').textContent=fmt(s.torque,3);el('inference').textContent=fmt(s.inference_ms)+' ms';el('freshness').textContent=fmt(data.snapshot_age_ms)+' ms';
 el('route').textContent=s.route_active===true?'Route in progress':s.route_active===false?'No active route':'—';el('interventions').textContent=fmt(s.human_interventions,0);
 el('pipeline').textContent=JSON.stringify({recording:s.recording??'not enabled',transport:s.transport??s.network??'local policy'},null,2);document.querySelectorAll('[data-event]').forEach(b=>b.disabled=data.controls.read_only);el('arm').disabled=!data.controls.allow_arm;plot(data.history);
 }catch(e){el('health').textContent='DISCONNECTED · status unavailable';el('health').className='badge bad';el('arm').disabled=true;}finally{setTimeout(refresh,250);}}
document.querySelectorAll('[data-event]').forEach(button=>button.addEventListener('click',async()=>{try{const response=await fetch('/api/events',{method:'POST',headers:{'Content-Type':'application/json','X-Forza-Token':token},body:JSON.stringify({event:button.dataset.event})});const result=await response.json();el('message').textContent=response.ok?'Requested '+result.queued+'; verify the mode above.':result.error;}catch(e){el('message').textContent='Command not delivered: '+e.message;}}));refresh();
</script></body></html>"""
