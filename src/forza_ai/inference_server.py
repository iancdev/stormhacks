"""Authenticated LAN model server; start explicitly on the GPU/inference PC."""

import argparse
from forza_ai.contracts import DrivingPrediction
import secrets
import select
import socket
import threading
import time

from forza_ai.network import (
    VERSION, AuthenticationError, ProtocolError, _angle, _decode_png, _identifier, _ipv4, _key_bytes,
    _port, _receive_message, _remaining, _schema, _send_message, _session, _speed, _timeout,
)


class _ReceiveCounter:
    """Observe byte counts only; socket deadlines and framing remain unchanged."""
    def __init__(self, client, state):
        self.client, self.state = client, state

    def settimeout(self, value):
        self.client.settimeout(value)

    def recv(self, size):
        data = self.client.recv(size)
        self.state['received_bytes'] += len(data)
        return data


class InferenceServer:
    """Single active connection with bounded I/O and independent model execution.

    A stalled predictor cannot bypass the client's deadline or local controller
    timeout. Python cannot forcibly stop model code: close shuts sockets first,
    then reports if its daemon worker has not stopped within one second.
    """

    def __init__(self, predictor, host="127.0.0.1", port=8765, key=None, timeout_s=2.0, event_log=None, *, saliency=False):
        self.event_log = event_log
        self._connection_sequence = 0
        self.predictor = predictor
        self.driving = getattr(predictor, "driving", False) is True
        self.version = 2 if self.driving else VERSION
        self.host = _ipv4(host)
        self.port = _port(port, allow_zero=True)
        self.timeout_s = _timeout(timeout_s)
        self._key = _key_bytes(key)
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._listener = None
        self._client = None
        self._thread = None
        self._address = None
        self._last_error = None
        self._running = False
        self._model_ready = threading.Condition()
        self._job = None
        self._model_thread = None
        self.preview = None
        if saliency:
            from forza_ai.visualbackprop import ActivationPreview
            self.preview = ActivationPreview(predictor)

    @property
    def address(self):
        with self._lock:
            return self._address

    @property
    def last_error(self):
        with self._lock:
            return self._last_error

    @property
    def running(self):
        with self._lock:
            return self._running

    def start(self):
        with self._lifecycle_lock:
            if (self._model_thread is not None and self._model_thread.is_alive()
                    and (self._stop.is_set() or self._thread is None or not self._thread.is_alive())):
                raise RuntimeError("inference worker is still stopping")
            if self._thread is not None and self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError("inference server is still stopping")
                return
            if self.preview is not None and self.preview._closed:
                from forza_ai.visualbackprop import ActivationPreview
                self.preview = ActivationPreview(self.predictor)
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listener.bind((self.host, self.port))
                listener.listen(1)
                listener.settimeout(0.1)
            except BaseException:
                listener.close()
                if self.preview is not None:
                    self.preview.close()
                raise
            self._stop.clear()
            self._listener = listener
            with self._lock:
                self._address = listener.getsockname()
                self._last_error = None
                self._running = True
            self._model_thread = threading.Thread(target=self._infer, name="model-inference", daemon=True)
            self._thread = threading.Thread(target=self._serve, args=(listener,),
                                            name="remote-inference", daemon=True)
            try:
                self._model_thread.start()
                self._thread.start()
            except BaseException:
                self._stop.set()
                with self._model_ready:
                    self._model_ready.notify_all()
                listener.close()
                self._listener = None
                self._thread = None
                with self._lock:
                    self._address = None
                    self._running = False
                raise

    def _log(self, event, state):
        if self.event_log is None:
            return
        try:
            elapsed = max(time.monotonic() - state['started'], 1e-9)
            fields = {key: state[key] for key in (
                'connection_id', 'peer_ip', 'peer_port', 'phase', 'cause',
                'authenticated', 'requests', 'responses', 'received_bytes', 'completed_model_calls', 'model_elapsed_ms')}
            fields.update(protocol_version=self.version, duration_ms=round(elapsed * 1000, 3),
                          response_hz=round(state['responses'] / elapsed, 3),
                          model_mean_ms=round(state['model_total_ms'] / max(state['completed_model_calls'], 1), 3),
                          model_max_ms=round(state['model_max_ms'], 3),
                          request_ms=round((time.monotonic() - state['request_started']) * 1000, 3),
                          timeout_ms=self.timeout_s * 1000)
            self.event_log.emit(event, **fields)
        except Exception:
            pass  # diagnostics must never alter protocol/control behavior

    @staticmethod
    def _close_cause(error, state):
        if state['cause']:
            return state['cause']
        if isinstance(error, AuthenticationError):
            return 'authentication_failed'
        if isinstance(error, TimeoutError):
            if state['phase'] == 'receive' and state['received_bytes'] == state['receive_start_bytes']:
                return 'idle_timeout'
            return 'timeout'
        if isinstance(error, ConnectionError):
            return 'peer_eof_or_reset'
        if isinstance(error, ProtocolError):
            return 'protocol_error'
        if isinstance(error, OSError):
            return 'transport_error'
        return 'internal_error'

    def _serve(self, listener):
        try:
            while not self._stop.is_set():
                try:
                    client, peer = listener.accept()
                except socket.timeout:
                    continue
                except OSError as error:
                    if not self._stop.is_set():
                        with self._lock:
                            self._last_error = error
                    break
                self._connection_sequence += 1
                now = time.monotonic()
                state = dict(connection_id=self._connection_sequence, peer_ip=peer[0], peer_port=peer[1],
                             started=now, request_started=now, next_summary=now + 5,
                             phase='hello', cause=None, authenticated=False, requests=0, responses=0,
                             received_bytes=0, receive_start_bytes=0, completed_model_calls=0, model_elapsed_ms=0.0,
                             model_total_ms=0.0, model_max_ms=0.0)
                self._log('connection_open', state)
                with self._lock:
                    self._client = client
                try:
                    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    if not self._stop.is_set():
                        self._serve_client(client, state)
                except Exception as error:
                    state['cause'] = 'server_stopping' if self._stop.is_set() else self._close_cause(error, state)
                    if not self._stop.is_set():
                        with self._lock:
                            self._last_error = error
                finally:
                    state['cause'] = state['cause'] or ('server_stopping' if self._stop.is_set() else 'completed')
                    self._log('connection_close', state)
                    client.close()
                    with self._lock:
                        self._client = None
        finally:
            listener.close()
            with self._lock:
                self._running = False
                self._address = None

    def _serve_client(self, client, state):
        session = secrets.token_hex(16)
        _send_message(client, self._key, {"version": self.version, "kind": "hello", "session": session},
                      b"", time.monotonic() + self.timeout_s)
        next_request_id = 1
        last_preview_id = 0
        while not self._stop.is_set():
            deadline = time.monotonic() + self.timeout_s
            state.update(phase='receive', request_started=time.monotonic(),
                         receive_start_bytes=state['received_bytes'])
            request, payload = _receive_message(_ReceiveCounter(client, state), self._key, deadline)
            state['phase'] = 'protocol'
            state['authenticated'] = True
            wants_preview = request.get("kind") == "predict_preview"
            _schema(request, "predict_preview" if wants_preview else "predict", {"session", "request_id", "nonce", "frame_id", "speed_mps", "width", "height"}, self.version)
            if (_session(request["session"]) != session
                    or _identifier(request["request_id"], positive=True) != next_request_id):
                raise ProtocolError("replayed or incorrectly ordered request")
            frame_id = _identifier(request["frame_id"])
            nonce = _session(request["nonce"])
            speed = _speed(request["speed_mps"])
            _remaining(deadline)
            if self._stop.is_set():
                return
            next_request_id += 1
            state['requests'] += 1
            if state['requests'] == 1:
                self._log('authenticated_protocol_ready', state)
            state['phase'] = 'model_wait'
            prediction = self._predict(client, payload, request["width"], request["height"], speed, deadline, state,
                                       dict(session=session, request_id=request["request_id"], frame_id=frame_id) if wants_preview else None)
            state['phase'] = 'prediction_validation'
            if self.driving != isinstance(prediction, DrivingPrediction):
                raise ProtocolError('predictor output does not match protocol version')
            angle = _angle(prediction.angle_deg if self.driving else prediction)
            response = {"version": self.version, "kind": "prediction", "session": session,
                        "request_id": request["request_id"], "nonce": nonce,
                        "frame_id": frame_id, "angle_deg": angle}
            if self.driving:
                response.update(throttle=prediction.throttle, brake=prediction.brake)
            preview_payload = b""
            if wants_preview:
                response['kind'] = 'prediction_preview'
                response['preview'] = None
                if self.preview is not None:
                    response['preview'], preview_payload = self.preview.latest(session, last_preview_id)
                    if response['preview'] is not None:
                        last_preview_id = response['preview']['request_id']
            state['phase'] = 'send'
            _send_message(client, self._key, response, preview_payload, deadline)
            state['responses'] += 1
            if state['responses'] == 1:
                self._log('first_prediction_sent', state)
            if time.monotonic() >= state['next_summary']:
                self._log('connection_summary', state)
                state['next_summary'] = time.monotonic() + 5

    def _infer(self):
        """One model owner, at most one job, never a queue of disconnected clients."""
        while not self._stop.is_set():
            with self._model_ready:
                self._model_ready.wait_for(lambda: self._job is not None or self._stop.is_set())
                if self._stop.is_set():
                    return
                job = self._job
            try:
                _remaining(job["deadline"])
                job['phase'] = 'decode'
                pixels = _decode_png(job["payload"], job["width"], job["height"])
                _remaining(job["deadline"])
                if not self._stop.is_set() and not job["cancelled"].is_set():
                    job['phase'] = 'model'
                    model_started = time.monotonic()
                    job['model_started'] = model_started
                    if self.preview is not None:
                        self.preview.begin(job.get("preview_identity"))
                    try:
                        job["prediction"] = self.predictor.predict(pixels, job["speed"])
                    finally:
                        if self.preview is not None:
                            self.preview.finish(job.get('prediction'))
                        job['model_ms'] = (time.monotonic() - model_started) * 1000
            except Exception as error:
                job["error"] = error
            finally:
                with self._model_ready:
                    self._job = None
                    job["done"].set()

    def _predict(self, client, payload, width, height, speed, deadline, state, preview_identity=None):
        job = dict(payload=payload, width=width, height=height, speed=speed, preview_identity=preview_identity,
                   deadline=deadline, done=threading.Event(), cancelled=threading.Event(), phase='model_wait')
        with self._model_ready:
            if self._job is not None:
                state['cause'] = 'busy'
                raise ProtocolError("inference busy; no queued work accepted")
            self._job = job
            self._model_ready.notify()
        try:
            while not job["done"].wait(min(.01, _remaining(deadline))):
                if self._stop.is_set():
                    raise OSError("inference server stopping")
                # Notice a timed-out client's disconnect while model code is
                # still running. Accept/authentication can then recover, without
                # launching another model invocation or keeping its request.
                readable, _, _ = select.select([client], [], [], 0)
                if readable:
                    peek = client.recv(1, socket.MSG_PEEK)
                    if not peek:
                        state['cause'] = 'peer_eof'
                        raise OSError("inference client disconnected")
                    state['cause'] = 'pipelined_request'
                    raise ProtocolError("pipelined requests are not supported")
            _remaining(deadline)
            if "error" in job:
                state['cause'] = ('model_error' if job['phase'] == 'model' else
                                  'timeout' if isinstance(job['error'], TimeoutError) else 'decode_error')
                raise job["error"]
            if "prediction" not in job:
                raise OSError("inference cancelled")
            return job["prediction"]
        finally:
            state['phase'] = job['phase']
            if 'model_started' in job:
                state['model_elapsed_ms'] = round(job.get('model_ms', (time.monotonic() - job['model_started']) * 1000), 3)
            if 'model_ms' in job:
                state['completed_model_calls'] += 1
                state['model_total_ms'] += job['model_ms']
                state['model_max_ms'] = max(state['model_max_ms'], job['model_ms'])
            job["cancelled"].set()

    def close(self):
        with self._lifecycle_lock:
            self._stop.set()
            with self._model_ready:
                self._model_ready.notify_all()
            with self._lock:
                client = self._client
                self._running = False
            if client is not None:
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                client.close()
            if self._listener is not None:
                self._listener.close()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
                if self._thread.is_alive():
                    raise RuntimeError("inference worker is still running; sockets have been closed")
            if self._model_thread is not None:
                self._model_thread.join(timeout=1.0)
                if self._model_thread.is_alive():
                    raise RuntimeError("inference worker is still running; sockets have been closed")
            if self.preview is not None:
                self.preview.close()
            self._listener = None
            self._thread = None
            self._model_thread = None
            with self._model_ready:
                self._job = None
            with self._lock:
                self._address = None


class _FixedPredictor:
    def __init__(self, angle):
        self.angle = _angle(angle)
        if abs(self.angle) > 15:
            raise ValueError("stationary test targets must be within +/-15 degrees")

    def predict(self, pixels, speed):
        return self.angle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1", help="numeric LAN IPv4 address; default loopback")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="CPU inference intra-op threads (default 4, measured batch-one setting)")
    parser.add_argument("--timeout", type=float, default=2.0, help="total per-request server I/O deadline, seconds")
    parser.add_argument("--saliency", action="store_true", help="opt-in 1 Hz positive-ELU VisualBackProp diagnostic (CPU PilotNet)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", help="export directory containing model.pt and metadata.json")
    mode.add_argument("--test-target", type=float, help="explicit fixed angle test; NOT a driving model")
    args = parser.parse_args(argv)
    if not 1 <= args.cpu_threads <= 1024:
        parser.error("--cpu-threads must be between 1 and 1024")
    if args.saliency and args.model is None:
        parser.error("--saliency requires --model")
    # Check configuration/key before loading a potentially expensive model.
    key = _key_bytes()
    _ipv4(args.bind)
    _port(args.port)
    _timeout(args.timeout)
    if args.model is not None:
        import torch
        from forza_ai.policies.predictor import load_predictor

        torch.set_num_threads(args.cpu_threads)
        predictor = load_predictor(args.model)
    else:
        predictor = _FixedPredictor(args.test_target)
    from forza_ai.inference_logging import InferenceLog
    event_log = InferenceLog()
    server = InferenceServer(predictor, args.bind, args.port, key, args.timeout, event_log=event_log, saliency=args.saliency)
    try:
        server.start()
        label = "CNN model" if args.model is not None else "FIXED TARGET TEST (not a driving model)"
        print(f"Authenticated {label} server listening on {server.address[0]}:{server.address[1]}", flush=True)
        while server.running:
            time.sleep(0.25)
        raise RuntimeError("inference listener stopped") from server.last_error
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            server.close()
        finally:
            event_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
