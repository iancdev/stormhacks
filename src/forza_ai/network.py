"""Authenticated, bounded LAN inference transport; no remote clock arithmetic.

Wire v1: uint32 network-order body length, 32-byte HMAC-SHA256, then a uint32
metadata length, canonical UTF-8 JSON, and optional lossless RGB PNG bytes. The
HMAC covers a protocol domain, the outer length, and the entire unsigned body.
Authentication is verified before parsing JSON or decoding an image. HMAC gives
integrity/authentication, not encryption; use this link on a trusted LAN.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import ipaddress
import json
import math
import os
import secrets
import socket
import struct
import threading
import time

import numpy as np
from PIL import Image

from forza_ai.contracts import ModelObservation, ObservationUnavailable


VERSION = 1
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024
MAX_WIDTH, MAX_HEIGHT = 3840, 2160
_U32 = struct.Struct("!I")
_DOMAIN = b"forza-inference-v1\0"


class ProtocolError(ValueError):
    """Authenticated transport contains invalid or incorrectly correlated data."""


class AuthenticationError(ProtocolError):
    """The shared key did not authenticate the received bytes."""


def _key_bytes(key=None):
    if key is None:
        key = os.environ.get("FORZA_LINK_KEY")
    if isinstance(key, str):
        key = key.encode("utf-8")
    if not isinstance(key, bytes) or not key:
        raise ValueError("set a nonempty FORZA_LINK_KEY on both hosts")
    return key


def _ipv4(host):
    try:
        return str(ipaddress.IPv4Address(host))
    except ipaddress.AddressValueError as error:
        raise ValueError("host must be a numeric IPv4 address; DNS is outside the I/O deadline") from error


def _port(port, allow_zero=False):
    if type(port) is not int or not (0 if allow_zero else 1) <= port <= 65535:
        raise ValueError("invalid TCP port")
    return port


def _timeout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_s must be finite and positive")
    return float(value)


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("inference operation deadline expired")
    return remaining


def _read_exact(sock, size, deadline):
    chunks = bytearray()
    while len(chunks) < size:
        # Some platforms do not wake another thread's socket select immediately
        # on close. Short waits bound cancellation without extending the deadline.
        sock.settimeout(min(_remaining(deadline), 0.05))
        try:
            chunk = sock.recv(min(size - len(chunks), 65536))
        except socket.timeout:
            continue
        if not chunk:
            raise ConnectionError("inference connection closed")
        chunks.extend(chunk)
    _remaining(deadline)
    return bytes(chunks)


def _canonical(metadata):
    try:
        return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise ProtocolError("invalid message metadata") from error


def _encode_message(key, metadata, payload=b""):
    encoded = _canonical(metadata)
    if len(encoded) > MAX_METADATA_BYTES:
        raise ProtocolError("message metadata exceeds size limit")
    content = _U32.pack(len(encoded)) + encoded + payload
    length = len(content) + hashlib.sha256().digest_size
    if length > MAX_MESSAGE_BYTES:
        raise ProtocolError("message exceeds 8 MiB size limit")
    prefix = _U32.pack(length)
    signature = hmac.digest(key, _DOMAIN + prefix + content, "sha256")
    return prefix + signature + content


def _send_message(sock, key, metadata, payload, deadline):
    data = memoryview(_encode_message(key, metadata, payload))
    while data:
        sock.settimeout(min(_remaining(deadline), 0.05))
        try:
            sent = sock.send(data)
        except socket.timeout:
            continue
        if not sent:
            raise ConnectionError("inference connection closed")
        data = data[sent:]
    _remaining(deadline)


def _receive_message(sock, key, deadline):
    prefix = _read_exact(sock, 4, deadline)
    length = _U32.unpack(prefix)[0]
    if not 38 <= length <= MAX_MESSAGE_BYTES:
        raise ProtocolError("invalid message length")
    body = _read_exact(sock, length, deadline)
    signature, content = body[:32], body[32:]
    expected = hmac.digest(key, _DOMAIN + prefix + content, "sha256")
    if not hmac.compare_digest(signature, expected):
        raise AuthenticationError("inference message authentication failed")
    metadata_length = _U32.unpack(content[:4])[0]
    if not 2 <= metadata_length <= min(MAX_METADATA_BYTES, len(content) - 4):
        raise ProtocolError("invalid metadata length")
    encoded = content[4:4 + metadata_length]
    try:
        metadata = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise ProtocolError("invalid JSON metadata") from error
    if not isinstance(metadata, dict) or _canonical(metadata) != encoded:
        raise ProtocolError("metadata must be a canonical JSON object")
    _remaining(deadline)
    return metadata, content[4 + metadata_length:]


def _schema(metadata, kind, fields):
    if set(metadata) != {"version", "kind", *fields}:
        raise ProtocolError("unexpected message fields")
    if type(metadata["version"]) is not int or metadata["version"] != VERSION or metadata["kind"] != kind:
        raise ProtocolError("unsupported protocol version or message kind")


def _session(value):
    if not isinstance(value, str) or len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise ProtocolError("invalid connection session")
    return value


def _identifier(value, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value <= 2**63 - 1:
        raise ProtocolError("invalid request or frame ID")
    return value


def _speed(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ProtocolError("speed must be finite and nonnegative")
    return float(value)


def _angle(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 450:
        raise ProtocolError("predicted angle must be finite and within +/-450 degrees")
    return float(value)


def _dimensions(width, height):
    if (type(width) is not int or type(height) is not int
            or not 0 < width <= MAX_WIDTH or not 0 < height <= MAX_HEIGHT):
        raise ProtocolError("image dimensions must be within 3840x2160")


def _encode_png(pixels):
    if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
            or pixels.ndim != 3 or pixels.shape[2] != 3):
        raise ProtocolError("expected uint8 HWC RGB road crop")
    height, width = pixels.shape[:2]
    _dimensions(width, height)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=1)
    payload = buffer.getvalue()
    if len(payload) > MAX_MESSAGE_BYTES - 1024:
        raise ProtocolError("PNG exceeds transport size limit")
    return payload, width, height


def _decode_png(payload, width, height):
    _dimensions(width, height)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height):
                raise ProtocolError("payload must be an RGB PNG matching declared dimensions")
            image.load()
            return np.array(image, dtype=np.uint8, copy=True)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ProtocolError("invalid PNG image payload") from error


class RemotePolicy:
    """One in-flight request, called only by the runtime's policy worker.

    The server creates a fresh authenticated session challenge per connection;
    monotonically increasing request IDs and a fresh client nonce bind each
    response to its request, even if an old signed hello is replayed to a client.
    No host monotonic timestamps cross the network. The caller must keep using
    the original local observation timestamp for command freshness checks.
    """

    requires_camera = True
    name = "remote CNN steering policy"

    def __init__(self, host, port=8765, key=None, timeout_s=0.2):
        self.host = _ipv4(host)
        self.port = _port(port)
        self.timeout_s = _timeout(timeout_s)
        self._key = _key_bytes(key)
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._socket = None
        self._session = None
        self._request_id = 1
        self._closed = False

    def _connect(self, deadline):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self._state_lock:
            if self._closed:
                sock.close()
                raise ObservationUnavailable("remote policy is closed")
            self._socket = sock
        sock.settimeout(_remaining(deadline))
        sock.connect((self.host, self.port))
        metadata, payload = _receive_message(sock, self._key, deadline)
        _schema(metadata, "hello", {"session"})
        if payload:
            raise ProtocolError("unexpected hello payload")
        self._session = _session(metadata["session"])
        self._request_id = 1
        return sock

    def predict(self, observation: ModelObservation) -> float:
        frame, vehicle = observation.frame, observation.vehicle
        if frame is None or vehicle is None:
            raise ObservationUnavailable("missing camera frame or telemetry")
        for timestamp in (frame.timestamp_ns, vehicle.timestamp_ns):
            if type(timestamp) is not int or timestamp < 0:
                raise ProtocolError("invalid local observation timestamp")
        if vehicle.is_race_on not in (True, False):
            raise ProtocolError("invalid race state")
        if not vehicle.is_race_on or not 0 <= frame.timestamp_ns - vehicle.timestamp_ns <= 100_000_000:
            raise ObservationUnavailable("game paused or telemetry unavailable at frame time")
        speed = _speed(vehicle.speed_mps)
        frame_id = _identifier(frame.frame_id)
        with self._lock:
            with self._state_lock:
                if self._closed:
                    raise ObservationUnavailable("remote policy is closed")
            deadline = time.monotonic() + self.timeout_s
            try:
                png, width, height = _encode_png(frame.rgb)
                _remaining(deadline)
                with self._state_lock:
                    if self._closed:
                        raise ObservationUnavailable("remote policy is closed")
                    sock = self._socket
                if sock is None:
                    sock = self._connect(deadline)
                request_id = self._request_id
                nonce = secrets.token_hex(16)  # also reject old server transcripts
                request = {"version": VERSION, "kind": "predict", "session": self._session,
                           "request_id": request_id, "nonce": nonce, "frame_id": frame_id, "speed_mps": speed,
                           "width": width, "height": height}
                _send_message(sock, self._key, request, png, deadline)
                response, payload = _receive_message(sock, self._key, deadline)
                _schema(response, "prediction", {"session", "request_id", "nonce", "frame_id", "angle_deg"})
                if (payload or _session(response["session"]) != self._session
                        or _identifier(response["request_id"], positive=True) != request_id
                        or _session(response["nonce"]) != nonce
                        or _identifier(response["frame_id"]) != frame_id):
                    raise ProtocolError("response does not match the in-flight observation")
                angle = _angle(response["angle_deg"])
                _remaining(deadline)
                with self._state_lock:
                    if self._closed:
                        raise ObservationUnavailable("remote policy is closed")
                self._request_id += 1
                return angle
            except (OSError, TimeoutError) as error:
                self._disconnect()
                raise ObservationUnavailable("remote inference timed out or disconnected") from error
            except BaseException:
                self._disconnect()
                raise

    def _disconnect(self):
        with self._state_lock:
            sock = self._socket
            self._socket = None
        self._close_socket(sock)
        self._session = None
        self._request_id = 1

    @staticmethod
    def _close_socket(sock):
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def close(self):
        # Do not wait behind predict's serialization lock or a blocked recv.
        with self._state_lock:
            self._closed = True
            sock = self._socket
            self._socket = None
        self._close_socket(sock)
