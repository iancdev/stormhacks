"""Strict FH4 Data Out parsing and a background UDP receiver.

Only the 324-byte Horizon 4 dash format used by ``utils/test.py`` is supported.
The receiver timestamps receipt with the host monotonic clock; it cannot measure
the time a packet spent in the game, network, or operating system's receive queue.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time

from .contracts import VehicleState


FH4_PACKET_SIZE = 324
DEFAULT_MAX_AGE_NS = 250_000_000


def _nonnegative_ns(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer number of nanoseconds")


def parse_fh4_packet(data: bytes, received_ns: int) -> VehicleState:
    """Decode known FH4 fields, raising ``ValueError`` for invalid packets.

    Speed stays in metres per second. Steering is the signed game input byte,
    not the physical wheel angle. Unknown fields are intentionally not decoded.
    """
    _nonnegative_ns(received_ns, "received_ns")
    if len(data) != FH4_PACKET_SIZE:
        raise ValueError(f"expected {FH4_PACKET_SIZE} FH4 bytes, received {len(data)}")
    race_on = struct.unpack_from("<i", data, 0)[0]
    if race_on not in (0, 1):
        raise ValueError("IsRaceOn must be 0 or 1")
    game_timestamp_ms = struct.unpack_from("<I", data, 4)[0]
    rpm = struct.unpack_from("<f", data, 16)[0]
    speed_mps = struct.unpack_from("<f", data, 256)[0]
    steering_input = struct.unpack_from("<b", data, 320)[0]
    if not math.isfinite(rpm) or not math.isfinite(speed_mps):
        raise ValueError("RPM and speed must be finite")
    return VehicleState(
        timestamp_ns=received_ns,
        speed_mps=speed_mps,
        is_race_on=bool(race_on),
        game_timestamp_ms=game_timestamp_ms,
        rpm=rpm,
        steering_input=steering_input,
    )


class TelemetryReceiver:
    """Receive telemetry without blocking the wheel loop.

    ``start`` and ``close`` are idempotent. ``latest`` returns ``None`` before a
    valid packet, after close, for future-dated samples, or when the sample is
    older than the requested limit. Reading a sample never refreshes its time.

    Duplicate/backward game timestamps are discarded using uint32 wrap ordering.
    After ``ordering_reset_ns`` without an accepted sample, the next valid packet
    establishes a new sequence. This lets a restarted game recover; at that
    boundary a delayed old packet is indistinguishable from a game restart.
    Consumers must still check ``is_race_on`` before using data for assistance.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        *,
        ordering_reset_ns: int = 2_000_000_000,
    ) -> None:
        _nonnegative_ns(ordering_reset_ns, "ordering_reset_ns")
        if ordering_reset_ns == 0:
            raise ValueError("ordering_reset_ns must be positive")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer from 0 to 65535")
        self._host = host
        self._port = port
        self._ordering_reset_ns = ordering_reset_ns
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._address: tuple[str, int] | None = None
        self._state: VehicleState | None = None
        self._running = False
        self._malformed_count = 0
        self._out_of_order_count = 0
        self._last_error: OSError | None = None

    @property
    def address(self) -> tuple[str, int] | None:
        """Actual bound IPv4 endpoint, including an OS-selected ephemeral port."""
        with self._lock:
            return self._address

    @property
    def malformed_count(self) -> int:
        with self._lock:
            return self._malformed_count

    @property
    def out_of_order_count(self) -> int:
        """Count rejected duplicate and backward packets across starts."""
        with self._lock:
            return self._out_of_order_count

    @property
    def last_error(self) -> OSError | None:
        """Unexpected receive error, cleared on a successful new start."""
        with self._lock:
            return self._last_error

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind((self._host, self._port))
                sock.settimeout(0.05)
            except BaseException:
                sock.close()
                raise
            self._stop.clear()
            with self._lock:
                self._address = sock.getsockname()
                self._state = None
                self._last_error = None
                self._running = True
            self._socket = sock
            self._thread = threading.Thread(
                target=self._receive, args=(sock,), name="fh4-telemetry", daemon=True
            )
            try:
                self._thread.start()
            except BaseException:
                sock.close()
                self._socket = None
                self._thread = None
                with self._lock:
                    self._running = False
                    self._address = None
                raise

    def latest(
        self, now_ns: int | None = None, max_age_ns: int = DEFAULT_MAX_AGE_NS
    ) -> VehicleState | None:
        _nonnegative_ns(max_age_ns, "max_age_ns")
        # Read the clock after taking the sample so a concurrent receiver update
        # cannot make the default 'now' older than that sample's timestamp.
        with self._lock:
            state = self._state if self._running else None
            if now_ns is None:
                now_ns = time.monotonic_ns()
        _nonnegative_ns(now_ns, "now_ns")
        if state is None or not 0 <= now_ns - state.timestamp_ns <= max_age_ns:
            return None
        return state

    def close(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            with self._lock:
                self._running = False
            if self._socket is not None:
                self._socket.close()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
                if self._thread.is_alive():
                    raise RuntimeError("telemetry receiver did not stop within one second")
            self._socket = None
            self._thread = None
            with self._lock:
                self._address = None
                self._state = None

    def _receive(self, sock: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                try:
                    # Receive the whole datagram so oversize packets cannot be
                    # truncated to 324 bytes and mistaken for valid FH4 data.
                    data, _ = sock.recvfrom(65535)
                    received_ns = time.monotonic_ns()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not self._stop.is_set():
                        with self._lock:
                            self._last_error = exc
                    break
                try:
                    state = parse_fh4_packet(data, received_ns)
                except ValueError:
                    with self._lock:
                        self._malformed_count += 1
                    continue
                with self._lock:
                    previous = self._state
                    if (
                        previous is not None
                        and received_ns - previous.timestamp_ns <= self._ordering_reset_ns
                    ):
                        delta = (state.game_timestamp_ms - previous.game_timestamp_ms) & 0xFFFFFFFF
                        if not 0 < delta < 0x80000000:
                            self._out_of_order_count += 1
                            continue
                    self._state = state
        finally:
            sock.close()
            with self._lock:
                self._running = False

    def __enter__(self) -> TelemetryReceiver:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
