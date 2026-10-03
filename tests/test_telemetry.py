"""Parser and local UDP integration tests; no game or Windows dependencies."""

import math
import socket
import struct
import time
import unittest

from forza_ai.telemetry import TelemetryReceiver, parse_fh4_packet


def packet(game_ms=1234, speed=12.5, rpm=3500.0, steer=-93, race_on=1):
    data = bytearray(324)
    struct.pack_into("<iI", data, 0, race_on, game_ms)
    struct.pack_into("<f", data, 16, rpm)
    struct.pack_into("<f", data, 256, speed)
    struct.pack_into("<b", data, 320, steer)
    return bytes(data)


class PacketTests(unittest.TestCase):
    def test_known_offsets_units_and_signed_steering(self):
        state = parse_fh4_packet(packet(game_ms=0xFFFFFFFF), 987654321)
        self.assertEqual(state.timestamp_ns, 987654321)
        self.assertEqual(state.game_timestamp_ms, 0xFFFFFFFF)
        self.assertEqual(state.speed_mps, 12.5)
        self.assertEqual(state.rpm, 3500.0)
        self.assertEqual(state.steering_input, -93)
        self.assertIs(state.is_race_on, True)
        self.assertIs(parse_fh4_packet(packet(race_on=0, steer=127), 0).is_race_on, False)

    def test_exact_packet_length_required(self):
        for size in (0, 232, 311, 323, 325, 331, 1024):
            with self.subTest(size=size), self.assertRaises(ValueError):
                parse_fh4_packet(bytes(size), 0)

    def test_invalid_race_flags_rejected(self):
        for flag in (-1, 2, 256):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                parse_fh4_packet(packet(race_on=flag), 0)

    def test_nonfinite_decoded_fields_rejected(self):
        for field in ("speed", "rpm"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    parse_fh4_packet(packet(**{field: value}), 0)

    def test_invalid_host_timestamps_rejected(self):
        for timestamp in (-1, 1.5, True):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                parse_fh4_packet(packet(), timestamp)


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.receiver = TelemetryReceiver(port=0)
        self.receiver.start()
        self.addCleanup(self.receiver.close)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(self.sender.close)

    def send(self, data):
        self.sender.sendto(data, self.receiver.address)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.002)
        self.fail("receiver condition was not reached within two seconds")

    def test_receive_timestamp_freshness_and_cache_identity(self):
        self.assertEqual(self.receiver.address[0], "127.0.0.1")
        self.assertIsNone(self.receiver.latest())
        before = time.monotonic_ns()
        self.send(packet())
        state = self.wait_for(self.receiver.latest)
        self.assertGreaterEqual(state.timestamp_ns, before)
        self.assertLessEqual(state.timestamp_ns, time.monotonic_ns())
        self.assertIs(self.receiver.latest(state.timestamp_ns + 100, 100), state)
        self.assertIsNone(self.receiver.latest(state.timestamp_ns + 101, 100))
        self.assertIsNone(self.receiver.latest(state.timestamp_ns - 1, 100))
        self.assertIs(self.receiver.latest(state.timestamp_ns, 0), state)
        self.assertEqual(state.speed_mps, 12.5)

    def test_malformed_packets_do_not_refresh_good_sample(self):
        self.send(packet())
        state = self.wait_for(self.receiver.latest)
        for data in (b"short", packet() + b"extra", packet(speed=math.nan), packet(race_on=2)):
            self.send(data)
        self.wait_for(lambda: self.receiver.malformed_count == 4)
        self.assertIs(self.receiver.latest(max_age_ns=2_000_000_000), state)
        self.assertIsNone(self.receiver.last_error)

    def test_frame_lookup_uses_past_sample_and_respects_age(self):
        self.send(packet(game_ms=1, speed=10))
        first = self.wait_for(self.receiver.latest)
        frame_time = time.monotonic_ns()
        self.send(packet(game_ms=2, speed=30))
        second = self.wait_for(lambda: (s if (s := self.receiver.latest()) and s.game_timestamp_ms == 2 else None))
        self.assertIs(self.receiver.at_or_before(frame_time), first)
        self.assertIs(self.receiver.at_or_before(second.timestamp_ns), second)
        self.assertIsNone(self.receiver.at_or_before(first.timestamp_ns - 1))
        self.assertIsNone(self.receiver.at_or_before(second.timestamp_ns + 101, max_age_ns=100))
        self.receiver.close()
        self.assertIsNone(self.receiver.at_or_before(frame_time))

    def test_duplicate_backward_and_wrapping_timestamps(self):
        self.send(packet(game_ms=0xFFFFFFFE))
        first = self.wait_for(self.receiver.latest)
        self.send(packet(game_ms=0xFFFFFFFE, speed=80.0))
        self.send(packet(game_ms=0xFFFFFFFD, speed=80.0))
        self.wait_for(lambda: self.receiver.out_of_order_count == 2)
        self.assertIs(self.receiver.latest(max_age_ns=2_000_000_000), first)
        self.send(packet(game_ms=3, speed=25.0))
        latest = self.wait_for(
            lambda: (state if (state := self.receiver.latest()) and state.game_timestamp_ms == 3 else None)
        )
        self.assertEqual(latest.speed_mps, 25.0)
        # A late packet from before wrap must not roll the state backward.
        self.send(packet(game_ms=0xFFFFFFFF, speed=99.0))
        self.wait_for(lambda: self.receiver.out_of_order_count == 3)
        self.assertIs(self.receiver.latest(max_age_ns=2_000_000_000), latest)

    def test_ordering_recovers_after_restart_gap(self):
        self.receiver.close()
        self.receiver = TelemetryReceiver(port=0, ordering_reset_ns=20_000_000)
        self.receiver.start()
        self.addCleanup(self.receiver.close)
        self.send(packet(game_ms=10000))
        self.wait_for(self.receiver.latest)
        time.sleep(0.03)
        self.send(packet(game_ms=1))
        self.wait_for(
            lambda: (state if (state := self.receiver.latest()) and state.game_timestamp_ms == 1 else None)
        )

    def test_close_releases_port_and_restart_drops_old_sample(self):
        self.send(packet())
        self.wait_for(self.receiver.latest)
        address = self.receiver.address
        self.receiver.start()  # Already started: preserve the original socket.
        self.assertEqual(self.receiver.address, address)
        self.receiver.close()
        self.receiver.close()
        self.assertIsNone(self.receiver.latest())
        self.assertIsNone(self.receiver.address)
        self.assertIsNone(self.receiver.last_error)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(address)
        self.receiver.start()
        self.assertIsNone(self.receiver.latest())
        self.send(packet(game_ms=1))
        self.wait_for(self.receiver.latest)

    def test_argument_validation(self):
        for age in (-1, 0.5, True):
            with self.subTest(age=age), self.assertRaises(ValueError):
                self.receiver.latest(max_age_ns=age)
        for now in (-1, 0.5, True):
            with self.subTest(now=now), self.assertRaises(ValueError):
                self.receiver.latest(now_ns=now)
        for port in (-1, 65536, 1.5, True):
            with self.subTest(port=port), self.assertRaises(ValueError):
                TelemetryReceiver(port=port)
        with self.assertRaises(ValueError):
            TelemetryReceiver(ordering_reset_ns=0)

    def test_bind_failure_does_not_leave_live_receiver(self):
        receiver = TelemetryReceiver(port=self.receiver.address[1])
        self.addCleanup(receiver.close)
        with self.assertRaises(OSError):
            receiver.start()
        self.assertIsNone(receiver.latest())
        self.assertIsNone(receiver.address)


if __name__ == "__main__":
    unittest.main()
