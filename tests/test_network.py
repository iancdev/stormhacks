"""Real loopback/socket-pair tests; no Windows, GPU, or trained model needed."""

import math
import socket
import struct
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from forza_ai import network
from forza_ai.contracts import CapturedFrame, ModelObservation, ObservationUnavailable, VehicleState
from forza_ai.inference_server import InferenceServer, main
from forza_ai.network import AuthenticationError, ProtocolError, RemotePolicy


KEY = b"loopback-test-key-not-a-production-secret"


def observation(frame_id=1, pixels=None, speed=21.25):
    if pixels is None:
        pixels = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
    now = time.monotonic_ns()
    return ModelObservation(CapturedFrame(frame_id, now, pixels),
                            VehicleState(now - 10_000_000, speed, True, 10, 2000, 0))


def receive(sock, key=KEY):
    return network._receive_message(sock, key, time.monotonic() + 1)


def send(sock, metadata, payload=b"", key=KEY):
    network._send_message(sock, key, metadata, payload, time.monotonic() + 1)


def request(session, request_id=1):
    png, width, height = network._encode_png(observation().frame.rgb)
    return ({"version": 1, "kind": "predict", "session": session, "request_id": request_id,
             "nonce": "c" * 32, "frame_id": 1, "speed_mps": 21.25, "width": width, "height": height}, png)


class FramingTests(unittest.TestCase):
    def setUp(self):
        self.reader, self.writer = socket.socketpair()
        self.addCleanup(self.reader.close)
        self.addCleanup(self.writer.close)

    def test_authentication_precedes_json_decode(self):
        for wrong_key, tamper in ((b"incorrect", False), (KEY, True)):
            with self.subTest(tamper=tamper):
                encoded = bytearray(network._encode_message(wrong_key, {"version": 1}, b"PNG bytes"))
                if tamper:
                    encoded[-1] ^= 1
                self.writer.sendall(encoded)
                with patch("forza_ai.network.json.loads") as decode:
                    with self.assertRaises(AuthenticationError):
                        receive(self.reader)
                    decode.assert_not_called()

    def test_message_length_rejected_before_body_read(self):
        for length in (0, 37, network.MAX_MESSAGE_BYTES + 1, 0xFFFFFFFF):
            with self.subTest(length=length):
                self.writer.sendall(struct.pack("!I", length))
                with self.assertRaisesRegex(ProtocolError, "length"):
                    receive(self.reader)

    def test_fragmented_read_has_total_deadline(self):
        stop = threading.Event()
        def trickle():
            for _ in range(20):
                if stop.wait(0.01):
                    return
                self.writer.sendall(b"x")
        thread = threading.Thread(target=trickle)
        thread.start()
        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                network._read_exact(self.reader, 20, started + 0.05)
            self.assertLess(time.monotonic() - started, 0.15)
        finally:
            stop.set()
            thread.join(timeout=1)

    def test_truncated_read_is_disconnect(self):
        self.writer.sendall(b"ab")
        self.writer.close()
        with self.assertRaises(ConnectionError):
            network._read_exact(self.reader, 3, time.monotonic() + 1)


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.predictor = Mock()
        self.predictor.predict.return_value = -12.75
        self.server = InferenceServer(self.predictor, port=0, key=KEY)
        self.server.start()
        self.addCleanup(self.server.close)
        self.policy = RemotePolicy(*self.server.address, key=KEY, timeout_s=0.5)
        self.addCleanup(self.policy.close)

    def test_lossless_png_speed_and_original_timestamp(self):
        original = observation()
        original_time = original.timestamp_ns
        self.assertEqual(self.policy.predict(original), -12.75)
        pixels, speed = self.predictor.predict.call_args.args
        np.testing.assert_array_equal(pixels, original.frame.rgb)
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertEqual(speed, 21.25)
        self.assertEqual(original.timestamp_ns, original_time)
        sock, session = self.policy._socket, self.policy._session
        self.assertEqual(self.policy.predict(observation(frame_id=2)), -12.75)
        self.assertIs(self.policy._socket, sock)
        self.assertEqual(self.policy._session, session)
        self.assertEqual(self.policy._request_id, 3)
        self.assertTrue(self.policy.requires_camera)

    def test_wrong_key_fails_authentication(self):
        wrong = RemotePolicy(*self.server.address, key=b"wrong", timeout_s=0.5)
        self.addCleanup(wrong.close)
        with self.assertRaises(AuthenticationError):
            wrong.predict(observation())
        self.assertIsNone(wrong._socket)
        self.predictor.predict.assert_not_called()

    def test_timeout_resets_connection_and_can_reconnect(self):
        entered, release = threading.Event(), threading.Event()
        def slow(pixels, speed):
            entered.set()
            release.wait(timeout=1)
            return 3.0
        self.predictor.predict.side_effect = slow
        self.policy.timeout_s = 0.04
        try:
            with self.assertRaises(ObservationUnavailable):
                self.policy.predict(observation())
            self.assertTrue(entered.is_set())
            self.assertIsNone(self.policy._socket)
        finally:
            release.set()
        self.predictor.predict.side_effect = None
        self.policy.timeout_s = 0.5
        self.assertEqual(self.policy.predict(observation()), -12.75)

    def test_close_interrupts_inflight_receive_without_waiting_for_predict_lock(self):
        entered, release = threading.Event(), threading.Event()
        self.predictor.predict.side_effect = lambda p, s: (entered.set(), release.wait(2), 0.0)[-1]
        self.policy.timeout_s = 5
        result = []
        def predict():
            try:
                self.policy.predict(observation())
            except Exception as error:
                result.append(error)
        worker = threading.Thread(target=predict)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=1))
            started = time.monotonic()
            self.policy.close()
            self.assertLess(time.monotonic() - started, 0.1)
            worker.join(timeout=0.5)
            self.assertFalse(worker.is_alive())
            self.assertIsInstance(result[0], ObservationUnavailable)
            self.policy.close()
            with self.assertRaises(ObservationUnavailable):
                self.policy.predict(observation())
        finally:
            release.set()
            worker.join(timeout=1)

    def test_encoding_counts_against_total_predict_deadline(self):
        encode = network._encode_png
        def slow(pixels):
            time.sleep(0.04)
            return encode(pixels)
        self.policy.timeout_s = 0.02
        with patch("forza_ai.network._encode_png", side_effect=slow):
            with self.assertRaises(ObservationUnavailable):
                self.policy.predict(observation())
        self.predictor.predict.assert_not_called()
        self.assertIsNone(self.policy._socket)

    def test_bad_frames_and_speeds_are_rejected_locally(self):
        images = [np.zeros((12, 20, 3)), np.zeros((12, 20, 4), dtype=np.uint8),
                  np.zeros((0, 20, 3), dtype=np.uint8), np.zeros((1, 3841, 3), dtype=np.uint8)]
        for pixels in images:
            with self.subTest(shape=pixels.shape), self.assertRaises(ProtocolError):
                self.policy.predict(observation(pixels=pixels))
        for speed in (-1, math.inf, math.nan, True):
            with self.subTest(speed=speed), self.assertRaises(ProtocolError):
                self.policy.predict(observation(speed=speed))
        self.predictor.predict.assert_not_called()

    def test_paused_and_noncausal_observations_do_not_leave_host(self):
        good = observation()
        for timestamp, race in ((good.timestamp_ns + 1, True), (good.timestamp_ns - 100_000_001, True),
                                 (good.timestamp_ns, False)):
            vehicle = VehicleState(timestamp, 20.0, race, 0, 0.0, 0)
            with self.subTest(timestamp=timestamp, race=race), self.assertRaises(ObservationUnavailable):
                self.policy.predict(ModelObservation(good.frame, vehicle))
        self.predictor.predict.assert_not_called()

    def test_duplicate_and_previous_connection_requests_rejected(self):
        with socket.create_connection(self.server.address, timeout=1) as sock:
            hello, _ = receive(sock)
            first, png = request(hello["session"])
            send(sock, first, png)
            self.assertEqual(receive(sock)[0]["angle_deg"], -12.75)
            send(sock, first, png)
            with self.assertRaises(ConnectionError):
                receive(sock)
        with socket.create_connection(self.server.address, timeout=1) as sock:
            second, _ = receive(sock)
            self.assertNotEqual(second["session"], hello["session"])
            send(sock, first, png)
            with self.assertRaises(ConnectionError):
                receive(sock)
        self.assertEqual(self.predictor.predict.call_count, 1)

    def test_authenticated_bad_png_rejected_without_inference(self):
        with socket.create_connection(self.server.address, timeout=1) as sock:
            hello, _ = receive(sock)
            metadata, _ = request(hello["session"])
            send(sock, metadata, b"not a PNG")
            with self.assertRaises(ConnectionError):
                receive(sock)
        self.predictor.predict.assert_not_called()

    def test_expired_decode_does_not_launch_model(self):
        decode = network._decode_png
        def slow(payload, width, height):
            time.sleep(0.06)
            return decode(payload, width, height)
        self.server.timeout_s = 0.03
        with patch("forza_ai.inference_server._decode_png", side_effect=slow):
            with self.assertRaises(ObservationUnavailable):
                self.policy.predict(observation())
        self.predictor.predict.assert_not_called()

    def test_server_shutdown_disconnects_and_releases_listener(self):
        self.policy.predict(observation())
        self.server.close()
        self.server.close()
        self.assertIsNone(self.server.address)
        self.assertFalse(self.server.running)
        with self.assertRaises(ObservationUnavailable):
            self.policy.predict(observation())

    def test_invalid_model_angle_does_not_reach_client(self):
        self.predictor.predict.return_value = 451.0
        with self.assertRaises(ObservationUnavailable):
            self.policy.predict(observation())
        self.assertIsInstance(self.server.last_error, ProtocolError)


class ResponseValidationTests(unittest.TestCase):
    def test_previously_signed_hello_and_prediction_cannot_be_replayed(self):
        # Both transcript messages have valid signatures and repeat request/frame
        # IDs from a previous client process; only fresh client material differs.
        hello = network._encode_message(KEY, {"version": 1, "kind": "hello", "session": "a" * 32})
        old = network._encode_message(KEY, {"version": 1, "kind": "prediction", "session": "a" * 32,
                                           "request_id": 1, "nonce": "e" * 32,
                                           "frame_id": 1, "angle_deg": 100.0})
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        errors, requests = [], []
        def replay():
            try:
                with listener.accept()[0] as sock:
                    sock.sendall(hello)
                    requests.append(receive(sock)[0])
                    sock.sendall(old)
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=replay)
        worker.start()
        policy = RemotePolicy(*listener.getsockname(), key=KEY, timeout_s=1)
        try:
            with patch("forza_ai.network.secrets.token_hex", return_value="f" * 32):
                with self.assertRaises(ProtocolError):
                    policy.predict(observation())
            self.assertIsNone(policy._socket)
        finally:
            policy.close()
            listener.close()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(requests[0]["nonce"], "f" * 32)

    def test_wrong_response_ids_session_angle_or_payload_are_fatal(self):
        for change in ({"request_id": 2}, {"session": "b" * 32}, {"nonce": "d" * 32},
                       {"frame_id": 9}, {"angle_deg": 451.0}, {"version": 2}):
            with self.subTest(change=change):
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                errors = []
                def serve():
                    try:
                        with listener.accept()[0] as sock:
                            send(sock, {"version": 1, "kind": "hello", "session": "a" * 32})
                            req, _ = receive(sock)
                            response = {"version": 1, "kind": "prediction", "session": req["session"],
                                        "request_id": req["request_id"], "nonce": req["nonce"],
                                        "frame_id": req["frame_id"], "angle_deg": 0.0}
                            response.update(change)
                            send(sock, response)
                    except Exception as error:
                        errors.append(error)
                thread = threading.Thread(target=serve)
                thread.start()
                policy = RemotePolicy(*listener.getsockname(), key=KEY, timeout_s=1)
                try:
                    with self.assertRaises(ProtocolError):
                        policy.predict(observation())
                    self.assertIsNone(policy._socket)
                finally:
                    policy.close()
                    listener.close()
                    thread.join(timeout=1)
                self.assertFalse(thread.is_alive())
                self.assertFalse(errors)

    def test_missing_key_invalid_config_and_cli_modes(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "FORZA_LINK_KEY"):
                RemotePolicy("127.0.0.1")
        for kwargs in ({"host": "localhost"}, {"host": "127.0.0.1", "port": 0},
                       {"host": "127.0.0.1", "timeout_s": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RemotePolicy(key=KEY, **kwargs)
        with patch.dict("os.environ", {"FORZA_LINK_KEY": KEY.decode()}):
            policy = RemotePolicy("127.0.0.1")
            policy.close()
            with self.assertRaisesRegex(ValueError, "15 degrees"):
                main(["--test-target", "16"])
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as missing:
                main([])
            self.assertEqual(missing.exception.code, 2)
            with self.assertRaises(SystemExit) as both:
                main(["--model", "artifact", "--test-target", "5"])
            self.assertEqual(both.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
