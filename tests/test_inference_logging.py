"""Real loopback diagnostics without model files, GPU or hardware."""
import json
import socket
import threading
import time
from unittest.mock import Mock

import pytest

from forza_ai.inference_logging import InferenceLog
from forza_ai.inference_server import InferenceServer
from forza_ai.network import RemotePolicy
from test_network import KEY, observation, receive, request, send


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate()


@pytest.fixture
def running():
    lines = []
    log = InferenceLog(sink=lines.append)
    predictor = Mock()
    predictor.predict.return_value = 12.5
    server = InferenceServer(predictor, port=0, key=KEY, timeout_s=.25, event_log=log)
    server.start()
    yield server, predictor, lines
    server.close()
    log.close()


def close_record(lines):
    wait_for(lambda: any(json.loads(line)['event'] == 'connection_close' for line in lines))
    return next(json.loads(line) for line in lines if json.loads(line)['event'] == 'connection_close')


def test_success_counts_no_per_frame_logging_or_sensitive_data(running):
    server, predictor, lines = running
    policy = RemotePolicy(*server.address, key=KEY, timeout_s=1)
    for frame in range(5):
        assert policy.predict(observation(frame)) == 12.5
    policy.close()
    record = close_record(lines)
    assert record['responses'] == record['requests'] == record['completed_model_calls'] == 5
    assert record['authenticated'] is True
    assert record['model_mean_ms'] >= 0 and record['model_max_ms'] >= record['model_mean_ms']
    assert record['cause'] == 'peer_eof_or_reset'
    assert [json.loads(line)['event'] for line in lines] == [
        'connection_open', 'authenticated_protocol_ready', 'first_prediction_sent', 'connection_close']
    assert all(json.loads(line)['utc'].endswith('+00:00') for line in lines)
    serialized = '\n'.join(lines)
    for secret in (KEY.decode(), 'nonce', 'session', 'payload', 'angle_deg', 'speed_mps'):
        assert secret not in serialized


def test_bad_authentication_is_distinct_from_protocol(running):
    server, predictor, lines = running
    with socket.create_connection(server.address) as client:
        hello, _ = receive(client)
        metadata, png = request(hello['session'])
        send(client, metadata, png, key=b'wrong-key-PRIVATE')
        record = close_record(lines)
    assert record['cause'] == 'authentication_failed'
    assert record['authenticated'] is False
    predictor.predict.assert_not_called()
    assert 'PRIVATE' not in '\n'.join(lines)


def test_authenticated_version_failure(running):
    server, predictor, lines = running
    with socket.create_connection(server.address) as client:
        hello, _ = receive(client)
        metadata, png = request(hello['session'])
        metadata['version'] = 99
        send(client, metadata, png)
        record = close_record(lines)
    assert record['cause'] == 'protocol_error'
    assert record['phase'] == 'protocol'
    assert record['authenticated'] is True
    predictor.predict.assert_not_called()


@pytest.mark.parametrize('partial,expected', [(False, 'idle_timeout'), (True, 'timeout')])
def test_idle_versus_partial_request_deadline(running, partial, expected):
    server, _, lines = running
    with socket.create_connection(server.address) as client:
        receive(client)
        if partial:
            client.sendall(b'\0')
        record = close_record(lines)
    assert record['phase'] == 'receive'
    assert record['cause'] == expected
    assert record['received_bytes'] == int(partial)


def test_model_exception_text_is_never_logged(running):
    server, predictor, lines = running
    predictor.predict.side_effect = RuntimeError('PRIVATE KEY payload image secret')
    policy = RemotePolicy(*server.address, key=KEY, timeout_s=1)
    try:
        with pytest.raises(Exception):
            policy.predict(observation())
    finally:
        policy.close()
    record = close_record(lines)
    assert record['cause'] == 'model_error' and record['phase'] == 'model'
    assert 'PRIVATE' not in '\n'.join(lines)
    assert server.running


def test_bad_png_reports_decode_phase_without_payload(running):
    server, predictor, lines = running
    with socket.create_connection(server.address) as client:
        hello, _ = receive(client)
        metadata, _ = request(hello['session'])
        send(client, metadata, b'PRIVATE IMAGE DATA')
        record = close_record(lines)
    assert record['cause'] == 'decode_error' and record['phase'] == 'decode'
    assert record['authenticated'] is True
    predictor.predict.assert_not_called()
    assert 'PRIVATE' not in '\n'.join(lines)


def test_stalled_model_deadline_and_busy_connection_are_distinct(running):
    server, predictor, lines = running
    entered, release = threading.Event(), threading.Event()
    def blocked(*args):
        entered.set()
        release.wait(2)
        return 0
    predictor.predict.side_effect = blocked
    try:
        with socket.create_connection(server.address) as client:
            hello, _ = receive(client)
            metadata, png = request(hello['session'])
            send(client, metadata, png)
            assert entered.wait(1)
            record = close_record(lines)
        assert record['cause'] == 'timeout' and record['phase'] == 'model'
        with socket.create_connection(server.address) as client:
            hello, _ = receive(client)
            metadata, png = request(hello['session'])
            send(client, metadata, png)
            wait_for(lambda: sum(json.loads(line)['event'] == 'connection_close' for line in lines) == 2)
        record = [json.loads(line) for line in lines if json.loads(line)['event'] == 'connection_close'][-1]
        assert record['cause'] == 'busy' and record['phase'] == 'model_wait'
        assert predictor.predict.call_count == 1
    finally:
        release.set()
        wait_for(lambda: server._job is None)


def test_blocked_sink_and_queue_overflow_do_not_block_predictions():
    entered, release = threading.Event(), threading.Event()
    def blocked(line):
        entered.set()
        release.wait(3)
    log = InferenceLog(sink=blocked, capacity=1)
    log.emit('test')
    assert entered.wait(1)
    predictor = Mock()
    predictor.predict.return_value = 3.0
    server = InferenceServer(predictor, port=0, key=KEY, event_log=log)
    server.start()
    policy = RemotePolicy(*server.address, key=KEY, timeout_s=.5)
    try:
        assert policy.predict(observation()) == 3
        assert log.dropped > 0
    finally:
        policy.close()
        server.close()
        release.set()
        log.close()


def test_broken_logger_does_not_break_server():
    broken = Mock()
    broken.emit.side_effect = RuntimeError('sink failed')
    predictor = Mock()
    predictor.predict.return_value = 7.0
    server = InferenceServer(predictor, port=0, key=KEY, event_log=broken)
    server.start()
    policy = RemotePolicy(*server.address, key=KEY, timeout_s=.5)
    try:
        assert policy.predict(observation()) == 7
    finally:
        policy.close()
        server.close()


def test_writer_survives_failed_output_and_drops_unknown_fields():
    output = []
    def flaky(line):
        if not output:
            output.append('failed')
            raise OSError('private output error')
        output.append(line)
    log = InferenceLog(sink=flaky)
    try:
        log.emit('first', key='PRIVATE', payload=b'PRIVATE')
        log.emit('second', cause='timeout', exception=ValueError('PRIVATE'))
        wait_for(lambda: len(output) == 2)
        assert log.dropped == 1
        assert json.loads(output[1])['cause'] == 'timeout'
        assert 'PRIVATE' not in output[1]
    finally:
        log.close()
