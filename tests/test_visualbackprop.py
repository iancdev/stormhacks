import base64
import io
import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from forza_ai.visualbackprop import ActivationPreview, SHAPES, visualbackprop, render_preview
from forza_ai.policies.predictor import DrivingPredictor
from forza_ai.policies.steering_model import DrivingModel, Preprocessing, preprocess_rgb
from forza_ai.network import RemotePolicy, ProtocolError
from forza_ai.preview_transport import validate_preview
from forza_ai.inference_server import InferenceServer
from forza_ai.dashboard import Dashboard
from forza_ai.contracts import CapturedFrame, ModelObservation, VehicleState

KEY = b'saliency-synthetic-test-only'


def predictor():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    p = object.__new__(DrivingPredictor)
    p.model = DrivingModel().cpu().eval()
    p.preprocessing = Preprocessing(crop=(.1, .1, .9, .9))
    return p


def observation(index=1):
    now = time.monotonic_ns()
    rgb = np.random.default_rng(8).integers(0, 256, (90, 320, 3), dtype=np.uint8)
    return ModelObservation(CapturedFrame(index, now, rgb), VehicleState(now - 1_000_000, 20, True, 10, 2000, 0))


def wait_for(fn):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = fn()
        if result:
            return result
        time.sleep(.005)
    raise AssertionError('diagnostic not ready')


def test_transposed_convolution_matches_torch_reference():
    rng = np.random.default_rng(4)
    maps = [rng.random(shape).astype(np.float32) for shape in SHAPES[1:]]
    result = visualbackprop(maps)
    value = torch.tensor(maps[-1], dtype=torch.float64)[None, None]
    for i, (k, s) in reversed(list(enumerate(((5, 2), (5, 2), (5, 2), (3, 1), (3, 1))))):
        target = SHAPES[i]
        padding = tuple(target[d] - ((value.shape[d + 2] - 1) * s + k) for d in range(2))
        value = torch.nn.functional.conv_transpose2d(value, torch.ones((1, 1, k, k), dtype=torch.float64), stride=s, output_padding=padding)
        if i:
            value *= torch.tensor(maps[i - 1])
    value = (value - value.min()) / (value.max() - value.min())
    np.testing.assert_allclose(result, value[0, 0].numpy(), atol=1e-7)


def test_zero_maps_no_nan_no_fake_heatmap():
    maps = [np.zeros(shape, np.float32) for shape in SHAPES[1:]]
    assert not visualbackprop(maps).any()
    image = np.zeros((3, 66, 200), np.float32)
    png, active = render_preview(image, maps)
    pixels = np.array(Image.open(io.BytesIO(png)))
    assert not active
    np.testing.assert_array_equal(pixels[:, :200], pixels[:, 200:])
    maps[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        visualbackprop(maps)


def test_exact_prediction_input_and_hooks_removed():
    p, obs = predictor(), observation()
    baseline = p.predict(obs.frame.rgb, 20)
    preview = ActivationPreview(p, hz=1)
    try:
        preview.begin(dict(session='a' * 32, request_id=1, frame_id=9))
        result = p.predict(obs.frame.rgb, 20)
        preview.finish(result)
        assert result == baseline
        metadata, png = wait_for(lambda: (v if (v := preview.latest('a' * 32, 0))[0] else None))
        expected = np.rint((preprocess_rgb(obs.frame.rgb, p.preprocessing).numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(png)))[:, :200], expected)
        assert metadata['prediction']['angle_deg'] == result.angle_deg
        assert preview.latest('b' * 32, 0) == (None, b'')
        assert preview.latest('a' * 32, 1) == (None, b'')
        preview.begin(dict(session='a' * 32, request_id=2, frame_id=10))
        assert preview._capture is None  # rate limited
        preview._latest[0]['started'] -= 3
        assert preview.latest('a' * 32, 0) == (None, b'')
    finally:
        preview.close()
    assert not p.model._forward_pre_hooks
    assert all(not m._forward_hooks for m in p.model.encoder)
    assert p.predict(obs.frame.rgb, 20) == baseline


def test_wrapped_adjustment_is_part_of_actual_input():
    p, obs = predictor(), observation()
    class Wrapper:
        def __init__(self): self.predictor = p
        def predict(self, rgb, speed): return p.predict(255 - rgb, speed)
    wrapped = Wrapper()
    preview = ActivationPreview(wrapped)
    try:
        preview.begin(dict(session='a' * 32, request_id=1, frame_id=1))
        result = wrapped.predict(obs.frame.rgb, 20)
        captured = preview._capture['image']
        np.testing.assert_array_equal(captured, preprocess_rgb(255 - obs.frame.rgb, p.preprocessing).numpy())
        preview.finish(result)
    finally:
        preview.close()


def test_renderer_does_not_block_prediction_and_pending_is_single(monkeypatch):
    import threading
    import forza_ai.visualbackprop as module
    entered, release = threading.Event(), threading.Event()
    render = module.render_preview
    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return render(*args)
    monkeypatch.setattr(module, 'render_preview', blocked)
    p, obs = predictor(), observation()
    preview = ActivationPreview(p)
    try:
        for i in range(3):
            preview._next_capture = 0
            preview.begin(dict(session='a' * 32, request_id=i + 1, frame_id=i))
            preview.finish(p.predict(obs.frame.rgb, 20))
            if i == 0:
                assert entered.wait(1)
        assert preview._pending['identity']['request_id'] == 3
        assert preview.latest('a' * 32, 0) == (None, b'')
    finally:
        release.set()
        preview.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_authenticated_roundtrip_backward_compat_and_stale(enabled):
    p = predictor()
    server = InferenceServer(p, port=0, key=KEY, saliency=enabled)
    server.start()
    client = RemotePolicy(*server.address, key=KEY, driving=True, timeout_s=2, saliency=enabled)
    legacy = None
    try:
        obs = observation()
        assert client.predict(obs) == p.predict(obs.frame.rgb, 20)
        if enabled:
            wait_for(lambda: server.preview._latest)
            assert client.predict(observation(2)) == p.predict(obs.frame.rgb, 20)
            snap = client.saliency_snapshot()
            assert snap['state'] == 'ready' and snap['frame_id'] == 1
            assert snap['request_id'] == 1
            preview, png, frame_ns, received = client._preview
            client._preview = (preview, png, frame_ns - 600_000_000, received)
            assert client.saliency_snapshot()['state'] == 'stale'
            assert 'png' not in client.saliency_snapshot()
        else:
            assert server.preview is None
            assert not p.model._forward_pre_hooks
            assert client.saliency_snapshot() == {'state': 'disabled'}
        client.close()
        assert client._preview is None
        # Old requests still get the exact old envelope even with server enabled.
        legacy = RemotePolicy(*server.address, key=KEY, driving=True, timeout_s=2)
        assert legacy.predict(observation(3)) == p.predict(obs.frame.rgb, 20)
    finally:
        client.close()
        if legacy: legacy.close()
        server.close()


def test_preview_validation_rejects_unbound_and_oversize():
    maps = [np.ones(shape, np.float32) for shape in SHAPES[1:]]
    png, active = render_preview(np.zeros((3, 66, 200)), maps)
    meta = dict(session='a' * 32, request_id=1, frame_id=1, age_ms=10., active=active, prediction={'angle_deg': 1.})
    validate_preview(meta, png, 'a' * 32, 1)
    for change in ({'session': 'b' * 32}, {'request_id': 2}, {'age_ms': float('nan')}, {'prediction': {'angle_deg': 999}}, {'active': 1}):
        with pytest.raises(ValueError): validate_preview(meta | change, png, 'a' * 32, 1)
    for bad in (b'', b'x' * 100001):
        with pytest.raises(ValueError): validate_preview(meta, bad, 'a' * 32, 1)
    with pytest.raises(ValueError): validate_preview(None, png, 'a' * 32, 1)


def test_dashboard_endpoint_respects_host_and_no_controls_added():
    import urllib.request
    import urllib.error
    import json
    d = Dashboard(queue.Queue(), port=0, read_only=True, saliency_source=lambda: {'state': 'waiting'})
    d.start()
    try:
        assert json.load(urllib.request.urlopen(d.url + '/api/saliency')) == {'state': 'waiting'}
        req = urllib.request.Request(d.url + '/api/saliency', headers={'Host': 'evil.example'})
        with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(req)
        assert error.value.code == 403
        req = urllib.request.Request(d.url + '/api/saliency', data=b'{}', method='POST')
        with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(req)
        assert error.value.code == 404
    finally:
        d.close()


def test_preview_server_can_restart_without_orphan_hooks():
    p = predictor()
    server = InferenceServer(p, port=0, key=KEY, saliency=True, saliency_hz=0)
    for _ in range(2):
        server.start()
        assert len(p.model._forward_pre_hooks) == 1
        assert server.preview.interval == 0
        server.close()
        assert len(p.model._forward_pre_hooks) == 0
        assert not server.preview._thread.is_alive()


def test_opted_client_disabled_server_waits_without_extra_model_work():
    p = predictor()
    server = InferenceServer(p, port=0, key=KEY)
    server.start()
    client = RemotePolicy(*server.address, key=KEY, driving=True, timeout_s=2, saliency=True)
    try:
        result = client.predict(observation())
        assert result == p.predict(observation().frame.rgb, 20)
        assert client.saliency_snapshot() == {'state': 'waiting'}
        assert not p.model._forward_pre_hooks
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize('hz', [0, 5, 10, 120])
def test_capture_cadence_and_unique_frames(monkeypatch, hz):
    import forza_ai.visualbackprop as module
    preview = ActivationPreview(predictor(), hz=hz)
    clock = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    def begin(request, frame, session='a' * 32):
        preview.begin(dict(session=session, request_id=request, frame_id=frame))
        return preview._capture
    try:
        assert begin(1, 1) is not None
        clock[0] += .001
        assert (begin(2, 2) is not None) == (hz == 0)
        clock[0] += 1
        assert begin(3, 3) is not None
        clock[0] += 1
        assert begin(4, 3) is None  # duplicate source frame, even with a new request
        assert begin(5, 1) is None  # no resampling an older source frame
        assert begin(1, 3, 'b' * 32) is not None  # a new authenticated session
    finally:
        preview.close()


@pytest.mark.parametrize('hz', [-1, float('nan'), float('inf'), True])
def test_invalid_rate_rejected_before_hooks(hz):
    p = predictor()
    with pytest.raises(ValueError): ActivationPreview(p, hz=hz)
    with pytest.raises(ValueError): InferenceServer(p, key=KEY, saliency_hz=hz)
    assert not p.model._forward_pre_hooks


def test_slow_renderer_drops_expired_result(monkeypatch):
    import threading
    import forza_ai.visualbackprop as module
    entered, release = threading.Event(), threading.Event()
    render = module.render_preview
    def slow(*args):
        entered.set()
        assert release.wait(3)
        return render(*args)
    monkeypatch.setattr(module, 'render_preview', slow)
    p, obs = predictor(), observation()
    preview = ActivationPreview(p, hz=0)
    try:
        preview.begin(dict(session='a' * 32, request_id=1, frame_id=1))
        snapshot = preview._capture
        preview.finish(p.predict(obs.frame.rgb, 20))
        assert entered.wait(1)
        snapshot['started'] -= 1
        release.set()
        # A subsequent fresh render proves the worker passed the expired result.
        preview.begin(dict(session='a' * 32, request_id=2, frame_id=2))
        preview.finish(p.predict(obs.frame.rgb, 20))
        result = wait_for(lambda: preview.latest('a' * 32, 0)[0])
        assert result['request_id'] == 2
    finally:
        release.set()
        preview.close()
