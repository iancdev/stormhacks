"""Opt-in, CPU-only positive-ELU VisualBackProp diagnostic; never a controller.

Adaptation of Bojarski et al., arXiv:1611.05418 section 3: average positive
post-activation maps, all-one transposed convolutions, multiply shallower maps.
The original uses ReLU. Here negative ELU evidence is explicitly omitted.
No gradients, additional model passes, output-specific or causal attribution.
"""
import io
import math
import threading
import time

import numpy as np
from PIL import Image

MAX_PREVIEW_BYTES = 100_000
SHAPES = ((66, 200), (31, 98), (14, 47), (5, 22), (3, 20), (1, 18))
KERNELS = ((5, 2), (5, 2), (5, 2), (3, 1), (3, 1))


def visualbackprop(maps):
    if len(maps) != 5:
        raise ValueError('five activation maps required')
    for value, shape in zip(maps, SHAPES[1:]):
        if value.shape != shape or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError('invalid positive activation map')
    mask = maps[-1].astype(np.float64)
    for index in range(4, -1, -1):
        kernel, stride = KERNELS[index]
        target = np.zeros(SHAPES[index], dtype=np.float64)
        height, width = mask.shape
        for y in range(kernel):
            for x in range(kernel):
                target[y:y + stride * height:stride, x:x + stride * width:stride] += mask
        mask = target * maps[index - 1] if index else target
        # Positive scalar rescaling preserves the final normalized result and
        # prevents overflow/underflow through the five multiplicative stages.
        maximum = mask.max()
        if maximum:
            mask /= maximum
    low, high = mask.min(), mask.max()
    return ((mask - low) / (high - low) if high > low else np.zeros_like(mask)).astype(np.float32)


def render_preview(image, maps):
    """Lossless composite: actual normalized input inverted for display, overlay."""
    if image.shape != (3, 66, 200) or not np.isfinite(image).all():
        raise ValueError('invalid model input')
    rgb = np.rint((image.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    mask = visualbackprop(maps)
    # Dimmed grayscale road with green activation, so highlights don't blend into road colours.
    gray = (rgb @ np.array([.299, .587, .114]))[..., None] * .55
    alpha = (mask ** .7 * .85)[..., None]
    overlay = np.rint(gray * (1 - alpha) + np.array([40, 255, 90]) * alpha).clip(0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(np.concatenate((rgb, overlay), axis=1)).save(output, format='PNG', compress_level=1)
    png = output.getvalue()
    if len(png) > MAX_PREVIEW_BYTES:
        raise ValueError('preview exceeds fixed budget')
    return png, bool(mask.max() > 0)


class ActivationPreview:
    """One in-progress render and one replaceable snapshot; configurable sampling, zero means uncapped.

    Hooks copy only one input plus five reduced maps during an existing forward.
    Rendering/PNG runs on a daemon independently; inference never waits for it.
    begin/finish belong exclusively to the existing model worker. No hooks exist
    unless explicitly constructed. CPU PilotNet only, including wrapped inputs.
    """
    def __init__(self, predictor, *, hz=10.0):
        if isinstance(hz, bool) or not math.isfinite(hz) or hz < 0:
            raise ValueError("saliency Hz must be finite and nonnegative (0 = uncapped)")
        self.interval = 1.0 / hz if hz else 0.0
        self._last_frame = None
        from torch import nn
        # Input-adjustment wrappers may expose their underlying predictor here.
        core = predictor
        while not hasattr(core, 'model') and hasattr(core, 'predictor'):
            core = core.predictor
        model = getattr(core, 'model', None)
        if model is None or not hasattr(model, 'encoder'):
            raise ValueError('saliency requires a CPU PilotNet predictor')
        layers = list(model.encoder)
        if (len(layers) != 11 or next(model.parameters()).device.type != 'cpu'
                or any(not isinstance(layers[i * 2], nn.Conv2d) or
                       not isinstance(layers[i * 2 + 1], nn.ELU) or
                       layers[i * 2].kernel_size != (k, k) or
                       layers[i * 2].stride != (s, s)
                       for i, (k, s) in enumerate(KERNELS))):
            raise ValueError('unsupported saliency encoder')
        self._condition = threading.Condition()
        self._pending = self._latest = self._capture = None
        self._next_capture = 0.0
        self._closed = False
        self._hooks = [model.register_forward_pre_hook(self._input)]
        self._hooks += [layers[i].register_forward_hook(self._activation) for i in (1, 3, 5, 7, 9)]
        self._thread = threading.Thread(target=self._render, name='activation-preview', daemon=True)
        self._thread.start()

    def begin(self, identity):
        self._capture = None
        now = time.monotonic()
        if (identity is not None and now >= self._next_capture and not self._closed
                and (self._last_frame is None or identity["session"] != self._last_frame[0]
                     or identity["frame_id"] > self._last_frame[1])):
            self._last_frame = (identity["session"], identity["frame_id"])
            self._next_capture = now + self.interval
            self._capture = dict(identity=identity, started=now, maps=[])

    def _input(self, module, args):
        if self._capture is not None:
            try:
                self._capture['image'] = args[0][0].detach().numpy().copy()
            except Exception:
                self._capture = None  # A diagnostic must not fail a prediction.

    def _activation(self, module, args, output):
        if self._capture is not None:
            try:
                value = output[0].detach().numpy()
                self._capture['maps'].append(np.maximum(value, 0).mean(axis=0))
            except Exception:
                self._capture = None

    def finish(self, prediction=None):
        snapshot, self._capture = self._capture, None
        if snapshot is None or prediction is None or 'image' not in snapshot or len(snapshot['maps']) != 5:
            return
        if hasattr(prediction, 'angle_deg'):
            snapshot['prediction'] = dict(angle_deg=prediction.angle_deg, throttle=prediction.throttle, brake=prediction.brake)
        else:
            snapshot['prediction'] = dict(angle_deg=float(prediction))
        # Never wait behind the renderer or a dashboard consumer.
        if self._condition.acquire(blocking=False):
            try:
                if not self._closed:
                    self._pending = snapshot
                    self._condition.notify()
            finally:
                self._condition.release()

    def _render(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                snapshot, self._pending = self._pending, None
            if time.monotonic() - snapshot['started'] > .5:
                continue
            try:
                png, active = render_preview(snapshot['image'], snapshot['maps'])
                result = (snapshot, png, active)
            except Exception:
                continue
            with self._condition:
                if not self._closed and time.monotonic() - snapshot["started"] <= .5:
                    self._latest = result

    def latest(self, session, after_id):
        if not self._condition.acquire(blocking=False):
            return None, b''
        try:
            result = self._latest
        finally:
            self._condition.release()
        if result is None:
            return None, b''
        snapshot, png, active = result
        identity = snapshot['identity']
        age_ms = (time.monotonic() - snapshot['started']) * 1000
        if identity['session'] != session or identity['request_id'] <= after_id or age_ms > 500:
            return None, b''
        metadata = dict(identity, age_ms=age_ms, active=active, prediction=snapshot['prediction'])
        return metadata, png

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        with self._condition:
            self._closed = True
            self._pending = self._latest = None
            self._condition.notify_all()
        self._thread.join(timeout=1)
