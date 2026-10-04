"""Offline synthetic fixture and Mac CPU overhead measurement; no hardware/weights.

Run PYTHONPATH=src python scripts/saliency_fixture.py. Never starts a listener.
"""
import base64
import io
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from forza_ai.dashboard import _PAGE
from forza_ai.policies.predictor import DrivingPredictor
from forza_ai.policies.steering_model import DrivingModel, Preprocessing
from forza_ai.visualbackprop import ActivationPreview, render_preview

out = Path('artifacts/saliency')
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7)
torch.set_num_threads(1)
p = object.__new__(DrivingPredictor)
p.model, p.preprocessing = DrivingModel().cpu().eval(), Preprocessing()
image = Image.new('RGB', (320, 66), '#758999')
d = ImageDraw.Draw(image)
d.rectangle((0, 24, 319, 65), fill='#527255')
d.polygon([(147, 24), (185, 24), (305, 65), (22, 65)], fill='#343d44')
d.line([(160, 25), (163, 37), (164, 43)], fill='#e6d7a8', width=2)
d.line([(165, 49), (167, 65)], fill='#e6d7a8', width=3)
rgb = np.asarray(image)
for _ in range(8): p.predict(rgb, 20)

def summary(values):
    return dict(median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)), max_ms=max(values))

report = dict(scope='Mac CPU, synthetic image, untrained seeded model; not Windows or driving evidence',
              machine=platform.machine(), python=platform.python_version(), torch=torch.__version__, threads=1)
times = []
for _ in range(64):
    start = time.perf_counter(); baseline = p.predict(rgb, 20); times.append((time.perf_counter()-start)*1000)
report['disabled'] = summary(times)
preview = ActivationPreview(p)
try:
    times = []
    for _ in range(64):
        preview.begin(None)
        start = time.perf_counter(); result = p.predict(rgb, 20); times.append((time.perf_counter()-start)*1000)
        assert result == baseline
    report['enabled_unsampled'] = summary(times)
    times = []
    # Force capture eligibility only for measuring snapshot overhead. Do not
    # submit to renderer, so this is not claimed as sustained 1Hz performance.
    for i in range(64):
        preview._next_capture = 0
        preview.begin(dict(session='a'*32, request_id=i+1, frame_id=i))
        start = time.perf_counter(); result = p.predict(rgb,20); times.append((time.perf_counter()-start)*1000)
        assert result == baseline
    report['sampled_forward_only'] = summary(times)
    snapshot = preview._capture
    times = []
    for _ in range(32):
        start=time.perf_counter(); png,active=render_preview(snapshot['image'],snapshot['maps']);times.append((time.perf_counter()-start)*1000)
    report['independent_render'] = summary(times)
    report['png_bytes'] = len(png)
    report['copied_snapshot_bytes'] = snapshot['image'].nbytes+sum(m.nbytes for m in snapshot['maps'])
    report['prediction_parity'] = 'All 128 enabled forwards exactly equal disabled prediction'
finally:
    preview.close()
(out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'synthetic-composite.png').write_bytes(png)
(out/'fixture.json').write_text(json.dumps(dict(state='ready',frame_id=42,request_id=18,age_ms=80,
    prediction=dict(angle_deg=baseline.angle_deg,throttle=baseline.throttle,brake=baseline.brake),active=active,
    png=base64.b64encode(png).decode())))
page=_PAGE.replace('__TOKEN__','synthetic-fixture-token-not-a-secret').replace('<body>','<body><div style="padding:14px;text-align:center;background:#47381c;color:#ffe1a3;font:13px system-ui">SIMULATED UI FIXTURE · SYNTHETIC IMAGE · UNTRAINED MODEL · NO HARDWARE</div>')
(out/'fixture.html').write_text(page)
print(json.dumps(report,indent=2))
