import csv
from dataclasses import replace
import numpy as np
from PIL import Image
import pytest
import torch

from forza_ai.data.synthetic import generate
from forza_ai.data.sessions import load_sessions, load_session
from forza_ai.data.dataset import SteeringDataset, cache_keys_for_sessions, estimate_cache_bytes
from forza_ai.policies.steering_model import Preprocessing, DrivingModel
from forza_ai.policies.predictor import DrivingPredictor, SteeringPredictor, load_predictor
from forza_ai.training.engine import TrainConfig, train, evaluate, export, load_checkpoint


def set_pedals(path):
    with path.open() as f:
        reader = csv.DictReader(f)
        names, rows = reader.fieldnames, list(reader)
    for i, row in enumerate(rows):
        row.update(throttle=str(.6 if i % 3 == 0 else 0), brake=str(.7 if i % 3 == 1 else 0))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)


def test_driving_train_resume_cache_export_parity(tmp_path):
    torch.set_num_threads(1)
    data = generate(tmp_path/'data', frames=6)
    for path in data.glob('*/wheel.csv'):
        set_pedals(path)
    config = TrainConfig(batch_size=3, task='driving', cache_mib=0)
    a, b = tmp_path/'resumed', tmp_path/'continuous'
    train(data, a, epochs=1, config=config, device='cpu')
    train(data, a, epochs=2, resume=a/'last.pt', device='cpu')
    train(data, b, epochs=2, config=replace(config, cache_mib=64), device='cpu')
    ca, cb = load_checkpoint(a/'last.pt'), load_checkpoint(b/'last.pt')
    assert ca['format_version'] == 2
    assert ca['history'] == cb['history']
    for k, v in ca['model_state'].items():
        assert torch.isfinite(v).all()
        torch.testing.assert_close(v, cb['model_state'][k], rtol=0, atol=0)
    assert set(ca['train_sessions']).isdisjoint(ca['validation_sessions'])
    assert {'throttle','brake'} <= evaluate(a/'last.pt', data).keys()
    model = DrivingModel(); model.load_state_dict(ca['model_state']); model.eval()
    sample = load_sessions(data)[0].samples[1]
    dataset = SteeringDataset(load_sessions(data), Preprocessing(), task='driving')
    image, speed, label = dataset[1]
    assert label[1:].tolist() == pytest.approx([0, .7])
    with torch.inference_mode():
        expected = model(image[None], speed[None])[0].tolist()
    artifact = tmp_path/'export'; metadata = export(a/'last.pt', artifact)
    predictor = load_predictor(artifact)
    assert isinstance(predictor, DrivingPredictor)
    prediction = predictor.predict(np.asarray(Image.open(sample.image_path).convert('RGB')), sample.speed_mps)
    assert [prediction.angle_deg/450, prediction.throttle, prediction.brake] == pytest.approx(expected)
    assert prediction.throttle * prediction.brake == 0
    assert metadata['output']['fields'] == ['angle_deg','throttle','brake']
    with pytest.raises(ValueError):
        SteeringPredictor(artifact)
    loss = model(image[None], speed[None]).sum(); loss.backward()
    assert model.head[-2].weight.grad is not None
    assert torch.isfinite(model.head[-2].weight.grad).all()
    assert (model.head[-2].weight.grad.abs().sum(dim=1) > 0).all()


def test_pedal_labels_and_cache_identity(tmp_path):
    data = generate(tmp_path/'data', frames=4)
    sessions = load_sessions(data)
    p = Preprocessing()
    assert sessions[0].samples[0].throttle == .4
    steering = cache_keys_for_sessions(sessions, p)
    driving = cache_keys_for_sessions(sessions, p, task='driving')
    assert steering != driving
    n = sum(len(s.samples) for s in sessions)
    assert estimate_cache_bytes(sessions,p,task='driving') - estimate_cache_bytes(sessions,p) == 8*n
    sessions[0].samples[0] = replace(sessions[0].samples[0], throttle=.9, brake=.2)
    assert driving != cache_keys_for_sessions(sessions,p,task='driving')
    label = SteeringDataset(sessions,p,task='driving')[0][2]
    assert label[1:].tolist() == pytest.approx([0,.2])
    sessions[0].samples[0] = replace(sessions[0].samples[0], throttle=None)
    with pytest.raises(ValueError, match='human throttle/brake'):
        SteeringDataset(sessions,p,task='driving')


def test_pedals_interpolate_at_label_time(tmp_path):
    data=generate(tmp_path/'data',frames=4)
    path=data/'synthetic-000'
    set_pedals(path/'wheel.csv')
    with (path/'frames.csv').open() as f:
        reader=csv.DictReader(f); names,rows=reader.fieldnames,list(reader)
    rows[0]['capture_time_ns']='1016666666'
    with (path/'frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)
    sample=load_session(path).samples[0]
    assert sample.throttle == pytest.approx(.3)
    assert sample.brake == pytest.approx(.35)
