from dataclasses import replace
import queue
import time
from unittest.mock import patch
import pytest

from forza_ai.contracts import DrivingPrediction, ControlMode, ControlStatus, WheelState, SteeringCommand, ObservationUnavailable, ActuationExpired
from forza_ai.inference_server import InferenceServer
from forza_ai.network import RemotePolicy, ProtocolError
from forza_ai.runtime import run, virtual_driving_state
from forza_ai.simulation import SimulatedAdapter
from test_network import observation
from test_live_runtime import FakeCamera, FakeTelemetry
from test_hardware import Rig


class DrivingPolicy:
    driving = True
    def predict(self, observation):
        return DrivingPrediction(5, .6, 0)


class Adapter(SimulatedAdapter):
    def __init__(self, pedal=None, events=None):
        super().__init__(); self.writes=[]; self.pedal=pedal; self.reads=0; self.events=events
    def read_state(self, now):
        self.reads += 1
        state=super().read_state(now)
        if self.events and self.reads == 30:
            self.events.put('arm')
        if self.pedal and 12 <= self.reads < 20:
            return replace(state, **{self.pedal:.5})
        return state
    def write_virtual_state(self,state):
        super().write_virtual_state(state);self.writes.append((self.reads,state))


@pytest.mark.parametrize('pedal',['throttle','brake'])
def test_auto_pedals_human_override_latches_and_explicit_rearm(pedal):
    events=queue.Queue();adapter=Adapter(pedal,events)
    summary=run(adapter,DrivingPolicy(),duration=.5,assist=True,auto_pedals=True,direct_vjoy=True,command_queue=events)
    assert any(s.throttle == .6 for i,s in adapter.writes if i < 12)
    assert all(getattr(s,pedal) == .5 for i,s in adapter.writes if 12 <= i < 20)
    assert all(s.throttle == s.brake == 0 for i,s in adapter.writes if 20 <= i < 30)
    assert any(s.throttle == .6 for i,s in adapter.writes if i > 32)
    assert summary['max_abs_torque']==0


def test_shadow_and_manual_pedal_compatibility():
    for kwargs in ({'auto_pedals':True,'shadow':True},{'auto_pedals':False,'assist':True}):
        a=Adapter(); run(a,DrivingPolicy(),duration=.1,**kwargs)
        assert all(s.throttle == s.brake == 0 for _,s in a.writes)
    from forza_ai.policies.placeholder import FixedAnglePolicy
    with pytest.raises(ValueError,match='v2 driving'):
        run(Adapter(),FixedAnglePolicy(0),auto_pedals=True)


def test_pause_releases_pedals_and_does_not_rearm():
    p=DrivingPolicy();p.requires_camera=True
    a=Adapter();r=run(a,p,duration=.35,assist=True,auto_pedals=True,direct_vjoy=True,
                     camera=FakeCamera(),receiver=FakeTelemetry(pause_after=.15))
    assert any(s.throttle == .6 for _,s in a.writes)
    assert a.writes[-1][1].throttle == a.writes[-1][1].brake == 0
    assert r['mode']=='takeover'


def test_transport_driving_roundtrip_and_version_mismatch():
    class Predictor:
        driving=True
        def predict(self,pixels,speed): return DrivingPrediction(-20,0,.7)
    server=InferenceServer(Predictor(),port=0,key=b'test');server.start()
    p=RemotePolicy(*server.address,key=b'test',driving=True)
    try:
        assert p.predict(observation()) == DrivingPrediction(-20,0,.7)
        p.close()
        old=RemotePolicy(*server.address,key=b'test')
        try:
            with pytest.raises(ProtocolError,match='version'):
                old.predict(observation())
        finally: old.close()
    finally: p.close();server.close()


@pytest.mark.parametrize('args',[(0,float('nan'),0),(0,1.1,0),(0,.1,.1),(0,True,0),(451,0,0)])
def test_bad_driving_predictions_rejected(args):
    with pytest.raises(ValueError): DrivingPrediction(*args)


def test_windows_axis_mapping_and_release_before_opposite_pedal():
    rig=Rig();a=rig.adapter()
    try:
        a.write_virtual_state_before(WheelState(1,0,1,0),time.monotonic_ns()+1_000_000_000)
        assert rig.virtual_axes[0x31]==32768 and rig.virtual_axes[0x32]==1
        rig.sdk.SetAxis.reset_mock()
        a.write_virtual_state_before(WheelState(1,0,0,1),time.monotonic_ns()+1_000_000_000)
        calls=[c.args for c in rig.sdk.SetAxis.call_args_list]
        assert calls[1][0]==32768 and calls[1][2]==0x32
        assert calls[2][0]==1 and calls[2][2]==0x31
        with pytest.raises(ActuationExpired):
            a.write_virtual_state_before(WheelState(1,0,1,0),0)
        assert rig.virtual_axes[0x31]==rig.virtual_axes[0x32]==32768
    finally:a.close()


def test_timeout_and_stale_commands_release_auto_pedals():
    class FailingPolicy(DrivingPolicy):
        def __init__(self):self.calls=0
        def predict(self,o):
            self.calls+=1
            if self.calls > 4:raise ObservationUnavailable('disconnected')
            return super().predict(o)
    a=Adapter();r=run(a,FailingPolicy(),duration=.3,assist=True,auto_pedals=True,direct_vjoy=True)
    assert any(s.throttle==.6 for _,s in a.writes)
    assert a.writes[-1][1].throttle==0
    assert r['mode']=='takeover'
    class ExpiredAdapter(Adapter):
        def write_virtual_state_before(self,state,deadline_ns):raise ActuationExpired('test')
    a=ExpiredAdapter();r=run(a,DrivingPolicy(),duration=.1,assist=True,auto_pedals=True,direct_vjoy=True)
    assert r['mode']=='fault'
    assert all(s.throttle==0 for _,s in a.writes)
