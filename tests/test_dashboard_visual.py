"""Dashboard presentation contracts with real status payloads and pure client logic.

No browser dependency is needed here; the coordinator also verifies real browser
layout. Node exercises the same inline view/plot functions served to operators.
"""
import json
import queue
import re
import shutil
import subprocess
import time

import pytest

from forza_ai.dashboard import Dashboard, _PAGE, _safe
from forza_ai.metrics import RunMetrics


def client(expression):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to exercise dashboard client behavior")
    script = re.search(r"<script>(.*?)</script>", _PAGE, re.S)[1]
    runner = "const fs=require('fs'),vm=require('vm');const p=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(vm.runInNewContext(p.script+'\\n;JSON.stringify('+p.expression+')',{}));"
    result = subprocess.run([node, "-e", runner], input=json.dumps({"script": script, "expression": expression}),
                            capture_output=True, text=True, check=True, timeout=5)
    return json.loads(result.stdout)


def view(status=None, *, stale=False, readonly=False, offline=False):
    payload = dict(status=status or {}, stale=stale, controls=dict(allow_arm=True, read_only=readonly))
    return client(f"viewState({json.dumps(payload)}, {str(offline).lower()})")


def test_nested_metrics_rates_and_limits_survive_without_secrets():
    metrics = RunMetrics()
    for index in range(3):
        metrics.update(dict(timestamp_ns=1_000_000_000 + index * 10_000_000, mode="assist",
                            actual_angle_deg=2, target_angle_deg=4, observation_age_ms=20,
                            inference_ms=7, prediction_id=index, input_status="ready", torque=.05))
    summary = metrics.summary()
    summary["latency"]["observation_age"]["access_token"] = "never expose"
    dashboard = Dashboard(queue.Queue())
    dashboard.publish(dict(mode="assist", metrics=summary,
                           rates=dict(capture_fps=None, policy_fps=29.3, control_hz=99.7, secret="hidden"),
                           limits=dict(observation_age_ms=250, target_angle_deg=90, torque=.15),
                           policy_name="Remote policy", expert_recording=False))
    result = dashboard.snapshot()["status"]
    assert result["metrics"]["latency"]["observation_age"]["count"] == 3
    assert result["metrics"]["latency"]["observation_age"]["p95_ms"] > 0
    assert result["metrics"]["routes"]["completed_duration"]["count"] == 0
    assert result["metrics"]["tracking_rmse_deg"] == 2
    assert result["rates"] == dict(capture_fps=None, policy_fps=29.3, control_hz=99.7)
    assert result["limits"]["observation_age_ms"] == 250
    assert "never expose" not in json.dumps(result)
    assert "hidden" not in json.dumps(result)


def test_nested_normalization_is_bounded_and_finite():
    tree = {str(i): {str(j): {str(k): {str(n): "x" * 1024 for n in range(8)}
                                     for k in range(8)} for j in range(8)} for i in range(2)}
    result = _safe(tree)
    # A single field has a shared 256-node budget, not an exponential depth cap.
    assert len(json.dumps(result)) < 75_000
    assert _safe({"a": float("nan"), "b": float("inf"), "values": [1, 2]}) == {"a": None, "b": None, "values": None}
    assert _safe({"a": {"b": {"c": {"d": {"password": "hidden"}}}}}) == {"a": {"b": {"c": {"d": None}}}}


@pytest.mark.parametrize("status,stale,readonly,allowed", [
    ({"mode": "manual"}, False, False, True),
    ({"mode": "takeover"}, False, False, True),
    ({"mode": "fault"}, False, False, True),
    ({"mode": "assist"}, False, False, False),
    ({"mode": "manual", "shadow": True}, False, False, False),
    ({"mode": "manual"}, True, False, False),
    ({"mode": "manual"}, False, True, False),
    ({}, False, False, False),
])
def test_server_engagement_availability_matches_operator_state(status, stale, readonly, allowed):
    dashboard = Dashboard(queue.Queue(), read_only=readonly)
    dashboard.publish(status)
    if stale:
        dashboard._published_ns = time.monotonic_ns() - 1_000_000_000
    assert dashboard.snapshot()["controls"]["allow_arm"] is allowed


@pytest.mark.parametrize("status,options,kind,arm", [
    ({}, {}, "waiting", False),
    ({"mode": "manual"}, {}, "manual", True),
    ({"mode": "assist"}, {}, "assist", False),
    ({"mode": "fault", "reason": "stale_wheel"}, {}, "fault", True),
    ({"mode": "manual", "shadow": True}, {}, "shadow", False),
    ({"mode": "assist"}, {"stale": True}, "stale", False),
    ({"mode": "assist"}, {"offline": True}, "offline", False),
    ({"mode": "manual"}, {"readonly": True}, "manual", False),
])
def test_client_state_does_not_declare_health_from_connection_alone(status, options, kind, arm):
    result = view(status, **options)
    assert result["kind"] == kind
    assert result["arm"] is arm
    if kind in ("offline", "stale"):
        assert "unknown" in (result["title"] + result["detail"]).lower()
    if kind == "fault":
        assert "fault" in result["title"].lower()


def test_route_controls_require_a_known_fresh_marker_state():
    idle = view({"mode": "manual", "route_active": False})
    active = view({"mode": "assist", "route_active": True})
    assert idle["startRoute"] and not idle["finishRoute"]
    assert active["finishRoute"] and not active["startRoute"]
    for options in ({"stale": True}, {"offline": True}, {"readonly": True}):
        result = view({"mode": "manual", "route_active": True}, **options)
        assert not result["finishRoute"] and not result["startRoute"]
    # Keep a manual takeover request available while status is stale but HTTP works.
    assert view({"mode": "assist"}, stale=True)["takeover"]
    assert not view({"mode": "assist"}, offline=True)["takeover"]


def test_empty_chart_and_null_values_never_become_zero_measurements():
    result = client("chartData([], ['actual_angle_deg'])")
    assert result["paths"] == [""] and result["hasData"] is False
    rows = [{"timestamp_ns": 1_000_000_000, "actual_angle_deg": None},
            {"timestamp_ns": 1_100_000_000, "actual_angle_deg": None}]
    result = client(f"chartData({json.dumps(rows)}, ['actual_angle_deg'])")
    assert result["paths"] == [""] and result["hasData"] is False
    rows[0]["actual_angle_deg"] = 0
    rows[1]["actual_angle_deg"] = 0
    result = client(f"chartData({json.dumps(rows)}, ['actual_angle_deg'])")
    assert result["hasData"] is True and " L" in result["paths"][0]


def test_chart_uses_timestamps_and_breaks_across_capture_gaps():
    rows = [{"timestamp_ns": 1_000_000_000, "actual_angle_deg": 1},
            {"timestamp_ns": 1_100_000_000, "actual_angle_deg": 2},
            {"timestamp_ns": 2_000_000_000, "actual_angle_deg": 3}]
    result = client(f"chartData({json.dumps(rows)}, ['actual_angle_deg'])")
    points = re.findall(r"[ML]([\d.]+),([\d.]+)", result["paths"][0])
    deltas = [float(points[i + 1][0]) - float(points[i][0]) for i in range(2)]
    assert deltas[1] / deltas[0] == pytest.approx(9, abs=.1)
    assert result["paths"][0].count(" M") == 2
    assert result["hasData"] is True


def test_plot_history_contains_real_source_age_and_torque():
    dashboard = Dashboard(queue.Queue())
    dashboard.publish(dict(timestamp_ns=42, actual_angle_deg=-3, torque=-.08,
                           observation_age_ms=14, inference_ms=9))
    sample = dashboard.snapshot()["history"][0]
    assert sample["torque"] == -.08 and sample["observation_age_ms"] == 14
    assert sample["inference_ms"] == 9
    assert sample["predicted_angle_deg"] is None


def test_quick_disengage_is_available_only_when_ai_may_have_authority():
    assert view({"mode": "assist"})["disengage"]
    assert view({"mode": "assist"}, stale=True)["disengage"]
    assert not view({"mode": "manual"})["disengage"]
    assert not view({"mode": "takeover"})["disengage"]
    assert not view({"mode": "fault"})["disengage"]
    assert not view({"mode": "assist"}, offline=True)["disengage"]
    assert not view({"mode": "assist"}, readonly=True)["disengage"]
    assert not view({"mode": "manual", "shadow": True})["disengage"]
    # Full manual takeover remains useful for declaring expert control separately.
    assert view({"mode": "manual"})["takeover"]


def test_quick_action_reuses_manual_queue_without_duplicate_element_ids():
    from html.parser import HTMLParser

    class Elements(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []
            self.manual_actions = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.append(attrs["id"])
            if tag == "button" and attrs.get("data-event") == "manual":
                self.manual_actions.append(attrs)

    elements = Elements()
    elements.feed(_PAGE)
    assert len(elements.ids) == len(set(elements.ids))
    assert len(elements.manual_actions) == 2
    assert sum(action.get("data-quick") == "true" for action in elements.manual_actions) == 1
    assert all("disabled" in action for action in elements.manual_actions)


def test_stale_wheel_explanation_names_sample_age_without_diagnosing_disconnect():
    result = view({"mode": "fault", "reason": "stale_wheel", "hardware_mode": "simulation"})
    assert "sample is too old" in result["detail"]
    assert "control-loop timing" in result["detail"]
    assert "device connection" in result["detail"]
    assert "disconnected" not in result["detail"]


def test_arm_acknowledgement_waits_for_a_subsequent_runtime_snapshot():
    baseline = {"status": {"timestamp_ns": 10, "ticks": 1, "mode": "manual"}, "stale": False}
    # An accepted request is insufficient; sending and unchanged snapshots stay pending.
    expression = f"(()=>{{const before={json.dumps(baseline)};const a=makePendingAction('arm',before,0);const sending=actionOutcome(a,{{status:{{timestamp_ns:20,mode:'assist'}}}},1);a.phase='queued';const unchanged=actionOutcome(a,before,2);const confirmed=actionOutcome(a,{{status:{{timestamp_ns:20,mode:'assist'}},stale:false}},3);return {{sending,unchanged,confirmed}};}})()"
    result = client(expression)
    assert result["sending"]["done"] is False
    assert result["unchanged"]["done"] is False
    assert result["confirmed"]["done"] is True
    assert result["confirmed"]["error"] is False
    assert "engaged" in result["confirmed"]["message"]


def test_manual_confirmation_requires_new_fresh_reported_mode():
    expression = "(()=>{const a=makePendingAction('manual',{status:{timestamp_ns:10,ticks:5,mode:'assist'}},0);a.phase='queued';return [actionOutcome(a,{status:{timestamp_ns:10,mode:'takeover'}},1),actionOutcome(a,{status:{timestamp_ns:20,mode:'takeover'},stale:true},2),actionOutcome(a,{status:{timestamp_ns:20,mode:'takeover'},stale:false},3)];})()"
    result = client(expression)
    assert [value["done"] for value in result] == [False, False, True]
    assert "disengaged" in result[-1]["message"]


@pytest.mark.parametrize("event", ["route_start", "route_complete", "route_abort"])
def test_route_confirmation_uses_the_matching_runtime_counter(event):
    before = {"status": {"timestamp_ns": 10, "route_active": True,
                          "metrics": {"events": {event: 3}}}, "stale": False}
    after = {"status": {"timestamp_ns": 20, "route_active": False,
                         "metrics": {"events": {event: 3}}}, "stale": False}
    expression = f"(()=>{{const a=makePendingAction('{event}',{json.dumps(before)},0);a.phase='queued';const after={json.dumps(after)};const unrelated=actionOutcome(a,after,1);after.status.metrics.events['{event}']=4;return {{unrelated,confirmed:actionOutcome(a,after,2)}};}})()"
    result = client(expression)
    assert result["unrelated"]["done"] is False
    assert result["confirmed"]["done"] is True
    assert "recorded" in result["confirmed"]["message"]


def test_confirmation_timeout_reports_reason_without_claiming_success():
    expression = "(()=>{const a=makePendingAction('arm',{status:{timestamp_ns:10,mode:'manual'}},0);a.phase='queued';const after={status:{timestamp_ns:20,mode:'fault',reason:'stale_wheel'},stale:false};return {waiting:actionOutcome(a,after,7999),expired:actionOutcome(a,after,8000),offline:actionOutcome(a,after,8000,true)};})()"
    result = client(expression)
    assert result["waiting"]["done"] is False
    assert result["expired"]["done"] and result["expired"]["error"]
    assert "Not confirmed" in result["expired"]["message"]
    assert "stale wheel" in result["expired"]["message"]
    assert "unavailable" in result["offline"]["message"]


def test_narrow_plot_time_ticks_remain_compact_and_do_not_mutate_history():
    result = client("({small:timeTicks(350),large:timeTicks(700)})")
    assert [tick["label"] for tick in result["small"]] == ["-18s", "-9s", "latest"]
    assert len(result["large"]) == 5
    assert all(len(tick["label"]) <= 6 for tick in result["large"])
    rows = [{"timestamp_ns": 1_000_000_000, "actual_angle_deg": 0},
            {"timestamp_ns": 1_100_000_000, "actual_angle_deg": 2}]
    outcome = client(f"(()=>{{const rows={json.dumps(rows)};chartData(rows,['actual_angle_deg'],{{width:350}});return rows;}})()")
    assert outcome == rows


def timed_client(body):
    """Run served client code with a deterministic clock and real promise jobs."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to exercise dashboard client behavior")
    script = re.search(r"<script>(.*?)</script>", _PAGE, re.S)[1]
    event_handlers = script.split("if(typeof document!=='undefined'){")[1].split(
        "document.querySelectorAll('[data-series]')"
    )[0]
    clock = """
let clockMs=0,nextTimer=1;
const timers=new Map(),performance={now:()=>clockMs};
function setTimeout(callback,delay){const id=nextTimer++;timers.set(id,{callback,at:clockMs+delay});return id;}
function clearTimeout(id){timers.delete(id);}
async function settle(){for(let i=0;i<30;i++)await Promise.resolve();}
async function advance(ms){
    const until=clockMs+ms;
    while(true){
        const entry=[...timers].filter(([,t])=>t.at<=until).sort((a,b)=>a[1].at-b[1].at)[0];
        if(!entry)break;
        const [id,timer]=entry;timers.delete(id);clockMs=timer.at;timer.callback();await settle();
    }
    clockMs=until;await settle();
}
"""
    runner = r"""
const fs=require('fs'),vm=require('vm'),p=JSON.parse(fs.readFileSync(0,'utf8'));
vm.runInNewContext(p.clock+p.script+'\n;(async()=>{'+p.body+'})()',
    {AbortController,eventHandlers:p.event_handlers})
    .then(result=>process.stdout.write(JSON.stringify(result)))
    .catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([node, "-e", runner], input=json.dumps(dict(
        clock=clock, script=script, body=body, event_handlers=event_handlers)),
        capture_output=True, text=True, check=True, timeout=5)
    assert result.stdout, "Client promise never settled after its deadline"
    return json.loads(result.stdout)


@pytest.mark.parametrize("stall", ["fetch", "body"])
def test_status_poll_deadline_disables_controls_and_retries_after_stalled_io(stall):
    result = timed_client("const stall=" + json.dumps(stall) + ";" + """
const fresh={status:{timestamp_ns:1,mode:'manual',route_active:false},snapshot_age_ms:0,
    stale:false,controls:{allow_arm:true}};
let calls=0,bodyCalls=0,signals=[],renders=[];
globalThis.fetch=async(url,options)=>{
    calls++;signals.push(options.signal);
    if(calls===1&&stall==='fetch')return new Promise(()=>{});
    return {ok:true,json:async()=>{bodyCalls++;return calls===1?new Promise(()=>{}):fresh;}};
};
render=(data,isOffline=false)=>{
    lastData=data;offline=isOffline;
    renders.push({offline,isStale:data.stale,view:viewState(data,isOffline)});
};
lastData=fresh;lastReceivedAt=0;
const first=refresh();await settle();
await advance(499);checkFreshness();const before=viewState(lastData,offline);
await advance(1);checkFreshness();const stale=viewState(lastData,offline);
await advance(500);await first;const failed=renders[renders.length-1];
const retryDelays=[...timers.values()].map(t=>t.at-clockMs);
checkFreshness();const stillOffline=offline;
await advance(100);const recovered=renders[renders.length-1];
return {before,stale,failed,retryDelays,stillOffline,recovered,calls,bodyCalls,
    firstAborted:signals[0].aborted,lastReceivedAt};
""")
    assert result["before"]["arm"] and result["before"]["startRoute"]
    assert result["stale"]["kind"] == "stale"
    assert not result["stale"]["arm"] and not result["stale"]["startRoute"]
    assert result["failed"]["offline"] and result["firstAborted"]
    assert not result["failed"]["view"]["arm"]
    assert not result["failed"]["view"]["startRoute"]
    assert result["retryDelays"] == [100]
    assert result["stillOffline"]
    assert not result["recovered"]["offline"] and not result["recovered"]["isStale"]
    assert result["recovered"]["view"]["arm"] and result["recovered"]["view"]["startRoute"]
    assert result["calls"] == 2
    assert result["bodyCalls"] == (1 if stall == "fetch" else 2)
    assert result["lastReceivedAt"] == 1100


def test_client_freshness_includes_server_age_and_preserves_offline_state():
    result = timed_client("""
let renders=[];
render=(data,isOffline=false)=>{lastData=data;offline=isOffline;renders.push(data.snapshot_age_ms);};
lastData={status:{mode:'assist'},snapshot_age_ms:350,stale:false,controls:{allow_arm:false}};
lastReceivedAt=0;
await advance(149);checkFreshness();const before=viewState(lastData,offline);
await advance(1);checkFreshness();const expired=viewState(lastData,offline);
offline=true;await advance(500);checkFreshness();
return {before,expired,renders,offline};
""")
    assert result["before"]["kind"] == "assist"
    assert result["expired"]["kind"] == "stale"
    assert result["renders"] == [500]
    assert result["offline"]


@pytest.mark.parametrize("stall", ["fetch", "body"])
def test_event_timeout_reports_unconfirmed_delivery_instead_of_not_delivered(stall):
    result = timed_client("const stall=" + json.dumps(stall) + ";" + """
let click,signal,feedback=[];
const button={dataset:{event:'arm'},addEventListener:(event,handler)=>{click=handler;}};
globalThis.document={querySelectorAll:()=>[button],getElementById:()=>({})};
commandFeedback=(button,message,error=false)=>feedback.push({message,error});
globalThis.fetch=async(url,options)=>{
    signal=options.signal;
    if(stall==='fetch')return new Promise(()=>{});
    return {ok:true,json:()=>new Promise(()=>{})};
};
lastData={status:{timestamp_ns:1,mode:'manual'},stale:false,controls:{allow_arm:true}};
eval(eventHandlers);
const request=click();await settle();await advance(1000);await request;
return {feedback,aborted:signal.aborted,pending:pendingAction};
""")
    assert result["aborted"] and result["pending"] is None
    message = result["feedback"][-1]
    assert message["error"]
    assert "Delivery not confirmed" in message["message"]
    assert "not delivered" not in message["message"]
    assert "runtime state" in message["message"]


def visual_status(**changes):
    status = dict(speed_mps=10, predicted_angle_deg=45, predicted_throttle=.4, predicted_brake=0,
                  physical_throttle=.2, physical_brake=.1, actual_angle_deg=-10,
                  prediction_remaining_ms=150, observation_age_ms=30,
                  input_status='ready', limits=dict(observation_age_ms=250))
    status.update(changes)
    return dict(status=status, snapshot_age_ms=0, stale=False)


def test_prediction_curve_sign_and_controls_are_distinct():
    right = client(f'predictionVisual({json.dumps(visual_status())})')
    left = client(f'predictionVisual({json.dumps(visual_status(predicted_angle_deg=-45))})')
    assert right['path'].endswith('440 25')
    assert left['path'].endswith('280 25')
    assert (right['angle'], right['throttle'], right['physicalAngle'], right['physicalThrottle'], right['physicalBrake']) == (45, 40, -10, 20, 10)


@pytest.mark.parametrize('change', [dict(predicted_angle_deg=None), dict(prediction_remaining_ms=0),
    dict(observation_age_ms=251), dict(input_status='frame_unavailable'), dict(closed=True)])
def test_prediction_curve_rejects_missing_expired_or_invalid_input(change):
    view = client(f'predictionVisual({json.dumps(visual_status(**change))})')
    assert not view['valid'] and not view['path'] and view['angle'] is None


def test_prediction_expiry_in_browser_without_new_snapshot():
    payload = json.dumps(visual_status())
    assert not client(f'predictionVisual({payload},false,150)')['valid']
    assert not client(f'predictionVisual({payload},true)')['valid']
    assert client(f'predictionVisual({payload},false,149)')['valid']
    assert client(f'predictionVisual({json.dumps(visual_status(predicted_throttle=None))})')['throttle'] is None


def test_prediction_fields_survive_snapshot_filter():
    dashboard = Dashboard(queue.Queue())
    dashboard.publish(visual_status()['status'])
    status = dashboard.snapshot()['status']
    assert status['physical_throttle'] == .2
    assert status['physical_brake'] == .1
    assert status['prediction_remaining_ms'] == 150


def test_nonfinite_and_out_of_range_predictions_do_not_draw():
    payload = json.dumps(visual_status())
    for field, value in [('predicted_angle_deg', 'NaN'), ('predicted_throttle', 'Infinity'), ('predicted_brake', '-.1')]:
        result = client(f'(()=>{{const d={payload};d.status.{field}={value};return predictionVisual(d)}})()')
        assert not result['valid'] and result['path'] == ''

def test_snapshot_age_counts_toward_command_and_observation_expiry():
    payload = visual_status()
    payload['snapshot_age_ms'] = 140
    assert not client(f'predictionVisual({json.dumps(payload)},false,10)')['valid']
    payload = visual_status(observation_age_ms=245)
    payload['snapshot_age_ms'] = 5
    assert not client(f'predictionVisual({json.dumps(payload)})')['valid']


def test_stationary_or_unknown_speed_hides_curve_but_retains_fresh_prediction():
    for speed in [0, None]:
        result = client(f'predictionVisual({json.dumps(visual_status(speed_mps=speed))})')
        assert result['valid'] and result['angle'] == 45 and result['path'] == ''
    result = client(f'predictionVisual({json.dumps(visual_status(predicted_brake=.2))})')
    assert not result['valid'] and result['path'] == ''
