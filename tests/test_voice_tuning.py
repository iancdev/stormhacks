"""APEX voice co-pilot: bounded live tuning, wake word, keyword fallback, runtime integration."""

import queue

import numpy as np

from forza_ai.contracts import DrivingPrediction
from forza_ai.runtime import run
from forza_ai.tuning import LiveTuning
from forza_ai.voice import KeywordBrain, Segmenter, VoiceAssistant, describe, strip_wake
from test_driving_control import Adapter


def tuning():
    return LiveTuning(throttle_cap=0.7, throttle_rate=0.5, max_speed_kmh=0.0, brake_gain=1.5,
                      corner_speed_kmh=130.0, steer_gain=1.6)


def test_changes_are_step_limited_and_clamped():
    t = tuning()
    r = t.apply([{"parameter": "throttle_cap", "mode": "set", "value": 1.0}])[0]
    assert r["new"] == 0.9 and "step" in r["note"]                  # at most +0.2 per command
    r = t.apply([{"parameter": "throttle_cap", "mode": "change", "value": 0.2}])[0]
    assert r["new"] == 1.0 and "clamp" in r["note"]                  # never above 1
    r = t.apply([{"parameter": "steer_gain", "mode": "change", "value": -0.1}])[0]
    assert r["applied"] and r["new"] == 1.5


def test_bad_input_is_skipped_not_raised():
    t = tuning()
    results = t.apply([{"parameter": "torque_limit", "mode": "set", "value": 1},
                       {"parameter": "throttle_cap", "mode": "set", "value": float("nan")},
                       {"parameter": "brake_gain", "mode": "set", "value": True}, "junk"])
    assert not any(r["applied"] for r in results)
    assert t.snapshot() == tuning().snapshot()


def test_speed_limits():
    t = tuning()
    assert not t.apply([{"parameter": "max_speed_kmh", "mode": "change", "value": 20}])[0]["applied"]   # off
    assert t.apply([{"parameter": "max_speed_kmh", "mode": "set", "value": 150}])[0]["new"] == 150
    assert t.apply([{"parameter": "max_speed_kmh", "mode": "set", "value": 15}])[0]["new"] == 40      # misheard
    assert t.apply([{"parameter": "corner_speed_kmh", "mode": "set", "value": 0}])[0]["new"] == 0      # off


def test_wake_word():
    assert strip_wake("APEX, can you speed it up a bit?") == "can you speed it up a bit?"
    assert strip_wake("Apex speed up") == "speed up"
    assert strip_wake("nice corner") is None
    assert strip_wake("speed up", required=False) == "speed up"


def test_keyword_brain_scales_with_wording():
    settings = tuning().snapshot()
    small = KeywordBrain().interpret(settings, text="APEX, speed it up a bit")
    plain = KeywordBrain().interpret(settings, text="Apex go faster")
    assert small["action"] == plain["action"] == "tune"
    assert 0 < small["adjustments"][0]["value"] < plain["adjustments"][0]["value"]
    assert KeywordBrain().interpret(settings, text="APEX stop, I've got it")["action"] == "take_over"
    assert KeywordBrain().interpret(settings, text="APEX what a view")["action"] == "answer"
    assert KeywordBrain().interpret(settings, text="go faster")["action"] == "ignore"        # no wake word


def test_segmenter_finds_speech_between_silence():
    seg, rng = Segmenter(), np.random.default_rng(0)
    quiet = lambda: (rng.normal(0, 50, 480)).astype(np.int16)
    loud = lambda: (rng.normal(0, 3000, 480)).astype(np.int16)
    out = [seg.feed(quiet()) for _ in range(30)] + [seg.feed(loud()) for _ in range(30)] + \
          [seg.feed(quiet()) for _ in range(40)]
    found = [o for o in out if o is not None]
    assert len(found) == 1 and 30 * 480 <= len(found[0]) <= 80 * 480
    assert all(seg.feed(quiet()) is None for _ in range(50))     # background alone never triggers


def test_assistant_round_trip_and_takeover():
    t, events = tuning(), queue.Queue()
    assistant = VoiceAssistant(KeywordBrain(), log=lambda *_: None)
    assistant._events, assistant._tuning = events, t
    import threading

    def loop():
        event = events.get(timeout=2)
        event[2].put(t.apply(event[1]))
    threading.Thread(target=loop, daemon=True).start()
    assert assistant.handle(text="Apex, faster please")["action"] == "tune"
    assert t.throttle_cap > 0.7
    assert assistant.handle(text="faster, no wake word here") is None
    assistant.handle(text="APEX stop")
    assert events.get_nowait() == "manual"


def test_describe():
    assert describe([{"parameter": "throttle_cap", "applied": True, "new": 0.8}]) == "throttle 80%."
    assert describe([{"parameter": "throttle_cap", "applied": False, "note": "clamped to 0.1-1"}]) == \
        "Already at the limit."


class FullThrottle:
    driving = True
    def predict(self, observation):
        return DrivingPrediction(0, 1.0, 0)


def test_runtime_applies_tune_events_live():
    events, reply = queue.Queue(), queue.Queue()
    events.put(("tune", [{"parameter": "throttle_cap", "mode": "set", "value": 0.8}], reply))
    a = Adapter()
    summary = run(a, FullThrottle(), duration=.4, assist=True, auto_pedals=True, direct_vjoy=True,
                  throttle_cap=0.6, command_queue=events)
    assert max(s.throttle for _, s in a.writes) == 0.8
    assert reply.get_nowait()[0]["new"] == 0.8
    assert summary["tuning"]["throttle_cap"] == 0.8 and summary["tuning_changes"]


def test_runtime_steer_gain_tune_reaches_controller():
    class Turn:
        driving = True
        def predict(self, observation):
            return DrivingPrediction(10, 0.0, 0)
    events = queue.Queue()
    events.put(("tune", [{"parameter": "steer_gain", "mode": "set", "value": 1.4}], None))
    a = Adapter()
    run(a, Turn(), duration=.6, assist=True, auto_pedals=True, direct_vjoy=True, command_queue=events,
        config=__import__("forza_ai.control.steering", fromlist=["SteeringConfig"]).SteeringConfig(
            target_rate_deg_s=400))
    assert max(abs(s.angle_deg) for _, s in a.writes) > 13          # 10 deg x 1.4 (step-limited from 1.0)


def test_gemini_brain_request_and_parsing():
    from types import SimpleNamespace
    from forza_ai.voice import GeminiBrain
    sent = {}

    class Interactions:
        def create(self, **kwargs):
            sent.update(kwargs)
            step = SimpleNamespace(type="function_call", name="adjust_driving", arguments={
                "heard": "APEX, speed it up a bit",
                "adjustments": [{"parameter": "throttle_cap", "mode": "change", "value": 0.05}],
                "spoken_reply": "Pushing a bit harder."})
            return SimpleNamespace(steps=[SimpleNamespace(type="thought"), step])

    brain = GeminiBrain.__new__(GeminiBrain)
    brain.client, brain.model, brain.name, brain._fast = SimpleNamespace(interactions=Interactions()), "m", "t", True
    audio = np.zeros(16000, np.int16)
    decision = brain.interpret({"throttle_cap": 0.7, "steer_gain": 1.6}, audio=audio)
    assert decision == {"action": "tune", "heard": "APEX, speed it up a bit", "reply": "Pushing a bit harder.",
                        "adjustments": [{"parameter": "throttle_cap", "mode": "change", "value": 0.05}]}
    assert sent["input"][1]["type"] == "audio" and sent["input"][1]["mime_type"] == "audio/wav"
    enum = sent["tools"][0]["parameters"]["properties"]["adjustments"]["items"]["properties"]["parameter"]["enum"]
    assert enum == ["steer_gain", "throttle_cap"]                    # only this run's settings
    assert "throttle_cap = 0.7" in sent["system_instruction"] and "APEX, ..." in sent["system_instruction"]
    assert sent["generation_config"]["tool_choice"] == "any" and sent["store"] is False


def test_gemini_ignore_means_no_event():
    from types import SimpleNamespace
    class Brain:
        name = "stub"
        def interpret(self, settings, **_):
            return {"action": "ignore", "heard": "nice pass", "adjustments": [], "reply": ""}
    events = queue.Queue()
    assistant = VoiceAssistant(Brain(), log=lambda *_: None)
    assistant._events, assistant._tuning = events, tuning()
    assert assistant.handle(audio=np.zeros(8000, np.int16)) is None and events.empty()
