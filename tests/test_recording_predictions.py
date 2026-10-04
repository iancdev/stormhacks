"""DAgger analysis: AI predictions are logged next to the human's labels without affecting training."""

from forza_ai.data.sessions import load_session
from forza_ai.recording import SessionRecorder
from test_recording import frame, rows, vehicle, wheel


def test_predictions_logged_deduplicated_and_loader_unaffected(tmp_path):
    path = tmp_path / "dagger"
    recorder = SessionRecorder(path).start()
    plan = [("assist", False, (1, 0, 12.0, 19.2, 0.5, 0.0)),
            ("assist", False, (1, 0, 12.0, 19.2, 0.5, 0.0)),       # same prediction: not repeated
            ("takeover", True, (3, 2, -4.0, -6.4, None, None)),   # AI keeps predicting while human drives
            ("takeover", True, None)]
    for index, (mode, expert, prediction) in enumerate(plan):
        ms = 100 + index * 20
        assert recorder.submit(wheel(ms), vehicle(ms), frame(index, ms), mode, expert=expert,
                               reason="test", prediction=prediction)
    recorder.close()
    logged = rows(path / "predictions.csv")
    assert [r["predicted_angle_deg"] for r in logged] == ["12.0", "-4.0"]
    assert logged[0]["applied_target_deg"] == "19.2" and logged[1]["control_mode"] == "takeover"
    assert logged[1]["predicted_throttle"] == ""
    session = load_session(path)                        # training loader still reads the session
    assert {s.control_mode for s in session.samples} == {"takeover"}
