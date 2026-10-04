"""Deployment checks must be useful without opening hardware or starting a server."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forza_ai import launch


@pytest.fixture
def game(tmp_path):
    (tmp_path / "capture.json").write_text(json.dumps({"crop": [0, 0, 640, 240], "monitor": 0}))
    return {
        "schema_version": 1, "role": "game", "inference": {"host": "192.168.1.42"},
        "capture": {"config_path": "capture.json"}, "buttons": {"takeover": 3},
        "run": {"output_dir": "runs"},
    }


@pytest.fixture
def desktop():
    return {"schema_version": 1, "role": "desktop", "inference": {"bind": "192.168.1.42"},
            "policy": {"kind": "fixed", "target_angle_deg": 5}, "run": {"output_dir": "runs"}}


def save(tmp_path, data, name="profile.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def value_after(plan, flag):
    return plan.argv[plan.argv.index(flag) + 1]


def test_game_defaults_to_shadow_without_assumed_buttons(tmp_path, game):
    plan = launch.build_plan(save(tmp_path, game), run_id="abc")
    assert plan.role == "game" and plan.mode == "shadow"
    assert "--shadow" in plan.argv and "--assist" not in plan.argv
    assert value_after(plan, "--takeover-button") == "3"
    assert "--arm-button" not in plan.argv and "--route-button" not in plan.argv
    assert "--record-session" not in plan.argv
    assert not plan.run_dir.exists()
    assert "--status-csv" not in plan.argv  # unlimited is bounded aggregate metrics only
    assert value_after(plan, "--run-report") == str(tmp_path / "runs/game-abc/report.json")


def test_profile_paths_are_relative_to_profile_not_cwd(tmp_path, game, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = save(tmp_path, game)
    monkeypatch.chdir(elsewhere)
    plan = launch.build_plan(path, run_id="abc")
    assert value_after(plan, "--capture-config") == str(tmp_path / "capture.json")
    assert plan.run_dir == tmp_path / "runs/game-abc"


def test_game_complete_profile_builds_structured_argv(tmp_path, game):
    game.update(mode="assist", buttons={"takeover": 3, "arm": 4, "route": 5,
                                      "virtual_map": [{"physical": 8, "virtual": 1}]},
                recording={"enabled": True, "include_manual": True, "output_dir": "sessions"},
                run={"output_dir": "runs", "duration_s": 12, "dashboard_port": 8766, "interactive": False})
    plan = launch.build_plan(save(tmp_path, game), run_id="test")
    assert "--assist" in plan.argv and "--shadow" not in plan.argv
    assert "--interactive" not in plan.argv
    assert value_after(plan, "--arm-button") == "4"
    assert value_after(plan, "--route-button") == "5"
    assert value_after(plan, "--button-map") == "8:1"
    assert value_after(plan, "--dashboard-port") == "8766"
    assert value_after(plan, "--status-csv") == str(tmp_path / "runs/game-test/control.csv")
    assert value_after(plan, "--record-session") == str(tmp_path / "sessions/session-test")
    assert "--record-manual" in plan.argv


def test_recording_uses_new_session_in_run_by_default(tmp_path, game):
    game["recording"] = {"enabled": True}
    plan = launch.build_plan(save(tmp_path, game), run_id="test")
    assert plan.recording_dir == plan.run_dir / "session"
    assert "--record-manual" not in plan.argv


@pytest.mark.parametrize("mode,expected", [("manual", "manual"), ("assist", "assist"), ("shadow", "shadow")])
def test_explicit_game_modes(tmp_path, game, mode, expected):
    game["mode"] = mode
    plan = launch.build_plan(save(tmp_path, game))
    assert plan.mode == expected
    assert ("--assist" in plan.argv) == (mode == "assist")
    assert ("--shadow" in plan.argv) == (mode == "shadow")


def test_assist_cli_override_is_explicit(tmp_path, game):
    plan = launch.build_plan(save(tmp_path, game), assist=True)
    assert plan.mode == "assist" and "--assist" in plan.argv and "--shadow" not in plan.argv


@pytest.mark.parametrize("section", [None, "inference", "capture", "buttons", "run", "control", "recording", "telemetry"])
def test_unknown_keys_rejected(tmp_path, game, section):
    target = game if section is None else game.setdefault(section, {})
    target["shell_command"] = "ignored?"
    with pytest.raises(launch.ProfileError, match="unknown"):
        launch.build_plan(save(tmp_path, game))


@pytest.mark.parametrize("version", [True, "1", 0, 2, None])
def test_schema_version_is_strict(tmp_path, desktop, version):
    desktop["schema_version"] = version
    with pytest.raises(launch.ProfileError, match="schema_version"):
        launch.build_plan(save(tmp_path, desktop))


def test_duplicate_json_keys_and_nonfinite_json_rejected(tmp_path):
    path = tmp_path / "profile.json"
    for raw, message in [('{"role":"desktop","role":"game"}', "duplicate"),
                         ('{"timeout_s": NaN}', "non-finite")]:
        path.write_text(raw)
        with pytest.raises(launch.ProfileError, match=message):
            launch.build_plan(path)


def test_role_mismatch_rejected_before_launch(tmp_path, desktop):
    with pytest.raises(launch.ProfileError, match="does not match"):
        launch.build_plan(save(tmp_path, desktop), expected_role="game")
    with pytest.raises(launch.ProfileError, match="game profile"):
        launch.build_plan(save(tmp_path, desktop), assist=True)


@pytest.mark.parametrize("field", ["mode", "capture", "buttons", "control", "telemetry", "recording"])
def test_desktop_cannot_contain_game_fields(tmp_path, desktop, field):
    desktop[field] = {}
    with pytest.raises(launch.ProfileError, match="game-only"):
        launch.build_plan(save(tmp_path, desktop))


@pytest.mark.parametrize("field", ["duration_s", "interactive", "dashboard_port"])
def test_desktop_run_rejects_game_fields(tmp_path, desktop, field):
    desktop["run"][field] = None
    with pytest.raises(launch.ProfileError, match="only accepts"):
        launch.build_plan(save(tmp_path, desktop))


@pytest.mark.parametrize("host", [None, "DESKTOP_LAN_IP", "example.com", "::1", "0.0.0.0", "224.0.0.1", "255.255.255.255"])
def test_game_host_requires_explicit_unicast_ipv4(tmp_path, game, host):
    game["inference"]["host"] = host
    with pytest.raises(launch.ProfileError, match="inference.host"):
        launch.build_plan(save(tmp_path, game))


def test_desktop_may_explicitly_bind_all_ipv4_interfaces(tmp_path, desktop):
    desktop["inference"]["bind"] = "0.0.0.0"
    assert value_after(launch.build_plan(save(tmp_path, desktop)), "--bind") == "0.0.0.0"


@pytest.mark.parametrize("port", [0, 65536, 1.5, True, "8765"])
def test_port_is_strict(tmp_path, desktop, port):
    desktop["inference"]["port"] = port
    with pytest.raises(launch.ProfileError, match="inference.port"):
        launch.build_plan(save(tmp_path, desktop))


@pytest.mark.parametrize("buttons", [
    {"takeover": None}, {"takeover": -1}, {"takeover": True}, {"takeover": 128},
    {"takeover": 3, "arm": 3}, {"takeover": 3, "route": 3},
    {"takeover": 3, "arm": 4, "route": 4},
    {"takeover": 3, "virtual_map": [{"physical": 3, "virtual": 1}]},
    {"takeover": 3, "virtual_map": [{"physical": 5, "virtual": 1}, {"physical": 5, "virtual": 2}]},
    {"takeover": 3, "virtual_map": [{"physical": 5, "virtual": 1}, {"physical": 6, "virtual": 1}]},
    {"takeover": 3, "virtual_map": [{"physical": 5, "virtual": 0}]},
    {"takeover": 3, "virtual_map": [{"physical": 5, "virtual": 129}]},
])
def test_invalid_button_combinations(tmp_path, game, buttons):
    game["buttons"] = buttons
    with pytest.raises(launch.ProfileError):
        launch.build_plan(save(tmp_path, game))


def test_capture_placeholder_and_invalid_crop_rejected(tmp_path, game):
    game["capture"]["config_path"] = None
    with pytest.raises(launch.ProfileError, match="null placeholder"):
        launch.build_plan(save(tmp_path, game))
    game["capture"]["config_path"] = "capture.json"
    (tmp_path / "capture.json").write_text('{"crop": [10, 10, 0, 0]}')
    with pytest.raises(ValueError, match="positive width"):
        launch.build_plan(save(tmp_path, game))


@pytest.mark.parametrize("section,field,value", [
    ("run", "interactive", "true"), ("run", "duration_s", -1), ("run", "dashboard_port", 0),
    ("capture", "hz", 0), ("capture", "game_process", "../bad.exe"),
    ("control", "torque_limit", 1.1), ("control", "hz", 9), ("control", "policy_hz", False),
    ("control", "target_limit_deg", 451), ("control", "kp", -1), ("control", "kd", -1),
    ("control", "target_rate_deg_s", 0), ("recording", "include_manual", True),
])
def test_invalid_runtime_values_fail_before_process(tmp_path, game, section, field, value):
    game.setdefault(section, {})[field] = value
    with pytest.raises(launch.ProfileError):
        launch.build_plan(save(tmp_path, game))


def test_fixed_target_explicit_and_bounded(tmp_path, desktop):
    for target in (None, 16, -16, True):
        desktop["policy"]["target_angle_deg"] = target
        with pytest.raises(launch.ProfileError):
            launch.build_plan(save(tmp_path, desktop))


def artifact(tmp_path, name="export"):
    path = tmp_path / name
    path.mkdir()
    (path / "model.pt").write_bytes(b"fixture; doctor must separately verify weights")
    (path / "metadata.json").write_text(json.dumps({"format_version": 1, "architecture": "pilotnet_speed_v1",
                                                   "image_stage": "road_crop", "preprocessing": {}}))
    return path


def test_model_profile_paths_are_single_argv_elements_without_shell(tmp_path, desktop, monkeypatch):
    path = artifact(tmp_path, "model; echo NEVER_RUN $(anything)")
    desktop["policy"] = {"kind": "model", "model_path": path.name}
    plan = launch.build_plan(save(tmp_path, desktop), run_id="test")
    assert value_after(plan, "--model") == str(path)
    calls = []
    monkeypatch.setenv("FORZA_LINK_KEY", "this-test-key-must-not-appear-in-output")
    monkeypatch.setattr(launch.subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs)) or SimpleNamespace(returncode=0, wait=lambda: 0))
    assert launch.launch(plan) == 0
    assert calls == [(list(plan.argv), {"shell": False})]
    assert calls[0][0][0] == launch.sys.executable
    assert "--test-target" not in plan.argv
    assert "this-test-key" not in (plan.run_dir / "launch.json").read_text()


def test_existing_run_and_session_cannot_be_overwritten(tmp_path, game):
    path = save(tmp_path, game)
    plan = launch.build_plan(path, run_id="same")
    plan.run_dir.mkdir(parents=True)
    with pytest.raises(launch.ProfileError, match="already exists"):
        launch.build_plan(path, run_id="same")
    game["recording"] = {"enabled": True, "output_dir": "sessions"}
    (tmp_path / "sessions/session-next").mkdir(parents=True)
    with pytest.raises(launch.ProfileError, match="already exists"):
        launch.build_plan(save(tmp_path, game), run_id="next")


def test_checks_never_launch_create_files_or_expose_key(tmp_path, desktop, monkeypatch, capsys):
    secret = "exact-secret-that-never-appears"
    monkeypatch.setenv("FORZA_LINK_KEY", secret)
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **kw: pytest.fail("check started a process"))
    path = save(tmp_path, desktop)
    assert launch.main(["--profile", str(path), "--check"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["profile_valid"] and report["shared_key_present"]
    assert secret not in out and "--test-target" in report["argv"]
    assert not (tmp_path / "runs").exists()


def test_key_is_redacted_even_in_arbitrary_profile_path(tmp_path, desktop, monkeypatch, capsys):
    secret = "embedded-secret-value"
    monkeypatch.setenv("FORZA_LINK_KEY", secret)
    desktop["run"]["output_dir"] = f"runs/{secret}"
    launch.main(["--profile", str(save(tmp_path, desktop)), "--check"])
    assert secret not in capsys.readouterr().out


def test_secret_redaction_occurs_before_json_escaping(monkeypatch, capsys):
    secret = 'quote"and\nnewline'
    monkeypatch.setenv("FORZA_LINK_KEY", secret)
    launch._emit({"argv": ["prefix/" + secret], "schema_version": 1})
    report = json.loads(capsys.readouterr().out)
    assert report == {"argv": ["prefix/<redacted>"], "schema_version": 1}


def test_huge_integer_reports_profile_error(tmp_path, desktop):
    desktop["inference"]["port"] = 10**400
    with pytest.raises(launch.ProfileError, match="inference.port"):
        launch.build_plan(save(tmp_path, desktop))


def test_missing_key_and_wrong_platform_report_unready(tmp_path, game, monkeypatch, capsys):
    monkeypatch.delenv("FORZA_LINK_KEY", raising=False)
    monkeypatch.setattr(launch.sys, "platform", "darwin")
    assert launch.main(["--profile", str(save(tmp_path, game)), "--check"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["profile_valid"] and not report["ready"]
    assert not report["shared_key_present"] and not report["platform_ready"]
    with pytest.raises(launch.ProfileError, match="launch requires"):
        launch.launch(launch.build_plan(save(tmp_path, game)))
    assert not (tmp_path / "runs").exists()


def test_unknown_secret_field_does_not_print_value(tmp_path, desktop, capsys):
    desktop["FORZA_LINK_KEY"] = "do-not-print-this"
    with pytest.raises(SystemExit) as exc:
        launch.main(["--profile", str(save(tmp_path, desktop)), "--check"])
    assert exc.value.code == 2
    out = capsys.readouterr()
    assert "do-not-print-this" not in out.err + out.out


def mock_packages(monkeypatch, missing=()):
    monkeypatch.setattr(launch, "_package", lambda distribution, module:
                        {"distribution": distribution, "installed": distribution not in missing, "version": "fixture"})


def test_desktop_doctor_checks_cuda_without_hardware_or_network(monkeypatch, capsys):
    mock_packages(monkeypatch)
    monkeypatch.setenv("FORZA_LINK_KEY", "not-output")
    monkeypatch.setattr(launch, "_cuda_report", lambda: {"checked": True, "available": True, "devices": ["mock GPU"]})
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **kw: pytest.fail("doctor started a process"))
    assert launch.main(["doctor", "--role", "desktop"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["training_ready"] and report["gpu_training_ready"] and report["ready"]
    assert not report["hardware_opened"] and not report["network"]["connectivity_checked"]
    assert report["cuda"]["devices"] == ["mock GPU"]


def test_doctor_separates_gpu_and_training_from_key_readiness(monkeypatch):
    mock_packages(monkeypatch)
    monkeypatch.delenv("FORZA_LINK_KEY", raising=False)
    monkeypatch.setattr(launch, "_cuda_report", lambda: {"checked": True, "available": False, "devices": []})
    report = launch.doctor_report("desktop")
    assert report["training_ready"] and not report["gpu_training_ready"] and not report["ready"]


def test_doctor_missing_torch_skips_import(monkeypatch):
    mock_packages(monkeypatch, missing=("torch",))
    monkeypatch.setattr(launch, "_cuda_report", lambda: pytest.fail("missing torch was imported"))
    report = launch.doctor_report("desktop")
    assert not report["training_ready"] and not report["cuda"]["checked"]


def test_doctor_installed_but_broken_torch_is_not_training_ready(monkeypatch):
    mock_packages(monkeypatch)
    monkeypatch.setenv("FORZA_LINK_KEY", "not-output")
    monkeypatch.setattr(launch, "_cuda_report", lambda: {"checked": False, "available": False, "error_type": "OSError"})
    report = launch.doctor_report("desktop")
    assert not report["ready"] and not report["training_ready"] and not report["gpu_training_ready"]


def test_game_doctor_only_discovers_packages(monkeypatch):
    mock_packages(monkeypatch)
    monkeypatch.setenv("FORZA_LINK_KEY", "not-output")
    monkeypatch.setattr(launch.sys, "platform", "win32")
    monkeypatch.setattr(launch, "_cuda_report", lambda: pytest.fail("game must not need Torch"))
    report = launch.doctor_report("game")
    assert report["ready"] and "cuda" not in report
    assert not report["hardware_opened"]


def test_cuda_query_does_not_echo_exception_secret(monkeypatch):
    def fail(_):
        raise RuntimeError("raw-private-value")
    monkeypatch.setattr(launch.importlib, "import_module", fail)
    report = launch._cuda_report()
    assert report["error_type"] == "RuntimeError"
    assert "raw-private-value" not in json.dumps(report)


def test_cuda_query_uses_installed_torch_only(monkeypatch):
    fake = SimpleNamespace(__version__="2.fixture", version=SimpleNamespace(cuda="12.fixture"),
                           cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                                get_device_name=lambda index: "mock GPU"))
    monkeypatch.setattr(launch.importlib, "import_module", lambda name: fake if name == "torch" else pytest.fail(name))
    assert launch._cuda_report() == {"checked": True, "available": True, "devices": ["mock GPU"],
                                      "torch_version": "2.fixture", "build_cuda": "12.fixture"}


def test_doctor_optional_artifact_load_is_explicit(tmp_path, monkeypatch):
    mock_packages(monkeypatch)
    monkeypatch.setattr(launch, "_cuda_report", lambda: {"checked": True, "available": False, "devices": []})
    monkeypatch.setenv("FORZA_LINK_KEY", "not-output")
    path = artifact(tmp_path)
    calls = []
    fake_predictor = SimpleNamespace(load_predictor=lambda model_path: calls.append(model_path))
    monkeypatch.setitem(launch.sys.modules, "forza_ai.policies.predictor", fake_predictor)
    assert launch.doctor_report("desktop")["model"] is None
    assert not calls
    report = launch.doctor_report("desktop", model_path=path)
    assert report["ready"] and report["model"]["weights_loaded"]
    assert calls == [path]


def test_example_profiles_require_actual_site_values():
    configs = Path(__file__).resolve().parents[1] / "configs"
    for path in configs.glob("*.example.json"):
        with pytest.raises(launch.ProfileError, match="null placeholder"):
            launch.build_plan(path)


def test_windows_launchers_use_argument_arrays_and_no_policy_changes():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for role in ("game", "desktop"):
        script = (scripts / f"start-{role}.ps1").read_text()
        assert "[Parameter(Mandatory=$true)]" in script
        assert '".venv/Scripts/python.exe"' in script
        assert "& $taskPython @taskArgs" in script
        assert "Invoke-Expression" not in script and "Set-ExecutionPolicy" not in script
        assert "New-NetFirewallRule" not in script and "$env:FORZA_LINK_KEY" not in script


def test_driving_profile_flags(tmp_path, game):
    game['control'] = {'auto_pedals': True, 'direct_vjoy': True, 'pedal_override': .1}
    plan = launch.build_plan(save(tmp_path, game), assist=True, run_id='drive')
    assert {'--auto-pedals', '--direct-vjoy', '--assist'} <= set(plan.argv)
    assert value_after(plan, '--pedal-override') == '0.1'
    assert '--shadow' not in plan.argv


def test_v2_model_profile(tmp_path, desktop):
    path = artifact(tmp_path)
    meta = json.loads((path / 'metadata.json').read_text())
    meta.update(format_version=2, architecture='pilotnet_driving_v2')
    (path / 'metadata.json').write_text(json.dumps(meta))
    desktop['policy'] = {'kind': 'model', 'model_path': str(path)}
    assert '--model' in launch.build_plan(save(tmp_path, desktop), run_id='v2').argv
