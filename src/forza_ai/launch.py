"""Validated, shell-free two-PC launch profiles and read-only setup checks.

    python -m forza_ai.launch --profile configs/game.local.json --check
    python -m forza_ai.launch doctor --role desktop

Profiles never contain the shared key: set FORZA_LINK_KEY in the process
environment on each host. Checking a profile never opens the wheel, starts a
server, creates output directories, or changes the machine's configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata
import importlib.util
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid


class ProfileError(ValueError):
    """A saved profile is incomplete, inconsistent, or unsupported."""


def _object(value, name, allowed, required=()):
    if not isinstance(value, dict):
        raise ProfileError(f"{name} must be a JSON object")
    if set(value) - set(allowed):
        raise ProfileError(f"unknown {name} fields: {', '.join(sorted(set(value) - set(allowed)))}")
    if set(required) - set(value):
        raise ProfileError(f"missing {name} fields: {', '.join(sorted(set(required) - set(value)))}")
    return value


def _unique_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ProfileError(f"duplicate JSON field: {key}")
        obj[key] = value
    return obj


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ProfileError("non-finite JSON number")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        # Do not echo arbitrary JSON contents, which could contain a secret.
        raise ProfileError(f"could not read valid JSON from {path} ({type(error).__name__})") from error


def _number(value, name, minimum=0, maximum=None, *, positive=False, integer=False):
    expected = type(value) is int if integer else type(value) in (int, float)
    try:
        finite = expected and math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or value < minimum
            or (positive and value == minimum) or (maximum is not None and value > maximum)):
        kind = "integer" if integer else "number"
        bounds = f"greater than {minimum}" if positive else f"at least {minimum}"
        if maximum is not None:
            bounds += f" and at most {maximum}"
        raise ProfileError(f"{name} must be a finite {kind} {bounds}")
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ProfileError(f"{name} must be true or false")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ProfileError(f"{name} must be a nonempty string; replace the example's null placeholder")
    return value


def _path(value, name, base, *, file=False, directory=False):
    path = Path(_text(value, name)).expanduser()
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if file and not path.is_file():
        raise ProfileError(f"{name} must point to an existing file: {path}")
    if directory and not path.is_dir():
        raise ProfileError(f"{name} must point to an existing directory: {path}")
    return path


def _ipv4(value, name, *, bind=False):
    try:
        address = ipaddress.IPv4Address(_text(value, name))
    except ipaddress.AddressValueError as error:
        raise ProfileError(f"{name} must be the numeric IPv4 address from that PC, not a placeholder or hostname") from error
    if address.is_multicast or str(address) == "255.255.255.255" or (address.is_unspecified and not bind):
        raise ProfileError(f"{name} must be a unicast IPv4 address")
    return str(address)


def _button(value, name, *, required=False):
    if value is None:
        if required:
            raise ProfileError(f"{name} is required: replace null with your verified zero-based SDL index from utils/test.py wheel")
        return None
    return _number(value, name, maximum=127, integer=True)


def _model_files(path):
    if not (path / "model.pt").is_file() or not (path / "metadata.json").is_file():
        raise ProfileError("model_path must be an exported artifact directory containing model.pt and metadata.json")
    metadata = _read_json(path / "metadata.json")
    if (not isinstance(metadata, dict) or type(metadata.get("format_version")) is not int
            or (metadata["format_version"], metadata.get("architecture")) not in {(1, "pilotnet_speed_v1"), (2, "pilotnet_driving_v2")}
            or metadata.get("image_stage") != "road_crop" or not isinstance(metadata.get("preprocessing"), dict)):
        raise ProfileError("unsupported model artifact metadata; export a supported v1 steering or v2 driving road-crop artifact")
    return metadata


@dataclass(frozen=True)
class LaunchPlan:
    role: str
    argv: tuple[str, ...]
    run_dir: Path
    mode: str
    recording_dir: Path | None = None


def build_plan(profile_path, *, expected_role=None, assist=False, run_id=None):
    """Validate configuration and build argv without I/O beyond reading files.

    Paths resolve beside the profile, never relative to the caller's cwd. The
    random run directory is reserved only when launching; --check is read-only.
    """
    profile_path = Path(profile_path).resolve()
    profile = _read_json(profile_path)
    _object(profile, "profile", {"schema_version", "role", "mode", "inference", "capture", "buttons",
                                 "control", "telemetry", "run", "recording", "policy"},
            {"schema_version", "role", "inference"})
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise ProfileError("schema_version must be integer 1")
    role = profile["role"]
    if role not in ("game", "desktop"):
        raise ProfileError("role must be game or desktop")
    if expected_role is not None and role != expected_role:
        raise ProfileError(f"profile role {role} does not match requested role {expected_role}")
    if role == "desktop" and assist:
        raise ProfileError("--assist applies only to a game profile")
    game_only = {"mode", "capture", "buttons", "control", "telemetry", "recording"}
    if role == "desktop" and set(profile) & game_only:
        raise ProfileError("desktop profile cannot contain game-only fields")
    if role == "game" and "policy" in profile:
        raise ProfileError("game profiles use remote inference; policy belongs to the desktop profile")
    base = profile_path.parent
    run = _object(profile.get("run", {}), "run", {"output_dir", "duration_s", "interactive", "dashboard_port"})
    if role == "desktop" and set(run) - {"output_dir"}:
        raise ProfileError("desktop run only accepts output_dir")
    output = _path(run.get("output_dir", "../runs"), "run.output_dir", base)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:12]
    if not isinstance(run_id, str) or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ProfileError("run_id must be a single alphanumeric path component")
    run_dir = output / f"{role}-{run_id}"
    if run_dir.exists():
        raise ProfileError("run output directory already exists; choose a fresh run")
    recording_dir = None

    if role == "desktop":
        link = _object(profile["inference"], "inference", {"bind", "port", "timeout_s"}, {"bind"})
        bind = _ipv4(link["bind"], "inference.bind", bind=True)
        port = _number(link.get("port", 8765), "inference.port", minimum=1, maximum=65535, integer=True)
        timeout = _number(link.get("timeout_s", 2.0), "inference.timeout_s", positive=True)
        policy = _object(profile.get("policy"), "policy", {"kind", "target_angle_deg", "model_path"}, {"kind"})
        argv = [sys.executable, "-m", "forza_ai.inference_server", "--bind", bind,
                "--port", str(port), "--timeout", str(timeout)]
        if policy["kind"] == "fixed":
            if "model_path" in policy:
                raise ProfileError("fixed policy must not contain model_path")
            target = _number(policy.get("target_angle_deg"), "policy.target_angle_deg", minimum=-15, maximum=15)
            argv += ["--test-target", str(target)]
            mode = "fixed_target_test_not_a_driving_model"
        elif policy["kind"] == "model":
            if "target_angle_deg" in policy:
                raise ProfileError("model policy must not contain target_angle_deg")
            artifact = _path(policy.get("model_path"), "policy.model_path", base, directory=True)
            _model_files(artifact)
            argv += ["--model", str(artifact)]
            mode = "model"
        else:
            raise ProfileError("policy.kind must be fixed or model")
    else:
        mode = profile.get("mode", "shadow")
        if mode not in ("shadow", "manual", "assist"):
            raise ProfileError("mode must be shadow, manual, or assist")
        if assist:
            mode = "assist"
        link = _object(profile["inference"], "inference", {"host", "port", "timeout_s"}, {"host"})
        host = _ipv4(link["host"], "inference.host")
        port = _number(link.get("port", 8765), "inference.port", minimum=1, maximum=65535, integer=True)
        timeout = _number(link.get("timeout_s", 0.2), "inference.timeout_s", positive=True)
        capture = _object(profile.get("capture"), "capture", {"config_path", "hz", "game_process"}, {"config_path"})
        capture_path = _path(capture["config_path"], "capture.config_path", base, file=True)
        # This only parses numeric transforms, never opens capture or hardware.
        from forza_ai.capture_config import CaptureConfig
        CaptureConfig.from_json(capture_path)
        capture_hz = _number(capture.get("hz", 30), "capture.hz", minimum=1, maximum=240)
        game_process = _text(capture.get("game_process", "ForzaHorizon4.exe"), "capture.game_process")
        if not game_process.lower().endswith(".exe") or any(c in game_process for c in "/\\\r\n"):
            raise ProfileError("capture.game_process must be an EXE basename")
        buttons = _object(profile.get("buttons"), "buttons", {"takeover", "arm", "route", "virtual_map"}, {"takeover"})
        indices = {name: _button(buttons.get(name), f"buttons.{name}", required=name == "takeover")
                   for name in ("takeover", "arm", "route")}
        selected = [index for index in indices.values() if index is not None]
        if len(selected) != len(set(selected)):
            raise ProfileError("takeover, arm, and route buttons must use distinct verified SDL indices")
        mappings = buttons.get("virtual_map", [])
        if not isinstance(mappings, list):
            raise ProfileError("buttons.virtual_map must be a list of {physical, virtual} objects")
        seen_physical, seen_virtual = set(), set()
        for mapping in mappings:
            _object(mapping, "button mapping", {"physical", "virtual"}, {"physical", "virtual"})
            physical = _button(mapping["physical"], "button mapping.physical", required=True)
            virtual = _number(mapping["virtual"], "button mapping.virtual", minimum=1, maximum=128, integer=True)
            if physical in seen_physical or virtual in seen_virtual:
                raise ProfileError("each physical and virtual button can appear only once in virtual_map")
            if physical in selected:
                raise ProfileError("control buttons cannot also be forwarded to the game in virtual_map")
            seen_physical.add(physical)
            seen_virtual.add(virtual)
        control = _object(profile.get("control", {}), "control",
                          {"hz", "policy_hz", "torque_limit", "kp", "kd", "target_rate_deg_s", "target_limit_deg",
                           "auto_pedals", "direct_vjoy", "pedal_override"})
        defaults = {"hz": 100, "policy_hz": 30, "torque_limit": 0.15, "kp": 0.008, "kd": 0.001,
                    "target_rate_deg_s": 60, "target_limit_deg": 90}
        config = dict(defaults, **control)
        _number(config["hz"], "control.hz", minimum=10, maximum=1000)
        _number(config["policy_hz"], "control.policy_hz", minimum=1, maximum=240)
        _number(config["torque_limit"], "control.torque_limit", maximum=1, positive=True)
        _number(config["kp"], "control.kp")
        _number(config["kd"], "control.kd")
        _number(config["target_rate_deg_s"], "control.target_rate_deg_s", positive=True)
        _number(config["target_limit_deg"], "control.target_limit_deg", maximum=450, positive=True)
        telemetry = _object(profile.get("telemetry", {}), "telemetry", {"port"})
        telemetry_port = _number(telemetry.get("port", 9999), "telemetry.port", minimum=1, maximum=65535, integer=True)
        duration = _number(run.get("duration_s", 0), "run.duration_s")
        interactive = _boolean(run.get("interactive", True), "run.interactive")
        dashboard_port = run.get("dashboard_port")
        if dashboard_port is not None:
            _number(dashboard_port, "run.dashboard_port", minimum=1, maximum=65535, integer=True)
        recording = _object(profile.get("recording", {}), "recording", {"enabled", "include_manual", "output_dir"})
        recording_enabled = _boolean(recording.get("enabled", False), "recording.enabled")
        record_manual = _boolean(recording.get("include_manual", False), "recording.include_manual")
        if record_manual and not recording_enabled:
            raise ProfileError("recording.include_manual requires recording.enabled=true")
        if "output_dir" in recording:
            recording_root = _path(recording["output_dir"], "recording.output_dir", base)
        else:
            recording_root = run_dir
        if recording_enabled:
            recording_dir = recording_root / ("session" if recording_root == run_dir else f"session-{run_id}")
            if recording_dir.exists():
                raise ProfileError("recording directory already exists; choose a fresh session")
        argv = [sys.executable, "-m", "forza_ai.runtime", "--backend", "windows",
                "--inference-host", host, "--inference-port", str(port), "--network-timeout", str(timeout),
                "--capture-config", str(capture_path), "--capture-hz", str(capture_hz),
                "--game-process", game_process, "--telemetry-port", str(telemetry_port),
                "--takeover-button", str(indices["takeover"]), "--duration", str(duration),
                "--run-report", str(run_dir / "report.json")]
        if _boolean(control.get('auto_pedals', False), 'control.auto_pedals'):
            argv += ['--auto-pedals']
        if _boolean(control.get('direct_vjoy', False), 'control.direct_vjoy'):
            argv += ['--direct-vjoy']
        if 'pedal_override' in control:
            threshold = _number(control['pedal_override'], 'control.pedal_override', positive=True, maximum=1)
            if threshold == 1:
                raise ProfileError('control.pedal_override must be below 1')
            argv += ['--pedal-override', str(threshold)]
        if mode != "manual":
            argv += ["--" + mode]
        for name in ("arm", "route"):
            if indices[name] is not None:
                argv += [f"--{name}-button", str(indices[name])]
        for mapping in mappings:
            argv += ["--button-map", f"{mapping['physical']}:{mapping['virtual']}"]
        for name, flag in (("hz", "control-hz"), ("policy_hz", "policy-hz"), ("torque_limit", "torque-limit"),
                           ("kp", "kp"), ("kd", "kd"), ("target_rate_deg_s", "target-rate"), ("target_limit_deg", "target-limit")):
            argv += ["--" + flag, str(config[name])]
        if interactive:
            argv += ["--interactive"]
        if dashboard_port is not None:
            argv += ["--dashboard-port", str(dashboard_port)]
        if duration > 0 and duration * config["hz"] <= 100_000:
            argv += ["--status-csv", str(run_dir / "control.csv")]
        if recording_dir is not None:
            argv += ["--record-session", str(recording_dir)]
            if record_manual:
                argv += ["--record-manual"]
    return LaunchPlan(role, tuple(argv), run_dir, mode, recording_dir)


def _redact(text):
    key = os.environ.get("FORZA_LINK_KEY")
    return text.replace(key, "<redacted>") if key else text


def _emit(value):
    print(_safe_json(value))


def _safe_json(value):
    # Redact strings before encoding, so keys containing quotes/newlines cannot
    # evade redaction through JSON escaping or damage the JSON output syntax.
    def scrub(item):
        if isinstance(item, str):
            return _redact(item)
        if isinstance(item, dict):
            return {_redact(key): scrub(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [scrub(entry) for entry in item]
        return item
    return json.dumps(scrub(value), indent=2)


def _package(distribution, module):
    result = {"distribution": distribution, "installed": False, "version": None}
    try:
        result["installed"] = importlib.util.find_spec(module) is not None
        result["version"] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception as error:
        result["error_type"] = type(error).__name__
    return result


def _cuda_report():
    report = {"checked": False, "available": False, "devices": []}
    try:
        torch = importlib.import_module("torch")
        report.update(checked=True, available=bool(torch.cuda.is_available()),
                      torch_version=str(torch.__version__), build_cuda=getattr(torch.version, "cuda", None))
        if report["available"]:
            report["devices"] = [str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())]
    except Exception as error:
        report["error_type"] = type(error).__name__
    return report


def doctor_report(role, *, model_path=None):
    """Read setup state without wheel handles, listeners, or package changes.

    CUDA detection queries Torch's installed runtime. It does not install
    drivers or prove workload speed. Network reachability remains untested.
    """
    if role not in ("game", "desktop"):
        raise ProfileError("role must be game or desktop")
    packages = [("numpy", "numpy"), ("Pillow", "PIL")]
    packages += ([("torch", "torch")] if role == "desktop" else
                 [("PySDL2", "sdl2"), ("pysdl2-dll", "sdl2dll"), ("pyvjoyffb", "pyvjoy"),
                  ("dxcam", "dxcam"), ("opencv-python", "cv2")])
    installed = [_package(distribution, module) for distribution, module in packages]
    python_ready = sys.version_info >= (3, 10)
    os_ready = role == "desktop" or sys.platform == "win32"
    report = {"schema_version": 1, "role": role, "scope": "read_only_environment_check",
              "python": {"version": platform.python_version(), "supported": python_ready},
              "platform": {"system": platform.system(), "supported": os_ready}, "packages": installed,
              "network": {"shared_key_present": bool(os.environ.get("FORZA_LINK_KEY")),
                          "requirement": "same trusted LAN; numeric IPv4 and matching TCP port/key on both hosts",
                          "connectivity_checked": False},
              "hardware_opened": False, "model": None}
    if role == "desktop":
        torch_installed = next(p["installed"] for p in installed if p["distribution"] == "torch")
        report["cuda"] = _cuda_report() if torch_installed else {"checked": False, "available": False, "devices": []}
        report["training_ready"] = (python_ready and all(p["installed"] for p in installed)
                                    and report["cuda"]["checked"])
        report["gpu_training_ready"] = report["training_ready"] and report["cuda"]["available"]
    if model_path is not None:
        path = Path(model_path).expanduser().resolve()
        model = {"path": str(path), "files_and_metadata_valid": False, "weights_loaded": False}
        try:
            _model_files(path)
            model["files_and_metadata_valid"] = True
            from forza_ai.policies.predictor import load_predictor
            load_predictor(path)
            model["weights_loaded"] = True
        except Exception as error:
            model["error_type"] = type(error).__name__
        report["model"] = model
    report["ready"] = (python_ready and os_ready and all(p["installed"] for p in installed)
                       and (role != "desktop" or report["training_ready"])
                       and report["network"]["shared_key_present"]
                       and (report["model"] is None or report["model"]["weights_loaded"]))
    report["not_checked"] = ["LAN connection, latency, and firewall rules", "actual game capture and telemetry",
                             "physical steering, calibration, and button identities"]
    return report


def _plan_report(plan):
    key_present = bool(os.environ.get("FORZA_LINK_KEY"))
    platform_ready = plan.role == "desktop" or sys.platform == "win32"
    return {"schema_version": 1, "profile_valid": True, "role": plan.role, "mode": plan.mode,
            "argv": list(plan.argv), "run_dir": str(plan.run_dir),
            "recording_dir": str(plan.recording_dir) if plan.recording_dir is not None else None,
            "shared_key_present": key_present, "platform_ready": platform_ready,
            "ready": key_present and platform_ready and sys.version_info >= (3, 10),
            "note": "Configuration only. Run doctor separately; real hardware/network operation is unverified."}


def _wait_gracefully(process, timeout):
    """Wait through repeated parent interrupts without resetting the deadline."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            returncode = process.poll()
            if returncode is not None:
                return returncode
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            return process.wait(timeout=remaining)
        except KeyboardInterrupt:
            continue


def launch(plan, *, shutdown_grace_s=15.0):
    """Run with the same Python and inherited environment/stdin, never a shell."""
    _number(shutdown_grace_s, "shutdown_grace_s", positive=True)
    report = _plan_report(plan)
    if not report["ready"]:
        raise ProfileError("launch requires Python 3.10+, FORZA_LINK_KEY, and Windows for the game role; run --check")
    # Atomic leaf reservation prevents reuse even when two launchers start together.
    plan.run_dir.parent.mkdir(parents=True, exist_ok=True)
    plan.run_dir.mkdir(exist_ok=False)
    (plan.run_dir / "launch.json").write_text(_safe_json(report) + "\n", encoding="utf-8")
    # Inherit the console/process group: its Ctrl+C reaches the runtime too.
    # subprocess.run kills its child on KeyboardInterrupt, which can cut off
    # recorder drain. Own the process and allow bounded normal cleanup instead.
    process = None
    interrupted, forced = False, False
    failure = None
    try:
        process = subprocess.Popen(list(plan.argv), shell=False)
        try:
            return process.wait()
        except KeyboardInterrupt:
            interrupted = True
            try:
                print("Stopping: waiting for runtime cleanup (Ctrl+C already sent to its console).", flush=True)
            except (KeyboardInterrupt, OSError):
                pass  # Diagnostic output must not abort the cleanup grace period.
            try:
                return _wait_gracefully(process, shutdown_grace_s)
            except subprocess.TimeoutExpired:
                forced = True
                process.terminate()
                try:
                    return _wait_gracefully(process, 1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return _wait_gracefully(process, 5.0)
    except BaseException as error:
        failure = type(error).__name__
        if process is not None and process.poll() is None:
            forced = True
            process.kill()
            try:
                _wait_gracefully(process, 5.0)
            except (subprocess.TimeoutExpired, OSError):
                pass
        raise
    finally:
        outcome = {"returncode": process.returncode if process is not None else None,
                   "interrupted": interrupted, "forced_termination": forced,
                   "error_type": failure}
        (plan.run_dir / "exit.json").write_text(json.dumps(outcome) + "\n", encoding="utf-8")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "doctor":
        parser = argparse.ArgumentParser(description="Read-only package/CUDA checks; no wheel or network actuation")
        parser.add_argument("--role", choices=("game", "desktop"), required=True)
        parser.add_argument("--model", type=Path, help="optionally verify an exported model by loading its weights on CPU")
        args = parser.parse_args(argv[1:])
        report = doctor_report(args.role, model_path=args.model)
        _emit(report)
        return 0 if report["ready"] else 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--role", choices=("game", "desktop"), help="reject a profile for the other PC")
    parser.add_argument("--check", action="store_true", help="validate and print redacted argv; never start anything")
    parser.add_argument("--assist", action="store_true", help="explicitly override a game profile to engage steering")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.profile, expected_role=args.role, assist=args.assist)
        report = _plan_report(plan)
        _emit(report)
        if args.check:
            return 0 if report["ready"] else 1
        return launch(plan)
    except (OSError, ValueError) as error:
        parser.error(_redact(str(error)))


if __name__ == "__main__":
    raise SystemExit(main())
