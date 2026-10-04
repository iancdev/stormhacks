"""Run physical steering tests or an exported camera-and-speed driving policy.

Example without Windows drivers:
    PYTHONPATH=src python -m forza_ai.runtime --backend sim --assist --duration 5

A fixed target/sweep is a stationary control test, never a driving policy.
Exported models require live RGB road capture and fresh, race-on telemetry.
"""

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time

from forza_ai.contracts import ActuationExpired, ControlMode, ModelObservation
from forza_ai.control import SteeringConfig, SteeringController
from forza_ai.policies.placeholder import FixedAnglePolicy, SweepPolicy, TestObservation
from forza_ai.simulation import SimulatedAdapter
from forza_ai.policy_worker import PolicyWorker


def _read_console(events, stop):
    while not stop.is_set():
        line = sys.stdin.readline()
        if not line:
            return
        try:
            events.put(line.strip().lower(), timeout=0.1)
        except queue.Full:
            print("Control command queue is full; try again.", flush=True)


def run(adapter, policy, *, duration=5.0, control_hz=100.0, policy_hz=30.0,
        assist=False, takeover_button=None, interactive=False, receiver=None,
        config=None, status_path=None, command_queue=None, camera=None, shadow=False,
        progress=False, arm_timeout=5.0, foreground_guard=None, arm_button=None,
        route_button=None, recorder=None, record_manual=False, takeover_settle_ms=100.0,
        dashboard_port=None, run_report=None, direct_vjoy=False, direct_override_deg=20.0,
        auto_pedals=False, pedal_override=0.05, dashboard_host="127.0.0.1"):
    """Own the adapter lifecycle, including cleanup on I/O or policy exceptions.

    ``direct_vjoy`` is the fallback when the motor path is unavailable: while
    assistance is engaged, the controller's rate-limited target is written to
    vJoy steering instead of driving the motor (torque stays zero), only after
    the same freshness/foreground/telemetry checks that gate motor output.
    Otherwise the measured wheel is forwarded as usual. Turning the physical
    wheel beyond ``direct_override_deg`` is a human takeover, like the button.

    Zero duration runs until interrupted. Status output is optional and buffered
    in memory during this finite test; unlimited runs cannot collect an unbounded
    status log. Integrated training recording uses a separate bounded writer;
    run metrics and the live dashboard use bounded memory for unlimited runs.
    """
    worker = None
    stop_console = threading.Event()
    rows = []
    summary = {"ticks": 0, "max_abs_torque": 0.0, "mode": "manual", "reason": "not_started",
               "hardware_mode": "simulation" if isinstance(adapter, SimulatedAdapter) else "physical adapter",
               "autocenter_disabled_confirmed": getattr(adapter, "autocenter_disabled_confirmed", None),
               "actuation": "direct_vjoy" if direct_vjoy else "motor"}
    error_text = None
    progress_worker = None
    dashboard = None
    from forza_ai.metrics import RunMetrics
    from forza_ai.reporting import LiveRates
    metrics = RunMetrics()
    live_rates = LiveRates()
    try:
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative")
        if not math.isfinite(control_hz) or not 10 <= control_hz <= 1000:
            raise ValueError("control_hz must be between 10 and 1000")
        if status_path is not None and (duration == 0 or duration * control_hz > 100_000):
            raise ValueError("status logging is limited to 100,000 ticks; use a finite test")
        control_buttons = [value for value in (takeover_button, arm_button, route_button) if value is not None]
        if any(type(value) is not int or value < 0 for value in control_buttons):
            raise ValueError("control buttons must be zero-based nonnegative integers")
        if len(set(control_buttons)) != len(control_buttons):
            raise ValueError("takeover, arm, and route buttons must be different")
        button_count = getattr(adapter, "button_count", None)
        if button_count is not None and any(value >= button_count for value in control_buttons):
            raise ValueError(f"control button is absent; device has {button_count} buttons")
        if not math.isfinite(arm_timeout) or arm_timeout <= 0:
            raise ValueError("arm_timeout must be positive")
        if shadow and assist:
            raise ValueError("shadow mode cannot engage assistance")
        if direct_vjoy and (not math.isfinite(direct_override_deg) or direct_override_deg <= 0):
            raise ValueError("direct_override_deg must be finite and positive")
        if auto_pedals and not getattr(policy, 'driving', False):
            raise ValueError('auto_pedals requires a v2 driving policy; steering-only models cannot control pedals')
        if not math.isfinite(pedal_override) or not 0 < pedal_override < 1:
            raise ValueError('pedal_override must be within (0,1)')
        if (direct_vjoy or auto_pedals) and not hasattr(adapter, 'write_virtual_state_before'):
            raise ValueError('AI virtual output requires a deadline-aware virtual adapter')
        model_mode = bool(getattr(policy, "requires_camera", False))
        if model_mode and (camera is None or receiver is None):
            raise ValueError("model policy requires a camera and telemetry receiver")
        if recorder is not None and (camera is None or receiver is None):
            raise ValueError("integrated recording requires a camera and telemetry receiver")
        if record_manual and recorder is None:
            raise ValueError("record_manual requires a session recorder")
        if not math.isfinite(takeover_settle_ms) or takeover_settle_ms < 0:
            raise ValueError("takeover_settle_ms must be finite and nonnegative")
        controller = SteeringController(config)
        events = command_queue if command_queue is not None else queue.Queue(maxsize=64)
        if interactive:
            threading.Thread(target=_read_console, args=(events, stop_console), daemon=True).start()
            print("Commands: arm, manual, route_start, route_complete, route_abort, quit + Enter.")
        if receiver is not None:
            receiver.start()
        if camera is not None:
            camera.start()
        if recorder is not None:
            recorder.start()
        if dashboard_port is not None:
            from forza_ai.dashboard import Dashboard
            dashboard = Dashboard(events, host=dashboard_host, port=dashboard_port)
            dashboard.start()
            if dashboard.address[0] == "0.0.0.0":
                print(f"Dashboard listening on 0.0.0.0:{dashboard.address[1]}; "
                      f"open http://<THIS-PC-LAN-IP>:{dashboard.address[1]} in your browser.", flush=True)
            else:
                print(f"Live dashboard: {dashboard.url}", flush=True)
        if progress:
            from forza_ai.reporting import ProgressReporter
            progress_worker = ProgressReporter()
            progress_worker.start()
        worker = PolicyWorker(policy, hz=policy_hz)
        worker.start()
        handled_failure = 0

        def consume_policy_failure():
            nonlocal handled_failure
            generation, reason = worker.failure_state()
            if generation > handled_failure:
                handled_failure = generation
                worker.invalidate("inference_failure")
                return True
            return False

        period = 1.0 / control_hz
        started = time.monotonic()
        deadline = started
        pending_arm = assist
        active_arm_until = started + arm_timeout if assist else 0.0
        previous_buttons = None
        previous_pedal_pressed = False
        route_active = False
        human_control = bool(record_manual)
        expert_after_ns = 0
        next_dashboard_ns = 0
        if assist:
            metrics.event("arm", time.monotonic_ns())
        while duration == 0 or time.monotonic() - started < duration:
            engaged_at_tick_start = controller.mode == ControlMode.ASSIST
            now_ns = time.monotonic_ns()
            wheel = adapter.read_state(now_ns)
            # Reacquire time after device I/O to account for a slow read.
            now_ns = time.monotonic_ns()
            wheel_error = controller.wheel_error(wheel, now_ns)
            if wheel_error:
                raise RuntimeError(wheel_error)
            # Read each cached producer with its own current clock, not a time
            # captured before another thread could publish a newer sample.
            vehicle = receiver.latest() if receiver is not None else None
            frame = camera.latest() if camera is not None else None
            input_error = None
            if model_mode:
                causal_vehicle = receiver.at_or_before(frame.timestamp_ns) if frame is not None else None
                if foreground_guard is not None and not foreground_guard.is_active():
                    input_error = "game_not_foreground"
                elif vehicle is None:
                    input_error = "telemetry_unavailable"
                elif not vehicle.is_race_on:
                    input_error = "race_inactive"
                elif frame is None:
                    input_error = "frame_unavailable"
                elif causal_vehicle is None:
                    input_error = "no_causal_telemetry"
                elif not causal_vehicle.is_race_on:
                    input_error = "frame_race_inactive"
                if input_error:
                    worker.invalidate(input_error)
                    if controller.mode == ControlMode.ASSIST:
                        human_control = False
                        controller.disengage(input_error)
                else:
                    worker.publish(ModelObservation(frame, causal_vehicle))
            else:
                worker.publish(TestObservation(wheel, vehicle))
            command = worker.latest()
            policy_failed = consume_policy_failure()
            if policy_failed:
                command = None
                pending_arm = False
                human_control = False
                input_error = "inference_failure"
                controller.disengage("inference_failure")
            buttons = set(wheel.buttons)
            # A held startup/arm button is never an engagement edge.
            rising = set() if previous_buttons is None else buttons - previous_buttons
            previous_buttons = buttons
            takeover = takeover_button is not None and takeover_button in wheel.buttons
            explicit_takeover = takeover_button is not None and takeover_button in rising
            wheel_arm = arm_button is not None and arm_button in rising
            wheel_route = route_button is not None and route_button in rising
            if direct_vjoy and engaged_at_tick_start and abs(wheel.angle_deg) > direct_override_deg:
                # No motor holds the wheel in this mode, so a deliberate turn is the human taking over.
                takeover = explicit_takeover = True
            pedal_pressed = auto_pedals and not shadow and max(wheel.throttle, wheel.brake) >= pedal_override
            if pedal_pressed:
                takeover = True
                explicit_takeover = explicit_takeover or not previous_pedal_pressed
            previous_pedal_pressed = pedal_pressed
            local_events = []
            if wheel_arm and not shadow:
                local_events.append("arm")
            if wheel_route:
                local_events.append("route_complete" if route_active else "route_start")
            should_quit = False
            # Bounded external work; a UI flood cannot starve the motor loop.
            for _ in range(64):
                try:
                    local_events.append(events.get_nowait())
                except queue.Empty:
                    break
            for event in local_events:
                if event in ("quit", "q"):
                    should_quit = True
                elif event in ("manual", "takeover"):
                    takeover = True
                    explicit_takeover = True
                elif event == "arm":
                    if not shadow and not engaged_at_tick_start and not policy_failed:
                        pending_arm = True
                        active_arm_until = time.monotonic() + arm_timeout
                        metrics.event("arm", time.monotonic_ns())
                elif event == "route_start" and not route_active:
                    metrics.event(event, time.monotonic_ns())
                    route_active = True
                elif event in ("route_complete", "route_abort") and route_active:
                    metrics.event(event, time.monotonic_ns())
                    route_active = False
            if should_quit:
                summary["stop_reason"] = "quit"
                break
            if takeover or time.monotonic() > active_arm_until:
                pending_arm = False
            if takeover:
                worker.invalidate("manual_takeover")
                command = None
                if explicit_takeover:
                    human_control = True
                    if engaged_at_tick_start:
                        metrics.event("human_takeover", time.monotonic_ns())
            engage_now = pending_arm and controller.command_error(command, time.monotonic_ns()) is None
            if engage_now:
                pending_arm = False
            now_ns = time.monotonic_ns()
            status = controller.step(wheel, command, now_ns, engage=engage_now, takeover=takeover)
            # Motor mode: always forward the measured inputs, never the model's target angle.
            # Direct-vJoy mode writes after validation below, where motor output would happen.
            if not direct_vjoy and not auto_pedals:
                adapter.write_virtual_state(wheel)
            live_error = None
            if model_mode and status.mode == ControlMode.ASSIST:
                current_vehicle = receiver.latest()
                if current_vehicle is None or not current_vehicle.is_race_on:
                    live_error = "telemetry_unavailable_or_paused"
                elif foreground_guard is not None and not foreground_guard.is_active():
                    live_error = "game_not_foreground"
                if live_error:
                    worker.invalidate(live_error)
                    command = None
                    input_error = live_error
            if consume_policy_failure():
                pending_arm = False
                human_control = False
                command = None
                live_error = "inference_failure"
                input_error = live_error
            # All producer/OS reads precede this clock: never validate a timestamp
            # and then perform a potentially slow external read before the motor.
            after_io = time.monotonic_ns()
            if after_io - now_ns > controller.config.max_wheel_age_ns:
                raise RuntimeError("hardware_output_stalled")
            # Revalidate ALL ages after I/O, without integrating the controller
            # twice. A sample can age out even when this write was individually fast.
            wheel_error = controller.wheel_error(wheel, after_io)
            command_error = controller.command_error(command, after_io)
            output_error = wheel_error or live_error or (command_error if status.mode == ControlMode.ASSIST else None)
            if output_error:
                controller.disengage(output_error, fault=wheel_error is not None)
                status = replace(status, mode=controller.mode, reason=output_error, torque=0.0)
            if direct_vjoy:
                status = replace(status, torque=0.0)
            virtual = virtual_driving_state(wheel, status, command, direct_vjoy, auto_pedals)
            try:
                if (direct_vjoy or auto_pedals) and status.mode == ControlMode.ASSIST:
                    actuation_deadline = min(wheel.timestamp_ns + controller.config.max_wheel_age_ns,
                                             command.valid_until_ns,
                                             command.generated_time_ns + controller.config.max_command_age_ns,
                                             command.observation_time_ns + controller.config.max_observation_age_ns)
                    if auto_pedals:
                        adapter.write_virtual_state_before(virtual, actuation_deadline)
                    else:
                        # Steering-only assistance preserves physical pedal input,
                        # including a human pressing both pedals simultaneously.
                        adapter.write_virtual_state_before(virtual, actuation_deadline, physical_pedals=True)
                elif direct_vjoy or auto_pedals:
                    adapter.write_virtual_state(virtual)
                if status.torque:
                    # Absolute host validity reaches the last native boundary;
                    # successful SDL preparation may still have consumed the budget.
                    actuation_deadline = min(wheel.timestamp_ns + controller.config.max_wheel_age_ns,
                                             command.valid_until_ns,
                                             command.generated_time_ns + controller.config.max_command_age_ns,
                                             command.observation_time_ns + controller.config.max_observation_age_ns)
                    adapter.set_torque_before(status.torque, actuation_deadline)
                else:
                    adapter.set_torque(0.0)
            except ActuationExpired:
                adapter.set_torque(0.0)
                if direct_vjoy or auto_pedals:
                    virtual = replace(wheel, throttle=0.0, brake=0.0) if auto_pedals else wheel
                    adapter.write_virtual_state(virtual)
                worker.invalidate("actuation_deadline_expired")
                pending_arm = False
                human_control = False
                command = None
                controller.disengage("actuation_deadline_expired", fault=True)
                status = replace(status, mode=controller.mode, reason=controller.reason, torque=0.0)
                input_error = controller.reason
            if explicit_takeover:
                # The handover interval starts after zero torque was sent, and
                # labels must be sampled after that boundary, not merely written later.
                expert_after_ns = time.monotonic_ns() + int(takeover_settle_ms * 1e6)
            if engaged_at_tick_start and status.mode != ControlMode.ASSIST:
                pending_arm = False
            if status.mode == ControlMode.ASSIST or status.mode == ControlMode.FAULT:
                human_control = False
            elif status.mode == ControlMode.TAKEOVER and status.reason != "manual_takeover":
                human_control = False
            expert = (human_control and status.mode in (ControlMode.MANUAL, ControlMode.TAKEOVER)
                      and wheel.timestamp_ns >= expert_after_ns and input_error is None)
            if recorder is not None:
                recorder.check()
                recorder.submit(wheel, vehicle, frame, status.mode, expert=expert, reason=status.reason)
            summary.update(ticks=summary["ticks"] + 1,
                           max_abs_torque=max(summary["max_abs_torque"], abs(status.torque)),
                           mode=status.mode.value, reason=status.reason,
                           actual_angle_deg=wheel.angle_deg, target_angle_deg=status.target_angle_deg,
                           speed_mps=vehicle.speed_mps if vehicle else None,
                           predicted_angle_deg=command.target_angle_deg if command else None,
                           observation_age_ms=(after_io - command.observation_time_ns) / 1e6 if command else None,
                           input_status=input_error or worker.unavailable_reason or "ready",
                           timestamp_ns=after_io, torque=status.torque,
                           inference_ms=command.inference_ms if command else None,
                           prediction_id=command.generated_time_ns if command else None,
                           shadow=shadow, allow_arm=not shadow, auto_pedals=auto_pedals,
                           predicted_throttle=command.throttle if command else None,
                           predicted_brake=command.brake if command else None,
                           output_throttle=virtual.throttle, output_brake=virtual.brake,
                           policy_name=getattr(policy, "name", "steering policy"),
                           expert_recording=expert if recorder is not None else False,
                           recording=recorder.stats if recorder is not None else None)
            metrics.update(summary)
            live_rates.update(after_io, frame=frame, prediction_id=summary["prediction_id"],
                              has_camera=camera is not None)
            if dashboard is not None and after_io >= next_dashboard_ns:
                dashboard_metrics = metrics.summary()
                summary["policy_worker"] = worker.stats() if hasattr(worker, "stats") else None
                dashboard.publish(dict(summary, route_active=route_active,
                                       human_interventions=dashboard_metrics["human_interventions"],
                                       metrics=dashboard_metrics, rates=live_rates.summary(),
                                       limits={"observation_age_ms": controller.config.max_observation_age_ns / 1e6,
                                               "target_angle_deg": controller.config.target_limit_deg,
                                               "torque": controller.config.torque_limit},
                                       network="LAN inference" if hasattr(policy, "host") else "local policy"))
                next_dashboard_ns = after_io + 100_000_000
            if progress_worker is not None:
                progress_worker.publish(summary)
            if status_path is not None:
                row = asdict(status)
                row.update(mode=status.mode.value, timestamp_ns=now_ns,
                           speed_mps=vehicle.speed_mps if vehicle else "",
                           requested_angle_deg=command.target_angle_deg if command else "",
                           observation_age_ms=summary["observation_age_ms"],
                           input_status=summary["input_status"])
                rows.append(row)
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()  # no backlog of catch-up motor commands
        summary.setdefault("stop_reason", "duration")
    except KeyboardInterrupt:
        # Operator stop is a normally completed recording, after writer drain.
        summary["stop_reason"] = "keyboard_interrupt"
        raise
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Release motor before joining threads or writing files. Preserve cleanup
        # order even if a driver throws; adapter.close() retries its own releases.
        stop_console.set()
        cleanup = [("zero_motor", lambda: adapter.set_torque(0.0)), ("adapter", adapter.close)]
        for name, resource in (("policy", worker), ("capture", camera),
                               ("telemetry", receiver), ("progress", progress_worker), ("dashboard", dashboard)):
            if resource is not None:
                cleanup.append((name, resource.close))
        cleanup_errors = []
        for name, operation in cleanup:
            try:
                operation()
            except BaseException as error:
                cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")
        if recorder is not None:
            try:
                recorder.close(completed=(error_text is None and not cleanup_errors))
            except BaseException as error:
                cleanup_errors.append(f"recording: {type(error).__name__}: {error}")
            summary["recording"] = recorder.stats
        summary["metrics"] = metrics.summary()
        summary["policy_worker"] = worker.stats() if hasattr(worker, "stats") else None
        report_error = error_text or ("Cleanup failed: " + "; ".join(cleanup_errors) if cleanup_errors else None)
        if status_path is not None:
            try:
                _save_status(status_path, rows, summary, report_error, cleanup_errors)
            except Exception:
                if error_text is None and not cleanup_errors:
                    raise
        if run_report is not None:
            path = Path(run_report)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                report = dict(summary, error=report_error, cleanup_errors=cleanup_errors, hardware_verified=False)
                temporary = path.with_name(path.name + ".tmp")
                temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                temporary.replace(path)
            except Exception:
                if error_text is None and not cleanup_errors:
                    raise
        if cleanup_errors and error_text is None:
            raise RuntimeError(report_error)
    return summary


def virtual_driving_state(wheel, status, command, direct_vjoy=False, auto_pedals=False):
    steer = status.target_angle_deg if direct_vjoy and status.mode == ControlMode.ASSIST else wheel.angle_deg
    throttle, brake = wheel.throttle, wheel.brake
    if auto_pedals:
        if status.mode == ControlMode.ASSIST:
            from forza_ai.contracts import DrivingPrediction
            if command is None:
                raise ValueError('missing driving command')
            prediction = DrivingPrediction(command.target_angle_deg, command.throttle, command.brake)
            throttle, brake = prediction.throttle, prediction.brake
        elif status.mode == ControlMode.FAULT or (status.mode == ControlMode.TAKEOVER and status.reason != 'manual_takeover'):
            throttle = brake = 0.0
        if brake > 0:
            throttle = 0.0
    return replace(wheel, angle_deg=steer, throttle=throttle, brake=brake)


def _save_status(status_path, rows, summary, error_text, cleanup_errors=()):
    """Preserve diagnostic evidence even when a run fails; hardware is closed."""
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(summary, error=error_text, cleanup_errors=list(cleanup_errors), hardware_verified=False)
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sim", "windows"), default="sim")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--assist", action="store_true", help="explicitly engage the selected steering policy")
    mode.add_argument("--shadow", action="store_true", help="show predictions; never apply AI torque")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", type=Path, help="export directory containing model.pt and metadata.json")
    selection.add_argument("--sweep", action="store_true", help="stationary 0/right/0/left/0 angle sequence")
    selection.add_argument("--inference-host", help="desktop LAN address for remote image-plus-speed inference")
    parser.add_argument("--auto-pedals", action="store_true",
                        help="use a v2 driving model for throttle/brake; either physical pedal takes over")
    parser.add_argument("--pedal-override", type=float, default=0.05,
                        help="physical pedal fraction that takes over (default .05)")
    parser.add_argument("--inference-port", type=int, default=8765)
    parser.add_argument("--network-timeout", type=float, default=0.2, help="total request deadline in seconds")
    parser.add_argument("--target-angle", type=float, default=None, help="fixed target, or positive sweep amplitude; default 5")
    parser.add_argument("--sweep-hold", type=float, default=2.0, help="seconds per stationary sweep target")
    parser.add_argument("--duration", type=float, help="seconds; 0 until stopped; default 5, sweep 12, model 0")
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--policy-hz", type=float, default=30.0)
    parser.add_argument("--torque-limit", type=float, default=0.15)
    parser.add_argument("--kp", type=float, default=0.008, help="PD gain per physical degree; requires tuning")
    parser.add_argument("--kd", type=float, default=0.001, help="PD damping gain per degree/second")
    parser.add_argument("--target-rate", type=float, default=60.0, help="maximum target slew in degrees/second")
    parser.add_argument("--target-limit", type=float, default=90.0, help="maximum absolute physical target degrees")
    parser.add_argument("--takeover-button", type=int, help="SDL zero-based index, from utils/test.py wheel")
    parser.add_argument("--arm-button", type=int, help="SDL button rising edge explicitly engages/re-engages")
    parser.add_argument("--route-button", type=int, help="SDL button rising edge toggles manual route start/complete")
    parser.add_argument("--button-map", action="append", default=[], metavar="SDL:VJOY",
                        help="explicit zero-based physical to one-based virtual button map; repeatable")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--telemetry", action="store_true", help="listen to FH4 Data Out on loopback")
    parser.add_argument("--telemetry-port", type=int, default=9999)
    crop_selection = parser.add_mutually_exclusive_group()
    crop_selection.add_argument("--crop", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                                help="absolute road crop for a custom recorder without masks")
    crop_selection.add_argument("--capture-config", type=Path,
                                help="record.py capture.json: exact monitor, crop, masks, saved-size transform")
    parser.add_argument("--display", type=int, default=None, help="DXcam output index, for --crop only")
    parser.add_argument("--capture-hz", type=float, default=30.0)
    parser.add_argument("--game-process", default="ForzaHorizon4.exe", help="foreground EXE required for live input")
    parser.add_argument("--status-csv", type=Path)
    parser.add_argument("--run-report", type=Path, help="bounded-memory metrics JSON, also for unlimited runs")
    parser.add_argument("--dashboard-host", default="127.0.0.1",
                        help="dashboard bind IP; 0.0.0.0 listens on all IPv4 interfaces (default localhost)")
    parser.add_argument("--dashboard-port", type=int, help="local browser dashboard port (0 selects a free port)")
    parser.add_argument("--record-session", type=Path, help="new session directory for integrated training data")
    parser.add_argument("--record-manual", action="store_true", help="declare initial manual driving as expert data")
    parser.add_argument("--takeover-settle-ms", type=float, default=100.0,
                        help="exclude the first milliseconds after a human takeover from training labels")
    parser.add_argument("--quiet", action="store_true", help="suppress twice-per-second progress")
    parser.add_argument("--direct-vjoy", action="store_true",
                        help="FALLBACK: send the AI's steering to vJoy instead of turning the wheel motor "
                             "(motor/haptics never opened; Forza must use the vJoy wheel)")
    parser.add_argument("--override-deg", type=float, default=20.0,
                        help="--direct-vjoy only: turning the wheel past this many degrees takes over")
    args = parser.parse_args(argv)
    from forza_ai.dashboard import validate_dashboard_host
    try:
        args.dashboard_host = validate_dashboard_host(args.dashboard_host)
    except ValueError as error:
        parser.error(str(error))
    live_mode = bool(args.model or args.inference_host)
    if args.auto_pedals and not live_mode:
        parser.error('--auto-pedals requires --model or --inference-host')
    needs_camera = live_mode or args.record_session is not None
    duration = args.duration if args.duration is not None else (0 if needs_camera else 12 if args.sweep else 5)
    if args.status_csv is not None and (duration == 0 or duration * args.control_hz > 100_000):
        parser.error("--status-csv requires a finite --duration of at most 100,000 control ticks")
    if args.record_manual and args.record_session is None:
        parser.error("--record-manual requires --record-session")
    if args.record_session is not None and args.record_session.exists():
        parser.error("--record-session must name a new directory")
    for output_path in (args.run_report, args.status_csv):
        if output_path is not None and output_path.exists():
            parser.error(f"output already exists; choose a new path: {output_path}")
    config = SteeringConfig(torque_limit=args.torque_limit, kp=args.kp, kd=args.kd,
                            target_rate_deg_s=args.target_rate, target_limit_deg=args.target_limit)
    mapping = {}
    for pair in args.button_map:
        try:
            physical, virtual = map(int, pair.split(":"))
        except ValueError:
            parser.error("--button-map must have the form SDL_INDEX:VJOY_BUTTON")
        if physical in mapping:
            parser.error("each physical button can be mapped only once")
        mapping[physical] = virtual
    camera = None
    foreground_guard = None
    capture_metadata = {}
    if needs_camera:
        if (args.crop is None and args.capture_config is None) or args.backend != "windows":
            parser.error("live model/recording requires --backend windows and --capture-config or --crop")
        from forza_ai.capture import DXCamCapture
        from forza_ai.foreground import ForegroundGameGuard
        foreground_guard = ForegroundGameGuard(args.game_process)
        frame_transform = None
        if args.capture_config:
            if args.display is not None:
                parser.error("capture config owns the monitor index; don't combine it with --display")
            from forza_ai.capture_config import CaptureConfig
            capture_config = CaptureConfig.from_json(args.capture_config)
            region, output_idx = capture_config.region, capture_config.output_idx
            frame_transform = capture_config.transform
            capture_metadata["capture_config"] = json.loads(args.capture_config.read_text(encoding="utf-8"))
        else:
            region, output_idx = tuple(args.crop), args.display or 0
            capture_metadata.update(crop=list(region), monitor=output_idx)
        camera = DXCamCapture(region=region, fps=args.capture_hz, output_idx=output_idx,
                              foreground_guard=foreground_guard, frame_transform=frame_transform)
    if live_mode:
        if args.target_angle is not None:
            parser.error("--target-angle cannot be combined with a driving model")
        if args.inference_host:
            from forza_ai.network import RemotePolicy
            policy = RemotePolicy(args.inference_host, port=args.inference_port, timeout_s=args.network_timeout, driving=args.auto_pedals)
        else:
            from forza_ai.policies.live import LiveModelPolicy
            policy = LiveModelPolicy(args.model)
    else:
        target = 5.0 if args.target_angle is None else args.target_angle
        if args.backend == "windows" and abs(target) > 15:
            parser.error("stationary Windows placeholder tests are limited to +/-15 degrees")
        policy = SweepPolicy(target, args.sweep_hold) if args.sweep else FixedAnglePolicy(target)
    if args.auto_pedals and not getattr(policy, 'driving', False):
        parser.error('--auto-pedals requires a v2 driving model; selected artifact is steering-only')
    reserved = {value for value in (args.takeover_button, args.arm_button, args.route_button) if value is not None}
    if reserved.intersection(mapping):
        parser.error("takeover/arm/route buttons are reserved; don't also map them to game actions")
    recorder = None
    if args.record_session is not None:
        from forza_ai.recording import SessionRecorder
        recorder = SessionRecorder(args.record_session, metadata=dict(capture_metadata, policy=policy.name,
                                                                     initial_manual_expert=args.record_manual,
                                                                     capture_target_hz=args.capture_hz,
                                                                     wheel_poll_target_hz=args.control_hz,
                                                                     policy_target_hz=args.policy_hz,
                                                                     takeover_settle_ms=args.takeover_settle_ms))
    if args.backend == "windows":
        if args.takeover_button is None:
            parser.error("Windows runs require --takeover-button with your verified SDL button index")
        from forza_ai.hardware import WindowsAdapter
        adapter = WindowsAdapter(torque_limit=args.torque_limit, button_map=mapping,
                                 use_motor=not args.direct_vjoy)
    else:
        adapter = SimulatedAdapter(torque_limit=args.torque_limit)
    receiver = None
    try:
        if args.telemetry or needs_camera:
            from forza_ai.telemetry import TelemetryReceiver
            receiver = TelemetryReceiver(port=args.telemetry_port)
    except BaseException:
        adapter.close()
        raise
    print(f"Policy: {policy.name}")
    if args.direct_vjoy:
        print(f"FALLBACK direct-vJoy: AI steering goes to vJoy, the wheel motor is not used. "
              f"Turn the wheel past {args.override_deg:g} deg or press the takeover button to take over.")
    try:
        result = run(adapter, policy, duration=duration, control_hz=args.control_hz,
                     policy_hz=args.policy_hz, assist=args.assist,
                     takeover_button=args.takeover_button, interactive=args.interactive,
                     receiver=receiver, config=config, status_path=args.status_csv,
                     camera=camera, shadow=args.shadow, progress=not args.quiet, foreground_guard=foreground_guard,
                     arm_button=args.arm_button, route_button=args.route_button,
                     recorder=recorder, record_manual=args.record_manual,
                     takeover_settle_ms=args.takeover_settle_ms,
                     dashboard_port=args.dashboard_port, run_report=args.run_report,
                     direct_vjoy=args.direct_vjoy, direct_override_deg=args.override_deg,
                     auto_pedals=args.auto_pedals, pedal_override=args.pedal_override,
                     dashboard_host=args.dashboard_host)
    except KeyboardInterrupt:
        print("Stopped; hardware cleanup requested.")
        return 0
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
