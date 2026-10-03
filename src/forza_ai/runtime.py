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

from forza_ai.contracts import ControlMode, ModelObservation, ObservationUnavailable, SteeringCommand
from forza_ai.control import SteeringConfig, SteeringController
from forza_ai.policies.placeholder import FixedAnglePolicy, SweepPolicy, TestObservation
from forza_ai.simulation import SimulatedAdapter


class PolicyWorker:
    """Latest-observation/latest-command exchange, no queue of old predictions.

    CPU model inference can replace predict(), but must provide the original
    image timestamp in its observation. Generating a command never refreshes
    the observation timestamp. Invalidated input also discards any in-flight
    prediction, so a slow model cannot republish a pre-takeover/pause command.
    """

    def __init__(self, policy, hz=30.0, command_ttl_ns=150_000_000):
        if not math.isfinite(hz) or hz <= 0 or command_ttl_ns <= 0:
            raise ValueError("policy frequency and TTL must be positive")
        self.policy = policy
        self.period = 1.0 / hz
        self.command_ttl_ns = command_ttl_ns
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._observation = None
        self._command = None
        self._error = None
        self._generation = 0
        self._unavailable_reason = None
        self._thread = threading.Thread(target=self._run, name="steering-policy", daemon=True)

    def start(self):
        self._thread.start()

    def publish(self, observation):
        with self._lock:
            self._observation = observation

    def invalidate(self, reason="input_unavailable"):
        with self._lock:
            self._generation += 1
            self._observation = None
            self._command = None
            self._unavailable_reason = reason

    @property
    def unavailable_reason(self):
        with self._lock:
            return self._unavailable_reason

    def latest(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError("policy failed") from self._error
            return self._command

    def _run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            with self._lock:
                observation = self._observation
                generation = self._generation
            if observation is not None:
                try:
                    target = float(self.policy.predict(observation))
                    if not math.isfinite(target):
                        raise ValueError("policy returned a non-finite target")
                    generated = time.monotonic_ns()
                    command = SteeringCommand(target, generated, observation.timestamp_ns,
                                               generated + self.command_ttl_ns)
                    with self._lock:
                        if generation == self._generation:
                            self._command = command
                            self._unavailable_reason = None
                except ObservationUnavailable as error:
                    with self._lock:
                        if generation == self._generation:
                            self._command = None
                            self._unavailable_reason = str(error)
                except Exception as error:
                    with self._lock:
                        self._error = error
                    return
            self._stop.wait(max(0.0, self.period - (time.monotonic() - started)))

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=0.5)
        # A hung model cannot block hardware cleanup; the daemon has no motor access.


def _read_console(events, stop):
    while not stop.is_set():
        line = sys.stdin.readline()
        if not line:
            return
        events.put(line.strip().lower())


def run(adapter, policy, *, duration=5.0, control_hz=100.0, policy_hz=30.0,
        assist=False, takeover_button=None, interactive=False, receiver=None,
        config=None, status_path=None, command_queue=None, camera=None, shadow=False,
        progress=False, arm_timeout=5.0):
    """Own the adapter lifecycle, including cleanup on I/O or policy exceptions.

    Zero duration runs until interrupted. Status output is optional and buffered
    in memory during this finite test; unlimited runs cannot collect an unbounded
    status log. The dataset recorder will use a separate asynchronous writer.
    """
    worker = None
    stop_console = threading.Event()
    rows = []
    summary = {"ticks": 0, "max_abs_torque": 0.0, "mode": "manual", "reason": "not_started"}
    error_text = None
    progress_worker = None
    try:
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative")
        if not math.isfinite(control_hz) or not 10 <= control_hz <= 1000:
            raise ValueError("control_hz must be between 10 and 1000")
        if status_path is not None and (duration == 0 or duration * control_hz > 100_000):
            raise ValueError("status logging is limited to 100,000 ticks; use a finite test")
        if takeover_button is not None and takeover_button < 0:
            raise ValueError("takeover button must be zero-based and nonnegative")
        button_count = getattr(adapter, "button_count", None)
        if takeover_button is not None and button_count is not None and takeover_button >= button_count:
            raise ValueError(f"takeover button {takeover_button} is absent; device has {button_count} buttons")
        if not math.isfinite(arm_timeout) or arm_timeout <= 0:
            raise ValueError("arm_timeout must be positive")
        if shadow and assist:
            raise ValueError("shadow mode cannot engage assistance")
        model_mode = bool(getattr(policy, "requires_camera", False))
        if model_mode and (camera is None or receiver is None):
            raise ValueError("model policy requires a camera and telemetry receiver")
        controller = SteeringController(config)
        events = command_queue if command_queue is not None else queue.Queue()
        if interactive:
            threading.Thread(target=_read_console, args=(events, stop_console), daemon=True).start()
            print("Commands: arm + Enter, manual + Enter, quit + Enter.")
        if receiver is not None:
            receiver.start()
        if camera is not None:
            camera.start()
        if progress:
            from forza_ai.reporting import ProgressReporter
            progress_worker = ProgressReporter()
            progress_worker.start()
        worker = PolicyWorker(policy, hz=policy_hz)
        worker.start()
        period = 1.0 / control_hz
        started = time.monotonic()
        deadline = started
        pending_arm = assist
        active_arm_until = started + arm_timeout if assist else 0.0
        while duration == 0 or time.monotonic() - started < duration:
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
            input_error = None
            if model_mode:
                frame = camera.latest()
                causal_vehicle = receiver.at_or_before(frame.timestamp_ns) if frame is not None else None
                if vehicle is None:
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
                        controller.disengage(input_error)
                else:
                    worker.publish(ModelObservation(frame, causal_vehicle))
            else:
                worker.publish(TestObservation(wheel, vehicle))
            command = worker.latest()
            takeover = takeover_button is not None and takeover_button in wheel.buttons
            should_quit = False
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    break
                if event in ("quit", "q"):
                    should_quit = True
                elif event in ("manual", "takeover"):
                    takeover = True
                elif event == "arm":
                    pending_arm = not shadow
                    active_arm_until = time.monotonic() + arm_timeout
            if should_quit:
                break
            if takeover or time.monotonic() > active_arm_until:
                pending_arm = False
            if takeover:
                worker.invalidate("manual_takeover")
                command = None
            engage_now = pending_arm and command is not None
            if engage_now:
                pending_arm = False
            now_ns = time.monotonic_ns()
            status = controller.step(wheel, command, now_ns, engage=engage_now, takeover=takeover)
            # Always forward the measured inputs, never the model's target angle.
            adapter.write_virtual_state(wheel)
            after_io = time.monotonic_ns()
            if after_io - now_ns > controller.config.max_wheel_age_ns:
                raise RuntimeError("hardware_output_stalled")
            # Revalidate ALL ages after I/O, without integrating the controller
            # twice. A sample can age out even when this write was individually fast.
            wheel_error = controller.wheel_error(wheel, after_io)
            command_error = controller.command_error(command, after_io)
            output_error = wheel_error or (command_error if status.mode == ControlMode.ASSIST else None)
            if model_mode and status.mode == ControlMode.ASSIST:
                current_vehicle = receiver.latest()
                if current_vehicle is None or not current_vehicle.is_race_on:
                    output_error = output_error or "telemetry_unavailable_or_paused"
            if output_error:
                controller.disengage(output_error, fault=wheel_error is not None)
                status = replace(status, mode=controller.mode, reason=output_error, torque=0.0)
            adapter.set_torque(status.torque)
            summary.update(ticks=summary["ticks"] + 1,
                           max_abs_torque=max(summary["max_abs_torque"], abs(status.torque)),
                           mode=status.mode.value, reason=status.reason,
                           actual_angle_deg=wheel.angle_deg, target_angle_deg=status.target_angle_deg,
                           speed_mps=vehicle.speed_mps if vehicle else None,
                           predicted_angle_deg=command.target_angle_deg if command else None,
                           observation_age_ms=(after_io - command.observation_time_ns) / 1e6 if command else None,
                           input_status=input_error or worker.unavailable_reason or "ready")
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
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Release motor before joining threads or writing files. Preserve cleanup
        # order even if a driver throws; adapter.close() retries its own releases.
        stop_console.set()
        try:
            adapter.set_torque(0.0)
        finally:
            try:
                adapter.close()
            finally:
                try:
                    if worker is not None:
                        worker.close()
                finally:
                    try:
                        if camera is not None:
                            camera.close()
                    finally:
                        try:
                            if receiver is not None:
                                receiver.close()
                        finally:
                            try:
                                if progress_worker is not None:
                                    progress_worker.close()
                            finally:
                                if status_path is not None:
                                    _save_status(status_path, rows, summary, error_text)
    return summary


def _save_status(status_path, rows, summary, error_text):
    """Preserve diagnostic evidence even when a run fails; hardware is closed."""
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(summary, error=error_text, hardware_verified=False)
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
    parser.add_argument("--button-map", action="append", default=[], metavar="SDL:VJOY",
                        help="explicit zero-based physical to one-based virtual button map; repeatable")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--telemetry", action="store_true", help="listen to FH4 Data Out on loopback")
    parser.add_argument("--telemetry-port", type=int, default=9999)
    parser.add_argument("--crop", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="absolute road crop matching the training recorder; required with --model")
    parser.add_argument("--display", type=int, default=0, help="DXcam output index")
    parser.add_argument("--capture-hz", type=float, default=30.0)
    parser.add_argument("--status-csv", type=Path)
    parser.add_argument("--quiet", action="store_true", help="suppress twice-per-second progress")
    args = parser.parse_args(argv)
    duration = args.duration if args.duration is not None else (0 if args.model else 12 if args.sweep else 5)
    if args.status_csv is not None and (duration == 0 or duration * args.control_hz > 100_000):
        parser.error("--status-csv requires a finite --duration of at most 100,000 control ticks")
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
    if args.model:
        if args.crop is None or args.backend != "windows":
            parser.error("live model requires --backend windows and --crop LEFT TOP RIGHT BOTTOM")
        if args.target_angle is not None:
            parser.error("--target-angle cannot be combined with a driving model")
        from forza_ai.capture import DXCamCapture
        from forza_ai.policies.live import LiveModelPolicy
        policy = LiveModelPolicy(args.model)
        camera = DXCamCapture(region=tuple(args.crop), fps=args.capture_hz, output_idx=args.display)
    else:
        target = 5.0 if args.target_angle is None else args.target_angle
        if args.backend == "windows" and abs(target) > 15:
            parser.error("stationary Windows placeholder tests are limited to +/-15 degrees")
        policy = SweepPolicy(target, args.sweep_hold) if args.sweep else FixedAnglePolicy(target)
    if args.backend == "windows":
        if args.takeover_button is None:
            parser.error("Windows runs require --takeover-button with your verified SDL button index")
        from forza_ai.hardware import WindowsAdapter
        adapter = WindowsAdapter(torque_limit=args.torque_limit, button_map=mapping)
    else:
        adapter = SimulatedAdapter(torque_limit=args.torque_limit)
    receiver = None
    try:
        if args.telemetry or args.model:
            from forza_ai.telemetry import TelemetryReceiver
            receiver = TelemetryReceiver(port=args.telemetry_port)
    except BaseException:
        adapter.close()
        raise
    print(f"Policy: {policy.name}")
    try:
        result = run(adapter, policy, duration=duration, control_hz=args.control_hz,
                     policy_hz=args.policy_hz, assist=args.assist,
                     takeover_button=args.takeover_button, interactive=args.interactive,
                     receiver=receiver, config=config, status_path=args.status_csv,
                     camera=camera, shadow=args.shadow, progress=not args.quiet)
    except KeyboardInterrupt:
        print("Stopped; hardware cleanup requested.")
        return 0
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
