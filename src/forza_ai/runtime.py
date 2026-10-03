"""Run the physical steering adapter with a clearly marked test policy.

Example without Windows drivers:
    PYTHONPATH=src python -m forza_ai.runtime --backend sim --assist --duration 5

This runtime has no camera or driving model yet. A fixed target is a stationary
control test, never an autonomous driving policy.
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

from forza_ai.contracts import ControlMode, SteeringCommand
from forza_ai.control import SteeringConfig, SteeringController
from forza_ai.policies.placeholder import FixedAnglePolicy, TestObservation
from forza_ai.simulation import SimulatedAdapter


class PolicyWorker:
    """Latest-observation/latest-command exchange, no queue of old predictions.

    CPU model inference can replace predict(), but must provide the original
    image timestamp in its observation. Generating a command never refreshes
    the observation timestamp. This first runtime publishes wheel observations
    solely for the stationary test policy.
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
        self._thread = threading.Thread(target=self._run, name="steering-policy", daemon=True)

    def start(self):
        self._thread.start()

    def publish(self, observation):
        with self._lock:
            self._observation = observation

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
            if observation is not None:
                try:
                    target = float(self.policy.predict(observation))
                    if not math.isfinite(target):
                        raise ValueError("policy returned a non-finite target")
                    generated = time.monotonic_ns()
                    command = SteeringCommand(target, generated, observation.timestamp_ns,
                                               generated + self.command_ttl_ns)
                    with self._lock:
                        self._command = command
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
        config=None, status_path=None, command_queue=None):
    """Own the adapter lifecycle, including cleanup on I/O or policy exceptions.

    Zero duration runs until interrupted. Status output is optional and buffered
    in memory during this finite test; unlimited runs cannot collect an unbounded
    status log. The dataset recorder will use a separate asynchronous writer.
    """
    worker = None
    stop_console = threading.Event()
    rows = []
    summary = {"ticks": 0, "max_abs_torque": 0.0, "mode": "manual", "reason": "not_started"}
    try:
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative")
        if not math.isfinite(control_hz) or not 10 <= control_hz <= 1000:
            raise ValueError("control_hz must be between 10 and 1000")
        if status_path is not None and (duration == 0 or duration * control_hz > 100_000):
            raise ValueError("status logging is limited to 100,000 ticks; use a finite test")
        if takeover_button is not None and takeover_button < 0:
            raise ValueError("takeover button must be zero-based and nonnegative")
        controller = SteeringController(config)
        events = command_queue if command_queue is not None else queue.Queue()
        if interactive:
            threading.Thread(target=_read_console, args=(events, stop_console), daemon=True).start()
            print("Commands: arm + Enter, manual + Enter, quit + Enter.")
        if receiver is not None:
            receiver.start()
        worker = PolicyWorker(policy, hz=policy_hz)
        worker.start()
        period = 1.0 / control_hz
        started = time.monotonic()
        deadline = started
        pending_arm = assist
        active_arm_until = started + 1.0 if assist else 0.0
        while duration == 0 or time.monotonic() - started < duration:
            now_ns = time.monotonic_ns()
            wheel = adapter.read_state(now_ns)
            # Reacquire time after device I/O to account for a slow read.
            now_ns = time.monotonic_ns()
            wheel_error = controller.wheel_error(wheel, now_ns)
            if wheel_error:
                raise RuntimeError(wheel_error)
            vehicle = receiver.latest(now_ns=now_ns) if receiver is not None else None
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
                    pending_arm = True
                    active_arm_until = time.monotonic() + 1.0
            if should_quit:
                break
            if takeover or time.monotonic() > active_arm_until:
                pending_arm = False
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
            if output_error:
                controller.disengage(output_error, fault=wheel_error is not None)
                status = replace(status, mode=controller.mode, reason=output_error, torque=0.0)
            adapter.set_torque(status.torque)
            summary.update(ticks=summary["ticks"] + 1,
                           max_abs_torque=max(summary["max_abs_torque"], abs(status.torque)),
                           mode=status.mode.value, reason=status.reason,
                           actual_angle_deg=wheel.angle_deg, target_angle_deg=status.target_angle_deg,
                           speed_mps=vehicle.speed_mps if vehicle else None)
            if status_path is not None:
                row = asdict(status)
                row.update(mode=status.mode.value, timestamp_ns=now_ns,
                           speed_mps=vehicle.speed_mps if vehicle else "")
                rows.append(row)
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()  # no backlog of catch-up motor commands
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
                    if receiver is not None:
                        receiver.close()
    if status_path is not None and rows:
        path = Path(status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sim", "windows"), default="sim")
    parser.add_argument("--assist", action="store_true", help="explicitly engage the stationary test policy")
    parser.add_argument("--target-angle", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=5.0, help="seconds; 0 runs until stopped")
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--policy-hz", type=float, default=30.0)
    parser.add_argument("--torque-limit", type=float, default=0.15)
    parser.add_argument("--takeover-button", type=int, help="SDL zero-based index, from utils/test.py wheel")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--telemetry", action="store_true", help="listen to FH4 Data Out on loopback")
    parser.add_argument("--telemetry-port", type=int, default=9999)
    parser.add_argument("--status-csv", type=Path)
    args = parser.parse_args(argv)
    config = SteeringConfig(torque_limit=args.torque_limit)
    policy = FixedAnglePolicy(args.target_angle)
    if args.backend == "windows":
        if args.takeover_button is None:
            parser.error("Windows runs require --takeover-button with your verified SDL button index")
        if abs(args.target_angle) > 15:
            parser.error("stationary Windows placeholder tests are limited to +/-15 degrees")
        from forza_ai.hardware import WindowsAdapter
        adapter = WindowsAdapter(torque_limit=args.torque_limit)
    else:
        adapter = SimulatedAdapter(torque_limit=args.torque_limit)
    receiver = None
    try:
        if args.telemetry:
            from forza_ai.telemetry import TelemetryReceiver
            receiver = TelemetryReceiver(port=args.telemetry_port)
    except BaseException:
        adapter.close()
        raise
    print(f"Policy: {policy.name}")
    try:
        result = run(adapter, policy, duration=args.duration, control_hz=args.control_hz,
                     policy_hz=args.policy_hz, assist=args.assist,
                     takeover_button=args.takeover_button, interactive=args.interactive,
                     receiver=receiver, config=config, status_path=args.status_csv)
    except KeyboardInterrupt:
        print("Stopped; hardware cleanup requested.")
        return 0
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
