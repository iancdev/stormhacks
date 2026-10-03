import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from forza_ai import launch


def plan(tmp_path):
    return launch.LaunchPlan("desktop", (sys.executable, "-c", "pass"), tmp_path / "run", "test")


def test_interrupt_waits_for_cleanup_and_records_outcome(tmp_path, monkeypatch):
    class Child:
        args = ["fake"]
        returncode = None
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            assert timeout is not None and timeout > 0
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            pytest.fail("graceful child must not be terminated")

        def kill(self):
            pytest.fail("graceful child must not be killed")

    monkeypatch.setenv("FORZA_LINK_KEY", "test-only")
    child = Child()
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *args, **kwargs: child)
    target = plan(tmp_path)
    assert launch.launch(target) == 0
    saved = json.loads((target.run_dir / "exit.json").read_text())
    assert saved == {"returncode": 0, "interrupted": True, "forced_termination": False, "error_type": None}


def test_nonterminating_child_is_escalated_after_grace(tmp_path, monkeypatch):
    class Child:
        args = ["fake"]
        returncode = None
        calls = 0
        terminated = killed = False

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            raise subprocess.TimeoutExpired(self.args, timeout)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            assert self.terminated
            self.killed = True
            self.returncode = -9

    child = Child()
    monkeypatch.setenv("FORZA_LINK_KEY", "test-only")
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *args, **kwargs: child)
    target = plan(tmp_path)
    assert launch.launch(target, shutdown_grace_s=.05) == -9
    assert child.killed
    assert json.loads((target.run_dir / "exit.json").read_text())["forced_termination"]


def test_repeated_interrupt_during_poll_still_allows_grace(tmp_path, monkeypatch):
    class Child:
        args = ["fake"]
        returncode = None
        waits = polls = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            self.returncode = 0
            return 0

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                raise KeyboardInterrupt
            return self.returncode

        def kill(self):
            pytest.fail("second parent interrupt must not kill a draining child")

    monkeypatch.setenv("FORZA_LINK_KEY", "test-only")
    child = Child()
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *args, **kwargs: child)
    target = plan(tmp_path)
    assert launch.launch(target) == 0
    assert child.waits == 2
    assert not json.loads((target.run_dir / "exit.json").read_text())["forced_termination"]


def test_interrupt_while_printing_stop_notice_does_not_kill_child(tmp_path, monkeypatch):
    class Child:
        args = ["fake"]
        returncode = None
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            pytest.fail("printing an interrupt notice must not kill the child")

    monkeypatch.setenv("FORZA_LINK_KEY", "test-only")
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert launch.launch(plan(tmp_path)) == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group signal repro; Windows console requires on-device acceptance")
def test_real_console_interrupt_preserves_slow_child_cleanup(tmp_path):
    ready, completed = tmp_path / "ready", tmp_path / "completed"
    child = (
        "from pathlib import Path\nimport time\n"
        f"Path({str(ready)!r}).write_text('ready')\n"
        "try:\n while True: time.sleep(.02)\n"
        "except KeyboardInterrupt:\n time.sleep(.7)\n"
        f" Path({str(completed)!r}).write_text('complete')\n"
    )
    run_directory = tmp_path / "launched"
    runner = ("from pathlib import Path\nfrom forza_ai.launch import LaunchPlan, launch\n"
              f"raise SystemExit(launch(LaunchPlan('desktop', ({sys.executable!r}, '-c', {child!r}), "
              f"Path({str(run_directory)!r}), 'test')))\n")
    environment = dict(os.environ, FORZA_LINK_KEY="test-only")
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    process = subprocess.Popen([sys.executable, "-c", runner], env=environment, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert completed.exists()
        outcome = json.loads((run_directory / "exit.json").read_text())
        assert outcome["interrupted"] and not outcome["forced_termination"]
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
