# Physical steering adapter, first increment

The runtime runs a local wheel loop and a separate policy worker. A continuously refreshed target drives a rate-limited PD controller. Forza receives the measured physical angle and the human pedals through vJoy; the target is never substituted as the game's input. Commands retain their source observation time and expire. Takeover and faults latch until explicit re-engagement.

The runtime supports a **fixed-angle or small stationary sweep policy**, plus live RGB capture and speed-conditioned inference. The primary deployment runs inference on the desktop over LAN while keeping motor control on the game/wheel PC; see [TWO_PC_SETUP.md](TWO_PC_SETUP.md). Game road-force playback is not implemented; the motor receives only our bounded steering controller torque, separate from any driver-native centering.

## Software verification without hardware

From the repository root, with Python 3.10 or later:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=src python3 -m forza_ai.runtime --backend sim --assist --duration 5
```

The simulation is a simple synthetic wheel plant, not a validated model of TMX mechanics. A passing run validates software wiring, not physical stability. `--status-csv /tmp/control.csv` saves a finite test's controller log; it is not the training recorder. Logging is limited to 100,000 ticks and writes after hardware cleanup.

## Windows acceptance

Keep `utils/test.py` as the known hardware diagnostic. Use the existing Windows environment and its pinned `requirements.txt`; training dependencies are separate. Confirm HidHide lets the intended Python process see the TMX while Forza sees only vJoy. The SDL button indices printed by `python utils/test.py wheel` are zero-based.

Set the source path in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m forza_ai.runtime --help
```

For the following commands, replace `BUTTON_INDEX` with the verified takeover button number. The placeholder is deliberately not an executable button value.

1. **Manual passthrough:** run `python -m forza_ai.runtime --backend windows --takeover-button BUTTON_INDEX --duration 30`. Verify steering, gas, and brake reach Forza correctly. Motor road feel is absent in this first increment. No wheel buttons are mapped to vJoy by default; explicit mappings can be supplied through the Python adapter API once verified.
2. **Stationary angle tracking:** close Forza for the first motor test. Run `python -m forza_ai.runtime --backend windows --takeover-button BUTTON_INDEX --assist --target-angle 5 --duration 5`. Confirm a small rightward movement; the tested TMX has a negative SDL force level for physical right. Repeat with -5. The initial gains and 15% torque cap require physical tuning; software tests do not establish comfortable force or stability.
3. **Takeover:** while running a longer stationary test, press the selected button. AI torque should stop and remain off after releasing the button. Physical inputs continue to vJoy.
4. **Explicit re-engagement:** add `--interactive`; enter `arm`, `manual`, or `quit` in the terminal. `arm` has a five-second window for the first valid command, and pressing takeover cancels it. Restarting the runtime with `--assist` is also an explicit engagement.
5. **Telemetry:** add `--telemetry` after configuring FH4 Data Out to `127.0.0.1:9999`. Final output includes the latest fresh speed in m/s. Verify speed and steering against the game. The receiver supports exact 324-byte FH4 packets only.
6. **Exit and error paths:** verify torque release on Ctrl+C, normal duration expiry, takeover, and wheel loss. Finite-duration force effects provide a bounded software fallback on a process stall; actual driver/device behavior must still be verified on Windows.

Windows placeholder targets are restricted to +/-15 degrees at the CLI. The controller has a configurable target limit and slew rate. This does not qualify the controller for autonomous driving or generalize its gains to another wheel/car.

## Interfaces

- `WindowsAdapter.read_state(now_ns)` returns a `WheelState` in physical units.
- Its timestamp is host poll time. SDL exposes cached axes without a USB report timestamp, so age checks detect host delays but cannot independently establish the age of an unchanged device report.
- `write_virtual_state(state)` forwards measured steering/pedals and explicitly mapped buttons.
- `set_torque(value)` consumes a normalized physical-right-positive value; the hardware layer handles SDL sign conversion and clamps it again.
- `close()` releases force effects, wheel handles, and vJoy ownership.
- `autocenter_disabled_confirmed` reports whether SDL confirmed disabling native centering. False is expected on some TMX drivers and does not prove centering is active or inactive. Opening SDL itself can reset or enable native effects; validate actual behavior on the wheel PC. Zero torque stops our effect, not a guarantee of mechanically force-free hardware.
- `SteeringController.step(...)` is pure Python and owns engagement, PD, target slew limits, and time validity.
- `PolicyWorker` keeps only the newest observation and command. Inference cannot block the motor thread; no queue of old steering commands accumulates.
- `TelemetryReceiver.latest(...)` returns an age-bounded sample without refreshing its timestamp. `at_or_before(frame_time)` supplies causal speed features. Live runtime rejects paused, missing, stale, or non-foreground game inputs and invalidates in-flight predictions.

See [CONTRACTS.md](CONTRACTS.md) for units, recording format, and the model export boundary.
