# Steering, acceleration and braking in Forza

Use `--task driving` to train a new model and `--auto-pedals --assist` on the game
client to drive with it. `--shadow` is optional: it computes predictions without
applying AI steering or pedal commands. This feature is for Forza Horizon 4 only.

## Shortest path to a driving run

Both PCs need the updated source. On the training/inference PC, use completed
human-driven recordings containing turns, acceleration, coasting and braking.
Import each independent recording with the existing manual-data importer; the
same gas/brake columns are now used as labels. Keep at least one independent
recording for validation; splitting the same clip into folders is not a held-out
recording. Choose new output directories in these example commands:

```powershell
python -m forza_ai.training.cli train C:\data\forza C:\runs\driving-v2 --task driving --device cuda --epochs 10
python -m forza_ai.training.cli evaluate C:\runs\driving-v2\best.pt C:\data\forza
python -m forza_ai.training.cli export C:\runs\driving-v2\best.pt C:\models\driving-v2
```

Ten epochs is a starting example, not a validated training duration. The only
real clip verified by this task was one short recording group; it cannot supply
independent training and validation or demonstrate reliable braking behavior.
No production driving weights were created by this change. The existing
`diagnostic_export_DO_NOT_DEPLOY` is steering-only and cannot supply pedals.

When the operator is ready to use the new export, start the server with the same
LAN address and shared `FORZA_LINK_KEY` used for steering inference:

```powershell
python -m forza_ai.inference_server --bind <INFERENCE_PC_IPV4> --model C:\models\driving-v2
```

On the game PC, keep the verified capture configuration, Data Out port and
button indices. Add `--auto-pedals --assist` to the existing runtime command.
A complete pattern (replace angle-bracket placeholders) is:

```powershell
python -m forza_ai.runtime --backend windows --inference-host <INFERENCE_PC_IPV4> --capture-config <CAPTURE_JSON> --telemetry-port 9999 --takeover-button <SDL_INDEX> --arm-button <OTHER_SDL_INDEX> --auto-pedals --assist --interactive --duration 0
```

This retains physical TMX steering. Add `--direct-vjoy` to use virtual steering
instead, with no motor acquisition. It preserves the RawInput startup fix from
`9948b02`: untouched physical pedals initialize released. Forza must use the
vJoy steering and pedal bindings. The adapter maps brake to **Y** and throttle
to **Z**, with **32768 released, 1 fully pressed**.

Saved profiles also support `control.auto_pedals: true`,
`control.direct_vjoy: true` (optional), and `control.pedal_override: 0.05`.
Use the launcher's existing `--assist` option or profile `mode: "assist"`.
Desktop profiles automatically select the export's protocol version.

## Driver controls

Pressing either physical pedal at least 5% takes over steering and pedals.
The takeover button or console `manual` command does the same. Release the
pedals, then use the arm button, console `arm`, or dashboard Engage driving to
resume. Holding a pedal prevents engagement; releasing it alone does not rearm.
`--pedal-override` adjusts the threshold. Ctrl+C stops the client and releases
virtual pedals. Pause, stale input and network failure release AI pedals and
require a new arm action. Manual pedal control remains available after takeover.

The dashboard/run report includes output gas/brake and predictions. With
`--shadow`, physical pedals remain manual even while predictions are available.
A local v2 model can also run steering with manual pedals by omitting
`--auto-pedals`; a remote v2 server requires a v2 (`--auto-pedals`) client.
A version mismatch fails explicitly instead of interpreting steering as pedals.

## Model and data contract

The original v1 steering architecture/checkpoints continue to work. New v2
checkpoints use `pilotnet_driving_v2`: unchanged RGB/speed preprocessing and
encoder, with a two-value tanh head for normalized angle and signed longitudinal
action. Positive action becomes throttle, negative becomes brake, and zero is
coast. The public result has `angle_deg`, `throttle`, `brake`; gas and brake cannot
both be positive. Loss equally weights normalized steering and signed pedal
action. Pedal fractions are 0 released to 1 pressed. If a demonstrator overlaps
pedals, the target gives braking priority. Original source labels stay intact.

Pedals use the same label-time interpolation and expert-mode eligibility as
steering. Integrated recording saves physical inputs, excludes AI-assist and
fault samples, and admits explicitly marked human correction samples only
after the existing handover interval. Never import mixed AI/human legacy
`record.py` output with `--expert-mode manual`: that format cannot identify AI
control. For driving training, the declaration applies to steering AND pedals.

Validation reports steering degrees and throttle/brake MAE/RMSE in fractions,
plus normalized aggregate RMSE for selecting the best v2 checkpoint. Resume
restores the task, optimizer and exact split; v1 weights cannot resume as v2.
The preprocessing cache includes task and pedal labels in its identity and
accounts for the larger target tensor in its existing fit-only memory budget.

## Verification scope

Offline tests cover label import/alignment, pedal-label cache isolation,
continuous-versus-resumed and cached-versus-uncached training equality,
checkpoint/export prediction parity, gradients, authenticated v2 transport,
v1 mismatch, fake Windows axis inversion/release ordering, simulator takeover,
rearming, optional shadow and release after pause/disconnection/deadline expiry.
They establish a functioning software path, not lap completion or driving
quality. This task did not restart the existing server, load a new live model,
move the TMX or send live vJoy commands.
