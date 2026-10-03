# Dataset Survey Report — Forza Horizon 4 steering behavior cloning

Synthesis of FINDINGS.md (tasks 1-8), dated 2026-10-03.
Target format: per-frame JPEG + CSV, bonnet cam, wheel angle in degrees
(-450..+450), speed in m/s, 30 fps, Forza Data Out telemetry.

## Exact match

**No exact match exists.** No public dataset pairs FH4/FH5/FM frames with
wheel-angle (or even continuous steering-axis) labels plus speed in a
directly usable, downloadable form. The closest candidates, with their
fatal flaws:

- **yuyuggg/ForzaHorizon6 (ModelScope)** — the only real Forza
  frame+steer+speed dataset found: 842,592 frames, ~15.1 GB, forward cam
  320x180, gamepad-axis steering conditioned on ego speed via
  `steer_map.json`. But: FH6 not FH4, gated download (ApprovalMode=1),
  license conflict (API says Apache-2.0, repo says CC BY-NC 4.0), ~70%
  AI-driver frames, labels are normalized axis not degrees.
- **opensima36/forza_horizon_5_bc_01 (HF)** — 63.7 GB of FH5
  behavior-cloning parquet, almost certainly frame+input data, but the
  7z archives are encrypted with no published key. Unusable as released.
- **montmejat/HorizonNet (FH4)** — the only directly downloadable FH4
  frame+steer data: 534 chase-cam PNGs + `inputs.csv` with Xbox joystick
  x-axis (~-32768..32767) and triggers. ~1 Hz sampling, no speed, no
  license, abandoned. Too small/sparse to pretrain on.
- **The Matrix Dataset (FH5, arXiv:2412.03568)** — 2,861.9 h of FH5 video
  at 1920x1080, 262,707 clips with control signals, Apache-2.0, actually
  downloadable (~12 TB). But labels are scripted 1 Hz discrete
  D/drive-left/drive-right commands — not human, not continuous, not
  degrees. Visual pretraining only.
- opensima05/forza_horizon_5_recordings_01 (17 raw FH5 sessions) and
  xiaoluo11/forza-horizon-5-gameplay-data (video + per-frame key/mouse
  event timeline) are real synced FH5 data but gated or labeled with
  keyboard/mouse events, not steering.

## Partial match

**Forza telemetry logs (no images):** exactly one published decoded
telemetry dataset exists anywhere — `alexhexan/fm7-rio-de-janeiro-race-telemetry`
on Kaggle (CC0, ~62 MB, 5 laps of Rio in FM7 by the author of the Rust
0x20F/forza-telemetry decoder). Caveat: FM7 uses the 311-byte dash
layout, not the FH4 324-byte Horizon layout. No public FH4/FH5
CSV/parquet capture was found; every recorder ships user-local files only.

**FH4 packet-layout documentation:** primary sources are gone — Forza
forums retired July 2026 (thread 74308 410-gone, survives in cache
excerpts) and the official support article 403'd (and documents FM2023
anyway). Best surviving artifact: `richstokes/Forza-data-tools/
FH4_packetformat.dat` (GPL-3.0, field-by-field 324-byte spec matching our
layout exactly).

**Other-game frame+steering sets (continuous labels only listed):**

| Dataset | Cam | Labels | Speed | Status |
|---|---|---|---|---|
| DeepDrive GTAV baseline 2016 (archive.org, 600k frames/42 h/52 GB) | forward | steer+throttle+yaw+speed, AI-driver, units unverified | yes | downloadable, no license |
| Alzaib GTA-5 (Google Drive, ~100k frames) | hood, 160x120 gray | joystick steer+throttle -1..1 | no | live, verified listing |
| europilot ETS2 (162k frames + wheel-axis CSV) | cabin?, 562x341 sample | G27 wheel axis [-32767,32767] | no | LINK DEAD (404) |
| CARLA CIL/CoRL-2017 (24 GB, ~14 h HUMAN) | front 200x88 | steer/gas/brake normalized -1..1 | yes | Drive link live |
| roboticslaburjc CARLA_e2e (~163 GB, Apache-2.0) | unverified | `image_name,throttle,steer` csv, normalized; includes human wheel captures | no (verified csv) | ungated HF |
| Bench2Drive (2M frames, Apache-2.0), PDM-Lite (581k), TCP (~123 GB), mmahdavi/carla_1.8m (~247 GB MIT) | multi-cam | normalized, autopilot/RL experts | yes | ungated HF |
| Princeton DeepDriving TORCS (484k human frames) | forward | 14 affordance indicators, NOT steering | n/a | LINKS DEAD (403) |

Label-unit warning: every CARLA/GTA/game set uses normalized -1..1 axis
or raw joystick values — same signal class as our `tele_steer` s8, not
wheel degrees. Keyboard-labeled sets (sartajbhuvaji ~1M frames,
kfk42kfk 70k/140k, EthanNCai, most sentdex clones) cannot supervise
steering regression at all.

**Human-driving telemetry corpora (no images):**
`dasgringuen/assettoCorsaGym` (HF, CC-BY-4.0, 64M steps, 2.3M human,
50 Hz, steerAngle+speed+pedals, MoTeC .ld) is the largest verified-
downloadable human driving corpus in the survey. TRAVEL/BeamNG Zenodo
sets are agent-generated.

## Adjacent / fallback

**Code-only Forza projects (no data shipped):** ricky5932TW/
End2End-autodrive-image-steer, zohairajmal/forza4selfdrive,
CallMeRZIBI/Forza-autopilot, shoal-rat/Horizon_FSD (recordings
deliberately excluded), castortroy05/ForzaAIMasters, EthanNCai/AI-Plays-
ForzaHorizon4 (dataset-0.npy behind a possibly-dead OneDrive link,
keyboard labels anyway). Academic: RealPlay (unreleased), Patadiya
CICN-2024 (unreleased), Pratama YOLOv8-FH4 (wrong task). codelion/
multi-drive-model-data (HF, 340k "realistic" frames) mixes Forza+NFS
with binary/flow-derived labels — no steering angle.

**Real-world fallbacks (all verified downloadable):**
- **comma2k19** (MIT, 94.6 GB, ~33.65 h, 2.4M frames @ 20 Hz,
  1164x874 dashcam): CAN `steering_angle` in DEGREES + `car_speed` m/s,
  timestamped npy channels. The only fallback carrying both a degrees
  wheel-angle label and speed — a superset of our CSV schema.
  Domain: pure CA-280 highway.
- **SullyChen/driving-datasets** (MIT, ~2.2+3.1 GB Google Drive,
  45,406 + 63,825 frames @ 455x256 suburban): wheel angle in DEGREES
  (verified ranges ±160..500 deg), no speed. His pipeline resizes to
  exactly 200x66 — identical input geometry to ours. Closest visual
  analog for lane-keeping.
- **Udacity CH2** (MIT, rosbag torrents): `steering_wheel_angle` in
  RADIANS + speed m/s, needs rosbag extraction + rad→deg relabel.
  Most famous, most work.
- **Udacity sim sample** (333 MB cloudfront, live): sim-internal
  steering float + mph — toy domain, smoke-test only.

## Forza Data Out parsers and offset agreement

Our offsets (IsRaceOn i32 @0, Speed f32 @256 m/s, Steer s8 @320, 324-byte
dash packet) are confirmed by **12 independent implementations, zero
disagreements**:

- richstokes/FH4_packetformat.dat — doc; agrees exactly (also documents
  older 323-byte packets).
- nettrom/forza_motorsport `fdp.py` (MIT, oldest confirmed Python) —
  agrees via `data[:232]+data[244:323]` patch; discards Horizon
  extension silently.
- makvoid/Blog-Articles `data_packet.py` — agrees; struct
  `'<iI27f4i20f5i'+'i19fH6B4b'` = 324 bytes.
- nikidziuba/Forza_horizon_data_out_python — agrees positionally;
  format doc covers only 323 bytes (drops trailing byte).
- jasperan/forza-horizon-5-telemetry-listener — agrees (nettrom
  derivative); richest dashboard/analytics tooling.
- Grvs44/Forza-Telemetry-Export (Clear BSD, pip) — agrees; documents
  all four packet sizes (SLED 232, DASH 311, DASHH 324, DASHM 331).
- bobbythehuman/RaceTelemetry (PyPI, LGPL-2.1) — agrees; only
  pip-installable package verified on the 324-byte layout.
- ricky5932TW `telemetry.py` — agrees byte-for-byte (Speed _f32 @256,
  Steer _s8 @320, PACKET_SIZE=324).
- shoal-rat `forza_telemetry.py` — agrees; asserted 324-byte spec
  "CONFIRMED against live FH6"; tolerates 323-byte packets. Strongest
  independent live confirmation.
- Ayin1412 `drive.py` — agrees (Speed f @256, Steer b @320, 324 total).
- 0x20F/forza-telemetry (Rust) — agrees (dash_base=244; Speed @256,
  Steer @320). Cross-language corroboration.
- austinbaccus/forza-telemetry (C#) — agrees on offsets; sloppy decode
  types (IsRaceOn read as float, ordinals as u8).

**One disagreement:** Estetika101/pacefinderapp mislabels the 331-byte
FM2023 packet as "FH4/FH5 Car Dash" and rejects real 324-byte Horizon
packets. Do not use as a reference.

Open items: the 12-byte Horizon extension @232-243 has two competing
naming conventions (CarCategory+Unknown1+Unknown2 vs
CarGroup+SmashableVelDiff+SmashableMass) — both community guesses, does
not affect our offsets. Also: `tele_steer` (s8, -127..127) is the
post-mapping game axis, NOT the wheel angle — expect divergence from
`steer_deg` under speed-sensitive steering limits.

## Verdict

(a) Pretraining: no usable Forza frame+degrees+speed dataset exists; the
gated FH6 set is the nearest and is neither FH4 nor degrees-labeled.
Realistic path: pretrain on comma2k19 (degrees + m/s, real road) or
Sully Chen (degrees, no speed), or skip pretraining and collect our own
laps — 30 fps captures reach Sully-Chen scale in under an hour.

(b) Telemetry parser validation: solved — our offsets are corroborated by
12 independent parsers and the richstokes spec; unit-test against a
synthetic 324-byte packet via RaceTelemetry or Grvs44 as oracle. No
public FH4 capture exists to validate against real data; record our own.

(c) Tiger Data analytics demo: seedable — the CC0 Kaggle FM7 Rio
telemetry CSV is the only published decoded Forza log; AssettoCorsaGym
(CC-BY-4.0, 2.3M human steps) is the best larger human-telemetry
corpus; or generate FH4 laps ourselves with jasperan/Grvs44/RaceTelemetry.
