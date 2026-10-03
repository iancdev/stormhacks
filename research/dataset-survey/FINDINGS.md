# Findings (append-only)

Add one entry per hit using this exact template. Never delete or rewrite earlier entries;
if you find a correction, add a new entry with "CORRECTION to <name>:" in the title.
Only record things you actually opened and read. Mark anything unverified as "unverified".

```
## <name>  [EXACT | PARTIAL | ADJACENT | FALLBACK]  (task N)
- URL:
- Maintainer:
- Date (created / last updated):
- License:
- Size (frames / hours / GB):
- Image resolution & camera view:
- Label columns & units (degrees / normalized / gamepad axis):
- Speed included: yes/no (units)
- Per-frame synced: yes/no/unknown (how)
- Download method:
- Known problems:
- Notes (offset agreement for parsers, etc.):
```

If a task produced no hits, add a short entry:
`## No hits for task N: <what was searched, where>`

---

## yuyuggg/ForzaHorizon6 (ModelScope, from Ayin1412/ForzaHorizon6-VisionAI)  [EXACT]  (task 1)
- URL: https://modelscope.cn/datasets/yuyuggg/ForzaHorizon6 ; repo https://github.com/Ayin1412/ForzaHorizon6-VisionAI
- Maintainer: Wenhao Li (GitHub Ayin1412, ModelScope yuyuggg), independent researcher
- Date (created / last updated): repo initial public release 2026-07-26; ModelScope GmtCreate epoch 1784262602 (~2026-07-17), GmtModified ~2026-07-26 (verified via ModelScope API)
- License: CONFLICT — ModelScope API metadata says "Apache License 2.0"; repo states dataset is under CC BY-NC 4.0 like other non-code artifacts, and all game imagery is © Microsoft. Treat as non-commercial research only.
- Size (frames / hours / GB): 842,592 cleaned frames from 33 recording sessions (69.4% game-AI "Anna", 30.6% human gamepad) per README; ModelScope API StorageSize = 15,095,383,165 bytes (~15.1 GB), 72 downloads. Both verified via API + README.
- Image resolution & camera view: single forward camera, training pipeline resizes captures to 320x180 (dataset storage res unverified); layout is `index.csv` + `images/`
- Label columns & units (degrees / normalized / gamepad axis): index.csv columns not public (gated). Per README/technical report: steer + throttle in gamepad-axis units (Anna-vs-human units aligned by a speed-conditioned map rho(v) stored in `steer_map.json`), plus derived geometry labels: lane offset, 9 arc-length waypoints, per-waypoint speed profile, tyre slip. NOT wheel degrees.
- Speed included: yes — ego speed is a control-head input sourced from Forza Data Out telemetry; steering-unit alignment is explicitly speed-conditioned (units in index.csv unverified)
- Per-frame synced: yes (recorded driving sessions; train.py validates sessions against a registry). Sync method undocumented.
- Download method: ModelScope dataset page; API shows ApprovalMode=1 / ProtectedMode=1 — access granted per requester (gated). Whether access is actually granted is unverified.
- Known problems: Forza Horizon 6, not FH4 (different map, physics, HUD); steering in normalized gamepad units, not degrees; gated download; license conflict Apache-2.0 vs CC BY-NC 4.0; copyrighted game imagery; session recorder not released (train.py is reference-only); ~70% of frames are AI-driver ("Anna"), not human.
- Notes: Closest thing found to a real public Forza frame+steer+speed dataset. Repo also ships released weights (model_30.pt), Forza "Sled+Dash" UDP telemetry handling on port 5606, and `steer_calibration.json`/`steer_map.json` which document the game's steering response curve — useful even without the dataset.

## montmejat/HorizonNet (formerly oralian/HorizonNet)  [PARTIAL]  (task 1)
- URL: https://github.com/montmejat/HorizonNet (data at https://github.com/montmejat/HorizonNet/tree/master/data ; CSV at https://raw.githubusercontent.com/montmejat/HorizonNet/master/data/inputs.csv)
- Maintainer: montmejat (repo moved/renamed from oralian)
- Date (created / last updated): unverified; Python 3.7-era project, 9 commits, ~2019-2020
- License: none (no LICENSE file in repo — verified via git tree API)
- Size (frames / hours / GB): 534 PNG frames + inputs.csv (534 data rows; verified by cloning git tree and fetching CSV — 535 lines incl. header). At ~1 capture/sec this is ~9 minutes of driving.
- Image resolution & camera view: 970x702 PNG (verified on 2 frames), chase/third-person camera behind the car, HUD off; mixed day and night conditions
- Label columns & units: `imagefile,x,y,gas,brake` — x,y are raw Xbox joystick axes (~-32768..32767, observed -24844..32767); gas/brake are trigger values 0..255. Steering = x axis in gamepad units, not degrees.
- Speed included: no
- Per-frame synced: yes — capture.py saves one frame + one inputs.csv row per iteration (~1 Hz)
- Download method: git clone (data is inside the repo)
- Known problems: tiny dataset; ~1 Hz sampling (not 30 fps); chase cam, not bonnet cam; no speed channel; no telemetry; unlicensed; "neural network is in development" per README (project abandoned early)
- Notes: The only FH4 frame+steer dataset verified to be directly downloadable on GitHub, but far too small/sparse to pretrain on. Format is otherwise close to ours (image file + CSV row).

## EthanNCai/AI-Plays-ForzaHorizon4  [PARTIAL]  (task 1)
- URL: https://github.com/EthanNCai/AI-Plays-ForzaHorizon4 ; dataset link "Download dataset-0.npy" -> https://1drv.ms/u/c/257cf703ad194ae1/EeFKGa0D93wggCUwEQAAAAABAWu52QyIaEb_sNwxk4dsWw
- Maintainer: EthanNCai (chickenbilibili@outlook.com)
- Date (created / last updated): unverified; 55 commits, older project
- License: none visible
- Size (frames / hours / GB): unknown — single `dataset-0.npy` file, size not stated
- Image resolution & camera view: screen grab of configurable capture zone (`grab_screen()`), preprocessed (crop/resize/grayscale/canny); resolution unverified
- Label columns & units: samples are (screen, keyboard actions) — discrete key presses, NOT wheel angle or gamepad axis
- Speed included: no
- Per-frame synced: yes (frame + key state recorded together)
- Download method: OneDrive share link in README. Link resolves (302 -> onedrive.live.com) but an unauthenticated HEAD request returned 403 — download unverified, possibly dead or login-gated.
- Known problems: keyboard-classification labels unusable for steering-angle regression; no speed; npy blob, format needs DataPreview.py to inspect; link status unverified.
- Notes: Data exists in principle but is the weakest of the "real" hits for our use case.

## ricky5932TW/End2End-autodrive-image-steer  [ADJACENT]  (task 1)
- URL: https://github.com/ricky5932TW/End2End-autodrive-image-steer
- Maintainer: ricky5932TW
- Date (created / last updated): 15 commits, recent (2026-era); exact dates unverified
- License: none stated in repo (unverified)
- Size: NO DATA SHIPPED — `sessions/` is not in the repo (verified via file listing); only `best_model.pth` checkpoint is shipped. README documents 12 sessions, 52,697 raw rows (52,015 after filtering).
- Image resolution & camera view: images resized 320x180 then road-cropped to (0,75,320,122); camera view unverified
- Label columns & units: `dataset.csv` per session with Speed, is_valid, steering scaled to [-127,127] (XInput-style axis units); EMA-smoothed labels
- Speed included: yes in their format (Speed > 0 filtering), but data not published
- Per-frame synced: n/a (data not published)
- Download method: n/a
- Known problems: dataset not released — README says training data lives in local `sessions/`; steering-only (throttle/brake are placeholders); game listed only as "Forza" with Dash UDP — which title (FH4/FH5/FM) is unverified.
- Notes: Worth revisiting in task 4 — `forza_autodrive/telemetry.py` is a Python Forza Dash UDP parser whose offsets can be checked against ours. Their [-127,127] steering convention matches Forza's s8 Steer field, not our wheel degrees.

## zohairajmal/forza4selfdrive  [ADJACENT]  (task 1)
- URL: https://github.com/zohairajmal/forza4selfdrive
- Maintainer: zohairajmal
- Date (created / last updated): unverified; 25 commits, older project
- License: none visible
- Size: no data — repo contains code + 4 readme images only (verified via file listing); user runs main.py to collect their own
- Label columns & units: keyboard keys (sentdex/GTA-style lane + key logging); not steering angle
- Speed included: no
- Download method: n/a
- Known problems: code only; AlexNet-style classifier on key presses.
- Notes: FH4, but no published dataset.

## CallMeRZIBI/Forza-autopilot  [ADJACENT]  (task 1)
- URL: https://github.com/CallMeRZIBI/Forza-autopilot
- Maintainer: CallMeRZIBI
- Date (created / last updated): unverified; 56 commits
- License: none visible
- Size: no data — README instructs `mkdir collected_data`; user collects their own
- Label columns & units: keyboard inputs via directkeys; not steering angle
- Speed included: no
- Download method: n/a
- Known problems: code only; keyboard labels.
- Notes: FH4 TF/Keras autopilot, code-only.

## shoal-rat/Horizon_FSD  [ADJACENT]  (task 1)
- URL: https://github.com/shoal-rat/Horizon_FSD
- Maintainer: shoal-rat
- Date (created / last updated): 22 commits; recordings_legacy dirs dated 2026-06-06
- License: unverified
- Size: no data — releases page explicitly states "Code and docs only — recordings, checkpoints, centerline.npy, logs excluded"; `recordings_legacy/` contains only meta.json stubs (verified via git tree API)
- Image resolution & camera view: 64x64x3 obs from WGC capture; chase cam
- Label columns & units: action [steer, throttle, brake] via virtual Xbox pad; human demo recordings used for BC loss — but not published
- Speed included: yes in obs (Data Out telemetry)
- Download method: n/a
- Known problems: recordings deliberately excluded from release.
- Notes: FH6 DreamerV3 world-model RL; its `forza_telemetry.py` is another Data Out parser candidate for task 4.

## codelion/multi-drive-model-data (Hugging Face — found during GitHub search)  [ADJACENT]  (task 1)
- URL: https://huggingface.co/datasets/codelion/multi-drive-model-data
- Maintainer: codelion (HF user)
- Date (created / last updated): unverified from fetched page
- License: "other" (per HF page); depicts copyrighted game footage
- Size (frames / hours / GB): 1,073,285 frames / 153 clips total; "realistic" theme = forza-horizon + need-for-speed, 25 clips / 340,722 frames (mix of two games, per-clip game tag in metadata.jsonl)
- Image resolution & camera view: 384x216 RGB JPEG; chase/third-person gameplay footage (screen recordings)
- Label columns & units: NOT recorded inputs — labels are derived from optical flow: 7-dim actions [accel, brake, left, right, drift(always 0), speed, turn]; speed/turn are per-clip p95-normalized floats in [-1,1], left/right are binary flow-derived bits. No wheel angle.
- Speed included: yes but normalized per-clip and flow-estimated (not telemetry, not m/s)
- Per-frame synced: yes (labels attached to i-1 -> i transition; alignment independently verified by maintainer)
- Download method: huggingface_hub / hf_hub_download (tar shards; viewer disabled)
- Known problems: pseudo-labels only (no ground-truth inputs — "inferred intent", maintainer's own warning); binary left/right not a steering angle; per-clip normalization makes speeds incomparable; forza-horizon mixed with need-for-speed in one theme; actual Forza title unverified.
- Notes: Suitable at most for visual pretraining, not steering supervision. Flagged here because it surfaced during the GitHub search; task 2 may re-verify on HF.

## castortroy05/ForzaAIMasters  [ADJACENT]  (task 1)
- URL: https://github.com/castortroy05/ForzaAIMasters
- Maintainer: castortroy05
- Date (created / last updated): unverified
- License: unverified
- Size: no data — DQN reinforcement learning project, learns online; no recorded dataset
- Label columns & units: n/a (discrete 5x5 steer x speed action space, learned not cloned)
- Download method: n/a
- Notes: Forza Motorsport 7, but RL agent with no published frames/labels.

## Task 1 summary note
GitHub search (queries: "Forza Horizon 4 self driving behavior cloning", "PilotNet steering angle dataset", "FH5 self driving dataset steering", "Forza Motorsport imitation learning", "forza horizon dataset steering csv/npy/zip") found NO public dataset pairing FH4/FH5/FM frames with wheel-angle + speed. The only real Forza frame+steer+speed dataset is the gated FH6 ModelScope release (entry 1). The only directly downloadable FH4 data is HorizonNet's 534 chase-cam frames with gamepad-axis labels (entry 2). Everything else is code-only or pseudo-labelled.
