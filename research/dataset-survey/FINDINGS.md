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

## opensima36/forza_horizon_5_bc_01 (Hugging Face)  [EXACT]  (task 2)
- URL: https://huggingface.co/datasets/opensima36/forza_horizon_5_bc_01
- Maintainer: opensima36 (HF user; dataset "Generated by the game data platform BC repository consolidator")
- Date (created / last updated): created 2026-09-04, lastModified 2026-09-04 (verified via HF API)
- License: "other" (per dataset card)
- Size (frames / hours / GB): 63.7 GB total — two 7z archives `part000.7z` (33.8 GB) + `part001.7z` (29.9 GB) with .sha256 sidecars, under `极限竞速地平线5/bc_parquet/`; README states "Encrypted bytes: 63741080884" (verified via file tree). 49 downloads.
- Image resolution & camera view: unverified — contents are inside encrypted 7z archives; "BC parquet" strongly suggests behavior-cloning parquet tables (likely frames+inputs) but nothing can be inspected without a decryption key.
- Label columns & units: unverified (encrypted)
- Speed included: unverified
- Per-frame synced: unverified
- Download method: hf_hub_download / direct resolve URLs — but archives are encrypted; no key published on the page (verified: repo contains only .gitattributes, README.md, and the two .7z + .sha256 files; no key/contact info).
- Known problems: encrypted payload makes every property unverifiable; "other" license; FH5 not FH4; anonymous uploader; no dataset schema documented. Cannot be used or even evaluated as published.
- Notes: Same "Game Data Platform" family as the gated opensima05/forza_horizon_5_recordings_01 (below) and the auto-generated xiaoluo11 repo style. If the archives really contain FH5 frame+input parquet this would be the closest hub-hosted exact match — but as published it is unusable. Flagged EXACT-candidate, effectiveness zero without key.

## opensima05/forza_horizon_5_recordings_01 (Hugging Face)  [PARTIAL]  (task 2)
- URL: https://huggingface.co/datasets/opensima05/forza_horizon_5_recordings_01
- Maintainer: opensima05 (HF user; "Game Data Platform")
- Date (created / last updated): created 2026-08-29, lastModified 2026-08-30 (verified via HF API)
- License: unverified (not stated in API metadata)
- Size (frames / hours / GB): 17 raw game recordings ("Recordings: 17", layout `recordings//`); total size unverified. 26 downloads.
- Image resolution & camera view: unverified — "raw game recordings" (probably video + event logs like the xiaoluo11 repo, same platform README template)
- Label columns & units: unverified
- Speed included: unverified
- Per-frame synced: unverified
- Download method: GATED — `gated: "manual"` per HF API; access requires manual approval
- Known problems: gated; no schema published; contents (video vs frames, input types) unverifiable without approval.
- Notes: Raw-recordings counterpart of the bc_01 consolidated archives; likely the upstream data for it.

## xiaoluo11/forza-horizon-5-gameplay-data (Hugging Face)  [PARTIAL]  (task 2)
- URL: https://huggingface.co/datasets/xiaoluo11/forza-horizon-5-gameplay-data
- Maintainer: xiaoluo11 (HF user); session folder named with person 孙世博
- Date (created / last updated): created 2026-06-03 (verified via HF API); session recorded 2026-04-29
- License: "other" (README warns to review game-footage rights)
- Size (frames / hours / GB): 4.77 GB total; single session dir with one video.mkv (5.08 GB), 3 PNG screenshots (~2 MB each), video-km-frames.json (34.8 MB), fps.jsonl, macroEvents.jsonl (2.4 MB), mouseMoveBy/ToMacroEvents.jsonl, pc.json, systemInfo.json, videoStartTime.txt (verified via HF tree API)
- Image resolution & camera view: video.mkv (resolution unverified); 3 PNG ~2 MB screenshots. Camera view unverified.
- Label columns & units: NOT steering angle — `video-km-frames.json` is a per-video-frame array `{frame, time_ns, time_rel_ns, video_ts_ns, events[]}` at ~60 fps (verified head of file); `macroEvents.jsonl` rows `{type, keyCode, mouseX, mouseY, time}` — keyCodes observed: 13 (Enter), 27 (Esc), mouse dx/dy deltas (verified head of file). Player appears to drive with keyboard/mouse.
- Speed included: no (no telemetry fields seen; unverified for full file)
- Per-frame synced: yes — events are indexed per video frame with ns timestamps
- Download method: hf_hub_download (ungated)
- Known problems: labels are raw key/mouse events, not wheel angle or normalized steer; no Forza telemetry/speed; single session; frames must be extracted from MKV; "other" license.
- Notes: Real per-frame synced FH5 data exists here in principle (video + input timeline), which distinguishes it from most code-only projects — but the label modality doesn't fit wheel-angle regression.

## ILoveCorn/forza-horizon (Hugging Face)  [ADJACENT]  (task 2)
- URL: https://huggingface.co/datasets/ILoveCorn/forza-horizon
- Maintainer: Yan MA (ILoveCorn), ENGG5104 final project
- Date (created / last updated): created 2023-05-03 (verified via HF API)
- License: AFL-3.0
- Size (frames / hours / GB): n<1K images (HF size category); 28 downloads
- Image resolution & camera view: car photos/screenshots from FH4/FH5 (caption dataset; resolution unverified)
- Label columns & units: image captions (text), not steering
- Speed included: no
- Per-frame synced: n/a
- Download method: datasets library / parquet (ungated)
- Known problems: captioning dataset, no control labels at all.
- Notes: Only listed to close out HF search completeness.

## alexhexan/fm7-rio-de-janeiro-race-telemetry "Race Telemetry - Rio de Janeiro" (Kaggle)  [PARTIAL]  (task 2)
- URL: https://www.kaggle.com/datasets/alexhexan/fm7-rio-de-janeiro-race-telemetry
- Maintainer: 0x20F (Kaggle alexhexan) — same author as github.com/0x20F/forza-telemetry (Rust FH4/FH5 Data Out decoder)
- Date (created / last updated): 2023-03-25 (verified via Kaggle API)
- License: CC0: Public Domain
- Size (frames / hours / GB): ~62.6 MB total bytes (24.1 MB download); 5 laps of the ~6.2 km Rio de Janeiro full circuit in Forza Motorsport 7, 2016 BMW M2 (RWD), Xbox controller, clear/sunny (per dataset description, verified via Kaggle API). 295 downloads.
- Image resolution & camera view: NO IMAGES — telemetry + track data only
- Label columns & units: column list unverified (Kaggle file/column metadata not exposed anonymously; dataset page not JS-renderable). Expected to be decoded FM7 "Dash" packet fields (311-byte FM layout, NOT the FH4 324-byte layout) — which include speed and steer, but exact column names/units unverified.
- Speed included: yes (expected — Dash packet; units unverified)
- Per-frame synced: n/a (no frames)
- Download method: Kaggle download (public, no special gating)
- Known problems: FM7 packet layout differs from FH4 (311 vs 324 bytes); 5 laps of one circuit only; controller inputs quantize to full-throw; column schema unverified.
- Notes: Found while searching Kaggle for Forza frame+steer data; directly relevant to task 4 (telemetry logs + same-author parser whose offsets can be cross-checked). Good Tiger Data analytics-demo seed candidate.

## CORRECTION to codelion/multi-drive-model-data (entry from task 1): label provenance  [ADJACENT]  (task 2)
- URL: https://huggingface.co/datasets/codelion/multi-drive-model-data
- Correction: the earlier entry stated all 7 action dims are optical-flow pseudo-labels. Re-reading the dataset card today (2026-10-03): indices 0-4 `accel, brake, left, right, drift` are explicitly "the player's intent" — actual recorded key presses; only indices 5-6 (`speed`, `turn_rate`, p95-normalized to [-1,1]) are optical-flow measurements. So left/right ARE ground-truth binary inputs, not pseudo-labels.
- Notes: Still no steering angle (left/right are binary keys, turn_rate lags input ~6 frames by design); forza-horizon clips remain mixed with need-for-speed in the "realistic" theme. Assessment unchanged for our purpose, but "no ground-truth inputs" was wrong.

## No hits for task 2 on: Zenodo, IEEE DataPort, Papers With Code, Roboflow Universe
- Zenodo: API queries `forza`, `"forza horizon"`, `"forza motorsport"` (verified via zenodo.org/api/records) — zero relevant records (only an unrelated forestry FORZA org, an author surnamed Forza, a Wikipedia-NLP set, and an FH5 driver-fatigue report with no released telemetry).
- IEEE DataPort: search UI is JS-gated (page fetch returns no results); site-targeted web search for Forza datasets on ieee-dataport.org returned nothing — no hits, but coverage marked unverified due to JS gating.
- Papers With Code: dataset search/API now serves the unified HF front-end (paperswithcode.com redirects into HF); no Forza datasets surfaced.
- Roboflow Universe: site search for "forza" returns only unrelated object-detection projects (Fortnite, F1Tenth, car classifiers); Universe hosts classification/detection datasets, not control-label regression data — no hits.
- Kaggle extras checked: `alimmusyaffa/forza4` (7.9 MB, no description/files), `akshatkhare23x3/fh4-videos` (313 MB "FH4 Videos", no description), `tunguz/forza-and-pascal` (24.7 MB, 2017, no description), `aditmagotra/gameplay-images` (10k images, 10 games), `deepcontractor`/`harryth129`/`jasonherton12` FH5 car-stats CSVs — none publish frames + steering/speed labels.

## The Matrix Dataset (the_matrix_dataset_8M_1920_1080) — FH5 video-control pairs  [PARTIAL]  (task 3)
- URL: https://huggingface.co/datasets/ztyang196/the_matrix_dataset_8M_1920_1080 and https://www.modelscope.cn/datasets/TheMatrixDataset/the_matrix_dataset_8M_1920_1080 ; paper arXiv:2412.03568 (NeurIPS 2025); docs https://matrixteam-ai.github.io/docs/TheMatrixDatasetDocs/
- Maintainer: Matrix team — Ruili Feng, Han Zhang, Zhantao Yang et al., Alibaba Tongyi Lab + Waterloo/SJTU/HKU (HF user ztyang196 = author Zhantao Yang)
- Date (created / last updated): paper 2024-12-04; ModelScope GmtCreate 1739428099 (~2025-02-13), LastUpdatedTime 1757000747 (~2025-09); docs last updated 2025-03-25 (all verified via ModelScope API)
- License: Apache License 2.0 (explicit in ModelScope License field AND README "## License" section — verified via API). Caveat: game footage is © Microsoft regardless.
- Size (frames / hours / GB): FH5 portion = 937,900 video clips at 60 FPS, 2,861.9 combined hours, of which 262,707 clips carry control signals (Stage 2: 208,933 clips avg 6.09 s; Stage 3: 53,774 clips avg 6.07 s; Stage 1: 675,193 unlabeled clips avg 12.8 s). Repo name says "8M ... 1920_1080"; ModelScope StorageSize 12,832,696,061,947 bytes (~12.8 TB), HF page reports 11.9 TB, ~1,000 zip shards of ~7 GB each under `data/` (verified via HF tree API). Cyberpunk 2077 portion (300k pairs) marked "coming soon".
- Image resolution & camera view: recorded 2560x1600 @ 60 FPS via OBS, HUD/GUI removed with ReshadeEffectShaderToggler; released set is 1920x1080 per repo name (zip contents unverified). Third-person/chase driving across mixed biomes (woods/grass/sea/field/river/others, ratio 12/15/18/16/15/9/15%).
- Label columns & units: per-second DISCRETE control labels: D (drive/forward) 51%, DR (drive-right) 25%, DL (drive-left) 24%, ~3 signal-change events per 6 s clip — i.e. 1 Hz 3-way commands generated by a randomized script driving with keyboard, NOT wheel angle and not even a continuous steer axis. README claims the release also includes "3D positional telemetry" (positions XYZ, velocities, accelerations collected via socket for collision/stagnation filtering) — telemetry fields inside the zips unverified.
- Speed included: telemetry logged during collection; whether speed/velocity ships per-frame in the released zips is unverified (docs describe it mainly as a filtering aid)
- Per-frame synced: partially — control signal is 1 Hz per clip timeline, synced to the 60 FPS video; not per-frame
- Download method: modelscope/huggingface_hub `snapshot_download`; HF file tree is public (verified); docs suggest a token but no gating flag observed on HF. 9,380 ModelScope downloads.
- Known problems: labels are scripted random 3-way keyboard commands (not human demonstration, not wheel degrees — unusable as steering regression targets); only ~28% of clips have any labels; no HUD-masked bonnet cam; 12 TB scale makes casual download impractical; zip internal schema unverified.
- Notes: First academically-published FH5 frame+input dataset that is actually downloadable. Their collection "GameData Platform" (OBS + simulated kbd + telemetry via socket + Reshade HUD toggler) is the same named pipeline behind the opensima05/opensima36/xiaoluo11 HF repos found in task 2 — strongly suggests those gated/encrypted repos are members of this data family. Could seed a visual-pretraining set, not steering supervision.

## RealPlay / "From Virtual Games to Real-World Play" (arXiv 2506.18901)  [ADJACENT]  (task 3)
- URL: https://arxiv.org/abs/2506.18901 ; project https://wenqsun.github.io/RealPlay/ ; code https://github.com/wenqsun/Real-Play
- Maintainer: Wenqiang Sun, Fangyun Wei et al. — HKUST, Microsoft Research, Univ. of Sydney, Tsinghua, Waterloo
- Date (created / last updated): arXiv June 2025 (paper year 2025)
- License: unverified (no LICENSE in repo; site CC BY-SA)
- Size: no data released — repo contains only README + assets ("All Code will be released soon", 5 commits, verified via repo page)
- Image resolution & camera view: FH5 gameplay video chunks (paper's labeled game data); resolution unverified
- Label columns & units: control signals "move forward / turn left / turn right" — discrete, same coarse granularity as Matrix-style labels; no wheel angle
- Speed included: no (not mentioned)
- Per-frame synced: yes within video chunks (frame-level control conditioning), but dataset not public
- Download method: n/a
- Known problems: code and data unreleased as of 2026-10-03; interactive video generation, not behavior cloning.
- Notes: Second academic work training on labeled FH5 data (author overlap with Matrix team: Hongyang Zhang). Confirms a FH5 labeled-collection exists academically but adds nothing downloadable.

## Patadiya et al., "Application of Deep Learning to Generate Auto Player Mode in Car Based Game" (IEEE CICN 2024)  [ADJACENT]  (task 3)
- URL: indexed at https://exa.ai/library/publication/st8k46cdbr3 ; TOC listing https://www.proceedings.com/content/078/078614webtoc.pdf (p. 233); ORCID record https://orcid.org/0000-0002-2502-5680
- Maintainer: Kunjesh Patadiya, Rituraj Jain, Jay Moteriya, Damodharan Palaniappan, Kumar J. Parmar, Premavathi T (IEEE CICN 2024, published ~2024-12-22)
- License: n/a (IEEE paper, no repo)
- Size: paper reports "hours of gameplay" recorded in FH5 for AlexNet-DRL training; dataset NOT released (no repo/data link found)
- Label columns & units: unverified — abstract mentions "vehicle telemetry" among recorded data but no schema published
- Speed included: claims telemetry collected; unverified, not released
- Download method: n/a
- Known problems: no data/code release; accuracy claims (92.5% "perceptual accuracy") look qualitative; low-tier venue.
- Notes: Confirms the pattern: several academic FH5 driving efforts exist, none publish their frame+input data.

## Pratama et al., "Car Detection over Network using YOLOv8 in Forza Horizon 4" (IEEE TSSA 2023)  [ADJACENT]  (task 3)
- URL: https://doi.org/10.1109/TSSA59948.2023.10366964 (metadata verified via Semantic Scholar API)
- Maintainer: Vitradisa Pratama et al. (TSSA 2023)
- License: n/a
- Size: FH4 vehicle-detection dataset used for YOLOv8 training; size and release status unstated — no repo/data link found
- Label columns & units: bounding-box detection labels, NOT driving controls
- Download method: n/a
- Notes: Only FH4-specific academic hit found; wrong task (perception, not steering).

## Task 3 summary note
Searched arXiv API (`all:forza` — 18 hits, almost all surname "Forza" false positives; only relevant: 2412.03568 Matrix, 2506.18901 RealPlay), Semantic Scholar API (`forza horizon`, `forza driving imitation learning` — only YOLOv8-FH4 paper + non-technical hits), thesis repositories via site-targeted web search (diva-portal.org, repository.tudelft.nl, ntnuopen, dspace.cvut.cz, publications.lib.chalmers.se, theseus.fi, IADT onshow — the nearest, Manu Jose's "Pinewood Rally" thesis, is a Unity waypoint-AI game project, not Forza and no dataset). Also surfaced: jimhoggey/SelfdrivingcarForza (FH3 OpenCV lane-detection research project, prerecorded-video approach, no data) and an H-BRS course project "Self-Driving car in Forza" (FH4 ConvNeXt-LSTM, Ibrahim Shakir Syed — LinkedIn only, no data). Verdict for task 3: the ONLY academically-originated downloadable Forza frame+control dataset is The Matrix Dataset (FH5, discrete 1 Hz D/DR/DL labels). No thesis or paper releases FH4/FH5 frames with wheel-angle or gamepad-axis steering + speed.

