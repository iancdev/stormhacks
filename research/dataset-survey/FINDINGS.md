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

## richstokes/Forza-data-tools — FH4_packetformat.dat layout doc + Go telemetry tool  [PARTIAL]  (task 4)
- URL: https://github.com/richstokes/Forza-data-tools ; layout doc https://github.com/richstokes/Forza-data-tools/blob/master/FH4_packetformat.dat (raw fetched and read in full)
- Maintainer: richstokes
- Date (created / last updated): repo created 2019-06-14, last push 2026-06-06 (verified via GitHub API); 115 stars
- License: GPL-3.0 (verified via GitHub API license field)
- Size (frames / hours / GB): n/a — tool + packet-format doc; no telemetry logs shipped (only `dash/sample.json`, a one-packet dashboard sample)
- Image resolution & camera view: n/a
- Label columns & units (degrees / normalized / gamepad axis): FH4_packetformat.dat lists the full 324-byte wire format field-by-field (s32 IsRaceOn, ..., s32 CarCategory, u32 HorizonUnknown1, u32 HorizonUnknown2, f32 PositionX/Y/Z, f32 Speed "meters per second", ..., s8 Steer, s8 NormalizedDrivingLine, s8 NormalizedAIBrakeDifference, u8 HorizonTrailingUnknown)
- Speed included: yes — f32 Speed explicitly documented as m/s
- Per-frame synced: n/a
- Download method: git clone / raw file; Go binary `fdt` logs telemetry to CSV with `-c log.csv`, `-z` flag selects the Horizon layout for FH4/5/6
- Known problems: the 12-byte Horizon extension semantics are community guesses (CarCategory + 2 unknowns); no published data
- Notes (offset agreement): AGREES with our offsets exactly — IsRaceOn s32 @0, Speed f32 @256 (m/s), Steer s8 @320, packet = 324 bytes incl. trailing u8 @323. This is the cleanest single-file FH4 layout doc found; it also notes "older 323-byte packets end before this field".

## Forza Data Out packet-layout documentation status  [PARTIAL]  (task 4)
- URL: forums.forza.net/t/data-output/74308 (original FH4 packet-format discovery thread); support.forzamotorsport.net/hc/en-us/articles/21742934024211-Forza-Motorsport-Data-Out-Documentation (official FM doc); forums.forzamotorsport.net turn10 post m926839 (original FM7 spec, linked from nettrom/fdp.py)
- Maintainer: Forza community / Turn 10
- Date (created / last updated): forum thread ~2018-2019; forums retired July 2026 (verified: forums.forza.net now serves a "Forza Forums Farewell" page, all threads 410-gone)
- License: n/a
- Size: n/a
- Label columns & units: n/a
- Speed included: n/a
- Per-frame synced: n/a
- Download method: n/a — the forum thread content survives only in search-cache excerpts and mirrors
- Known problems: BOTH primary sources are now unreachable — forums.forza.net retired (verified farewell page 2026-10-03), and the support.forzamotorsport.net article returned HTTP 403 to our fetch (unverified; it documents the FM2023 311+20=331-byte format, not FH4 anyway). The de-facto FH4 spec now lives in code comments/struct definitions of the parsers below.
- Notes: Cached forum excerpts document the FH4 packet as: [0]-[231] FM7 sled, [232]-[243] "new unknown data" (12 bytes), [244]-[322] FM7 Car Dash data, [323] unknown trailing byte — i.e. exactly our layout (Speed @244+12=256, Steer @244+76=320). geeooff/forza-data-web (C#) was cited by Grvs44 as an independent layout source (repo confirmed to exist with ForzaDataReader.HorizonExtras.cs; offsets unverified).

## nettrom/forza_motorsport — Python FM7/FH4 Data Out parser + CSV/TSV logger  [PARTIAL]  (task 4)
- URL: https://github.com/nettrom/forza_motorsport (fdp.py fetched raw and read in full)
- Maintainer: Morten Wang (nettrom)
- Date (created / last updated): copyright 2018 in fdp.py header; 14 commits, 56 stars (verified via repo page)
- License: MIT (LICENSE file + header comment verified)
- Size: no data shipped — user records own TSV/CSV via data2file.py
- Label columns & units: all sled+dash fields as named props; `speed` = wire value (m/s), `steer` = s8 -127..127
- Speed included: yes (m/s as sent on the wire)
- Per-frame synced: n/a
- Download method: pip-able repo; `python data2file.py -p fh4 <port> out.csv`
- Known problems: FH4 path does `patched_data = data[:232] + data[244:323]` — it silently DISCARDS the 12-byte Horizon extension and the trailing byte rather than documenting them; no length validation before slicing.
- Notes (offset agreement): AGREES (functionally) — after patching, the FM7 dash struct places Speed at original offset 256 (f32) and Steer at original offset 320 (s8); IsRaceOn i32 @0. Oldest confirmed Python FH4 parser; jasperan's code below is a clear derivative.

## makvoid/Blog-Articles Forza-Telemetry DataPacket.py — Python parser  [PARTIAL]  (task 4)
- URL: https://github.com/makvoid/Blog-Articles/blob/main/Forza-Telemetry/util/data_packet.py (fetched raw, read in full); companion to a telemetry blog series
- Maintainer: makvoid
- Date (created / last updated): unverified
- License: unverified (no license check performed on repo)
- Size: code only
- Label columns & units: `speed` wire f32 m/s converted to mph (*2.237); `steering_angle` = s8 -127..127; throttle/brake/clutch/handbrake 0-255 -> %
- Speed included: yes (m/s raw; displayed as mph)
- Per-frame synced: n/a
- Download method: git clone
- Known problems: the 12-byte Horizon extension is guessed as `car_type`(i32 @232) + `impact_x`/`impact_y` (f32 @236/240); display-unit conversion makes raw values non-obvious; `_convert` returns int-percent for pedals.
- Notes (offset agreement): AGREES — `_horizon_format = '<iI27f4i20f5i' + 'i19fH6B4b'` = exactly 324 bytes; attribute order places `speed` at wire offset 256 (f32 m/s) and `steering_angle` at 320 (s8); `active` = IsRaceOn i32 @0. 323-vs-324 handled by naming the last byte `unknown`.

## nikidziuba/Forza_horizon_data_out_python — minimal Python UDP server  [PARTIAL]  (task 4)
- URL: https://github.com/nikidziuba/Forza_horizon_data_out_python (server.py + data_format.txt both fetched and read in full)
- Maintainer: nikidziuba
- Date (created / last updated): 3 commits; dates unverified; 14 stars
- License: none visible
- Size: code only
- Label columns & units: dict of all wire fields; Speed f32 m/s, Steer s8
- Speed included: yes
- Per-frame synced: n/a
- Download method: git clone
- Known problems: data_format.txt ENDS at NormalizedAIBrakeDifference (@322) — the format covers only 323 bytes, silently dropping the FH4 trailing byte; no packet-length validation (it just consumes fields sequentially, so a 324-byte packet still parses correctly); decoder reads sequentially rather than by absolute offset.
- Notes (offset agreement): AGREES — `hzn HorizonPlaceholder` is an explicit 12-byte skip @232-243, so Speed lands @256 (f32), Steer s8 @320, IsRaceOn s32 @0. Agreement is positional (sequential decode), not by offset constants.

## jasperan/forza-horizon-5-telemetry-listener — Python FH4/FH5 listener + dashboard/analytics platform  [PARTIAL]  (task 4)
- URL: https://github.com/jasperan/forza-horizon-5-telemetry-listener (src/data_packet.py fetched raw and read in full; repo tree verified)
- Maintainer: jasperan
- Date (created / last updated): created 2021-11-10, pushed 2026-09-15 (verified via GitHub API); 11 stars
- License: none detected by GitHub (LICENSE file is a 56-byte stub — effectively unlicensed)
- Size: no telemetry data shipped; runtime DB/JSON records gitignored
- Label columns & units: identical prop names to nettrom (`speed` m/s wire, `steer` s8)
- Speed included: yes (m/s)
- Per-frame synced: n/a
- Download method: git clone; `python app.py --no-db` for dashboard-only mode
- Known problems: `data_packet.py` is a trimmed copy of nettrom's fdp.py — same `data[:232] + data[244:323]` patch, drops Horizon extension + trailing byte; Oracle-DB heritage makes the full stack heavy, but --no-db mode works.
- Notes (offset agreement): AGREES (same scheme as nettrom — Speed @256, Steer s8 @320, IsRaceOn i32 @0). Most feature-rich Python FH telemetry platform found (dashboard, lap/session tracking, track auto-mapping, coach) — relevant tooling reference for a Tiger Data analytics demo even though it ships no data.

## Grvs44/Forza-Telemetry-Export — Python package exporting telemetry to CSV/SQLite/binary  [PARTIAL]  (task 4)
- URL: https://github.com/Grvs44/Forza-Telemetry-Export (export.py, dashh_fields.csv, unpack_horizon.py, LICENSE, README all fetched and read)
- Maintainer: Elli Greaves (Grvs44)
- Date (created / last updated): copyright 2025; current HEAD verified via git tree API
- License: Clear BSD (LICENSE file read in full)
- Size: no data shipped
- Label columns & units: dash_fields.csv column order = PositionX/Y/Z,Speed,Power,Torque,TireTemps,Boost,Fuel,DistanceTraveled,BestLap,LastLap,CurrentLap,CurrentRaceTime,LapNumber,RacePosition,Accel,Brake,Clutch,HandBrake,Gear,Steer,NormalizedDrivingLine,NormalizedAIBrakeDifference; dashh_fields.csv adds CarCategory,Unknown1,Unknown2 for the 12-byte extension
- Speed included: yes (f32 m/s wire)
- Per-frame synced: n/a
- Download method: pip package (`python -m forza_telemetry_export.export_csv ... {sled,dash,dashh,dashm}`)
- Known problems: README calls the Horizon extension "3 8-byte numbers" (should be 3x4 bytes — the struct format `'III'` is correct, the prose is wrong); Horizon trailing byte handling is a workaround (`unpack_horizon` yields `t[:-1]`); README describes extension semantics as unknown.
- Notes (offset agreement): AGREES — `get_format(DASHH) = sled + 'III' + dash + 'B'` = 232+12+79+1 = 324 bytes exactly; Speed @256 f32, Steer @320 (dash 'bbb' tail: Steer, DrivingLine, AIBrakeDiff @320-322, then trailing 'B' @323); IsRaceOn i32 @0. Also documents the four packet sizes: SLED 232, DASH 311, DASHH 324, DASHM 331 (FM2023).

## bobbythehuman/Race-Telemetry-Package (PyPI "RaceTelemetry") — multi-game Python telemetry lib, FH4/5/6  [PARTIAL]  (task 4)
- URL: https://github.com/bobbythehuman/Race-Telemetry-Package ; PyPI https://pypi.org/project/RaceTelemetry/ ; FH4 struct https://raw.githubusercontent.com/bobbythehuman/Race-Telemetry-Package/HEAD/src/RaceTelemetry/data_structures/FH4_struct.py (fetched and read in full)
- Maintainer: bobbythehuman
- Date (created / last updated): PyPI latest 5.10.11.post1 (verified via pypi.org/pypi/RaceTelemetry/json); PyPI page itself JS-gated
- License: LGPL-2.1 (LICENSE file read — GNU Lesser GPL v2.1 header)
- Size: library only; has tests/Game_Specific/Forza Horizon 4,5,6 examples
- Label columns & units: ctypes struct with full field names; Speed f32 m/s, Steer s8 -127..127; commonFieldMap exposes speed/steering/throttle/brake
- Speed included: yes (m/s)
- Per-frame synced: n/a
- Download method: `pip install RaceTelemetry`
- Known problems: `_pack_ = 1` is commented out in the ctypes struct — works only because natural alignment introduces no padding before the byte fields; struct declares 323 bytes of fields and ctypes pads sizeof to 324. Horizon extension is named CarCategory(i32)/SmashableVelDiff(f32)/SmashableMass(f32) — the Smashable* interpretation is a community guess (same names as Ayin1412's drive.py).
- Notes (offset agreement): AGREES — DashData struct: IsRaceOn c_int32 @0, 3-field Horizon extension @232-243, Speed c_float @256, Steer c_int8 @320. The only pip-installable Python package verified to handle the FH4 324-byte layout.

## ricky5932TW/End2End-autodrive-image-steer — forza_autodrive/telemetry.py  [PARTIAL]  (task 4)
- URL: https://github.com/ricky5932TW/End2End-autodrive-image-steer/blob/main/forza_autodrive/telemetry.py (fetched raw, read in full)
- Maintainer: ricky5932TW
- Date (created / last updated): 2026-era (from task 1 entry)
- License: none stated
- Size: parser only (dataset not released, per task 1)
- Label columns & units: returns {Speed: f32 @256 (m/s), Gear: u8 @319, Steer: s8 @320}; ~25 more fields listed but commented out, including IsRaceOn i32 @0
- Speed included: yes (m/s)
- Per-frame synced: n/a — threaded receiver keeps latest packet + age
- Download method: git clone
- Known problems: most fields commented out (parser only emits Speed/Gear/Steer); raises on any non-324-byte packet.
- Notes (offset agreement): AGREES exactly — `DASH_PACKET_SIZE = 324`, `Speed: _f32(data, 256)`, `Steer: _s8(data, 320)`, `Gear: _u8(data, 319)`, commented `#"IsRaceOn": _i32(data, 0)`. Byte-for-byte our offsets.

## shoal-rat/Horizon_FSD — forza_telemetry.py + docs/telemetry_format.md  [PARTIAL]  (task 4)
- URL: https://github.com/shoal-rat/Horizon_FSD/blob/main/forza_telemetry.py (fetched raw, read in full)
- Maintainer: shoal-rat
- Date (created / last updated): 2026-06 (task 1 entry)
- License: unverified
- Size: parser only; also `telemetry_probe` companion per docstring
- Label columns & units: full ~94-field dataclass; Speed m/s, AccelInput/BrakeInput 0..255, Steer s8 -127..127 (derived `steer_norm` = steer/127, `throttle` = accel/255); GEAR_SHIFTING = 11 sentinel documented
- Speed included: yes (m/s; speed_kmh/mph properties)
- Per-frame synced: n/a
- Download method: git clone
- Known problems: targets FH6 (same wire format as FH4/5); asserts finite values on a subset of physics fields — drops NaN packets entirely.
- Notes (offset agreement): AGREES exactly — SPEC struct is byte-identical to ours: is_race_on i @0, horizon_car_category i @232 + 2 unknowns @236/240, position @244, speed f @256, gear B @319, steer b @320, trailing B @323; `assert PACKET_SIZE == 324`. Docstring says layout "CONFIRMED against live FH6 in Phase 0"; also tolerates 323-byte older-build packets (pads with \x00). Strongest independent live confirmation found.

## Ayin1412/ForzaHorizon6-VisionAI — telemetry block inside scripts/drive.py  [PARTIAL]  (task 4)
- URL: https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/scripts/drive.py (fetched raw; TELEMETRY_FIELDS block read in full)
- Maintainer: Wenhao Li (Ayin1412)
- Date (created / last updated): 2026-07 (task 1 entry)
- License: Apache-2.0 for code (LICENSE/NOTICE present); dataset artifacts CC BY-NC 4.0 (per task 1)
- Size: parser only
- Label columns & units: full field list; Speed f m/s; Steer 'b' s8; Steer is used with steer_calibration.json/steer_map.json for gamepad-unit mapping
- Speed included: yes (m/s)
- Per-frame synced: n/a
- Download method: git clone
- Known problems: Horizon extension named `("CarGroup","I")`, `("SmashableVelDiff","f")`, `("SmashableMass","f")` — a different guess at the 12 bytes than the CarCategory/Unknown convention; WheelInPuddle fields typed 'i' instead of 'f' (4 bytes either way, offsets unaffected, values misinterpreted); format string ends with 'x' pad byte for the trailing unknown.
- Notes (offset agreement): AGREES — IsRaceOn 'i' @0, Speed f @256, Steer b @320, calcsize = 324 (incl. trailing 'x'). The alternative CarGroup/Smashable* naming for bytes 232-243 is worth noting since two repos (this + RaceTelemetry) now use it vs the CarCategory convention elsewhere.

## Estetika101/pacefinderapp — parsers/forza.py  [ADJACENT]  (task 4)
- URL: https://github.com/Estetika101/pacefinderapp/blob/main/parsers/forza.py (fetched raw, read in full)
- Maintainer: Estetika101
- Date (created / last updated): unverified
- License: unverified
- Size: parser only
- Label columns & units: standard FM7 dash field names; speed raw (m/s) + derived speed_mph; steer raw s8
- Speed included: yes
- Download method: git clone
- Known problems: MISLABELED packet sizes — `FM_PACKET_SIZE_FH = 331` is claimed as "Forza Horizon 4/5 Car Dash" but 331 is actually the FM2023 extended packet; the FH4/FH5 324-byte packet is REJECTED by parse_forza (returns None for any length other than 311/331). It also appends tireWear+trackOrdinal as if they were a Horizon extension.
- Notes (offset agreement): DISAGREES / cannot parse FH4 — its dash block is the FM layout at base 232 (Speed would be @244, Steer @308 in FM packets). Using this parser on FH4 data yields nothing; using it as an offset reference would be actively wrong for Horizon packets.

## 0x20F/forza-telemetry — Rust Data Out decoder + session CSV recorder (author of the task-2 Kaggle dataset)  [PARTIAL]  (task 4)
- URL: https://github.com/0x20F/forza-telemetry (src/decoder/formats.rs fetched raw, read in full; repo + captures/ tree verified)
- Maintainer: 0x20F (= Kaggle alexhexan, per task 2 entry)
- Date (created / last updated): created 2023-03-24, pushed 2026-05-09 (verified via GitHub API); 4 stars
- License: none (GitHub license field = null)
- Size: tool + tiny capture artifacts — captures/ contains ONE `*.session.toml` (schema_version=2, source=fm2023_extras, car_ordinal 1655, track 1640, frames_written=20848) + one `frame_14538_drift.json`; the referenced CSV itself is NOT in the repo
- Label columns & units: RawPacket fields; CSV schema v2 (see src/csv_writer); tire temps converted F->C on ingest
- Speed included: yes (f32 m/s at dash base+12)
- Per-frame synced: per-packet; adds recv_time_ns timestamps on receipt
- Download method: git clone / cargo build (`forza-telemetry record --port 7777 --out captures/`)
- Known problems: not Python (Rust); ships no usable telemetry log — only a session sidecar for a missing CSV; unlicensed.
- Notes (offset agreement): AGREES — Source::DashHorizon (324 bytes) sets dash_base=244; Speed @244+12=256 f32, Steer @244+76=320 i8, IsRaceOn i32 @0. Cross-language corroboration. Its Kaggle FM7 dataset (task 2 entry) is the only published decoded-telemetry log found anywhere in this survey.

## austinbaccus/forza-telemetry — C#/Electron recorder + dashboard  [ADJACENT]  (task 4)
- URL: https://github.com/austinbaccus/forza-telemetry (ForzaCore/FMData.cs, PacketParse.cs, Program.cs, DataPacket.cs fetched raw and read; tree verified)
- Maintainer: austinbaccus
- Date (created / last updated): unverified
- License: present (1070-byte LICENSE — MIT-sized; type unverified)
- Size: app only — `data/` dir contains an empty `default` file; recorded CSVs are user-local
- Label columns & units: DataPacket class fields; Speed m/s wire, Steer int (s8)
- Speed included: yes
- Per-frame synced: recorder decimates to recordRateMS = 50 ms (20 Hz), not per-packet
- Download method: git clone; ElectronCgi app
- Known problems: type sloppiness — IsRaceOn read as float at @0 (works accidentally since any nonzero i32 != 0f... reads 01 00 00 00 as 1.4e-45 > 0); CarOrdinal/CarClass/PI/Drivetrain/NumCylinders read as GetUInt8 (only the low byte of each i32!); WheelOnRumbleStrip read as f32 instead of i32. Offsets are right; decode types are sloppy.
- Notes (offset agreement): AGREES — `AdjustToBufferType` sets `FMData.BufferOffset = 12` for 324-byte FH4 packets, shifting dash reads so Speed = 244+12 = 256 and Steer = 308+12 = 320; 232/311/324/331 lengths all dispatched.

## Task 4 summary note
TELEMETRY LOGS: the only published decoded-telemetry dataset found anywhere remains the Kaggle `alexhexan/fm7-rio-de-janeiro-race-telemetry` (task 2 entry — FM7 311-byte layout, CC0, ~62 MB, 5 laps). NO public FH4/FH5 CSV/parquet telemetry log was found; every tool records user-local files (nettrom, Grvs44, richstokes, jasperan, austinbaccus, Yurikada, 0x20F) and none ship their captures. 0x20F's repo has one session sidecar for a CSV that isn't included.
PACKET LAYOUT: FH4/FH5 dash = 232-byte sled + 12-byte Horizon extension (@232-243, semantics unverified — two naming conventions: CarCategory+Unknown1+Unknown2 vs CarGroup+SmashableVelDiff+SmashableMass) + 79-byte FM dash block @244-322 + 1 trailing byte @323 = 324 bytes. Our offsets (IsRaceOn i32 @0, Speed f32 @256 m/s, Steer s8 @320, Gear u8 @319, Accel u8 @315, Brake u8 @316) are confirmed by EVERY FH4-capable parser checked: nettrom, jasperan, nikidziuba, makvoid, Grvs44, RaceTelemetry, ricky5932TW, shoal-rat (asserted + live-verified on FH6), Ayin1412, plus Rust (0x20F) and C# (austinbaccus) corroboration — 12 independent implementations, zero disagreements. One parser (Estetika101/pacefinderapp) mislabels the 331-byte FM2023 packet as FH4/FH5 and would reject real 324-byte Horizon packets. Official docs: Forza forums retired July 2026 (thread 74308 gone, survives in cache excerpts); support.forzamotorsport.net article 403'd to our fetch (FM2023-format doc anyway). Best single doc artifact: richstokes/FH4_packetformat.dat.
PARSER VALIDATION RECOMMENDATION: unit-test our decoder against makvoid/Grvs44/nettrom equivalent fields on a shared synthetic packet; note tele_steer (s8 game axis, -127..127) is NOT the wheel angle — it is the post-mapping input the game applied, so expect it to differ from steer_deg under speed-sensitive steering limits.

## DeepDrive GTAV baseline dataset — archive.org deepdrive-baseline-uint8 (600K frames, 42 h)  [PARTIAL]  (task 5)
- URL: https://archive.org/details/deepdrive-baseline-uint8 (single file https://archive.org/download/deepdrive-baseline-uint8/gtav-42-hours-uint8.tar.gz); 2017 site also lists Dropbox `gtav-42-hours-uint8.tar.gz` and an 80 GB uint32 Google Drive variant — those two unverified
- Maintainer: Craig Quiter (deepdrive.io, later Voyage/OpenAI-Universe ecosystem); archive.org upload by Matthew Kleinsmith (mwksmith@gmail.com)
- Date (created / last updated): dataset dated 2016-11-14, uploaded to archive.org 2016-12-15 (verified via archive.org metadata API)
- License: none stated on the item page or in metadata — unverified
- Size (frames / hours / GB): 600K images / 42 hours of driving per the 2017 deepdrive.io page (Wayback 2017-01-11 capture read); tarball `gtav-42-hours-uint8.tar.gz` = 52,368,286,067 bytes (~52.4 GB), item_size verified via metadata API
- Image resolution & camera view: "forward mounted camera" per 2017 site (hood-equivalent); resolution unverified — uint8 RGB frames, era suggests ~227x227-640x480
- Label columns & units (degrees / normalized / gamepad axis): steering, throttle, yaw, and forward-speed control values produced by the in-game AI driver (NOT a human). Units unverified on the item; the companion universe-windows-envs vnc-gtav README defines the action space as continuous joystick axes -1..1 (steering x-axis, throttle z-axis)
- Speed included: yes — "forward-speed" is one of the four recorded control values (units unverified)
- Per-frame synced: yes — frame + control pairs recorded from the running game (sync method undocumented)
- Download method: archive.org direct HTTP download or BitTorrent — item verified live today, metadata API returns workable servers
- Known problems: labels come from the game's AI driver, not human wheel input — AI steering is quantized to GTA's input model; no license; dataset README was a Google Doc (link likely dead, unverified); camera is forward-mounted but not a masked bonnet cam; 2016-era GTA V build; tar-in-one-blob (no per-file random access)
- Notes: This is THE canonical "GTA V steering dataset" — it predates DeepGTAV and is what most 2016-2018 GTA-V-driving work refers to. Different from the modern deepdrive.io dataset below.

## deepdrive.io modern dataset (Unreal sim, NOT GTA V) — 100 GB / 8.2 h  [ADJACENT]  (task 5)
- URL: https://docs.deepdrive.io/ and https://github.com/deepdrive/deepdrive ; data at s3://deepdrive/data/baseline_tfrecords (+ legacy s3://deepdrive/data/baseline HDF5)
- Maintainer: deepdrive (Craig Quiter et al.)
- Date (created / last updated): current-gen UE4 simulator era (~2019+); exact dates unverified
- License: unverified
- Size (frames / hours / GB): 100 GB, 8.2 hours, camera+depth+steering+throttle+brake of an oracle path-following agent plus DAgger corrective data
- Image resolution & camera view: rotates between three cameras (normal, wide, semi-truck) with random intrinsic/extrinsic perturbations per episode — not a fixed bonnet cam
- Label columns & units: steering/throttle/brake, normalized -1..1 action space per docs
- Speed included: vehicle data available (depth + vehicle state) — per-frame speed column unverified
- Per-frame synced: yes (recorded episodes)
- Download method: `aws s3 sync s3://deepdrive/data/baseline_tfrecords .` — bucket LIST returned HTTP 403 to anonymous fetch today; download accessibility unverified
- Known problems: NOT GTA V (own Unreal map/physics); AI-oracle + DAgger labels, not human; S3 access unverified.
- Notes: Included to disambiguate the DeepDrive name — only the 2016 archive.org item above is actual GTA V.

## Alzaib/Autonomous-Self-Driving-Car-GTA-5 — published 100k-frame hood-cam dataset (Google Drive)  [PARTIAL]  (task 5)
- URL: https://github.com/Alzaib/Autonomous-Self-Driving-Car-GTA-5 ; data folder https://drive.google.com/drive/folders/1R787vkWaMe5nsWyLpbXTG55aUv4YteTo
- Maintainer: Alzaib Nasiruddin
- Date (created / last updated): npy files dated Jul 15-22, 2020 (verified in live Drive listing)
- License: none for the data (repo LICENSE status unverified; README asserts none)
- Size (frames / hours / GB): README claims 100,000 images collected (39,046 after balancing); Drive folder contains training_data-1.npy .. training_data-25.npy at ~100-132 MB each (~2.9 GB total) — consistent with 25 files x 4,000 samples = 100k raw (collect_data.py saves every 4,000 samples)
- Image resolution & camera view: screen region (0,40,800,640) grabbed then resized to 160x120 GRAYSCALE (verified in collect_data.py); README requires "turn on hood camera" — hood cam confirmed
- Label columns & units: `output = [axis_0, axis_3]` = [steering, throttle] read via pygame `joystick.get_axis()` → float -1.0..1.0 joystick-axis units (verified in collect_data.py). NOT wheel degrees. No brake channel.
- Speed included: no
- Per-frame synced: yes — frame grab and axis read happen in the same loop iteration at ~20 fps (clock.tick(20))
- Download method: public Google Drive folder — anonymous file listing verified today (2026-10-03); per-file download links present
- Known problems: grayscale + tiny 160x120 images (weak match to our 200x66 RGB pipeline); joystick-axis labels, not degrees; highway driving only; no speed; no license; Google Drive long-term availability not guaranteed; raw set is heavily unbalanced (only 39k usable after balancing).
- Notes: The closest GTA V analog to our record format: continuous steering axis + throttle + per-frame image, hood cam, PilotNet trainer. Best downloadable GTA V set for steering regression found.

## sartajbhuvaji "Self Driving in GTA V" (Kaggle copy of HF self-driving-GTA-V)  [PARTIAL]  (task 5)
- URL: https://www.kaggle.com/datasets/sartajbhuvaji/self-driving-in-gta-v ; canonical original https://huggingface.co/datasets/sartajbhuvaji/self-driving-GTA-V
- Maintainer: Sartaj Bhuvaji
- Date (created / last updated): Kaggle v1 created 2023-12-25 (verified via Kaggle API); HF commits "over 2 years ago" per search index
- License: MIT (Kaggle API licenseName verified)
- Size (frames / hours / GB): Kaggle copy totalBytes 1,944,347,225 (~1.94 GB; file list not exposed anonymously — contents unverified). HF original claims ~1M frames / ~362 GB in ~200 files of ~1.81 GB each, plus a "mini" subset and per-file key-count CSVs (per dataset-card text in search index — unverified)
- Image resolution & camera view: card says 800x600 windowed capture, stored images 480x270 RGB; "Camera: Hood Cam", Vehicle Camera Height Low, head-bobbing off (card text, unverified)
- Label columns & units: one-of-9 one-hot keyboard classes {W,S,A,D,WA,WD,SA,SD,NK} — discrete classification labels, NOT steering angle or continuous axis; ~74% of frames are 'W' (counts verified via the published data-count figures)
- Speed included: no
- Per-frame synced: yes (sentdex-style npy frame+key pairs)
- Download method: Kaggle public (57 downloads, isPrivate=false). HF original returned HTTP 401 to anonymous fetches today (page, API, tree, raw README all 401) — repo possibly made private/gated; status unverified
- Known problems: keyboard-class labels cannot supervise steering regression; the public Kaggle copy (~1.9 GB) is a tiny fraction of the claimed 362 GB set; HF canonical copy currently inaccessible.
- Notes: Largest GTA V frame+input dataset found anywhere, but the label modality is keys. A→D one-hot could at best seed a turn-left/turn-right classifier, not a wheel-angle regressor.

## kfk42kfk GTA V sentdex-format datasets (Kaggle, 70k + 140k)  [PARTIAL]  (task 5)
- URL: v1 https://www.kaggle.com/datasets/kfk42kfk/gta-v-self-driving-car ("70K"); v2 https://www.kaggle.com/datasets/kfk42kfk/gtav-new ("GTA-V 140k dataset")
- Maintainer: Furkan K (kfk42kfk)
- Date (created / last updated): v1 last updated 2021-06-04 (dataset version 3); v2 created 2022-04-09 (both verified via Kaggle API)
- License: v1 "Unknown"; v2 "GPL 2" (per Kaggle API)
- Size (frames / hours / GB): v1 ~13.4 GB (70k per subtitle); v2 ~7.03 GB, 140k images
- Image resolution & camera view: sentdex pygta5 format — 800x600 windowed screen grabs expected; resolution/camera unverified
- Label columns & units: sentdex one-hot WASD key classes (per "for Sentdex's self driving car series" description); continuous steering unlikely — unverified inside archives
- Speed included: no
- Per-frame synced: yes (frame+key npy pairs)
- Download method: public Kaggle downloads (59 and 29 downloads respectively)
- Known problems: keyboard labels, not wheel angle; license inconsistent between versions (Unknown vs GPL-2); camera view unverified.
- Notes: These are community recreations — Sentdex himself never published his pygta5 training data; these two plus sartajbhuvaji's are the de-facto "sentdex-format" public sets.

## dhruv-sirohi/GTAV-Imitation-Learning — 50k joystick-steer frames, data NOT published  [ADJACENT]  (task 5)
- URL: https://github.com/dhruv-sirohi/GTAV-Imitation-Learning
- Maintainer: dhruv-sirohi
- Date (created / last updated): 72 commits; dates unverified
- License: none stated (no LICENSE in root listing)
- Size: NO DATA SHIPPED — repo contains only `data collection scripts/` and `training_scripts/`; README reports ~3.5 h recorded, ~50,000 balanced datapoints (frame + right-joystick x-axis continuous steer, merged npy)
- Image resolution & camera view: screen-recorded frames (resolution unverified); camera view unverified
- Label columns & units: continuous joystick x-axis (regression target), units per xinput read (likely -1..1 or 0..65535) — unverified
- Speed included: no
- Download method: n/a — dataset not released
- Notes: Collection pipeline (screen_record.py + xinput.py + timestamp merge) is a decent reference for our own recorder design.

## AutoAILab/End2EndDriving — 200k GTA V frames @30fps recorded, data NOT published  [ADJACENT]  (task 5)
- URL: https://github.com/AutoAILab/End2EndDriving
- Maintainer: AutoAILab
- Date (created / last updated): ~2020 (TensorFlow 2.1 era); unverified
- License: none stated
- Size: NO DATA SHIPPED — README reports 200,000+ images recorded at 30 fps to .npy (800x600 windowed capture); no download link anywhere in README
- Image resolution & camera view: 800x600 windowed, resized 100x100 HLS for training; camera view unverified
- Label columns & units: left analogue stick (steering) + triggers (throttle) from a physical controller — continuous axes, units unverified
- Speed included: no
- Download method: n/a
- Notes: VGG-16 regression variant of the same pattern; data collected but never released.

## marsolmos/gtautodrive — 150k images collected, data NOT published  [ADJACENT]  (task 5)
- URL: https://github.com/marsolmos/gtautodrive
- Maintainer: marsolmos
- Date (created / last updated): ~2020-2021 (TF 2.3 era); unverified
- License: MIT (LICENSE file present)
- Size: NO DATA SHIPPED — README reports 150,000 images + key labels collected, 11,235 after balancing; no download link
- Label columns & units: keyboard keys (sentdex-style), not steering angle
- Speed included: no
- Download method: n/a
- Notes: Code-only; README even lists "hood camera" as a future improvement (their data is presumably chase cam).

## mrclgl/gta-v-driver — hood-cam + speed-input wheel mod project, data NOT published  [ADJACENT]  (task 5)
- URL: https://github.com/mrclgl/gta-v-driver
- Maintainer: mrclgl (a.k.a. Check2016)
- Date (created / last updated): ~2017-2018 (TensorFlow 1.3/CUDA 8 era); unverified
- License: none stated
- Size: NO DATA SHIPPED — only a trained TF checkpoint is published (mediafire link)
- Image resolution & camera view: 640x160 RGB crop; hood cam via manual-transmission mod's preconfigured "9" view — closest camera match to ours found in GTA V work
- Label columns & units: steering wheel axis + throttle/brake fed through x360ce/vJoy to the game's wheel input (manual transmission mod "Wheel" mode); units unverified but effectively a continuous wheel axis, not degrees
- Speed included: yes — current vehicle speed is a model input (read via the mod to a file), unique among the GTA V projects surveyed
- Download method: n/a
- Notes: Architecturally the closest GTA V analog to our project (image + speed -> wheel/throttle/brake, hood cam, wheel mod) — shame the training set was never released.

## aitorzip/DeepGTAV (and David0tt/DeepGTAV-PreSIL forks) — collection framework, no steering dataset  [ADJACENT]  (task 5)
- URL: https://github.com/aitorzip/DeepGTAV ; fork https://github.com/David0tt/DeepGTAV ; PreSIL https://github.com/bradenhurl/DeepGTAV-PreSIL
- Maintainer: aitorzip (Aitor Ruano); David0tt / bradenhurl (U Waterloo forks)
- Date (created / last updated): 1.2k stars, ~2017-era; forks maintained into 2021+
- License: LICENSE file present in repo (type unverified)
- Size: NO DATASET of driving-control data shipped by the framework itself; David0tt's fork links pregenerated datasets for UAV object detection (VisDrone-style: images + 2D boxes) at cloud.cs.uni-tuebingen.de — perception labels, not steering
- Label columns & units: the FRAMEWORK can stream per-frame throttle, brake, steering (float, from game memory offsets), speed, yawRate, location, time via JSON at configurable Hz with a front-center vehicle camera — i.e. it can produce exactly our record format, but the user must record it
- Speed included: supported field
- Download method: git clone; requires GTAV <= 1.0.1180.2 + ScriptHookV
- Known problems: a tool, not a dataset; memory offsets break across game versions; requires downgrading the game.
- Notes: If we ever wanted a GTA V cross-domain pretraining set in our exact format, DeepGTAV+VPilot is the mature way to generate it. PreSIL (bradenhurl) ships LiDAR/KITTI-format perception data — wrong task.

## Task 5 summary note
GTA V frame+steering landscape: the only published datasets with CONTINUOUS steering labels are (a) the 2016 DeepDrive GTAV baseline on archive.org — 600K frames / 42 h / 52 GB, forward camera, steering+throttle+yaw+speed labels generated by the in-game AI (not human, units unverified, no license) — and (b) Alzaib's 100k-frame hood-cam set on a live public Google Drive — 160x120 grayscale + pygame joystick steer/throttle in -1..1 (verified from collect_data.py). Everything else with real downloads (sartajbhuvaji ~1M frames, kfk42kfk 70k/140k) uses sentdex one-hot WASD keyboard labels — unusable for steering regression. Projects that recorded proper continuous steer (dhruv-sirohi 50k, AutoAILab 200k, mrclgl wheel-mod+speed) never released their data. Perception GTAV sets (Playing for Data / GTAV segmentation ~25k frames, PreSIL) are not driving-control data and were not pursued. DeepGTAV remains the right tool if we ever want to generate a GTA V set in our exact schema.


## marsauto/europilot — ETS2 dataset (162,495 images + wheel-axis CSV) — advertised download link DEAD  [PARTIAL]  (task 6)
- URL: https://github.com/marsauto/europilot ; dataset link in README -> https://drive.google.com/file/d/0B42sVbnSOCJ4bnZhWF80b0xUY28/view
- Maintainer: marsauto org (marsauto.com team)
- Date (created / last updated): repo created 2017-07-12, last push 2020-10-25 (verified via GitHub API); 1,512 stars. Sample CSV frames timestamped 2017-07-27
- License: MIT (repo LICENSE verified via GitHub API); dataset itself has no separate license statement
- Size (frames / hours / GB): README claims 162,495 images / 17 GB — contents unverified because the download is dead (below). Only a stub sample survives in-repo: sample/9d0c3c2b.csv (1,357 rows, fetched and read) + ONE front image
- Image resolution & camera view: user-definable screen-capture BOX (default 500x500); verified sample `front` crop = 562x341 JPEG; CSV also references `side_left`/`side_right` camera crops per row, so the format supports multi-cam like NVIDIA's paper. Camera view unverified (ETS2 cabin/hood view likely given screenshots in README gifs, but not confirmed)
- Label columns & units: verified CSV header: `id,img,wheel-axis,clutch,brake,gas,paddle-left,paddle-right,<13 button cols>,dpad-left/right,dpad-up/down,<4 shifter buttons>,gear-1..6,gear-R,front,side_left,side_right`. joystick.py (read in full) documents wheel-axis normalized to [-32767,32767] (Logitech G27 wheel, NOT degrees), pedals [0,65535]. No speed column.
- Speed included: NO — joystick/wheel values only
- Per-frame synced: yes — each CSV row maps image filenames to one joystick snapshot; capture at DEFAULT_FPS=20 (generate_training_data.py, verified)
- Download method: DEAD — the Google Drive link returns HTTP 404 "the file you have requested does not exist" (verified 2026-10-03 via curl, page body confirms). All forks (parnec, vjekoslavdiklic, marshq) point to the same dead link; no mirror found in web search
- Known problems: dataset is effectively LOST (link dead ~2020s); wheel-axis is a raw 16-bit joystick axis, not degrees; no speed channel; camera view unverified; MIT covers code only, not the (gone) data
- Notes: This WAS the canonical ETS2 PilotNet dataset and its format (filename + wheel axis + pedals + multi-cam filenames) is nearly identical to ours minus speed. The surviving sample/ CSV + joystick.py still document the exact schema if a mirror ever surfaces. Worth a Wayback/annas-archive check if the data is ever needed.

## yinhuankuang/rl-game-traces-euro-truck-simulator-2 (Hugging Face)  [PARTIAL]  (task 6)
- URL: https://huggingface.co/datasets/yinhuankuang/rl-game-traces-euro-truck-simulator-2
- Maintainer: yinhuankuang (HF user); sessions by driver 刘靖; same auto-generated "game data platform" pipeline as the FH5 xiaoluo11/opensima repos (task 2)
- Date (created / last updated): created ~2026-06-11, lastModified 2026-06-12 (verified via HF API); 1,033 downloads
- License: "other" (dataset card)
- Size (frames / hours / GB): card reports 471 files / 344.36 GB; 36 session dirs verified via HF tree API, each with video.mkv + one .parquet + 3 PNGs + fps.jsonl + macroEvents.jsonl + mouseMoveBy/ToMacroEvents.jsonl + pc.json + systemInfo.json + video-km-frames.json + videoStartTime.txt
- Image resolution & camera view: video.mkv per session (~60 fps per video-km-frames timestamps, verified head of file); resolution and camera view unverified
- Label columns & units: raw key/mouse macro events — macroEvents.jsonl rows like `{"type":3,"mouseX":-1,"mouseY":0,"time":...}` = mouse deltas (verified head of file); video-km-frames.json indexes events per video frame with ns timestamps (verified). NOT wheel angle. The per-session .parquet contents unverified (likely consolidated per-frame events/state)
- Speed included: no telemetry fields seen (unverified for parquet contents)
- Per-frame synced: yes — events indexed per video frame
- Download method: hf_hub_download, ungated (verified via API gated:false)
- Known problems: mouse/keyboard input events, not wheel degrees; player appears to steer by mouse; 344 GB of video; "other" license; anonymous uploader
- Notes: ETS2 member of the same data family as the FH5 Matrix/game-data-platform repos — useful context: this pipeline produces video+input-event traces, never wheel-angle datasets.

## dasgringuen/assettoCorsaGym (Hugging Face) — Assetto Corsa human+SAC telemetry, no images  [PARTIAL]  (task 6)
- URL: https://huggingface.co/datasets/dasgringuen/assettoCorsaGym ; code https://github.com/dasGringuen/assetto_corsa_gym ; site https://assetto-corsa-gym.github.io/
- Maintainer: Adrian Remonda et al. (dasGringuen); academic project with accompanying AssettoCorsaGym paper
- Date (created / last updated): HF repo created ~2024-06, lastModified 2024-11-13 (verified via HF API)
- License: CC-BY-4.0 (verified via HF API license tag)
- Size (frames / hours / GB): card states 64M steps total including 2.3M steps from human drivers; 15 drivers (1 pro e-sports, 4 expert, 5 casual, 5 beginner), >=5 laps each on 4 tracks (Indianapolis, Barcelona, Red Bull Ring, Monza) x 3 cars (Miata, Dallara F317, BMW Z4 GT3). Total repo bytes unverified (HF size endpoint errors on this repo; one stint .ld verified = 6.2 MB); file tree = data_sets/<track>/<session>/laps/*.ld + .ldx + eval_summary.csvs (verified via HF tree API)
- Image resolution & camera view: NO IMAGES — raw MoTeC i2 telemetry files (.ld/.ldx) plus derived eval CSVs only
- Label columns & units: verified column set from HF dataset preview: steerAngle, brakeStatus, accStatus, actualGear, RPM, speed, world_position_x/y, velocity_x/y/z, yaw, roll, angular_velocity_x/y, LapCount, packetId, currentTime + reward/done/gap (RL fields). Telemetry at 50 Hz (per card). steerAngle units unverified (likely the MoTeC channel value; AC shared memory steering angle — degrees-vs-radians unverified)
- Speed included: yes — `speed` column (units unverified, likely km/h or m/s per AC telemetry convention)
- Per-frame synced: n/a (no frames)
- Download method: hf_hub_download / git clone, ungated (verified gated:false)
- Known problems: telemetry-only (cannot pretrain a CNN); MoTeC .ld binary format needs their motec_to_pickle.py toolchain; only ~3.6% of steps are human (rest SAC-generated); steerAngle/speed units undocumented on the card
- Notes: The largest VERIFIED-downloadable human driving dataset found in any racing sim — 2.3M human steps with real wheel/pedal inputs. Value for us is analytics/validation (e.g., human steer-vs-speed distributions), not frame supervision. Repo also documents the data-collection rig (ACTI MoTeC plugin).

## briansfma/AC-Synced-Logger — Assetto Corsa frame+telemetry recorder, no data shipped  [ADJACENT]  (task 6)
- URL: https://github.com/briansfma/AC-Synced-Logger
- Maintainer: briansfma
- Date (created / last updated): unverified
- License: unverified
- Size: tool only — user records own captures into `captures/` inside the AC app
- Label columns & units: verified CSV schema from README: image file, gas 0..1, brake 0..1, clutch 0..1, gear -1..6, STEERING WHEEL ANGLE IN DEGREES (depends on user hardware), lat/long G, SPEED IN MPH, lap valid, lap times, lap proportion, performance delta/delta-rate
- Speed included: yes (MPH)
- Per-frame synced: yes — the tool's whole purpose is time-aligning AC app telemetry with video frames
- Download method: n/a
- Known problems: in-app Python runs at AC's app-refresh rate (not 30+ fps guaranteed); recording drops render framerate; no dataset published
- Notes: Notable because its CSV schema (wheel degrees + speed + frame filename) is the closest published analog to our record format in any sim — a ready-made reference if we wanted an AC capture rig.

## TRAVEL dataset (Zenodo 5911161) — BeamNG.tech test executions, telemetry JSON, no images  [PARTIAL]  (task 6)
- URL: https://zenodo.org/records/5911161 ; pipeline https://github.com/se2p/tool-competition-av
- Maintainer: Derakhshanfar, Panichella, Gambi, Riccio, Birchler, S. Panichella (TU Delft / U Passau / USI / ZHAW)
- Date (created / last updated): published 2022-01-27 (verified via Zenodo API; record modified 2024-07)
- License: CC-BY-4.0 (verified via Zenodo API license field "cc-by-4.0")
- Size (frames / hours / GB): files verified via API: competition.tar.gz 1,608,957,232 B (~1.6 GB) + sdc-prioritizer.zip + README.pdf + tool zips; thousands of test.<id>.json executions from the SBST CPS tool competition (generators: Deeper, Frenetic, AdaFrenetic, Swat) vs BeamNG.AI and DAVE2 agents; 462 downloads
- Image resolution & camera view: NO IMAGES — virtual-road definitions (road points) + execution time series only
- Label columns & units: execution_data array per test: timer, pos[3], vel[3], vel_kmh, steering, brake, throttle, is_oob, oob_percentage (schema verified on the record page). Steering = driving-agent control input; units unverified (BeamNG electrics steering input — normalized vs degrees not stated on the record)
- Speed included: yes — vel_kmh (km/h) + velocity vector
- Per-frame synced: n/a (state sampled at constant frequency, not frames)
- Download method: Zenodo HTTP download, ungated
- Known problems: agent-generated (test-generator + BeamNG.AI/DAVE2) not human driving; no camera data; JSON-per-test layout, not frame+CSV
- Notes: A second same-family Zenodo set exists — record 14599223 "Dataset for Regression Testing of Self-Driving Cars" (10,000 Frenetic test cases, executed-10000.rar 329.2 MB, 2025-01-04) — same no-image telemetry schema, not separately verified. BeamNG's own Deep Layers 2022 blog also announced a 50k-frame SEMANTIC-ANNOTATION dataset (perception, no steering labels).

## SohaibBazaz/Autonomous-Driving-using-Camera — BeamNG.tech CNN steering, no data  [ADJACENT]  (task 6)
- URL: https://github.com/SohaibBazaz/Autonomous-Driving-using-Camera
- Maintainer: SohaibBazaz
- Date (created / last updated): unverified
- License: none visible (tree verified — no LICENSE)
- Size: NO DATA — repo verified to contain only beamng_png.py capture script, 3 CNN defs + train scripts, .pth checkpoints, and small output/log.csv training logs
- Label columns & units: predicts steering angles from front + side camera images (units unverified; BeamNG agents typically normalized)
- Download method: n/a
- Known problems: code + weights only; maintainer notes models "not trustworthy"
- Notes: Confirms the BeamNG pattern: people train PilotNet-style models on self-collected BeamNG.tech captures but do not publish the frame+steer data.

## Task 6 summary note
ETS2: the only ever-published ETS2 frame+wheel dataset is europilot's (162,495 imgs + CSV, G27 wheel-axis [-32767,32767], no speed) — and its Google Drive link is verified DEAD (404 "does not exist"); no mirror found. Everything else ETS2 is collect-your-own code: boris-ns/ats-autopilot (README cites two 40k-image datasets trained on; only a 6-jpg dataset-example stub ships), manvydasu/Euro-truck-simulator2_self_driving (~40k imgs collected, none shipped), Dodecahedrane/ETS2-Self-Driving-AI (steering inferred from video of the wheel, no data), aleju/self-driving-truck (RL, no data), kdimon15/self-driving-ETS2 (no data). The 344 GB yinhuankuang HF repo has real per-frame synced ETS2 sessions but labels are mouse/key events, not wheel angle. ASSETTO CORSA: no AC frame+steer dataset exists publicly; the verified downloadable ACGym set is telemetry-only (CC-BY-4.0, 64M steps, 2.3M human, 50 Hz, steerAngle/speed/pedals, MoTeC .ld) — the best human-sim-driving corpus found in this whole survey, but unusable for CNN pretraining. AC-Synced-Logger is only a (well-designed, degrees+MPH schema) recorder. BEAMNG: only test-generation execution telemetry exists publicly (TRAVEL CC-BY-4.0 + Zenodo 14599223), no images; BeamNG's 50k set is semantic annotation. NET: zero downloadable frame+steering datasets in any of the three sims; europilot's format is the design to copy and ACGym is the best telemetry corpus.
