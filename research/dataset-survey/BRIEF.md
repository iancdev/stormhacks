You are researching whether public datasets exist that match a specific behavior-cloning project. Be precise and skeptical; report only things you actually found with URLs, license, size, and format. Separate "exact match", "partial match", and "adjacent / fallback".

PROJECT: Lane-keeping / steering-only behavior cloning in Forza Horizon 4 (PC, Steam), PilotNet-style CNN. Input: cropped bonnet-camera road frame (HUD masked, resized to ~320 px wide, model input 200x66 RGB) plus vehicle speed (m/s). Label: physical steering-wheel angle in degrees (Thrustmaster TMX, 900-degree rotation, right positive, range about -450..+450). Human drives one car, one repeatable route, consistent weather, racing line off, bonnet cam, HUD off, automatic gearbox.

OUR RECORD FORMAT (what an ideal dataset would resemble): per-frame JPEG + CSV row with columns
frame, segment, t (s), steer_raw, steer_deg, brake (0..1), gas (0..1), wheel_age_ms, speed_mps, race_on, tele_steer (Forza Data Out s8 steer), tele_age_ms. Captured at 30 fps with dxcam. Telemetry is Forza "Data Out" UDP, FH4 324-byte "dash" layout (IsRaceOn int32 @0, Speed float m/s @256, Steer int8 @320).

FIND, IN PRIORITY ORDER:
1. Exact: public Forza Horizon 4 (or FH5 / Forza Motorsport 7/2023) datasets pairing screen frames with steering input (wheel angle or gamepad steer axis) and speed. Search GitHub, Kaggle, Hugging Face Datasets, Zenodo, IEEE DataPort, Papers With Code, Roboflow Universe, academic theses. Terms: "Forza Horizon 4 self-driving dataset", "Forza behavior cloning", "Forza Horizon imitation learning", "Forza steering angle dataset", "Forza PilotNet", "Forza Horizon end-to-end driving", "FH4 autonomous driving CNN", "Forza Data Out dataset", "Forza telemetry CSV".
2. Partial: Forza Data Out telemetry logs (CSV/parquet) without images (useful for validating packet offsets and for Tiger Data analytics demos). Also any FH4 packet layout docs that confirm Speed @256 and Steer @320.
3. Partial: frame + steering datasets from other driving games with a bonnet/hood camera and PC screen capture (GTA V e.g. DeepGTAV / GTAV-Dataset, Euro Truck Simulator 2, Assetto Corsa, BeamNG, TORCS, CARLA). Note camera view, label type (wheel angle vs normalized steer), resolution, size, license.
4. Fallback: real-world steering datasets commonly used for PilotNet (Udacity self-driving-car dataset, comma2k19, Sully Chen driving dataset). State label units so we can judge transferability.

FOR EACH HIT REPORT: URL, maintainer, date, license, size (frames/hours/GB), image resolution and camera view, label columns and units (degrees vs normalized vs gamepad axis), whether speed is included, whether it is per-frame synced, download method, and any known problems. Also note any open-source Forza Data Out parsers for FH4 in Python and whether their offsets agree with ours.

END WITH: a 5-line verdict on whether any dataset could (a) pretrain our CNN before we have our own laps, (b) validate our telemetry parser, (c) seed a Tiger Data analytics demo. If nothing matches, say so plainly.
