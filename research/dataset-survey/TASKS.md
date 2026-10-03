# Dataset survey tasks

One task per loop iteration. Do ONLY the first unchecked task, then check it off.
Tasks 1-8 append entries to FINDINGS.md. Task 9 writes REPORT.md.

- [x] 1. EXACT / GitHub: search GitHub for Forza Horizon 4/5 and Forza Motorsport behavior-cloning, self-driving, PilotNet, imitation-learning repos. For each repo, check whether it actually publishes frames + steering data (not just training code). Record repos with data under "Exact"; repos with code-only under "Adjacent" with a one-line note.
- [x] 2. EXACT / dataset hubs: search Kaggle, Hugging Face Datasets, Zenodo, IEEE DataPort, Papers With Code, Roboflow Universe for Forza frame + steering/speed datasets.
- [x] 3. EXACT / academic: search Google Scholar, arXiv, Semantic Scholar, university thesis repositories for Forza Horizon end-to-end driving / CNN steering papers and theses; check whether they released data.
- [x] 4. PARTIAL / telemetry: find Forza Data Out telemetry logs (CSV/parquet, no images) and FH4 packet-layout documentation. Find open-source Python FH4 Data Out parsers and state explicitly whether each one's offsets agree with ours (IsRaceOn int32 @0, Speed float @256, Steer int8 @320, 324-byte "dash" packet).
- [x] 5. PARTIAL / GTA V: DeepGTAV, GTAV-Dataset, and any other GTA V frame + steering datasets with hood/bonnet camera.
- [x] 6. PARTIAL / ETS2, Assetto Corsa, BeamNG: frame + steering datasets from these sims.
- [x] 7. PARTIAL / TORCS, CARLA: frame + steering datasets (note CARLA labels are normalized -1..1).
- [ ] 8. FALLBACK / real-world: Udacity self-driving-car dataset, comma2k19, Sully Chen driving dataset. State label units precisely.
- [ ] 9. SYNTHESIZE: read all of FINDINGS.md and write REPORT.md with sections "Exact match", "Partial match", "Adjacent / fallback", "Forza Data Out parsers and offset agreement", and the 5-line verdict on (a) pretraining, (b) telemetry parser validation, (c) Tiger Data analytics demo seed. If nothing matches a category, say so plainly.
