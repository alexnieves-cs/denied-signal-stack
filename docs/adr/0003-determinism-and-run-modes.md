# ADR-0003: determinism and run modes

Status: accepted (Phase 0).

- RNG: every stream is `SeedSequence(root, spawn_key=(DOMAIN, crc32(name)))`; `root.spawn()` is forbidden (order-dependent). Adding a stream never changes an existing one (tested).
- Tier A: bit-identical sensor streams and GT for the same seed (SHA-256 per stream, `test_determinism.py`).
- Tier B: bit-identical rendered images on the same machine and manifest.
- Tier C: estimator outputs within tolerance across platforms.
- `--lockstep`: deterministic; `metrics.json` contains no wall-clock fields. `--realtime`: writes `timing.json`.
