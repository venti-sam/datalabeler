# Dev / TODO

Working notes and outstanding work for the datalabeler pipeline. Stages refer to
the four in the [README](README.md): rosbag → SAM 3 → CVAT → COCO.

## Status

| Stage | State |
|-------|-------|
| Backbone (config, manifest, IDs, COCO) | ✅ implemented + unit-tested |
| 1 — rosbag extract | ✅ code complete; ⚠️ unrun on a real bag |
| 2 — SAM 3 autolabel | ✅ real API wired; ⚠️ unrun on real weights (gated HF) |
| 3 — CVAT round-trip | ✅ manual + automated (cvat-sdk); ⚠️ unrun on a live server |
| 4 — package + splits | ✅ implemented + unit-tested |

Offline test suite (`pytest -q`) is green: 3 end-to-end tests, no GPU/weights.

## TODO — verification on real inputs

- [ ] Run Stage 1 against one real rosbag (ROS 1 `.bag` and a ROS 2 `.mcap`);
      confirm `Image`/`CompressedImage` decode and the phash sampling gate.
- [ ] Run `scripts/smoke_sam3.py` in the `sam3` container with real
      `facebook/sam3` weights (needs `HF_TOKEN` + access granted). Eyeball
      `work/sam3_smoke.png`, then tune per-class `score_threshold` / prompts.
- [ ] Stand up CVAT (its own compose), run `cvat-push` → correct → `cvat-pull`
      end to end; confirm category remap, polygon→RLE, and void-drop on a real
      export.

## TODO — features / hardening

- [ ] `motion` sampling: time-sync odom to image stamps; keep a frame only after
      moving > `trans_m` / rotating > `rot_deg`. Currently falls back to interval.
- [ ] Offline Stage 1 test: synthesize a tiny rosbag fixture (via `rosbags`
      writer) so extraction gets end-to-end coverage without a real bag.
- [ ] CVAT networking: wire `network_mode: host` (or join CVAT's network)
      concretely and document the working `cvat.host` value.
- [ ] Capture LiDAR + TF alongside camera_info for the downstream projection use
      mentioned in the original design.
- [ ] Active-learning loop: swap SAM 3 for a trained model as the pre-labeler in
      Stage 2 (backend abstraction already supports this).
- [x] Rename the Python package to `datalabeler` to match the repo (done;
      package, console script, and imports are all `datalabeler`).

## TODO — nice to have

- [ ] `webdataset` layout in Stage 4 (`package.layout: webdataset`) — tar shards
      for training throughput. Currently only the `coco` layout is implemented.
- [ ] DVC/git-lfs wiring for versioning the growing dataset across rounds.
- [ ] `ultralytics` backend: verify SAM 3 support / weight name on a real install.
