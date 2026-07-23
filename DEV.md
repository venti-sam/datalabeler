# Dev / TODO

Working notes and outstanding work for the datalabeler pipeline. Stages refer to
the four in the [README](README.md): rosbag → SAM 3 → CVAT → COCO.

## Status

| Stage | State |
|-------|-------|
| Backbone (config, manifest, IDs, COCO) | ✅ implemented + unit-tested |
| 1 — rosbag extract | ✅ verified on real **ROS 1** `.bag` **and ROS 2** `.db3`; ⚠️ ROS 2 `.mcap` / `CompressedImage` still untried |
| 2 — SAM 3 autolabel | ✅ verified end to end on real gated weights (RTX 4090) |
| 3 — CVAT round-trip | ✅ full push→correct→pull verified on live CVAT v2.70.0 (20 frames) |
| 4 — package + splits | ✅ implemented + unit-tested |

Offline test suite (`pytest -q`) is green: 4 end-to-end tests, no GPU/weights.

**The remaining open verifications** (most has now been exercised on real data):

1. **Stage 1 ROS 2 loose ends** — a real ROS 2 `.db3` bag now extracts end to end
   (`camera_lidar_20260721_093621`: raw `Image` decode, lowercase `k/d/p`
   `CameraInfo`, bag-record stamp — all confirmed). Two ROS 2 sub-cases remain
   untried: a `.mcap`-backed bag, and `CompressedImage` decode (this bag carried
   raw `Image`). Note ROS 2 `.db3` bags need a `default_typestore`
   (`extract.ros2_typestore`, default `ROS2_HUMBLE`) since they carry no embedded
   type defs.
2. **Stage 3 CVAT — done.** Full `cvat-push` → correct in the UI → `cvat-pull`
   verified on a live CVAT v2.70.0 (20 frames → `corrected`). Real gotcha the
   fixture missed: CVAT's COCO export uses *uncompressed* RLE (`counts` as an int
   list), which `rle_to_mask` now compresses before decode (regression-tested).
   Remaining Stage 4 loose end: `package` has still not run on a corrected dataset
   (now unblocked).

## Gotchas worth not rediscovering

- **SAM 3's image path needs an autocast context you supply.** Its weights are
  bfloat16 and, unlike the video predictor, `Sam3Processor` has no `@torch.autocast`
  of its own — without one every call dies on a dtype mismatch in vitdet. `scores`
  also come back bfloat16, which numpy cannot convert; `.float()` first.
- **`Sam3Processor(confidence_threshold=0.5)` filters before you see anything**, so
  `autolabel.score_threshold` must be passed *into* it, not just applied after.
  Real pipe detections here score 0.42–0.46 — the default would drop them all.
- **One concept per prompt.** A conjunction ("horizontal and vertical metal pipe")
  matches nothing. Use multiple list entries; nothing dedupes across them.
- **sam3 runs fine on Python 3.11** despite its README saying 3.12+ (its pyproject
  only requires >=3.8). It does pin `numpy<2`, which is why `[sam3]` holds opencv
  below 5.0 — opencv 5 wants numpy>=2 and you'd get an ABI mismatch.
- **Both Dockerfiles install editable (`pip install -e`).** Compose mounts the repo
  over `/app`, so a plain install pins the CLI to the build-time copy and silently
  ignores your edits — while `scripts/*.py` (which `sys.path.insert` the source) do
  pick them up. Same code, two different behaviours.
- **`labeled: 0` from `autolabel` is usually not a failure.** It only picks up
  `status=extracted`; once frames are `auto` it correctly does nothing. The CLI now
  says so. `--reannotate` redoes them.

## If the workdir looks broken

Symptom: `preview`/`package` report `missing_image`, or cv2 logs
`can't open/read file` for every frame, while `status` still lists them.

The manifest and the files on disk have diverged — a manifest row is a claim that
a frame *was* extracted, not that its bytes still exist. Re-run `datalabeler
extract`: it checks each image is really there and reports `repaired_missing`
for the ones it rewrites, leaving each row's status untouched. (Before that check
existed, extract skipped on the row alone and the pipeline could not be unstuck
short of deleting the manifest.)

If the per-frame COCO is what's missing (`not_labeled_yet` from `preview`), that
is Stage 2's output: `datalabeler autolabel --reannotate`.

## docker/.env — machine-local settings

Compose auto-loads `docker/.env` (it sits beside the compose file), so it reaches
both `./join.sh sam3` and hand-run `docker compose -f docker/... run`. Gitignored,
chmod 600, template in `docker/.env.example`. It holds:

| var | why |
|-----|-----|
| `DL_UID` / `DL_GID` | container user, so output in `work/` is deletable without sudo |
| `HF_TOKEN` | gated `facebook/sam3` weights for Stage 2 |
| `DL_BAGS_DIR` | read bags from outside the repo without copying them in |
| `CVAT_USER` / `CVAT_PASSWORD` | Stage 3 round-trip |

`docker/_env.sh` (run by `build.sh`/`start.sh`) keeps `DL_UID`/`DL_GID` current and
**preserves every other line** — it must never clobber the file, since the secrets
live there. It also pre-creates bind sources (`data/bags`, `work`, `checkpoints`):
docker creates a missing bind source as root, which would reintroduce the very
root-owned dirs the uid pinning exists to prevent.

Services also set `HOME=/tmp`: the pinned uid has no passwd entry of its own in the
image, so its default home (`/`) is unwritable and torch/triton/HF caches would
fail. The read-only `/etc/passwd` + `/etc/group` mounts give that uid a name —
without them the shell greets you as `I have no name!` and `groups` errors.

Bags must be visible **inside** the container. A symlink under `data/bags` cannot
work: it resolves to a host path that isn't mounted, so it dangles. Use
`DL_BAGS_DIR`, a hard link, or a copy.

Files written by an older root container stay root-owned; hand them back with:

```bash
docker compose -f docker/docker-compose.yml run --rm --user root \
    --entrypoint chown extract -R "$(id -u):$(id -g)" /app/work
```

## TODO — verification on real inputs

- [x] **Stage 1 on a ROS 2 `.db3` bag** (`camera_lidar_20260721_093621`): raw
      `Image` decode, lowercase `k/d/p` `CameraInfo`, bag-record stamp all confirmed.
      Needed two fixes: `bag_files` now discovers ROS 2 bag *directories* (not just
      `.bag` globs), and `AnyReader` gets a `default_typestore` (ROS 2 `.db3` bags
      carry no embedded type defs). Static scene → phash kept 1 of 590 (correct).
- [ ] **Stage 1 ROS 2 loose ends**: a `.mcap`-backed bag, and `CompressedImage`
      decode (the `.db3` above carried raw `Image`). Candidate `.mcap` on this box:
      `htc_vive_pro2_socket/rosbags/teleop/` — point `DL_BAGS_DIR` at it.
- [ ] **Stage 3 CVAT, against a live server** — the #2 open item. Stand up CVAT (its
      own compose), then `cvat-push` → correct in the UI → `cvat-pull`. Confirm on a
      *real* export what the fixture only simulates: name-based category remap,
      polygon→RLE, void-drop — plus the untested-by-anything bits: auth via
      `CVAT_USER`/`CVAT_PASSWORD`, `project_id`/label setup, and whether the
      container can reach the server at all (see the networking item below).
- [x] Stage 1 on a real ROS 1 `.bag` (`htx_gemini_pipesmall.bag`: 12 frames kept
      of 481 msgs).
- [x] `scripts/smoke_sam3.py` in the `sam3` container on real `facebook/sam3`
      weights, then `autolabel` over all 12 frames (~2.9 img/s on an RTX 4090).
      Masks verified by eye via `datalabeler preview`.
- [ ] Tune the remaining prompts: `grass` matches nothing on this bag, and no bag
      with a mannequin has been through Stage 2 yet, so only `pipe` is really
      exercised. Per-class `score_threshold` is still global (see below).

## TODO — features / hardening

- [ ] `motion` sampling: time-sync odom to image stamps; keep a frame only after
      moving > `trans_m` / rotating > `rot_deg`. Currently falls back to interval.
- [ ] Offline Stage 1 test: synthesize a tiny rosbag fixture (via `rosbags`
      writer) so extraction gets end-to-end coverage without a real bag.
- [x] CVAT networking: `extract` uses `network_mode: host`, config `cvat.host` is
      `http://localhost:8080`. `docker/cvat.sh` clones + runs the CVAT server.
      **Reachability + SDK auth from the extract container verified** (CVAT
      v2.70.0). Gotcha: CVAT's traefik only routes `/api/` for the localhost Host
      header — `host.docker.internal` 404s, which is why host networking (Host =
      localhost) rather than an `extra_hosts` mapping.
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
