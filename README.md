# Datalabeler

A decoupled auto-labeling pipeline: **rosbag → SAM 3 → CVAT → COCO**. Four
stages that hand off through files, tied together by one config and one
manifest, so each can run in its own environment and be re-run independently.

```
 Stage 1            Stage 2              Stage 3               Stage 4
 rosbag ─▶ images ─▶ SAM 3 pre-labels ─▶ CVAT correction ─▶ (image,label) pairs
   │         │            │                    │                    │
   └─────────┴────────────┴───── manifest.sqlite (source of truth) ─┘
                         canonical format: COCO (RLE masks)
```

## Status

| Stage | Verified on real data? |
|-------|------------------------|
| 1 — rosbag extract | ✅ **ROS 1** `.bag` **and ROS 2** `.db3` — ⚠️ ROS 2 `.mcap` / `CompressedImage` not yet checked |
| 2 — SAM 3 autolabel | ✅ end to end on the real gated weights (RTX 4090) |
| 3 — CVAT round-trip | ✅ full push→correct→pull verified on a live server (CVAT v2.70.0, 20 frames) |
| 4 — package + splits | ✅ unit-tested (not yet run on a corrected dataset) |

Stages 1–3 have now run end to end on real data. Remaining gaps: ROS 2
`.mcap`/`CompressedImage` extraction (only `.db3` + raw `Image` checked), and
Stage 4 packaging on a real corrected dataset (now unblocked — 20 frames are
`corrected`). See [DEV.md](DEV.md) for specifics.

## Design backbone

- **One config** (`config/pipeline.yaml`) spanning all stages — topics,
  sampling, class→prompt map, thresholds, paths.
- **One manifest** (`work/manifest.sqlite`) — one row per frame with a
  deterministic ID, source bag/topic/timestamp, sampling reason, label status
  (`extracted`/`auto`/`corrected`), split, class distribution, provenance.
  Every stage reads and updates it; this is what makes the pipeline resumable,
  queryable, and idempotent.
- **Deterministic frame IDs** (`{bag}__{topic}__{stamp}__{hash}`) — re-running
  skips work already recorded instead of duplicating it.
- **COCO everywhere** — SAM 3 emits instances, CVAT imports/exports COCO
  natively, and it flattens to a semantic class-map at export. One COCO file per
  frame under `work/coco_annotations/`.

## Environments (Docker)

Two images because the ROS-reading stack and recent PyTorch/CUDA fight in one
env. They share the project dir; the files on disk are the only handoff.

| Image | Stages | Base | Notes |
|-------|--------|------|-------|
| `extract` | 1, 3, 4 | `python:3.11-slim` + `rosbags` + `cvat-sdk` | CPU |
| `sam3` | 2 | `pytorch:2.7-cuda12.8` + `facebookresearch/sam3` | GPU, gated HF weights |

```bash
docker compose -f docker/docker-compose.yml build
```

(The `sam3` image is on the CUDA 12.8 base so it runs on Blackwell GPUs such as
the RTX 5080 / sm_120; the 12.6 wheels only ship kernels up to sm_90.)

SAM 3 weights are **gated**: request access on
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), mint a token,
and put it in `docker/.env` (start from `docker/.env.example`):

```bash
cp docker/.env.example docker/.env    # then set HF_TOKEN=hf_...
```

Compose reads that file automatically — it sits next to the compose file — so the
token reaches Stage 2 whether you use `./join.sh sam3` or a hand-run
`docker compose ... run --rm sam3 autolabel`, with nothing to export each session.
`.env` is gitignored and kept chmod 600; **it is the only place a secret belongs** —
not the compose file, not the config YAML. Weights cache to `docker/checkpoints/`
across runs, so the token is only needed for the first download of a version.

### Terminal workflow (`docker/*.sh`)

To work inside an image interactively instead of one-shot `run` commands, use the
helper scripts in `docker/`. Each takes a service (`extract` or `sam3`), or shows
a menu with no argument:

```bash
cd docker
./build.sh sam3      # build one image (no arg = build all)
./start.sh sam3      # start a persistent dev container (datalabeler-<service>)
./join.sh sam3       # open a shell in it — then run `datalabeler <cmd>` by hand
./stop.sh sam3       # stop + remove the dev container
```

`start.sh` keeps the container alive (the services are batch jobs whose entrypoint
would otherwise exit); `join.sh` auto-starts it if it isn't running.

## Run

Each stage runs **inside** one of the two containers. The container hop is
**`extract` (Stage 1) → `sam3` (Stage 2) → `extract` (Stages 3 & 4)**, and
`work/manifest.sqlite` carries state across every hop — the two images never talk
directly, so you can quit one container and pick up the next stage in the other.
Statuses advance `extracted → auto → corrected` as you go, and every stage is
resumable: re-running skips frames already recorded.

**Setup (once, on the host):**

```bash
cp config/pipeline.example.yaml config/pipeline.yaml   # edit topics / classes / prompts / paths
cp docker/.env.example docker/.env                     # set HF_TOKEN (+ CVAT_USER/PASSWORD, DL_BAGS_DIR if needed)
# put bags under data/bags/ — or set DL_BAGS_DIR in docker/.env to a directory
# elsewhere (a symlink into data/bags will NOT work: it resolves to a host path
# the container hasn't mounted)
cd docker && ./build.sh                                # no arg = build both images
```

**Stage 1 — extract frames** (`extract` container):

```bash
./join.sh extract
datalabeler extract      # bags → sampled frames in work/extracted_rosbag_images/, rows @ status=extracted
datalabeler status       # sanity-check the counts, any time
```

**Stage 2 — SAM 3 auto-label** (`sam3` container, GPU):

```bash
exit; ./join.sh sam3
datalabeler autolabel    # per-class prompts → per-frame COCO in work/coco_annotations/, rows → status=auto
```

First run downloads the gated weights (via `HF_TOKEN`) into `docker/checkpoints/`;
later runs reuse the cache. `autolabel` only touches `status=extracted` frames, so
a re-run reports `labeled: 0` — that's correct, not a failure; use `--reannotate`
to redo them.

**Eyeball the labels** (`extract` container — no GPU needed):

```bash
exit; ./join.sh extract
datalabeler preview --status auto    # overlays → work/preview_overlays/ (see below)
```

**Stage 3 — human correction in CVAT** (`extract` container):

One-time — start the CVAT server (its own stack: `docker/cvat.sh` clones + runs
cvat-ai/cvat; first `up` pulls several GB) and create the login you put in
`docker/.env` as `CVAT_USER`/`CVAT_PASSWORD`:

```bash
cd docker && ./cvat.sh up && ./cvat.sh superuser    # UI at http://localhost:8080
```

Then, in the `extract` container:

```bash
datalabeler cvat-push                     # status=auto frames → CVAT tasks + images + SAM masks as pre-annotations

# Now correct in the browser: open http://localhost:8080/tasks/1 (login admin/admin),
#   open the job, and go frame by frame fixing SAM 3's masks:
#     - drag polygon points to tighten boundaries
#     - delete false masks
#     - draw any object SAM 3 missed
#     - paint ambiguous pixels with the "void" label (dropped to background on ingest)
#   Save as you go (Ctrl+S).

datalabeler cvat-pull --annotator alice   # corrected COCO back → status=corrected
```

**Stage 4 — package** (`extract` container):

```bash
datalabeler package                       # status=corrected frames → work/packaged_dataset/ (train/val/test by bag)
```

For a dry run before any correction, `datalabeler package --use-status auto`
packages the raw SAM 3 output instead.

### Eyeballing the labels (`datalabeler preview`)

Writes `work/preview_overlays/<frame_id>.png` — each frame with its masks burned
on, coloured by class with a legend of what was found. It renders from the
canonical per-frame COCO rather than from model output, so it needs no GPU and
runs in the **extract** container, and what you see is what the next stage
actually consumes (RLE decode included). Overlapping instances are flattened by
the same `priority` the packaged masks use.

```bash
datalabeler preview                       # every frame in the manifest
datalabeler preview --status auto         # just SAM 3's output (or: corrected)
datalabeler preview --limit 20 --alpha 0.6
```

Prefer one-shot runs without a persistent shell? Each command also works as
`docker compose -f docker/docker-compose.yml run --rm <extract|sam3> <cmd>`
(e.g. `... run --rm extract extract`). Or fully outside Docker: `pip install -e .`
then `datalabeler <cmd>` (add `.[cvat]` for Stage 3, `.[sam3]` + the sam3 git
package for Stage 2 on a CUDA host).

## Shutting down

This runs a lot of containers — our two dev shells plus CVAT's ~17-container
stack. To stop everything, from `docker/`:

```bash
./stop.sh extract        # stop + remove the extract dev container
./stop.sh sam3           # stop + remove the sam3 dev container
./cvat.sh down           # stop CVAT's whole stack
```

Nothing above deletes data. `./cvat.sh down` keeps CVAT's projects/tasks/
annotations in Docker volumes (so `./cvat.sh up` brings them back), and your
pipeline outputs stay in `work/` regardless.

**Reclaiming disk (optional).** The images are large (the sam3 CUDA image + CVAT's
many images). See what's using space with `docker system df`. To remove CVAT
entirely, including its annotation volumes:

```bash
cd "$(git rev-parse --show-toplevel)/../cvat"   # or wherever CVAT_DIR points
docker compose down --rmi all -v                # -v ALSO deletes annotations — export first
```

## Output dataset (`work/packaged_dataset/`)

```
packaged_dataset/
├── train|val|test/
│   ├── images/              JPEG frames
│   ├── masks/               indexed-PNG semantic masks (0=void), optional
│   └── annotations/instances.json   merged COCO (RLE instances)
├── categories.json
└── manifest.jsonl           per-sample provenance
```

Splits are assigned **by bag** (`package.split.by`), never randomly — random
splits leak near-duplicate video frames across train/val/test and inflate
metrics.

## Testing

```bash
pip install -e ".[dev]"
pytest -q                          # offline: no GPU, no SAM 3 weights
```

`tests/test_pipeline.py` injects a fake segmentation backend and a simulated
CVAT export to exercise Stage 2 → 3 → 4 end to end (COCO RLE round-trip,
name-based category remap, void drop, polygon→RLE, semantic-priority flatten,
splits). The **real** SAM 3 check runs inside the GPU container (`./join.sh sam3`;
the gated weights download using the `HF_TOKEN` from `docker/.env`):

```bash
python scripts/smoke_sam3.py --image data/sample.jpg
```

It loads the actual model, runs every config prompt, and writes an overlay to
`work/sam3_smoke.png`.

## Sampling

Stage 1 doesn't dump every frame (a 30 fps bag is mostly near-duplicates).
Strategies (`extract.sampling.strategy`): `interval` (time floor) or `phash`
(interval floor **then** perceptual-hash novelty gate). `motion` (odometry-gated)
is stubbed and falls back to interval.

## Notes / TODO

- Stage 2 supports two backends: `sam3` (facebookresearch/sam3, the default —
  `build_sam3_image_model` → `Sam3Processor` → text prompt) and `ultralytics`.
  Verified offline with a fake backend; run `scripts/smoke_sam3.py` for the real
  model check (needs GPU + HF access).
- `motion` sampling needs odom time-sync (not yet implemented; falls back to
  interval).
- Not yet exercised on real inputs here: an actual rosbag through Stage 1 and
  the real SAM 3 weights (gated download). The code paths are written and unit-
  covered but unrun on real data.
