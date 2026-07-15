# Datalabeller

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
  frame under `work/annotations/`.

## Environments (Docker)

Two images because the ROS-reading stack and recent PyTorch/CUDA fight in one
env. They share the project dir; the files on disk are the only handoff.

| Image | Stages | Base | Notes |
|-------|--------|------|-------|
| `extract` | 1, 3, 4 | `python:3.11-slim` + `rosbags` + `cvat-sdk` | CPU |
| `sam3` | 2 | `pytorch:2.7-cuda12.6` + `facebookresearch/sam3` | GPU, gated HF weights |

```bash
docker compose -f docker/docker-compose.yml build
```

SAM 3 weights are **gated**: request access on
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), then pass a
token — `HF_TOKEN=hf_xxx docker compose ... run --rm sam3 autolabel`. Weights
cache to `docker/checkpoints/` across runs.

## Run

```bash
cp config/pipeline.example.yaml config/pipeline.yaml   # then edit topics/classes/paths
# put bags under data/bags/ (or point paths.bags elsewhere)

# Stage 1 — extract sampled frames
docker compose -f docker/docker-compose.yml run --rm extract extract

# Stage 2 — SAM 3 pre-annotations (GPU)
docker compose -f docker/docker-compose.yml run --rm sam3 autolabel

# Stage 3 — human correction in CVAT (automated round-trip, see below)
CVAT_USER=admin CVAT_PASSWORD=... \
  docker compose -f docker/docker-compose.yml run --rm extract cvat-push
#   ... humans correct in the CVAT UI ...
CVAT_USER=admin CVAT_PASSWORD=... \
  docker compose -f docker/docker-compose.yml run --rm extract cvat-pull --annotator alice

# Stage 4 — package (image,label) pairs + splits
docker compose -f docker/docker-compose.yml run --rm extract package

# any time
docker compose -f docker/docker-compose.yml run --rm extract status
```

Without Docker: `pip install -e .` then `datalabeller <cmd>` (add `.[cvat]` for
the automated Stage 3, `.[sam3]` + the sam3 git package for Stage 2 on a CUDA host).

## Output dataset (`work/dataset/`)

```
dataset/
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

## Stage 3 — CVAT integration

CVAT ships as its **own** multi-container docker-compose stack
([github.com/cvat-ai/cvat](https://github.com/cvat-ai/cvat)) — server, UI, db,
redis, workers. We do **not** merge it into our compose; we run it independently
and talk to it over HTTP with `cvat-sdk`. Bring it up once:

```bash
git clone https://github.com/cvat-ai/cvat && cd cvat
docker compose up -d                 # serves the UI on http://localhost:8080
docker exec -it cvat_server bash -ic \
  'python3 ~/manage.py createsuperuser'   # make the account you'll put in CVAT_USER/PASSWORD
```

Then two ways to run the correction loop — both feed the **same** ingest, so
canonical annotations stay uniform (RLE, our category ids):

**Automated (recommended, `cvat-sdk`)**

| Command | What it does |
|---------|--------------|
| `datalabeller cvat-push` | Groups `status=auto` frames into tasks (by bag, or `--batch N`), creates each CVAT task, uploads the images, and uploads the SAM 3 masks as pre-annotations (`import_annotations`, COCO 1.0). Records task ids in `work/cvat/tasks.json`. |
| `datalabeller cvat-pull` | For each recorded task, `export_dataset` (COCO 1.0), then ingest → per-frame canonical COCO + `status=corrected` in the manifest. |

Config lives under `cvat:` in the pipeline YAML (host, `project_id`, whether to
import masks as editable polygons). **Credentials come from env vars**
(`CVAT_USER`/`CVAT_PASSWORD`), never the config file.

**Manual (no server scripting)** — `cvat-export` writes `work/cvat/<name>/`
(images + `instances_default.json`) that you upload/export through the CVAT UI;
`cvat-import <exported.json> --task-dir work/cvat/<name>` reads it back.

Three correctness details the ingest handles, because CVAT's export differs from
what we sent:

- **Category remap by name.** CVAT assigns its own category ids by label order;
  we remap by *name* back to the config's canonical ids.
- **Polygon ↔ RLE.** With `conv_mask_to_poly: true`, annotators edit polygons;
  the export then carries polygons, which we re-rasterize to RLE so downstream
  stays uniform.
- **Void/ignore.** A `void` label (added to each task when `add_void_label:
  true`) is *dropped* on ingest, leaving those pixels as background rather than
  forcing a class.

**Networking:** if CVAT runs via its own compose on the same host, the simplest
setup is `network_mode: host` on our `extract` service with
`cvat.host: http://localhost:8080` (commented in `docker-compose.yml`);
otherwise join CVAT's docker network and use `http://cvat_server:8080`.

**Active learning:** later rounds swap SAM 3 for your trained model as the
pre-labeler; `cvat-push`/`cvat-pull` are unchanged.

## Testing

```bash
pip install -e ".[dev]"
pytest -q                          # offline: no GPU, no SAM 3 weights
```

`tests/test_pipeline.py` injects a fake segmentation backend and a simulated
CVAT export to exercise Stage 2 → 3 → 4 end to end (COCO RLE round-trip,
name-based category remap, void drop, polygon→RLE, semantic-priority flatten,
splits). The **real** SAM 3 check runs in the GPU container:

```bash
HF_TOKEN=hf_xxx docker compose -f docker/docker-compose.yml \
  run --rm sam3 python scripts/smoke_sam3.py --image data/sample.jpg
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
