#!/usr/bin/env python
"""Real SAM 3 smoke test - run inside the sam3 Docker image (GPU + HF access).

Loads the actual facebookresearch/sam3 model via the pipeline's Sam3RepoBackend,
runs every class prompt from the config on one image, prints per-prompt
detections, and writes an overlay so you can eyeball the masks. This is the
"is the SAM 3 integration actually working" check the offline pytest can't do.

    python scripts/smoke_sam3.py --image sample.jpg
    python scripts/smoke_sam3.py --image sample.jpg --prompt "horizontal metal pipe"

Exits non-zero if the model fails to load or run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datalabeler.config import load_config  # noqa: E402
from datalabeler.stages.autolabel import Sam3RepoBackend  # noqa: E402

_COLORS = [(0, 200, 0), (0, 0, 220), (220, 120, 0), (200, 0, 200), (0, 200, 200)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to a test image")
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--prompt", default=None, help="single prompt (default: all config prompts)")
    ap.add_argument("--out", default="work/sam3_smoke.png")
    ap.add_argument("--min-score", type=float, default=0.4)
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"cannot read image: {args.image}", file=sys.stderr)
        return 2

    # backend config comes from the pipeline config when present, else defaults
    version, checkpoint, device = "sam3", None, "cuda"
    prompts: list[str] = []
    if Path(args.config).exists():
        cfg = load_config(args.config)
        a = cfg.autolabel
        version = a.get("version", "sam3")
        checkpoint = a.get("checkpoint")
        device = a.get("device", "cuda")
        prompts = [p for c in cfg.classes for p in c.prompts]
    if args.prompt:
        prompts = [args.prompt]
    if not prompts:
        prompts = ["person"]

    print(f"[smoke] building SAM 3 ({version}, device={device}) ...")
    backend = Sam3RepoBackend(checkpoint, device, version=version,
                              score_threshold=args.min_score)
    print("[smoke] model ready")

    overlay = bgr.copy()
    total = 0
    for i, prompt in enumerate(prompts):
        dets = backend.segment(bgr, prompt)
        kept = [(m, s) for m, s in dets if s >= args.min_score]
        print(f"[smoke] prompt {prompt!r:32s} -> {len(dets)} masks, "
              f"{len(kept)} >= {args.min_score} "
              f"(scores: {', '.join(f'{s:.2f}' for _, s in dets[:5])})")
        color = _COLORS[i % len(_COLORS)]
        for m, _ in kept:
            overlay[m.astype(bool)] = (0.5 * np.array(color)
                                       + 0.5 * overlay[m.astype(bool)]).astype(np.uint8)
        total += len(kept)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, overlay)
    print(f"[smoke] wrote overlay -> {args.out} ({total} masks total)")
    if total == 0:
        print("[smoke] WARNING: model ran but found nothing above threshold; "
              "try a clearer image/prompt or lower --min-score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
