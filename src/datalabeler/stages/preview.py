"""Verification overlays - canonical COCO -> semantic masks burned onto frames.

Not a pipeline stage: a read-only debug view for eyeballing what SAM 3 (or a
human in CVAT) actually produced. It renders from the per-frame COCO files
rather than from model output, so it needs no GPU or torch and runs in the
extract env -- and what you see is the artifact the next stage really consumes,
RLE decode included.

Instances are flattened to semantic by the config's class priority, exactly as
Stage 4 does it, so overlapping instances of different classes resolve the same
way here as in the packaged dataset.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..config import Config
from ..coco import flatten_to_semantic, load_coco
from ..manifest import Manifest

# Okabe-Ito: distinguishable under the common forms of colour blindness, which
# matters when the whole point is a human judging classes apart by eye.
_PALETTE_RGB = [
    (213, 94, 0),    # vermillion
    (0, 114, 178),   # blue
    (0, 158, 115),   # bluish green
    (230, 159, 0),   # orange
    (86, 180, 233),  # sky blue
    (204, 121, 167), # reddish purple
    (240, 228, 66),  # yellow
]


def _palette(cfg: Config) -> dict[int, tuple[int, int, int]]:
    """class id -> BGR. Keyed off sorted class ids so a class keeps its colour
    whether or not the other classes happen to appear in a given frame."""
    out = {}
    for i, c in enumerate(sorted(cfg.classes, key=lambda c: c.id)):
        r, g, b = _PALETTE_RGB[i % len(_PALETTE_RGB)]
        out[c.id] = (b, g, r)
    return out


def _draw_legend(img: np.ndarray, rows: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Swatch + label per class present, on a dimmed panel in the top-left.

    Skipped when it would dominate the frame: numpy would silently clip the
    panel to the image bounds, dimming the very thing we're here to look at.
    """
    if not rows:
        rows = [("no detections", (200, 200, 200))]
    pad, sw, lh = 6, 14, 18
    w = max(int(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0])
            for t, _ in rows) + sw + 3 * pad
    h = lh * len(rows) + 2 * pad
    if h > 0.4 * img.shape[0] or w > 0.6 * img.shape[1]:
        return
    panel = img[0:h, 0:w].copy()
    img[0:h, 0:w] = (0.45 * panel).astype(np.uint8)  # dim, don't blank: keep context
    for i, (text, bgr) in enumerate(rows):
        y = pad + i * lh
        cv2.rectangle(img, (pad, y + 3), (pad + sw, y + lh - 5), bgr, -1)
        cv2.putText(img, text, (pad + sw + pad, y + lh - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def render_overlay(bgr: np.ndarray, coco: dict, cfg: Config,
                   alpha: float = 0.5) -> np.ndarray:
    """One frame + its COCO -> an BGR overlay image."""
    colors = _palette(cfg)
    names = {c.id: c.name for c in cfg.classes}
    out = bgr.copy()
    if not coco["images"]:
        return out
    sem = flatten_to_semantic(coco["images"][0], coco["annotations"], cfg.priority_ids)

    counts: dict[int, int] = {}
    for ann in coco["annotations"]:
        counts[ann["category_id"]] = counts.get(ann["category_id"], 0) + 1

    for cid in sorted(np.unique(sem)):
        if cid == 0:  # void
            continue
        m = sem == cid
        color = np.array(colors.get(int(cid), (255, 255, 255)), dtype=np.float32)
        out[m] = (alpha * color + (1 - alpha) * out[m]).astype(np.uint8)
        # Outline the region too: a blend alone gets ambiguous over busy scenes.
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cont, -1, colors.get(int(cid), (255, 255, 255)), 1)

    rows = [(f"{names.get(cid, cid)} x{counts.get(cid, 0)}", colors.get(cid, (255, 255, 255)))
            for cid in sorted(counts)]
    _draw_legend(out, rows)
    return out


def preview(cfg: Config, status: str | None = None, alpha: float = 0.5,
            limit: int | None = None) -> dict[str, int]:
    ann_dir = cfg.path("annotations")
    # Fall back rather than require a paths.preview entry: configs written before
    # this existed should still work.
    raw = cfg.paths.get("preview")
    out_dir = cfg.path("preview") if raw else cfg.path("workdir") / "preview_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skips are split by cause, not lumped into one "skipped": the two mean very
    # different things -- a missing image is a broken workdir (`extract` rewrites
    # it), a missing .coco.json just means Stage 2 hasn't run for that frame yet.
    stats = {"rendered": 0, "no_annotations": 0,
             "missing_image": 0, "not_labeled_yet": 0}
    with Manifest(cfg.path("manifest")) as mani:
        frames = list(mani.iter_frames(status=status))
    if limit:
        frames = frames[:limit]

    for fr in tqdm(frames, desc="preview", unit="img"):
        ann_path = ann_dir / f"{fr['frame_id']}.coco.json"
        bgr = cv2.imread(fr["image_path"])
        if bgr is None:
            stats["missing_image"] += 1
            continue
        if not ann_path.exists():
            stats["not_labeled_yet"] += 1
            continue
        coco = load_coco(ann_path)
        if not coco["annotations"]:
            stats["no_annotations"] += 1
        out = render_overlay(bgr, coco, cfg, alpha=alpha)
        cv2.imwrite(str(out_dir / f"{fr['frame_id']}.png"), out)
        stats["rendered"] += 1
    return stats
