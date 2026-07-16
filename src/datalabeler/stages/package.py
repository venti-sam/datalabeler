"""Stage 4 - storage-efficient (image, label) pairs + reproducible splits.

Emits a COCO-layout dataset (images + one instances.json per split) and,
optionally, indexed-PNG semantic masks flattened by class priority. Splits are
assigned by bag/scene (never random) so near-duplicate video frames can't leak
across train/val/test and inflate metrics. A manifest.jsonl travels with the
dataset for human inspection and provenance.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from ..config import Config
from ..coco import (flatten_to_semantic, load_coco, merge_cocos, save_coco)
from ..manifest import CORRECTED, AUTO, Manifest


def _assign_splits(bags: list[str], ratios: dict, seed: int) -> dict[str, str]:
    """Deterministically map each bag to a split by hashing bag+seed."""
    order = sorted(bags, key=lambda b: hashlib.sha1(f"{seed}:{b}".encode()).hexdigest())
    n = len(order)
    n_train = round(n * ratios.get("train", 0.8))
    n_val = round(n * ratios.get("val", 0.1))
    out = {}
    for i, b in enumerate(order):
        out[b] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def package(cfg: Config, use_status: str = CORRECTED) -> dict[str, int]:
    ann_dir = cfg.path("annotations")
    ds = cfg.path("dataset")
    classes = cfg.classes
    pkg = cfg.package
    split_cfg = pkg.get("split", {})
    render_png = bool(pkg.get("render_semantic_png", True))

    with Manifest(cfg.path("manifest")) as mani:
        frames = list(mani.iter_frames(status=use_status))
        if not frames:
            raise SystemExit(f"no frames with status={use_status}; run earlier stages")
        splits = _assign_splits(mani.bags(),
                                split_cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1}),
                                int(split_cfg.get("seed", 42)))

        by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for fr in frames:
            sp = splits.get(fr["bag"], "train")
            mani.set_split(fr["frame_id"], sp)
            by_split[sp].append(fr)
        mani.commit()

    stats = {"train": 0, "val": 0, "test": 0}
    manifest_rows = []
    for split, group in by_split.items():
        if not group:
            continue
        img_out = ds / split / "images"
        img_out.mkdir(parents=True, exist_ok=True)
        if render_png:
            (ds / split / "masks").mkdir(parents=True, exist_ok=True)

        cocos = []
        for fr in tqdm(group, desc=f"package/{split}", unit="img"):
            ann_path = ann_dir / f"{fr['frame_id']}.coco.json"
            if not ann_path.exists():
                continue
            coco = load_coco(ann_path)
            src = Path(fr["image_path"])
            shutil.copy2(src, img_out / src.name)

            if render_png and coco["images"]:
                sem = flatten_to_semantic(coco["images"][0], coco["annotations"],
                                          cfg.priority_ids)
                cv2.imwrite(str(ds / split / "masks" / f"{fr['frame_id']}.png"), sem)

            cocos.append(coco)
            manifest_rows.append({
                "frame_id": fr["frame_id"], "split": split, "bag": fr["bag"],
                "topic": fr["topic"], "stamp_ns": fr["stamp_ns"],
                "image": f"{split}/images/{src.name}",
                "status": fr["status"], "class_dist": fr.get("class_dist"),
            })
            stats[split] += 1

        merged = merge_cocos(cocos, classes)
        save_coco(merged, ds / split / "annotations" / "instances.json")

    (ds / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest_rows) + "\n")
    (ds / "categories.json").write_text(json.dumps(
        [{"id": c.id, "name": c.name} for c in classes], indent=2))
    return stats
