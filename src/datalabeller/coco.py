"""Canonical COCO helpers (instance masks as RLE).

COCO is the currency end to end: SAM 3 emits instances, CVAT imports/exports
it natively, and it flattens cleanly to a semantic class-map at export time.
One COCO file per frame (`annotations/{frame_id}.coco.json`, single image
entry) keeps status tracking and merging trivial.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pycocotools import mask as mask_utils

from .config import ClassDef


def empty_coco(classes: Iterable[ClassDef]) -> dict[str, Any]:
    return {
        "info": {"description": "datalabeller canonical COCO"},
        "images": [],
        "annotations": [],
        "categories": [{"id": c.id, "name": c.name} for c in classes],
    }


def add_image(coco: dict, image_id: int, file_name: str, w: int, h: int) -> None:
    coco["images"].append(
        {"id": image_id, "file_name": file_name, "width": w, "height": h})


def mask_to_rle(mask: np.ndarray) -> dict:
    """Binary HxW mask -> COCO RLE with a JSON-safe (ascii) counts string."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def rle_to_mask(rle: dict) -> np.ndarray:
    r = dict(rle)
    if isinstance(r["counts"], str):
        r = {**r, "counts": r["counts"].encode("ascii")}
    return mask_utils.decode(r).astype(np.uint8)


def seg_to_mask(seg, height: int, width: int) -> np.ndarray:
    """Decode any COCO segmentation to a binary HxW mask.

    Handles both RLE dicts (what SAM 3 / we produce) and polygon lists (what
    CVAT emits when annotators edit as polygons), so ingest is uniform.
    """
    if isinstance(seg, dict):
        return rle_to_mask(seg)
    # polygon(s): list of [x,y,x,y,...]
    rles = mask_utils.frPyObjects(seg, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(np.uint8)


def seg_to_rle(seg, height: int, width: int) -> dict:
    """Normalize any COCO segmentation to a JSON-safe RLE dict."""
    if isinstance(seg, dict) and isinstance(seg.get("counts"), str):
        return seg
    return mask_to_rle(seg_to_mask(seg, height, width))


def add_instance(coco: dict, ann_id: int, image_id: int, category_id: int,
                 mask: np.ndarray, score: float | None = None) -> None:
    rle = mask_to_rle(mask)
    bbox = mask_utils.toBbox(
        {**rle, "counts": rle["counts"].encode("ascii")}).tolist()
    area = float(mask_utils.area(
        {**rle, "counts": rle["counts"].encode("ascii")}))
    ann = {
        "id": ann_id, "image_id": image_id, "category_id": category_id,
        "segmentation": rle, "bbox": bbox, "area": area, "iscrowd": 0,
    }
    if score is not None:
        ann["score"] = float(score)
    coco["annotations"].append(ann)


def save_coco(coco: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco))


def load_coco(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def merge_cocos(cocos: list[dict], classes: Iterable[ClassDef]) -> dict:
    """Merge per-frame COCOs into one, remapping image/annotation ids."""
    out = empty_coco(classes)
    next_img, next_ann = 1, 1
    for coco in cocos:
        remap: dict[int, int] = {}
        for img in coco["images"]:
            remap[img["id"]] = next_img
            out["images"].append({**img, "id": next_img})
            next_img += 1
        for ann in coco["annotations"]:
            out["annotations"].append(
                {**ann, "id": next_ann, "image_id": remap[ann["image_id"]]})
            next_ann += 1
    return out


def flatten_to_semantic(coco_image: dict, annotations: list[dict],
                        priority_ids: list[int]) -> np.ndarray:
    """Flatten instances of one image into an indexed HxW semantic mask.

    Pixel value 0 = background/void. Overlaps resolved by `priority_ids`
    (earlier wins); classes not listed are painted first (lowest priority),
    ordered by ascending score so higher-confidence instances land on top.
    """
    h, w = coco_image["height"], coco_image["width"]
    out = np.zeros((h, w), dtype=np.uint8)

    def rank(ann: dict) -> tuple:
        cid = ann["category_id"]
        pr = priority_ids.index(cid) if cid in priority_ids else len(priority_ids)
        # paint low priority first: reverse-sort by priority, then by score asc
        return (-pr, ann.get("score", 1.0))

    for ann in sorted(annotations, key=rank):
        m = seg_to_mask(ann["segmentation"], h, w).astype(bool)
        out[m] = ann["category_id"]
    return out
