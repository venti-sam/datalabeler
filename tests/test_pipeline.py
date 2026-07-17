"""End-to-end pipeline test with NO GPU and NO SAM 3 weights.

Injects a fake concept-segmentation backend so Stage 2's plumbing (image ->
COCO RLE -> manifest) is exercised for real, then simulates a CVAT COCO export
(with CVAT-style category ids, a `void` label, and a polygon segmentation) to
prove the Stage 3 ingest and Stage 4 packaging are correct.

Run: pip install .[dev] && pytest -q
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from datalabeler.config import load_config
from datalabeler.coco import load_coco, rle_to_mask, seg_to_mask
from datalabeler.manifest import AUTO, CORRECTED, EXTRACTED, Frame, Manifest
from datalabeler.stages.autolabel import Sam3Backend, autolabel
from datalabeler.stages.cvat import ingest_coco
from datalabeler.stages.package import package
from datalabeler.stages.preview import preview

H, W = 40, 60


class FakeBackend(Sam3Backend):
    """Deterministic masks keyed by prompt, standing in for SAM 3."""

    def segment(self, bgr, prompt):
        if prompt == "grass":
            m = np.ones((H, W), np.uint8)
            return [(m, 0.95)]
        if prompt == "horizontal metal pipe":
            m = np.zeros((H, W), np.uint8); m[18:22, :] = 1
            return [(m, 0.80)]
        if prompt in ("mannequin", "person"):
            m = np.zeros((H, W), np.uint8); m[0:8, 0:8] = 1
            return [(m, 0.70)]
        return []


@pytest.fixture()
def project(tmp_path: Path):
    cfg_dict = {
        "paths": {
            "bags": [], "workdir": str(tmp_path / "work"),
            "images": "${workdir}/images", "annotations": "${workdir}/annotations",
            "dataset": "${workdir}/dataset", "cvat": "${workdir}/cvat",
            "manifest": "${workdir}/manifest.sqlite",
        },
        "extract": {"topics": ["/cam"], "sampling": {"strategy": "interval"}},
        "classes": [
            {"name": "grass", "id": 1, "prompts": ["grass"]},
            {"name": "mannequin", "id": 2, "prompts": ["mannequin", "person"]},
            {"name": "pipe", "id": 3, "prompts": ["horizontal metal pipe"]},
        ],
        "priority": ["mannequin", "pipe", "grass"],
        "autolabel": {"backend": "sam3", "score_threshold": 0.4, "max_instances": 100},
        "package": {"render_semantic_png": True,
                    "split": {"by": "bag", "ratios": {"train": 1.0, "val": 0.0, "test": 0.0}}},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    cfg = load_config(cfg_path)

    # seed two extracted frames with real image files on disk
    images = cfg.path("images"); images.mkdir(parents=True, exist_ok=True)
    frame_ids = []
    with Manifest(cfg.path("manifest")) as mani:
        for i in range(2):
            fid = f"bagA__cam__{i}"
            p = images / f"{fid}.jpg"
            cv2.imwrite(str(p), np.full((H, W, 3), 127, np.uint8))
            mani.upsert(Frame(frame_id=fid, bag="/data/bagA.bag", topic="/cam",
                              stamp_ns=i, image_path=str(p), width=W, height=H,
                              status=EXTRACTED))
            frame_ids.append(fid)
        mani.commit()
    return cfg, frame_ids


def test_stage2_autolabel_writes_valid_coco(project):
    cfg, frame_ids = project
    stats = autolabel(cfg, backend=FakeBackend())
    assert stats["labeled"] == 2
    assert stats["instances"] == 8   # (grass + pipe + mannequin*2 prompts) * 2 frames

    coco = load_coco(cfg.path("annotations") / f"{frame_ids[0]}.coco.json")
    cats = {c["id"] for c in coco["categories"]}
    assert cats == {1, 2, 3}
    # every mask decodes to the right footprint via RLE
    areas = {a["category_id"]: int(rle_to_mask(a["segmentation"]).sum())
             for a in coco["annotations"]}
    assert areas[1] == H * W          # grass full frame
    assert areas[3] == 4 * W          # pipe stripe

    with Manifest(cfg.path("manifest")) as mani:
        assert set(mani.counts_by_status()) == {AUTO}


def test_preview_renders_overlays_by_class_priority(project):
    cfg, frame_ids = project
    autolabel(cfg, backend=FakeBackend())

    stats = preview(cfg, alpha=0.5)
    assert stats == {"rendered": 2, "no_annotations": 0,
                     "missing_image": 0, "not_labeled_yet": 0}

    # the fixture config has no paths.preview, so this also covers the fallback
    out = cfg.path("workdir") / "preview_overlays" / f"{frame_ids[0]}.png"
    assert out.exists()
    img = cv2.imread(str(out))
    assert img.shape == (H, W, 3)

    # base frame is uniform 127 grey and alpha is 0.5, so an overlaid pixel is
    # exactly halfway between grey and its class colour.
    def blended(class_id: int) -> np.ndarray:
        from datalabeler.stages.preview import _palette
        return (0.5 * np.array(_palette(cfg)[class_id]) + 0.5 * 127).astype(np.uint8)

    # priority mannequin(2) > pipe(3) > grass(1), same as the packaged masks.
    # All three pixels are region interiors: boundary pixels carry the contour
    # outline, which is drawn in pure colour rather than blended.
    assert np.array_equal(img[20, 30], blended(3))   # pipe stripe wins over grass
    assert np.array_equal(img[4, 4], blended(2))     # mannequin corner wins over both
    assert np.array_equal(img[35, 35], blended(1))   # grass elsewhere


def _fake_cvat_export(frame_ids):
    """COCO as CVAT would export it: its own category ids, a void label,
    and one annotation edited into a POLYGON instead of RLE."""
    return {
        "images": [{"id": 1, "file_name": f"{frame_ids[0]}.jpg", "width": W, "height": H}],
        "categories": [  # note: ids differ from our config on purpose
            {"id": 10, "name": "grass"}, {"id": 11, "name": "pipe"},
            {"id": 12, "name": "mannequin"}, {"id": 13, "name": "void"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 12,        # mannequin as polygon
             "segmentation": [[0, 0, 8, 0, 8, 8, 0, 8]], "bbox": [0, 0, 8, 8],
             "area": 64, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 13,        # void -> must be dropped
             "segmentation": [[30, 30, 40, 30, 40, 40, 30, 40]], "iscrowd": 0},
        ],
    }


def test_stage3_ingest_remaps_by_name_and_drops_void(project):
    cfg, frame_ids = project
    autolabel(cfg, backend=FakeBackend())

    export = _fake_cvat_export(frame_ids)
    stats = ingest_coco(cfg, export, name_to_fid={f"{frame_ids[0]}.jpg": frame_ids[0]},
                        annotator="tester")
    assert stats["imported"] == 1
    assert stats["dropped_unknown"] == 1     # the void annotation

    coco = load_coco(cfg.path("annotations") / f"{frame_ids[0]}.coco.json")
    assert len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    assert ann["category_id"] == 2           # mannequin remapped to OUR id, not 12
    assert isinstance(ann["segmentation"], dict)  # polygon normalized to RLE
    assert isinstance(ann["segmentation"]["counts"], str)
    assert int(seg_to_mask(ann["segmentation"], H, W).sum()) == 64

    with Manifest(cfg.path("manifest")) as mani:
        row = next(mani.iter_frames(status=CORRECTED))
        assert row["annotator"] == "tester"


def test_stage4_package_layout_and_semantic_priority(project):
    cfg, frame_ids = project
    autolabel(cfg, backend=FakeBackend())
    # correct one frame; package the raw auto labels for the rest of coverage
    ingest_coco(cfg, _fake_cvat_export(frame_ids),
                name_to_fid={f"{frame_ids[0]}.jpg": frame_ids[0]}, annotator="t")

    stats = package(cfg, use_status="auto")   # both frames are 'auto' except one
    # frame[0] is now 'corrected', frame[1] still 'auto'
    assert stats["train"] == 1

    ds = cfg.path("dataset")
    inst = load_coco(ds / "train" / "annotations" / "instances.json")
    assert inst["images"] and inst["annotations"]
    mask_png = ds / "train" / "masks" / f"{frame_ids[1]}.png"
    assert mask_png.exists()
    sem = cv2.imread(str(mask_png), cv2.IMREAD_UNCHANGED)
    # priority mannequin(2) > pipe(3) > grass(1): corner is mannequin, stripe pipe
    assert sem[0, 0] == 2
    assert sem[20, 30] == 3
    assert sem[35, 35] == 1
    rows = (ds / "manifest.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["split"] == "train"
