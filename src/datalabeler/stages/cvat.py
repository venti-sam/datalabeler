"""Stage 3 - CVAT round-trip for human correction after SAM 3.

CVAT runs as its OWN docker-compose stack (github.com/cvat-ai/cvat); we do not
merge it into our compose. Two integration paths, both feeding the same shared
ingest so canonical annotations stay uniform (RLE, our category ids):

  * MANUAL  - `export_for_cvat` writes task folders (images + merged COCO) that
    a human uploads/exports through the CVAT web UI; `import_from_cvat` reads the
    exported COCO back.

  * AUTOMATED (cvat-sdk) - `cvat_push` creates a CVAT task over HTTP, uploads the
    images and the SAM 3 pre-annotations; `cvat_pull` exports the corrected COCO
    and ingests it. Requires `pip install .[cvat]` and a running CVAT server.

Annotators fix boundaries, delete false pipes, add missed objects, and mark
genuinely ambiguous pixels with the `void` label rather than forcing a class.
Later loops become active learning: a trained model pre-labels the next batch
instead of SAM 3, and humans correct that.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from ..config import Config
from ..coco import empty_coco, load_coco, merge_cocos, save_coco, seg_to_rle
from ..manifest import AUTO, CORRECTED, Manifest

VOID_LABEL = "void"


# --------------------------------------------------------------------------- #
# Shared ingest: any CVAT COCO export -> canonical per-frame COCO + manifest.
# --------------------------------------------------------------------------- #
def ingest_coco(cfg: Config, coco: dict, name_to_fid: dict[str, str],
                annotator: Optional[str] = None) -> dict[str, int]:
    """Split a COCO export into canonical per-frame files and sync the manifest.

    Remaps categories BY NAME (CVAT assigns its own category ids by label order,
    so copying ids verbatim would corrupt classes), normalizes every
    segmentation to RLE, and drops annotations whose label isn't a known class
    (e.g. `void`/ignore) so those pixels fall back to background.
    """
    ann_dir = cfg.path("annotations")
    ann_dir.mkdir(parents=True, exist_ok=True)
    classes = cfg.classes
    name_to_id = {c.name: c.id for c in classes}

    # map export's category_id -> our canonical id (via name)
    src_cat_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    imgs = {im["id"]: im for im in coco["images"]}
    per_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        per_img.setdefault(ann["image_id"], []).append(ann)

    stats = {"imported": 0, "dropped_unknown": 0}
    with Manifest(cfg.path("manifest")) as mani:
        for img_id, im in imgs.items():
            fname = Path(im["file_name"]).name
            fid = name_to_fid.get(fname, Path(fname).stem)
            h, w = im.get("height"), im.get("width")

            single = empty_coco(classes)
            single["images"].append({**im, "id": 1})
            dist: dict[int, int] = {}
            out_id = 1
            for ann in per_img.get(img_id, []):
                cname = src_cat_name.get(ann["category_id"])
                if cname not in name_to_id:      # void / unknown -> background
                    stats["dropped_unknown"] += 1
                    continue
                cid = name_to_id[cname]
                single["annotations"].append({
                    "id": out_id, "image_id": 1, "category_id": cid,
                    "segmentation": seg_to_rle(ann["segmentation"], h, w),
                    "bbox": ann.get("bbox"), "area": ann.get("area"),
                    "iscrowd": ann.get("iscrowd", 0),
                })
                dist[cid] = dist.get(cid, 0) + 1
                out_id += 1
            save_coco(single, ann_dir / f"{fid}.coco.json")
            mani.set_status(fid, CORRECTED, class_dist=dist, annotator=annotator)
            stats["imported"] += 1
        mani.commit()
    return stats


def _grouped_frames(cfg: Config, bag: Optional[str], batch: Optional[int]):
    with Manifest(cfg.path("manifest")) as mani:
        frames = list(mani.iter_frames(status=AUTO, bag=bag))
    tasks: dict[str, list[dict]] = {}
    if batch:
        for i, fr in enumerate(frames):
            tasks.setdefault(f"batch_{i // batch:04d}", []).append(fr)
    else:
        for fr in frames:
            tasks.setdefault(Path(fr["bag"]).stem, []).append(fr)
    return tasks


def _merged_annotations(cfg: Config, group: list[dict]) -> dict:
    ann_dir = cfg.path("annotations")
    cocos = [load_coco(ann_dir / f"{fr['frame_id']}.coco.json")
             for fr in group
             if (ann_dir / f"{fr['frame_id']}.coco.json").exists()]
    return merge_cocos(cocos, cfg.classes) if cocos else empty_coco(cfg.classes)


# --------------------------------------------------------------------------- #
# Manual path (no server needed).
# --------------------------------------------------------------------------- #
def export_for_cvat(cfg: Config, bag: Optional[str] = None,
                    batch: Optional[int] = None) -> list[Path]:
    """Build CVAT-importable task folders (images + merged COCO)."""
    cvat_root = cfg.path("cvat")
    cvat_root.mkdir(parents=True, exist_ok=True)
    tasks = _grouped_frames(cfg, bag, batch)

    created: list[Path] = []
    for name, group in tasks.items():
        tdir = cvat_root / name
        img_dir = tdir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for fr in group:
            dst = img_dir / Path(fr["image_path"]).name
            if not dst.exists():
                shutil.copy2(fr["image_path"], dst)
        save_coco(_merged_annotations(cfg, group),
                  tdir / "annotations" / "instances_default.json")
        (tdir / "manifest_map.json").write_text(json.dumps(
            {Path(fr["image_path"]).name: fr["frame_id"] for fr in group}, indent=2))
        created.append(tdir)
        print(f"[cvat] task {name}: {len(group)} frames -> {tdir}")
    return created


def import_from_cvat(cfg: Config, export_json: str | Path,
                     task_dir: Optional[str | Path] = None,
                     annotator: Optional[str] = None) -> dict[str, int]:
    """Ingest a CVAT COCO export file (manual UI export)."""
    coco = load_coco(export_json)
    name_to_fid = {}
    if task_dir:
        mp = Path(task_dir) / "manifest_map.json"
        if mp.exists():
            name_to_fid = json.loads(mp.read_text())
    return ingest_coco(cfg, coco, name_to_fid, annotator=annotator)


# --------------------------------------------------------------------------- #
# Automated path (cvat-sdk over HTTP).
# --------------------------------------------------------------------------- #
def _registry_path(cfg: Config) -> Path:
    return cfg.path("cvat") / "tasks.json"


def _load_registry(cfg: Config) -> dict:
    p = _registry_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {}


def _save_registry(cfg: Config, reg: dict) -> None:
    p = _registry_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2))


def _make_client(cfg: Config):
    from cvat_sdk import make_client  # lazy: only when using the SDK path
    c = cfg.cvat
    user = os.environ.get(c.get("username_env", "CVAT_USER"))
    pw = os.environ.get(c.get("password_env", "CVAT_PASSWORD"))
    if not user or not pw:
        raise SystemExit("set CVAT credentials in the configured env vars")
    client = make_client(c.get("host", "http://localhost:8080"),
                         credentials=(user, pw))
    if c.get("org"):
        client.organization_slug = c["org"]
    return client


def _label_spec(cfg: Config) -> list[dict]:
    labels = [{"name": cls.name} for cls in cfg.classes]
    if cfg.cvat.get("add_void_label", True):
        labels.append({"name": VOID_LABEL})
    return labels


def cvat_push(cfg: Config, bag: Optional[str] = None,
              batch: Optional[int] = None) -> dict[str, int]:
    """Create CVAT tasks, upload images + SAM 3 pre-annotations over HTTP."""
    from cvat_sdk.core.proxies.tasks import ResourceType

    tasks = _grouped_frames(cfg, bag, batch)
    reg = _load_registry(cfg)
    stats = {"tasks": 0, "frames": 0}

    with _make_client(cfg) as client:
        for name, group in tasks.items():
            if reg.get(name, {}).get("task_id"):
                print(f"[cvat] task {name} already pushed "
                      f"(id={reg[name]['task_id']}); skipping")
                continue
            spec = {"name": f"datalabeler_{name}", "labels": _label_spec(cfg)}
            if cfg.cvat.get("project_id"):
                spec["project_id"] = cfg.cvat["project_id"]
            images = [fr["image_path"] for fr in group]
            task = client.tasks.create_from_data(
                spec=spec, resource_type=ResourceType.LOCAL, resources=images)

            # upload SAM 3 pre-annotations so humans correct rather than draw
            merged = _merged_annotations(cfg, group)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
                json.dump(merged, fh)
                ann_file = fh.name
            try:
                task.import_annotations(
                    format_name="COCO 1.0", filename=ann_file,
                    conv_mask_to_poly=cfg.cvat.get("conv_mask_to_poly", True))
            finally:
                os.unlink(ann_file)

            reg[name] = {
                "task_id": task.id,
                "file_to_fid": {Path(fr["image_path"]).name: fr["frame_id"]
                                for fr in group},
            }
            _save_registry(cfg, reg)
            stats["tasks"] += 1
            stats["frames"] += len(group)
            print(f"[cvat] pushed task {name} -> id={task.id} ({len(group)} frames)")
    return stats


def cvat_pull(cfg: Config, name: Optional[str] = None,
              annotator: Optional[str] = None) -> dict[str, int]:
    """Export corrected COCO from CVAT task(s) and ingest to canonical."""
    reg = _load_registry(cfg)
    names = [name] if name else list(reg.keys())
    total = {"imported": 0, "dropped_unknown": 0}

    with _make_client(cfg) as client:
        for nm in names:
            entry = reg.get(nm)
            if not entry or not entry.get("task_id"):
                print(f"[cvat] no pushed task named {nm}; skipping")
                continue
            task = client.tasks.retrieve(entry["task_id"])
            with tempfile.TemporaryDirectory() as td:
                zpath = os.path.join(td, "export.zip")
                task.export_dataset(format_name="COCO 1.0", filename=zpath,
                                    include_images=False)
                with zipfile.ZipFile(zpath) as zf:
                    inst = next(n for n in zf.namelist()
                                if n.endswith(".json") and "instances" in n)
                    zf.extract(inst, td)
                    coco = load_coco(os.path.join(td, inst))
            stats = ingest_coco(cfg, coco, entry.get("file_to_fid", {}),
                                annotator=annotator)
            for k in total:
                total[k] += stats[k]
            print(f"[cvat] pulled task {nm}: {stats}")
    return total
