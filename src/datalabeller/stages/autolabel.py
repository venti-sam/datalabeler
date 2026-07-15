"""Stage 2 - SAM 3 promptable concept segmentation -> COCO pre-annotations.

Runs as a batch GPU job in its own env (see docker/Dockerfile.sam3). Reads
Stage 1 images + manifest, prompts SAM 3 with the config's per-class noun
phrases, and writes one canonical COCO file per frame for humans to correct.

The model backend is abstracted so the rest of the pipeline never imports
torch. Two backends: `ultralytics` (pip install ultralytics) and `sam3`
(facebookresearch/sam3). Only the chosen one is imported, lazily.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..config import Config
from ..coco import add_image, add_instance, empty_coco, save_coco
from ..manifest import AUTO, EXTRACTED, Manifest


class Sam3Backend:
    """Abstract concept-segmentation backend."""

    def segment(self, bgr: np.ndarray, prompt: str) -> list[tuple[np.ndarray, float]]:
        """Return [(binary HxW mask, score), ...] for one text prompt."""
        raise NotImplementedError


class UltralyticsBackend(Sam3Backend):
    def __init__(self, checkpoint: str, device: str):
        from ultralytics import SAM  # lazy: only in the sam3 env
        self.model = SAM(checkpoint)
        self.device = device

    def segment(self, bgr, prompt):
        res = self.model(bgr, prompts=prompt, device=self.device, verbose=False)
        out = []
        for r in res:
            if r.masks is None:
                continue
            confs = (r.boxes.conf.cpu().numpy()
                     if r.boxes is not None and r.boxes.conf is not None
                     else np.ones(len(r.masks.data)))
            for m, s in zip(r.masks.data.cpu().numpy(), confs):
                out.append(((m > 0.5).astype(np.uint8), float(s)))
        return out


class Sam3RepoBackend(Sam3Backend):
    """facebookresearch/sam3 Promptable Concept Segmentation.

    Uses the official image API: build_sam3_image_model -> Sam3Processor ->
    set_image(PIL) -> set_text_prompt(prompt, state). The returned state carries
    masks (N,1,H,W) binary, boxes (N,4) xyxy, scores (N,) as torch tensors.

    Checkpoint resolution:
      * a local file path in `checkpoint` -> load it directly
      * else `version` ("sam3" | "sam3.1") -> download from the gated HF repo
        (requires `hf auth login` with access granted on facebook/sam3[.1]).
    """

    def __init__(self, checkpoint: str | None, device: str, version: str = "sam3"):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        local = checkpoint if (checkpoint and os.path.exists(checkpoint)) else None
        if local is None and version and version != "sam3":
            # non-default HF repo (e.g. sam3.1): fetch explicitly, then load local
            from sam3.model_builder import download_ckpt_from_hf  # type: ignore
            local = download_ckpt_from_hf(version=version)

        model = build_sam3_image_model(
            device=device,
            checkpoint_path=local,
            load_from_HF=local is None,
        )
        self.processor = Sam3Processor(model)
        self.device = device

    def segment(self, bgr, prompt):
        from PIL import Image

        rgb = np.ascontiguousarray(bgr[:, :, ::-1])  # BGR -> RGB for PIL
        state = self.processor.set_image(Image.fromarray(rgb))
        out = self.processor.set_text_prompt(prompt=prompt, state=state)
        masks, scores = out.get("masks"), out.get("scores")
        if masks is None or masks.shape[0] == 0:
            return []
        masks = masks.squeeze(1).detach().cpu().numpy().astype(np.uint8)  # (N,H,W)
        scores = scores.detach().cpu().numpy().reshape(-1)
        return [(masks[i], float(scores[i])) for i in range(masks.shape[0])]


def _make_backend(cfg: Config) -> Sam3Backend:
    a = cfg.autolabel
    backend = a.get("backend", "ultralytics")
    if backend == "ultralytics":
        return UltralyticsBackend(a["checkpoint"], a.get("device", "cuda"))
    if backend == "sam3":
        return Sam3RepoBackend(a.get("checkpoint"), a.get("device", "cuda"),
                               version=a.get("version", "sam3"))
    raise ValueError(f"unknown autolabel backend: {backend}")


def autolabel(cfg: Config, reannotate: bool = False,
              backend: Sam3Backend | None = None) -> dict[str, int]:
    ann_dir = cfg.path("annotations")
    ann_dir.mkdir(parents=True, exist_ok=True)
    classes = cfg.classes
    score_thr = float(cfg.autolabel.get("score_threshold", 0.4))
    max_inst = int(cfg.autolabel.get("max_instances", 100))

    # `backend` is injectable so tests can exercise the full pipeline without a
    # GPU or the gated SAM 3 weights.
    if backend is None:
        backend = _make_backend(cfg)
    stats = {"labeled": 0, "skipped": 0, "instances": 0}

    with Manifest(cfg.path("manifest")) as mani:
        # By default only label freshly extracted frames; --reannotate redoes them.
        target = None if reannotate else EXTRACTED
        frames = list(mani.iter_frames(status=target))
        for fr in tqdm(frames, desc="autolabel", unit="img"):
            out_path = ann_dir / f"{fr['frame_id']}.coco.json"
            if out_path.exists() and not reannotate:
                stats["skipped"] += 1
                continue
            bgr = cv2.imread(fr["image_path"])
            if bgr is None:
                stats["skipped"] += 1
                continue

            coco = empty_coco(classes)
            add_image(coco, 1, Path(fr["image_path"]).name, fr["width"], fr["height"])
            ann_id = 1
            dist: dict[int, int] = {}
            for cls in classes:
                for prompt in cls.prompts:
                    for mask, score in backend.segment(bgr, prompt):
                        if score < score_thr:
                            continue
                        if ann_id > max_inst:
                            break
                        add_instance(coco, ann_id, 1, cls.id, mask, score)
                        dist[cls.id] = dist.get(cls.id, 0) + 1
                        ann_id += 1
                        stats["instances"] += 1
            save_coco(coco, out_path)
            mani.set_status(fr["frame_id"], AUTO, class_dist=dist)
            stats["labeled"] += 1
        mani.commit()
    return stats
