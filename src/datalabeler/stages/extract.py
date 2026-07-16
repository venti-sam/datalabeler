"""Stage 1 - rosbag -> sampled images + manifest rows.

Reads ROS 1 and ROS 2 bags via `rosbags` (no sourced ROS install). Samples so
we don't dump 30 fps of near-duplicates: an interval floor, then optional
perceptual-hash novelty gating. Captures camera_info intrinsics when present.
"""
from __future__ import annotations

from pathlib import Path

import cv2
from rosbags.highlevel import AnyReader
from tqdm import tqdm

from ..config import Config
from ..ids import frame_id
from ..imageio import (ahash, compressed_msg_to_bgr, hamming, image_msg_to_bgr)
from ..manifest import EXTRACTED, Frame, Manifest


def _msgtype(conn) -> str:
    # rosbags Connection exposes the message type on .msgtype
    return getattr(conn, "msgtype", "")


def _decode(msg, msgtype: str):
    if "CompressedImage" in msgtype:
        return compressed_msg_to_bgr(msg)
    if "Image" in msgtype:
        return image_msg_to_bgr(msg)
    return None


def _write_image(bgr, out_dir: Path, fid: str, cfg: Config) -> str:
    fmt = cfg.extract.get("image_format", "jpg").lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fid}.{fmt}"
    if fmt in ("jpg", "jpeg"):
        cv2.imwrite(str(path), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, int(cfg.extract.get("jpeg_quality", 92))])
    else:
        cv2.imwrite(str(path), bgr)
    return str(path)


def _ci_field(msg, name: str):
    # CameraInfo intrinsics are lowercase (k/d/p) in ROS 2 but uppercase
    # (K/D/P) in ROS 1; rosbags preserves the source casing. Accept both.
    val = getattr(msg, name.lower(), None)
    if val is None:
        val = getattr(msg, name.upper(), None)
    return [] if val is None else list(val)


def _capture_camera_info(mani: Manifest, topic: str, msg) -> None:
    if mani.has_camera_info(topic):
        return
    mani.put_camera_info(
        topic=topic, width=int(msg.width), height=int(msg.height),
        distortion_model=getattr(msg, "distortion_model", ""),
        K=_ci_field(msg, "k"), D=_ci_field(msg, "d"), P=_ci_field(msg, "p"))


def extract(cfg: Config) -> dict[str, int]:
    topics = set(cfg.extract["topics"])
    samp = cfg.extract["sampling"]
    strategy = samp.get("strategy", "interval")
    interval_ns = int(float(samp.get("interval_sec", 1.0)) * 1e9)
    phash_thr = int(samp.get("phash_threshold", 8))
    capture_ci = bool(cfg.extract.get("capture_camera_info", True))
    images_dir = cfg.path("images")

    if strategy == "motion":
        # Odometry-gated sampling needs time-synced odom lookup; not yet wired.
        # Fall back to the interval floor so the pipeline still runs.
        print("[extract] motion sampling not implemented; using interval floor")
        strategy = "interval"

    stats = {"kept": 0, "skipped_existing": 0, "seen": 0}
    bags = cfg.bag_files
    if not bags:
        raise SystemExit("no bags matched paths.bags")

    with Manifest(cfg.path("manifest")) as mani:
        for bag in bags:
            last_stamp: dict[str, int] = {}   # per topic
            last_hash: dict[str, int] = {}    # per topic
            with AnyReader([bag]) as reader:
                conns = [c for c in reader.connections if c.topic in topics]
                ci_conns = [c for c in reader.connections
                            if "CameraInfo" in _msgtype(c)] if capture_ci else []
                gen = reader.messages(connections=conns + ci_conns)
                for conn, tstamp, rawdata in tqdm(gen, desc=bag.name, unit="msg"):
                    mtype = _msgtype(conn)
                    if "CameraInfo" in mtype:
                        msg = reader.deserialize(rawdata, mtype)
                        _capture_camera_info(mani, conn.topic, msg)
                        continue
                    stats["seen"] += 1
                    topic = conn.topic
                    stamp = tstamp  # ns

                    # interval floor
                    if topic in last_stamp and stamp - last_stamp[topic] < interval_ns:
                        continue

                    msg = reader.deserialize(rawdata, mtype)
                    bgr = _decode(msg, mtype)
                    if bgr is None:
                        continue

                    reason = strategy
                    if strategy == "phash":
                        h = ahash(bgr)
                        if topic in last_hash and hamming(h, last_hash[topic]) < phash_thr:
                            continue
                        last_hash[topic] = h
                        reason = "phash-novel"
                    last_stamp[topic] = stamp

                    fid = frame_id(bag, topic, stamp)
                    if mani.has(fid):
                        stats["skipped_existing"] += 1
                        continue

                    path = _write_image(bgr, images_dir, fid, cfg)
                    hh, ww = bgr.shape[:2]
                    mani.upsert(Frame(
                        frame_id=fid, bag=str(bag), topic=topic, stamp_ns=stamp,
                        image_path=path, width=ww, height=hh,
                        sampling_reason=reason, status=EXTRACTED))
                    stats["kept"] += 1
            mani.commit()
    return stats
