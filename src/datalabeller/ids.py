"""Deterministic, filesystem-safe frame IDs so re-runs are idempotent."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SLUG = re.compile(r"[^A-Za-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG.sub("_", text).strip("_")


def frame_id(bag: str | Path, topic: str, stamp_ns: int) -> str:
    """Stable ID from (bag, topic, timestamp).

    Same (bag, topic, stamp) always yields the same ID, so a repeated
    extraction skips work already recorded in the manifest instead of
    duplicating it. A short hash of the full triple guards against slug
    collisions between different bags/topics.
    """
    bag_stem = _slug(Path(bag).stem)
    topic_slug = _slug(topic)
    h = hashlib.sha1(f"{bag}|{topic}|{stamp_ns}".encode()).hexdigest()[:8]
    return f"{bag_stem}__{topic_slug}__{stamp_ns}__{h}"
