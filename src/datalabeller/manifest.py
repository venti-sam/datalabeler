"""SQLite manifest: the pipeline's source of truth, one row per kept frame.

Every stage reads and updates it, which is what makes the pipeline resumable,
queryable and idempotent. Export to JSONL at packaging time for humans.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

# label status lifecycle
EXTRACTED = "extracted"  # Stage 1 wrote the image
AUTO = "auto"            # Stage 2 wrote SAM 3 pre-annotations
CORRECTED = "corrected"  # Stage 3 human correction synced back

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id     TEXT PRIMARY KEY,
    bag          TEXT NOT NULL,
    topic        TEXT NOT NULL,
    stamp_ns     INTEGER NOT NULL,
    image_path   TEXT NOT NULL,
    width        INTEGER,
    height       INTEGER,
    sampling_reason TEXT,
    status       TEXT NOT NULL,
    split        TEXT,
    class_dist   TEXT,          -- json {category_id: pixel_or_instance_count}
    annotator    TEXT,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frames_bag    ON frames(bag);
CREATE INDEX IF NOT EXISTS idx_frames_status ON frames(status);

CREATE TABLE IF NOT EXISTS camera_info (
    topic         TEXT PRIMARY KEY,
    width         INTEGER,
    height        INTEGER,
    distortion_model TEXT,
    K             TEXT,   -- json 9
    D             TEXT,   -- json
    P             TEXT    -- json 12
);
"""


@dataclass
class Frame:
    frame_id: str
    bag: str
    topic: str
    stamp_ns: int
    image_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    sampling_reason: Optional[str] = None
    status: str = EXTRACTED
    split: Optional[str] = None
    class_dist: Optional[dict] = None
    annotator: Optional[str] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- frames ------------------------------------------------------------
    def has(self, frame_id: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM frames WHERE frame_id=?", (frame_id,))
        return cur.fetchone() is not None

    def upsert(self, f: Frame) -> None:
        self.conn.execute(
            """INSERT INTO frames
               (frame_id,bag,topic,stamp_ns,image_path,width,height,
                sampling_reason,status,split,class_dist,annotator,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(frame_id) DO UPDATE SET
                 image_path=excluded.image_path,
                 width=excluded.width, height=excluded.height,
                 sampling_reason=excluded.sampling_reason,
                 status=excluded.status, updated_at=excluded.updated_at""",
            (f.frame_id, f.bag, f.topic, f.stamp_ns, f.image_path, f.width,
             f.height, f.sampling_reason, f.status, f.split,
             json.dumps(f.class_dist) if f.class_dist else None,
             f.annotator, _now()),
        )

    def set_status(self, frame_id: str, status: str,
                   class_dist: Optional[dict] = None,
                   annotator: Optional[str] = None) -> None:
        self.conn.execute(
            """UPDATE frames SET status=?, class_dist=COALESCE(?,class_dist),
                   annotator=COALESCE(?,annotator), updated_at=?
               WHERE frame_id=?""",
            (status, json.dumps(class_dist) if class_dist else None,
             annotator, _now(), frame_id),
        )

    def set_split(self, frame_id: str, split: str) -> None:
        self.conn.execute("UPDATE frames SET split=?, updated_at=? WHERE frame_id=?",
                          (split, _now(), frame_id))

    def iter_frames(self, status: Optional[str] = None,
                    bag: Optional[str] = None) -> Iterator[dict]:
        q = "SELECT * FROM frames"
        conds, args = [], []
        if status:
            conds.append("status=?"); args.append(status)
        if bag:
            conds.append("bag=?"); args.append(bag)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY bag, topic, stamp_ns"
        for row in self.conn.execute(q, args):
            d = dict(row)
            if d.get("class_dist"):
                d["class_dist"] = json.loads(d["class_dist"])
            yield d

    def bags(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT bag FROM frames ORDER BY bag")]

    def counts_by_status(self) -> dict[str, int]:
        return {r[0]: r[1] for r in self.conn.execute(
            "SELECT status, COUNT(*) FROM frames GROUP BY status")}

    # -- camera info -------------------------------------------------------
    def put_camera_info(self, topic: str, width: int, height: int,
                        distortion_model: str, K: list, D: list, P: list) -> None:
        self.conn.execute(
            """INSERT INTO camera_info (topic,width,height,distortion_model,K,D,P)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(topic) DO NOTHING""",
            (topic, width, height, distortion_model,
             json.dumps(list(K)), json.dumps(list(D)), json.dumps(list(P))),
        )

    def has_camera_info(self, topic: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM camera_info WHERE topic=?", (topic,)).fetchone() is not None

    def commit(self) -> None:
        self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
