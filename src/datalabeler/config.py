"""Load and validate the pipeline config, with ${workdir} path expansion."""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_VAR = re.compile(r"\$\{(\w+)\}")


@dataclass
class ClassDef:
    name: str
    id: int
    prompts: list[str]


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path

    # convenience accessors -------------------------------------------------
    @property
    def paths(self) -> dict[str, str]:
        return self.raw["paths"]

    def path(self, key: str) -> Path:
        """Absolute path for a `paths.<key>` entry, resolved against config dir."""
        p = Path(self.paths[key])
        return p if p.is_absolute() else (self.root / p)

    @property
    def bag_files(self) -> list[Path]:
        """Resolve `paths.bags` to concrete bag paths, ROS 1 and ROS 2 alike.

        A ROS 1 bag is a single `.bag` (or `.mcap`) file; a ROS 2 bag is a
        *directory* holding `metadata.yaml` + its `.db3`/`.mcap`. Each configured
        entry may point straight at a bag, be a glob, or be a directory that
        *contains* bags -- in that last case we discover the `.bag` files and
        ROS 2 bag dirs one level inside it, so `paths.bags: [data/bags/]` picks up
        whatever kind of bag you dropped in there.
        """
        out: list[Path] = []
        seen: set[Path] = set()

        def add(p: Path) -> None:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(rp)

        def take(p: Path) -> None:
            if p.is_file() and p.suffix in (".bag", ".mcap"):
                add(p)                                   # ROS 1 bag / standalone mcap
            elif p.is_dir() and (p / "metadata.yaml").is_file():
                add(p)                                   # ROS 2 bag directory
            elif p.is_dir():
                # A directory of bags: look one level in for either kind.
                for child in sorted(p.iterdir()):
                    if child.is_file() and child.suffix in (".bag", ".mcap"):
                        add(child)
                    elif child.is_dir() and (child / "metadata.yaml").is_file():
                        add(child)

        for pattern in self.paths["bags"]:
            pat = pattern if Path(pattern).is_absolute() else str(self.root / pattern)
            matches = sorted(glob.glob(pat))
            for m in matches:
                take(Path(m))
        return sorted(out)

    @property
    def classes(self) -> list[ClassDef]:
        return [ClassDef(**c) for c in self.raw["classes"]]

    @property
    def class_by_id(self) -> dict[int, ClassDef]:
        return {c.id: c for c in self.classes}

    @property
    def priority_ids(self) -> list[int]:
        by_name = {c.name: c.id for c in self.classes}
        return [by_name[n] for n in self.raw.get("priority", []) if n in by_name]

    @property
    def extract(self) -> dict[str, Any]:
        return self.raw["extract"]

    @property
    def autolabel(self) -> dict[str, Any]:
        return self.raw["autolabel"]

    @property
    def package(self) -> dict[str, Any]:
        return self.raw["package"]

    @property
    def cvat(self) -> dict[str, Any]:
        return self.raw.get("cvat", {})


def _expand(obj: Any, vars: dict[str, str]) -> Any:
    if isinstance(obj, str):
        return _VAR.sub(lambda m: vars.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, list):
        return [_expand(v, vars) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand(v, vars) for k, v in obj.items()}
    return obj


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    _validate(raw)
    # Expand ${workdir} (and any other top-level paths key) inside paths.
    vars = {k: v for k, v in raw["paths"].items() if isinstance(v, str)}
    # two passes so ${workdir} nested references resolve
    for _ in range(2):
        raw["paths"] = _expand(raw["paths"], vars)
        vars = {k: v for k, v in raw["paths"].items() if isinstance(v, str)}
    return Config(raw=raw, root=path.parent)


def _validate(raw: dict[str, Any]) -> None:
    for section in ("paths", "extract", "classes"):
        if section not in raw:
            raise ValueError(f"config missing required section: {section}")
    ids = [c["id"] for c in raw["classes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("class ids must be unique")
    if 0 in ids:
        raise ValueError("class id 0 is reserved for background/void")
