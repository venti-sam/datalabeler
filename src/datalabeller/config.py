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
        out: list[Path] = []
        for pattern in self.paths["bags"]:
            pat = pattern if Path(pattern).is_absolute() else str(self.root / pattern)
            out.extend(Path(p) for p in sorted(glob.glob(pat)))
        return out

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
