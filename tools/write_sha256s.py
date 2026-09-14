#!/usr/bin/env python3
"""Write a sorted SHA-256 manifest for the public release."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


paths = sorted(
    path
    for path in ROOT.rglob("*")
    if path.is_file()
    and path != OUTPUT
    and ".git" not in path.relative_to(ROOT).parts
)
OUTPUT.write_text(
    "".join(f"{digest(path)}  {path.relative_to(ROOT)}\n" for path in paths),
    encoding="utf-8",
)
