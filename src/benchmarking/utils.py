"""Small filesystem and naming helpers shared by benchmark workflows."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    """Return a compact UTC timestamp suitable for run identifiers."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_name(value: str) -> str:
    """Make an arbitrary label safe for filenames and run identifiers."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write formatted JSON without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
