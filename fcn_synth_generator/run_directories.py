from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_TIMESTAMP_SUFFIX = re.compile(r"_(\d{8}_\d{6})(?:_(\d+))?$")


def format_run_timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime(RUN_TIMESTAMP_FORMAT)


def is_timestamped_directory(path: str | Path) -> bool:
    return _TIMESTAMP_SUFFIX.search(Path(path).name) is not None


def timestamped_directory(
    path: str | Path,
    timestamp: str | None = None,
) -> Path:
    base = Path(path)
    if is_timestamped_directory(base):
        return base

    timestamp = timestamp or format_run_timestamp()
    candidate = base.with_name(f"{base.name}_{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{timestamp}_{suffix:02d}")
        suffix += 1
    return candidate


def latest_timestamped_directory(
    path: str | Path,
    required_file: str | None = None,
) -> Path | None:
    base = Path(path)
    if is_timestamped_directory(base):
        if base.is_dir() and (required_file is None or (base / required_file).is_file()):
            return base
        return None

    prefix = f"{base.name}_"
    candidates: list[tuple[str, int, int, Path]] = []
    if not base.parent.is_dir():
        return None

    for candidate in base.parent.glob(f"{base.name}_*"):
        if not candidate.is_dir() or not candidate.name.startswith(prefix):
            continue
        match = _TIMESTAMP_SUFFIX.search(candidate.name)
        if match is None or match.start() != len(base.name):
            continue
        if required_file is not None and not (candidate / required_file).is_file():
            continue
        sequence = int(match.group(2) or 0)
        candidates.append(
            (
                match.group(1),
                sequence,
                candidate.stat().st_mtime_ns,
                candidate,
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[:3])[3]
