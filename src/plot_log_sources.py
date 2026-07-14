"""Resolve plot log sources from either run_* or matrix_* directories."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _existing_path_candidates(path_text: str, base_dir: Path) -> list[Path]:
    raw = Path(path_text)
    if raw.is_absolute():
        return [raw]
    return [
        PROJECT_ROOT / raw,
        base_dir.parent / raw,
        base_dir / raw,
    ]


def _resolve_existing_path(path_text: object, base_dir: Path) -> Path | None:
    if not path_text:
        return None
    for candidate in _existing_path_candidates(str(path_text), base_dir):
        if candidate.exists():
            return candidate.resolve()
    return None


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_run_dir(path: Path | str) -> Path:
    candidate = Path(path).resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Log directory not found: {path}")

    if candidate.name.startswith("run_"):
        return candidate

    if candidate.name.startswith("matrix_"):
        status_payload = _read_json(candidate / "current_status.json") or {}
        run_dir = _resolve_existing_path(status_payload.get("run_dir"), candidate)
        if run_dir is not None and run_dir.is_dir():
            return run_dir

        csv_path = _resolve_existing_path(status_payload.get("csv"), candidate)
        if csv_path is not None and csv_path.is_file():
            return csv_path.parent.resolve()

        manifest_payload = _read_json(candidate / "manifest.json") or {}
        runs = manifest_payload.get("runs")
        if isinstance(runs, list):
            for item in reversed(runs):
                if not isinstance(item, dict):
                    continue
                run_dir = _resolve_existing_path(item.get("run_dir"), candidate)
                if run_dir is not None and run_dir.is_dir():
                    return run_dir
                csv_path = _resolve_existing_path(item.get("csv"), candidate)
                if csv_path is not None and csv_path.is_file():
                    return csv_path.parent.resolve()

        raise ValueError(f"Matrix log does not point to a usable run directory: {candidate}")

    return candidate


def list_resolved_run_dirs(logs_dir: Path | str) -> list[Path]:
    logs_dir = Path(logs_dir)
    resolved = []
    seen = set()
    for source_dir in sorted(
        (
            path
            for path in logs_dir.iterdir()
            if path.is_dir() and (path.name.startswith("run_") or path.name.startswith("matrix_"))
        ),
        key=lambda path: path.name,
        reverse=True,
    ):
        try:
            run_dir = resolve_run_dir(source_dir)
        except (FileNotFoundError, ValueError):
            continue
        key = str(run_dir)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(run_dir)
    return resolved


def latest_resolved_run_dir(
    logs_dir: Path | str,
    *,
    require_obstacles: bool = False,
) -> Path:
    for run_dir in list_resolved_run_dirs(logs_dir):
        if not require_obstacles or any(run_dir.glob("obstacle_*.json")):
            return run_dir
    raise FileNotFoundError(f"No usable run directory found in {logs_dir}")
