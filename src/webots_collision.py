"""Read collision results produced by the Webots ShipObstacle contact test."""

from __future__ import annotations

import json
from pathlib import Path


COLLISION_RECORD_NAME = "webots_collision.json"


def read_collision_record(source: Path | str, required: bool = False) -> dict | None:
    source = Path(source)
    run_dir = source if source.is_dir() else source.parent
    path = run_dir / COLLISION_RECORD_NAME
    if not path.exists():
        if required:
            raise ValueError(f"No Webots collision sensor record in {run_dir}")
        return None

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        payload.get("source") != "webots_touch_sensor"
        or payload.get("obstacle_model") != "ShipObstacle"
        or payload.get("sensor_available") is not True
        or not isinstance(payload.get("detected"), bool)
    ):
        if required:
            raise ValueError(f"Invalid Webots ShipObstacle collision record: {path}")
        return None
    return payload


def collision_detected(source: Path | str, required: bool = False) -> bool | None:
    record = read_collision_record(source, required=required)
    return None if record is None else record["detected"]


def avoidance_succeeded(source: Path | str, required: bool = False) -> bool | None:
    detected = collision_detected(source, required=required)
    return None if detected is None else not detected


def collision_outcome_text(source: Path | str) -> str:
    detected = collision_detected(source)
    if detected is None:
        return "collision unknown"
    return "collision detected" if detected else "no collision"
