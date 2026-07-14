"""Generate APF trajectory snapshots from obstacle JSON logs."""

from __future__ import annotations

import argparse
import csv
import json
import copy
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np

from webots_collision import collision_outcome_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_OUTPUT_DIR = DEFAULT_LOGS_DIR / "generated_figures"
DEFAULT_BATCH_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "plot_apf_snapshot"
DEFAULT_K_GOAL = 1.0
DEFAULT_K_OBSTACLE = 150.0
TIME_COLUMNS = ("TimeFromStart(s)", "TimeFromStart", "elapsed [s]")
OWN_NORTH_COLUMNS = ("North(m)", "North", "x [m]")
OWN_EAST_COLUMNS = ("East(m)", "East", "y [m]")
ARUCO_NORTH_COLUMNS = ("ARUCOSensedNorth(m)", "ARUCOSensedNorth")
ARUCO_EAST_COLUMNS = ("ARUCOSensedEast(m)", "ARUCOSensedEast")
SHIP_OBSTACLE_LOCAL_GEOMETRY_CENTER_M = np.array([0.025, 0.0, 0.0], dtype=float)
WEBOTS_MOTION_TARGET_CONFIGS = {
    "CROSSING_LEFT_TO_RIGHT_ROBOT": {
        "speed": 0.20,
        "acceleration": 0.0,
        "yaw_rate": 0.0,
        "turn_radius": 0.0,
        "lock_x": 4.8,
        "wrap_y_top": 4.67,
        "wrap_y_bottom": -5.0,
        "start_with_zero_speed": True,
    },
    "CROSSING_RIGHT_TO_LEFT_ROBOT": {
        "speed": 0.20,
        "acceleration": 0.0,
        "yaw_rate": 0.0,
        "turn_radius": 0.0,
        "lock_x": 4.8,
        "wrap_y_top": 2.6,
        "wrap_y_bottom": -5.0,
        "start_with_zero_speed": True,
    },
    "CROSSING_PORT_TO_STARBOARD_ROBOT": {
        "speed": 0.20,
        "acceleration": 0.0,
        "yaw_rate": 0.0,
        "turn_radius": 0.0,
        "lock_x": 2.2,
        "wrap_y_top": 2.0,
        "wrap_y_bottom": -5.0,
        "start_with_zero_speed": True,
    },
    "CROSSING_STARBOARD_TO_PORT_ROBOT": {
        "speed": 0.20,
        "acceleration": 0.0,
        "yaw_rate": 0.0,
        "turn_radius": 0.0,
        "lock_x": 7.4,
        "wrap_y_top": 2.0,
        "wrap_y_bottom": -5.0,
        "start_with_zero_speed": True,
    },
    "MOVING_ROBOT": {
        "speed": 0.10,
        "acceleration": 0.0,
        "yaw_rate": 0.0,
        "turn_radius": 0.0,
        "lock_x": 4.8,
        "wrap_y_top": 2.6,
        "wrap_y_bottom": -3.8,
        "start_with_zero_speed": True,
    },
    "MOVING_ROBOT_2": {
        "speed": 0.10,
        "acceleration": 0.0,
        "yaw_rate": 0.0,
        "turn_radius": 0.0,
        "lock_x": 1.8,
        "bounce_y_top": 6.0,
        "bounce_y_bottom": -6.0,
        "start_with_zero_speed": True,
        "match_ego_route_speed": False,
    },
    "MOVING_ROBOT_3": {
        "speed": 0.10,
        "acceleration": 0.0,
        "yaw_rate": 0.0,
        "turn_radius": 0.0,
        "lock_x": 7.8,
        "bounce_y_top": 6.0,
        "bounce_y_bottom": -6.0,
        "start_with_zero_speed": True,
    },
    "FRONT_OBSTACLE_ROBOT": {
        "speed": 0.074,
        "acceleration": 0.025,
        "yaw_rate": 0.0,
        "turn_radius": 0.0,
        "lock_y": -1.0,
        "stop_x": 10.0,
        "start_with_zero_speed": True,
    },
    "HEAD_ON_OBSTACLE_ROBOT": {
        "speed": 0.062,
        "acceleration": 0.035,
        "yaw_rate": 0.0,
        "turn_radius": 0.0,
        "lock_y": -1.0,
        "start_with_zero_speed": True,
    },
}


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def first_existing(row, names):
    return next((row[name] for name in names if name in row), None)


def find_pseudo_aruco_csv(run_dir):
    candidates = list(Path(run_dir).glob("log_*_pseudo_aruco.csv"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _read_json(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolved_entry_run_dir(entry, base_dir):
    if not isinstance(entry, dict):
        return None
    for key in ("run_dir", "csv"):
        value = entry.get(key)
        if not value:
            continue
        raw = Path(str(value))
        candidates = [raw] if raw.is_absolute() else [
            PROJECT_ROOT / raw,
            base_dir.parent / raw,
            base_dir / raw,
        ]
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate.resolve()
                return resolved.parent if key == "csv" else resolved
    return None


def resolve_run_dir(path):
    candidate = Path(path).resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Log directory not found: {path}")
    if not candidate.name.startswith("matrix_"):
        return candidate

    run_dir = _resolved_entry_run_dir(
        _read_json(candidate / "current_status.json"), candidate
    )
    if run_dir is not None and run_dir.is_dir():
        return run_dir

    runs = _read_json(candidate / "manifest.json").get("runs", [])
    if isinstance(runs, list):
        for entry in reversed(runs):
            run_dir = _resolved_entry_run_dir(entry, candidate)
            if run_dir is not None and run_dir.is_dir():
                return run_dir
    raise ValueError(f"Matrix log does not point to a usable run directory: {candidate}")


def list_resolved_run_dirs(logs_dir):
    resolved = []
    seen = set()
    for source_dir in sorted(Path(logs_dir).iterdir(), key=lambda path: path.name, reverse=True):
        if not source_dir.is_dir() or not source_dir.name.startswith(("run_", "matrix_")):
            continue
        try:
            run_dir = resolve_run_dir(source_dir)
        except (FileNotFoundError, ValueError):
            continue
        if run_dir not in seen:
            seen.add(run_dir)
            resolved.append(run_dir)
    return resolved


@dataclass(frozen=True)
class RunSelection:
    run_dir: Path
    world_name: str
    combination: str


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return None


def switch_combination_name(row):
    switch_name = str(row.get("SwitchCombination", "")).strip().lower()
    if switch_name:
        return switch_name
    ekf_on = parse_bool(row.get("EKFPredictionEnabled")) is True
    cluster_on = parse_bool(
        row.get("ClusterSizeAPFEnabled")
        if "ClusterSizeAPFEnabled" in row
        else row.get("ClusterAPFEnabled")
    ) is True
    ekf_token = "on" if ekf_on else "off"
    cluster_token = "on" if cluster_on else "off"
    return f"ekf_{ekf_token}_cluster_{cluster_token}"


def find_primary_csv(run_dir):
    candidates = [
        path
        for path in Path(run_dir).glob("log_*.csv")
        if not path.name.endswith("_pseudo_aruco.csv")
    ]
    if not candidates:
        raise FileNotFoundError(f"No primary log CSV found in {run_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_run_metadata_from_csv(run_dir):
    with find_primary_csv(run_dir).open(newline="", encoding="utf-8-sig") as stream:
        first_row = next(csv.DictReader(stream), None)
    if first_row is None:
        return None
    world_name = Path(str(first_row.get("WebotsEnvironment") or "").strip()).name
    combination = switch_combination_name(first_row)
    if not world_name or not combination:
        return None
    return RunSelection(Path(run_dir).resolve(), world_name, combination)


def latest_matrix_dirs(logs_dir):
    return sorted(
        (
            path
            for path in Path(logs_dir).iterdir()
            if path.is_dir() and path.name.startswith("matrix_")
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def has_snapshot_logs(run_dir):
    return any(Path(run_dir).glob("obstacle_*.json"))


def load_latest_world_runs(logs_dir, combination="ekf_on_cluster_on"):
    latest_by_world = {}
    for matrix_dir in latest_matrix_dirs(logs_dir):
        manifest_path = matrix_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            continue
        for item in manifest.get("runs", []):
            if not isinstance(item, dict):
                continue
            if str(item.get("combination", "")).strip().lower() != combination:
                continue
            world_name = Path(str(item.get("world") or "")).name
            run_dir_text = item.get("run_dir")
            if not world_name or not run_dir_text or world_name in latest_by_world:
                continue
            run_dir = (PROJECT_ROOT / str(run_dir_text)).resolve()
            if run_dir.is_dir() and has_snapshot_logs(run_dir):
                latest_by_world[world_name] = RunSelection(
                    run_dir=run_dir,
                    world_name=world_name,
                    combination=combination,
                )
    if latest_by_world:
        return sorted(latest_by_world.values(), key=lambda item: Path(item.world_name).stem)

    for run_dir in list_resolved_run_dirs(logs_dir):
        if not has_snapshot_logs(run_dir):
            continue
        try:
            selection = load_run_metadata_from_csv(run_dir)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if selection is None or selection.combination != combination:
            continue
        latest_by_world.setdefault(selection.world_name, selection)
    return sorted(latest_by_world.values(), key=lambda item: Path(item.world_name).stem)


def world_output_dir(output_dir, world_name):
    return Path(output_dir) / Path(world_name).stem


def point(value):
    try:
        value = np.asarray(value, dtype=float).reshape(2)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value).all() else None


def point_cloud(value):
    try:
        cloud = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    cloud = cloud[:, :2]
    return cloud[np.isfinite(cloud).all(axis=1)]


def positive(value, fallback):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return value if np.isfinite(value) and value > 0.0 else float(fallback)


def load_snapshots(run_dir):
    snapshots = []
    for path in sorted(Path(run_dir).glob("obstacle_*.json")):
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        try:
            time_s = float(payload.get("t"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(time_s) and point(payload.get("robot_pos")) is not None:
            snapshots.append({"path": path, "time_s": time_s, "payload": payload})

    snapshots.sort(key=lambda snapshot: snapshot["time_s"])
    if not snapshots:
        raise ValueError(f"No valid obstacle_*.json snapshots found in {run_dir}")

    start_s = snapshots[0]["time_s"]
    for snapshot in snapshots:
        snapshot["relative_time_s"] = snapshot["time_s"] - start_s
    return snapshots


def read_trajectory_samples_csv(log_path, north_columns, east_columns):
    times = []
    points = []
    with Path(log_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{log_path} has no CSV header")
        for row in reader:
            time_s = parse_float(first_existing(row, TIME_COLUMNS))
            north_m = parse_float(first_existing(row, north_columns))
            east_m = parse_float(first_existing(row, east_columns))
            if not (
                np.isfinite(time_s)
                and np.isfinite(north_m)
                and np.isfinite(east_m)
            ):
                continue
            times.append(float(time_s))
            points.append([north_m, east_m])
    if not times:
        raise ValueError(f"No valid trajectory samples in {log_path}")
    time_array = np.asarray(times, dtype=float)
    trajectory_ne_m = np.asarray(points, dtype=float)
    order = np.argsort(time_array)
    return time_array[order], trajectory_ne_m[order]


def load_robot_trajectory(run_dir):
    pseudo_aruco_path = find_pseudo_aruco_csv(run_dir)
    if pseudo_aruco_path is not None:
        try:
            return read_trajectory_samples_csv(
                pseudo_aruco_path,
                OWN_NORTH_COLUMNS,
                OWN_EAST_COLUMNS,
            )
        except ValueError:
            pass

    log_path = find_primary_csv(run_dir)
    try:
        return read_trajectory_samples_csv(
            log_path,
            ARUCO_NORTH_COLUMNS,
            ARUCO_EAST_COLUMNS,
        )
    except ValueError:
        return read_trajectory_samples_csv(
            log_path,
            OWN_NORTH_COLUMNS,
            OWN_EAST_COLUMNS,
        )


def wrap_to_pi(angle_rad):
    return (float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi


def advance_speed(current_speed, target_speed, acceleration, dt):
    current_speed = float(current_speed)
    target_speed = float(target_speed)
    acceleration = float(acceleration)
    dt = float(dt)
    if acceleration <= 1e-9:
        return target_speed
    delta_speed = target_speed - current_speed
    max_step = acceleration * dt
    if abs(delta_speed) <= max_step:
        return target_speed
    return current_speed + np.sign(delta_speed) * max_step


def parse_world_motion_targets(world_name):
    world_path = PROJECT_ROOT / "webots" / "worlds" / Path(str(world_name)).name
    if not world_path.is_file():
        return []
    lines = world_path.read_text(encoding="utf-8").splitlines()
    custom_speed = None
    for line in lines:
        match = re.search(r'front_obstacle_speed=([0-9.]+)', line)
        if match:
            custom_speed = float(match.group(1))
            break

    targets = []
    current = None
    for raw_line in lines:
        line = raw_line.strip()
        match = re.match(r"DEF\s+([A-Z0-9_]+)\s+ShipObstacle\s*\{", line)
        if match:
            name = match.group(1)
            if name in WEBOTS_MOTION_TARGET_CONFIGS:
                current = {
                    "name": name,
                    "translation": None,
                    "yaw": 0.0,
                    "scale": np.array([1.0, 1.0, 1.0], dtype=float),
                }
            else:
                current = None
            continue
        if current is None:
            continue
        if line.startswith("translation "):
            parts = line.split()
            current["translation"] = np.asarray(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=float,
            )
        elif line.startswith("rotation "):
            parts = line.split()
            current["yaw"] = float(parts[4])
        elif line.startswith("scale "):
            parts = line.split()
            current["scale"] = np.asarray(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=float,
            )
        elif line == "}":
            if current.get("translation") is not None:
                config = copy.deepcopy(WEBOTS_MOTION_TARGET_CONFIGS[current["name"]])
                if current["name"] == "FRONT_OBSTACLE_ROBOT" and custom_speed is not None:
                    config.update(
                        speed=float(custom_speed),
                        acceleration=0.0,
                        start_with_zero_speed=False,
                    )
                current_speed = (
                    0.0
                    if config.get("start_with_zero_speed") and config.get("acceleration", 0.0) > 0.0
                    else float(config["speed"])
                )
                targets.append(
                    {
                        "name": current["name"],
                        "position": current["translation"].copy(),
                        "initial_position": current["translation"].copy(),
                        "yaw": float(current["yaw"]),
                        "initial_yaw": float(current["yaw"]),
                        "scale": current["scale"].copy(),
                        "current_speed": float(current_speed),
                        **config,
                    }
                )
            current = None
    return targets


def simulate_webots_target(target, elapsed_s, dt=0.05):
    target = copy.deepcopy(target)
    elapsed_s = max(float(elapsed_s), 0.0)
    simulated_s = 0.0
    while simulated_s < elapsed_s - 1e-9:
        step_s = min(float(dt), elapsed_s - simulated_s)
        target["current_speed"] = advance_speed(
            target["current_speed"],
            target["speed"],
            target["acceleration"],
            step_s,
        )
        yaw_rate = float(target.get("yaw_rate", 0.0))
        turn_radius = float(target.get("turn_radius", 0.0))
        if abs(yaw_rate) <= 1e-9 and abs(turn_radius) > 1e-9 and abs(target["current_speed"]) > 1e-9:
            yaw_rate = target["current_speed"] / turn_radius
        target["yaw"] = wrap_to_pi(target["yaw"] + yaw_rate * step_s)
        direction = np.array(
            [np.cos(target["yaw"]), np.sin(target["yaw"]), 0.0],
            dtype=float,
        )
        target["position"] = target["position"] + direction * target["current_speed"] * step_s
        if target.get("lock_x") is not None and abs(yaw_rate) <= 1e-9 and abs(turn_radius) <= 1e-9:
            target["position"][0] = float(target["lock_x"])
        if target.get("lock_y") is not None and abs(yaw_rate) <= 1e-9 and abs(turn_radius) <= 1e-9:
            target["position"][1] = float(target["lock_y"])
        if target.get("stop_x") is not None and target["position"][0] >= float(target["stop_x"]):
            target["position"][0] = float(target["stop_x"])
            target["speed"] = 0.0
            target["current_speed"] = 0.0

        wrap_y_top = target.get("wrap_y_top")
        wrap_y_bottom = target.get("wrap_y_bottom")
        if wrap_y_top is not None and wrap_y_bottom is not None:
            if target["position"][1] < float(wrap_y_bottom):
                target["position"] = target["initial_position"].copy()
                target["position"][1] = float(wrap_y_top)
                target["yaw"] = float(target["initial_yaw"])
                target["current_speed"] = 0.0 if target["acceleration"] > 0.0 else float(target["speed"])
            elif target["position"][1] > float(wrap_y_top):
                target["position"] = target["initial_position"].copy()
                target["position"][1] = float(wrap_y_bottom)
                target["yaw"] = float(target["initial_yaw"])
                target["current_speed"] = 0.0 if target["acceleration"] > 0.0 else float(target["speed"])

        bounce_y_top = target.get("bounce_y_top")
        bounce_y_bottom = target.get("bounce_y_bottom")
        if bounce_y_top is not None and bounce_y_bottom is not None:
            if target["position"][1] < float(bounce_y_bottom):
                target["position"][1] = float(bounce_y_bottom)
                target["yaw"] = wrap_to_pi(target["yaw"] + np.pi)
                target["current_speed"] = 0.0 if target["acceleration"] > 0.0 else float(target["speed"])
            elif target["position"][1] > float(bounce_y_top):
                target["position"][1] = float(bounce_y_top)
                target["yaw"] = wrap_to_pi(target["yaw"] + np.pi)
                target["current_speed"] = 0.0 if target["acceleration"] > 0.0 else float(target["speed"])
        simulated_s += step_s
    return target


def simulate_webots_targets(payload, elapsed_s):
    run_context = payload.get("run_context", {})
    if not isinstance(run_context, dict):
        return []
    world_name = run_context.get("webots_environment")
    if not world_name:
        return []
    targets = parse_world_motion_targets(world_name)
    return [simulate_webots_target(target, elapsed_s) for target in targets]


def webots_geometry_center_position(target):
    position = np.asarray(target["position"], dtype=float).reshape(3)
    scale = np.asarray(target.get("scale", [1.0, 1.0, 1.0]), dtype=float).reshape(3)
    local_offset = SHIP_OBSTACLE_LOCAL_GEOMETRY_CENTER_M * scale
    yaw_rad = float(target["yaw"])
    rotation = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return position + rotation @ local_offset


def webots_position_to_ne(target):
    position = webots_geometry_center_position(target)
    return np.array([position[0], -position[1]], dtype=float)


def webots_heading_vector_to_ne(target):
    yaw_rad = float(target["yaw"])
    return np.array([np.cos(yaw_rad), -np.sin(yaw_rad)], dtype=float)


def matched_webots_target(obstacle, webots_targets, max_distance_m=2.5):
    centre_ne = point(obstacle.get("centre_ne")) if isinstance(obstacle, dict) else None
    if centre_ne is None or not webots_targets:
        return None
    nearest_target = None
    nearest_distance = float("inf")
    for target in webots_targets:
        distance_m = float(np.linalg.norm(webots_position_to_ne(target) - centre_ne))
        if distance_m < nearest_distance:
            nearest_distance = distance_m
            nearest_target = target
    return nearest_target if nearest_distance <= float(max_distance_m) else None


def select_snapshot_indices(snapshots, interval_s):
    interval_s = positive(interval_s, 5.0)
    duration_s = snapshots[-1]["relative_time_s"]
    targets = np.arange(0.0, duration_s + 1e-9, interval_s)
    times = np.asarray(
        [snapshot["relative_time_s"] for snapshot in snapshots],
        dtype=float,
    )
    indices = []
    for target_s in targets:
        index = int(np.argmin(np.abs(times - target_s)))
        if not indices or index != indices[-1]:
            indices.append(index)
    return indices


def obstacle_dimensions(obstacle, settings):
    minimum_pc1_m = positive(settings.get("minimum_pc1_m"), 0.30)
    minimum_pc2_m = positive(settings.get("minimum_pc2_m"), 0.16)
    equivalent_radius_m = positive(
        obstacle.get("equivalent_radius_m"),
        0.5 * minimum_pc1_m,
    )
    pc1_m = positive(obstacle.get("pc1_m"), 2.0 * equivalent_radius_m)
    pc2_m = positive(obstacle.get("pc2_m"), min(pc1_m, 2.0 * equivalent_radius_m))
    return max(pc1_m, minimum_pc1_m), max(min(pc2_m, pc1_m), minimum_pc2_m)


def normalized_axis(value):
    axis = point(value)
    if axis is None:
        return np.array([1.0, 0.0], dtype=float)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm >= 1e-9 else np.array([1.0, 0.0], dtype=float)


def obstacle_ellipse_geometry(obstacle, settings):
    centre = point(obstacle.get("centre_ne"))
    if centre is None:
        return None
    pc1_m, pc2_m = obstacle_dimensions(obstacle, settings)
    axis_ne = normalized_axis(obstacle.get("length_axis_ne"))
    angle_deg = float(np.degrees(np.arctan2(axis_ne[0], axis_ne[1])))
    return centre, pc1_m, pc2_m, angle_deg


def attractive_potential(north_grid, east_grid, target_ne, k_goal):
    target_ne = point(target_ne)
    if target_ne is None:
        return np.zeros_like(north_grid)
    return 0.5 * float(k_goal) * (
        (north_grid - target_ne[0]) ** 2
        + (east_grid - target_ne[1]) ** 2
    )


def classic_repulsive_potential(
    distance_grid,
    influence_distance_m,
    repulsive_gain,
):
    influence_distance_m = positive(influence_distance_m, 3.0)
    distance_grid = np.asarray(distance_grid, dtype=float)
    safe_distance = np.maximum(distance_grid, 1e-3)
    potential = np.zeros_like(safe_distance)
    active = safe_distance <= influence_distance_m
    potential[active] = 0.5 * float(repulsive_gain) * (
        1.0 / safe_distance[active] - 1.0 / influence_distance_m
    ) ** 2
    return potential


def ellipse_potential(
    north_grid,
    east_grid,
    obstacle,
    settings,
    k_obstacle=DEFAULT_K_OBSTACLE,
):
    centre = point(obstacle.get("centre_ne"))
    if centre is None:
        return np.zeros_like(north_grid)

    delta_n = north_grid - centre[0]
    delta_e = east_grid - centre[1]
    if settings.get("cluster_range_enabled", True) is False:
        return classic_repulsive_potential(
            np.hypot(delta_n, delta_e),
            settings.get("classic_influence_distance_m"),
            k_obstacle,
        )

    pc1_m, pc2_m = obstacle_dimensions(obstacle, settings)
    scale = positive(
        settings.get("avoidance_pc_scale"),
        settings.get("cluster_influence_scale", 6.0),
    )
    own_radius_m = positive(settings.get("own_equivalent_radius_m"), 0.0)
    semi_length_m = max(0.5 * scale * pc1_m + own_radius_m, 1e-6)
    semi_width_m = max(0.5 * scale * pc2_m + own_radius_m, 1e-6)
    axis = normalized_axis(obstacle.get("length_axis_ne"))
    width_axis = np.array([-axis[1], axis[0]], dtype=float)

    along = delta_n * axis[0] + delta_e * axis[1]
    across = delta_n * width_axis[0] + delta_e * width_axis[1]
    # Keep the plotted repulsive peak at centre_ne while preserving the
    # U=k/e domain boundary at the scaled pc1/pc2 ellipse.
    exponent = -(
        (along / semi_length_m) ** 2
        + (across / semi_width_m) ** 2
    )
    return float(k_obstacle) * np.exp(exponent)


def segment_potential(
    north_grid,
    east_grid,
    obstacle,
    settings,
    k_obstacle=DEFAULT_K_OBSTACLE,
):
    start = point(obstacle.get("segment_start_ne"))
    end = point(obstacle.get("segment_end_ne"))
    if start is None or end is None:
        return np.zeros_like(north_grid)

    pc1_m, pc2_m = obstacle_dimensions(obstacle, settings)
    scale = positive(
        settings.get("virtual_pc_scale"),
        settings.get("avoidance_pc_scale", 6.0),
    )
    segment = end - start
    segment_length_m = float(np.linalg.norm(segment))
    cluster_range_enabled = settings.get("cluster_range_enabled", True) is not False
    if cluster_range_enabled and segment_length_m >= 1e-9:
        axis = segment / segment_length_m
        extension_m = max(0.5 * scale * pc1_m, 1e-6)
        start = start - extension_m * axis
        end = end + extension_m * axis
        segment = end - start

    segment_length_sq = max(float(np.dot(segment, segment)), 1e-12)
    delta_n = north_grid - start[0]
    delta_e = east_grid - start[1]
    ratio = np.clip(
        (delta_n * segment[0] + delta_e * segment[1]) / segment_length_sq,
        0.0,
        1.0,
    )
    closest_n = start[0] + ratio * segment[0]
    closest_e = start[1] + ratio * segment[1]
    distance_grid = np.hypot(
        north_grid - closest_n,
        east_grid - closest_e,
    )
    if not cluster_range_enabled:
        return classic_repulsive_potential(
            distance_grid,
            settings.get("classic_influence_distance_m"),
            k_obstacle,
        )

    corridor_radius_m = max(0.5 * scale * pc2_m, 1e-6)
    level = distance_grid / corridor_radius_m
    return float(k_obstacle) * np.exp(-(level ** 4))


def potential_components(
    north_grid,
    east_grid,
    payload,
    default_target_ne,
    k_goal,
    k_obstacle,
):
    settings = payload.get("apf_settings", {})
    settings = settings if isinstance(settings, dict) else {}
    apf = payload.get("apf", {})
    apf = apf if isinstance(apf, dict) else {}
    target_ne = point(apf.get("target_ne"))
    if target_ne is None:
        target_ne = point(default_target_ne)
    attractive = attractive_potential(
        north_grid,
        east_grid,
        target_ne,
        k_goal,
    )
    real = np.zeros_like(north_grid)
    virtual = np.zeros_like(north_grid)

    for obstacle in payload.get("clusters", []):
        if isinstance(obstacle, dict):
            real += ellipse_potential(
                north_grid,
                east_grid,
                obstacle,
                settings,
                k_obstacle,
            )

    for obstacle in payload.get("virtual_obstacles", []):
        if not isinstance(obstacle, dict):
            continue
        if "segment_start_ne" in obstacle and "segment_end_ne" in obstacle:
            virtual += segment_potential(
                north_grid,
                east_grid,
                obstacle,
                settings,
                k_obstacle,
            )
        else:
            virtual += ellipse_potential(
                north_grid,
                east_grid,
                obstacle,
                settings,
                k_obstacle,
            )

    return attractive, real, virtual, target_ne


def all_run_points(snapshots):
    points = []
    for snapshot in snapshots:
        payload = snapshot["payload"]
        candidate = point(payload.get("robot_pos"))
        if candidate is not None:
            points.append(candidate)
        cloud = point_cloud(payload.get("cloud", []))
        if len(cloud):
            points.extend(cloud)
        for collection_name in ("clusters", "tracks"):
            for item in payload.get(collection_name, []):
                if not isinstance(item, dict):
                    continue
                candidate = point(
                    item.get("centre_ne", item.get("position_ne"))
                )
                if candidate is not None:
                    points.append(candidate)
        for virtual in payload.get("virtual_obstacles", []):
            if not isinstance(virtual, dict):
                continue
            for key in ("segment_start_ne", "segment_end_ne", "centre_ne"):
                candidate = point(virtual.get(key))
                if candidate is not None:
                    points.append(candidate)
    return np.asarray(points, dtype=float)


def run_bounds(snapshots, map_size_m=20.0):
    points = all_run_points(snapshots)
    if len(points) == 0:
        return (-2.0, 2.0, -2.0, 2.0)
    north_min, east_min = np.min(points, axis=0)
    north_max, east_max = np.max(points, axis=0)
    north_centre = 0.5 * (north_min + north_max)
    east_centre = 0.5 * (east_min + east_max)
    half_span_m = 0.5 * positive(map_size_m, 20.0)
    return (
        north_centre - half_span_m,
        north_centre + half_span_m,
        east_centre - half_span_m,
        east_centre + half_span_m,
    )


def prediction_points(track):
    for key in ("prediction_ne", "predicted_trajectory_ne"):
        try:
            values = np.asarray(track.get(key, []), dtype=float)
        except (TypeError, ValueError):
            continue
        if values.ndim == 2 and values.shape[1] >= 2:
            values = values[:, :2]
            return values[np.isfinite(values).all(axis=1)]
    return np.empty((0, 2), dtype=float)


def straight_line_display_prediction(track, fallback_prediction, horizon_s):
    fallback_prediction = np.asarray(fallback_prediction, dtype=float)
    if fallback_prediction.ndim != 2 or fallback_prediction.shape[1] < 2:
        fallback_prediction = np.empty((0, 2), dtype=float)

    start_ne = (
        point(track.get("position_ne")) if isinstance(track, dict) else None
    )
    if start_ne is None and len(fallback_prediction):
        start_ne = fallback_prediction[0]
    if start_ne is None:
        return np.empty((0, 2), dtype=float)

    sample_count = max(len(fallback_prediction), 2)
    velocity_ne = velocity_vector(track) if isinstance(track, dict) else None
    if velocity_ne is not None and float(np.linalg.norm(velocity_ne)) > 1e-9:
        times = np.linspace(0.0, max(float(horizon_s), 0.0), sample_count)
        return np.asarray(
            [start_ne + velocity_ne * dt for dt in times],
            dtype=float,
        )

    if len(fallback_prediction) >= 2:
        end_ne = fallback_prediction[-1]
        ratios = np.linspace(0.0, 1.0, sample_count)
        return np.asarray(
            [start_ne + (end_ne - start_ne) * ratio for ratio in ratios],
            dtype=float,
        )

    return np.asarray([start_ne], dtype=float)


def finite_speed(value):
    speed = parse_float(value)
    return float(speed) if np.isfinite(speed) and speed >= 0.0 else np.nan


def velocity_vector(entry):
    for key in ("velocity_ne", "velocity_mean_ne"):
        vector = point(entry.get(key))
        if vector is not None:
            return vector
    return None


def speed_from_vector(vector):
    if vector is None:
        return np.nan
    return float(np.linalg.norm(vector))


def heading_deg_from_vector(vector):
    if vector is None:
        return np.nan
    if float(np.linalg.norm(vector)) <= 1e-9:
        return np.nan
    return float((np.degrees(np.arctan2(vector[1], vector[0])) + 360.0) % 360.0)


def format_speed(speed):
    return f"{speed:.2f} m/s" if np.isfinite(speed) else "n/a"


def format_heading_deg(heading_deg):
    return f"{heading_deg:.0f}°" if np.isfinite(heading_deg) else "n/a"


def matched_track(obstacle, tracks):
    if not isinstance(obstacle, dict):
        return None
    track_id = obstacle.get("track_id")
    if track_id is not None:
        for track in tracks:
            if isinstance(track, dict) and track.get("id") == track_id:
                return track
    centre_ne = point(obstacle.get("centre_ne"))
    if centre_ne is None:
        return None
    nearest_track = None
    nearest_distance = float("inf")
    for track in tracks:
        if not isinstance(track, dict):
            continue
        position_ne = point(track.get("position_ne"))
        if position_ne is None:
            continue
        distance_m = float(np.linalg.norm(position_ne - centre_ne))
        if distance_m < nearest_distance:
            nearest_distance = distance_m
            nearest_track = track
    return nearest_track


def plan_direction_vector(track, prediction):
    if len(prediction) >= 2:
        direction = prediction[-1] - prediction[0]
        if float(np.linalg.norm(direction)) > 1e-9:
            return direction
    if isinstance(track, dict):
        direction = velocity_vector(track)
        if direction is not None and float(np.linalg.norm(direction)) > 1e-9:
            return direction
    return None


def plot_snapshot(
    snapshot,
    robot_history,
    robot_position,
    bounds,
    output_path,
    grid_size,
    target_time_s,
    default_target_ne,
    k_goal,
    k_obstacle,
    quiver_step,
    run_collision_outcome,
):
    payload = snapshot["payload"]
    webots_targets = simulate_webots_targets(payload, snapshot["time_s"])
    potential_payload = payload
    north_min, north_max, east_min, east_max = bounds
    north_axis = np.linspace(north_min, north_max, grid_size)
    east_axis = np.linspace(east_min, east_max, grid_size)
    east_grid, north_grid = np.meshgrid(east_axis, north_axis)
    (
        attractive_potential_map,
        real_potential,
        virtual_potential,
        target_ne,
    ) = potential_components(
        north_grid,
        east_grid,
        potential_payload,
        default_target_ne,
        k_goal,
        k_obstacle,
    )
    total_potential = attractive_potential_map + real_potential + virtual_potential

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    minimum = float(np.min(total_potential))
    maximum = float(np.max(total_potential))
    if maximum - minimum > 1e-9:
        contour = ax.contourf(
            east_grid,
            north_grid,
            total_potential,
            levels=np.linspace(minimum, maximum, 32),
            cmap="coolwarm",
            alpha=0.72,
            vmin=minimum,
            vmax=maximum,
        )
        colorbar = fig.colorbar(contour, ax=ax, pad=0.02)
        colorbar.set_label("Total APF potential, U")

    settings = payload.get("apf_settings", {})
    settings = settings if isinstance(settings, dict) else {}
    prediction_horizon_s = positive(
        settings.get("obstacle_prediction_horizon_s"),
        30.0,
    )
    domain_level = float(k_obstacle) / np.e
    for obstacle in potential_payload.get("clusters", []):
        if not isinstance(obstacle, dict):
            continue
        field = ellipse_potential(
            north_grid,
            east_grid,
            obstacle,
            settings,
            k_obstacle,
        )
        if float(np.min(field)) <= domain_level <= float(np.max(field)):
            ax.contour(
                east_grid,
                north_grid,
                field,
                levels=[domain_level],
                colors="black",
                linewidths=1.1,
            )

    for obstacle in potential_payload.get("virtual_obstacles", []):
        if not isinstance(obstacle, dict):
            continue
        field = (
            segment_potential(north_grid, east_grid, obstacle, settings, k_obstacle)
            if "segment_start_ne" in obstacle and "segment_end_ne" in obstacle
            else ellipse_potential(north_grid, east_grid, obstacle, settings, k_obstacle)
        )
        if float(np.min(field)) <= domain_level <= float(np.max(field)):
            ax.contour(
                east_grid,
                north_grid,
                field,
                levels=[domain_level],
                colors="#ff7f0e",
                linestyles="--",
                linewidths=1.1,
            )

    gradient_north, gradient_east = np.gradient(
        total_potential,
        north_axis,
        east_axis,
    )
    gradient_norm = np.hypot(gradient_east, gradient_north)
    finite_gradient = gradient_norm > 1e-9
    descent_east = np.zeros_like(gradient_east)
    descent_north = np.zeros_like(gradient_north)
    descent_east[finite_gradient] = (
        -gradient_east[finite_gradient] / gradient_norm[finite_gradient]
    )
    descent_north[finite_gradient] = (
        -gradient_north[finite_gradient] / gradient_norm[finite_gradient]
    )
    step = max(int(quiver_step), 1)
    ax.quiver(
        east_grid[::step, ::step],
        north_grid[::step, ::step],
        descent_east[::step, ::step],
        descent_north[::step, ::step],
        color="black",
        alpha=0.28,
        pivot="mid",
        scale=35,
        width=0.002,
        zorder=3,
    )

    history = np.asarray(robot_history, dtype=float)
    ax.plot(
        history[:, 1],
        history[:, 0],
        color="#1f77b4",
        linewidth=2.2,
        label="OS trajectory",
        zorder=5,
    )
    ax.scatter(
        [robot_position[1]],
        [robot_position[0]],
        marker="*",
        s=180,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.8,
        label="OS current position",
        zorder=8,
    )
    cloud = point_cloud(payload.get("cloud", []))
    if len(cloud):
        ax.scatter(
            cloud[:, 1],
            cloud[:, 0],
            marker=".",
            s=8,
            color="#7f7f7f",
            alpha=0.65,
            label="LiDAR point cloud (earth frame)",
            zorder=4,
        )
    if target_ne is not None:
        ax.scatter(
            [target_ne[1]],
            [target_ne[0]],
            marker="X",
            s=90,
            color="#39ff14",
            edgecolor="black",
            linewidth=0.7,
            label="APF attractive target",
            zorder=9,
        )

    clusters = [
        obstacle
        for obstacle in payload.get("clusters", [])
        if isinstance(obstacle, dict) and point(obstacle.get("centre_ne")) is not None
    ]
    if clusters:
        centres = np.asarray(
            [point(obstacle.get("centre_ne")) for obstacle in clusters],
            dtype=float,
        )
        ax.scatter(
            centres[:, 1],
            centres[:, 0],
            marker="s",
            s=75,
            color="#d62728",
            edgecolor="white",
            linewidth=0.7,
            label="Obstacle ship current position",
            zorder=8,
        )
        tracks = [
            track
            for track in payload.get("tracks", [])
            if isinstance(track, dict)
        ]
        has_webots_real_position = False
        has_webots_real_direction = False
        for cluster_index, obstacle in enumerate(clusters):
            geometry = obstacle_ellipse_geometry(obstacle, settings)
            if geometry is None:
                continue
            centre_ne, pc1_m, pc2_m, angle_deg = geometry
            track = matched_track(obstacle, tracks)
            webots_target = matched_webots_target(obstacle, webots_targets)
            raw_prediction = (
                prediction_points(track) if track is not None else np.empty((0, 2))
            )
            prediction = straight_line_display_prediction(
                track,
                raw_prediction,
                prediction_horizon_s,
            )
            webots_position_ne = (
                webots_position_to_ne(webots_target)
                if webots_target is not None
                else None
            )
            real_heading_vector = (
                webots_heading_vector_to_ne(webots_target)
                if webots_target is not None
                else None
            )
            real_speed = (
                float(webots_target["current_speed"])
                if webots_target is not None
                else np.nan
            )
            if not np.isfinite(real_speed):
                real_speed = finite_speed(obstacle.get("speed_m_s"))
            if not np.isfinite(real_speed):
                real_speed = speed_from_vector(velocity_vector(obstacle))
            predicted_speed = (
                finite_speed(track.get("speed_m_s")) if track is not None else np.nan
            )
            if not np.isfinite(predicted_speed):
                direction = velocity_vector(track) if track is not None else None
                predicted_speed = speed_from_vector(direction)
            planned_direction = plan_direction_vector(track, prediction)
            planned_heading_deg = heading_deg_from_vector(planned_direction)
            real_heading_deg = heading_deg_from_vector(real_heading_vector)
            ax.add_patch(
                Ellipse(
                    xy=(centre_ne[1], centre_ne[0]),
                    width=pc1_m,
                    height=pc2_m,
                    angle=angle_deg,
                    facecolor="#d62728",
                    edgecolor="white",
                    linewidth=1.2,
                    alpha=0.35,
                    label=(
                        "LiDAR clustered obstacle size (pc1 × pc2)"
                        if cluster_index == 0
                        else None
                    ),
                    zorder=7,
                )
            )
            ax.annotate(
                (
                    f"pc1={pc1_m:.2f} m\n"
                    f"pc2={pc2_m:.2f} m\n"
                    f"v_real={format_speed(real_speed)}\n"
                    f"v_pred={format_speed(predicted_speed)}\n"
                    f"real_dir={format_heading_deg(real_heading_deg)}\n"
                    f"plan={format_heading_deg(planned_heading_deg)}"
                ),
                xy=(centre_ne[1], centre_ne[0]),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=7,
                color="#7f0000",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.72,
                    "edgecolor": "none",
                },
                zorder=10,
            )
            if webots_position_ne is not None:
                has_webots_real_position = True
                ax.scatter(
                    [webots_position_ne[1]],
                    [webots_position_ne[0]],
                    marker="D",
                    s=60,
                    facecolors="#00c2c7",
                    edgecolors="black",
                    linewidth=0.6,
                    label="Webots true obstacle position" if cluster_index == 0 else None,
                    zorder=9,
                )
            if real_heading_vector is not None and webots_position_ne is not None:
                has_webots_real_direction = True
                real_heading_norm = float(np.linalg.norm(real_heading_vector))
                scaled_heading = real_heading_vector / real_heading_norm
                real_arrow_length_m = np.clip(
                    real_speed * 4.0 if np.isfinite(real_speed) else 1.0,
                    0.8,
                    1.8,
                )
                real_arrow_end_ne = (
                    webots_position_ne + scaled_heading * real_arrow_length_m
                )
                ax.annotate(
                    "",
                    xy=(real_arrow_end_ne[1], real_arrow_end_ne[0]),
                    xytext=(webots_position_ne[1], webots_position_ne[0]),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#00c2c7",
                        "linewidth": 1.8,
                    },
                    zorder=9,
                )
            if planned_direction is not None:
                norm = float(np.linalg.norm(planned_direction))
                scaled_direction = planned_direction / norm
                arrow_length_m = np.clip(
                    predicted_speed * 4.0 if np.isfinite(predicted_speed) else 1.0,
                    0.8,
                    1.8,
                )
                arrow_end_ne = centre_ne + scaled_direction * arrow_length_m
                ax.annotate(
                    "",
                    xy=(arrow_end_ne[1], arrow_end_ne[0]),
                    xytext=(centre_ne[1], centre_ne[0]),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#ff8c00",
                        "linewidth": 1.8,
                    },
                    zorder=9,
                )

    for track_index, track in enumerate(payload.get("tracks", [])):
        if not isinstance(track, dict):
            continue
        position = point(track.get("position_ne"))
        if position is not None:
            ax.scatter(
                [position[1]],
                [position[0]],
                marker="o",
                s=55,
                facecolors="none",
                edgecolors="#2ca02c",
                linewidth=1.4,
                label="Obstacle EKF position" if track_index == 0 else None,
                zorder=8,
            )
        prediction = straight_line_display_prediction(
            track,
            prediction_points(track),
            prediction_horizon_s,
        )
        if len(prediction) >= 2:
            ax.plot(
                prediction[:, 1],
                prediction[:, 0],
                color="#ff7f0e",
                linestyle="--",
                linewidth=2.0,
                marker=".",
                markersize=4,
                label=(
                    f"EKF predicted trajectory ({prediction_horizon_s:g} s)"
                    if track_index == 0
                    else None
                ),
                zorder=7,
            )
            ax.scatter(
                [prediction[-1, 1]],
                [prediction[-1, 0]],
                marker="X",
                s=65,
                color="#ff7f0e",
                edgecolor="black",
                linewidth=0.6,
                label=(
                    f"Obstacle position at +{prediction_horizon_s:g} s"
                    if track_index == 0
                    else None
                ),
                zorder=8,
            )

    apf = payload.get("apf", {})
    mode = apf.get("navigation_mode", "unknown") if isinstance(apf, dict) else "unknown"
    encounter = apf.get("encounter", "none") if isinstance(apf, dict) else "none"
    sample_time_s = snapshot["relative_time_s"]
    time_text = f"t={target_time_s:.1f} s"
    if abs(sample_time_s - target_time_s) >= 0.05:
        time_text += f" (nearest log sample {sample_time_s:.1f} s)"
    ax.set_title(
        f"APF trajectory snapshot at {time_text}\n"
        f"mode={mode}, encounter={encounter}; "
        f"{run_collision_outcome} (Webots ShipObstacle)"
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_xlim(east_min, east_max)
    ax.set_ylim(north_min, north_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if float(np.max(real_potential)) > 1e-9:
        handles.append(Line2D([0], [0], color="black", linewidth=1.0))
        labels.append("Measured-obstacle potential boundary")
    if float(np.max(virtual_potential)) > 1e-9:
        handles.append(Line2D([0], [0], color="#ff7f0e", linestyle="--", linewidth=1.0))
        labels.append("Predicted-obstacle potential boundary")
    if clusters:
        if has_webots_real_position:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    linestyle="none",
                    markerfacecolor="#00c2c7",
                    markeredgecolor="black",
                    markersize=7,
                )
            )
            labels.append("Webots true obstacle position")
        if has_webots_real_direction:
            handles.append(Line2D([0], [0], color="#00c2c7", linewidth=1.8))
            labels.append("Webots true obstacle heading")
        handles.append(Line2D([0], [0], color="#ff8c00", linewidth=1.8))
        labels.append("Obstacle planned direction")
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker=r"$\rightarrow$",
            linestyle="none",
            alpha=0.45,
        )
    )
    labels.append("Negative potential gradient")
    unique = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def generate_snapshots(
    run_dir,
    output_dir=None,
    interval_s=5.0,
    grid_size=240,
    k_goal=DEFAULT_K_GOAL,
    k_obstacle=DEFAULT_K_OBSTACLE,
    quiver_step=14,
    map_size_m=20.0,
):
    run_dir = Path(run_dir)
    snapshots = load_snapshots(run_dir)
    run_collision_outcome = collision_outcome_text(run_dir)
    selected_indices = select_snapshot_indices(snapshots, interval_s)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR / f"{run_dir.name}_apf_snapshots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_snapshot in output_dir.glob("apf_snapshot_*.png"):
        old_snapshot.unlink()
    bounds = run_bounds(snapshots, map_size_m)
    default_target_ne = point(snapshots[-1]["payload"].get("robot_pos"))
    trajectory_time_s, trajectory_ne_m = load_robot_trajectory(run_dir)

    outputs = []
    selected_targets = {
        index: selected_order * positive(interval_s, 5.0)
        for selected_order, index in enumerate(selected_indices)
    }
    for index, snapshot in enumerate(snapshots):
        if index not in selected_targets:
            continue
        target_time_s = selected_targets[index]
        sample_time_s = float(snapshot["time_s"])
        history_mask = trajectory_time_s <= sample_time_s
        if np.count_nonzero(history_mask) >= 2:
            robot_history = trajectory_ne_m[history_mask]
            robot_position = robot_history[-1]
        elif len(trajectory_ne_m):
            nearest_index = int(np.argmin(np.abs(trajectory_time_s - sample_time_s)))
            robot_position = trajectory_ne_m[nearest_index]
            robot_history = trajectory_ne_m[: nearest_index + 1]
        else:
            robot_position = point(snapshot["payload"].get("robot_pos"))
            if robot_position is None:
                continue
            robot_history = np.asarray([robot_position], dtype=float)
        output_path = output_dir / f"apf_snapshot_{target_time_s:06.1f}s.png"
        plot_snapshot(
            snapshot=snapshot,
            robot_history=robot_history,
            robot_position=robot_position,
            bounds=bounds,
            output_path=output_path,
            grid_size=max(int(grid_size), 80),
            target_time_s=target_time_s,
            default_target_ne=default_target_ne,
            k_goal=float(k_goal),
            k_obstacle=float(k_obstacle),
            quiver_step=quiver_step,
            run_collision_outcome=run_collision_outcome,
        )
        outputs.append(output_path)
    return outputs


def generate_latest_world_snapshots(
    logs_dir,
    output_dir=None,
    interval_s=5.0,
    grid_size=240,
    k_goal=DEFAULT_K_GOAL,
    k_obstacle=DEFAULT_K_OBSTACLE,
    quiver_step=14,
    map_size_m=20.0,
    combination="ekf_on_cluster_on",
):
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_BATCH_OUTPUT_DIR
    selections = load_latest_world_runs(logs_dir, combination=combination)
    if not selections:
        raise FileNotFoundError(
            f"No runs found in {logs_dir} with combination {combination}"
        )

    generated = {}
    for selection in selections:
        generated[selection.world_name] = generate_snapshots(
            run_dir=selection.run_dir,
            output_dir=world_output_dir(output_dir, selection.world_name),
            interval_s=interval_s,
            grid_size=grid_size,
            k_goal=k_goal,
            k_obstacle=k_obstacle,
            quiver_step=quiver_step,
            map_size_m=map_size_m,
        )
    return generated


def _self_check():
    assert switch_combination_name({"SwitchCombination": "ekf_on_cluster_on"}) == "ekf_on_cluster_on"
    assert switch_combination_name({"EKFPredictionEnabled": "1", "ClusterAPFEnabled": "0"}) == "ekf_on_cluster_off"
    assert world_output_dir(Path("x"), "mr_webots_head_on_small_ship.wbt").as_posix().endswith(
        "x/mr_webots_head_on_small_ship"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate APF snapshots containing the earth-frame LiDAR cloud, "
            "OS history, obstacle positions, and EKF predictions."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing obstacle_*.json; defaults to latest run.",
    )
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--k-goal", type=float, default=DEFAULT_K_GOAL)
    parser.add_argument("--k-obstacle", type=float, default=DEFAULT_K_OBSTACLE)
    parser.add_argument("--quiver-step", type=int, default=14)
    parser.add_argument(
        "--map-size",
        type=float,
        default=20.0,
        help="Square map width and height in metres.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run a minimal internal sanity check and exit.",
    )
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        print("self-check passed")
        return

    if args.run_dir is not None:
        run_dir = resolve_run_dir(args.run_dir)
        outputs = generate_snapshots(
            run_dir=run_dir,
            output_dir=args.output_dir,
            interval_s=args.interval,
            grid_size=args.grid_size,
            k_goal=args.k_goal,
            k_obstacle=args.k_obstacle,
            quiver_step=args.quiver_step,
            map_size_m=args.map_size,
        )
        print(f"Run: {run_dir}")
        print(f"Generated {len(outputs)} snapshots")
        for path in outputs:
            print(path)
        return

    generated = generate_latest_world_snapshots(
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        interval_s=args.interval,
        grid_size=args.grid_size,
        k_goal=args.k_goal,
        k_obstacle=args.k_obstacle,
        quiver_step=args.quiver_step,
        map_size_m=args.map_size,
    )
    print(f"Generated latest ekf_on_cluster_on snapshots for {len(generated)} worlds")
    for world_name, paths in generated.items():
        print(f"{Path(world_name).stem}: {len(paths)} snapshots")
        if paths:
            print(paths[0].parent)


if __name__ == "__main__":
    main()
