"""Overlay avoidance trajectories for large and small obstacle ships.

The script scans recent valid ``run_*`` logs, estimates one obstacle-size
metric per run from the active obstacle snapshots, and automatically labels
each trajectory as large or small from the log data before plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from plot_log_sources import list_resolved_run_dirs, resolve_run_dir
from webots_collision import collision_detected, collision_outcome_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_OUTPUT_DIR = DEFAULT_LOGS_DIR / "generated_figures" / "colreg_size_trajectories"
DEFAULT_LATEST_LIMIT = 0

TIME_COLUMNS = ("TimeFromStart(s)", "TimeFromStart", "elapsed [s]")
OWN_NORTH_COLUMNS = ("North(m)", "North", "x [m]")
OWN_EAST_COLUMNS = ("East(m)", "East", "y [m]")
ARUCO_NORTH_COLUMNS = ("ARUCOSensedNorth(m)", "ARUCOSensedNorth")
ARUCO_EAST_COLUMNS = ("ARUCOSensedEast(m)", "ARUCOSensedEast")
SWITCH_FLAGS = {
    "ekf_on_cluster_on": (True, True),
    "ekf_on_cluster_off": (True, False),
    "ekf_off_cluster_on": (False, True),
    "ekf_off_cluster_off": (False, False),
}


@dataclass
class SnapshotSample:
    time_s: float
    obstacle_pc1_m: float
    obstacle_pc2_m: float
    equivalent_radius_m: float


@dataclass
class RunRecord:
    run_dir: Path
    log_path: Path
    ekf_enabled: bool
    size_enabled: bool
    webots_environment: str
    webots_pair_key: str
    trajectory_time_s: np.ndarray
    trajectory_ne_m: np.ndarray
    active_window_s: tuple[float, float]
    median_pc1_m: float
    median_pc2_m: float
    median_equivalent_radius_m: float
    size_metric_m: float
    collision_detected: bool | None
    size_label: str = ""


def parse_float(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def positive_float(value):
    value = parse_float(value)
    return value if np.isfinite(value) and value > 0.0 else np.nan


def first_existing(row, names):
    for name in names:
        if name in row:
            return row[name]
    return None


def parse_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return None


def normalise_webots_name(value):
    if value is None:
        return "webots_unknown"
    text = str(value).strip()
    if not text:
        return "webots_unknown"
    if text.lower().endswith(".wbt"):
        text = Path(text).stem
    text = text.lower().replace("-", "_").replace(" ", "_")
    token = "".join(char for char in text if char.isalnum() or char == "_")
    return token or "webots_unknown"


def webots_size_label(webots_environment):
    tokens = webots_environment.split("_")
    if "large" in tokens:
        return "Large"
    if "small" in tokens:
        return "Small"
    return None


def webots_pair_key(webots_environment):
    label = webots_size_label(webots_environment)
    if label is None:
        return None
    return "_".join(
        "size" if token in {"large", "small"} else token
        for token in webots_environment.split("_")
    )


def flags_from_switch_name(value):
    text = str(value or "").strip()
    return SWITCH_FLAGS.get(text)


def read_csv_metadata(log_path):
    webots_environment = None
    ekf_enabled = None
    size_enabled = None

    with log_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return ekf_enabled, size_enabled, webots_environment

        for row in reader:
            if webots_environment is None and "WebotsEnvironment" in row:
                name = normalise_webots_name(row.get("WebotsEnvironment"))
                if name not in {"webots_unknown", "not_webots"}:
                    webots_environment = name

            switch_flags = flags_from_switch_name(row.get("SwitchCombination"))
            if switch_flags is not None:
                ekf_enabled, size_enabled = switch_flags

            if ekf_enabled is None:
                ekf_enabled = parse_bool(row.get("EKFPredictionEnabled"))
            if size_enabled is None:
                size_enabled = parse_bool(
                    row.get("ClusterSizeAPFEnabled")
                    if "ClusterSizeAPFEnabled" in row
                    else row.get("ClusterAPFEnabled")
                )

            if (
                webots_environment is not None
                and ekf_enabled is not None
                and size_enabled is not None
            ):
                break

    return ekf_enabled, size_enabled, webots_environment


def read_json_metadata(run_dir):
    snapshot_paths = sorted(run_dir.glob("obstacle_*.json"))
    if not snapshot_paths:
        return None, None, None

    sample_indices = sorted({0, len(snapshot_paths) // 2, len(snapshot_paths) - 1})
    flags = []
    names = []
    for index in sample_indices:
        try:
            with snapshot_paths[index].open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue

        for key in ("webots_environment", "WebotsEnvironment", "world", "world_file"):
            name = normalise_webots_name(payload.get(key))
            if name not in {"webots_unknown", "not_webots"}:
                names.append(name)

        settings = payload.get("apf_settings", {})
        if not isinstance(settings, dict):
            continue
        if (
            "obstacle_ekf_prediction_enabled" in settings
            and "cluster_range_enabled" in settings
        ):
            flags.append(
                (
                    bool(settings["obstacle_ekf_prediction_enabled"]),
                    bool(settings["cluster_range_enabled"]),
                )
            )

    if flags and any(flag != flags[0] for flag in flags[1:]):
        raise ValueError("Switch metadata changes within the run")
    webots_environment = max(set(names), key=names.count) if names else None
    return (
        flags[0][0] if flags else None,
        flags[0][1] if flags else None,
        webots_environment,
    )


def read_run_metadata(run_dir, log_path):
    ekf_enabled, size_enabled, webots_environment = read_csv_metadata(log_path)
    json_ekf, json_size, json_webots = read_json_metadata(run_dir)
    return (
        ekf_enabled if ekf_enabled is not None else json_ekf,
        size_enabled if size_enabled is not None else json_size,
        webots_environment or json_webots or "webots_unknown",
    )


def point(value):
    if value is None:
        return None
    try:
        point_array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if point_array.size < 2:
        return None
    point_array = point_array[:2]
    return point_array if np.isfinite(point_array).all() else None


def first_point(*values):
    for value in values:
        parsed = point(value)
        if parsed is not None:
            return parsed
    return None


def find_primary_csv(run_dir):
    candidates = [
        path
        for path in run_dir.glob("log_*.csv")
        if not path.name.endswith("_pseudo_aruco.csv")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_pseudo_aruco_csv(run_dir):
    candidates = list(run_dir.glob("log_*_pseudo_aruco.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_trajectory_csv(log_path, north_columns, east_columns):
    times = []
    points = []
    first_time_s = np.nan

    with log_path.open(newline="", encoding="utf-8-sig") as stream:
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
            if not np.isfinite(first_time_s):
                first_time_s = float(time_s)
            times.append(float(time_s))
            points.append([north_m, east_m])

    if not times:
        raise ValueError(f"No valid trajectory samples in {log_path}")

    time_array = np.asarray(times, dtype=float)
    trajectory_ne_m = np.asarray(points, dtype=float)
    order = np.argsort(time_array)
    time_array = time_array[order]
    trajectory_ne_m = trajectory_ne_m[order]
    time_array -= float(first_time_s) if np.isfinite(first_time_s) else time_array[0]
    return time_array, trajectory_ne_m


def read_trajectory(log_path, run_dir):
    pseudo_aruco_path = find_pseudo_aruco_csv(run_dir)
    if pseudo_aruco_path is not None:
        try:
            return read_trajectory_csv(
                pseudo_aruco_path,
                OWN_NORTH_COLUMNS,
                OWN_EAST_COLUMNS,
            )
        except ValueError:
            pass

    try:
        return read_trajectory_csv(
            log_path,
            ARUCO_NORTH_COLUMNS,
            ARUCO_EAST_COLUMNS,
        )
    except ValueError:
        return read_trajectory_csv(
            log_path,
            OWN_NORTH_COLUMNS,
            OWN_EAST_COLUMNS,
        )


def obstacle_candidates(payload):
    candidates = []
    for key in ("tracks", "clusters"):
        entries = payload.get(key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            pc1_m = positive_float(entry.get("pc1_m"))
            if not np.isfinite(pc1_m):
                continue
            pc2_m = positive_float(entry.get("pc2_m"))
            equivalent_radius_m = positive_float(entry.get("equivalent_radius_m"))
            centre_ne = first_point(
                entry.get("position_ne"),
                entry.get("centre_ne"),
                entry.get("measurement_centre_ne"),
                entry.get("virtual_position_ne"),
            )
            candidates.append(
                {
                    "pc1_m": pc1_m,
                    "pc2_m": pc2_m if np.isfinite(pc2_m) else np.nan,
                    "equivalent_radius_m": (
                        equivalent_radius_m if np.isfinite(equivalent_radius_m) else np.nan
                    ),
                    "centre_ne": centre_ne,
                }
            )
    return candidates


def choose_active_obstacle(payload):
    robot_pos = point(payload.get("robot_pos"))
    candidates = obstacle_candidates(payload)
    if not candidates:
        return None
    if robot_pos is None:
        return candidates[0]

    def distance(candidate):
        centre_ne = candidate.get("centre_ne")
        if centre_ne is None:
            return float("inf")
        return float(np.linalg.norm(robot_pos - centre_ne))

    return min(candidates, key=distance)


def collect_snapshot_samples(run_dir):
    samples = []
    for path in sorted(run_dir.glob("obstacle_*.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue

        time_s = parse_float(payload.get("t"))
        if not np.isfinite(time_s):
            continue

        obstacle = choose_active_obstacle(payload)
        if obstacle is None:
            continue

        samples.append(
            SnapshotSample(
                time_s=float(time_s),
                obstacle_pc1_m=float(obstacle["pc1_m"]),
                obstacle_pc2_m=float(obstacle["pc2_m"]),
                equivalent_radius_m=float(obstacle["equivalent_radius_m"]),
            )
        )

    return samples
def median_of(values):
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return np.nan
    return float(np.median(np.asarray(finite, dtype=float)))


def build_run_record(run_dir):
    log_path = find_primary_csv(run_dir)
    if log_path is None:
        return None

    ekf_enabled, size_enabled, webots_environment = read_run_metadata(run_dir, log_path)
    webots_pair = webots_pair_key(webots_environment)
    size_label = webots_size_label(webots_environment)
    if ekf_enabled is not True or size_enabled is not True:
        return None
    if webots_pair is None or size_label is None:
        return None

    samples = collect_snapshot_samples(run_dir)
    if len(samples) < 3:
        return None

    trajectory_time_s, trajectory_ne_m = read_trajectory(log_path, run_dir)
    active_times = np.asarray([sample.time_s for sample in samples], dtype=float)
    active_window_s = (float(np.min(active_times)), float(np.max(active_times)))

    median_pc1_m = median_of(sample.obstacle_pc1_m for sample in samples)
    median_pc2_m = median_of(sample.obstacle_pc2_m for sample in samples)
    median_equivalent_radius_m = median_of(
        sample.equivalent_radius_m for sample in samples
    )
    size_metric_m = (
        median_pc1_m
        if np.isfinite(median_pc1_m)
        else median_equivalent_radius_m
    )
    if not np.isfinite(size_metric_m):
        return None

    return RunRecord(
        run_dir=run_dir,
        log_path=log_path,
        ekf_enabled=bool(ekf_enabled),
        size_enabled=bool(size_enabled),
        webots_environment=webots_environment,
        webots_pair_key=webots_pair,
        trajectory_time_s=trajectory_time_s,
        trajectory_ne_m=trajectory_ne_m,
        active_window_s=active_window_s,
        median_pc1_m=median_pc1_m,
        median_pc2_m=median_pc2_m,
        median_equivalent_radius_m=median_equivalent_radius_m,
        size_metric_m=float(size_metric_m),
        collision_detected=collision_detected(run_dir),
        size_label=size_label,
    )


def latest_run_dirs(logs_dir, latest_limit=DEFAULT_LATEST_LIMIT):
    return [
        path
        for path in list_resolved_run_dirs(logs_dir)
        if any(path.glob("obstacle_*.json"))
    ]


def selected_run_dirs(logs_dir, run_dirs):
    resolved = []
    for run_dir in run_dirs:
        candidate = Path(run_dir)
        if candidate.is_absolute() or candidate.exists():
            resolved.append(resolve_run_dir(candidate))
            continue
        resolved.append(resolve_run_dir(Path(logs_dir) / candidate))
    return resolved


def segment_for_plot(record, padding_s):
    start_s, end_s = record.active_window_s
    lower_s = max(record.trajectory_time_s[0], start_s - padding_s)
    upper_s = min(record.trajectory_time_s[-1], end_s + padding_s)
    mask = (
        (record.trajectory_time_s >= lower_s)
        & (record.trajectory_time_s <= upper_s)
    )
    if np.count_nonzero(mask) < 2:
        return record.trajectory_ne_m
    return record.trajectory_ne_m[mask]


def merged_output_name(records):
    merged = "__".join(record.webots_environment for record in records)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", merged).strip("_")
    return safe_name or "large_vs_small"


def plot_group(records, output_path, padding_s):
    colors = {"Small": "#0072B2", "Large": "#D55E00"}
    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    for record in sorted(records, key=lambda item: (item.size_label, item.size_metric_m, item.run_dir.name)):
        segment_ne_m = segment_for_plot(record, padding_s)
        label = (
            f"{record.size_label} {record.run_dir.name} "
            f"(pc1={record.median_pc1_m:.2f} m; "
            f"{collision_outcome_text(record.run_dir)})"
        )
        color = colors.get(record.size_label, "#333333")
        ax.plot(
            segment_ne_m[:, 1],
            segment_ne_m[:, 0],
            linewidth=2.0,
            color=color,
            alpha=0.88,
            label=label,
        )
        ax.scatter(
            segment_ne_m[0, 1],
            segment_ne_m[0, 0],
            color=color,
            marker="o",
            s=22,
            alpha=0.8,
        )
        ax.scatter(
            segment_ne_m[-1, 1],
            segment_ne_m[-1, 0],
            color=color,
            marker="x",
            s=42,
            alpha=0.9,
        )
        ax.annotate(
            record.size_label,
            xy=(segment_ne_m[-1, 1], segment_ne_m[-1, 0]),
            xytext=(6, 6),
            textcoords="offset points",
            color=color,
            fontsize=9,
            weight="bold",
        )

    ax.set_title(
        "Avoidance Trajectory Comparison\n"
        f"{records[0].webots_pair_key.replace('_', ' ').title()}: Large vs Small\n"
        "Collision from Webots ShipObstacle contact sensor"
    )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def collect_grouped_runs(
    logs_dir,
    latest_limit=DEFAULT_LATEST_LIMIT,
    run_dirs=None,
):
    records = []
    collected = 0
    source_run_dirs = (
        selected_run_dirs(logs_dir, run_dirs)
        if run_dirs
        else latest_run_dirs(logs_dir, latest_limit=latest_limit)
    )
    for run_dir in source_run_dirs:
        try:
            record = build_run_record(run_dir)
        except (OSError, ValueError):
            continue
        if record is None:
            continue
        records.append(record)
        collected += 1
        if (
            not run_dirs
            and latest_limit is not None
            and latest_limit > 0
            and collected >= int(latest_limit)
        ):
            break
    return records


def pair_large_small_records(records):
    grouped = {}
    for record in sorted(records, key=lambda item: item.run_dir.name, reverse=True):
        group = grouped.setdefault(record.webots_pair_key, {})
        group.setdefault(record.size_label, record)
    return [
        [group["Small"], group["Large"]]
        for _, group in sorted(grouped.items())
        if "Small" in group and "Large" in group
    ]


def plot_colreg_size_trajectories(
    logs_dir,
    output_dir,
    padding_s=2.0,
    latest_limit=DEFAULT_LATEST_LIMIT,
    run_dirs=None,
):
    records = collect_grouped_runs(
        logs_dir,
        latest_limit=latest_limit,
        run_dirs=run_dirs,
    )
    if not records:
        raise FileNotFoundError(
            "No EKF-on and size-on runs with usable obstacle snapshots were found"
        )

    pairs = pair_large_small_records(records)
    if not pairs:
        raise ValueError(
            "No large/small Webots pairs were found with EKF and size both enabled"
        )

    outputs = []
    for pair in pairs:
        output_path = output_dir / f"{merged_output_name(pair)}.png"
        plot_group(pair, output_path, padding_s=padding_s)
        outputs.append((output_path, pair))
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay robot avoidance trajectories under different obstacle-ship "
            "sizes, then label each trajectory as a large-ship or small-ship case."
        )
    )
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--padding-s",
        type=float,
        default=2.0,
        help="Trajectory padding before and after the active obstacle window.",
    )
    parser.add_argument(
        "--latest-limit",
        type=int,
        default=DEFAULT_LATEST_LIMIT,
        help="Only scan the latest N run_* directories (0 means all; default: 0).",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        help="Explicit run_* directory to include. Repeat for multiple runs.",
    )
    args = parser.parse_args()

    outputs = plot_colreg_size_trajectories(
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        padding_s=float(args.padding_s),
        latest_limit=args.latest_limit,
        run_dirs=args.run_dirs,
    )

    for output_path, records in outputs:
        for record in sorted(records, key=lambda item: (item.size_label, item.size_metric_m)):
            print(
                "  "
                f"{record.size_label} {record.run_dir.name} "
                f"pc1={record.median_pc1_m:.3f}m "
                f"pc2={record.median_pc2_m:.3f}m "
                f"radius={record.median_equivalent_radius_m:.3f}m"
            )
        print(f"  Figure: {output_path}")


if __name__ == "__main__":
    main()
