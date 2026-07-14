"""Plot Webots ship size reference lines and cluster PC dimensions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from plot_apf_snapshots import load_latest_world_runs
from plot_log_sources import resolve_run_dir
from plot_colreg_size_trajectories import (
    collect_grouped_runs,
    collect_snapshot_samples,
    merged_output_name,
    pair_large_small_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_OUTPUT_DIR = DEFAULT_LOGS_DIR / "generated_figures"
DEFAULT_BATCH_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "pc_size"
DEFAULT_SIZE_COMPARISON_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "pc_size_comparisons"

SMALL_SHIP_BOUNDS_M = (0.9, 0.32)
LARGE_SHIP_BOUNDS_M = (1.8, 0.528)


@dataclass
class ClusterRun:
    run_dir: Path
    webots_environment: str
    switch_combination: str
    time_s: np.ndarray
    pc1_m: np.ndarray
    pc2_m: np.ndarray


def positive_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0.0 else None


def normalise_webots_name(value):
    text = str(value or "").strip()
    if text.lower().endswith(".wbt"):
        text = Path(text).stem
    text = text.lower().replace("-", "_").replace(" ", "_")
    token = "".join(char for char in text if char.isalnum() or char == "_")
    return token or "webots_unknown"


def switch_from_flags(ekf_enabled, size_enabled):
    if ekf_enabled is None or size_enabled is None:
        return None
    return (
        f"ekf_{'on' if ekf_enabled else 'off'}_"
        f"cluster_{'on' if size_enabled else 'off'}"
    )


def parse_bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return None


def find_primary_csv(run_dir):
    candidates = [
        path
        for path in Path(run_dir).glob("log_*.csv")
        if not path.name.endswith("_pseudo_aruco.csv")
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def read_log_metadata(log_path):
    import csv

    with log_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")
        for row in reader:
            webots_environment = normalise_webots_name(row.get("WebotsEnvironment"))
            switch_combination = str(row.get("SwitchCombination", "")).strip()
            if switch_combination not in {"ekf_on_cluster_on", "ekf_on_cluster_off", "ekf_off_cluster_on", "ekf_off_cluster_off"}:
                switch_combination = switch_from_flags(
                    parse_bool(row.get("EKFPredictionEnabled")),
                    parse_bool(
                        row.get("ClusterSizeAPFEnabled")
                        if "ClusterSizeAPFEnabled" in row
                        else row.get("ClusterAPFEnabled")
                    ),
                )
            if (
                webots_environment not in {"webots_unknown", "not_webots"}
                and switch_combination is not None
            ):
                return webots_environment, switch_combination
    raise ValueError("No Webots/EKF/size metadata in CSV")


def load_cluster_dimensions(run_dir):
    import csv

    samples = []
    for path in sorted(Path(run_dir).glob("obstacle_*.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue

        try:
            time_s = float(payload.get("t"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(time_s):
            continue

        clusters = payload.get("clusters", [])
        if not isinstance(clusters, list):
            continue
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            pc1_m = positive_float(cluster.get("pc1_m"))
            pc2_m = positive_float(cluster.get("pc2_m"))
            if pc1_m is None or pc2_m is None:
                continue
            track_id = cluster.get("track_id")
            if track_id is None:
                track_id = f"cluster-{cluster.get('label', 'unknown')}"
            samples.append(
                {
                    "time_s": time_s,
                    "track_id": str(track_id),
                    "pc1_m": pc1_m,
                    "pc2_m": pc2_m,
                }
            )

    if not samples:
        raise ValueError(f"No valid cluster pc1_m/pc2_m values found in {run_dir}")

    samples.sort(key=lambda sample: sample["time_s"])
    first_time_s = samples[0]["time_s"]
    grouped = defaultdict(list)
    for sample in samples:
        sample["time_s"] -= first_time_s
        grouped[sample["track_id"]].append(sample)
    return dict(grouped)


def plot_cluster_dimensions(run_dir, output_path=None):
    """Plot cluster PC1/PC2 values with Webots large/small ship size lines."""
    run_dir = Path(run_dir)
    grouped = load_cluster_dimensions(run_dir)
    all_samples = [sample for track_samples in grouped.values() for sample in track_samples]

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{run_dir.name}_pc1_pc2_over_time.png"
    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for track_index, track_id in enumerate(
        sorted(grouped, key=lambda value: (not value.isdigit(), value))
    ):
        samples = grouped[track_id]
        time_s = [sample["time_s"] for sample in samples]
        pc1_m = [sample["pc1_m"] for sample in samples]
        pc2_m = [sample["pc2_m"] for sample in samples]
        ax.plot(
            time_s,
            pc1_m,
            color="#0072B2",
            linewidth=1.4,
            marker="o",
            markersize=2.5,
            alpha=0.85,
            label="PC1" if track_index == 0 else "_nolegend_",
        )
        ax.plot(
            time_s,
            pc2_m,
            color="#D55E00",
            linewidth=1.4,
            marker="o",
            markersize=2.5,
            alpha=0.85,
            label="PC2" if track_index == 0 else "_nolegend_",
        )

    small_length_m, small_width_m = SMALL_SHIP_BOUNDS_M
    large_length_m, large_width_m = LARGE_SHIP_BOUNDS_M
    for value, label, color, linestyle in [
        (small_length_m, "Small ship length", "#2ca02c", "--"),
        (small_width_m, "Small ship width", "#2ca02c", ":"),
        (large_length_m, "Large ship length", "#9467bd", "--"),
        (large_width_m, "Large ship width", "#9467bd", ":"),
    ]:
        ax.axhline(
            value,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )

    ax.set_title(f"Cluster principal dimensions over time\n{run_dir.name}")
    ax.set_xlabel("Time from first cluster sample (s)")
    ax.set_ylabel("Size (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path, len(all_samples)


def plot_latest_world_cluster_dimensions(logs_dir, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_BATCH_OUTPUT_DIR
    selections = load_latest_world_runs(logs_dir, combination="ekf_on_cluster_on")
    if not selections:
        raise FileNotFoundError(
            f"No runs found in {logs_dir} with combination ekf_on_cluster_on"
        )

    run_dirs = [selection.run_dir for selection in selections]
    records = collect_grouped_runs(logs_dir, run_dirs=run_dirs)
    pairs = pair_large_small_records(records)
    if not pairs:
        raise ValueError(
            "No matched large/small Webots pairs were found in latest ekf_on_cluster_on runs"
        )

    outputs = []
    for pair in pairs:
        output_path = Path(output_dir) / f"{merged_output_name(pair)}_pc1_pc2.png"
        outputs.append((plot_large_small_pc_pair(pair, output_path), pair))
    return outputs


def _self_check():
    assert normalise_webots_name("mr_webots_head_on_small_ship.wbt") == "mr_webots_head_on_small_ship"
    assert switch_from_flags(True, True) == "ekf_on_cluster_on"
    assert merged_output_name(
        [
            type("R", (), {"webots_environment": "mr_webots_head_on_small_ship"})(),
            type("R", (), {"webots_environment": "mr_webots_head_on_large_ship"})(),
        ]
    ).endswith(
        "mr_webots_head_on_small_ship__mr_webots_head_on_large_ship"
    )


def plot_large_small_pc_pair(records, output_path):
    """Plot PC1 and PC2 time series for one matched large/small Webots pair."""
    colors = {"Small": "#0072B2", "Large": "#D55E00"}
    scenario = (
        records[0].webots_pair_key.removeprefix("mr_webots_")
        .replace("_size_ship", "")
        .replace("_", " ")
        .title()
    )
    fig, ax = plt.subplots(figsize=(10, 5.8))

    for record in sorted(records, key=lambda item: item.size_label, reverse=True):
        samples = collect_snapshot_samples(record.run_dir)
        if not samples:
            continue
        first_time_s = samples[0].time_s
        time_s = [sample.time_s - first_time_s for sample in samples]
        color = colors[record.size_label]
        ax.plot(
            time_s,
            [sample.obstacle_pc1_m for sample in samples],
            color=color,
            linewidth=1.8,
            label=f"{record.size_label} ship PC1",
        )
        ax.plot(
            time_s,
            [sample.obstacle_pc2_m for sample in samples],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"{record.size_label} ship PC2",
        )

    ax.axhline(
        SMALL_SHIP_BOUNDS_M[0],
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.4,
        label="Small ship length",
    )
    ax.axhline(
        SMALL_SHIP_BOUNDS_M[1],
        color="#2ca02c",
        linestyle=":",
        linewidth=1.4,
        label="Small ship width",
    )
    ax.axhline(
        LARGE_SHIP_BOUNDS_M[0],
        color="#9467bd",
        linestyle="-.",
        linewidth=1.4,
        label="Large ship length",
    )
    ax.axhline(
        LARGE_SHIP_BOUNDS_M[1],
        color="#9467bd",
        linestyle=":",
        linewidth=1.4,
        label="Large ship width",
    )

    ax.set_title(
        "Principal Component Size over Time\n"
        f"{scenario}: Large vs Small"
    )
    ax.set_xlabel("Time from first obstacle sample (s)")
    ax.set_ylabel("Size (m)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_large_small_pc_comparisons(logs_dir, output_dir, run_dirs=None):
    records = collect_grouped_runs(logs_dir, run_dirs=run_dirs)
    pairs = pair_large_small_records(records)
    if not pairs:
        raise ValueError(
            "No matched large/small Webots runs with EKF and size both enabled"
        )

    outputs = []
    for pair in pairs:
        output_path = Path(output_dir) / f"{merged_output_name(pair)}_pc1_pc2.png"
        outputs.append((plot_large_small_pc_pair(pair, output_path), pair))
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot cluster PC dimensions with Webots large/small ship size lines."
        )
    )
    parser.add_argument("--run-dir", type=Path, help="Run directory to plot.")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run a minimal internal sanity check and exit.",
    )
    parser.add_argument(
        "--size-comparison",
        action="store_true",
        help="Compare PC1/PC2 for matched large/small EKF-on and size-on runs.",
    )
    parser.add_argument(
        "--size-comparison-output-dir",
        type=Path,
        default=DEFAULT_SIZE_COMPARISON_OUTPUT_DIR,
        help="Output directory for large/small PC1/PC2 figures.",
    )
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        print("self-check passed")
        return

    if args.size_comparison:
        outputs = plot_large_small_pc_comparisons(
            args.logs_dir,
            args.size_comparison_output_dir,
            run_dirs=[args.run_dir] if args.run_dir else None,
        )
        for output_path, pair in outputs:
            print(f"Figure: {output_path}")
            for record in pair:
                print(f"  {record.size_label}: {record.run_dir.name}")
        return

    if not args.run_dir:
        outputs = plot_latest_world_cluster_dimensions(args.logs_dir, args.output)
        print(f"Generated latest ekf_on_cluster_on PC-size comparison figures for {len(outputs)} COLREG pairs")
        for figure_path, pair in outputs:
            for record in sorted(pair, key=lambda item: item.size_label):
                print(f"{record.size_label}: {record.run_dir}")
            print(f"Figure: {figure_path}")
        return

    run_dir = resolve_run_dir(args.run_dir)
    output_path, sample_count = plot_cluster_dimensions(run_dir, args.output)
    print(f"Run: {run_dir}")
    print(f"Samples: {sample_count}")
    print(f"Figure: {output_path}")


if __name__ == "__main__":
    main()
