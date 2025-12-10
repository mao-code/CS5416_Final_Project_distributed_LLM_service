#!/usr/bin/env python3
"""
Plot latency and throughput trends from the timing CSVs in output/.

Creates three figures:
- output/timing_overall_plots.png       (system-level)
- output/timing_per_node_plots.png      (per-node)
- output/timing_per_stage_plots.png     (per-stage)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    output_dir = Path("output")

    overall_path = output_dir / "timing_overall.csv"
    per_node_path = output_dir / "timing_per_node.csv"
    per_stage_path = output_dir / "timing_per_stage.csv"

    _plot_overall(overall_path, output_dir / "timing_overall_plots.png")
    _plot_grouped(
        per_node_path,
        output_dir / "timing_per_node_plots.png",
        group_col="node_number",
        group_label_fmt="node={}",
        latency_col="avg_node_latency_sec",
        throughput_col="avg_node_throughput_rps",
        title_prefix="Per-node",
    )
    _plot_grouped(
        per_stage_path,
        output_dir / "timing_per_stage_plots.png",
        group_col="stage",
        group_label_fmt="{}",
        latency_col="avg_stage_latency_sec",
        throughput_col="avg_stage_throughput_rps",
        title_prefix="Per-stage",
    )


def _read_with_load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path} to exist")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty")
    df["request_load_rps"] = df["num_requests"] / df["interval_sec"]
    return df


def _plot_overall(csv_path: Path, output_path: Path) -> None:
    df = _read_with_load(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.flatten()

    _plot_vs_batch(
        axes[0],
        df,
        metric="avg_system_latency_sec",
        ylabel="Avg system latency (s)",
        title="Latency vs. batch size (grouped by request load)",
        legend_label="{} req/s",
        group_col="request_load_rps",
    )
    _plot_vs_batch(
        axes[1],
        df,
        metric="avg_system_throughput_rps",
        ylabel="Avg system throughput (req/s)",
        title="Throughput vs. batch size (grouped by request load)",
        legend_label="{} req/s",
        group_col="request_load_rps",
    )
    _plot_vs_load(
        axes[2],
        df,
        metric="avg_system_latency_sec",
        ylabel="Avg system latency (s)",
        title="Latency vs. request load (grouped by batch size)",
        legend_label="batch={}",
        group_col="batch_size",
    )
    _plot_vs_load(
        axes[3],
        df,
        metric="avg_system_throughput_rps",
        ylabel="Avg system throughput (req/s)",
        title="Throughput vs. request load (grouped by batch size)",
        legend_label="batch={}",
        group_col="batch_size",
    )

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")


def _plot_grouped(
    csv_path: Path,
    output_path: Path,
    *,
    group_col: str,
    group_label_fmt: str,
    latency_col: str,
    throughput_col: str,
    title_prefix: str,
) -> None:
    df = _read_with_load(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.flatten()

    _plot_vs_batch(
        axes[0],
        df,
        metric=latency_col,
        ylabel="Avg latency (s)",
        title=f"{title_prefix} latency vs. batch size",
        legend_label=group_label_fmt,
        group_col=group_col,
    )
    _plot_vs_batch(
        axes[1],
        df,
        metric=throughput_col,
        ylabel="Avg throughput (req/s)",
        title=f"{title_prefix} throughput vs. batch size",
        legend_label=group_label_fmt,
        group_col=group_col,
    )
    _plot_vs_load(
        axes[2],
        df,
        metric=latency_col,
        ylabel="Avg latency (s)",
        title=f"{title_prefix} latency vs. request load",
        legend_label=group_label_fmt,
        group_col=group_col,
    )
    _plot_vs_load(
        axes[3],
        df,
        metric=throughput_col,
        ylabel="Avg throughput (req/s)",
        title=f"{title_prefix} throughput vs. request load",
        legend_label=group_label_fmt,
        group_col=group_col,
    )

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")


def _plot_vs_batch(
    ax,
    df,
    *,
    metric: str,
    ylabel: str,
    title: str,
    legend_label: str,
    group_col: str,
) -> None:
    """Draw metric against batch size with curves per grouping."""
    for key, group in df.groupby(group_col):
        group_sorted = group.sort_values("batch_size")
        ax.plot(
            group_sorted["batch_size"],
            group_sorted[metric],
            marker="o",
            label=legend_label.format(key),
        )
    ax.set_xticks(sorted(df["batch_size"].unique()))
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _plot_vs_load(
    ax,
    df,
    *,
    metric: str,
    ylabel: str,
    title: str,
    legend_label: str,
    group_col: str,
) -> None:
    """Draw metric against request load with curves per grouping."""
    for key, group in df.groupby(group_col):
        group_sorted = group.sort_values("request_load_rps")
        ax.plot(
            group_sorted["request_load_rps"],
            group_sorted[metric],
            marker="o",
            label=legend_label.format(key),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Request load (req/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


if __name__ == "__main__":
    main()
