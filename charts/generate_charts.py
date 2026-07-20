#!/usr/bin/env python3
"""charts/generate_charts.py — Publication-quality chart generator for benchmark results.

Reads benchmark, speculative comparison, and quantization comparison output
files and produces ten standardized charts.

Supported inputs (auto-detected by filename):
    benchmark_results*.csv      → latency, ttft, tokens_per_second, cpu,
                                   memory, duration, success_rate
    speculative_results.csv     → speedup
    quantization_results.csv    → tokens_per_second (per model)
    leaderboard.json            → leaderboard
    summary.json                → radar

Usage:
    python -m charts.generate_charts \\
        --input results/ \\
        --output charts/

    python -m charts.generate_charts \\
        --input results/benchmark_results.csv results/leaderboard.json \\
        --output charts/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

DPI = 300
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 6
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0",
          "#607D8B", "#00BCD4", "#795548", "#E91E63", "#3F51B5"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


class ChartGenerator:
    """Loads benchmark data and generates all applicable charts."""

    def __init__(self, input_path: Path, output_dir: Path) -> None:
        """Initialize the chart generator.

        Args:
            input_path: Path to a directory of result files, or a single file.
            output_dir: Directory to write chart PNGs into.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.benchmark_df: pd.DataFrame | None = None
        self.speculative_df: pd.DataFrame | None = None
        self.quantization_df: pd.DataFrame | None = None
        self.leaderboard: dict[str, Any] | None = None
        self.summary: dict[str, Any] | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Loaders
    # ──────────────────────────────────────────────────────────────────────────

    def load_csv(self, path: Path) -> pd.DataFrame | None:
        """Load a CSV file into a DataFrame.

        Args:
            path: Path to the CSV file.

        Returns:
            DataFrame, or None if the file is missing/empty/unparseable.
        """
        if not path.exists():
            logger.warning("CSV not found: %s", path)
            return None
        try:
            df = pd.read_csv(path)
            if df.empty:
                logger.warning("CSV is empty: %s", path)
                return None
            logger.info("Loaded %s (%d rows, %d cols)", path.name, len(df), len(df.columns))
            return df
        except pd.errors.EmptyDataError:
            logger.warning("CSV has no data: %s", path)
            return None
        except Exception:
            logger.exception("Failed to parse CSV: %s", path)
            return None

    def load_json(self, path: Path) -> dict[str, Any] | None:
        """Load a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed dict, or None if the file is missing/invalid.
        """
        if not path.exists():
            logger.warning("JSON not found: %s", path)
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded %s", path.name)
            return data
        except Exception:
            logger.exception("Failed to parse JSON: %s", path)
            return None

    def discover_and_load(self) -> None:
        """Scan input path(s) and load all recognized result files."""
        if self.input_path.is_file():
            self._load_single_file(self.input_path)
        elif self.input_path.is_dir():
            d = self.input_path
            # Benchmark results (may be timestamped)
            for f in sorted(d.glob("benchmark_results*.csv")):
                self.benchmark_df = self.load_csv(f)
                break
            # Speculative
            spec_csv = d / "speculative_results.csv"
            self.speculative_df = self.load_csv(spec_csv)
            # Quantization
            quant_csv = d / "quantization_results.csv"
            self.quantization_df = self.load_csv(quant_csv)
            # Leaderboard JSON
            lb_json = d / "leaderboard.json"
            self.leaderboard = self.load_json(lb_json)
            # Summary JSON
            summary_json = d / "summary.json"
            self.summary = self.load_json(summary_json)
        else:
            logger.error("Input path does not exist: %s", self.input_path)

    def _load_single_file(self, path: Path) -> None:
        """Load a single file by detecting its type from the name."""
        name = path.name.lower()
        if "benchmark_results" in name and name.endswith(".csv"):
            self.benchmark_df = self.load_csv(path)
        elif "speculative_results" in name and name.endswith(".csv"):
            self.speculative_df = self.load_csv(path)
        elif "quantization_results" in name and name.endswith(".csv"):
            self.quantization_df = self.load_csv(path)
        elif name == "leaderboard.json":
            self.leaderboard = self.load_json(path)
        elif name == "summary.json":
            self.summary = self.load_json(path)
        else:
            logger.warning("Unrecognized file: %s", path.name)

    # ──────────────────────────────────────────────────────────────────────────
    # Chart generation helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _save(fig: plt.Figure, name: str, output_dir: Path) -> Path:
        """Save a figure to disk and close it.

        Args:
            fig: Matplotlib figure.
            name: Filename stem (without extension).
            output_dir: Output directory.

        Returns:
            Path to the saved PNG.
        """
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved chart: %s", path)
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Latency Comparison
    # ──────────────────────────────────────────────────────────────────────────

    def chart_latency(self) -> Path | None:
        """Bar chart of average latency per prompt.

        Returns:
            Path to saved chart, or None if data unavailable.
        """
        df = self.benchmark_df
        if df is None or "latency" not in df.columns or "prompt_id" not in df.columns:
            logger.info("Skipping latency chart — no benchmark data")
            return None

        ok = df[df["status"] == "success"] if "status" in df.columns else df
        if ok.empty:
            return None

        grouped = ok.groupby("prompt_id")["latency"].mean().sort_values()

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        bars = ax.bar(range(len(grouped)), grouped.values, color=COLORS[0], edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Prompt ID", fontsize=11)
        ax.set_ylabel("Average Latency (ms)", fontsize=11)
        ax.set_title("Inter-Token Latency by Prompt", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        return self._save(fig, "latency", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. TTFT Comparison
    # ──────────────────────────────────────────────────────────────────────────

    def chart_ttft(self) -> Path | None:
        """Bar chart of average time-to-first-token per prompt."""
        df = self.benchmark_df
        if df is None or "ttft" not in df.columns or "prompt_id" not in df.columns:
            logger.info("Skipping TTFT chart — no benchmark data")
            return None

        ok = df[df["status"] == "success"] if "status" in df.columns else df
        if ok.empty:
            return None

        grouped = ok.groupby("prompt_id")["ttft"].mean().sort_values()

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        bars = ax.bar(range(len(grouped)), grouped.values, color=COLORS[1], edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Prompt ID", fontsize=11)
        ax.set_ylabel("Average TTFT (ms)", fontsize=11)
        ax.set_title("Time to First Token by Prompt", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        return self._save(fig, "ttft", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Tokens/sec Comparison
    # ──────────────────────────────────────────────────────────────────────────

    def chart_tokens_per_second(self) -> Path | None:
        """Bar chart of average tokens/second per prompt."""
        df = self.benchmark_df
        if df is None or "tokens_per_second" not in df.columns or "prompt_id" not in df.columns:
            logger.info("Skipping tokens/sec chart — no benchmark data")
            return None

        ok = df[df["status"] == "success"] if "status" in df.columns else df
        if ok.empty:
            return None

        grouped = ok.groupby("prompt_id")["tokens_per_second"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        bars = ax.bar(range(len(grouped)), grouped.values, color=COLORS[2], edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Prompt ID", fontsize=11)
        ax.set_ylabel("Average Tokens / Second", fontsize=11)
        ax.set_title("Generation Throughput by Prompt", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        return self._save(fig, "tokens_per_second", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. CPU Usage
    # ──────────────────────────────────────────────────────────────────────────

    def chart_cpu(self) -> Path | None:
        """Bar chart of average CPU usage per prompt."""
        df = self.benchmark_df
        if df is None or "cpu_usage" not in df.columns or "prompt_id" not in df.columns:
            logger.info("Skipping CPU chart — no benchmark data")
            return None

        ok = df[df["status"] == "success"] if "status" in df.columns else df
        if ok.empty:
            return None

        grouped = ok.groupby("prompt_id")["cpu_usage"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        bars = ax.bar(range(len(grouped)), grouped.values, color=COLORS[3], edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Prompt ID", fontsize=11)
        ax.set_ylabel("Average CPU Usage (%)", fontsize=11)
        ax.set_title("CPU Usage by Prompt", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        return self._save(fig, "cpu", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Memory Usage
    # ──────────────────────────────────────────────────────────────────────────

    def chart_memory(self) -> Path | None:
        """Bar chart of average memory usage per prompt."""
        df = self.benchmark_df
        if df is None or "memory_usage" not in df.columns or "prompt_id" not in df.columns:
            logger.info("Skipping memory chart — no benchmark data")
            return None

        ok = df[df["status"] == "success"] if "status" in df.columns else df
        if ok.empty:
            return None

        grouped = ok.groupby("prompt_id")["memory_usage"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        bars = ax.bar(range(len(grouped)), grouped.values, color=COLORS[4], edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Prompt ID", fontsize=11)
        ax.set_ylabel("Average Memory Usage (MB)", fontsize=11)
        ax.set_title("Memory Usage by Prompt", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7)

        return self._save(fig, "memory", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Speculative Speedup
    # ──────────────────────────────────────────────────────────────────────────

    def chart_speedup(self) -> Path | None:
        """Bar chart of per-prompt speedup (baseline / speculative duration)."""
        df = self.speculative_df
        if df is None:
            logger.info("Skipping speedup chart — no speculative data")
            return None

        required = {"baseline_duration", "speculative_duration", "prompt_id"}
        if not required.issubset(df.columns):
            logger.info("Skipping speedup chart — missing columns: %s", required - set(df.columns))
            return None

        ok = df.copy()
        ok = ok[(ok["baseline_duration"] > 0) & (ok["speculative_duration"] > 0)]
        if ok.empty:
            return None

        ok["speedup"] = ok["baseline_duration"] / ok["speculative_duration"]
        grouped = ok.groupby("prompt_id")["speedup"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

        bar_colors = [COLORS[2] if v >= 1.0 else COLORS[3] for v in grouped.values]
        bars = ax.bar(range(len(grouped)), grouped.values, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2, label="Baseline (1.0x)")

        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Prompt ID", fontsize=11)
        ax.set_ylabel("Speedup (baseline / speculative)", fontsize=11)
        ax.set_title("Speculative Decoding Speedup by Prompt", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}x", ha="center", va="bottom", fontsize=7)

        return self._save(fig, "speedup", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Quantization Leaderboard
    # ──────────────────────────────────────────────────────────────────────────

    def chart_leaderboard(self) -> Path | None:
        """Horizontal bar chart of models ranked by average TPS."""
        data = self.leaderboard
        if data is None:
            logger.info("Skipping leaderboard chart — no leaderboard data")
            return None

        entries = data.get("leaderboard", [])
        if not entries:
            return None

        # Already sorted by rank in the JSON
        models = [e["model"] for e in entries]
        tps_vals = [e["avg_tokens_per_second"] for e in entries]
        ranks = [e["rank"] for e in entries]

        # Reverse for horizontal bar (top rank at top)
        models_r = models[::-1]
        tps_r = tps_vals[::-1]
        ranks_r = ranks[::-1]

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, max(4, len(models) * 0.8)))
        bar_colors = [COLORS[i % len(COLORS)] for i in range(len(models_r))]
        bars = ax.barh(range(len(models_r)), tps_r, color=bar_colors, edgecolor="white", linewidth=0.5)

        ax.set_yticks(range(len(models_r)))
        ax.set_yticklabels([f"#{r}  {m}" for r, m in zip(ranks_r, models_r)], fontsize=10)
        ax.set_xlabel("Average Tokens / Second", fontsize=11)
        ax.set_title("Quantization Leaderboard", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for bar, val in zip(bars, tps_r):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", ha="left", va="center", fontsize=9)

        return self._save(fig, "leaderboard", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Success Rate
    # ──────────────────────────────────────────────────────────────────────────

    def chart_success_rate(self) -> Path | None:
        """Bar chart of success rate per prompt (from benchmark data)."""
        df = self.benchmark_df
        if df is None or "prompt_id" not in df.columns or "status" not in df.columns:
            logger.info("Skipping success rate chart — no benchmark data")
            return None

        total = df.groupby("prompt_id").size()
        success = df[df["status"] == "success"].groupby("prompt_id").size()
        rate = (success.reindex(total.index, fill_value=0) / total * 100).sort_values()

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        bar_colors = [COLORS[2] if v >= 100 else COLORS[1] if v >= 50 else COLORS[3] for v in rate.values]
        bars = ax.bar(range(len(rate)), rate.values, color=bar_colors, edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(rate)))
        ax.set_xticklabels(rate.index, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Prompt ID", fontsize=11)
        ax.set_ylabel("Success Rate (%)", fontsize=11)
        ax.set_title("Prompt Success Rate", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.axhline(y=100, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, rate.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

        return self._save(fig, "success_rate", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 9. Average Duration
    # ──────────────────────────────────────────────────────────────────────────

    def chart_duration(self) -> Path | None:
        """Bar chart of average duration per prompt."""
        df = self.benchmark_df
        if df is None or "duration" not in df.columns or "prompt_id" not in df.columns:
            logger.info("Skipping duration chart — no benchmark data")
            return None

        ok = df[df["status"] == "success"] if "status" in df.columns else df
        if ok.empty:
            return None

        grouped = ok.groupby("prompt_id")["duration"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        bars = ax.bar(range(len(grouped)), grouped.values, color=COLORS[5], edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Prompt ID", fontsize=11)
        ax.set_ylabel("Average Duration (s)", fontsize=11)
        ax.set_title("Average Inference Duration by Prompt", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        return self._save(fig, "duration", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 10. Radar Chart (Overall Performance)
    # ──────────────────────────────────────────────────────────────────────────

    def chart_radar(self) -> Path | None:
        """Radar chart normalizing TPS, Latency, Memory, CPU, TTFT.

        Each metric is normalized to [0, 1] where 1 = best.
        For TPS higher is better; for the rest lower is better.
        """
        df = self.benchmark_df
        if df is None:
            logger.info("Skipping radar chart — no benchmark data")
            return None

        required = {"tokens_per_second", "latency", "memory_usage", "cpu_usage", "ttft"}
        if not required.issubset(df.columns):
            logger.info("Skipping radar chart — missing columns: %s", required - set(df.columns))
            return None

        ok = df[df["status"] == "success"] if "status" in df.columns else df
        if ok.empty:
            return None

        metrics = {
            "TPS": ("tokens_per_second", True),      # higher is better
            "Latency": ("latency", False),             # lower is better
            "Memory": ("memory_usage", False),         # lower is better
            "CPU": ("cpu_usage", False),               # lower is better
            "TTFT": ("ttft", False),                   # lower is better
        }

        raw: dict[str, float] = {}
        for label, (col, higher_better) in metrics.items():
            val = ok[col].mean()
            raw[label] = val

        # Normalize: for each metric, map to [0, 1] where 1 = best
        normalized: dict[str, float] = {}
        for label, (col, higher_better) in metrics.items():
            val = raw[label]
            # Use a simple min-max style normalization based on observed range
            col_min = ok[col].min()
            col_max = ok[col].max()
            span = col_max - col_min
            if span == 0:
                normalized[label] = 1.0
            elif higher_better:
                normalized[label] = (val - col_min) / span
            else:
                normalized[label] = (col_max - val) / span

        labels = list(normalized.keys())
        values = [normalized[l] for l in labels]
        # Close the polygon
        values_closed = values + [values[0]]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles_closed, values_closed, color=COLORS[0], alpha=0.2)
        ax.plot(angles_closed, values_closed, color=COLORS[0], linewidth=2)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title("Overall Performance Profile\n(1.0 = best)", fontsize=13,
                      fontweight="bold", pad=20)

        # Annotate raw values
        for angle, label, norm_val, raw_val in zip(angles, labels, values, raw.values()):
            ax.annotate(
                f"{raw_val:.1f}",
                xy=(angle, norm_val),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=COLORS[0],
            )

        return self._save(fig, "radar", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Generate all
    # ──────────────────────────────────────────────────────────────────────────

    def generate_all(self) -> list[Path]:
        """Generate all applicable charts.

        Returns:
            List of paths to successfully generated chart files.
        """
        self.discover_and_load()

        generators = [
            self.chart_latency,
            self.chart_ttft,
            self.chart_tokens_per_second,
            self.chart_cpu,
            self.chart_memory,
            self.chart_speedup,
            self.chart_leaderboard,
            self.chart_success_rate,
            self.chart_duration,
            self.chart_radar,
        ]

        paths: list[Path] = []
        for gen in generators:
            try:
                result = gen()
                if result is not None:
                    paths.append(result)
            except Exception:
                logger.exception("Chart generation failed: %s", gen.__name__)

        return paths


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to sys.argv[1:].

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="generate_charts",
        description="Generate publication-quality charts from benchmark results.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing result files, or a single result file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("charts"),
        help="Output directory for chart PNGs (default: charts/).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    """Configure root logger.

    Args:
        level: Log level string.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    args = parse_args(argv)
    setup_logging(args.log_level)

    logger.info("Chart generator starting")
    logger.info("Input:  %s", args.input)
    logger.info("Output: %s", args.output)

    generator = ChartGenerator(args.input, args.output)

    try:
        paths = generator.generate_all()
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    except Exception:
        logger.exception("Chart generation failed")
        return 1

    logger.info("Generated %d charts in %s", len(paths), args.output)
    for p in paths:
        logger.info("  %s", p.name)

    return 0 if paths else 1


if __name__ == "__main__":
    sys.exit(main())
