#!/usr/bin/env python3
"""evaluation/evaluate.py — Final evaluation report generator for benchmark runs.

Reads benchmark outputs produced by benchmark.py, compare_speculative.py,
and compare_quantizations.py, then generates a unified evaluation summary
as JSON, CSV, and a Markdown report with automated recommendations.

Usage:
    python -m evaluation.evaluate \\
        --input results/ \\
        --output reports/

    python -m evaluation.evaluate \\
        --input results/benchmark_results.csv results/speculative_results.csv \\
        --output reports/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation report."""

    input_directory: Path
    output_directory: Path

    def validate(self) -> None:
        """Validate configuration, raising ValueError on failure."""
        if not self.input_directory.exists():
            raise ValueError(f"Input directory does not exist: {self.input_directory}")
        if not self.input_directory.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_directory}")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation summary
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationSummary:
    """Unified summary across all benchmark result types."""

    total_prompts: int = 0
    successful_runs: int = 0
    failed_runs: int = 0

    average_ttft: float = 0.0
    average_latency: float = 0.0
    average_tokens_per_second: float = 0.0
    average_cpu_usage: float = 0.0
    average_memory_usage: float = 0.0
    average_duration: float = 0.0
    success_rate: float = 0.0

    best_model: str = ""
    best_quantization: str = ""
    best_speedup: float = 0.0

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


def load_csv(path: Path) -> list[dict[str, Any]]:
    """Load a CSV file into a list of row dicts.

    Args:
        path: Path to the CSV file.

    Returns:
        List of row dicts, or empty list if file is missing/empty/invalid.
    """
    if not path.exists():
        logger.warning("CSV not found: %s", path)
        return []
    if path.stat().st_size == 0:
        logger.warning("CSV is empty: %s", path)
        return []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            logger.warning("CSV has no data rows: %s", path)
            return []
        logger.info("Loaded %s (%d rows)", path.name, len(rows))
        return rows
    except Exception:
        logger.exception("Failed to parse CSV: %s", path)
        return []


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dict, or None if file is missing/invalid.
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


# ──────────────────────────────────────────────────────────────────────────────
# Safe numeric helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning default on failure."""
    try:
        result = float(value)
        return result if result == result else default  # NaN check
    except (TypeError, ValueError):
        return default


def _safe_mean(values: list[float]) -> float:
    """Compute mean of a list, returning 0.0 for empty lists."""
    finite = [v for v in values if v == v]  # filter NaN
    return statistics.mean(finite) if finite else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation functions
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_benchmark(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate standard benchmark results.

    Args:
        rows: List of row dicts from benchmark_results.csv.

    Returns:
        Dict with aggregate metrics.
    """
    if not rows:
        return {}

    total = len(rows)
    successful = [r for r in rows if r.get("status") == "success"]
    failed = total - len(successful)

    if not successful:
        return {
            "total_prompts": total,
            "successful_runs": 0,
            "failed_runs": failed,
            "success_rate": 0.0,
        }

    ttft_vals = [_safe_float(r.get("ttft")) for r in successful]
    lat_vals = [_safe_float(r.get("latency")) for r in successful]
    tps_vals = [_safe_float(r.get("tokens_per_second")) for r in successful]
    cpu_vals = [_safe_float(r.get("cpu_usage")) for r in successful]
    mem_vals = [_safe_float(r.get("memory_usage")) for r in successful]
    dur_vals = [_safe_float(r.get("duration")) for r in successful]

    return {
        "total_prompts": total,
        "successful_runs": len(successful),
        "failed_runs": failed,
        "success_rate": len(successful) / total * 100,
        "average_ttft": _safe_mean(ttft_vals),
        "average_latency": _safe_mean(lat_vals),
        "average_tokens_per_second": _safe_mean(tps_vals),
        "average_cpu_usage": _safe_mean(cpu_vals),
        "average_memory_usage": _safe_mean(mem_vals),
        "average_duration": _safe_mean(dur_vals),
    }


def evaluate_speculative(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate speculative decoding comparison results.

    Args:
        rows: List of row dicts from speculative_results.csv.

    Returns:
        Dict with comparative metrics.
    """
    if not rows:
        return {}

    valid = [
        r for r in rows
        if r.get("winner") in ("speculative", "baseline", "tie")
    ]

    if not valid:
        total = len(rows)
        return {
            "total_comparisons": total,
            "valid_comparisons": 0,
        }

    speedups = [_safe_float(r.get("speedup")) for r in valid]
    lat_improvements = [
        _safe_float(r.get("baseline_latency")) - _safe_float(r.get("speculative_latency"))
        for r in valid
    ]
    ttft_improvements = [
        _safe_float(r.get("baseline_ttft")) - _safe_float(r.get("speculative_ttft"))
        for r in valid
    ]
    acceptance_rates = [_safe_float(r.get("acceptance_rate")) for r in valid]

    spec_wins = sum(1 for r in valid if r.get("winner") == "speculative")
    base_wins = sum(1 for r in valid if r.get("winner") == "baseline")
    ties = sum(1 for r in valid if r.get("winner") == "tie")

    return {
        "total_comparisons": len(rows),
        "valid_comparisons": len(valid),
        "speculative_wins": spec_wins,
        "baseline_wins": base_wins,
        "ties": ties,
        "average_speedup": _safe_mean(speedups),
        "average_latency_improvement_ms": _safe_mean(lat_improvements),
        "average_ttft_improvement_ms": _safe_mean(ttft_improvements),
        "average_acceptance_rate": _safe_mean(acceptance_rates),
    }


def evaluate_quantizations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate quantization comparison results and produce a leaderboard.

    Args:
        rows: List of row dicts from quantization_results.csv.

    Returns:
        Dict with per-model aggregates and a ranked leaderboard.
    """
    if not rows:
        return {}

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        model = r.get("model", "")
        if model:
            by_model[model].append(r)

    if not by_model:
        return {}

    model_stats: list[dict[str, Any]] = []

    for model_name, model_rows in by_model.items():
        total = len(model_rows)
        successful = [r for r in model_rows if r.get("status") == "success"]
        n_ok = len(successful)

        if n_ok == 0:
            model_stats.append({
                "model": model_name,
                "total_prompts": total,
                "successful": 0,
                "success_rate": 0.0,
                "avg_ttft": 0.0,
                "avg_latency": 0.0,
                "avg_tokens_per_second": 0.0,
                "avg_cpu_usage": 0.0,
                "avg_memory_usage": 0.0,
                "avg_duration": 0.0,
            })
            continue

        model_stats.append({
            "model": model_name,
            "total_prompts": total,
            "successful": n_ok,
            "success_rate": n_ok / total * 100,
            "avg_ttft": _safe_mean([_safe_float(r.get("ttft")) for r in successful]),
            "avg_latency": _safe_mean([_safe_float(r.get("latency")) for r in successful]),
            "avg_tokens_per_second": _safe_mean([_safe_float(r.get("tokens_per_second")) for r in successful]),
            "avg_cpu_usage": _safe_mean([_safe_float(r.get("cpu_usage")) for r in successful]),
            "avg_memory_usage": _safe_mean([_safe_float(r.get("memory_usage")) for r in successful]),
            "avg_duration": _safe_mean([_safe_float(r.get("duration")) for r in successful]),
        })

    # Rank by TPS (desc), then latency (asc), then TTFT (asc), then memory (asc)
    ranked = sorted(
        [m for m in model_stats if m["successful"] > 0],
        key=lambda m: (
            -m["avg_tokens_per_second"],
            m["avg_latency"],
            m["avg_ttft"],
            m["avg_memory_usage"],
        ),
    )
    for i, entry in enumerate(ranked):
        entry["rank"] = i + 1

    # Also include models with 0 successes at the bottom
    failed_models = [m for m in model_stats if m["successful"] == 0]
    for m in failed_models:
        m["rank"] = len(ranked) + 1
    ranked.extend(failed_models)

    return {
        "total_models": len(by_model),
        "models": model_stats,
        "leaderboard": ranked,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────


def _generate_recommendations(
    bench: dict[str, Any],
    spec: dict[str, Any],
    quant: dict[str, Any],
) -> list[str]:
    """Generate automated recommendations based on evaluation results.

    Args:
        bench: Benchmark evaluation results.
        spec: Speculative decoding evaluation results.
        quant: Quantization evaluation results.

    Returns:
        List of recommendation strings.
    """
    recs: list[str] = []

    # Best latency
    if bench.get("average_latency", 0) > 0:
        recs.append(
            f"**Best Latency**: Average inter-token latency is "
            f"{bench['average_latency']:.1f} ms. "
            f"{'Good responsiveness.' if bench['average_latency'] < 30 else 'Consider a smaller quantization for lower latency.'}"
        )

    # Best throughput
    if bench.get("average_tokens_per_second", 0) > 0:
        tps = bench["average_tokens_per_second"]
        recs.append(
            f"**Best Throughput**: Average generation speed is "
            f"{tps:.2f} tokens/sec. "
            f"{'Excellent throughput.' if tps > 30 else 'Throughput is moderate; check quantization options.'}"
        )

    # Best memory
    if bench.get("average_memory_usage", 0) > 0:
        mem = bench["average_memory_usage"]
        recs.append(
            f"**Best Memory Efficiency**: Average RSS is "
            f"{mem:.1f} MB. "
            f"{'Efficient memory usage.' if mem < 200 else 'Memory usage is high; consider a more aggressive quantization.'}"
        )

    # Best quantization
    leaderboard = quant.get("leaderboard", [])
    if leaderboard:
        best = leaderboard[0]
        recs.append(
            f"**Best Quantization Model**: {best['model']} ranks #1 with "
            f"{best['avg_tokens_per_second']:.2f} tok/s, "
            f"{best['avg_latency']:.1f} ms latency, "
            f"{best['avg_memory_usage']:.1f} MB memory."
        )

    # Speculative decoding
    if spec.get("valid_comparisons", 0) > 0:
        avg_speedup = spec.get("average_speedup", 0)
        spec_wins = spec.get("speculative_wins", 0)
        total_valid = spec.get("valid_comparisons", 0)
        recs.append(
            f"**Speculative Decoding**: Average speedup is {avg_speedup:.3f}x "
            f"({spec_wins}/{total_valid} prompts faster). "
            f"{'Recommended for production.' if avg_speedup > 1.1 else 'Marginal improvement; evaluate cost/benefit.'}"
        )

    # Success rate
    if bench.get("success_rate", 0) < 100:
        recs.append(
            f"**Reliability**: Success rate is {bench.get('success_rate', 0):.1f}%. "
            f"Investigate failures for production readiness."
        )

    if not recs:
        recs.append("Insufficient data to generate recommendations.")

    return recs


def _build_markdown_report(
    summary: EvaluationSummary,
    bench: dict[str, Any],
    spec: dict[str, Any],
    quant: dict[str, Any],
    recommendations: list[str],
) -> str:
    """Build a Markdown evaluation report.

    Args:
        summary: Overall evaluation summary.
        bench: Benchmark evaluation results.
        spec: Speculative decoding evaluation results.
        quant: Quantization evaluation results.
        recommendations: List of recommendation strings.

    Returns:
        Complete Markdown report string.
    """
    lines: list[str] = []

    lines.append("# Benchmark Evaluation Report")
    lines.append("")
    lines.append(f"*Generated: {summary.timestamp}*")
    lines.append("")

    # ── Overall Summary ──────────────────────────────────────────────────────
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Prompts | {summary.total_prompts} |")
    lines.append(f"| Successful Runs | {summary.successful_runs} |")
    lines.append(f"| Failed Runs | {summary.failed_runs} |")
    lines.append(f"| Success Rate | {summary.success_rate:.1f}% |")
    lines.append(f"| Best Quantization | {summary.best_quantization or 'N/A'} |")
    lines.append(f"| Best Speedup | {summary.best_speedup:.3f}x |")
    lines.append("")

    # ── Benchmark Metrics ────────────────────────────────────────────────────
    lines.append("## Benchmark Metrics")
    lines.append("")
    if bench:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Average TTFT | {bench.get('average_ttft', 0):.1f} ms |")
        lines.append(f"| Average Latency | {bench.get('average_latency', 0):.1f} ms |")
        lines.append(f"| Average Tokens/sec | {bench.get('average_tokens_per_second', 0):.2f} |")
        lines.append(f"| Average CPU Usage | {bench.get('average_cpu_usage', 0):.1f}% |")
        lines.append(f"| Average Memory | {bench.get('average_memory_usage', 0):.1f} MB |")
        lines.append(f"| Average Duration | {bench.get('average_duration', 0):.2f} s |")
    else:
        lines.append("*No benchmark results available.*")
    lines.append("")

    # ── Speculative Decoding ─────────────────────────────────────────────────
    lines.append("## Speculative Decoding Results")
    lines.append("")
    if spec and spec.get("valid_comparisons", 0) > 0:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Valid Comparisons | {spec.get('valid_comparisons', 0)} |")
        lines.append(f"| Speculative Wins | {spec.get('speculative_wins', 0)} |")
        lines.append(f"| Baseline Wins | {spec.get('baseline_wins', 0)} |")
        lines.append(f"| Ties | {spec.get('ties', 0)} |")
        lines.append(f"| Average Speedup | {spec.get('average_speedup', 0):.3f}x |")
        lines.append(f"| Avg Latency Improvement | {spec.get('average_latency_improvement_ms', 0):+.1f} ms |")
        lines.append(f"| Avg TTFT Improvement | {spec.get('average_ttft_improvement_ms', 0):+.1f} ms |")
        lines.append(f"| Avg Acceptance Rate | {spec.get('average_acceptance_rate', 0):.1%} |")
    else:
        lines.append("*No speculative decoding results available.*")
    lines.append("")

    # ── Quantization Comparison ──────────────────────────────────────────────
    lines.append("## Quantization Comparison")
    lines.append("")
    leaderboard = quant.get("leaderboard", [])
    if leaderboard:
        lines.append("| Rank | Model | TPS | TTFT (ms) | Latency (ms) | Memory (MB) | Success Rate |")
        lines.append("|------|-------|-----|-----------|--------------|-------------|--------------|")
        for entry in leaderboard:
            if entry.get("successful", 0) == 0:
                lines.append(
                    f"| {entry.get('rank', '-')} | {entry['model']} | -- | -- | -- | -- | 0% |"
                )
            else:
                lines.append(
                    f"| {entry.get('rank', '-')} | {entry['model']} "
                    f"| {entry.get('avg_tokens_per_second', 0):.2f} "
                    f"| {entry.get('avg_ttft', 0):.1f} "
                    f"| {entry.get('avg_latency', 0):.1f} "
                    f"| {entry.get('avg_memory_usage', 0):.1f} "
                    f"| {entry.get('success_rate', 0):.0f}% |"
                )
    else:
        lines.append("*No quantization results available.*")
    lines.append("")

    # ── Leaderboard ──────────────────────────────────────────────────────────
    lines.append("## Leaderboard")
    lines.append("")
    if leaderboard and leaderboard[0].get("successful", 0) > 0:
        lines.append(
            f"**#1 {leaderboard[0]['model']}** — "
            f"{leaderboard[0].get('avg_tokens_per_second', 0):.2f} tok/s, "
            f"{leaderboard[0].get('avg_latency', 0):.1f} ms latency"
        )
        lines.append("")
        if len(leaderboard) > 1:
            for entry in leaderboard[1:]:
                if entry.get("successful", 0) == 0:
                    continue
                lines.append(
                    f"- **#{entry.get('rank', '?')} {entry['model']}** — "
                    f"{entry.get('avg_tokens_per_second', 0):.2f} tok/s, "
                    f"{entry.get('avg_latency', 0):.1f} ms latency"
                )
        lines.append("")
    else:
        lines.append("*No ranked models available.*")
        lines.append("")

    # ── Recommendations ──────────────────────────────────────────────────────
    lines.append("## Recommendations")
    lines.append("")
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────────────────────


class EvaluationRunner:
    """Orchestrates the full evaluation: loads data, computes metrics,
    generates outputs."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize the evaluation runner.

        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.benchmark_rows: list[dict[str, Any]] = []
        self.speculative_rows: list[dict[str, Any]] = []
        self.quantization_rows: list[dict[str, Any]] = []
        self.leaderboard_data: dict[str, Any] | None = None
        self.summary_data: dict[str, Any] | None = None

        self.bench_result: dict[str, Any] = {}
        self.spec_result: dict[str, Any] = {}
        self.quant_result: dict[str, Any] = {}
        self.evaluation_summary: EvaluationSummary | None = None

    def discover_and_load(self) -> None:
        """Scan input directory and load all recognized result files."""
        d = self.config.input_directory

        # Benchmark CSV (may be timestamped)
        for f in sorted(d.glob("benchmark_results*.csv")):
            self.benchmark_rows = load_csv(f)
            if self.benchmark_rows:
                break

        # Speculative CSV
        spec_csv = d / "speculative_results.csv"
        self.speculative_rows = load_csv(spec_csv)

        # Quantization CSV
        quant_csv = d / "quantization_results.csv"
        self.quantization_rows = load_csv(quant_csv)

        # Leaderboard JSON
        lb_json = d / "leaderboard.json"
        self.leaderboard_data = load_json(lb_json)

        # Summary JSON (speculative)
        summary_json = d / "summary.json"
        self.summary_data = load_json(summary_json)

    def evaluate(self) -> EvaluationSummary:
        """Run all evaluations and build the summary.

        Returns:
            Populated EvaluationSummary.
        """
        self.discover_and_load()

        # Evaluate each section
        self.bench_result = evaluate_benchmark(self.benchmark_rows)
        self.spec_result = evaluate_speculative(self.speculative_rows)
        self.quant_result = evaluate_quantizations(self.quantization_rows)

        # Build unified summary
        summary = EvaluationSummary()
        summary.total_prompts = self.bench_result.get("total_prompts", 0)
        summary.successful_runs = self.bench_result.get("successful_runs", 0)
        summary.failed_runs = self.bench_result.get("failed_runs", 0)
        summary.average_ttft = self.bench_result.get("average_ttft", 0.0)
        summary.average_latency = self.bench_result.get("average_latency", 0.0)
        summary.average_tokens_per_second = self.bench_result.get("average_tokens_per_second", 0.0)
        summary.average_cpu_usage = self.bench_result.get("average_cpu_usage", 0.0)
        summary.average_memory_usage = self.bench_result.get("average_memory_usage", 0.0)
        summary.average_duration = self.bench_result.get("average_duration", 0.0)
        summary.success_rate = self.bench_result.get("success_rate", 0.0)

        # Best quantization from leaderboard
        leaderboard = self.quant_result.get("leaderboard", [])
        if leaderboard and leaderboard[0].get("successful", 0) > 0:
            summary.best_quantization = leaderboard[0]["model"]

        # Best speedup from speculative
        summary.best_speedup = self.spec_result.get("average_speedup", 0.0)

        # Best model = best quantization (same thing)
        summary.best_model = summary.best_quantization

        self.evaluation_summary = summary
        return summary

    def save_results(self) -> list[Path]:
        """Save evaluation outputs: JSON, CSV, and Markdown report.

        Returns:
            List of paths to written files.
        """
        if self.evaluation_summary is None:
            raise RuntimeError("Must call evaluate() before save_results()")

        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        # ── JSON ─────────────────────────────────────────────────────────────
        json_path = self.config.output_directory / "evaluation_summary.json"
        payload = {
            "metadata": {
                "timestamp": self.evaluation_summary.timestamp,
                "input_directory": str(self.config.input_directory),
            },
            "summary": self.evaluation_summary.to_dict(),
            "benchmark": self.bench_result,
            "speculative": self.spec_result,
            "quantization": self.quant_result,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("JSON saved to %s", json_path)
        paths.append(json_path)

        # ── CSV ──────────────────────────────────────────────────────────────
        csv_path = self.config.output_directory / "evaluation_summary.csv"
        summary_dict = self.evaluation_summary.to_dict()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_dict.keys()))
            writer.writeheader()
            writer.writerow(summary_dict)
        logger.info("CSV saved to %s", csv_path)
        paths.append(csv_path)

        # ── Markdown ─────────────────────────────────────────────────────────
        recommendations = _generate_recommendations(
            self.bench_result, self.spec_result, self.quant_result,
        )
        md_content = _build_markdown_report(
            self.evaluation_summary,
            self.bench_result,
            self.spec_result,
            self.quant_result,
            recommendations,
        )
        md_path = self.config.output_directory / "evaluation_report.md"
        md_path.write_text(md_content, encoding="utf-8")
        logger.info("Markdown saved to %s", md_path)
        paths.append(md_path)

        return paths

    def print_summary(self) -> None:
        """Print a formatted evaluation summary to stdout."""
        s = self.evaluation_summary
        if s is None:
            return

        print()
        print("=" * 70)
        print("  EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  Total Prompts        : {s.total_prompts}")
        print(f"  Successful           : {s.successful_runs}")
        print(f"  Failed               : {s.failed_runs}")
        print(f"  Success Rate         : {s.success_rate:.1f}%")
        print("-" * 70)

        if s.average_tokens_per_second > 0:
            print(f"  Avg TTFT             : {s.average_ttft:.1f} ms")
            print(f"  Avg Latency          : {s.average_latency:.1f} ms")
            print(f"  Avg Tokens/sec       : {s.average_tokens_per_second:.2f}")
            print(f"  Avg CPU              : {s.average_cpu_usage:.1f}%")
            print(f"  Avg Memory           : {s.average_memory_usage:.1f} MB")
            print(f"  Avg Duration         : {s.average_duration:.2f} s")

        if s.best_quantization:
            print("-" * 70)
            print(f"  Best Quantization    : {s.best_quantization}")

        if s.best_speedup > 0:
            print(f"  Avg Speculative Speedup : {s.best_speedup:.3f}x")

        print("=" * 70)
        print()

        # Print recommendations
        recs = _generate_recommendations(
            self.bench_result, self.spec_result, self.quant_result,
        )
        print("  RECOMMENDATIONS")
        print("  " + "-" * 66)
        for rec in recs:
            # Strip markdown bold for terminal
            clean = rec.replace("**", "")
            print(f"  - {clean}")
        print()


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
        prog="evaluate",
        description="Generate evaluation report from benchmark results.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing benchmark result files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports (default: reports/).",
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

    logger.info("Evaluation starting")
    logger.info("Input:  %s", args.input)
    logger.info("Output: %s", args.output)

    config = EvaluationConfig(
        input_directory=args.input,
        output_directory=args.output,
    )

    try:
        config.validate()
    except ValueError as e:
        logger.error("Invalid configuration: %s", e)
        return 1

    runner = EvaluationRunner(config)

    try:
        runner.evaluate()
        runner.print_summary()
        paths = runner.save_results()
        logger.info("Reports saved to: %s", ", ".join(str(p) for p in paths))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Evaluation failed")
        return 1

    summary = runner.evaluation_summary
    if summary and summary.total_prompts > 0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
