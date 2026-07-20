#!/usr/bin/env python3
"""benchmark/compare_speculative.py — Speculative vs baseline decoding comparison.

Runs the same prompt dataset against two llama-server instances
(one normal, one with speculative decoding enabled) via
benchmark.BenchmarkRunner, then computes per-prompt and aggregate
comparative metrics.

Usage:
    python -m benchmark.compare_speculative \\
        --baseline-url http://localhost:8080 \\
        --baseline-model qwen2.5 \\
        --speculative-url http://localhost:8081 \\
        --speculative-model qwen2.5 \\
        --trials 3 \\
        --output results/

    python benchmark/compare_speculative.py \\
        --baseline-url http://10.0.0.1:8080 \\
        --baseline-model llama-70b \\
        --speculative-url http://10.0.0.2:8080 \\
        --speculative-model llama-70b \\
        --trials 5 \\
        --output /tmp/spec_compare/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from benchmark.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner  # noqa: E402

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SpeculativeConfig:
    """Configuration for a speculative decoding A/B comparison."""

    baseline_server_url: str
    baseline_model: str
    speculative_server_url: str
    speculative_model: str
    trials: int = 3
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 120.0
    output_directory: Path = Path("results")

    def validate(self) -> None:
        """Validate configuration, raising ValueError on failure."""
        if not self.baseline_server_url:
            raise ValueError("baseline_server_url must not be empty")
        if not self.baseline_model:
            raise ValueError("baseline_model must not be empty")
        if not self.speculative_server_url:
            raise ValueError("speculative_server_url must not be empty")
        if not self.speculative_model:
            raise ValueError("speculative_model must not be empty")
        if self.trials < 1:
            raise ValueError(f"trials must be >= 1, got {self.trials}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {self.temperature}")

    def baseline_benchmark_config(self) -> BenchmarkConfig:
        """Build a BenchmarkConfig for the baseline server."""
        return BenchmarkConfig(
            server_url=self.baseline_server_url,
            model_name=self.baseline_model,
            trials=self.trials,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            output_directory=self.output_directory,
            save_json=False,
            save_csv=False,
        )

    def speculative_benchmark_config(self) -> BenchmarkConfig:
        """Build a BenchmarkConfig for the speculative server."""
        return BenchmarkConfig(
            server_url=self.speculative_server_url,
            model_name=self.speculative_model,
            trials=self.trials,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            output_directory=self.output_directory,
            save_json=False,
            save_csv=False,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Per-prompt comparison result
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SpeculativeComparisonResult:
    """Head-to-head comparison for a single prompt + trial."""

    prompt_id: str
    category: str
    trial: int

    baseline_ttft: float
    speculative_ttft: float

    baseline_latency: float
    speculative_latency: float

    baseline_tokens_per_second: float
    speculative_tokens_per_second: float

    baseline_cpu: float
    speculative_cpu: float

    baseline_memory: float
    speculative_memory: float

    baseline_duration: float
    speculative_duration: float

    acceptance_rate: float
    speedup: float
    latency_reduction: float
    winner: str

    baseline_status: str = "success"
    speculative_status: str = "success"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate summary
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AggregateSummary:
    """Aggregated comparison statistics across all prompt/trial pairs."""

    total_comparisons: int = 0
    valid_comparisons: int = 0

    avg_speedup: float = 0.0
    median_speedup: float = 0.0
    min_speedup: float = 0.0
    max_speedup: float = 0.0

    avg_ttft_improvement_ms: float = 0.0
    avg_latency_improvement_ms: float = 0.0
    avg_tps_improvement: float = 0.0

    avg_acceptance_rate: float = 0.0

    baseline_wins: int = 0
    speculative_wins: int = 0
    ties: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────


class SpeculativeComparator:
    """Orchestrates an A/B comparison between baseline and speculative decoding.

    Creates two BenchmarkRunner instances pointed at different servers,
    runs both against the same prompt dataset, pairs up results, and
    computes comparative metrics.
    """

    def __init__(self, config: SpeculativeConfig) -> None:
        """Initialize the comparator.

        Args:
            config: Comparison configuration.
        """
        self.config = config
        self.baseline_runner = BenchmarkRunner(config.baseline_benchmark_config())
        self.speculative_runner = BenchmarkRunner(config.speculative_benchmark_config())
        self.comparisons: list[SpeculativeComparisonResult] = []
        self.summary: AggregateSummary | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Runners
    # ──────────────────────────────────────────────────────────────────────────

    def run_baseline(self) -> list[BenchmarkResult]:
        """Run the baseline benchmark via BenchmarkRunner.

        Returns:
            List of BenchmarkResult from the baseline server.
        """
        logger.info(
            "Running baseline: %s (model=%s, trials=%d)",
            self.config.baseline_server_url,
            self.config.baseline_model,
            self.config.trials,
        )
        return self.baseline_runner.run_all()

    def run_speculative(self) -> list[BenchmarkResult]:
        """Run the speculative benchmark via BenchmarkRunner.

        Returns:
            List of BenchmarkResult from the speculative server.
        """
        logger.info(
            "Running speculative: %s (model=%s, trials=%d)",
            self.config.speculative_server_url,
            self.config.speculative_model,
            self.config.trials,
        )
        return self.speculative_runner.run_all()

    # ──────────────────────────────────────────────────────────────────────────
    # Comparison logic
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _pair_key(result: BenchmarkResult) -> tuple[str, int]:
        """Build a lookup key from a BenchmarkResult (prompt_id, trial)."""
        return (result.prompt_id, result.trial)

    @staticmethod
    def _compute_acceptance_rate(
        baseline: BenchmarkResult,
        speculative: BenchmarkResult,
    ) -> float:
        """Estimate draft-token acceptance rate from duration and TPS.

        When the speculative server is faster but generates a similar
        number of tokens, the ratio of durations approximates the
        acceptance-rate benefit.  We define acceptance_rate as the
        fraction of time saved:

            1 - (speculative_duration / baseline_duration)

        Clamped to [0, 1].

        Args:
            baseline: Baseline BenchmarkResult.
            speculative: Speculative BenchmarkResult.

        Returns:
            Estimated acceptance rate (0.0 – 1.0).  0.0 if estimation
            is not possible.
        """
        if baseline.duration <= 0 or speculative.duration <= 0:
            return 0.0
        rate = 1.0 - (speculative.duration / baseline.duration)
        return max(0.0, min(1.0, rate))

    def compare(
        self,
        baseline_results: list[BenchmarkResult],
        speculative_results: list[BenchmarkResult],
    ) -> list[SpeculativeComparisonResult]:
        """Pair up baseline and speculative results and compute comparative metrics.

        For every (prompt_id, trial) pair present in both sets, computes:
            - speedup = baseline_duration / speculative_duration
            - latency_reduction = baseline_latency - speculative_latency
            - winner = "speculative" if speedup > 1.0 else "baseline"
            - acceptance_rate (estimated)

        Results where either side failed are still included (winner="inconclusive")
        but excluded from aggregate statistics.

        Args:
            baseline_results: Results from the baseline server.
            speculative_results: Results from the speculative server.

        Returns:
            List of SpeculativeComparisonResult.
        """
        spec_map: dict[tuple[str, int], BenchmarkResult] = {
            self._pair_key(r): r for r in speculative_results
        }

        self.comparisons = []

        for b_res in baseline_results:
            key = self._pair_key(b_res)
            s_res = spec_map.get(key)

            if s_res is None:
                logger.warning(
                    "No speculative result for prompt=%s trial=%d — skipping",
                    b_res.prompt_id,
                    b_res.trial,
                )
                continue

            both_ok = b_res.status == "success" and s_res.status == "success"

            if both_ok:
                speedup = b_res.duration / s_res.duration if s_res.duration > 0 else 0.0
                latency_reduction = b_res.latency - s_res.latency
                acceptance = self._compute_acceptance_rate(b_res, s_res)

                if speedup > 1.001:
                    winner = "speculative"
                elif speedup < 0.999:
                    winner = "baseline"
                else:
                    winner = "tie"
            else:
                speedup = 0.0
                latency_reduction = 0.0
                acceptance = 0.0
                winner = "inconclusive"

            err_parts: list[str] = []
            if b_res.status != "success":
                err_parts.append(f"baseline:{b_res.status}")
            if s_res.status != "success":
                err_parts.append(f"speculative:{s_res.status}")

            comp = SpeculativeComparisonResult(
                prompt_id=b_res.prompt_id,
                category=b_res.category,
                trial=b_res.trial,
                baseline_ttft=b_res.ttft,
                speculative_ttft=s_res.ttft,
                baseline_latency=b_res.latency,
                speculative_latency=s_res.latency,
                baseline_tokens_per_second=b_res.tokens_per_second,
                speculative_tokens_per_second=s_res.tokens_per_second,
                baseline_cpu=b_res.cpu_usage,
                speculative_cpu=s_res.cpu_usage,
                baseline_memory=b_res.memory_usage,
                speculative_memory=s_res.memory_usage,
                baseline_duration=b_res.duration,
                speculative_duration=s_res.duration,
                acceptance_rate=acceptance,
                speedup=speedup,
                latency_reduction=latency_reduction,
                winner=winner,
                baseline_status=b_res.status,
                speculative_status=s_res.status,
                error="; ".join(err_parts),
            )
            self.comparisons.append(comp)

        self.summary = self._aggregate(self.comparisons)
        return self.comparisons

    # ──────────────────────────────────────────────────────────────────────────
    # Aggregation
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(comparisons: list[SpeculativeComparisonResult]) -> AggregateSummary:
        """Compute aggregate statistics from per-prompt comparisons.

        Only comparisons where both sides succeeded are included in
        numeric aggregates.

        Args:
            comparisons: Per-prompt comparison results.

        Returns:
            Populated AggregateSummary.
        """
        total = len(comparisons)
        valid = [c for c in comparisons if c.winner in ("speculative", "baseline", "tie")]

        if not valid:
            return AggregateSummary(
                total_comparisons=total,
                valid_comparisons=0,
            )

        speedups = [c.speedup for c in valid]
        ttft_diffs = [c.baseline_ttft - c.speculative_ttft for c in valid]
        lat_diffs = [c.baseline_latency - c.speculative_latency for c in valid]
        tps_diffs = [
            c.speculative_tokens_per_second - c.baseline_tokens_per_second
            for c in valid
        ]
        acceptance_vals = [c.acceptance_rate for c in valid]

        return AggregateSummary(
            total_comparisons=total,
            valid_comparisons=len(valid),
            avg_speedup=statistics.mean(speedups),
            median_speedup=statistics.median(speedups),
            min_speedup=min(speedups),
            max_speedup=max(speedups),
            avg_ttft_improvement_ms=statistics.mean(ttft_diffs),
            avg_latency_improvement_ms=statistics.mean(lat_diffs),
            avg_tps_improvement=statistics.mean(tps_diffs),
            avg_acceptance_rate=statistics.mean(acceptance_vals),
            baseline_wins=sum(1 for c in valid if c.winner == "baseline"),
            speculative_wins=sum(1 for c in valid if c.winner == "speculative"),
            ties=sum(1 for c in valid if c.winner == "tie"),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Print summary
    # ──────────────────────────────────────────────────────────────────────────

    def print_comparison(self) -> None:
        """Print a formatted comparison table to stdout."""
        if not self.comparisons:
            logger.warning("No comparisons to print")
            return

        s = self.summary
        if s is None:
            return

        print()
        print("=" * 78)
        print("  SPECULATIVE vs BASELINE DECODING COMPARISON")
        print("=" * 78)
        print(f"  Baseline server      : {self.config.baseline_server_url}")
        print(f"  Baseline model       : {self.config.baseline_model}")
        print(f"  Speculative server   : {self.config.speculative_server_url}")
        print(f"  Speculative model    : {self.config.speculative_model}")
        print(f"  Trials               : {self.config.trials}")
        print("-" * 78)

        if s.valid_comparisons == 0:
            print("  No valid comparisons (all prompts failed on at least one side).")
            print("=" * 78)
            return

        print(f"  Valid comparisons    : {s.valid_comparisons} / {s.total_comparisons}")
        print(f"  Speculative wins     : {s.speculative_wins}")
        print(f"  Baseline wins        : {s.baseline_wins}")
        print(f"  Ties                 : {s.ties}")
        print("-" * 78)
        print("  AGGREGATE METRICS")
        print(f"    Avg speedup              : {s.avg_speedup:.3f}x")
        print(f"    Median speedup           : {s.median_speedup:.3f}x")
        print(f"    Min / Max speedup        : {s.min_speedup:.3f}x / {s.max_speedup:.3f}x")
        print(f"    Avg TTFT improvement     : {s.avg_ttft_improvement_ms:+.1f} ms")
        print(f"    Avg latency improvement  : {s.avg_latency_improvement_ms:+.1f} ms")
        print(f"    Avg TPS improvement      : {s.avg_tps_improvement:+.2f} tok/s")
        print(f"    Avg acceptance rate      : {s.avg_acceptance_rate:.1%}")
        print("-" * 78)

        print(f"  {'Prompt':<12} {'Trial':>5} {'Base TPS':>10} {'Spec TPS':>10} {'Speedup':>9} {'Winner':<14}")
        print("  " + "-" * 62)
        for c in self.comparisons[:30]:
            print(
                f"  {c.prompt_id:<12} {c.trial:>5} "
                f"{c.baseline_tokens_per_second:>10.2f} "
                f"{c.speculative_tokens_per_second:>10.2f} "
                f"{c.speedup:>8.3f}x "
                f"{c.winner:<14}"
            )
        if len(self.comparisons) > 30:
            print(f"  ... and {len(self.comparisons) - 30} more")

        print("=" * 78)
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # Save outputs
    # ──────────────────────────────────────────────────────────────────────────

    def save_results(self) -> list[Path]:
        """Save comparison results to CSV, per-prompt JSON, and summary JSON.

        Returns:
            List of paths to written files.
        """
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        paths: list[Path] = []

        # ── CSV ──────────────────────────────────────────────────────────────
        csv_path = self.config.output_directory / "speculative_results.csv"
        fieldnames = [
            "prompt_id", "category", "trial",
            "baseline_ttft", "speculative_ttft",
            "baseline_latency", "speculative_latency",
            "baseline_tokens_per_second", "speculative_tokens_per_second",
            "baseline_cpu", "speculative_cpu",
            "baseline_memory", "speculative_memory",
            "baseline_duration", "speculative_duration",
            "acceptance_rate", "speedup", "latency_reduction", "winner",
            "baseline_status", "speculative_status", "error",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in self.comparisons:
                writer.writerow({k: v for k, v in c.to_dict().items() if k in fieldnames})

        logger.info("CSV saved to %s", csv_path)
        paths.append(csv_path)

        # ── Per-prompt JSON ──────────────────────────────────────────────────
        json_path = self.config.output_directory / "speculative_results.json"
        payload = {
            "metadata": {
                "baseline_server_url": self.config.baseline_server_url,
                "baseline_model": self.config.baseline_model,
                "speculative_server_url": self.config.speculative_server_url,
                "speculative_model": self.config.speculative_model,
                "trials": self.config.trials,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "results": [c.to_dict() for c in self.comparisons],
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("JSON saved to %s", json_path)
        paths.append(json_path)

        # ── Summary JSON ─────────────────────────────────────────────────────
        summary_path = self.config.output_directory / "summary.json"
        summary_payload = {
            "metadata": payload["metadata"],
            "aggregate": self.summary.to_dict() if self.summary else {},
            "per_prompt_summary": self._per_prompt_summary(),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
        logger.info("Summary saved to %s", summary_path)
        paths.append(summary_path)

        return paths

    def _per_prompt_summary(self) -> list[dict[str, Any]]:
        """Aggregate metrics per prompt_id (averaging across trials).

        Returns:
            List of per-prompt summary dicts.
        """
        from collections import defaultdict

        by_prompt: dict[str, list[SpeculativeComparisonResult]] = defaultdict(list)
        for c in self.comparisons:
            by_prompt[c.prompt_id].append(c)

        summaries: list[dict[str, Any]] = []
        for pid in sorted(by_prompt):
            items = by_prompt[pid]
            valid = [c for c in items if c.winner in ("speculative", "baseline", "tie")]

            if not valid:
                summaries.append({
                    "prompt_id": pid,
                    "category": items[0].category if items else "",
                    "n_trials": len(items),
                    "avg_speedup": 0.0,
                    "avg_ttft_improvement_ms": 0.0,
                    "avg_latency_improvement_ms": 0.0,
                    "avg_tps_improvement": 0.0,
                    "winner": "inconclusive",
                })
                continue

            avg_spd = statistics.mean(c.speedup for c in valid)
            avg_ttft = statistics.mean(c.baseline_ttft - c.speculative_ttft for c in valid)
            avg_lat = statistics.mean(c.baseline_latency - c.speculative_latency for c in valid)
            avg_tps = statistics.mean(
                c.speculative_tokens_per_second - c.baseline_tokens_per_second for c in valid
            )

            spec_wins = sum(1 for c in valid if c.winner == "speculative")
            base_wins = sum(1 for c in valid if c.winner == "baseline")
            if spec_wins > base_wins:
                overall_winner = "speculative"
            elif base_wins > spec_wins:
                overall_winner = "baseline"
            else:
                overall_winner = "tie"

            summaries.append({
                "prompt_id": pid,
                "category": valid[0].category,
                "n_trials": len(items),
                "avg_speedup": round(avg_spd, 4),
                "avg_ttft_improvement_ms": round(avg_ttft, 2),
                "avg_latency_improvement_ms": round(avg_lat, 2),
                "avg_tps_improvement": round(avg_tps, 2),
                "winner": overall_winner,
            })

        return summaries


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
        prog="compare_speculative",
        description="Compare baseline vs speculative decoding across the same prompt dataset.",
    )
    parser.add_argument(
        "--baseline-url",
        type=str,
        required=True,
        help="Base URL of the baseline llama-server.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        required=True,
        help="Model name for the baseline server.",
    )
    parser.add_argument(
        "--speculative-url",
        type=str,
        required=True,
        help="Base URL of the speculative-decoding llama-server.",
    )
    parser.add_argument(
        "--speculative-model",
        type=str,
        required=True,
        help="Model name for the speculative server.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per server (default: 3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response (default: 512).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results/).",
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

    config = SpeculativeConfig(
        baseline_server_url=args.baseline_url,
        baseline_model=args.baseline_model,
        speculative_server_url=args.speculative_url,
        speculative_model=args.speculative_model,
        trials=args.trials,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        output_directory=args.output,
    )

    try:
        config.validate()
    except ValueError as e:
        logger.error("Invalid configuration: %s", e)
        return 1

    comparator = SpeculativeComparator(config)

    try:
        logger.info("Starting baseline benchmark ...")
        baseline_results = comparator.run_baseline()

        logger.info("Starting speculative benchmark ...")
        speculative_results = comparator.run_speculative()

        comparator.compare(baseline_results, speculative_results)
        comparator.print_comparison()
        paths = comparator.save_results()
        logger.info("Results saved to: %s", ", ".join(str(p) for p in paths))

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Comparison failed")
        return 1

    if comparator.summary and comparator.summary.valid_comparisons > 0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
