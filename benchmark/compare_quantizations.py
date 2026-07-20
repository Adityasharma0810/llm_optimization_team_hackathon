#!/usr/bin/env python3
"""benchmark/compare_quantizations.py — Quantization format comparison.

Benchmarks multiple quantized models served by the same llama-server
(or different servers) via benchmark.BenchmarkRunner, then computes
per-model aggregate statistics and a ranked leaderboard.

Each model name in --models is sent as the ``model`` field in the
/v1/chat/completions request.  The server must already be loaded with
the corresponding GGUF file (via ``--model`` at server startup or
via the server's model-switching API).

Usage:
    python -m benchmark.compare_quantizations \\
        --server-url http://localhost:8080 \\
        --models Q4_K_M Q5_K_M Q8_0 \\
        --trials 3

    python benchmark/compare_quantizations.py \\
        --server-url http://10.0.0.1:8080 \\
        --models Q4_K_M Q5_K_M Q6_K Q8_0 \\
        --trials 5 \\
        --output /tmp/quant_compare/
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
class QuantizationConfig:
    """Configuration for a quantization comparison sweep."""

    server_url: str
    models: list[str]
    trials: int = 3
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 120.0
    output_directory: Path = Path("results")

    def validate(self) -> None:
        """Validate configuration, raising ValueError on failure."""
        if not self.server_url:
            raise ValueError("server_url must not be empty")
        if not self.models:
            raise ValueError("At least one model must be specified")
        for m in self.models:
            if not m:
                raise ValueError("Model names must not be empty strings")
        if self.trials < 1:
            raise ValueError(f"trials must be >= 1, got {self.trials}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {self.temperature}")

    def benchmark_config_for(self, model_name: str) -> BenchmarkConfig:
        """Build a BenchmarkConfig for a specific model.

        Args:
            model_name: The model name to benchmark.

        Returns:
            BenchmarkConfig with save_json/save_csv disabled
            (the comparator handles output).
        """
        return BenchmarkConfig(
            server_url=self.server_url,
            model_name=model_name,
            trials=self.trials,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            output_directory=self.output_directory,
            save_json=False,
            save_csv=False,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Per-prompt result (extends BenchmarkResult with response_length)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class QuantizationResult:
    """Result of benchmarking a single prompt with a single quantization."""

    model: str
    prompt_id: str
    category: str
    trial: int

    ttft: float
    latency: float
    tokens_per_second: float
    cpu_usage: float
    memory_usage: float
    duration: float
    response_length: int

    status: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Per-model aggregate
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelAggregate:
    """Aggregated statistics for a single quantization model."""

    model: str
    total_prompts: int = 0
    successful: int = 0
    failed: int = 0
    success_rate: float = 0.0

    avg_ttft_ms: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_cpu_usage: float = 0.0
    avg_memory_usage_mb: float = 0.0
    avg_duration_sec: float = 0.0
    avg_response_length: float = 0.0

    median_ttft_ms: float = 0.0
    median_latency_ms: float = 0.0
    median_tokens_per_second: float = 0.0

    p95_ttft_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p95_tokens_per_second: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Leaderboard entry
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class LeaderboardEntry:
    """A single row in the quantization leaderboard."""

    rank: int
    model: str
    avg_tokens_per_second: float
    avg_ttft_ms: float
    avg_latency_ms: float
    avg_memory_usage_mb: float
    success_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: safe statistics
# ──────────────────────────────────────────────────────────────────────────────


def _safe_mean(values: list[float]) -> float:
    """Compute mean of a list, returning 0.0 for empty lists."""
    return statistics.mean(values) if values else 0.0


def _safe_median(values: list[float]) -> float:
    """Compute median of a list, returning 0.0 for empty lists."""
    return statistics.median(values) if values else 0.0


def _safe_p95(values: list[float]) -> float:
    """Compute 95th percentile of a list, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = 0.95 * (len(sorted_vals) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ──────────────────────────────────────────────────────────────────────────────
# Comparator
# ──────────────────────────────────────────────────────────────────────────────


class QuantizationComparator:
    """Orchestrates quantization comparison across multiple models.

    Iterates over model names, creates a BenchmarkRunner for each,
    collects results, and computes aggregate statistics and rankings.
    """

    def __init__(self, config: QuantizationConfig) -> None:
        """Initialize the comparator.

        Args:
            config: Quantization comparison configuration.
        """
        self.config = config
        self.results: dict[str, list[QuantizationResult]] = defaultdict(list)
        self.aggregates: dict[str, ModelAggregate] = {}
        self.leaderboard: list[LeaderboardEntry] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Runners
    # ──────────────────────────────────────────────────────────────────────────

    def run_model(self, model_name: str) -> list[QuantizationResult]:
        """Benchmark a single quantized model via BenchmarkRunner.

        Creates a BenchmarkRunner configured for the given model name,
        runs all trials, and converts BenchmarkResult objects into
        QuantizationResult objects.

        Args:
            model_name: The model name to benchmark.

        Returns:
            List of QuantizationResult for this model.
        """
        logger.info(
            "Running model: %s (trials=%d, server=%s)",
            model_name,
            self.config.trials,
            self.config.server_url,
        )

        b_config = self.config.benchmark_config_for(model_name)
        runner = BenchmarkRunner(b_config)

        try:
            benchmark_results = runner.run_all()
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("Model %s failed entirely", model_name)
            return []

        q_results: list[QuantizationResult] = []
        for br in benchmark_results:
            q_results.append(
                QuantizationResult(
                    model=model_name,
                    prompt_id=br.prompt_id,
                    category=br.category,
                    trial=br.trial,
                    ttft=br.ttft,
                    latency=br.latency,
                    tokens_per_second=br.tokens_per_second,
                    cpu_usage=br.cpu_usage,
                    memory_usage=br.memory_usage,
                    duration=br.duration,
                    response_length=len(br.response),
                    status=br.status,
                    error=br.error,
                )
            )

        succeeded = sum(1 for r in q_results if r.status == "success")
        logger.info(
            "Model %s complete: %d/%d succeeded",
            model_name,
            succeeded,
            len(q_results),
        )
        return q_results

    def run_all_models(self) -> dict[str, list[QuantizationResult]]:
        """Run benchmarks for every model in the configuration.

        Continues to the next model if one fails entirely.

        Returns:
            Dict mapping model name to its list of QuantizationResult.
        """
        logger.info(
            "Starting quantization comparison: %d models × %d trials",
            len(self.config.models),
            self.config.trials,
        )

        for model_name in self.config.models:
            try:
                results = self.run_model(model_name)
                self.results[model_name] = results
            except KeyboardInterrupt:
                logger.warning("Interrupted — saving partial results for %d models", len(self.results))
                break
            except Exception:
                logger.exception("Unexpected failure for model %s", model_name)
                self.results[model_name] = []

        return dict(self.results)

    # ──────────────────────────────────────────────────────────────────────────
    # Compare / aggregate
    # ──────────────────────────────────────────────────────────────────────────

    def compare(self) -> dict[str, ModelAggregate]:
        """Compute aggregated statistics for each model and build the leaderboard.

        Must be called after run_all_models().

        Returns:
            Dict mapping model name to its ModelAggregate.
        """
        self.aggregates = {}

        for model_name, model_results in self.results.items():
            successful = [r for r in model_results if r.status == "success"]
            total = len(model_results)
            n_ok = len(successful)

            agg = ModelAggregate(
                model=model_name,
                total_prompts=total,
                successful=n_ok,
                failed=total - n_ok,
                success_rate=n_ok / total if total > 0 else 0.0,
            )

            if successful:
                agg.avg_ttft_ms = _safe_mean([r.ttft for r in successful])
                agg.avg_latency_ms = _safe_mean([r.latency for r in successful])
                agg.avg_tokens_per_second = _safe_mean([r.tokens_per_second for r in successful])
                agg.avg_cpu_usage = _safe_mean([r.cpu_usage for r in successful])
                agg.avg_memory_usage_mb = _safe_mean([r.memory_usage for r in successful])
                agg.avg_duration_sec = _safe_mean([r.duration for r in successful])
                agg.avg_response_length = _safe_mean([float(r.response_length) for r in successful])

                agg.median_ttft_ms = _safe_median([r.ttft for r in successful])
                agg.median_latency_ms = _safe_median([r.latency for r in successful])
                agg.median_tokens_per_second = _safe_median([r.tokens_per_second for r in successful])

                agg.p95_ttft_ms = _safe_p95([r.ttft for r in successful])
                agg.p95_latency_ms = _safe_p95([r.latency for r in successful])
                agg.p95_tokens_per_second = _safe_p95([r.tokens_per_second for r in successful])

            self.aggregates[model_name] = agg

        self.leaderboard = self._build_leaderboard()
        return self.aggregates

    def _build_leaderboard(self) -> list[LeaderboardEntry]:
        """Build a ranked leaderboard sorted by primary TPS, then latency.

        Ranking criteria applied in order:
            1. TPS (highest first)
            2. Latency (lowest first)
            3. TTFT (lowest first)
            4. Memory (lowest first)

        Only models with at least one successful result are ranked.

        Returns:
            Sorted list of LeaderboardEntry with 1-based ranks.
        """
        ranked: list[LeaderboardEntry] = []

        for model_name, agg in self.aggregates.items():
            if agg.successful == 0:
                continue
            ranked.append(
                LeaderboardEntry(
                    rank=0,  # placeholder, set below
                    model=model_name,
                    avg_tokens_per_second=agg.avg_tokens_per_second,
                    avg_ttft_ms=agg.avg_ttft_ms,
                    avg_latency_ms=agg.avg_latency_ms,
                    avg_memory_usage_mb=agg.avg_memory_usage_mb,
                    success_rate=agg.success_rate,
                )
            )

        ranked.sort(
            key=lambda e: (
                -e.avg_tokens_per_second,   # highest TPS first
                e.avg_latency_ms,           # lowest latency first
                e.avg_ttft_ms,              # lowest TTFT first
                e.avg_memory_usage_mb,      # lowest memory first
            ),
        )

        for i, entry in enumerate(ranked):
            entry.rank = i + 1

        return ranked

    # ──────────────────────────────────────────────────────────────────────────
    # Print
    # ──────────────────────────────────────────────────────────────────────────

    def print_comparison(self) -> None:
        """Print a formatted comparison table and leaderboard to stdout."""
        if not self.aggregates:
            logger.warning("No aggregates to print — run compare() first")
            return

        print()
        print("=" * 82)
        print("  QUANTIZATION COMPARISON RESULTS")
        print("=" * 82)
        print(f"  Server  : {self.config.server_url}")
        print(f"  Trials  : {self.config.trials}")
        print(f"  Models  : {len(self.config.models)}")
        print("-" * 82)

        # Per-model summary table
        print(f"  {'Model':<14} {'OK':>5} {'TPS':>9} {'TTFT ms':>9} {'Lat ms':>9} {'Mem MB':>9} {'Dur s':>8}")
        print("  " + "-" * 64)

        for model_name in self.config.models:
            agg = self.aggregates.get(model_name)
            if agg is None:
                print(f"  {model_name:<14} {'--':>5} {'--':>9} {'--':>9} {'--':>9} {'--':>9} {'--':>8}")
                continue
            print(
                f"  {model_name:<14} "
                f"{agg.successful:>5} "
                f"{agg.avg_tokens_per_second:>9.2f} "
                f"{agg.avg_ttft_ms:>9.1f} "
                f"{agg.avg_latency_ms:>9.1f} "
                f"{agg.avg_memory_usage_mb:>9.1f} "
                f"{agg.avg_duration_sec:>8.2f}"
            )

        # Leaderboard
        if self.leaderboard:
            print()
            print("  LEADERBOARD (ranked by TPS → Latency → TTFT → Memory)")
            print("  " + "-" * 64)
            print(f"  {'Rank':>4}  {'Model':<14} {'TPS':>9} {'TTFT ms':>9} {'Lat ms':>9} {'Mem MB':>9}")
            print("  " + "-" * 64)
            for entry in self.leaderboard:
                medal = ""
                if entry.rank == 1:
                    medal = " *"
                print(
                    f"  {entry.rank:>4}  {entry.model:<14} "
                    f"{entry.avg_tokens_per_second:>9.2f} "
                    f"{entry.avg_ttft_ms:>9.1f} "
                    f"{entry.avg_latency_ms:>9.1f} "
                    f"{entry.avg_memory_usage_mb:>9.1f}{medal}"
                )

        print("=" * 82)
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # Save outputs
    # ──────────────────────────────────────────────────────────────────────────

    def save_results(self) -> list[Path]:
        """Save comparison results to CSV, JSON, and leaderboard JSON.

        Returns:
            List of paths to written files.
        """
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        paths: list[Path] = []

        # ── CSV ──────────────────────────────────────────────────────────────
        csv_path = self.config.output_directory / "quantization_results.csv"
        fieldnames = [
            "model", "prompt_id", "category", "trial",
            "ttft", "latency", "tokens_per_second",
            "cpu_usage", "memory_usage", "duration",
            "response_length", "status", "error",
        ]

        all_results = [
            r for model_results in self.results.values() for r in model_results
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: v for k, v in r.to_dict().items() if k in fieldnames})

        logger.info("CSV saved to %s", csv_path)
        paths.append(csv_path)

        # ── JSON (per-prompt) ────────────────────────────────────────────────
        json_path = self.config.output_directory / "quantization_results.json"
        payload = {
            "metadata": {
                "server_url": self.config.server_url,
                "models": self.config.models,
                "trials": self.config.trials,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "aggregates": {
                name: agg.to_dict() for name, agg in self.aggregates.items()
            },
            "results": [r.to_dict() for r in all_results],
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("JSON saved to %s", json_path)
        paths.append(json_path)

        # ── Leaderboard JSON ─────────────────────────────────────────────────
        leaderboard_path = self.config.output_directory / "leaderboard.json"
        lb_payload = {
            "metadata": payload["metadata"],
            "leaderboard": [entry.to_dict() for entry in self.leaderboard],
            "ranking_criteria": [
                "tokens_per_second (highest)",
                "latency (lowest)",
                "ttft (lowest)",
                "memory (lowest)",
            ],
        }
        leaderboard_path.write_text(json.dumps(lb_payload, indent=2) + "\n")
        logger.info("Leaderboard saved to %s", leaderboard_path)
        paths.append(leaderboard_path)

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
        prog="compare_quantizations",
        description="Compare inference across multiple quantized models.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="Base URL of the llama-server.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to compare (e.g. Q4_K_M Q5_K_M Q8_0).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per model (default: 3).",
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

    config = QuantizationConfig(
        server_url=args.server_url,
        models=args.models,
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

    comparator = QuantizationComparator(config)

    try:
        comparator.run_all_models()
        comparator.compare()
        comparator.print_comparison()
        paths = comparator.save_results()
        logger.info("Results saved to: %s", ", ".join(str(p) for p in paths))

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Quantization comparison failed")
        return 1

    any_success = any(
        agg.successful > 0 for agg in comparator.aggregates.values()
    )
    return 0 if any_success else 1


if __name__ == "__main__":
    sys.exit(main())
