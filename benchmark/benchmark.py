#!/usr/bin/env python3
"""benchmark/benchmark.py — HTTP-based benchmark runner for llama-server.

Sends prompts to a running llama-server instance via the OpenAI-compatible
/v1/chat/completions endpoint, measures throughput, latency, memory, and
CPU usage using evaluation.metrics, and saves results as JSON and CSV.

Usage:
    python -m benchmark.benchmark \\
        --server-url http://localhost:8080 \\
        --model-name qwen2.5 \\
        --trials 3 \\
        --output results/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running as a direct script: `python benchmark/benchmark.py`
# When invoked this way the project root is not on sys.path, so sibling
# packages (evaluation, etc.) cannot be found.  We detect the situation
# and insert the project root automatically.
if __package__ is None or __package__ == "":
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

import requests  # noqa: E402

from evaluation.metrics import (  # noqa: E402
    compute_cpu_usage,
    compute_latency,
    compute_memory_usage,
    compute_summary_statistics,
    compute_tokens_per_second,
    compute_ttft,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    All fields map directly to CLI arguments or constructor parameters.
    """

    server_url: str
    model_name: str
    trials: int = 3
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 120.0
    output_directory: Path = Path("results")
    save_json: bool = True
    save_csv: bool = True

    def validate(self) -> None:
        """Validate configuration, raising ValueError on failure."""
        if not self.server_url:
            raise ValueError("server_url must not be empty")
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.trials < 1:
            raise ValueError(f"trials must be >= 1, got {self.trials}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {self.temperature}")


@dataclass
class PromptRecord:
    """A single evaluation prompt loaded from prompts.json."""

    id: str
    category: str
    name: str
    prompt: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to dictionary."""
        return {"id": self.id, "category": self.category, "name": self.name, "prompt": self.prompt}


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single prompt for a single trial."""

    prompt_id: str
    category: str
    trial: int
    response: str
    status: str
    tokens_generated: int
    timestamps: list[float]
    ttft: float
    latency: float
    tokens_per_second: float
    cpu_usage: float
    memory_usage: float
    duration: float
    error: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for CSV/JSON output."""
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "trial": self.trial,
            "response": self.response,
            "status": self.status,
            "tokens_generated": self.tokens_generated,
            "timestamps": json.dumps(self.timestamps),
            "ttft": self.ttft,
            "latency": self.latency,
            "tokens_per_second": self.tokens_per_second,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "duration": self.duration,
            "error": self.error,
        }


# ──────────────────────────────────────────────────────────────────────────────
# BenchmarkRunner
# ──────────────────────────────────────────────────────────────────────────────


class BenchmarkRunner:
    """Orchestrates benchmarking of an LLM server via HTTP.

    Loads prompts from evaluation/prompts.json, sends each prompt to
    the server, measures performance metrics using evaluation.metrics,
    and saves structured results.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self.results: list[BenchmarkResult] = []
        self._session = requests.Session()

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt loading
    # ──────────────────────────────────────────────────────────────────────────

    def load_prompts(self) -> list[PromptRecord]:
        """Load evaluation prompts from evaluation/prompts.json.

        Returns:
            List of PromptRecord objects.

        Raises:
            FileNotFoundError: If prompts.json does not exist.
            json.JSONDecodeError: If prompts.json is malformed.
            KeyError: If required fields are missing from a prompt.
        """
        dataset_path = Path(__file__).resolve().parent.parent / "evaluation" / "prompts.json"
        logger.info("Loading prompts from %s", dataset_path)

        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)

        prompts: list[PromptRecord] = []
        for entry in data["prompts"]:
            record = PromptRecord(
                id=entry["id"],
                category=entry["category"],
                name=entry["name"],
                prompt=entry["prompt"],
            )
            prompts.append(record)

        logger.info("Loaded %d prompts across %d categories", len(prompts), len(data.get("categories", [])))
        return prompts

    # ──────────────────────────────────────────────────────────────────────────
    # HTTP request
    # ──────────────────────────────────────────────────────────────────────────

    def send_request(
        self,
        prompt_text: str,
    ) -> dict[str, Any]:
        """Send a streaming chat completion request and collect per-token timestamps.

        Opens a streaming connection to the server, reads SSE chunks,
        records the wall-clock timestamp of each token as it arrives,
        and returns the complete response text plus raw timing data.

        Args:
            prompt_text: The user prompt string.

        Returns:
            Dict with keys:
                - response_text: str — the complete generated text.
                - timestamps: list[float] — per-token elapsed ms from request start.
                - tokens: int — number of generated tokens.
                - duration_sec: float — total wall-clock seconds.
                - prompt_tokens: int — prompt token count from usage.
                - completion_tokens: int — completion token count from usage.

        Raises:
            requests.Timeout: If the request exceeds self.config.timeout.
            requests.ConnectionError: If the server is unreachable.
            requests.HTTPError: If the server returns a non-2xx status.
            ValueError: If the response body cannot be parsed as JSON.
        """
        url = f"{self.config.server_url.rstrip('/')}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        logger.debug("POST %s (streaming)", url)
        t_start = time.perf_counter()

        response = self._session.post(url, json=payload, timeout=self.config.timeout, stream=True)
        response.raise_for_status()

        collected_text = ""
        timestamps_ms: list[float] = []
        prompt_tokens = 0
        completion_tokens = 0

        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
            line = line.strip()
            if not line or line == "data: [DONE]":
                continue
            if not line.startswith("data: "):
                continue

            payload_str = line[len("data: "):]
            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse SSE chunk: %s", e)
                continue

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content is not None:
                    collected_text += content
                    timestamps_ms.append(elapsed_ms)

            usage = chunk.get("usage")
            if usage is not None:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

        duration_sec = time.perf_counter() - t_start

        if completion_tokens == 0:
            completion_tokens = len(collected_text.split())

        return {
            "response_text": collected_text,
            "timestamps": timestamps_ms,
            "tokens": len(timestamps_ms) if timestamps_ms else completion_tokens,
            "duration_sec": duration_sec,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Single prompt benchmark
    # ──────────────────────────────────────────────────────────────────────────

    def run_prompt(
        self,
        prompt: PromptRecord,
        trial: int,
    ) -> BenchmarkResult:
        """Benchmark a single prompt for one trial.

        Sends the prompt, collects raw timing data, computes metrics
        using evaluation.metrics, and returns a BenchmarkResult.
        All metric computations are delegated to evaluation.metrics.

        Args:
            prompt: The PromptRecord to benchmark.
            trial: Current trial number (0-indexed).

        Returns:
            BenchmarkResult with all metrics populated.
        """
        logger.info("[%s] trial=%d category=%s", prompt.id, trial, prompt.category)

        try:
            raw = self.send_request(prompt.prompt)
        except requests.Timeout as e:
            logger.error("[%s] Timeout after %.0fs: %s", prompt.id, self.config.timeout, e)
            return self._error_result(prompt, trial, "timeout", str(e))
        except requests.ConnectionError as e:
            logger.error("[%s] Connection refused: %s", prompt.id, e)
            return self._error_result(prompt, trial, "connection_error", str(e))
        except requests.HTTPError as e:
            logger.error("[%s] HTTP error %s", prompt.id, e)
            return self._error_result(prompt, trial, "http_error", str(e))
        except ValueError as e:
            logger.error("[%s] Invalid response: %s", prompt.id, e)
            return self._error_result(prompt, trial, "invalid_response", str(e))
        except Exception as e:
            logger.error("[%s] Unexpected error: %s", prompt.id, e)
            return self._error_result(prompt, trial, "error", str(e))

        timestamps = raw["timestamps"]
        duration_sec = raw["duration_sec"]
        tokens = raw["tokens"]

        # Compute metrics using evaluation.metrics — no logic duplicated here
        try:
            ttft = compute_ttft(timestamps) if timestamps else 0.0
        except (ValueError, IndexError):
            ttft = 0.0

        try:
            latency_result = compute_latency(timestamps, total_time_ms=duration_sec * 1000.0)
            latency_val = latency_result.itl_mean_ms
        except (ValueError, IndexError):
            latency_val = 0.0

        try:
            tps = compute_tokens_per_second(tokens, duration_sec)
        except ValueError:
            tps = 0.0

        try:
            cpu = compute_cpu_usage(interval=0.0)
            cpu_val = cpu.cpu_percent_process
        except Exception:
            cpu_val = 0.0

        try:
            mem = compute_memory_usage(generated_tokens=tokens)
            mem_val = mem.rss_current_mb
        except Exception:
            mem_val = 0.0

        logger.info(
            "[%s] tokens=%d ttft=%.1fms tps=%.1f duration=%.2fs",
            prompt.id,
            tokens,
            ttft,
            tps,
            duration_sec,
        )

        return BenchmarkResult(
            prompt_id=prompt.id,
            category=prompt.category,
            trial=trial,
            response=raw["response_text"],
            status="success",
            tokens_generated=tokens,
            timestamps=timestamps,
            ttft=ttft,
            latency=latency_val,
            tokens_per_second=tps,
            cpu_usage=cpu_val,
            memory_usage=mem_val,
            duration=duration_sec,
            error="",
        )

    def _error_result(
        self,
        prompt: PromptRecord,
        trial: int,
        status: str,
        error_msg: str,
    ) -> BenchmarkResult:
        """Build a BenchmarkResult representing a failed prompt.

        Args:
            prompt: The prompt that failed.
            trial: Current trial number.
            status: Error status string.
            error_msg: Human-readable error message.

        Returns:
            BenchmarkResult with zeroed metrics and error populated.
        """
        return BenchmarkResult(
            prompt_id=prompt.id,
            category=prompt.category,
            trial=trial,
            response="",
            status=status,
            tokens_generated=0,
            timestamps=[],
            ttft=0.0,
            latency=0.0,
            tokens_per_second=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            duration=0.0,
            error=error_msg,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Trial and full run
    # ──────────────────────────────────────────────────────────────────────────

    def run_trial(
        self,
        prompts: list[PromptRecord],
        trial: int,
    ) -> list[BenchmarkResult]:
        """Execute one full trial: benchmark every prompt once.

        Args:
            prompts: Prompts to benchmark.
            trial: Trial number (0-indexed).

        Returns:
            List of BenchmarkResult for this trial.
        """
        trial_results: list[BenchmarkResult] = []
        logger.info("=== Trial %d/%d (%d prompts) ===", trial + 1, self.config.trials, len(prompts))

        for i, prompt in enumerate(prompts):
            logger.info("  [%d/%d] %s — %s", i + 1, len(prompts), prompt.id, prompt.name)
            result = self.run_prompt(prompt, trial)
            trial_results.append(result)

        succeeded = sum(1 for r in trial_results if r.status == "success")
        logger.info("Trial %d complete: %d/%d succeeded", trial + 1, succeeded, len(prompts))
        return trial_results

    def run_all(self) -> list[BenchmarkResult]:
        """Run the full benchmark: load prompts, execute all trials, aggregate.

        Returns:
            List of all BenchmarkResult across all trials.

        Raises:
            KeyboardInterrupt: Re-raised after saving partial results.
            RuntimeError: If no prompts could be loaded.
        """
        prompts = self.load_prompts()
        if not prompts:
            raise RuntimeError("No prompts loaded from evaluation/prompts.json")

        all_results: list[BenchmarkResult] = []

        try:
            for trial in range(self.config.trials):
                trial_results = self.run_trial(prompts, trial)
                all_results.extend(trial_results)
        except KeyboardInterrupt:
            logger.warning("Benchmark interrupted by user — saving %d partial results", len(all_results))

        self.results = all_results
        self._print_summary()

        if self.config.save_json:
            self.save_json()
        if self.config.save_csv:
            self.save_csv()

        return all_results

    # ──────────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        """Print aggregated benchmark summary to stdout."""
        if not self.results:
            logger.warning("No results to summarize")
            return

        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status != "success"]

        if not successful:
            logger.warning("All prompts failed — no metrics to summarize")
            return

        tps_values = [r.tokens_per_second for r in successful]
        ttft_values = [r.ttft for r in successful]
        latency_values = [r.latency for r in successful]
        duration_values = [r.duration for r in successful]

        tps_stats = compute_summary_statistics(tps_values)
        ttft_stats = compute_summary_statistics(ttft_values)
        lat_stats = compute_summary_statistics(latency_values)
        dur_stats = compute_summary_statistics(duration_values)

        print()
        print("=" * 72)
        print("  BENCHMARK SUMMARY")
        print("=" * 72)
        print(f"  Model:              {self.config.model_name}")
        print(f"  Server:             {self.config.server_url}")
        print(f"  Trials:             {self.config.trials}")
        print(f"  Prompts:            {len(set(r.prompt_id for r in self.results))}")
        print(f"  Successful:         {len(successful)}")
        print(f"  Failed:             {len(failed)}")
        print("-" * 72)
        print("  TOKENS/SECOND")
        print(f"    Mean:             {tps_stats.mean:.2f}")
        print(f"    Median:           {tps_stats.median:.2f}")
        print(f"    Std:              {tps_stats.std:.2f}")
        print(f"    Min:              {tps_stats.min:.2f}")
        print(f"    Max:              {tps_stats.max:.2f}")
        print("-" * 72)
        print("  TIME TO FIRST TOKEN (ms)")
        print(f"    Mean:             {ttft_stats.mean:.1f}")
        print(f"    Median:           {ttft_stats.median:.1f}")
        print(f"    p95:              {ttft_stats.p95:.1f}")
        print("-" * 72)
        print("  INTER-TOKEN LATENCY (ms)")
        print(f"    Mean:             {lat_stats.mean:.1f}")
        print(f"    Median:           {lat_stats.median:.1f}")
        print(f"    p95:              {lat_stats.p95:.1f}")
        print("-" * 72)
        print("  DURATION (s)")
        print(f"    Mean:             {dur_stats.mean:.2f}")
        print(f"    Median:           {dur_stats.median:.2f}")
        print("=" * 72)
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # Output: JSON
    # ──────────────────────────────────────────────────────────────────────────

    def save_json(self) -> Path:
        """Save all benchmark results to a JSON file.

        Returns:
            Path to the written file.
        """
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"benchmark_results_{timestamp}.json"
        output_path = self.config.output_directory / filename

        successful = [r for r in self.results if r.status == "success"]

        aggregate: dict[str, Any] = {}
        if successful:
            tps_vals = [r.tokens_per_second for r in successful]
            ttft_vals = [r.ttft for r in successful]
            lat_vals = [r.latency for r in successful]
            dur_vals = [r.duration for r in successful]

            tps_s = compute_summary_statistics(tps_vals)
            ttft_s = compute_summary_statistics(ttft_vals)
            lat_s = compute_summary_statistics(lat_vals)
            dur_s = compute_summary_statistics(dur_vals)

            aggregate = {
                "tokens_per_second": tps_s.to_dict(),
                "ttft_ms": ttft_s.to_dict(),
                "inter_token_latency_ms": lat_s.to_dict(),
                "duration_sec": dur_s.to_dict(),
            }

        payload = {
            "metadata": {
                "server_url": self.config.server_url,
                "model_name": self.config.model_name,
                "trials": self.config.trials,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_prompts": len(set(r.prompt_id for r in self.results)),
                "total_results": len(self.results),
                "successful": sum(1 for r in self.results if r.status == "success"),
                "failed": sum(1 for r in self.results if r.status != "success"),
            },
            "aggregate": aggregate,
            "results": [r.to_dict() for r in self.results],
        }

        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("Results saved to %s", output_path)
        return output_path

    # ──────────────────────────────────────────────────────────────────────────
    # Output: CSV
    # ──────────────────────────────────────────────────────────────────────────

    def save_csv(self) -> Path:
        """Save all benchmark results to a CSV file (one row per result).

        Returns:
            Path to the written file.
        """
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"benchmark_results_{timestamp}.csv"
        output_path = self.config.output_directory / filename

        if not self.results:
            logger.warning("No results to write to CSV")
            output_path.touch()
            return output_path

        fieldnames = [
            "prompt_id",
            "category",
            "trial",
            "status",
            "tokens_generated",
            "ttft",
            "latency",
            "tokens_per_second",
            "cpu_usage",
            "memory_usage",
            "duration",
            "error",
            "response",
            "timestamps",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                row = result.to_dict()
                writer.writerow({k: row[k] for k in fieldnames})

        logger.info("CSV saved to %s", output_path)
        return output_path


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
        prog="benchmark",
        description="Benchmark an LLM server using evaluation prompts and metrics.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="Base URL of the llama-server (e.g. http://localhost:8080).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name to send in the API request.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials to run (default: 3).",
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
        help="Maximum tokens to generate per response (default: 512).",
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
        "--no-json",
        action="store_true",
        help="Skip JSON output.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV output.",
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

    config = BenchmarkConfig(
        server_url=args.server_url,
        model_name=args.model_name,
        trials=args.trials,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        output_directory=args.output,
        save_json=not args.no_json,
        save_csv=not args.no_csv,
    )

    try:
        config.validate()
    except ValueError as e:
        logger.error("Invalid configuration: %s", e)
        return 1

    runner = BenchmarkRunner(config)

    try:
        results = runner.run_all()
    except KeyboardInterrupt:
        logger.warning("Interrupted — partial results may have been saved")
        return 130
    except Exception:
        logger.exception("Benchmark failed")
        return 1

    succeeded = sum(1 for r in results if r.status == "success")
    logger.info("Benchmark complete: %d/%d succeeded", succeeded, len(results))
    return 0 if succeeded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
