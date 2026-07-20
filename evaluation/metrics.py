#!/usr/bin/env python3
"""evaluation/metrics.py — Pure metric computation functions for LLM inference evaluation.

Provides stateless, unit-test-friendly functions for computing throughput,
latency, memory usage, CPU usage, acceptance rate, and summary statistics.
No benchmark orchestration logic — only mathematical calculations.

Usage:
    from evaluation.metrics import compute_ttft, compute_tokens_per_second

    ttft = compute_ttft(token_timestamps_ms=[42.1, 68.3, 89.7, ...])
    tps = compute_tokens_per_second(n_tokens=128, elapsed_sec=5.6)
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence

import psutil

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LatencyResult:
    """Computed latency metrics from a single inference run.

    All time values are in milliseconds.
    """

    ttft_ms: float
    itl_mean_ms: float
    itl_median_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    itl_min_ms: float
    itl_max_ms: float
    itl_std_ms: float
    total_time_ms: float
    n_tokens: int

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to a flat dictionary."""
        return {
            "ttft_ms": self.ttft_ms,
            "itl_mean_ms": self.itl_mean_ms,
            "itl_median_ms": self.itl_median_ms,
            "itl_p95_ms": self.itl_p95_ms,
            "itl_p99_ms": self.itl_p99_ms,
            "itl_min_ms": self.itl_min_ms,
            "itl_max_ms": self.itl_max_ms,
            "itl_std_ms": self.itl_std_ms,
            "total_time_ms": self.total_time_ms,
            "n_tokens": self.n_tokens,
        }


@dataclass(frozen=True)
class ThroughputResult:
    """Computed throughput metrics."""

    prompt_tok_per_sec: float
    generation_tok_per_sec: float
    total_tok_per_sec: float
    prompt_tokens: int
    generated_tokens: int
    prompt_time_sec: float
    generation_time_sec: float

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to a flat dictionary."""
        return {
            "prompt_tok_per_sec": self.prompt_tok_per_sec,
            "generation_tok_per_sec": self.generation_tok_per_sec,
            "total_tok_per_sec": self.total_tok_per_sec,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "prompt_time_sec": self.prompt_time_sec,
            "generation_time_sec": self.generation_time_sec,
        }


@dataclass(frozen=True)
class MemoryResult:
    """Computed memory usage metrics."""

    rss_current_mb: float
    rss_peak_mb: float
    vms_current_mb: float
    vms_peak_mb: float
    rss_percent: float
    model_size_mb: float
    memory_efficiency_tok_per_mb: float

    def to_dict(self) -> dict[str, float]:
        """Serialize to a flat dictionary."""
        return {
            "rss_current_mb": self.rss_current_mb,
            "rss_peak_mb": self.rss_peak_mb,
            "vms_current_mb": self.vms_current_mb,
            "vms_peak_mb": self.vms_peak_mb,
            "rss_percent": self.rss_percent,
            "model_size_mb": self.model_size_mb,
            "memory_efficiency_tok_per_mb": self.memory_efficiency_tok_per_mb,
        }


@dataclass(frozen=True)
class CpuResult:
    """Computed CPU usage metrics."""

    cpu_percent_process: float
    cpu_percent_system: float
    cpu_user_sec: float
    cpu_system_sec: float
    cpu_count_logical: int
    cpu_count_physical: int

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to a flat dictionary."""
        return {
            "cpu_percent_process": self.cpu_percent_process,
            "cpu_percent_system": self.cpu_percent_system,
            "cpu_user_sec": self.cpu_user_sec,
            "cpu_system_sec": self.cpu_system_sec,
            "cpu_count_logical": self.cpu_count_logical,
            "cpu_count_physical": self.cpu_count_physical,
        }


@dataclass(frozen=True)
class AcceptanceResult:
    """Computed acceptance rate metrics for speculative decoding."""

    total_draft_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    acceptance_rate: float
    rejection_rate: float
    mean_run_length: float
    max_run_length: int
    n_runs: int

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to a flat dictionary."""
        return {
            "total_draft_tokens": self.total_draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "acceptance_rate": self.acceptance_rate,
            "rejection_rate": self.rejection_rate,
            "mean_run_length": self.mean_run_length,
            "max_run_length": self.max_run_length,
            "n_runs": self.n_runs,
        }


@dataclass(frozen=True)
class SummaryStatistics:
    """Summary statistics (mean, std, min, max, percentiles) for a sample set."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    p5: float
    p25: float
    p75: float
    p95: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to a flat dictionary."""
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p5": self.p5,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "n": self.n,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Compute a percentile from an already-sorted sequence using linear interpolation.

    Args:
        sorted_values: Ascending-sorted numeric sequence.
        p: Percentile to compute, 0 <= p <= 100.

    Returns:
        Interpolated percentile value.

    Raises:
        ValueError: If sorted_values is empty or p is out of range.
    """
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty sequence")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"Percentile must be 0-100, got {p}")

    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    k = (p / 100.0) * (n - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_values[int(k)])
    fraction = k - lo
    return float(sorted_values[lo] * (1.0 - fraction) + sorted_values[hi] * fraction)


def _validate_positive(value: float, name: str) -> None:
    """Raise ValueError if value is not a finite positive number.

    Args:
        value: Number to validate.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is not positive or not finite.
    """
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")


def _validate_non_negative(value: float, name: str) -> None:
    """Raise ValueError if value is negative or not finite.

    Args:
        value: Number to validate.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is negative or not finite.
    """
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value}")


# ──────────────────────────────────────────────────────────────────────────────
# compute_ttft
# ──────────────────────────────────────────────────────────────────────────────


def compute_ttft(token_timestamps_ms: Sequence[float]) -> float:
    """Compute time to first token (TTFT) from per-token timestamps.

    TTFT is defined as the elapsed time from the start of inference
    to the moment the first token is emitted. This is typically the
    smallest meaningful latency measurement for interactive use.

    The first element of ``token_timestamps_ms`` is interpreted as the
    timestamp of the first generated token, measured in milliseconds
    from the start of the request.

    Args:
        token_timestamps_ms: Chronologically ordered timestamps in
            milliseconds for each generated token. The first element
            is the time at which token 0 was produced.

    Returns:
        Time to first token in milliseconds.

    Raises:
        ValueError: If the sequence is empty or contains non-finite values.

    Examples:
        >>> compute_ttft([42.0, 68.0, 91.0])
        42.0
        >>> compute_ttft([15.3])
        15.3
    """
    if not token_timestamps_ms:
        raise ValueError("token_timestamps_ms must not be empty")

    ttft = token_timestamps_ms[0]
    _validate_non_negative(ttft, "first timestamp (ttft)")

    logger.debug("TTFT: %.2f ms", ttft)
    return ttft


# ──────────────────────────────────────────────────────────────────────────────
# compute_latency
# ──────────────────────────────────────────────────────────────────────────────


def compute_latency(
    token_timestamps_ms: Sequence[float],
    total_time_ms: float | None = None,
) -> LatencyResult:
    """Compute comprehensive latency metrics from per-token timestamps.

    Calculates inter-token latency (ITL) as the difference between
    consecutive token timestamps, then derives mean, median, percentiles,
    and spread statistics.

    Args:
        token_timestamps_ms: Chronologically ordered timestamps in
            milliseconds for each generated token. Must contain at
            least one element. The first element is TTFT.
        total_time_ms: Total wall-clock inference time in milliseconds.
            If None, inferred as the last timestamp minus the first.

    Returns:
        LatencyResult with TTFT, ITL statistics, and total time.

    Raises:
        ValueError: If fewer than 2 timestamps, non-finite values,
            or timestamps are not monotonically non-decreasing.

    Examples:
        >>> timestamps = [50.0, 75.0, 105.0, 130.0, 160.0]
        >>> result = compute_latency(timestamps)
        >>> result.ttft_ms
        50.0
        >>> result.itl_mean_ms
        27.5
    """
    if len(token_timestamps_ms) < 1:
        raise ValueError(
            f"token_timestamps_ms must have at least 1 element, "
            f"got {len(token_timestamps_ms)}"
        )

    for i, ts in enumerate(token_timestamps_ms):
        if not math.isfinite(ts):
            raise ValueError(f"Non-finite value at index {i}: {ts}")

    if len(token_timestamps_ms) >= 2:
        for i in range(1, len(token_timestamps_ms)):
            if token_timestamps_ms[i] < token_timestamps_ms[i - 1]:
                raise ValueError(
                    f"Timestamps must be monotonically non-decreasing: "
                    f"index {i} ({token_timestamps_ms[i]}) < "
                    f"index {i - 1} ({token_timestamps_ms[i - 1]})"
                )

    ttft = token_timestamps_ms[0]

    if total_time_ms is None:
        total_time_ms = token_timestamps_ms[-1]
    else:
        _validate_positive(total_time_ms, "total_time_ms")

    n_tokens = len(token_timestamps_ms)

    if n_tokens < 2:
        itl_values: list[float] = []
    else:
        itl_values = [
            token_timestamps_ms[i] - token_timestamps_ms[i - 1]
            for i in range(1, n_tokens)
        ]

    if itl_values:
        itl_sorted = sorted(itl_values)
        itl_mean = statistics.mean(itl_values)
        itl_median = statistics.median(itl_values)
        itl_p95 = _percentile(itl_sorted, 95.0)
        itl_p99 = _percentile(itl_sorted, 99.0)
        itl_min = itl_sorted[0]
        itl_max = itl_sorted[-1]
        itl_std = statistics.stdev(itl_values) if len(itl_values) >= 2 else 0.0
    else:
        itl_mean = 0.0
        itl_median = 0.0
        itl_p95 = 0.0
        itl_p99 = 0.0
        itl_min = 0.0
        itl_max = 0.0
        itl_std = 0.0

    result = LatencyResult(
        ttft_ms=ttft,
        itl_mean_ms=itl_mean,
        itl_median_ms=itl_median,
        itl_p95_ms=itl_p95,
        itl_p99_ms=itl_p99,
        itl_min_ms=itl_min,
        itl_max_ms=itl_max,
        itl_std_ms=itl_std,
        total_time_ms=total_time_ms,
        n_tokens=n_tokens,
    )

    logger.debug(
        "Latency: TTFT=%.2f ms, ITL mean=%.2f ms, p95=%.2f ms, p99=%.2f ms",
        ttft,
        itl_mean,
        itl_p95,
        itl_p99,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# compute_tokens_per_second
# ──────────────────────────────────────────────────────────────────────────────


def compute_tokens_per_second(
    n_tokens: int,
    elapsed_sec: float,
) -> float:
    """Compute tokens per second (throughput) from a token count and elapsed time.

    Args:
        n_tokens: Number of tokens processed or generated. Must be >= 0.
        elapsed_sec: Elapsed wall-clock time in seconds. Must be positive.

    Returns:
        Tokens per second as a float. Returns 0.0 if n_tokens is 0.

    Raises:
        ValueError: If elapsed_sec is not positive or n_tokens is negative.

    Examples:
        >>> compute_tokens_per_second(128, 5.0)
        25.6
        >>> compute_tokens_per_second(0, 1.0)
        0.0
    """
    if n_tokens < 0:
        raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
    _validate_positive(elapsed_sec, "elapsed_sec")

    if n_tokens == 0:
        return 0.0

    tok_per_sec = float(n_tokens) / elapsed_sec
    logger.debug("Throughput: %d tokens / %.3f sec = %.2f tok/s", n_tokens, elapsed_sec, tok_per_sec)
    return tok_per_sec


def compute_throughput(
    prompt_tokens: int,
    generated_tokens: int,
    prompt_time_sec: float,
    generation_time_sec: float,
) -> ThroughputResult:
    """Compute prompt and generation throughput from raw timing data.

    Args:
        prompt_tokens: Number of prompt tokens processed.
        generated_tokens: Number of tokens generated.
        prompt_time_sec: Wall-clock time in seconds for prompt processing.
        generation_time_sec: Wall-clock time in seconds for token generation.

    Returns:
        ThroughputResult with prompt, generation, and total throughput.

    Raises:
        ValueError: If any time value is not positive or token counts are negative.

    Examples:
        >>> r = compute_throughput(512, 128, 10.0, 5.6)
        >>> round(r.prompt_tok_per_sec, 1)
        51.2
        >>> round(r.generation_tok_per_sec, 1)
        22.9
    """
    if prompt_tokens < 0:
        raise ValueError(f"prompt_tokens must be >= 0, got {prompt_tokens}")
    if generated_tokens < 0:
        raise ValueError(f"generated_tokens must be >= 0, got {generated_tokens}")
    _validate_positive(prompt_time_sec, "prompt_time_sec")
    _validate_positive(generation_time_sec, "generation_time_sec")

    ptps = compute_tokens_per_second(prompt_tokens, prompt_time_sec) if prompt_tokens > 0 else 0.0
    gtps = compute_tokens_per_second(generated_tokens, generation_time_sec) if generated_tokens > 0 else 0.0

    total_tokens = prompt_tokens + generated_tokens
    total_time = prompt_time_sec + generation_time_sec
    ttps = compute_tokens_per_second(total_tokens, total_time) if total_tokens > 0 else 0.0

    result = ThroughputResult(
        prompt_tok_per_sec=ptps,
        generation_tok_per_sec=gtps,
        total_tok_per_sec=ttps,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        prompt_time_sec=prompt_time_sec,
        generation_time_sec=generation_time_sec,
    )

    logger.debug(
        "Throughput: prompt=%.2f tok/s, generation=%.2f tok/s, total=%.2f tok/s",
        ptps,
        gtps,
        ttps,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# compute_memory_usage
# ──────────────────────────────────────────────────────────────────────────────


def compute_memory_usage(
    pid: int | None = None,
    model_size_mb: float = 0.0,
    generated_tokens: int = 0,
) -> MemoryResult:
    """Measure current memory usage of a process via psutil.

    Reads RSS, VMS, and system memory percentage for the given process.
    Optionally computes memory efficiency (tokens generated per MB of RSS).

    Args:
        pid: Process ID to measure. If None, measures the current process.
        model_size_mb: Model file size on disk in megabytes for reference.
            Must be >= 0.
        generated_tokens: Number of tokens generated during inference.
            Used to compute memory_efficiency_tok_per_mb.

    Returns:
        MemoryResult with current and peak memory measurements.

    Raises:
        psutil.NoSuchProcess: If the given PID does not exist.
        psutil.AccessDenied: If permission to read the process is denied.

    Examples:
        >>> result = compute_memory_usage()
        >>> result.rss_current_mb >= 0
        True
    """
    if model_size_mb < 0.0:
        raise ValueError(f"model_size_mb must be >= 0, got {model_size_mb}")
    if generated_tokens < 0:
        raise ValueError(f"generated_tokens must be >= 0, got {generated_tokens}")

    proc = psutil.Process(pid)

    mem_info = proc.memory_info()
    mem_full = proc.memory_full_info() if hasattr(proc, "memory_full_info") else mem_info

    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)

    rss_peak_mb = getattr(mem_full, "rss", mem_info.rss) / (1024 * 1024)
    vms_peak_mb = getattr(mem_full, "vms", mem_info.vms) / (1024 * 1024)

    mem_percent = proc.memory_percent()

    efficiency = 0.0
    if generated_tokens > 0 and rss_mb > 0.0:
        efficiency = float(generated_tokens) / rss_mb

    result = MemoryResult(
        rss_current_mb=rss_mb,
        rss_peak_mb=rss_peak_mb,
        vms_current_mb=vms_mb,
        vms_peak_mb=vms_peak_mb,
        rss_percent=mem_percent,
        model_size_mb=model_size_mb,
        memory_efficiency_tok_per_mb=efficiency,
    )

    logger.debug(
        "Memory: RSS=%.1f MB, VMS=%.1f MB, RSS%%=%.1f%%, efficiency=%.2f tok/MB",
        rss_mb,
        vms_mb,
        mem_percent,
        efficiency,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# compute_cpu_usage
# ──────────────────────────────────────────────────────────────────────────────


def compute_cpu_usage(
    pid: int | None = None,
    interval: float = 0.1,
) -> CpuResult:
    """Measure CPU usage of a process via psutil.

    Reads per-process CPU percentage, cumulative user/system time,
    and system CPU count.

    Args:
        pid: Process ID to measure. If None, measures the current process.
        interval: Seconds to wait for CPU percentage measurement.
            Pass 0.0 for non-blocking (percentage since last call).
            Default is 0.1 seconds.

    Returns:
        CpuResult with process and system CPU metrics.

    Raises:
        psutil.NoSuchProcess: If the given PID does not exist.
        psutil.AccessDenied: If permission to read the process is denied.

    Examples:
        >>> result = compute_cpu_usage(interval=0.0)
        >>> result.cpu_count_logical > 0
        True
    """
    proc = psutil.Process(pid)

    cpu_percent = proc.cpu_percent(interval=interval)
    cpu_times = proc.cpu_times()

    result = CpuResult(
        cpu_percent_process=cpu_percent,
        cpu_percent_system=psutil.cpu_percent(interval=0.0),
        cpu_user_sec=cpu_times.user,
        cpu_system_sec=cpu_times.system,
        cpu_count_logical=psutil.cpu_count(logical=True) or 0,
        cpu_count_physical=psutil.cpu_count(logical=False) or 0,
    )

    logger.debug(
        "CPU: process=%.1f%%, system=%.1f%%, user=%.3fs, system=%.3fs",
        cpu_percent,
        result.cpu_percent_system,
        cpu_times.user,
        cpu_times.system,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# compute_acceptance_rate
# ──────────────────────────────────────────────────────────────────────────────


def compute_acceptance_rate(
    accepted: Sequence[bool],
) -> AcceptanceResult:
    """Compute speculative decoding acceptance rate from token-level booleans.

    Analyzes a sequence of True/False values indicating whether each
    draft token was accepted or rejected. Computes overall acceptance
    rate, rejection rate, and run-length statistics (consecutive
    accepted tokens between rejections).

    Args:
        accepted: Sequence of booleans where True means the draft token
            was accepted and False means it was rejected.

    Returns:
        AcceptanceResult with acceptance/rejection rates and run-length
        statistics.

    Raises:
        ValueError: If the sequence is empty.

    Examples:
        >>> r = compute_acceptance_rate([True, True, False, True, False, False, True])
        >>> r.acceptance_rate
        0.5714285714285714
        >>> r.mean_run_length
        1.5
    """
    if not accepted:
        raise ValueError("accepted sequence must not be empty")

    total = len(accepted)
    n_accepted = sum(1 for a in accepted if a)
    n_rejected = total - n_accepted
    acceptance_rate = n_accepted / total
    rejection_rate = 1.0 - acceptance_rate

    # Compute run lengths of consecutive True values
    run_lengths: list[int] = []
    current_run = 0
    for a in accepted:
        if a:
            current_run += 1
        else:
            if current_run > 0:
                run_lengths.append(current_run)
            current_run = 0
    if current_run > 0:
        run_lengths.append(current_run)

    n_runs = len(run_lengths)
    mean_run = statistics.mean(run_lengths) if run_lengths else 0.0
    max_run = max(run_lengths) if run_lengths else 0

    result = AcceptanceResult(
        total_draft_tokens=total,
        accepted_tokens=n_accepted,
        rejected_tokens=n_rejected,
        acceptance_rate=acceptance_rate,
        rejection_rate=rejection_rate,
        mean_run_length=mean_run,
        max_run_length=max_run,
        n_runs=n_runs,
    )

    logger.debug(
        "Acceptance: %d/%d (%.1f%%), mean_run=%.2f, max_run=%d",
        n_accepted,
        total,
        acceptance_rate * 100,
        mean_run,
        max_run,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# compute_summary_statistics
# ──────────────────────────────────────────────────────────────────────────────


def compute_summary_statistics(values: Sequence[float]) -> SummaryStatistics:
    """Compute descriptive statistics for a numeric sample.

    Calculates mean, median, standard deviation, min, max, and
    the 5th, 25th, 75th, and 95th percentiles.

    Args:
        values: Numeric values to summarize. Must have at least one element.

    Returns:
        SummaryStatistics with all computed statistics.

    Raises:
        ValueError: If values is empty or contains non-finite values.

    Examples:
        >>> s = compute_summary_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> s.mean
        3.0
        >>> s.n
        5
    """
    if not values:
        raise ValueError("values must not be empty")

    for i, v in enumerate(values):
        if not math.isfinite(v):
            raise ValueError(f"Non-finite value at index {i}: {v}")

    n = len(values)
    sorted_vals = sorted(values)

    mean_val = statistics.mean(values)
    median_val = statistics.median(values)
    std_val = statistics.stdev(values) if n >= 2 else 0.0
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    p5 = _percentile(sorted_vals, 5.0)
    p25 = _percentile(sorted_vals, 25.0)
    p75 = _percentile(sorted_vals, 75.0)
    p95 = _percentile(sorted_vals, 95.0)

    result = SummaryStatistics(
        mean=mean_val,
        median=median_val,
        std=std_val,
        min=min_val,
        max=max_val,
        p5=p5,
        p25=p25,
        p75=p75,
        p95=p95,
        n=n,
    )

    logger.debug(
        "Summary: mean=%.4f, std=%.4f, min=%.4f, max=%.4f, n=%d",
        mean_val,
        std_val,
        min_val,
        max_val,
        n,
    )
    return result
