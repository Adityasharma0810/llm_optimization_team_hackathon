# evaluation/__init__.py
"""Evaluation framework for LLM inference quality and performance.

Provides metric computation, prompt management, and automated evaluation
of quantized and speculative decoding configurations.
"""

from evaluation.metrics import (
    compute_ttft,
    compute_latency,
    compute_tokens_per_second,
    compute_throughput,
    compute_memory_usage,
    compute_cpu_usage,
    compute_acceptance_rate,
    compute_summary_statistics,
    LatencyResult,
    ThroughputResult,
    MemoryResult,
    CpuResult,
    AcceptanceResult,
    SummaryStatistics,
)

__all__ = [
    "compute_ttft",
    "compute_latency",
    "compute_tokens_per_second",
    "compute_throughput",
    "compute_memory_usage",
    "compute_cpu_usage",
    "compute_acceptance_rate",
    "compute_summary_statistics",
    "LatencyResult",
    "ThroughputResult",
    "MemoryResult",
    "CpuResult",
    "AcceptanceResult",
    "SummaryStatistics",
]
