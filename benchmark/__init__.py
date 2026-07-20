# benchmark/__init__.py
"""Benchmark suite for llama.cpp + KleidiAI on ARM64.

Provides automated benchmarking of LLM inference throughput, latency,
memory usage, and speculative decoding performance.
"""

from benchmark.benchmark import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
]
