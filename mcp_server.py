#!/usr/bin/env python3
"""mcp_server.py — SpecArm MCP (Model Context Protocol) server.

Exposes three tools that any MCP-compatible AI client (Claude Desktop,
Cursor, Kiro, etc.) can call to drive the SpecArm inference pipeline:

  run_benchmark   — trigger P3's benchmark harness, return live results
  get_last_result — return the most recent saved benchmark result (no re-run)
  get_sysinfo     — return CPU / Arm ISA / KleidiAI capability information

Transport: stdio (the standard MCP transport — host process communicates
           via stdin/stdout JSON-RPC 2.0 messages).

Usage (register in your MCP client config):
    {
      "mcpServers": {
        "specarm": {
          "command": "python3",
          "args": ["/path/to/mcp_server.py"],
          "env": {
            "SPECARM_SERVER_URL": "http://13.211.208.159:8080"
          }
        }
      }
    }

Environment variables (all optional — sane defaults shown):
    SPECARM_SERVER_URL   http://127.0.0.1:8080   llama-server base URL
    SPECARM_MODEL_NAME   Q4_K_M                  model label in reports
    SPECARM_TRIALS       3                        benchmark trial count
    SPECARM_MAX_TOKENS   64                       max tokens per prompt
    SPECARM_TIMEOUT      120                      HTTP timeout (seconds)
    SPECARM_RESULTS_DIR  results                  where result files live
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "ERROR: 'mcp' package not installed.\n"
        "  Run: pip install mcp\n"
        "  Or:  pip install 'mcp[cli]'",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Logging (goes to stderr so it doesn't pollute the MCP stdio stream) ───────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[specarm-mcp] %(levelname)s %(message)s",
)
logger = logging.getLogger("specarm.mcp")

# ── Project root — lets us invoke sibling scripts regardless of cwd ───────────
_PROJECT_ROOT = Path(__file__).resolve().parent

# ── Runtime configuration from environment ───────────────────────────────────
_SERVER_URL   = os.environ.get("SPECARM_SERVER_URL",  "http://127.0.0.1:8080")
_MODEL_NAME   = os.environ.get("SPECARM_MODEL_NAME",  "Q4_K_M")
_TRIALS       = int(os.environ.get("SPECARM_TRIALS",       "3"))
_MAX_TOKENS   = int(os.environ.get("SPECARM_MAX_TOKENS",   "64"))
_TIMEOUT      = float(os.environ.get("SPECARM_TIMEOUT",    "120"))
_RESULTS_DIR  = Path(os.environ.get("SPECARM_RESULTS_DIR", str(_PROJECT_ROOT / "results")))

# ── MCP server instance ────────────────────────────────────────────────────────
mcp = FastMCP(
    "SpecArm",
    instructions=(
        "SpecArm runs speculative decoding on Arm Graviton hardware using "
        "llama.cpp + KleidiAI. Use run_benchmark to measure live performance, "
        "get_last_result to inspect the most recent run, and get_sysinfo to "
        "verify Arm-specific CPU capabilities are active."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_subprocess(cmd: list[str], *, timeout: float = 300.0) -> tuple[int, str, str]:
    """Run a child process, return (returncode, stdout, stderr)."""
    logger.info("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_PROJECT_ROOT),
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Process timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return -1, "", str(exc)


def _latest_result_file() -> Path | None:
    """Return the most recently modified benchmark JSON in the results dir."""
    if not _RESULTS_DIR.is_dir():
        return None
    candidates = sorted(
        _RESULTS_DIR.glob("benchmark_results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _summarise_benchmark_json(data: dict[str, Any]) -> dict[str, Any]:
    """Collapse a full benchmark JSON into a judge-friendly summary."""
    meta = data.get("metadata", {})
    agg  = data.get("aggregate", {})
    results = data.get("results", [])

    successful = [r for r in results if r.get("status") == "success"]
    failed     = [r for r in results if r.get("status") != "success"]

    tps_stats   = agg.get("tokens_per_second", {})
    dur_stats   = agg.get("duration_sec", {})
    ttft_stats  = agg.get("ttft_ms", {})

    # Per-category breakdown
    categories: dict[str, list[float]] = {}
    for r in successful:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r.get("tokens_per_second", 0.0))
    category_avg_tps = {
        cat: round(sum(vals) / len(vals), 2)
        for cat, vals in categories.items()
    }

    return {
        "server_url":          meta.get("server_url"),
        "model":               meta.get("model_name"),
        "timestamp":           meta.get("timestamp"),
        "trials":              meta.get("trials"),
        "max_tokens":          meta.get("max_tokens"),
        "total_prompts":       meta.get("total_prompts"),
        "total_results":       meta.get("total_results"),
        "successful":          meta.get("successful"),
        "failed":              meta.get("failed"),
        "avg_tokens_per_sec":  round(tps_stats.get("mean",   0.0), 3),
        "median_tokens_per_sec": round(tps_stats.get("median", 0.0), 3),
        "p95_tokens_per_sec":  round(tps_stats.get("p95",    0.0), 3),
        "avg_duration_sec":    round(dur_stats.get("mean",   0.0), 3),
        "avg_ttft_ms":         round(ttft_stats.get("mean",  0.0), 3),
        "category_avg_tps":    category_avg_tps,
        "failed_prompts":      [r.get("prompt_id") for r in failed],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tool 1 — run_benchmark
# ──────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def run_benchmark(
    server_url: str = "",
    model_name: str = "",
    trials: int = 0,
    max_tokens: int = 0,
) -> dict[str, Any]:
    """Run the SpecArm benchmark against the live llama-server.

    Invokes P3's benchmark harness (benchmark/benchmark.py), waits for it
    to complete, then returns a structured summary of the results.

    Parameters
    ----------
    server_url  : Base URL of the llama-server, e.g. "http://127.0.0.1:8080".
                  Defaults to the SPECARM_SERVER_URL env var.
    model_name  : Label used in result files.  Defaults to SPECARM_MODEL_NAME.
    trials      : Number of trials per prompt (1–10).  Defaults to SPECARM_TRIALS.
    max_tokens  : Max tokens generated per request.  Defaults to SPECARM_MAX_TOKENS.

    Returns
    -------
    A dict with benchmark summary stats, per-category throughput, and the
    path of the saved result file.
    """
    url    = server_url  or _SERVER_URL
    model  = model_name  or _MODEL_NAME
    n_tri  = trials      or _TRIALS
    n_tok  = max_tokens  or _MAX_TOKENS

    # Clamp to sane ranges
    n_tri = max(1, min(n_tri, 10))
    n_tok = max(1, min(n_tok, 2048))

    logger.info("run_benchmark: url=%s model=%s trials=%d max_tokens=%d", url, model, n_tri, n_tok)

    cmd = [
        sys.executable, "-m", "benchmark.benchmark",
        "--server-url",  url,
        "--model-name",  model,
        "--trials",      str(n_tri),
        "--max-tokens",  str(n_tok),
        "--timeout",     str(int(_TIMEOUT)),
        "--output",      str(_RESULTS_DIR),
        "--log-level",   "WARNING",   # keep stderr clean while MCP is running
    ]

    rc, stdout, stderr = _run_subprocess(cmd, timeout=_TIMEOUT * n_tri * 35 + 60)

    if rc != 0:
        return {
            "ok":     False,
            "error":  f"Benchmark process exited with code {rc}",
            "stderr": stderr[-2000:] if stderr else "",
            "hint":   (
                "Check that llama-server is running at "
                f"{url} before calling run_benchmark."
            ),
        }

    # Find the result file that was just written
    result_file = _latest_result_file()
    if result_file is None:
        return {
            "ok":     False,
            "error":  "Benchmark completed but no result file found",
            "stdout": stdout[-1000:],
        }

    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Could not parse result file: {exc}"}

    summary = _summarise_benchmark_json(data)
    summary["ok"]          = True
    summary["result_file"] = str(result_file)
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Tool 2 — get_last_result
# ──────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_last_result(full: bool = False) -> dict[str, Any]:
    """Return the most recent benchmark result without running a new benchmark.

    Finds the newest benchmark_results_*.json file in the results directory
    and returns a summary (or the full JSON when full=True).

    Parameters
    ----------
    full : If True, return the complete raw JSON (can be large).
           If False (default), return a concise summary with key metrics.

    Returns
    -------
    A dict with benchmark metadata and aggregate performance metrics.
    """
    result_file = _latest_result_file()

    if result_file is None:
        return {
            "ok":    False,
            "error": f"No benchmark result files found in {_RESULTS_DIR}",
            "hint":  "Run run_benchmark() first to generate results.",
        }

    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Could not read {result_file.name}: {exc}"}

    if full:
        data["ok"]          = True
        data["result_file"] = str(result_file)
        return data

    summary = _summarise_benchmark_json(data)
    summary["ok"]          = True
    summary["result_file"] = str(result_file)
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Tool 3 — get_sysinfo
# ──────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_sysinfo() -> dict[str, Any]:
    """Return hardware and Arm capability info from the inference server.

    Runs probe.py (P1's hardware detector) and returns structured JSON
    covering CPU model, Arm ISA extensions (NEON, DOTPROD, SVE, I8MM),
    memory, and KleidiAI build status.

    This tool is the definitive answer to "are Arm-specific optimisations
    actually active?" — important for judges verifying the project claims.

    Returns
    -------
    A dict with hardware, arm_isa, os, and build sections, plus a
    readiness_summary block listing which Arm features are enabled.
    """
    probe_script = _PROJECT_ROOT / "probe.py"

    if not probe_script.exists():
        return {"ok": False, "error": f"probe.py not found at {probe_script}"}

    cmd = [sys.executable, str(probe_script), "--json"]
    rc, stdout, stderr = _run_subprocess(cmd, timeout=30.0)

    if rc != 0 or not stdout.strip():
        return {
            "ok":     False,
            "error":  f"probe.py exited with code {rc}",
            "stderr": stderr[-1000:] if stderr else "",
        }

    try:
        data: dict[str, Any] = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Could not parse probe output: {exc}"}

    # Build a human-readable readiness summary
    isa   = data.get("arm_isa",  {})
    build = data.get("build",    {})
    hw    = data.get("hardware", {})
    os_d  = data.get("os",       {})

    key_features = {
        "NEON (SIMD baseline)":    isa.get("NEON",    False),
        "DOTPROD (int8 dot)":      isa.get("DOTPROD", False),
        "SVE (scalable vector)":   isa.get("SVE",     False),
        "SVE2":                    isa.get("SVE2",    False),
        "I8MM (int8 matrix mul)":  isa.get("I8MM",    False),
        "BF16":                    isa.get("BF16",    False),
    }

    kleidai_active = (
        build.get("kleidai_cpu", "").upper() == "ON"
        and build.get("kleidai_top", "").upper() == "ON"
    )

    data["ok"] = True
    data["readiness_summary"] = {
        "cpu_model":        hw.get("cpu_model"),
        "cores":            hw.get("cores"),
        "memory_total":     hw.get("memory", {}).get("total"),
        "os":               os_d.get("pretty", os_d.get("name")),
        "architecture":     os_d.get("machine"),
        "arm_features":     key_features,
        "kleidiai_active":  kleidai_active,
        "llama_version":    build.get("llama_version"),
        "build_type":       build.get("build_type"),
        "specarm_ready":    kleidai_active and isa.get("NEON", False),
    }

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(
        "SpecArm MCP server starting (server_url=%s, results_dir=%s)",
        _SERVER_URL,
        _RESULTS_DIR,
    )
    mcp.run(transport="stdio")
