#!/usr/bin/env python3
"""specarm/cli.py — SpecArm command-line interface.

Three sub-commands that give judges and developers a clean terminal UX
on top of the underlying P3/P4 pipeline components.

    specarm benchmark   — run benchmark harness, show live results table
    specarm probe       — show hardware / Arm ISA capability report
    specarm autotune    — run P4's config sweep, show ranked results table

Install:
    pip install -e .          # then `specarm` is available on PATH
    python -m specarm.cli     # or run directly without installing

Quick examples:
    specarm benchmark --host 13.211.208.159
    specarm probe
    specarm autotune --model-path ~/models/qwen2.5-0.5b-instruct-fp16.gguf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from specarm import __version__

# ── Project root (one level up from this file) ────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers (auto-disabled when not a TTY)
# ──────────────────────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


class _C:
    RESET   = "\033[0m"   if _USE_COLOR else ""
    BOLD    = "\033[1m"   if _USE_COLOR else ""
    DIM     = "\033[2m"   if _USE_COLOR else ""
    CYAN    = "\033[36m"  if _USE_COLOR else ""
    GREEN   = "\033[32m"  if _USE_COLOR else ""
    YELLOW  = "\033[33m"  if _USE_COLOR else ""
    RED     = "\033[31m"  if _USE_COLOR else ""
    MAGENTA = "\033[35m"  if _USE_COLOR else ""


def _banner(text: str) -> None:
    width = 62
    print(f"\n{_C.BOLD}{_C.CYAN}{'─' * width}{_C.RESET}")
    print(f"{_C.BOLD}{_C.CYAN}  {text}{_C.RESET}")
    print(f"{_C.BOLD}{_C.CYAN}{'─' * width}{_C.RESET}")


def _row(label: str, value: str, *, ok: bool | None = None) -> None:
    if ok is True:
        tag = f" {_C.GREEN}✓{_C.RESET}"
    elif ok is False:
        tag = f" {_C.RED}✗{_C.RESET}"
    else:
        tag = ""
    print(f"  {_C.CYAN}{label:<30}{_C.RESET} {value}{tag}")


def _ok(msg: str) -> None:
    print(f"  {_C.GREEN}✓{_C.RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_C.YELLOW}⚠{_C.RESET}  {msg}")


def _err(msg: str) -> None:
    print(f"  {_C.RED}✗{_C.RESET}  {msg}", file=sys.stderr)


def _run(cmd: list[str], *, timeout: float = 600.0, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, streaming output live unless capture=True."""
    try:
        if capture:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(_ROOT),
            )
        else:
            return subprocess.run(
                cmd,
                text=True,
                timeout=timeout,
                cwd=str(_ROOT),
            )
    except subprocess.TimeoutExpired:
        _err(f"Command timed out after {timeout}s")
        sys.exit(1)
    except FileNotFoundError:
        _err(f"Command not found: {cmd[0]}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-command: benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark harness and display a summary table."""

    host = args.host.strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    server_url = f"{host}:{args.port}"

    _banner(f"SpecArm Benchmark  →  {server_url}")
    print(
        f"  model      {_C.BOLD}{args.model}{_C.RESET}\n"
        f"  trials     {args.trials}\n"
        f"  max-tokens {args.max_tokens}\n"
        f"  output     {args.output}\n"
    )

    # Build the benchmark.py command
    cmd = [
        sys.executable, "-m", "benchmark.benchmark",
        "--server-url",  server_url,
        "--model-name",  args.model,
        "--trials",      str(args.trials),
        "--max-tokens",  str(args.max_tokens),
        "--timeout",     str(args.timeout),
        "--output",      str(args.output),
        "--log-level",   "WARNING" if args.quiet else "INFO",
    ]

    print(f"{_C.DIM}  Running: {' '.join(cmd)}{_C.RESET}\n")

    # Stream output live so the user sees progress
    result = _run(cmd, timeout=float(args.timeout) * args.trials * 35 + 120)

    if result.returncode != 0:
        _err(f"Benchmark failed (exit code {result.returncode})")
        _err("Check that llama-server is running and reachable.")
        return 1

    # Parse and pretty-print the freshest result file
    results_dir = Path(args.output)
    candidates = sorted(
        results_dir.glob("benchmark_results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        _warn("Benchmark completed but no result file was found.")
        return 0

    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    _print_benchmark_summary(data, candidates[0])
    return 0


def _print_benchmark_summary(data: dict[str, Any], path: Path) -> None:
    meta = data.get("metadata", {})
    agg  = data.get("aggregate", {})
    results = data.get("results", [])

    successful = [r for r in results if r.get("status") == "success"]
    failed     = [r for r in results if r.get("status") != "success"]

    tps  = agg.get("tokens_per_second", {})
    dur  = agg.get("duration_sec", {})
    ttft = agg.get("ttft_ms", {})

    _banner("Benchmark Results")
    _row("Result file",  path.name)
    _row("Server",       meta.get("server_url", "N/A"))
    _row("Model",        meta.get("model_name", "N/A"))
    _row("Timestamp",    meta.get("timestamp",  "N/A"))
    _row("Trials",       str(meta.get("trials", "N/A")))
    _row("Max tokens",   str(meta.get("max_tokens", "N/A")))

    print()
    success_rate = (len(successful) / len(results) * 100.0) if results else 0.0
    sr_ok = success_rate == 100.0
    _row("Prompts run",    str(meta.get("total_prompts", "N/A")))
    _row("Success rate",  f"{success_rate:.1f}%", ok=sr_ok)

    print()
    _banner("Throughput  (tokens / second)")
    _row("Mean",   f"{tps.get('mean',   0):.2f}")
    _row("Median", f"{tps.get('median', 0):.2f}")
    _row("p95",    f"{tps.get('p95',    0):.2f}")
    _row("Min",    f"{tps.get('min',    0):.2f}")
    _row("Max",    f"{tps.get('max',    0):.2f}")

    if ttft.get("mean", 0) > 0:
        print()
        _banner("Latency")
        _row("Avg TTFT (ms)",     f"{ttft.get('mean', 0):.2f}")
        _row("Avg duration (s)",  f"{dur.get('mean',  0):.3f}")

    # Per-category TPS
    categories: dict[str, list[float]] = {}
    for r in successful:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r.get("tokens_per_second", 0.0))

    if categories:
        print()
        _banner("Throughput by Category")
        for cat, vals in sorted(categories.items()):
            avg = sum(vals) / len(vals)
            _row(cat, f"{avg:.2f} tok/s")

    if failed:
        print()
        _warn(f"{len(failed)} prompt(s) failed: {[r.get('prompt_id') for r in failed]}")

    print()
    _ok(f"Full results saved to: {path}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Sub-command: probe
# ──────────────────────────────────────────────────────────────────────────────

def _cmd_probe(args: argparse.Namespace) -> int:
    """Run hardware probe and display capability report."""

    probe_script = _ROOT / "probe.py"
    if not probe_script.exists():
        _err(f"probe.py not found at {probe_script}")
        return 1

    if args.json:
        # Pass --json through to probe.py and relay its output
        result = _run([sys.executable, str(probe_script), "--json"], capture=True)
        if result.returncode != 0:
            _err(f"probe.py failed: {result.stderr}")
            return 1
        # Pretty-print the JSON
        try:
            data = json.loads(result.stdout)
            print(json.dumps(data, indent=2))
        except json.JSONDecodeError:
            print(result.stdout)
        return 0

    if args.compact:
        result = _run([sys.executable, str(probe_script), "--compact"], capture=True)
        print(result.stdout.strip())
        return result.returncode

    # Default: stream the full human-readable report live
    _banner("SpecArm  /  System Capability Probe")
    result = _run([sys.executable, str(probe_script)])
    return result.returncode


# ──────────────────────────────────────────────────────────────────────────────
# Sub-command: autotune
# ──────────────────────────────────────────────────────────────────────────────

def _cmd_autotune(args: argparse.Namespace) -> int:
    """Run P4's auto-tuner and show a ranked results table."""

    _banner("SpecArm Auto-Tune")
    print(
        f"  model-path     {_C.BOLD}{args.model_path}{_C.RESET}\n"
        f"  threads        {args.threads}\n"
        f"  batch-sizes    {args.batch_sizes}\n"
        f"  ubatch-sizes   {args.ubatch_sizes}\n"
        f"  context-sizes  {args.context_sizes}\n"
        f"  output         {args.output}\n"
    )

    cmd = [
        sys.executable, str(_ROOT / "auto_tune.py"),
        "--model-path",    args.model_path,
        "--output",        args.output,
    ]

    # Threads / batch / ubatch / context — all accept multiple values
    for t in args.threads.split():
        cmd += ["--threads", t]
    for b in args.batch_sizes.split():
        cmd += ["--batch-sizes", b]
    for u in args.ubatch_sizes.split():
        cmd += ["--ubatch-sizes", u]
    for c in args.context_sizes.split():
        cmd += ["--context-sizes", c]

    # Optional speculative decoding
    if args.draft_model_path:
        cmd += ["--draft-model-path", args.draft_model_path]
    if args.draft_lengths:
        for d in args.draft_lengths.split():
            cmd += ["--draft-lengths", d]

    if args.trials:
        cmd += ["--trials", str(args.trials)]

    if not args.quiet:
        cmd += ["--log-level", "INFO"]
    else:
        cmd += ["--log-level", "WARNING"]

    print(f"{_C.DIM}  Running: {' '.join(cmd)}{_C.RESET}\n")

    # Stream output live
    result = _run(cmd, timeout=3600.0)

    if result.returncode != 0:
        _err(f"auto_tune.py failed (exit code {result.returncode})")
        return 1

    # Parse and display results table
    output_dir = Path(args.output)
    json_file  = output_dir / "tuning_results.json"

    if not json_file.exists():
        # Search one level deeper (auto_tune sometimes creates a sub-dir)
        candidates = sorted(output_dir.rglob("tuning_results.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        json_file = candidates[0] if candidates else None

    if json_file is None or not json_file.exists():
        _warn("auto_tune.py finished but no tuning_results.json was found.")
        return 0

    data = json.loads(json_file.read_text(encoding="utf-8"))
    _print_autotune_summary(data, json_file)
    return 0


def _print_autotune_summary(data: dict[str, Any], path: Path) -> None:
    meta  = data.get("metadata", {})
    runs  = data.get("runs", [])
    best  = data.get("best_configuration")
    spec  = data.get("speculative_decision")

    _banner("Auto-Tune Results")
    _row("Result file",  path.name)
    _row("Timestamp",    meta.get("timestamp", "N/A"))
    _row("Trials",       str(meta.get("trials", "N/A")))

    # ── Ranked table ──────────────────────────────────────────────────────────
    completed = [
        r for r in runs
        if r.get("status") == "completed" and r.get("result")
    ]
    failed = [r for r in runs if r.get("status") != "completed"]

    if not completed:
        _warn("No completed benchmark runs to rank.")
    else:
        print()
        _banner("Ranked Configurations  (by avg tokens/sec)")

        # Header
        col = _C.BOLD + _C.CYAN
        rst = _C.RESET
        print(
            f"\n  {col}{'Rank':<5}{'Threads':<9}{'Batch':<7}{'uBatch':<9}"
            f"{'Context':<10}{'Avg TPS':<11}{'TTFT ms':<11}"
            f"{'Success%':<10}{'vs Baseline':<14}{rst}"
        )
        print(f"  {'─' * 80}")

        # Sort by avg_tps descending
        ranked = sorted(
            completed,
            key=lambda r: r["result"].get("avg_tps", 0),
            reverse=True,
        )

        for i, run in enumerate(ranked, start=1):
            cfg = run.get("server_config", {})
            res = run.get("result", {})
            cmp = run.get("comparison_to_baseline", {})

            tps       = res.get("avg_tps",         0.0)
            ttft      = res.get("avg_ttft_ms",      0.0)
            success   = res.get("success_rate",     0.0)
            threads   = cfg.get("threads",          "?")
            batch     = cfg.get("batch_size",       "?")
            ubatch    = cfg.get("ubatch_size",      "?")
            context   = cfg.get("context_size",     "?")
            tps_diff  = cmp.get("tps_improvement_percent")

            # Colour the best row
            row_color  = _C.GREEN + _C.BOLD if i == 1 else ""
            row_reset  = _C.RESET

            if tps_diff is not None:
                sign  = "+" if tps_diff >= 0 else ""
                vs_bl = f"{sign}{tps_diff:.1f}%"
                vs_color = _C.GREEN if tps_diff >= 0 else _C.RED
            else:
                vs_bl    = "N/A"
                vs_color = ""

            print(
                f"  {row_color}{i:<5}{threads:<9}{batch:<7}{ubatch:<9}"
                f"{context:<10}{tps:<11.2f}{ttft:<11.1f}"
                f"{success:<10.1f}"
                f"{vs_color}{vs_bl:<14}{row_reset}"
            )

        print()

    # ── Best config ──────────────────────────────────────────────────────────
    if best:
        print()
        _banner("Recommended Configuration")
        cfg = best.get("server_config", {})
        res = best.get("result", {})
        _row("Threads",       str(cfg.get("threads",      "N/A")))
        _row("Batch size",    str(cfg.get("batch_size",   "N/A")))
        _row("uBatch size",   str(cfg.get("ubatch_size",  "N/A")))
        _row("Context size",  str(cfg.get("context_size", "N/A")))
        _row("Avg TPS",       f"{res.get('avg_tps', 0):.2f}")
        _row("Avg TTFT",      f"{res.get('avg_ttft_ms', 0):.1f} ms")
        _row("Success rate",  f"{res.get('success_rate', 0):.1f}%")

        # Print the launch command so the user can copy-paste it
        binary = cfg.get("server_binary", "~/llama.cpp/build/bin/llama-server")
        model  = cfg.get("model_path",    "<model>.gguf")
        print(f"\n  {_C.DIM}Suggested launch command:{_C.RESET}")
        launch = (
            f"  {binary} \\\n"
            f"    -m {model} \\\n"
            f"    -t {cfg.get('threads')} \\\n"
            f"    -b {cfg.get('batch_size')} \\\n"
            f"    -ub {cfg.get('ubatch_size')} \\\n"
            f"    -c {cfg.get('context_size')} \\\n"
            f"    --host 0.0.0.0 --port 8080"
        )
        print(f"{_C.DIM}{launch}{_C.RESET}")

    # ── Speculative decoding decision ─────────────────────────────────────────
    if spec:
        print()
        _banner("Speculative Decoding Decision")
        enabled = spec.get("speculation_enabled", False)
        reason  = spec.get("decision_reason", "N/A")
        _row("Enabled",         str(enabled), ok=enabled)
        _row("Reason",          textwrap.shorten(reason, 60))
        if spec.get("selected_draft_length"):
            _row("Draft length",   str(spec["selected_draft_length"]))
        if spec.get("improvement") is not None:
            _row("TPS improvement", f"{spec['improvement']:.1%}")
        tested = spec.get("draft_lengths_tested", [])
        if tested:
            _row("Lengths tested", str(list(tested)))

    if failed:
        print()
        _warn(f"{len(failed)} configuration(s) failed and were skipped.")

    print()
    _ok(f"Full results saved to: {path.parent}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="specarm",
        description="SpecArm — Arm AI Optimization developer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              specarm benchmark --host 13.211.208.159
              specarm probe
              specarm probe --compact
              specarm autotune --model-path ~/models/qwen2.5-0.5b-instruct-fp16.gguf
            """
        ),
    )
    parser.add_argument("-V", "--version", action="version", version=f"specarm {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── benchmark ─────────────────────────────────────────────────────────────
    p_bench = sub.add_parser(
        "benchmark",
        help="Run benchmark harness against the live llama-server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Send the standard prompt suite to llama-server and report throughput metrics.",
    )
    p_bench.add_argument("--host",       default="127.0.0.1",  help="llama-server host/IP  (default: 127.0.0.1)")
    p_bench.add_argument("--port",       type=int, default=8080, help="llama-server port  (default: 8080)")
    p_bench.add_argument("--model",      default="Q4_K_M",     help="Model label for reports  (default: Q4_K_M)")
    p_bench.add_argument("--trials",     type=int, default=3,   help="Trials per prompt  (default: 3)")
    p_bench.add_argument("--max-tokens", type=int, default=64,  help="Max tokens to generate  (default: 64)")
    p_bench.add_argument("--timeout",    type=float, default=120.0, help="HTTP timeout seconds  (default: 120)")
    p_bench.add_argument("--output",     default="results",    help="Output directory  (default: results)")
    p_bench.add_argument("--quiet", "-q", action="store_true", help="Suppress benchmark INFO logs")
    p_bench.set_defaults(func=_cmd_benchmark)

    # ── probe ─────────────────────────────────────────────────────────────────
    p_probe = sub.add_parser(
        "probe",
        help="Show hardware and Arm ISA capability report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Detect CPU, Arm ISA extensions, memory, and KleidiAI build status.",
    )
    p_probe.add_argument("--json",    action="store_true", help="Output raw JSON")
    p_probe.add_argument("--compact", action="store_true", help="Single-line summary")
    p_probe.set_defaults(func=_cmd_probe)

    # ── autotune ──────────────────────────────────────────────────────────────
    p_tune = sub.add_parser(
        "autotune",
        help="Run P4's configuration sweep and show ranked results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Sweep thread/batch/context combinations, benchmark each one,
            and rank by average tokens/second.  Requires a local GGUF model
            and a built llama-server binary.
            """
        ),
    )
    p_tune.add_argument(
        "--model-path", required=True,
        help="Path to target GGUF model file (required)",
    )
    p_tune.add_argument(
        "--threads", default="1 2",
        metavar="'N [N ...]'",
        help="Space-separated thread counts to test  (default: '1 2')",
    )
    p_tune.add_argument(
        "--batch-sizes", default="256 512",
        metavar="'N [N ...]'",
        help="Space-separated batch sizes  (default: '256 512')",
    )
    p_tune.add_argument(
        "--ubatch-sizes", default="256 512",
        metavar="'N [N ...]'",
        help="Space-separated micro-batch sizes  (default: '256 512')",
    )
    p_tune.add_argument(
        "--context-sizes", default="2048",
        metavar="'N [N ...]'",
        help="Space-separated context sizes  (default: '2048')",
    )
    p_tune.add_argument(
        "--draft-model-path", default=None,
        help="Optional draft GGUF model for speculative decoding evaluation",
    )
    p_tune.add_argument(
        "--draft-lengths", default=None,
        metavar="'N [N ...]'",
        help="Draft lengths to test, e.g. '1 2 4 8'  (requires --draft-model-path)",
    )
    p_tune.add_argument(
        "--trials", type=int, default=None,
        help="Trials per config  (auto_tune.py default if omitted)",
    )
    p_tune.add_argument(
        "--output", default="results/auto_tune",
        help="Output directory  (default: results/auto_tune)",
    )
    p_tune.add_argument("--quiet", "-q", action="store_true", help="Suppress INFO logs")
    p_tune.set_defaults(func=_cmd_autotune)

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser  = _build_parser()
    args    = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
