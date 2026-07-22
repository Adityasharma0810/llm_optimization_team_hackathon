#!/usr/bin/env python3
"""server_client.py — Minimal client for the llama.cpp /completion endpoint.

Connects to a running llama-server instance, sends a prompt, and prints the
response along with basic timing and token-count metrics.  This is the
foundation layer for the SpecArm developer-experience tooling (P5).

Usage:
    # Talk to a local server (default):
    python3 server_client.py --prompt "Explain speculative decoding in one paragraph."

    # Talk to the remote AWS server:
    python3 server_client.py --host 13.211.208.159 --prompt "Hello, Arm!"

    # Full control:
    python3 server_client.py \\
        --host 13.211.208.159 \\
        --port 8080 \\
        --prompt "What is KleidiAI?" \\
        --max-tokens 256 \\
        --temperature 0.7 \\
        --timeout 120

    # JSON output (for piping / scripting):
    python3 server_client.py --prompt "Hi" --json

    # Health-check only (no inference):
    python3 server_client.py --health
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is not installed.  Run: pip install requests", file=sys.stderr)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Default values (mirrors llama-server / benchmark.py conventions)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT = 120.0


# ──────────────────────────────────────────────────────────────────────────────
# Core client functions
# ──────────────────────────────────────────────────────────────────────────────


def build_server_url(host: str, port: int) -> str:
    """Return the base URL for the llama-server, normalised."""
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return f"{host}:{port}"


def health_check(server_url: str, timeout: float = 10.0) -> dict[str, Any]:
    """GET /health and return the parsed JSON body.

    Returns a dict with at minimum a 'status' key.  On any failure the dict
    contains 'status': 'error' and an 'error' key with the reason string.
    """
    url = f"{server_url}/health"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"status": "error", "error": f"Connection refused — is llama-server running at {server_url}?"}
    except requests.Timeout:
        return {"status": "error", "error": f"Health check timed out after {timeout}s"}
    except requests.HTTPError as exc:
        return {"status": "error", "error": f"HTTP {exc.response.status_code}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def send_completion(
    server_url: str,
    prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST /completion and return a structured result dict.

    Matches the payload format used by benchmark/benchmark.py so both tools
    talk to the server the same way.

    Returns:
        On success:
            {
              "ok": True,
              "content":           str,   # generated text
              "tokens_generated":  int,   # completion tokens
              "tokens_prompt":     int,   # prompt tokens
              "duration_sec":      float, # wall-clock time
              "tokens_per_second": float, # generation throughput
              "raw":               dict,  # full server response
            }
        On failure:
            {
              "ok": False,
              "error": str,
              "error_type": str,  # "connection" | "timeout" | "http" | "parse" | "unknown"
            }
    """
    url = f"{server_url.rstrip('/')}/completion"
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    t_start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.ConnectionError as exc:
        return {"ok": False, "error": str(exc), "error_type": "connection"}
    except requests.Timeout:
        return {"ok": False, "error": f"Request timed out after {timeout}s", "error_type": "timeout"}
    except requests.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.response.status_code}: {exc}", "error_type": "http"}

    duration_sec = time.perf_counter() - t_start

    try:
        data = resp.json()
    except ValueError as exc:
        return {"ok": False, "error": f"Could not parse server response as JSON: {exc}", "error_type": "parse"}

    content: str = data.get("content", "")
    tokens_generated: int = data.get("tokens_predicted", 0)
    tokens_prompt: int = data.get("tokens_evaluated", 0)
    tps: float = tokens_generated / duration_sec if duration_sec > 0 else 0.0

    return {
        "ok": True,
        "content": content,
        "tokens_generated": tokens_generated,
        "tokens_prompt": tokens_prompt,
        "duration_sec": round(duration_sec, 3),
        "tokens_per_second": round(tps, 2),
        "raw": data,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()


class _C:
    RESET  = "\033[0m"   if _USE_COLOR else ""
    BOLD   = "\033[1m"   if _USE_COLOR else ""
    CYAN   = "\033[36m"  if _USE_COLOR else ""
    GREEN  = "\033[32m"  if _USE_COLOR else ""
    YELLOW = "\033[33m"  if _USE_COLOR else ""
    RED    = "\033[31m"  if _USE_COLOR else ""
    DIM    = "\033[2m"   if _USE_COLOR else ""


def _print_health(result: dict[str, Any], server_url: str) -> None:
    ok = result.get("status") == "ok"
    icon = f"{_C.GREEN}✓{_C.RESET}" if ok else f"{_C.RED}✗{_C.RESET}"
    label = f"{_C.BOLD}{server_url}/health{_C.RESET}"
    status = result.get("status", "unknown")
    print(f"\n{icon} {label}  →  status: {_C.GREEN if ok else _C.RED}{status}{_C.RESET}")
    if not ok:
        print(f"  {_C.RED}{result.get('error', '')}{_C.RESET}")
    print()


def _print_response(result: dict[str, Any], prompt: str, server_url: str) -> None:
    sep = f"{_C.DIM}{'─' * 60}{_C.RESET}"

    print(f"\n{_C.BOLD}{_C.CYAN}SpecArm / server_client.py{_C.RESET}  {_C.DIM}→ {server_url}{_C.RESET}")
    print(sep)

    print(f"{_C.BOLD}Prompt:{_C.RESET}  {prompt}")
    print(sep)

    if not result["ok"]:
        print(f"{_C.RED}ERROR ({result['error_type']}): {result['error']}{_C.RESET}\n")
        return

    content: str = result["content"].strip()
    print(f"{_C.BOLD}Response:{_C.RESET}\n")
    print(f"  {content}\n")
    print(sep)

    tok_gen = result["tokens_generated"]
    tok_pmt = result["tokens_prompt"]
    dur     = result["duration_sec"]
    tps     = result["tokens_per_second"]

    print(
        f"{_C.DIM}  prompt tokens: {tok_pmt}   "
        f"generated: {tok_gen}   "
        f"duration: {dur:.2f}s   "
        f"throughput: {_C.RESET}{_C.GREEN}{_C.BOLD}{tps:.1f} tok/s{_C.RESET}"
    )
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="server_client.py",
        description="Send a prompt to a running llama-server and print the response.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Connection
    conn = parser.add_argument_group("connection")
    conn.add_argument(
        "--host",
        default=DEFAULT_HOST,
        metavar="HOST",
        help=f"llama-server host or IP  (default: {DEFAULT_HOST})",
    )
    conn.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        metavar="PORT",
        help=f"llama-server port  (default: {DEFAULT_PORT})",
    )

    # Inference
    inf = parser.add_argument_group("inference")
    inf.add_argument(
        "--prompt", "-p",
        default=None,
        metavar="TEXT",
        help="Prompt to send.  Required unless --health is given.",
    )
    inf.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help=f"Maximum tokens to generate  (default: {DEFAULT_MAX_TOKENS})",
    )
    inf.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        metavar="T",
        help=f"Sampling temperature 0.0–2.0  (default: {DEFAULT_TEMPERATURE})",
    )
    inf.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SEC",
        help=f"HTTP request timeout in seconds  (default: {DEFAULT_TIMEOUT})",
    )

    # Modes
    mode = parser.add_argument_group("modes")
    mode.add_argument(
        "--health",
        action="store_true",
        help="Run a health check only — no inference",
    )
    mode.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print result as JSON instead of human-readable text",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate args
    if not args.health and args.prompt is None:
        parser.error("--prompt is required (or use --health for a connectivity check)")

    if not (0.0 <= args.temperature <= 2.0):
        parser.error("--temperature must be between 0.0 and 2.0")

    server_url = build_server_url(args.host, args.port)

    # ── Health check ──────────────────────────────────────────────────────────
    if args.health:
        result = health_check(server_url, timeout=args.timeout)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            _print_health(result, server_url)
        return 0 if result.get("status") == "ok" else 1

    # ── Always do a quick health check first so the error is clear ───────────
    health = health_check(server_url, timeout=min(args.timeout, 10.0))
    if health.get("status") != "ok":
        if args.json_output:
            print(json.dumps({"ok": False, "error": health.get("error", "server unhealthy"), "health": health}, indent=2))
        else:
            _print_health(health, server_url)
            print(f"{_C.YELLOW}Tip: start llama-server with{_C.RESET}")
            print(f"  ~/llama.cpp/build/bin/llama-server -m ~/models/<model>.gguf --host 0.0.0.0 --port {args.port}")
            print()
        return 1

    # ── Completion ────────────────────────────────────────────────────────────
    result = send_completion(
        server_url,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    if args.json_output:
        # Drop the raw server response by default — it's large and noisy
        output = {k: v for k, v in result.items() if k != "raw"}
        print(json.dumps(output, indent=2))
    else:
        _print_response(result, args.prompt, server_url)

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
