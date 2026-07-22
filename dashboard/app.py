#!/usr/bin/env python3
"""dashboard/app.py — SpecArm Web Dashboard API server.

Serves the interactive frontend at http://localhost:5000 and exposes
a set of JSON API endpoints the frontend calls via fetch().

Endpoints:
  GET  /                          → serves index.html
  GET  /api/status                → server health check
  GET  /api/results               → list all saved benchmark runs
  GET  /api/results/latest        → most recent benchmark run (summary)
  GET  /api/results/<filename>    → specific run by filename
  GET  /api/sysinfo               → hardware / Arm ISA capabilities (from probe.py)
  POST /api/benchmark             → trigger a new benchmark run (async via SSE stream)
  GET  /api/benchmark/stream      → SSE stream of live benchmark log output
  POST /api/prompt                → send a single prompt to llama-server, stream response

Usage:
    python dashboard/app.py
    python dashboard/app.py --host 13.211.208.159 --port 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Generator

from flask import Flask, Response, jsonify, request, send_from_directory

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_RESULTS_DIR  = _PROJECT_ROOT / "results"
_PROBE_SCRIPT = _PROJECT_ROOT / "probe.py"

# ── Config from env ────────────────────────────────────────────────────────────
DEFAULT_LLAMA_URL = os.environ.get("SPECARM_SERVER_URL", "http://13.211.208.159:8080")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(_HERE), static_url_path="")

logging.basicConfig(level=logging.INFO, format="[dashboard] %(levelname)s %(message)s")
logger = logging.getLogger("specarm.dashboard")

# ── Shared state for the live-streaming benchmark job ─────────────────────────
_bench_lock   = threading.Lock()
_bench_active = False          # True while a benchmark is running
_bench_queue: queue.Queue[str] = queue.Queue(maxsize=2000)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _all_benchmark_files() -> list[Path]:
    if not _RESULTS_DIR.is_dir():
        return []
    return sorted(
        _RESULTS_DIR.glob("benchmark_results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _summarise(data: dict[str, Any], filename: str) -> dict[str, Any]:
    """Build a compact summary dict from a full benchmark JSON."""
    meta     = data.get("metadata", {})
    agg      = data.get("aggregate", {})
    results  = data.get("results", [])

    successful = [r for r in results if r.get("status") == "success"]
    tps        = agg.get("tokens_per_second", {})
    dur        = agg.get("duration_sec", {})
    ttft       = agg.get("ttft_ms", {})

    cats: dict[str, list[float]] = {}
    for r in successful:
        cat = r.get("category", "unknown")
        cats.setdefault(cat, []).append(r.get("tokens_per_second", 0.0))
    cat_tps = {c: round(sum(v) / len(v), 2) for c, v in cats.items()}

    mem_vals = [r.get("memory_usage", 0) for r in successful if r.get("memory_usage", 0) > 0]
    avg_mem  = round(sum(mem_vals) / len(mem_vals), 1) if mem_vals else 0.0

    return {
        "filename":             filename,
        "server_url":           meta.get("server_url", ""),
        "model":                meta.get("model_name", ""),
        "timestamp":            meta.get("timestamp", ""),
        "trials":               meta.get("trials", 0),
        "max_tokens":           meta.get("max_tokens", 0),
        "total_prompts":        meta.get("total_prompts", 0),
        "total_results":        meta.get("total_results", 0),
        "successful":           meta.get("successful", 0),
        "failed":               meta.get("failed", 0),
        "success_rate":         round(len(successful) / max(len(results), 1) * 100, 1),
        "avg_tps":              round(tps.get("mean",   0.0), 2),
        "median_tps":           round(tps.get("median", 0.0), 2),
        "p95_tps":              round(tps.get("p95",    0.0), 2),
        "min_tps":              round(tps.get("min",    0.0), 2),
        "max_tps":              round(tps.get("max",    0.0), 2),
        "std_tps":              round(tps.get("std",    0.0), 2),
        "avg_duration_sec":     round(dur.get("mean",   0.0), 3),
        "avg_ttft_ms":          round(ttft.get("mean",  0.0), 2),
        "avg_memory_mb":        avg_mem,
        "category_tps":         cat_tps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Static routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(_HERE), "index.html")


# ─────────────────────────────────────────────────────────────────────────────
# API: status
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """Quick server health + config check."""
    llama_url  = request.args.get("server_url", DEFAULT_LLAMA_URL)
    llama_ok   = False
    llama_msg  = ""

    try:
        import requests as req
        r = req.get(f"{llama_url}/health", timeout=5)
        llama_ok  = r.status_code == 200
        llama_msg = r.json().get("status", "ok") if llama_ok else f"HTTP {r.status_code}"
    except Exception as exc:
        llama_msg = str(exc)

    return jsonify({
        "dashboard":   "ok",
        "llama_server": {"url": llama_url, "ok": llama_ok, "message": llama_msg},
        "results_dir": str(_RESULTS_DIR),
        "result_count": len(_all_benchmark_files()),
        "benchmark_active": _bench_active,
    })


# ─────────────────────────────────────────────────────────────────────────────
# API: results
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/results")
def api_results():
    """List all saved benchmark runs, newest first."""
    files   = _all_benchmark_files()
    summaries = []
    for f in files:
        data = _load_json(f)
        if data:
            summaries.append(_summarise(data, f.name))
    return jsonify({"ok": True, "count": len(summaries), "runs": summaries})


@app.route("/api/results/latest")
def api_results_latest():
    """Return the most recent benchmark run summary."""
    files = _all_benchmark_files()
    if not files:
        return jsonify({"ok": False, "error": "No benchmark results found.  Run a benchmark first."})
    data = _load_json(files[0])
    if not data:
        return jsonify({"ok": False, "error": f"Could not read {files[0].name}"})
    return jsonify({"ok": True, **_summarise(data, files[0].name)})


@app.route("/api/results/<filename>")
def api_results_file(filename: str):
    """Return a specific result file by name (summary or full with ?full=1)."""
    # Safety: only allow files inside results dir
    path = _RESULTS_DIR / filename
    if not path.exists() or path.parent != _RESULTS_DIR:
        return jsonify({"ok": False, "error": "File not found"}), 404
    data = _load_json(path)
    if not data:
        return jsonify({"ok": False, "error": "Could not parse file"}), 500

    if request.args.get("full"):
        return jsonify({"ok": True, **data})
    return jsonify({"ok": True, **_summarise(data, filename)})


# ─────────────────────────────────────────────────────────────────────────────
# API: sysinfo
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/sysinfo")
def api_sysinfo():
    """Run probe.py and return structured hardware/ISA info."""
    if not _PROBE_SCRIPT.exists():
        return jsonify({"ok": False, "error": "probe.py not found"})
    try:
        proc = subprocess.run(
            [sys.executable, str(_PROBE_SCRIPT), "--json"],
            capture_output=True, text=True, timeout=20,
            cwd=str(_PROJECT_ROOT),
        )
        if proc.returncode != 0:
            return jsonify({"ok": False, "error": proc.stderr[:500]})
        data = json.loads(proc.stdout)
        isa   = data.get("arm_isa", {})
        build = data.get("build",   {})
        hw    = data.get("hardware", {})
        os_d  = data.get("os", {})
        data["ok"] = True
        data["summary"] = {
            "cpu":          hw.get("cpu_model", "N/A"),
            "cores":        hw.get("cores", "N/A"),
            "memory":       hw.get("memory", {}).get("total", "N/A"),
            "os":           os_d.get("pretty", os_d.get("name", "N/A")),
            "arch":         os_d.get("machine", "N/A"),
            "kleidiai":     build.get("kleidai_cpu", "N/A"),
            "llama_build":  build.get("source_exists", "N/A"),
            "neon":         isa.get("NEON",    False),
            "dotprod":      isa.get("DOTPROD", False),
            "sve":          isa.get("SVE",     False),
            "i8mm":         isa.get("I8MM",    False),
        }
        return jsonify(data)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# API: benchmark (trigger + SSE stream)
# ─────────────────────────────────────────────────────────────────────────────

def _run_benchmark_thread(server_url: str, model: str, trials: int, max_tokens: int) -> None:
    """Worker thread: runs benchmark.py and pushes log lines into _bench_queue."""
    global _bench_active
    cmd = [
        sys.executable, "-m", "benchmark.benchmark",
        "--server-url",  server_url,
        "--model-name",  model,
        "--trials",      str(trials),
        "--max-tokens",  str(max_tokens),
        "--timeout",     "120",
        "--output",      str(_RESULTS_DIR),
        "--log-level",   "INFO",
    ]
    _bench_queue.put(json.dumps({"type": "start", "cmd": " ".join(cmd)}))
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(_PROJECT_ROOT),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _bench_queue.put(json.dumps({"type": "log", "line": line.rstrip()}))
        proc.wait()
        rc = proc.returncode

        # Find the result file just written
        files = _all_benchmark_files()
        summary = None
        if files:
            data = _load_json(files[0])
            if data:
                summary = _summarise(data, files[0].name)

        if rc == 0 and summary:
            _bench_queue.put(json.dumps({"type": "done", "ok": True,  "summary": summary}))
        else:
            _bench_queue.put(json.dumps({"type": "done", "ok": False, "exit_code": rc}))
    except Exception as exc:
        _bench_queue.put(json.dumps({"type": "done", "ok": False, "error": str(exc)}))
    finally:
        with _bench_lock:
            # noinspection PyGlobalUndefined
            globals()["_bench_active"] = False


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark_start():
    """Kick off a benchmark run (non-blocking — stream output via /api/benchmark/stream)."""
    global _bench_active

    with _bench_lock:
        if _bench_active:
            return jsonify({"ok": False, "error": "A benchmark is already running."}), 409
        _bench_active = True

    body         = request.get_json(silent=True) or {}
    server_url   = body.get("server_url",  DEFAULT_LLAMA_URL)
    model        = body.get("model",       "Q4_K_M")
    trials       = int(body.get("trials",       3))
    max_tokens   = int(body.get("max_tokens",  64))

    # Drain any stale messages
    while not _bench_queue.empty():
        try:
            _bench_queue.get_nowait()
        except queue.Empty:
            break

    t = threading.Thread(
        target=_run_benchmark_thread,
        args=(server_url, model, trials, max_tokens),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True, "message": "Benchmark started.  Connect to /api/benchmark/stream for live output."})


@app.route("/api/benchmark/stream")
def api_benchmark_stream():
    """SSE endpoint — streams benchmark log lines to the browser."""
    def generate() -> Generator[str, None, None]:
        yield "data: {\"type\":\"connected\"}\n\n"
        while True:
            try:
                msg = _bench_queue.get(timeout=30)
                yield f"data: {msg}\n\n"
                parsed = json.loads(msg)
                if parsed.get("type") == "done":
                    break
            except queue.Empty:
                # Heartbeat so the connection doesn't time out
                yield "data: {\"type\":\"ping\"}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# API: single prompt (live inference)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/prompt", methods=["POST"])
def api_prompt():
    """Send a single prompt to llama-server and return the response."""
    body       = request.get_json(silent=True) or {}
    server_url = body.get("server_url",  DEFAULT_LLAMA_URL)
    prompt     = body.get("prompt",      "").strip()
    max_tokens = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 0.7))

    if not prompt:
        return jsonify({"ok": False, "error": "prompt is required"}), 400

    try:
        import requests as req
        import time as _time
        t0   = _time.perf_counter()
        resp = req.post(
            f"{server_url.rstrip('/')}/completion",
            json={"prompt": prompt, "n_predict": max_tokens, "temperature": temperature, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        dur  = _time.perf_counter() - t0
        data = resp.json()
        n    = data.get("tokens_predicted", 0)
        return jsonify({
            "ok":              True,
            "content":         data.get("content", ""),
            "tokens_generated": n,
            "tokens_prompt":   data.get("tokens_evaluated", 0),
            "duration_sec":    round(dur, 3),
            "tokens_per_second": round(n / dur, 2) if dur > 0 else 0,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SpecArm Web Dashboard")
    p.add_argument("--host",       default="0.0.0.0",            help="Dashboard listen host  (default: 0.0.0.0)")
    p.add_argument("--port",       type=int, default=5000,        help="Dashboard listen port  (default: 5000)")
    p.add_argument("--llama-url",  default=DEFAULT_LLAMA_URL,     help="llama-server URL override")
    p.add_argument("--debug",      action="store_true",           help="Flask debug mode")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    os.environ["SPECARM_SERVER_URL"] = args.llama_url
    DEFAULT_LLAMA_URL = args.llama_url
    logger.info("SpecArm Dashboard starting on http://%s:%d", args.host, args.port)
    logger.info("llama-server URL: %s", args.llama_url)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
