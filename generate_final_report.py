#!/usr/bin/env python3
"""Render a judge-facing report from existing auto-tuning artifacts only.

This script never starts a server, runs a benchmark, or changes an autotuning
decision.  It is deliberately a formatter over the JSON artifacts produced by
``auto_tune.py`` and a previously saved ``probe.py --json`` output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ReportInputError(RuntimeError):
    """Raised when an existing artifact needed for an honest report is absent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"Cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportInputError(f"Expected a JSON object in {path}.")
    return value


def _find_artifact(directory: Path, filename: str) -> Path:
    direct = directory / filename
    candidates = [direct] if direct.is_file() else sorted(directory.rglob(filename))
    if not candidates:
        raise ReportInputError(
            f"Required {filename} was not found under {directory}. "
            "Run the existing auto_tune.py workflow first; this reporting script does not generate measurements."
        )
    if len(candidates) > 1:
        raise ReportInputError(
            f"Multiple {filename} files were found under {directory}: "
            f"{', '.join(map(str, candidates))}. Supply an auto-tune directory containing one run."
        )
    return candidates[0]


def _find_probe_output(project_root: Path, auto_tune_dir: Path) -> Path:
    names = ("probe_output.json", "probe.json", "hardware_probe.json")
    candidates = [
        path for base in (auto_tune_dir, project_root, project_root / "results", project_root / "reports")
        for name in names for path in [base / name] if path.is_file()
    ]
    candidates = sorted(set(candidates))
    if not candidates:
        raise ReportInputError(
            "No saved probe.py JSON output was found. Save the already-collected probe output "
            "as probe_output.json (for example, results/auto_tune/probe_output.json) and rerun this formatter."
        )
    if len(candidates) > 1:
        raise ReportInputError(f"Multiple probe outputs found: {', '.join(map(str, candidates))}.")
    return candidates[0]


def _value(value: Any, *, digits: int | None = None, suffix: str = "") -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float) and digits is not None:
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _fraction(value: Any) -> str:
    """Format a stored fractional threshold or improvement as a percentage."""
    return "N/A" if value is None or value == "" else f"{float(value) * 100:.1f}%"


def _nested(record: dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _hardware(probe: dict[str, Any]) -> dict[str, str]:
    hardware = probe.get("hardware", {}) if isinstance(probe.get("hardware"), dict) else {}
    os_info = probe.get("os", {}) if isinstance(probe.get("os"), dict) else {}
    build = probe.get("build", {}) if isinstance(probe.get("build"), dict) else {}
    # ``hardware.architecture`` comes from /proc/cpuinfo and can be a numeric
    # ARM ISA revision (for example, "8"), not the machine architecture.  The
    # latter is reported by probe.py as ``os.machine`` (for example, aarch64).
    machine = os_info.get("machine")
    cpu_architecture = hardware.get("architecture")
    architecture = machine or (cpu_architecture if not str(cpu_architecture).isdigit() else None)
    kleidai = build.get("kleidai_cpu")
    return {
        "architecture": _value(architecture),
        "cores": _value(hardware.get("cores") or hardware.get("cpu_count")),
        "model": _value(hardware.get("cpu_model")),
        "kleidai": "Enabled" if kleidai == "ON" else "Disabled" if kleidai == "OFF" else "Unavailable",
    }


def _baseline_max_tokens(baseline: dict[str, Any], metadata: dict[str, Any]) -> Any:
    """Read a recorded baseline token limit without guessing a default."""
    return (
        baseline.get("max_tokens")
        or metadata.get("baseline_max_tokens")
        or _nested(metadata, "baseline", "max_tokens")
    )


def _comparison_caveat(baseline: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    """Flag baseline deltas when the JSON records non-comparable token limits."""
    baseline_tokens = _baseline_max_tokens(baseline, metadata)
    sweep_tokens = metadata.get("max_tokens")
    if baseline_tokens is None or sweep_tokens is None or str(baseline_tokens) == str(sweep_tokens):
        return None
    return (
        f"**Baseline-comparison caveat:** the baseline used max_tokens={baseline_tokens}, "
        f"while this sweep used max_tokens={sweep_tokens}. TPS, TTFT, and latency deltas are not a fair "
        "apples-to-apples comparison and must not be read as configuration improvement or regression until "
        "both runs are repeated with matching settings."
    )


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def render_report(tuning: dict[str, Any], speculative: dict[str, Any], probe: dict[str, Any]) -> str:
    """Return deterministic Markdown using only fields already stored in JSON."""
    baseline = tuning.get("baseline") if isinstance(tuning.get("baseline"), dict) else {}
    runs = tuning.get("runs") if isinstance(tuning.get("runs"), list) else []
    winner = tuning.get("best_configuration") if isinstance(tuning.get("best_configuration"), dict) else {}
    metadata = tuning.get("metadata") if isinstance(tuning.get("metadata"), dict) else {}
    decision = speculative.get("decision") if isinstance(speculative.get("decision"), dict) else tuning.get("speculative_decision")
    if not isinstance(decision, dict):
        raise ReportInputError("Speculative results contain no existing adaptive decision.")
    spec_metadata = speculative.get("metadata") if isinstance(speculative.get("metadata"), dict) else {}
    spec_results = speculative.get("results") if isinstance(speculative.get("results"), list) else []
    hw = _hardware(probe)
    winner_config = winner.get("server_config") if isinstance(winner.get("server_config"), dict) else {}
    winner_result = winner.get("result") if isinstance(winner.get("result"), dict) else {}
    comparison = winner.get("comparison_to_baseline") if isinstance(winner.get("comparison_to_baseline"), dict) else {}
    enabled = bool(decision.get("speculation_enabled"))
    best_spec = decision.get("best_speculative") if isinstance(decision.get("best_speculative"), dict) else None
    best_spec_result = best_spec.get("result") if isinstance(best_spec, dict) and isinstance(best_spec.get("result"), dict) else {}
    acceptance = spec_metadata.get("acceptance_metric") or decision.get("acceptance_metric") or "Not available"
    comparison_caveat = _comparison_caveat(baseline, metadata)

    winner_label = (
        f"threads={_value(winner_config.get('threads'))}, batch={_value(winner_config.get('batch_size'))}, "
        f"ubatch={_value(winner_config.get('ubatch_size'))}"
    ) if winner else "N/A"
    tps_change = comparison.get("tps_improvement_percent")
    decision_word = "ENABLE" if enabled else "DISABLE"
    executive = (
        "This report summarizes the existing llama.cpp configuration sweep, which compared thread, batch, "
        f"micro-batch, and context settings and selected {winner_label} as the recorded non-speculative winner. "
        f"Its measured average throughput was {_value(winner_result.get('avg_tps'), digits=2)} TPS, "
        f"with a recorded baseline change of {_value(tps_change, digits=1, suffix='%')}. "
        "Speculative decoding, where a smaller draft model proposes tokens for the main model to check, "
        "was also evaluated. A draft model is that smaller proposing model; an acceptance rate is the fraction "
        "of its proposed tokens accepted by the main model. "
        f"The existing controller decision is {decision_word}: {decision.get('decision_reason', 'N/A')}"
    )
    lines = ["# Final LLM Optimization Report", "", "## 1. Executive Summary", "", executive]
    if comparison_caveat:
        lines.extend(["", comparison_caveat])
    lines.extend(["", "## 2. Hardware Summary", ""])
    lines.extend(_table(["Property", "Value"], [["CPU architecture", hw["architecture"]], ["CPU count", hw["cores"]], ["CPU model", hw["model"]], ["KleidiAI", hw["kleidai"]]]))
    if hw["kleidai"] == "Enabled":
        lines.extend(["", "KleidiAI's ARM-optimized kernels were enabled in the recorded llama.cpp build."])
    lines.extend(["", "## 3. Baseline", ""])
    lines.extend(_table(["Model", "Average TPS", "Average TTFT (ms)", "Average latency (ms)", "Success rate", "Threads", "Batch size"], [[
        _value(baseline.get("model")), _value(baseline.get("avg_tps"), digits=2), _value(baseline.get("avg_ttft_ms"), digits=1),
        _value(baseline.get("avg_latency_ms"), digits=1), _value(baseline.get("success_rate"), digits=1, suffix="%"),
        _value(baseline.get("threads")), _value(baseline.get("batch_size")),
    ]]))
    lines.extend(["", "## 4. Non-Speculative Configuration Sweep", ""])
    if comparison_caveat:
        lines.extend([comparison_caveat, ""])
    sweep_rows: list[list[str]] = []
    for run in runs:
        run = run if isinstance(run, dict) else {}
        config = run.get("server_config") if isinstance(run.get("server_config"), dict) else {}
        result = run.get("result") if isinstance(run.get("result"), dict) else {}
        delta = run.get("comparison_to_baseline") if isinstance(run.get("comparison_to_baseline"), dict) else {}
        sweep_rows.append([_value(result.get("rank")), _value(config.get("threads")), _value(config.get("batch_size")), _value(config.get("ubatch_size")), _value(config.get("context_size")), _value(result.get("avg_tps"), digits=2), _value(result.get("avg_ttft_ms"), digits=1), _value(result.get("avg_latency_ms"), digits=1), _value(result.get("memory_mb"), digits=1), _value(result.get("success_rate"), digits=1, suffix="%"), _value(delta.get("tps_improvement_percent"), digits=1, suffix="%"), _value(run.get("status")), _value(run.get("error"))])
    lines.extend(_table(["Rank", "Threads", "Batch", "UBatch", "Context", "TPS", "TTFT (ms)", "Latency (ms)", "Memory (MB)", "Success", "TPS Δ", "Status", "Error"], sweep_rows or [["N/A"] * 13]))
    lines.extend(["", "## 5. Winning Non-Speculative Configuration", ""])
    lines.extend(_table(["Threads", "Batch", "UBatch", "Context", "Average TPS", "Success", "TPS Δ", "TTFT Δ", "Latency Δ"], [[_value(winner_config.get("threads")), _value(winner_config.get("batch_size")), _value(winner_config.get("ubatch_size")), _value(winner_config.get("context_size")), _value(winner_result.get("avg_tps"), digits=2), _value(winner_result.get("success_rate"), digits=1, suffix="%"), _value(comparison.get("tps_improvement_percent"), digits=1, suffix="%"), _value(comparison.get("ttft_improvement_percent"), digits=1, suffix="%"), _value(comparison.get("latency_improvement_percent"), digits=1, suffix="%")]]))
    lines.extend([""])
    if comparison_caveat:
        lines.extend([comparison_caveat, ""])
    lines.extend([f"This configuration was selected using the recorded rule: {metadata.get('selection_rule', 'Not available')}.", "", "## 6. Speculative Decoding Evaluation", "", "### Configuration tested", "", f"- Draft model: {_value((best_spec or {}).get('draft_model_path') if best_spec else None)}", f"- Draft lengths tested: {', '.join(map(str, decision.get('draft_lengths_tested') or spec_metadata.get('draft_lengths') or [])) or 'N/A'}", f"- Minimum improvement threshold: {_fraction(decision.get('decision_threshold', spec_metadata.get('minimum_improvement')))}", "", "### Results", ""])
    spec_rows = []
    for item in spec_results:
        item = item if isinstance(item, dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        spec_rows.append([_value(item.get("draft_length")), "completed" if item.get("success") else "failed", _value(result.get("avg_tps"), digits=2), _value(result.get("success_rate"), digits=1, suffix="%"), _value(item.get("error")), _value(item.get("acceptance_metric_label") or acceptance)])
    lines.extend(_table(["Draft length", "Status", "Average TPS", "Success rate", "Error", "Acceptance metric status"], spec_rows or [["N/A"] * 6]))
    lines.extend(["", "### Best speculative result", "", f"- Best draft length: {_value((best_spec or {}).get('draft_length') if best_spec else None)}", f"- Best speculative TPS: {_value(best_spec_result.get('avg_tps'), digits=2)}", f"- Improvement/degradation versus winner: {_fraction(decision.get('improvement'))}", "", "### Final decision", "", f"**{decision_word}**", "", _value(decision.get("decision_reason")), "", "Token-level acceptance statistics were not directly extracted from llama-server by the reporting pipeline, so no verified acceptance-rate figure is used as a decision metric." if "not extracted" in str(acceptance).lower() else f"Acceptance metric status: {acceptance}", "", "## 7. Limitations and Caveats", ""])
    limitations = []
    if metadata.get("trials") is not None:
        limitations.append(f"- The recorded normal sweep used {metadata['trials']} trial(s).")
    if metadata.get("max_tokens") is not None:
        limitations.append(f"- The recorded benchmark limit was {metadata['max_tokens']} output tokens.")
    if decision.get("draft_lengths_tested") or spec_metadata.get("draft_lengths"):
        limitations.append("- The speculative evaluation covers only the configured draft lengths shown above.")
    if "not extracted" in str(acceptance).lower():
        limitations.append("- Token-level acceptance statistics were not directly extracted from llama-server.")
    limitations.append("- Results apply to the recorded ARM64 environment and configuration, not all ARM systems.")
    lines.extend(limitations)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a final report from existing auto-tune JSON artifacts.")
    parser.add_argument("--auto-tune-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        tuning_path = _find_artifact(args.auto_tune_dir, "tuning_results.json")
        speculative_path = _find_artifact(args.auto_tune_dir, "speculative_results.json")
        probe_path = _find_probe_output(Path.cwd(), args.auto_tune_dir)
        report = render_report(_read_json(tuning_path), _read_json(speculative_path), _read_json(probe_path))
    except ReportInputError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
