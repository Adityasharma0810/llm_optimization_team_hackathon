#!/usr/bin/env python3
"""Reproducible Graviton llama.cpp configuration and autotuning utilities."""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import product
import json
import logging
from pathlib import Path
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.error import URLError
from urllib.request import urlopen

if TYPE_CHECKING:
    from benchmark.benchmark import BenchmarkConfig


logger = logging.getLogger(__name__)


class ServerStartupError(RuntimeError):
    """Raised when llama-server exits or fails its readiness check at startup."""


@dataclass(frozen=True)
class ServerConfig:
    """A complete llama-server launch configuration.

    The generated flags follow the project's documented llama.cpp conventions:
    ``-m`` for the GGUF model, ``-t`` for threads, ``-c`` for context size,
    and ``--host``/``--port`` for the HTTP listener.  Batch and speculative
    decoding flags are llama.cpp server options used by the tuning layer.
    """

    threads: int
    batch_size: int
    ubatch_size: int
    context_size: int
    model_path: str
    draft_model_path: Optional[str] = None
    draft_max: Optional[int] = None
    host: str = "127.0.0.1"
    port: int = 8080
    server_binary: Path = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"

    def __post_init__(self) -> None:
        """Reject incomplete or nonsensical configuration values."""
        if self.threads < 1:
            raise ValueError("threads must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.ubatch_size < 1:
            raise ValueError("ubatch_size must be >= 1")
        if self.context_size < 1:
            raise ValueError("context_size must be >= 1")
        if not self.model_path:
            raise ValueError("model_path must not be empty")
        if not self.host:
            raise ValueError("host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.draft_max is not None and self.draft_max < 1:
            raise ValueError("draft_max must be >= 1 when provided")
        if self.draft_max is not None and not self.draft_model_path:
            raise ValueError("draft_model_path is required when draft_max is set")
        if self.draft_model_path is not None and self.draft_max is None:
            raise ValueError("draft_max is required when draft_model_path is set")

    @property
    def speculative_enabled(self) -> bool:
        """Whether this configuration enables llama.cpp speculative decoding."""
        return (
            self.draft_model_path is not None
            and self.draft_max is not None
        )

    @property
    def server_url(self) -> str:
        """Base URL consumed by :class:`benchmark.benchmark.BenchmarkConfig`."""
        return f"http://{self.host}:{self.port}"

    def to_cli_args(self) -> list[str]:
        """Return the exact argv list for launching ``llama-server``.

        The caller may pass this list directly to a future process-launching
        layer.  No process is created here.
        """
        args = [
            str(self.server_binary),
            "-m", self.model_path,
            "-t", str(self.threads),
            "-b", str(self.batch_size),
            "-ub", str(self.ubatch_size),
            "-c", str(self.context_size),
            "--host", self.host,
            "--port", str(self.port),
        ]
        if self.draft_model_path is not None:
            args.extend([
                "--spec-type", "draft-simple",
                "--spec-draft-model", self.draft_model_path,
            ])
        if self.draft_max is not None:
            args.extend(["--spec-draft-n-max", str(self.draft_max)])
        return args

    def to_benchmark_config(
        self,
        *,
        model_name: Optional[str] = None,
        trials: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
        output_directory: Path = Path("results"),
    ) -> "BenchmarkConfig":
        """Create the existing benchmark configuration for this server.

        ``BenchmarkConfig`` has no model-path field because llama-server loads
        its GGUF at startup.  Its ``model_name`` is therefore a report label;
        by default this method uses the target model filename.
        """
        from benchmark.benchmark import BenchmarkConfig

        return BenchmarkConfig(
            server_url=self.server_url,
            model_name=model_name or Path(self.model_path).name,
            trials=trials,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            output_directory=output_directory,
        )


@dataclass(frozen=True)
class TuningResult:
    """A leaderboard-compatible result augmented with tuning metadata.

    The first seven fields mirror ``LeaderboardEntry`` in
    ``benchmark/compare_quantizations.py``.  The remaining fields capture the
    candidate selected by the future autotuning execution layer.
    """

    rank: int
    model: str
    avg_tokens_per_second: float
    avg_ttft_ms: float
    avg_latency_ms: float
    avg_memory_usage_mb: float
    success_rate: float
    threads: int
    batch_size: int
    draft_max: Optional[int]
    speculative_enabled: bool
    avg_tps: float
    p95_duration_s: float
    memory_mb: float
    score: float

    def to_dict(self) -> dict[str, object]:
        """Serialize this result using the same field names as leaderboard rows."""
        return asdict(self)


@dataclass(frozen=True)
class BaselineMetrics:
    """Metrics from a completed, fixed baseline benchmark."""

    model: str
    avg_tps: float
    avg_ttft_ms: float
    avg_latency_ms: float
    success_rate: float
    threads: int
    batch_size: int
    # The completed manual baseline used the same 128-token benchmark workload
    # recorded in the project benchmark artifacts.  Keep it with the baseline
    # metrics so reports can flag non-comparable later sweeps.
    max_tokens: int = 128

    def to_dict(self) -> dict[str, object]:
        """Serialize baseline information for reports and result files."""
        return asdict(self)


@dataclass(frozen=True)
class TuningRun:
    """One real configuration attempt, including failures without inventing metrics."""

    config: ServerConfig
    result: TuningResult | None
    error: str = ""

    @property
    def status(self) -> str:
        """Return completed only when a real benchmark result was produced."""
        return "completed" if self.result is not None else "failed"


@dataclass(frozen=True)
class SpeculativeCliSupport:
    """Result of checking the installed llama-server speculative CLI surface."""

    supported: bool
    missing_options: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class SpeculativeResult:
    """One actual speculative-decoding benchmark attempt.

    No acceptance-rate field is present: the benchmark API does not expose
    token-level draft acceptance statistics, so this controller never labels a
    derived throughput value as an acceptance rate.
    """

    target_config: ServerConfig
    draft_model_path: str
    draft_length: int
    result: TuningResult | None
    error: str = ""

    @property
    def success(self) -> bool:
        """Return true only for a fully successful measured benchmark."""
        return self.result is not None and self.result.success_rate == 100.0

    @property
    def acceptance_metric_label(self) -> str:
        """Document that token-level acceptance statistics were not collected."""
        return "Not available (llama-server acceptance statistics not extracted)"


@dataclass(frozen=True)
class SpeculativeDecision:
    """Evidence-based decision to enable or disable speculative decoding."""

    speculation_enabled: bool
    best_non_speculative_result: TuningRun
    best_speculative_result: SpeculativeResult | None
    selected_draft_length: int | None
    improvement: float | None
    decision_threshold: float
    decision_reason: str
    support: SpeculativeCliSupport
    draft_lengths_tested: tuple[int, ...] = ()


class ServerManager:
    """Launch, verify, and stop one llama-server process.

    This class owns only process lifecycle management.  It intentionally does
    not invoke the benchmark runner or make tuning decisions.
    """

    def __init__(
        self,
        config: ServerConfig,
        *,
        startup_timeout: float = 60.0,
        poll_interval: float = 0.25,
        shutdown_timeout: float = 10.0,
    ) -> None:
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be > 0")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be > 0")

        self.config = config
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval
        self.shutdown_timeout = shutdown_timeout
        self.process: subprocess.Popen[str] | None = None
        self._stdout_lines: deque[str] = deque(maxlen=200)
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._log_threads: list[threading.Thread] = []

    def command(self) -> list[str]:
        """Return the argv used to start the configured server."""
        return self.config.to_cli_args()

    @property
    def readiness_url(self) -> str:
        """Return the local health endpoint used to verify server readiness."""
        host = self.config.host
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        elif ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.config.port}/health"

    def start(self) -> subprocess.Popen[str]:
        """Start llama-server and wait until its health endpoint responds.

        Raises:
            RuntimeError: If a managed process is already running.
            ServerStartupError: If the process exits or does not become ready.
        """
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("llama-server is already running")

        self._cleanup_finished_process()
        self._stdout_lines.clear()
        self._stderr_lines.clear()
        command = self.command()
        logger.info("Starting llama-server: %s", " ".join(command))

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise ServerStartupError(
                f"Could not launch llama-server at {self.config.server_binary}: {exc}"
            ) from exc

        self._start_log_capture()
        try:
            self._wait_until_ready()
        except ServerStartupError:
            self.stop()
            raise

        logger.info("llama-server is ready at %s (pid=%d)", self.readiness_url, self.process.pid)
        return self.process

    def stop(self) -> None:
        """Gracefully stop the managed server, killing it only if necessary."""
        process = self.process
        if process is None:
            return

        if process.poll() is None:
            logger.info("Stopping llama-server (pid=%d)", process.pid)
            process.terminate()
            try:
                process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "llama-server did not stop within %.1fs; killing pid=%d",
                    self.shutdown_timeout,
                    process.pid,
                )
                process.kill()
                process.wait()
        else:
            logger.info("llama-server already exited with code %s", process.returncode)

        self._cleanup_finished_process()
        logger.info("llama-server stopped")

    def restart(self) -> subprocess.Popen[str]:
        """Stop any managed server and start a fresh instance."""
        logger.info("Restarting llama-server")
        self.stop()
        return self.start()

    def _wait_until_ready(self) -> None:
        """Poll the health endpoint until ready, exited, or timed out."""
        assert self.process is not None
        deadline = time.monotonic() + self.startup_timeout

        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise ServerStartupError(
                    "llama-server exited during startup "
                    f"with code {return_code}. Logs:\n{self._log_tail()}"
                )

            try:
                with urlopen(self.readiness_url, timeout=self.poll_interval) as response:
                    if 200 <= response.status < 300:
                        return
            except (URLError, TimeoutError, OSError):
                pass

            time.sleep(self.poll_interval)

        raise ServerStartupError(
            f"llama-server did not become ready at {self.readiness_url} "
            f"within {self.startup_timeout:.1f}s. Logs:\n{self._log_tail()}"
        )

    def _start_log_capture(self) -> None:
        """Drain stdout and stderr so verbose server logs cannot block the process."""
        assert self.process is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._log_threads = [
            self._capture_stream(self.process.stdout, self._stdout_lines, logging.DEBUG),
            self._capture_stream(self.process.stderr, self._stderr_lines, logging.WARNING),
        ]
        for thread in self._log_threads:
            thread.start()

    def _capture_stream(
        self,
        stream: object,
        lines: deque[str],
        level: int,
    ) -> threading.Thread:
        """Create a daemon thread that stores and logs a server output stream."""
        def drain() -> None:
            for line in stream:  # type: ignore[union-attr]
                message = line.rstrip()
                lines.append(message)
                logger.log(level, "llama-server: %s", message)

        return threading.Thread(target=drain, daemon=True)

    def _log_tail(self) -> str:
        """Return recent captured server output for an actionable failure message."""
        lines = [*self._stdout_lines, *self._stderr_lines]
        return "\n".join(lines[-20:]) or "<no server output captured>"

    def _cleanup_finished_process(self) -> None:
        """Close pipes and release the completed process handle."""
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            return

        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        for thread in self._log_threads:
            thread.join(timeout=1.0)
        self._log_threads.clear()
        self.process = None


class BenchmarkRunner:
    """Run the project's existing benchmark against one managed server config.

    This wrapper owns one start/benchmark/stop cycle.  It does not create
    candidates, compare configurations, or assign a meaningful score.
    """

    def __init__(
        self,
        *,
        trials: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
        output_directory: Path = Path("results"),
        startup_timeout: float = 60.0,
    ) -> None:
        self.trials = trials
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.output_directory = output_directory
        self.startup_timeout = startup_timeout

    def run(self, config: ServerConfig) -> TuningResult:
        """Launch one server, run the existing benchmark, and stop the server.

        The server is always stopped after the benchmark attempt, including if
        the benchmark raises an exception.
        """
        manager = ServerManager(config, startup_timeout=self.startup_timeout)
        manager.start()
        try:
            benchmark_config = config.to_benchmark_config(
                trials=self.trials,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                output_directory=self.output_directory,
            )
            benchmark_config.validate()

            # Import lazily so the lifecycle-only smoke test remains usable
            # without the benchmark package's optional runtime dependencies.
            from benchmark.benchmark import BenchmarkRunner as ProjectBenchmarkRunner

            project_runner = ProjectBenchmarkRunner(benchmark_config)
            results = project_runner.run_all()
            return self._to_tuning_result(config, results)
        finally:
            manager.stop()

    @staticmethod
    def _to_tuning_result(
        config: ServerConfig,
        results: list[object],
    ) -> TuningResult:
        """Translate existing benchmark result records into one result object."""
        from evaluation.metrics import compute_summary_statistics

        successful = [result for result in results if result.status == "success"]
        total = len(results)

        if successful:
            tps = compute_summary_statistics(
                [result.tokens_per_second for result in successful]
            )
            ttft = compute_summary_statistics([result.ttft for result in successful])
            latency = compute_summary_statistics([result.latency for result in successful])
            memory = compute_summary_statistics(
                [result.memory_usage for result in successful]
            )
            duration = compute_summary_statistics(
                [result.duration for result in successful]
            )
            avg_tps = tps.mean
            avg_ttft_ms = ttft.mean
            avg_latency_ms = latency.mean
            avg_memory_mb = memory.mean
            p95_duration_s = duration.p95
        else:
            avg_tps = 0.0
            avg_ttft_ms = 0.0
            avg_latency_ms = 0.0
            avg_memory_mb = 0.0
            p95_duration_s = 0.0

        return TuningResult(
            # Rank and score are retained solely for the existing result
            # schema.  This single-run layer performs no ranking or scoring.
            rank=0,
            model=Path(config.model_path).name,
            avg_tokens_per_second=avg_tps,
            avg_ttft_ms=avg_ttft_ms,
            avg_latency_ms=avg_latency_ms,
            avg_memory_usage_mb=avg_memory_mb,
            success_rate=(len(successful) / total * 100.0) if total else 0.0,
            threads=config.threads,
            batch_size=config.batch_size,
            draft_max=config.draft_max,
            speculative_enabled=config.speculative_enabled,
            avg_tps=avg_tps,
            p95_duration_s=p95_duration_s,
            memory_mb=avg_memory_mb,
            score=0.0,
        )


_SPECULATIVE_OPTIONS = (
    "--spec-type",
    "--spec-draft-model",
    "--spec-draft-n-max",
)


def inspect_speculative_cli(server_binary: Path) -> SpeculativeCliSupport:
    """Verify the installed binary supports this project's speculative flags."""
    try:
        completed = subprocess.run(
            [str(server_binary), "--help"],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return SpeculativeCliSupport(supported=False, error=str(exc))

    help_text = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        return SpeculativeCliSupport(
            supported=False,
            error=f"llama-server --help exited with code {completed.returncode}",
        )
    missing = tuple(option for option in _SPECULATIVE_OPTIONS if option not in help_text)
    return SpeculativeCliSupport(supported=not missing, missing_options=missing)


class AdaptiveSpeculativeController:
    """Evaluate a small set of draft lengths on the normal tuning winner only."""

    def __init__(
        self,
        *,
        draft_model_path: str | None,
        draft_lengths: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        minimum_improvement: float = 0.05,
        trials: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
        startup_timeout: float = 60.0,
        output_directory: Path = Path("results/auto_tune/speculative"),
        support_checker: Callable[[Path], SpeculativeCliSupport] = inspect_speculative_cli,
        runner_factory: Callable[..., BenchmarkRunner] = BenchmarkRunner,
    ) -> None:
        if minimum_improvement < 0.0:
            raise ValueError("minimum_improvement must be >= 0")
        if not draft_lengths or any(length < 1 for length in draft_lengths):
            raise ValueError("draft_lengths must contain positive integers")
        self.draft_model_path = draft_model_path
        self.draft_lengths = tuple(dict.fromkeys(draft_lengths))
        self.minimum_improvement = minimum_improvement
        self.trials = trials
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.output_directory = output_directory
        self.support_checker = support_checker
        self.runner_factory = runner_factory

    def run(self, best_non_speculative: TuningRun) -> SpeculativeDecision:
        """Run every configured draft length and make a thresholded decision."""
        if best_non_speculative.result is None:
            raise ValueError("best_non_speculative must contain a measured result")

        try:
            support = self.support_checker(best_non_speculative.config.server_binary)
        except Exception as exc:
            logger.exception("Could not inspect llama-server speculative CLI support")
            support = SpeculativeCliSupport(supported=False, error=str(exc))
        if not self.draft_model_path:
            return self._save_disabled(
                best_non_speculative, support,
                "Speculative decoding was not evaluated: no draft model path was configured.",
            )
        if not Path(self.draft_model_path).is_file():
            return self._save_disabled(
                best_non_speculative, support,
                f"Speculative decoding was not evaluated: draft model is unavailable at {self.draft_model_path}.",
            )
        if not support.supported:
            details = ", ".join(support.missing_options) or support.error or "unknown CLI validation error"
            return self._save_disabled(
                best_non_speculative, support,
                f"Speculative decoding was not evaluated: installed llama-server lacks required support ({details}).",
            )

        attempts: list[SpeculativeResult] = []
        self.output_directory.mkdir(parents=True, exist_ok=True)
        for draft_length in self.draft_lengths:
            config = replace(
                best_non_speculative.config,
                draft_model_path=self.draft_model_path,
                draft_max=draft_length,
            )
            try:
                runner = self.runner_factory(
                    trials=self.trials,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    output_directory=self.output_directory / f"draft_length_{draft_length}",
                    startup_timeout=self.startup_timeout,
                )
                result = runner.run(config)
            except Exception as exc:
                logger.exception("Speculative benchmark failed for draft length %d", draft_length)
                attempts.append(SpeculativeResult(
                    target_config=best_non_speculative.config,
                    draft_model_path=self.draft_model_path,
                    draft_length=draft_length,
                    result=None,
                    error=str(exc),
                ))
            else:
                attempts.append(SpeculativeResult(
                    target_config=best_non_speculative.config,
                    draft_model_path=self.draft_model_path,
                    draft_length=draft_length,
                    result=result,
                ))

        best_speculative = self._best_result(attempts)
        if best_speculative is None or best_speculative.result is None:
            decision = self._disabled(
                best_non_speculative, support,
                "Speculative decoding could not be evaluated successfully for any configured draft length.",
            )
        else:
            improvement = _fractional_improvement(
                best_speculative.result.avg_tps,
                best_non_speculative.result.avg_tps,
            )
            enabled = improvement >= self.minimum_improvement
            decision = SpeculativeDecision(
                speculation_enabled=enabled,
                best_non_speculative_result=best_non_speculative,
                best_speculative_result=best_speculative,
                selected_draft_length=best_speculative.draft_length if enabled else None,
                improvement=improvement,
                decision_threshold=self.minimum_improvement,
                decision_reason=(
                    "Speculative decoding exceeded the required throughput improvement "
                    f"threshold ({improvement:.1%} >= {self.minimum_improvement:.1%})."
                    if enabled else
                    "Speculative decoding did not meet the required throughput improvement "
                    f"threshold ({improvement:.1%} < {self.minimum_improvement:.1%})."
                ),
                support=support,
                draft_lengths_tested=tuple(item.draft_length for item in attempts),
            )
        self.save_results(attempts, decision)
        return decision

    @staticmethod
    def _best_result(results: list[SpeculativeResult]) -> SpeculativeResult | None:
        """Choose only fully successful speculative runs by their measured TPS."""
        eligible = [result for result in results if result.success and result.result is not None]
        return max(eligible, key=lambda result: result.result.avg_tps) if eligible else None

    def _disabled(
        self,
        best_non_speculative: TuningRun,
        support: SpeculativeCliSupport,
        reason: str,
    ) -> SpeculativeDecision:
        return SpeculativeDecision(
            speculation_enabled=False,
            best_non_speculative_result=best_non_speculative,
            best_speculative_result=None,
            selected_draft_length=None,
            improvement=None,
            decision_threshold=self.minimum_improvement,
            decision_reason=reason,
            support=support,
            draft_lengths_tested=(),
        )

    def _save_disabled(
        self,
        best_non_speculative: TuningRun,
        support: SpeculativeCliSupport,
        reason: str,
    ) -> SpeculativeDecision:
        decision = self._disabled(best_non_speculative, support, reason)
        self.save_results([], decision)
        return decision

    def save_results(self, results: list[SpeculativeResult], decision: SpeculativeDecision) -> list[Path]:
        """Persist actual speculative attempts and an honest adaptive decision."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "draft_lengths": list(self.draft_lengths),
                "minimum_improvement": self.minimum_improvement,
                "acceptance_metric": "Not available; token-level acceptance statistics were not extracted from llama-server.",
            },
            "results": [self._result_to_dict(result) for result in results],
            "decision": self.decision_to_dict(decision),
        }
        json_path = self.output_directory / "speculative_results.json"
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        csv_path = self.output_directory / "speculative_results.csv"
        fields = ["draft_model_path", "draft_length", "status", "error", "avg_tps", "avg_ttft_ms", "avg_latency_ms", "memory_mb", "success_rate", "acceptance_metric_label"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in results:
                row: dict[str, object] = {
                    "draft_model_path": item.draft_model_path,
                    "draft_length": item.draft_length,
                    "status": "completed" if item.success else "failed",
                    "error": item.error,
                    "acceptance_metric_label": item.acceptance_metric_label,
                }
                if item.result:
                    row.update({
                        "avg_tps": item.result.avg_tps,
                        "avg_ttft_ms": item.result.avg_ttft_ms,
                        "avg_latency_ms": item.result.avg_latency_ms,
                        "memory_mb": item.result.memory_mb,
                        "success_rate": item.result.success_rate,
                    })
                writer.writerow(row)
        return [json_path, csv_path]

    @staticmethod
    def _result_to_dict(result: SpeculativeResult) -> dict[str, Any]:
        return {
            "target_config": _server_config_to_dict(result.target_config),
            "draft_model_path": result.draft_model_path,
            "draft_length": result.draft_length,
            "success": result.success,
            "error": result.error,
            "result": result.result.to_dict() if result.result else None,
            "acceptance_metric_label": result.acceptance_metric_label,
        }

    @staticmethod
    def decision_to_dict(decision: SpeculativeDecision) -> dict[str, Any]:
        return {
            "speculation_enabled": decision.speculation_enabled,
            "selected_draft_length": decision.selected_draft_length,
            "improvement": decision.improvement,
            "decision_threshold": decision.decision_threshold,
            "decision_reason": decision.decision_reason,
            "draft_lengths_tested": list(decision.draft_lengths_tested),
            "support": asdict(decision.support),
            "best_non_speculative": {
                "config": _server_config_to_dict(decision.best_non_speculative_result.config),
                "result": decision.best_non_speculative_result.result.to_dict() if decision.best_non_speculative_result.result else None,
            },
            "best_speculative": AdaptiveSpeculativeController._result_to_dict(decision.best_speculative_result) if decision.best_speculative_result else None,
            "acceptance_metric": "Not available; token-level acceptance statistics were not extracted from llama-server.",
        }


class AutoTuner:
    """Run a sequential, reproducible grid of real llama-server benchmarks."""

    def __init__(
        self,
        *,
        baseline: BaselineMetrics,
        trials: int,
        temperature: float,
        max_tokens: int,
        timeout: float,
        startup_timeout: float,
        output_directory: Path,
    ) -> None:
        if trials < 1:
            raise ValueError("trials must be >= 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")

        self.baseline = baseline
        self.trials = trials
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.output_directory = output_directory
        self.speculative_decision: SpeculativeDecision | None = None

    @staticmethod
    def build_candidates(
        *,
        model_path: str,
        threads: list[int],
        batch_sizes: list[int],
        ubatch_sizes: list[int],
        context_sizes: list[int],
        host: str,
        port: int,
        server_binary: Path,
    ) -> list[ServerConfig]:
        """Build the requested Cartesian product of server configurations."""
        return [
            ServerConfig(
                threads=thread_count,
                batch_size=batch_size,
                ubatch_size=ubatch_size,
                context_size=context_size,
                model_path=model_path,
                host=host,
                port=port,
                server_binary=server_binary,
            )
            for thread_count, batch_size, ubatch_size, context_size in product(
                threads, batch_sizes, ubatch_sizes, context_sizes
            )
        ]

    def run(
        self,
        candidates: list[ServerConfig],
        *,
        speculative_controller: AdaptiveSpeculativeController | None = None,
    ) -> list[TuningRun]:
        """Benchmark the normal grid, then optionally evaluate its one winner.

        The return type deliberately remains the normal-grid run list so callers
        from Prompts 0--5 continue to work unchanged.  The adaptive decision is
        exposed on ``speculative_decision`` and persisted in the normal report.
        """
        if not candidates:
            raise ValueError("At least one configuration is required")

        self.output_directory.mkdir(parents=True, exist_ok=True)
        runs: list[TuningRun] = []
        logger.info("Starting tuning run with %d configuration(s)", len(candidates))

        for index, config in enumerate(candidates, start=1):
            candidate_directory = self.output_directory / "benchmarks" / f"candidate_{index:03d}"
            logger.info(
                "Benchmarking candidate %d/%d: threads=%d batch=%d ubatch=%d context=%d",
                index,
                len(candidates),
                config.threads,
                config.batch_size,
                config.ubatch_size,
                config.context_size,
            )
            runner = BenchmarkRunner(
                trials=self.trials,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                output_directory=candidate_directory,
                startup_timeout=self.startup_timeout,
            )
            try:
                result = runner.run(config)
            except Exception as exc:
                logger.exception("Candidate %d failed", index)
                runs.append(TuningRun(config=config, result=None, error=str(exc)))
            else:
                runs.append(TuningRun(config=config, result=result))

        ranked_runs = self._rank_completed_runs(runs)
        self.save_results(ranked_runs)
        if speculative_controller is not None:
            best = self._best_run(ranked_runs)
            if best is not None:
                # The controller records individual failed draft lengths and
                # never permits a speculative issue to invalidate this sweep.
                try:
                    self.speculative_decision = speculative_controller.run(best)
                except Exception as exc:  # defensive boundary for the pipeline
                    logger.exception("Adaptive speculative evaluation failed unexpectedly")
                    self.speculative_decision = SpeculativeDecision(
                        speculation_enabled=False,
                        best_non_speculative_result=best,
                        best_speculative_result=None,
                        selected_draft_length=None,
                        improvement=None,
                        decision_threshold=speculative_controller.minimum_improvement,
                        decision_reason=(
                            "Speculative decoding could not be evaluated successfully: "
                            f"{exc}"
                        ),
                        support=SpeculativeCliSupport(supported=False, error=str(exc)),
                        draft_lengths_tested=(),
                    )
                # Re-render the normal report with the outcome; no normal
                # benchmark is rerun.
                self.save_results(ranked_runs)
        return ranked_runs

    @staticmethod
    def _rank_completed_runs(runs: list[TuningRun]) -> list[TuningRun]:
        """Assign ranks by measured TPS; failed runs remain unranked."""
        successful_indices = [
            index for index, run in enumerate(runs)
            if run.result is not None and run.result.success_rate == 100.0
        ]
        ranked_indices = sorted(
            successful_indices,
            key=lambda index: runs[index].result.avg_tps if runs[index].result else 0.0,
            reverse=True,
        )
        rank_by_index = {index: rank for rank, index in enumerate(ranked_indices, start=1)}

        ranked: list[TuningRun] = []
        for index, run in enumerate(runs):
            if run.result is None:
                ranked.append(run)
            else:
                ranked.append(
                    TuningRun(
                        config=run.config,
                        result=replace(run.result, rank=rank_by_index.get(index, 0)),
                        error=run.error,
                    )
                )
        return ranked

    def _best_run(self, runs: list[TuningRun]) -> TuningRun | None:
        """Return the fastest completed run that preserved 100% success."""
        eligible = [
            run for run in runs
            if run.result is not None and run.result.success_rate == 100.0
        ]
        return max(eligible, key=lambda run: run.result.avg_tps) if eligible else None

    def _comparison(self, result: TuningResult) -> dict[str, float]:
        """Compute display-only percentage deltas against the supplied baseline."""
        return {
            "tps_improvement_percent": _percent_change(result.avg_tps, self.baseline.avg_tps),
            "ttft_improvement_percent": -_percent_change(
                result.avg_ttft_ms, self.baseline.avg_ttft_ms
            ),
            "latency_improvement_percent": -_percent_change(
                result.avg_latency_ms, self.baseline.avg_latency_ms
            ),
        }

    def save_results(self, runs: list[TuningRun]) -> list[Path]:
        """Save complete tuning data as JSON, CSV, and a Markdown report."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        best = self._best_run(runs)
        payload = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trials": self.trials,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "selection_rule": "maximum average TPS with exactly 100% success rate",
            },
            "baseline": self.baseline.to_dict(),
            "runs": [self._run_to_dict(run) for run in runs],
            "best_configuration": self._run_to_dict(best) if best else None,
            "speculative_decision": (
                AdaptiveSpeculativeController.decision_to_dict(self.speculative_decision)
                if self.speculative_decision else None
            ),
        }
        json_path = self.output_directory / "tuning_results.json"
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        csv_path = self.output_directory / "tuning_results.csv"
        self._write_csv(csv_path, runs)

        report_path = self.output_directory / "tuning_report.md"
        report_path.write_text(self._render_report(runs, best), encoding="utf-8")
        logger.info("Tuning results saved under %s", self.output_directory)
        return [json_path, csv_path, report_path]

    def _run_to_dict(self, run: TuningRun) -> dict[str, Any]:
        """Serialize a run while keeping failed measurements explicitly empty."""
        record: dict[str, Any] = {
            "status": run.status,
            "error": run.error,
            "server_config": _server_config_to_dict(run.config),
            "result": run.result.to_dict() if run.result is not None else None,
            "comparison_to_baseline": self._comparison(run.result) if run.result else None,
        }
        return record

    def _write_csv(self, path: Path, runs: list[TuningRun]) -> None:
        """Write one flat structured row per configuration attempt."""
        fields = [
            "status", "error", "rank", "threads", "batch_size", "ubatch_size",
            "context_size", "model_path", "avg_tps", "avg_ttft_ms", "avg_latency_ms",
            "memory_mb", "p95_duration_s", "success_rate", "tps_improvement_percent",
            "ttft_improvement_percent", "latency_improvement_percent",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for run in runs:
                row: dict[str, object] = {
                    "status": run.status,
                    "error": run.error,
                    "threads": run.config.threads,
                    "batch_size": run.config.batch_size,
                    "ubatch_size": run.config.ubatch_size,
                    "context_size": run.config.context_size,
                    "model_path": run.config.model_path,
                }
                if run.result is not None:
                    row.update({
                        "rank": run.result.rank,
                        "avg_tps": run.result.avg_tps,
                        "avg_ttft_ms": run.result.avg_ttft_ms,
                        "avg_latency_ms": run.result.avg_latency_ms,
                        "memory_mb": run.result.memory_mb,
                        "p95_duration_s": run.result.p95_duration_s,
                        "success_rate": run.result.success_rate,
                    })
                    row.update(self._comparison(run.result))
                writer.writerow(row)

    def _render_report(self, runs: list[TuningRun], best: TuningRun | None) -> str:
        """Render a concise Markdown report from actual saved tuning results."""
        lines = [
            "# Graviton Auto-Tuning Report",
            "",
            "## Baseline",
            "",
            "| Model | TPS | TTFT (ms) | Latency (ms/token) | Success | Threads | Batch |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| {self.baseline.model} | {self.baseline.avg_tps:.2f} | "
                f"{self.baseline.avg_ttft_ms:.1f} | {self.baseline.avg_latency_ms:.1f} | "
                f"{self.baseline.success_rate:.1f}% | {self.baseline.threads} | "
                f"{self.baseline.batch_size} |"
            ),
            "",
            "## Tested Configurations",
            "",
            "| Rank | Threads | Batch | UBatch | Context | TPS | TTFT (ms) | Latency (ms/token) | Memory (MB) | Success | TPS Δ | Status |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for run in runs:
            if run.result is None:
                lines.append(
                    f"| -- | {run.config.threads} | {run.config.batch_size} | "
                    f"{run.config.ubatch_size} | {run.config.context_size} | -- | -- | -- | -- | -- | -- | failed |"
                )
                continue
            comparison = self._comparison(run.result)
            lines.append(
                f"| {run.result.rank or '--'} | {run.config.threads} | {run.config.batch_size} | "
                f"{run.config.ubatch_size} | {run.config.context_size} | {run.result.avg_tps:.2f} | "
                f"{run.result.avg_ttft_ms:.1f} | {run.result.avg_latency_ms:.1f} | "
                f"{run.result.memory_mb:.1f} | {run.result.success_rate:.1f}% | "
                f"{comparison['tps_improvement_percent']:+.1f}% | completed |"
            )

        lines.extend(["", "## Best Configuration", ""])
        if best is None or best.result is None:
            lines.append("No tested configuration completed with 100% success rate.")
        else:
            comparison = self._comparison(best.result)
            lines.extend([
                (
                    f"Selected configuration: **threads={best.config.threads}, "
                    f"batch={best.config.batch_size}, ubatch={best.config.ubatch_size}, "
                    f"context={best.config.context_size}**."
                ),
                "",
                f"- Average TPS: {best.result.avg_tps:.2f}",
                f"- Success rate: {best.result.success_rate:.1f}%",
                f"- TPS improvement over baseline: {comparison['tps_improvement_percent']:+.1f}%",
                f"- TTFT change versus baseline: {comparison['ttft_improvement_percent']:+.1f}%",
                f"- Latency change versus baseline: {comparison['latency_improvement_percent']:+.1f}%",
            ])
        lines.append("")
        if self.speculative_decision is not None:
            decision = self.speculative_decision
            lines.extend([
                "## Speculative Decoding Evaluation",
                "",
                f"- Draft lengths tested: {', '.join(map(str, decision.draft_lengths_tested)) or 'none'}",
                f"- Decision threshold: {decision.decision_threshold:.1%}",
                f"- Speculation enabled: **{decision.speculation_enabled}**",
                f"- Decision: {decision.decision_reason}",
                "- Acceptance metric: Not available; token-level acceptance statistics were not extracted from llama-server.",
            ])
            if decision.best_speculative_result and decision.best_speculative_result.result:
                speculative = decision.best_speculative_result
                lines.extend([
                    f"- Draft model: `{speculative.draft_model_path}`",
                    f"- Best speculative draft length: {speculative.draft_length}",
                    f"- Best speculative TPS: {speculative.result.avg_tps:.2f}",
                    f"- Measured throughput change: {decision.improvement:.1%}" if decision.improvement is not None else "- Measured throughput change: unavailable",
                ])
            else:
                lines.append("- Best speculative result: unavailable (fallback retained).")
            lines.append("")
        return "\n".join(lines)

def _server_config_to_dict(config: ServerConfig) -> dict[str, object]:
    """Serialize paths as strings for JSON output."""
    return {
        "threads": config.threads,
        "batch_size": config.batch_size,
        "ubatch_size": config.ubatch_size,
        "context_size": config.context_size,
        "model_path": config.model_path,
        "draft_model_path": config.draft_model_path,
        "draft_max": config.draft_max,
        "host": config.host,
        "port": config.port,
        "server_binary": str(config.server_binary),
    }


def _percent_change(new_value: float, reference_value: float) -> float:
    """Return percent change, avoiding a divide-by-zero crash in reports."""
    return ((new_value - reference_value) / reference_value * 100.0) if reference_value else 0.0


def _fractional_improvement(new_value: float, reference_value: float) -> float:
    """Return a fractional performance change, safely handling a zero baseline."""
    return ((new_value - reference_value) / reference_value) if reference_value else 0.0


def _main() -> int:
    """Run one reproducible real-benchmark grid against a fixed GGUF model."""
    parser = argparse.ArgumentParser(description="Auto-tune llama-server on one fixed model.")
    parser.add_argument("--model-path", required=True, help="Path to a local GGUF model.")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--ubatch-sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[2048])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--server-binary", type=Path, default=ServerConfig.__dataclass_fields__["server_binary"].default)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("results/auto_tune"))
    parser.add_argument("--draft-model-path", help="Optional compatible GGUF draft model for adaptive speculation.")
    parser.add_argument("--draft-lengths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--minimum-speculative-improvement", type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    baseline = BaselineMetrics(
        model="qwen2.5-0.5b-instruct-fp16.gguf",
        avg_tps=29.63,
        avg_ttft_ms=515.6,
        avg_latency_ms=28.8,
        success_rate=100.0,
        threads=2,
        batch_size=512,
        max_tokens=128,
    )
    if Path(args.model_path).name != baseline.model:
        parser.error(
            "--model-path must reference the completed baseline model "
            f"({baseline.model}); model changes are outside this tuner"
        )
    candidates = AutoTuner.build_candidates(
        model_path=args.model_path,
        threads=args.threads,
        batch_sizes=args.batch_sizes,
        ubatch_sizes=args.ubatch_sizes,
        context_sizes=args.context_sizes,
        host=args.host,
        port=args.port,
        server_binary=args.server_binary,
    )
    tuner = AutoTuner(
        baseline=baseline,
        trials=args.trials,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        output_directory=args.output,
        startup_timeout=args.startup_timeout,
    )

    try:
        controller = None
        if args.draft_model_path:
            controller = AdaptiveSpeculativeController(
                draft_model_path=args.draft_model_path,
                draft_lengths=args.draft_lengths,
                minimum_improvement=args.minimum_speculative_improvement,
                trials=args.trials,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                startup_timeout=args.startup_timeout,
                output_directory=args.output / "speculative",
            )
        runs = tuner.run(candidates, speculative_controller=controller)
    except ValueError as exc:
        print(f"Auto-tuning setup failed: {exc}")
        return 1

    best = tuner._best_run(runs)
    print(f"Auto-tuning completed: {len(runs)} configuration(s) tested.")
    if best is not None and best.result is not None:
        print("Best configuration:", _server_config_to_dict(best.config))
        print("Best result:", best.result)
    else:
        print("No configuration completed with 100% success rate.")
    if tuner.speculative_decision is not None:
        print("Speculative decision:", "ENABLE" if tuner.speculative_decision.speculation_enabled else "DISABLE")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
