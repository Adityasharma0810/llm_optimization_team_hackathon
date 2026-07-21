"""Mocked unit coverage for the autotuner's non-network components."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

from auto_tune import (
    AutoTuner,
    BaselineMetrics,
    BenchmarkRunner,
    ServerConfig,
    ServerManager,
    TuningResult,
)


def _result(tps: float = 10.0) -> TuningResult:
    return TuningResult(
        rank=0, model="target.gguf", avg_ttft_ms=5.0, avg_latency_ms=1.0,
        success_rate=100.0, threads=2, batch_size=512, draft_max=None,
        speculative_enabled=False, avg_tps=tps, p95_duration_s=1.0,
        memory_mb=42.0, score=0.0,
    )


class ServerConfigTests(TestCase):
    def test_speculative_cli_arguments_and_legacy_metric_aliases(self) -> None:
        config = ServerConfig(2, 512, 256, 2048, "target.gguf", "draft.gguf", 4)
        self.assertTrue(config.speculative_enabled)
        self.assertIn("--spec-draft-n-max", config.to_cli_args())
        result = _result()
        self.assertEqual(result.avg_tokens_per_second, result.avg_tps)
        self.assertEqual(result.avg_memory_usage_mb, result.memory_mb)
        self.assertNotIn("avg_tokens_per_second", result.to_dict())

    def test_draft_configuration_requires_both_values(self) -> None:
        with self.assertRaises(ValueError):
            ServerConfig(2, 512, 256, 2048, "target.gguf", draft_model_path="draft.gguf")


class ServerManagerTests(TestCase):
    @mock.patch.object(ServerManager, "_wait_until_ready")
    @mock.patch("auto_tune.subprocess.Popen")
    def test_start_and_stop_manage_one_process(self, popen: mock.Mock, ready: mock.Mock) -> None:
        process = mock.Mock(pid=1234, stdout=StringIO(), stderr=StringIO())
        process.poll.side_effect = [None, None, 0]
        popen.return_value = process
        manager = ServerManager(ServerConfig(2, 512, 256, 2048, "target.gguf"))
        manager.start()
        manager.stop()
        ready.assert_called_once()
        process.terminate.assert_called_once()


class BenchmarkWrapperTests(TestCase):
    @mock.patch("auto_tune.ServerManager")
    @mock.patch("benchmark.benchmark.BenchmarkRunner")
    def test_wrapper_converts_existing_benchmark_records(
        self, project_runner: mock.Mock, manager: mock.Mock
    ) -> None:
        manager.return_value.start.return_value = None
        project_runner.return_value.run_all.return_value = [
            SimpleNamespace(status="success", tokens_per_second=12.0, ttft=2.0, latency=1.0, memory_usage=10.0, duration=3.0)
        ]
        output = BenchmarkRunner(trials=1).run(ServerConfig(2, 512, 256, 2048, "target.gguf"))
        self.assertEqual(output.avg_tps, 12.0)
        manager.return_value.stop.assert_called_once()


class AutoTunerTests(TestCase):
    def test_candidates_rank_and_save_without_a_real_server(self) -> None:
        baseline = BaselineMetrics("target.gguf", 8.0, 1.0, 1.0, 100.0, 2, 512, 128)
        with mock.patch("auto_tune.BenchmarkRunner") as runner:
            runner.return_value.run.side_effect = [_result(9.0), _result(11.0)]
            with __import__("tempfile").TemporaryDirectory() as temp_dir:
                tuner = AutoTuner(baseline=baseline, trials=1, temperature=0.0, max_tokens=32, timeout=1.0, startup_timeout=1.0, output_directory=Path(temp_dir))
                candidates = AutoTuner.build_candidates(model_path="target.gguf", threads=[1, 2], batch_sizes=[512], ubatch_sizes=[256], context_sizes=[2048], host="127.0.0.1", port=8080, server_binary=Path("llama-server"))
                runs = tuner.run(candidates)
                self.assertEqual(tuner._best_run(runs).result.avg_tps, 11.0)  # type: ignore[union-attr]
                self.assertTrue((Path(temp_dir) / "tuning_results.json").is_file())
