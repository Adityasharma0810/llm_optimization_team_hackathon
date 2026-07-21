"""Unit tests for Prompt 6 without a llama-server or benchmark dependencies."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from auto_tune import (
    AdaptiveSpeculativeController,
    ServerConfig,
    SpeculativeCliSupport,
    TuningResult,
    TuningRun,
)


def result(tps: float, *, success_rate: float = 100.0) -> TuningResult:
    return TuningResult(
        rank=1, model="target.gguf", avg_ttft_ms=10.0, avg_latency_ms=2.0,
        success_rate=success_rate, threads=2, batch_size=512, draft_max=None,
        speculative_enabled=False, avg_tps=tps, p95_duration_s=1.0,
        memory_mb=100.0, score=tps,
    )


class FakeRunner:
    outcomes: dict[int, object] = {}

    def __init__(self, **_: object) -> None:
        pass

    def run(self, config: ServerConfig) -> TuningResult:
        outcome = self.outcomes[config.draft_max or 0]
        if isinstance(outcome, Exception):
            raise outcome
        return result(float(outcome))


class AdaptiveSpeculativeControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.draft = Path(self.tempdir.name) / "draft.gguf"
        self.draft.touch()
        self.best = TuningRun(
            ServerConfig(2, 512, 512, 2048, "target.gguf", server_binary=Path("/fake/server")),
            result(100.0),
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def controller(self, **kwargs: object) -> AdaptiveSpeculativeController:
        return AdaptiveSpeculativeController(
            draft_model_path=str(self.draft), draft_lengths=[1, 2, 4, 8],
            support_checker=lambda _: SpeculativeCliSupport(supported=True),
            runner_factory=FakeRunner, output_directory=Path(self.tempdir.name) / "output",
            **kwargs,
        )

    def test_creates_speculative_config_and_sweeps_each_length(self) -> None:
        FakeRunner.outcomes = {1: 95.0, 2: 101.0, 4: 110.0, 8: 105.0}
        decision = self.controller().run(self.best)
        self.assertTrue(decision.speculation_enabled)
        self.assertEqual(decision.selected_draft_length, 4)
        self.assertEqual(decision.draft_lengths_tested, (1, 2, 4, 8))
        self.assertAlmostEqual(decision.improvement or 0, 0.10)
        self.assertTrue((Path(self.tempdir.name) / "output" / "speculative_results.json").is_file())

    def test_threshold_disables_a_smaller_gain(self) -> None:
        FakeRunner.outcomes = {1: 100.0, 2: 104.0, 4: 101.0, 8: 99.0}
        decision = self.controller(minimum_improvement=0.05).run(self.best)
        self.assertFalse(decision.speculation_enabled)
        self.assertIsNone(decision.selected_draft_length)
        self.assertAlmostEqual(decision.improvement or 0, 0.04)

    def test_failed_length_is_recorded_and_other_lengths_continue(self) -> None:
        FakeRunner.outcomes = {1: RuntimeError("startup failed"), 2: 106.0, 4: 0.0, 8: 101.0}
        decision = self.controller().run(self.best)
        self.assertTrue(decision.speculation_enabled)
        self.assertEqual(decision.selected_draft_length, 2)
        self.assertEqual(decision.draft_lengths_tested, (1, 2, 4, 8))

    def test_missing_draft_model_falls_back_without_running_server(self) -> None:
        self.draft.unlink()
        FakeRunner.outcomes = {}
        decision = self.controller().run(self.best)
        self.assertFalse(decision.speculation_enabled)
        self.assertIn("unavailable", decision.decision_reason)
        self.assertEqual(decision.draft_lengths_tested, ())

    def test_all_failures_fall_back_and_never_claim_acceptance_rate(self) -> None:
        FakeRunner.outcomes = {1: RuntimeError("x"), 2: RuntimeError("x"), 4: RuntimeError("x"), 8: RuntimeError("x")}
        controller = self.controller()
        decision = controller.run(self.best)
        self.assertFalse(decision.speculation_enabled)
        self.assertIsNone(decision.best_speculative_result)
        self.assertNotIn("true acceptance rate", decision.decision_reason.lower())
        report = (Path(self.tempdir.name) / "output" / "speculative_results.json").read_text()
        self.assertIn("token-level acceptance statistics were not extracted", report)


if __name__ == "__main__":
    unittest.main()
