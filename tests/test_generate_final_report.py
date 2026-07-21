"""Tests for report formatting from saved data, with no benchmark execution."""

from unittest import TestCase

from generate_final_report import _hardware, render_report


class FinalReportTests(TestCase):
    def test_uses_machine_architecture_and_warns_about_token_mismatch(self) -> None:
        hardware = _hardware({"hardware": {"architecture": "8", "cores": 2}, "os": {"machine": "aarch64"}, "build": {"kleidai_cpu": "ON"}})
        self.assertEqual(hardware["architecture"], "aarch64")
        tuning = {"baseline": {"model": "target.gguf", "max_tokens": 128}, "metadata": {"max_tokens": 32, "selection_rule": "maximum TPS"}, "runs": [], "best_configuration": {"server_config": {}, "result": {}, "comparison_to_baseline": {}}}
        speculative = {"metadata": {"acceptance_metric": "not extracted"}, "results": [], "decision": {"speculation_enabled": False, "decision_reason": "threshold not met", "decision_threshold": 0.05}}
        report = render_report(tuning, speculative, {"hardware": {"architecture": "8"}, "os": {"machine": "aarch64"}, "build": {}})
        self.assertIn("baseline used max_tokens=128, while this sweep used max_tokens=32", report)
