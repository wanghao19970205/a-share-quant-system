import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from quant import daily_optimization_pipeline as pipeline


def decision_metrics(daily_return: float, **overrides):
    metrics = {
        "mean_return": daily_return,
        "mean_filled_names": 1.5,
        "max_drawdown": -0.2,
        "positive_months": 4,
        "daily_returns": [
            {
                "date": str(pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)),
                "return": daily_return,
            }
            for index in range(12)
        ],
    }
    metrics.update(overrides)
    return metrics


class DailyOptimizationPipelineTest(unittest.TestCase):
    def test_failed_branch_routes_in_registered_order(self):
        result = pipeline.choose_next_branch(
            "open_buyin_ridge",
            {"metrics": decision_metrics(-0.001)},
            decision_metrics(0.0),
        )
        self.assertEqual(result["status"], "branch_failed")
        self.assertEqual(result["next_branch"], "open_ridge")

    def test_paired_gain_and_top2_fill_gate_accept_one_position(self):
        result = pipeline.choose_next_branch(
            "open_ridge",
            {"metrics": decision_metrics(0.001, mean_filled_names=1.0)},
            decision_metrics(0.0),
        )
        self.assertEqual(result["status"], "candidate_requires_independent_reproduction")
        self.assertEqual(result["next_branch"], "independent_reproduction")
        self.assertGreater(
            result["paired_incumbent_comparison"]["ci95"][0], 0.0,
        )

    def test_positive_candidate_without_relative_gain_fails(self):
        result = pipeline.choose_next_branch(
            "open_ridge",
            {"metrics": decision_metrics(0.001)},
            decision_metrics(0.001),
        )
        self.assertEqual(result["status"], "branch_failed")

    def test_last_failed_branch_requires_human_review(self):
        result = pipeline.choose_next_branch(
            "open_random_forest",
            {"metrics": decision_metrics(-0.001, mean_filled_names=2.0)},
            decision_metrics(0.0),
        )
        self.assertEqual(result["next_branch"], "human_review_required")

    def test_metrics_use_no_refill_and_next_open_buyability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "predictions.parquet"
            pd.DataFrame({
                "date": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-02"] * 3),
                "code": ["000001", "000002", "000003"] * 2,
                "pred": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
                "open_ret_1d": [0.10, 0.02, 0.01, -0.01, 0.03, 0.02],
                "buyable_next": [False, True, True, True, False, True],
            }).to_parquet(path)
            metrics = pipeline._metrics(path)
        self.assertEqual(metrics["mean_filled_names"], 1.0)
        self.assertAlmostEqual(metrics["fill_rate"], 0.5)
        self.assertLess(metrics["mean_return"], 0.01)

    def test_command_uses_isolated_target_and_scheduler_disabled(self):
        command = pipeline._train_command(
            "open_buyin_ridge", "daily_auto_open_buyin_ridge_v1", Path("/tmp/cache"), 12
        )
        self.assertIn("--train-target-mode", command)
        self.assertIn("open-buyin-mask", command)
        self.assertNotIn("--positive-only", command)
        self.assertIn("--purge-horizon", command)
        extra_trees_command = pipeline._train_command(
            "open_extratrees", "extra_trees_run", Path("/tmp/cache"), 12
        )
        self.assertIn("--extra-trees-weight", extra_trees_command)
        self.assertEqual(extra_trees_command[extra_trees_command.index("--extra-trees-weight") + 1], "0.2")
        random_forest_command = pipeline._train_command(
            "open_random_forest", "random_forest_run", Path("/tmp/cache"), 12
        )
        self.assertIn("--random-forest", random_forest_command)
        self.assertEqual(random_forest_command[random_forest_command.index("--random-forest-weight") + 1], "0.2")
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(pipeline, "_train_command", return_value=["true"]), mock.patch.object(
                pipeline, "_run_subprocess", return_value=subprocess.CompletedProcess(["true"], 0, "", "")
            ), mock.patch.object(pipeline.config, "QUANT_DIR", temporary):
                result = pipeline.run_once("open_buyin_ridge", Path(temporary), 1, execute=True)
            self.assertEqual(result["status"], "branch_artifact_missing")

    def test_run_once_sets_explicit_execution_environment(self):
        captured = {}

        def fake_run(command, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(command, 1, stdout="out", stderr="err")

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            pipeline, "_train_command", return_value=["train"]
        ), mock.patch.object(pipeline, "_run_subprocess", side_effect=fake_run), mock.patch.object(
            pipeline.config, "QUANT_DIR", temporary
        ):
            result = pipeline.run_once(
                "open_buyin_ridge", Path(temporary) / "research", 1, max_retries=0
            )
        self.assertEqual(result["status"], "branch_execution_failed")
        self.assertEqual(captured["env"]["SCHEDULER_DISABLED"], "1")
        self.assertEqual(captured["env"]["QUANT_BT_FILL"], "next_open")
        self.assertEqual(captured["env"]["QUANT_BT_FILTER_UNTRADABLE"], "1")
        self.assertEqual(captured["env"]["QUANT_BT_COST_ROUNDTRIP"], "0.002")
        self.assertEqual(captured["timeout"], 8 * 60 * 60)

    def test_transient_failure_retries_and_persists_failure(self):
        completed = subprocess.CompletedProcess(["train"], 137, stdout="", stderr="killed")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            pipeline, "_train_command", return_value=["train"]
        ), mock.patch.object(pipeline, "_run_subprocess", return_value=completed) as run, mock.patch.object(
            pipeline.config, "QUANT_DIR", temporary
        ):
            result = pipeline.run_once(
                "open_buyin_ridge", Path(temporary) / "research", 1, max_retries=1
            )
            manifest_path = Path(result["output_dir"]) / "attempt.json"
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(result["status"], "branch_execution_failed")
        self.assertEqual(persisted["last_failure"]["returncode"], 137)
        pipeline.validate_attempt_manifest(persisted, verify_artifact=False)

    def test_unique_prefix_prevents_stale_prediction_reuse(self):
        prefixes = iter(["run_a", "run_b"])
        completed = subprocess.CompletedProcess(["true"], 0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            pipeline, "_new_run_id", side_effect=lambda branch: next(prefixes)
        ), mock.patch.object(pipeline, "_train_command", return_value=["true"]), mock.patch.object(
            pipeline.subprocess, "run", return_value=completed
        ), mock.patch.object(pipeline.config, "QUANT_DIR", temporary):
            old = Path(temporary) / "daily_auto_open_buyin_ridge_v1_bt_ridge_lightgbm_ranker_ensemble_predictions.parquet"
            old.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"stale": [1]}).to_parquet(old)
            first = pipeline.run_once("open_buyin_ridge", Path(temporary) / "research", 1)
            second = pipeline.run_once("open_buyin_ridge", Path(temporary) / "research", 1)
        self.assertEqual(first["status"], "branch_artifact_missing")
        self.assertEqual(second["status"], "branch_artifact_missing")
        self.assertNotEqual(first["prediction_path"], second["prediction_path"])

    def test_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            pipeline, "_train_command", return_value=["true"]
        ), mock.patch.object(pipeline.config, "QUANT_DIR", temporary):
            result = pipeline.run_once(
                "open_buyin_ridge", Path(temporary) / "research", 1, execute=False
            )
            manifest_path = Path(result["output_dir"]) / "attempt.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["branch"] = "open_ridge"
            with self.assertRaisesRegex(RuntimeError, "state hash mismatch"):
                pipeline.validate_attempt_manifest(manifest, verify_artifact=False)

    def test_pipeline_automatically_runs_next_failed_branch(self):
        failed = decision_metrics(-0.001)
        passed = decision_metrics(0.001, mean_filled_names=1.2)
        baseline = decision_metrics(0.0)
        calls = []

        def fake_run_once(branch, research_root, **kwargs):
            calls.append(branch)
            metrics = failed if branch == "open_buyin_ridge" else passed
            output_dir = Path(research_root) / "attempts" / branch
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / "attempt.json"
            pipeline.atomic_json({"branch": branch}, manifest_path)
            return {
                "branch": branch,
                "run_id": branch,
                "output_dir": str(output_dir),
                "status": "evaluated",
                "metrics": metrics,
                "baseline_metrics": baseline,
                "decision": pipeline.choose_next_branch(
                    branch, {"metrics": metrics}, baseline,
                ),
            }

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            pipeline, "run_once", side_effect=fake_run_once
        ):
            result = pipeline.run_pipeline(
                "open_buyin_ridge", Path(temporary), recent_windows=1
            )
        self.assertEqual(calls, ["open_buyin_ridge", "open_ridge"])
        self.assertEqual(result["status"], "candidate_requires_independent_reproduction")
        self.assertEqual(result["selected"], "open_ridge")

    def test_pipeline_stops_on_execution_failure(self):
        calls = []

        def fake_run_once(branch, research_root, **kwargs):
            calls.append(branch)
            output_dir = Path(research_root) / "attempts" / branch
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / "attempt.json"
            pipeline.atomic_json({"branch": branch}, manifest_path)
            return {
                "branch": branch,
                "run_id": branch,
                "output_dir": str(output_dir),
                "status": "branch_execution_failed",
            }

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            pipeline, "run_once", side_effect=fake_run_once
        ):
            result = pipeline.run_pipeline("open_buyin_ridge", Path(temporary), recent_windows=1)
        self.assertEqual(calls, ["open_buyin_ridge"])
        self.assertEqual(result["status"], "pipeline_blocked")
        self.assertEqual(result["reason"], "branch_execution_failed")

    def test_cycle_lock_rejects_concurrent_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with pipeline._cycle_lock(root):
                with self.assertRaisesRegex(RuntimeError, "another daily optimization"):
                    with pipeline._cycle_lock(root):
                        pass


if __name__ == "__main__":
    unittest.main()
