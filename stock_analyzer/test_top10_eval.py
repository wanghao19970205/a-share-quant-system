import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from stock_analyzer import top10_eval


class Top10CacheFingerprintTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "code": ["600881"], "date": ["2026-07-20"],
            "rank": [1], "pred": [0.12345678],
        })

    def test_quote_file_change_invalidates_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            price_dir = Path(directory) / "price"
            price_dir.mkdir()
            quote = price_dir / "600881.parquet"
            quote.write_bytes(b"old")
            with mock.patch.dict(os.environ, {"QUANT_DATA_DIR": directory}):
                before = top10_eval._fingerprint(self.frame)
                quote.write_bytes(b"new-content")
                after = top10_eval._fingerprint(self.frame)

        self.assertNotEqual(before, after)

    def test_same_input_requires_current_cache_version(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"QUANT_DATA_DIR": directory}):
                entry = {
                    "cache_version": top10_eval._CACHE_VERSION,
                    "date": "2026-07-20",
                    "codes": ["600881"],
                    "fingerprint": top10_eval._fingerprint(self.frame),
                    "rows": [{"code": "600881"}],
                }
                self.assertTrue(top10_eval._same_input(entry, self.frame))
                entry["cache_version"] -= 1
                self.assertFalse(top10_eval._same_input(entry, self.frame))

    def test_failed_codes_force_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"QUANT_DATA_DIR": directory}):
                entry = {
                    "cache_version": top10_eval._CACHE_VERSION,
                    "date": "2026-07-20",
                    "codes": ["600881"],
                    "fingerprint": top10_eval._fingerprint(self.frame),
                    "failed_codes": ["600881"],
                    "rows": [{"code": "600881", "stale": True}],
                }
                self.assertFalse(top10_eval._same_input(entry, self.frame))

    def test_refresh_uses_configured_workers_and_reports_timing(self):
        frames = {
            "白名单": self.frame,
            "全A": pd.DataFrame(),
            "创新药": pd.DataFrame(),
        }
        evaluated = {}

        def evaluate(codes, **kwargs):
            evaluated["workers"] = kwargs["max_workers"]
            return {
                "model": "test",
                "rows": [{
                    "code": code,
                    "available": True,
                    "rank_score": 1.0,
                } for code in codes],
            }

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {
                    "SNAPSHOT_DIR": directory,
                    "TOP10_EVAL_WORKERS": "4",
                }), \
                mock.patch.object(top10_eval, "ranking_frames", return_value=frames), \
                mock.patch.object(
                    top10_eval.candidate_eval,
                    "latest_quotes",
                    return_value={"600881": {"quote_date": "2026-07-20"}},
                ) as latest_quotes, \
                mock.patch.object(top10_eval.candidate_eval, "evaluate_top", side_effect=evaluate) as evaluate_top:
                summary = top10_eval.refresh()
                cache = top10_eval.load()

        self.assertEqual(evaluated["workers"], 4)
        self.assertEqual(summary["unique_evaluated"], 1)
        self.assertEqual(summary["timing"]["workers"], 4)
        self.assertEqual(summary["timing"]["quotes_refreshed"], 1)

        self.assertEqual(
            cache["白名单"]["quant_fingerprint"],
            top10_eval._fingerprint(self.frame),
        )
        self.assertNotEqual(
            cache["白名单"]["fingerprint"],
            cache["白名单"]["quant_fingerprint"],
        )
        latest_quotes.assert_called_once()
        self.assertEqual(
            evaluate_top.call_args.kwargs["freshness_bucket"],
            latest_quotes.call_args.kwargs["freshness_bucket"],
        )
        self.assertIn("evaluate_seconds", summary["timing"])

    def test_mobile_snapshot_records_model_metadata(self):
        frames = {"白名单": self.frame, "全A": pd.DataFrame(), "创新药": pd.DataFrame()}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {
                    "SNAPSHOT_DIR": directory,
                    "QUANT_DATA_DIR": directory,
                    "TOP10_SOURCE_JOB": "daily-light",
                }), \
                mock.patch.object(top10_eval, "_published_model_metadata", return_value={
                    "model": "active_quant",
                    "published_at": "2026-07-22T15:05:00",
                    "job": "daily-light",
                    "snapshot_dir": directory,
                }):
            top10_eval._write_mobile_snapshot(frames, {})
            with open(Path(directory) / "mobile_snapshot.json", encoding="utf-8") as fh:
                snapshot = __import__("json").load(fh)

        self.assertEqual(snapshot["version"], 2)
        self.assertEqual(snapshot["model"]["published_at"], "2026-07-22T15:05:00")
        self.assertEqual(snapshot["model"]["job"], "daily-light")

    def test_invalid_worker_setting_uses_default(self):
        with mock.patch.dict(os.environ, {"TOP10_EVAL_WORKERS": "invalid"}):
            self.assertEqual(top10_eval._worker_count(), 4)


if __name__ == "__main__":
    unittest.main()
