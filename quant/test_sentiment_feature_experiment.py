import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant import sentiment_feature_experiment as experiment


def _news(rows):
    defaults = {
        "category": "stock_news", "sentiment": 1.0, "llm_impact": 1.0,
        "llm_relevance": 1.0, "llm_certainty": 1.0, "llm_novelty": 1.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class SentimentFeatureExperimentTest(unittest.TestCase):
    def test_parse_publish_time_accepts_mixed_date_and_datetime(self):
        parsed = experiment.parse_publish_time(pd.Series(["2026-07-01", "2026-07-01 14:30:00"]))

        self.assertTrue(parsed.notna().all())
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2026-07-01"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2026-07-01 14:30:00"))

    def test_daily_features_respect_close_cutoff_and_date_only_delay(self):
        news = _news([
            {"publish_time": "2026-07-01 14:30:00", "sentiment": 1.0},
            {"publish_time": "2026-07-01 15:01:00", "sentiment": -1.0},
            {"publish_time": "2026-07-01", "sentiment": -1.0},
        ])
        dates = pd.DatetimeIndex(["2026-07-01", "2026-07-02"])

        features = experiment._daily_features(news, dates, cutoff_hour=15, half_lives=(3,))

        self.assertEqual(features.loc[pd.Timestamp("2026-07-01"), "news_sent_article_count_1d"], 1)
        self.assertEqual(features.loc[pd.Timestamp("2026-07-02"), "news_sent_article_count_1d"], 2)
        self.assertGreater(features.loc[pd.Timestamp("2026-07-01"), "news_sent_local_decay_3d"], 0)
        self.assertLess(features.loc[pd.Timestamp("2026-07-02"), "news_sent_local_decay_3d"], 1)

    def test_build_feature_panel_limits_rows_to_watchlist(self):
        panel = pd.DataFrame({
            "code": ["000001", "000002"], "date": ["2026-07-01", "2026-07-01"],
            "target_ret_3d": [0.01, -0.01], "base_factor": [1.0, 2.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            news_dir = Path(directory)
            _news([{"publish_time": "2026-07-01 10:00:00"}]).to_parquet(
                news_dir / "000001.parquet", index=False)
            result = experiment.build_feature_panel(panel, news_dir, ["000001"], half_lives=(3,))

        self.assertEqual(result["code"].tolist(), ["000001"])
        self.assertIn("news_sent_local_decay_3d", result.columns)

    def test_run_experiment_refuses_existing_output(self):
        panel = pd.DataFrame({
            "code": ["000001"], "date": [pd.Timestamp("2026-07-01")],
            "target_ret_3d": [0.01], "base_factor": [1.0],
            "news_sent_local_decay_3d": [0.2],
        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                experiment.run_experiment(
                    panel, output, 3, pd.Timestamp("2026-06-01"),
                    pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-16"), models=("ridge",),
                )

    def test_base_features_exclude_sentiment_columns(self):
        panel = pd.DataFrame({
            "code": ["000001"], "date": [pd.Timestamp("2026-07-01")],
            "target_ret_3d": [0.01], "base_factor": [1.0],
            "news_sent_local_decay_3d": [0.2],
        })

        self.assertEqual(experiment._base_features(panel, 3), ["base_factor"])

    def test_prediction_metrics_use_test_predictions(self):
        predictions = pd.DataFrame({
            "target_ret_3d": [0.1, -0.1, 0.2], "pred": [0.2, -0.2, 0.1],
        })

        metrics = experiment._prediction_metrics(predictions, 3)

        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["direction_accuracy"], 1.0)
        self.assertIsNotNone(metrics["rank_ic"])


if __name__ == "__main__":
    unittest.main()
