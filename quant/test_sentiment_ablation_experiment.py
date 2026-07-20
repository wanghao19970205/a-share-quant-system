import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant import sentiment_ablation_experiment as ablation


class SentimentAblationExperimentTest(unittest.TestCase):
    def test_feature_groups_are_complete_and_disjoint(self):
        columns = [
            "news_sent_local_decay_3d", "news_sent_structured_decay_3d",
            "news_sent_hybrid_decay_3d", "news_sent_article_count_1d",
            "news_sent_positive_ratio_7d", "news_sent_llm_coverage_7d",
            "news_sent_relevance_7d", "news_sent_announcement_count_7d",
        ]

        groups = ablation.sentiment_feature_groups(columns)
        flattened = [column for values in groups.values() for column in values]

        self.assertEqual(set(flattened), set(columns))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_selection_gate_requires_consistent_non_negative_gain(self):
        passing = {
            "rank_ic_gain": 0.01, "sharpe_gain": 0.0,
            "monthly_win_rate": 0.5, "drawdown_change": -0.03,
        }

        self.assertTrue(ablation._selection_pass(passing))
        self.assertFalse(ablation._selection_pass({**passing, "rank_ic_gain": 0.0}))
        self.assertFalse(ablation._selection_pass({**passing, "sharpe_gain": -0.01}))
        self.assertFalse(ablation._selection_pass({**passing, "monthly_win_rate": 0.49}))
        self.assertFalse(ablation._selection_pass({**passing, "drawdown_change": -0.031}))

    def test_confirmation_gate_is_stricter(self):
        passing = {
            "rank_ic_gain": 0.01, "sharpe_gain": 0.10,
            "monthly_win_rate": 0.55, "drawdown_change": -0.02,
        }

        self.assertTrue(ablation._confirmation_pass(passing))
        self.assertFalse(ablation._confirmation_pass({**passing, "sharpe_gain": 0.09}))
        self.assertFalse(ablation._confirmation_pass({**passing, "monthly_win_rate": 0.54}))
        self.assertFalse(ablation._confirmation_pass({**passing, "drawdown_change": -0.021}))

    def test_run_ablation_refuses_existing_output(self):
        panel = pd.DataFrame({
            "code": ["000001"], "date": [pd.Timestamp("2026-07-01")],
            "target_ret_3d": [0.01], "base_factor": [1.0],
            "news_sent_local_decay_3d": [0.2],
        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                ablation.run_ablation(
                    panel, output, 3,
                    pd.Timestamp("2026-01-31"), pd.Timestamp("2026-03-31"),
                    pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-31"),
                    pd.Timestamp("2026-01-31"), pd.Timestamp("2026-04-18"),
                    pd.Timestamp("2026-04-19"), models=("ridge",),
                )


if __name__ == "__main__":
    unittest.main()
