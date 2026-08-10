from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from intraday_1400 import daily_h1_executable_payoff as payoff
from intraday_1400.target_redesign_backfill import FOLD_POSITIONS


class ExecutablePayoffTests(unittest.TestCase):
    def test_executable_value_and_fixed_blends_use_predictions_only(self):
        date = pd.Timestamp("2026-08-04")
        daily = pd.DataFrame({"code": ["600001", "600002", "600003"], "date": date, "score": [3.0, 2.0, 1.0]})
        buy = pd.DataFrame({"code": daily.code, "date": date, "pred": [.5, .8, .2]})
        liq = pd.DataFrame({"code": daily.code, "date": date, "pred": [.5, .25, .9]})
        ret = pd.DataFrame({"code": daily.code, "date": date, "pred": [.2, -.2, .03]})
        first = payoff.build_payoff_scores(daily, buy, liq, ret)
        expected = np.array([.5 * (.5 * .10 + .5 * -.10), .8 * (.25 * -.10 + .75 * -.10), .2 * (.9 * .03 + .1 * -.10)])
        # Recover the payoff by repeating the public formula inputs and check clipping explicitly.
        raw = buy.pred.to_numpy() * (liq.pred.to_numpy() * np.clip(ret.pred.to_numpy(), -.10, .10) + (1 - liq.pred.to_numpy()) * -.10)
        self.assertTrue(np.allclose(raw, expected))
        self.assertEqual(
            first["h1_top50_payoff_rerank"]["code"].tolist(),
            ["600003", "600001", "600002"],
        )
        self.assertEqual(
            first["payoff_top50_buy_rerank"]["code"].tolist(),
            ["600002", "600001", "600003"],
        )
        changed = daily.assign(realized_outcome=[999, -999, 999])
        second = payoff.build_payoff_scores(changed, buy, liq, ret)
        for name in payoff.CANDIDATES:
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_selection_requires_four_folds_and_three_wins(self):
        def metrics(value, filled=10.0):
            return {"mean_return": value, "compound_return": value * 10, "max_drawdown": -.10, "mean_filled_names": filled}
        aggregate = {name: metrics(0.0) for name in payoff.CANDIDATES}
        folds = []
        for index, fold in enumerate(FOLD_POSITIONS):
            models = {name: metrics(0.0) for name in payoff.CANDIDATES}
            models["h1_top50_payoff_rerank"] = metrics(.01 if index < 3 else -.01)
            folds.append({"name": fold["name"], "models": models})
        decision = payoff.select_enhancement(aggregate, folds)
        self.assertEqual(decision["status"], "no_payoff_enhancement_passed")
        self.assertEqual(decision["next_branch"], "daily_baseline_retained")
        aggregate["h1_top50_payoff_rerank"] = metrics(.01)
        self.assertEqual(
            payoff.select_enhancement(aggregate, folds)["selected"],
            "h1_top50_payoff_rerank",
        )

    def test_recipe_is_isolated_and_has_no_calibration(self):
        self.assertEqual(payoff.MODEL_RECIPE["buyability_head"]["n_estimators"], 160)
        self.assertEqual(payoff.MODEL_RECIPE["liquidation_head"]["minority_weight"], 1.0)
        self.assertEqual(payoff.MODEL_RECIPE["return_head"]["n_estimators"], 200)
        self.assertEqual(payoff.MODEL_RECIPE["return_head"]["learning_rate"], .015)
        self.assertEqual(payoff.MODEL_RECIPE["return_head"]["max_train_rows"], 400000)
        self.assertFalse(payoff.MODEL_RECIPE["return_head"]["early_stopping"])
        self.assertFalse(payoff.protocol_payload()["production_publication"])


if __name__ == "__main__":
    unittest.main()
