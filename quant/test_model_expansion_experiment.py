from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant import model_expansion_experiment as experiment


class ModelExpansionExperimentTest(unittest.TestCase):
    def test_neutralize_target_removes_industry_and_size_exposure(self):
        rng = np.random.default_rng(42)
        rows = []
        for date in pd.date_range("2026-01-01", periods=3):
            for index in range(80):
                industry = "A" if index < 40 else "B"
                size = rng.normal()
                target = (0.2 if industry == "A" else -0.2) + 0.3 * size + rng.normal(scale=0.01)
                rows.append({
                    "date": date,
                    "target": target,
                    "log_mv_total": size,
                    "industry": industry,
                })
        frame = pd.DataFrame(rows)

        residual = experiment.neutralize_target_cross_section(
            frame,
            "target",
            industry=frame["industry"],
            exposure_columns=("log_mv_total",),
        )

        self.assertLess(residual.groupby(frame["date"]).mean().abs().max(), 1e-10)
        self.assertLess(abs(residual.corr(frame["log_mv_total"])), 1e-10)
        industry_means = residual.groupby([frame["date"], frame["industry"]]).mean()
        self.assertLess(industry_means.abs().max(), 1e-10)

    def test_market_regime_requires_trailing_history(self):
        rng = np.random.default_rng(7)
        dates = pd.date_range("2025-01-01", periods=300)
        market = pd.DataFrame({
            "date": dates,
            "median_return": rng.normal(0, 0.01, len(dates)),
            "breadth": rng.uniform(0, 1, len(dates)),
        })

        result = experiment.build_market_regimes(
            market,
            trend_window=20,
            volatility_window=20,
            history_window=252,
        )

        self.assertTrue(result.iloc[:62]["regime"].eq("insufficient_history").all())
        self.assertTrue(result.iloc[-1]["regime"] != "insufficient_history")

    def test_regime_weight_selection_uses_supplied_returns(self):
        dates = pd.date_range("2026-01-01", periods=40)
        returns = pd.DataFrame({
            "date": list(dates) * 2,
            "regime": ["up"] * 80,
            "weight": [0.0] * 40 + [0.1] * 40,
            "ret": [0.0] * 40 + [0.01, 0.02] * 20,
        })

        selected = experiment.select_regime_weights(returns, [0.0, 0.1], min_observations=20)

        self.assertEqual(selected, {"up": 0.1})


if __name__ == "__main__":
    unittest.main()
