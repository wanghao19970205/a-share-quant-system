from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from stock_analyzer import all_a_meta


class SwIndustryHistoryTest(unittest.TestCase):
    def test_build_history_uses_publication_date_as_pit_boundary(self):
        raw = pd.DataFrame({
            "symbol": ["600001", "600001", "000001"],
            "start_date": ["2020-01-01", "2024-02-01", "2021-07-30"],
            "industry_code": ["480101", "480301", "480101"],
            "update_time": ["2020-01-03", "2024-03-01", "2024-09-27"],
        })
        with mock.patch.object(all_a_meta.ak, "stock_industry_clf_hist_sw", return_value=raw):
            history = all_a_meta.build_sw_industry_history()

        first = history[(history["code"] == "600001") & (history["industry"] == "480101")].iloc[0]
        second = history[(history["code"] == "600001") & (history["industry"] == "480301")].iloc[0]
        retroactive = history[history["code"] == "000001"].iloc[0]
        self.assertEqual(first["valid_to"], pd.Timestamp("2024-02-01"))
        self.assertEqual(second["available_from"], pd.Timestamp("2024-03-01"))
        self.assertEqual(retroactive["available_from"], pd.Timestamp("2024-09-27"))
        self.assertEqual(first["source"], "akshare.stock_industry_clf_hist_sw")


if __name__ == "__main__":
    unittest.main()
