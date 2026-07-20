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


if __name__ == "__main__":
    unittest.main()
