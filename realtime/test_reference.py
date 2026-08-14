import tempfile
import unittest
from pathlib import Path

from realtime.config import RealtimeConfig
from realtime.reference import (
    _build_calibration,
    _load_expected_return,
    assert_prediction_fresh,
    build,
)
from realtime.watchlist import load_codes


class PredictionFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.predictions = Path(self.tmp.name) / "missing_predictions.parquet"
        self.cfg = RealtimeConfig(predictions_file=self.predictions)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_prediction_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "prediction artifact is missing"):
            assert_prediction_fresh(self.cfg)

    def test_expected_return_loader_does_not_bypass_missing_gate(self):
        with self.assertRaisesRegex(RuntimeError, "prediction artifact is missing"):
            _load_expected_return(self.cfg)

    def test_calibration_loader_does_not_bypass_missing_gate(self):
        with self.assertRaisesRegex(RuntimeError, "prediction artifact is missing"):
            _build_calibration(self.cfg)

    def test_reference_build_does_not_allow_fallback_candidates(self):
        with self.assertRaisesRegex(RuntimeError, "prediction artifact is missing"):
            build(self.cfg, ["000001"])

    def test_watchlist_does_not_use_fallback_candidates_without_prediction(self):
        with self.assertRaisesRegex(RuntimeError, "prediction artifact is missing"):
            load_codes(self.cfg)


if __name__ == "__main__":
    unittest.main()
