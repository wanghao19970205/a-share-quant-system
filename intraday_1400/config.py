from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT = Path(os.environ.get("INTRADAY_1400_DATA_DIR", "intraday_1400_data")).resolve()
ASOF_PRICE_DIR = DATA_ROOT / "asof_price"
FEATURE_DIR = DATA_ROOT / "features"
PREPARED_DIR = DATA_ROOT / "prepared_monthly"
LABEL_DIR = DATA_ROOT / "labels"
MODEL_DIR = DATA_ROOT / "models"
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
REPORT_DIR = DATA_ROOT / "reports"
LOCK_PATH = Path(os.environ.get("INTRADAY_1400_LOCK", "/tmp/stock-intraday-1400.lock"))
SCHEMA_VERSION = 2
FEATURE_RECIPE_VERSION = 8
PREPARE_RECIPE_VERSION = 8
LABEL_RECIPE_VERSION = 10
TRAIN_RECIPE_VERSION = 8
CUTOFF_TIME = os.environ.get("INTRADAY_1400_CUTOFF", "13:55")
MINUTE_BATCH_SIZE = max(int(os.environ.get("INTRADAY_1400_MINUTE_BATCH_SIZE", "50") or 50), 1)
PARTITION_SIZE = max(int(os.environ.get("INTRADAY_1400_PARTITION_SIZE", "50") or 50), 1)
MINUTE_TIMEOUT = float(os.environ.get("INTRADAY_1400_MINUTE_TIMEOUT", "180") or 180)
FEATURE_WORKERS = max(int(os.environ.get("INTRADAY_1400_FEATURE_WORKERS", "4") or 4), 1)


def ensure_dirs() -> None:
    for path in (
        DATA_ROOT,
        ASOF_PRICE_DIR,
        FEATURE_DIR,
        PREPARED_DIR,
        LABEL_DIR,
        MODEL_DIR,
        CHECKPOINT_DIR,
        REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
