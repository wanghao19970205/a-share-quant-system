"""实时层配置：订阅清单来源、推送凭证、交易时段、规避窗口、节流参数。

全部走环境变量，带安全默认值。绝不硬编码任何 Token/密钥。
本模块只读复用 quant.config 的路径常量，不改动生产配置。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from quant import config as _qconfig


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v not in {"0", "false", "no", "off"}


def _default_env_file() -> Path:
    """凭证 env 文件默认路径：与账本同在 logs/realtime（容器已挂载 + git 排除）。

    放在这里而非 /app 下，是为了不让密钥文件混进代码仓库（logs/ 已被 git 忽略）。
    """
    return Path(_qconfig.QUANT_DIR).parent / "logs" / "realtime" / "notify.env"


def _load_env_file() -> None:
    """把 REALTIME_ENV_FILE（默认 logs/realtime/notify.env）里的 KEY=VALUE 注入环境。

    只填「尚未设置」的变量（真实环境变量优先），因此不会覆盖 docker -e 传入的值。
    文件不存在则静默跳过。绝不打印文件内容/密钥。
    """
    path = Path(os.environ.get("REALTIME_ENV_FILE", "") or _default_env_file())
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as e:  # noqa: BLE001 - env 文件问题不拦启动
        print(f"[config] 读取 env 文件失败(忽略)：{type(e).__name__}", flush=True)


@dataclass
class RealtimeConfig:
    """一份实时层运行配置。字段全部可被环境变量覆盖。"""

    # ---- 订阅清单来源 --------------------------------------------------------
    # 选股清单：最新一期短线预测（rt_probe 已验证可读）。
    predictions_file: Path = field(
        default_factory=lambda: Path(_qconfig.QUANT_DIR)
        / "active_quant_short_predictions.parquet")
    # 持仓清单（可选）：一行一个 6 位代码，可选跟买入日期（"代码 YYYY-MM-DD" /
    # "代码,YYYYMMDD"，支持行内 # 注释）；存在则并入订阅，买入日期供 T+N 到期卖出判定。
    holdings_file: Path = field(
        default_factory=lambda: Path(
            os.environ.get("REALTIME_HOLDINGS_FILE",
                           str(Path(_qconfig.QUANT_DIR) / "realtime_holdings.txt"))))
    # 兜底股票池文件（预测清单缺失时用）。
    universe_file: Path = field(
        default_factory=lambda: Path(_qconfig.MAINBOARD_UNIVERSE_FILE))
    # 订阅上限（受券商 SubscribeLimitNum 约束）。默认 100 保底：验收套餐订阅数
    # 下限即 100，且防止清单退回全市场兜底池(数千只)时直接打爆券商上限。
    # 下周一盘中实测拿到真实 SubscribeLimitNum 后，用 REALTIME_MAX_SUBSCRIBE 调。
    max_subscribe: int = field(default_factory=lambda: _env_int("REALTIME_MAX_SUBSCRIBE", 100))

    # ---- 账本 ----------------------------------------------------------------
    # 独立账本目录：默认 logs/realtime（容器已挂载，且与 quant_data 业务数据隔离）。
    ledger_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("REALTIME_LEDGER_DIR",
                           str(Path(_qconfig.QUANT_DIR).parent / "logs" / "realtime"))))

    # ---- 推送（Bark / Server酱 / PushDeer）---------------------------------
    # Bark（iOS 首选，免费不限量）：BARK_KEY 为设备 key；endpoint 默认官方 api.day.app。
    bark_key: str = field(default_factory=lambda: os.environ.get("BARK_KEY", "").strip())
    bark_endpoint: str = field(
        default_factory=lambda: os.environ.get("BARK_ENDPOINT", "https://api.day.app").strip())
    # SCKEY / PUSHDEER_KEY 从环境变量读；缺失则只记账不推送（干跑）。
    serverchan_key: str = field(default_factory=lambda: os.environ.get("SERVERCHAN_SCKEY", "").strip())
    pushdeer_key: str = field(default_factory=lambda: os.environ.get("PUSHDEER_KEY", "").strip())
    pushdeer_endpoint: str = field(
        default_factory=lambda: os.environ.get("PUSHDEER_ENDPOINT", "https://api2.pushdeer.com").strip())
    # 同一 (code, 信号类型) 的推送最小间隔秒数，防刷屏。
    notify_cooldown_sec: int = field(default_factory=lambda: _env_int("REALTIME_NOTIFY_COOLDOWN", 300))

    # ---- 卖点纪律 ------------------------------------------------------------
    # 持有期到期卖出的目标 horizon（交易日）。Phase1 holdout 对比表定档 T+1
    # （年化264%/Sharpe12/回撤-4.2% 三项全优）。对比表结论更新后改此值即可，不动代码。
    sell_horizon: int = field(default_factory=lambda: _env_int("REALTIME_SELL_HORIZON", 1))

    # ---- 交易时段 / 生命周期（本地时间，HHMM 整数）--------------------------
    session_start: int = field(default_factory=lambda: _env_int("REALTIME_SESSION_START", 925))
    session_end: int = field(default_factory=lambda: _env_int("REALTIME_SESSION_END", 1505))
    morning_close: int = field(default_factory=lambda: _env_int("REALTIME_MORNING_CLOSE", 1132))
    afternoon_open: int = field(default_factory=lambda: _env_int("REALTIME_AFTERNOON_OPEN", 1300))
    # daily-light 规避窗（11:40 启动）：这段时间暂停订阅，避免同机抢券商连接。
    avoid_start: int = field(default_factory=lambda: _env_int("REALTIME_AVOID_START", 0))
    avoid_end: int = field(default_factory=lambda: _env_int("REALTIME_AVOID_END", 0))

    # ---- 运行参数 ------------------------------------------------------------
    # 干跑：不真正登录/订阅，用于本地校验骨架（喂假快照）。
    dry_run: bool = field(default_factory=lambda: _env_bool("REALTIME_DRY_RUN", False))
    # 主循环心跳打印间隔秒。
    heartbeat_sec: int = field(default_factory=lambda: _env_int("REALTIME_HEARTBEAT", 60))

    def ensure_dirs(self) -> None:
        self.ledger_dir.mkdir(parents=True, exist_ok=True)


def load() -> RealtimeConfig:
    """构造一份运行配置（读环境变量）。

    先尝试从 REALTIME_ENV_FILE（默认 logs/realtime/notify.env）补齐未设置的环境变量，
    这样 BARK_KEY 等推送凭证可持久化在挂载盘上，无需每次 docker exec -e 传入。
    真实环境变量优先，不会被文件覆盖。
    """
    _load_env_file()
    return RealtimeConfig()
