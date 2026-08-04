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


# 推送信号白名单（kind）默认集合（Plan A）：只推离散、可行动的事件——
# 封板/炸板（关注）、急速拉升跳水（关注）、吊灯止损 + 持有到期（卖出纪律）。
# 刻意挡掉「状态徘徊」噪音：vwap_*（每票每5分钟反复报偏贵/偏宜）、near_limit_*
# （14只票反复逼近涨跌停刷屏）、volume_surge（开盘首笔当基线致集体误报，修好基线前不推）。
_DEFAULT_NOTIFY_KINDS = (
    "limit_up,limit_down,limit_open,chandelier_stop,holding_expiry,surge_up,surge_down"
)


def _parse_notify_kinds() -> frozenset:
    """解析 REALTIME_NOTIFY_KINDS（逗号分隔）为推送白名单。

    缺省用 Plan A 白名单；设为 "all"/"*"/空 则返回空集 = 不过滤（全推，慎用会刷屏）。
    低于白名单的信号仍照常记账（含金量复盘靠账本，不靠推送）。
    """
    raw = os.environ.get("REALTIME_NOTIFY_KINDS", _DEFAULT_NOTIFY_KINDS).strip()
    if not raw or raw.lower() in {"all", "*"}:
        return frozenset()
    return frozenset(k.strip() for k in raw.replace("，", ",").split(",") if k.strip())


def _realtime_dir() -> Path:
    """实时态持久目录：项目根 /logs/realtime。

    必须基于 _qconfig._ROOT（= 项目根，容器内 /app）而非 QUANT_DIR.parent：
    容器里 QUANT_DATA_DIR 指到子目录 /app/quant_data/full_a_2018_wide，其 .parent
    是 /app/quant_data（非挂载层，rebuild 即失），只有 _ROOT/logs（= /app/logs，
    docker 挂载 /www/A/logs）才是持久盘。notify.env / paper_state / 账本全落这里，
    才能扛住 rebuild。本地无 QUANT_DATA_DIR 时 _ROOT 即项目根，行为不变。
    """
    return Path(_qconfig._ROOT) / "logs" / "realtime"


def _default_env_file() -> Path:
    """凭证 env 文件默认路径：与账本同在 logs/realtime（容器挂载盘 + git 排除）。

    放在这里而非代码目录，是为了不让密钥文件混进代码仓库（logs/ 已被 git 忽略）。
    """
    return _realtime_dir() / "notify.env"


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
    # 主路径：手机 UI 的 3 个 Top10 名单（白名单/全A/创新药），即 top10_eval 落盘的
    # mobile_snapshot.json。每次 top10-eval 跑完重写；引擎盘中检测其变化即自动重载订阅。
    # 默认取 SNAPSHOT_DIR（容器 /app/snapshots 已挂载持久），可用 REALTIME_MOBILE_SNAPSHOT 覆盖。
    mobile_snapshot_file: Path = field(
        default_factory=lambda: Path(
            os.environ.get("REALTIME_MOBILE_SNAPSHOT", "")
            or (Path(os.environ.get("SNAPSHOT_DIR", "snapshots")) / "mobile_snapshot.json")))
    # 兜底：最新一期短线预测（Top10 快照缺失时退回按模型分取 top，保旧行为不空订阅）。
    predictions_file: Path = field(
        default_factory=lambda: Path(_qconfig.QUANT_DIR)
        / "active_quant_short_predictions.parquet")
    # 持仓清单（可选）：一行一个 6 位代码，可选跟买入日期（"代码 YYYY-MM-DD" /
    # "代码,YYYYMMDD"，支持行内 # 注释）；存在则并入订阅，买入日期供 T+N 到期卖出判定。
    # 默认落挂载盘 logs/realtime（与 notify.env/账本同处），宿主机可直接编辑、无需进容器。
    holdings_file: Path = field(
        default_factory=lambda: Path(
            os.environ.get("REALTIME_HOLDINGS_FILE",
                           str(_realtime_dir() / "realtime_holdings.txt"))))
    # 兜底股票池文件（预测清单也缺失时用）。
    universe_file: Path = field(
        default_factory=lambda: Path(_qconfig.MAINBOARD_UNIVERSE_FILE))
    # 本地订阅安全上限。账号不限制代码数，但总带宽上限为 2 MB；保留该阈值防名单异常
    # 退回全市场时打满带宽。行业 ETF 在股票清单截断后另行加入同一 Push 会话。
    max_subscribe: int = field(default_factory=lambda: _env_int("REALTIME_MAX_SUBSCRIBE", 100))
    # 盘中订阅自动重载：检测到 mobile_snapshot.json / 持仓文件变化即 execv 重启换新名单。
    watchlist_reload: bool = field(default_factory=lambda: _env_bool("REALTIME_WATCHLIST_RELOAD", True))

    # ---- 实时买入候选榜（RankBoard）----------------------------------------
    # 跨票聚合器：模型看多且校准净收益覆盖成本后取 Top-N，配盘中量拼 digest 推送。
    # 排序仍由模型 alpha 决定，盘中信号只做有界纠偏；仅榜单指纹变化才推。
    rank_board_enabled: bool = field(default_factory=lambda: _env_bool("REALTIME_RANK_BOARD", True))
    rank_interval_sec: int = field(default_factory=lambda: _env_int("REALTIME_RANK_INTERVAL", 300))
    rank_top_n: int = field(default_factory=lambda: _env_int("REALTIME_RANK_TOP_N", 5))

    # ---- 盘中动态重排（RerankScorer，RankBoard + PaperTrader 共用）----------
    # 候选池 = 模型看多且校准净收益达标的 Top-rank_pool_n；绝不拉池外票。
    # score = exp*(1+clamp(adj,±rerank_cap))：模型分为主序，盘中信号只在 ±cap 内微调名次。
    # VWAP位置/买卖失衡/盘口价差全 env 可调；旧高开惩罚保留开关但默认关闭。
    rerank_enabled: bool = field(default_factory=lambda: _env_bool("REALTIME_RERANK", True))
    rank_pool_n: int = field(default_factory=lambda: _env_int("REALTIME_RANK_POOL_N", 30))
    rerank_cap: float = field(default_factory=lambda: _env_float("REALTIME_RERANK_CAP", 0.30))
    rerank_w_vwap: float = field(default_factory=lambda: _env_float("REALTIME_RERANK_W_VWAP", 0.35))
    rerank_w_imb: float = field(default_factory=lambda: _env_float("REALTIME_RERANK_W_IMB", 0.30))
    # 离线 IC 已确认高开是正向隔夜动量，旧的“高开吃预期”惩罚方向错误，默认关闭。
    rerank_w_gap: float = field(default_factory=lambda: _env_float("REALTIME_RERANK_W_GAP", 0.0))
    rerank_w_spread: float = field(default_factory=lambda: _env_float("REALTIME_RERANK_W_SPREAD", 0.15))
    # 历史分档校准后扣 round-trip 成本的最低净收益；默认至少覆盖交易成本。
    rank_min_net_return: float = field(
        default_factory=lambda: _env_float("REALTIME_RANK_MIN_NET_RETURN", 0.0))

    # ---- 实时模拟盘（PaperTrader）------------------------------------------
    # T+N 到期腿在收盘前执行，随后从净收益达标候选中按模型分买 Top-N，保持 close→close 口径。
    # 风险退出（止损/止盈/移动止盈/VWAP破位）不受 time_cap_start 限制，T+1 全天有效。
    paper_trade_enabled: bool = field(default_factory=lambda: _env_bool("REALTIME_PAPER_TRADE", True))
    paper_buy_n: int = field(default_factory=lambda: _env_int("REALTIME_PAPER_BUY_N", 2))
    paper_buy_start: int = field(default_factory=lambda: _env_int("REALTIME_PAPER_BUY_START", 1455))
    paper_time_cap_start: int = field(
        default_factory=lambda: _env_int("REALTIME_PAPER_TIME_CAP_START", 1450))
    paper_start_equity: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_START_EQUITY", 100000.0))
    paper_cost: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_COST", 0.002))
    paper_state_file: Path = field(
        default_factory=lambda: Path(
            os.environ.get("REALTIME_PAPER_STATE_FILE",
                           str(_realtime_dir() / "paper_state.json"))))

    # 出场策略（先触发先走）：硬止损 / 止盈上限 / 移动止盈(ATR吊灯) / 破位(跌破VWAP) / T+N时间上限。
    # 卖出腿全交易时段每轮评估；买入腿仍只在收盘前窗口。全 env 可调，实盘复盘后按流水再调。
    paper_stop_loss: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_STOP_LOSS", 0.05))
    paper_take_profit: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_TAKE_PROFIT", 0.09))
    paper_trail_k: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_TRAIL_K", 3.0))
    paper_vwap_break: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_VWAP_BREAK", 0.02))
    # 买入择时过滤（方向2）：买入腿按重排后 score 降序取 Top-N，买前逐票查以下项，命中即跳过（顺延下一名）。
    # 全 env 可关（设 0 即不过滤该项）；全部候选被过滤则当日不建仓。
    # 与重排一致：高开是已验证的正向隔夜动量，不再按“吃掉预期”阻断入场。
    paper_entry_gap_eaten: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_ENTRY_GAP_EATEN", 0.0))
    paper_entry_rich: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_ENTRY_RICH", 0.01))
    paper_entry_ask_strong: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_ENTRY_ASK_STRONG", 0.2))
    # 盘口价差流动性门槛：买一/卖一价差率超阈值即跳过（宽价差=薄盘口，14:50 建仓滑点吃 alpha）。
    # 默认 0.006(0.6%)；设 0 关闭。缺盘口价(spread_pct=None)不拦，不误跳。
    paper_entry_spread: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_ENTRY_SPREAD", 0.006))

    # ---- V2 模拟盘（赛马对照）------------------------------------------------
    # V2 与 V1 在同一引擎内并行，读取独立状态文件（_v2 后缀），共享 ctx 与 notifier。
    # V1 现役策略完全不动；V2 在这里增加保护性止盈、动态分配、持仓上限、买窗收窄和跌停阻塞。
    paper_v2_enabled: bool = field(default_factory=lambda: _env_bool("REALTIME_PAPER_V2", True))
    paper_buy_end: int = field(default_factory=lambda: _env_int("REALTIME_PAPER_BUY_END", 1457))
    paper_max_positions: int = field(default_factory=lambda: _env_int("REALTIME_PAPER_MAX_POSITIONS", 4))
    paper_breakeven_arm: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_BREAKEVEN_ARM", 0.03))
    paper_breakeven_margin: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_BREAKEVEN_MARGIN", 0.005))
    paper_take_profit_tighten: float = field(default_factory=lambda: _env_float("REALTIME_PAPER_TAKE_PROFIT_TIGHTEN", 0.03))
    paper_limit_down_roll_max: int = field(default_factory=lambda: _env_int("REALTIME_PAPER_LIMIT_DOWN_ROLL_MAX", 3))

    # ---- V3 模拟盘（执行确认 + ATR 自适应出场）-------------------------------
    # 与 V2 共用模型池、资金分配、买窗和持仓上限；按 ask1/bid1 模拟可成交报价，
    # 只使用当日预测和新鲜快照，卖出统一采用 ATR 风险单位。
    paper_v3_enabled: bool = field(default_factory=lambda: _env_bool("REALTIME_PAPER_V3", True))
    paper_v3_quote_max_age_sec: float = field(
        default_factory=lambda: max(
            1.0, _env_float("REALTIME_PAPER_V3_QUOTE_MAX_AGE_SEC", 90.0)))
    paper_v3_atr_k: float = field(
        default_factory=lambda: max(0.1, _env_float("REALTIME_PAPER_V3_ATR_K", 2.0)))

    # ---- V4 模拟盘（V3 + 行业 ETF 弱势回避）---------------------------------
    # ETF 与个股复用同一 AmazingData Push 会话；ETF 快照只进入板块上下文，不进入个股策略。
    # V4 第一版只过滤显著弱于沪深300 ETF 的行业，不改 V3 卖点/仓位，保持差异可归因。
    paper_v4_enabled: bool = field(default_factory=lambda: _env_bool("REALTIME_PAPER_V4", True))
    sector_etf_enabled: bool = field(default_factory=lambda: _env_bool("REALTIME_SECTOR_ETF", True))
    sector_etf_benchmark: str = field(
        default_factory=lambda: os.environ.get("REALTIME_SECTOR_ETF_BENCHMARK", "510300.SH").strip().upper())
    sector_etf_specs: str = field(
        default_factory=lambda: os.environ.get("REALTIME_SECTOR_ETFS", "").strip())
    sector_etf_quote_max_age_sec: float = field(
        default_factory=lambda: max(
            1.0, _env_float("REALTIME_SECTOR_ETF_QUOTE_MAX_AGE_SEC", 90.0)))
    paper_v4_sector_weak_excess: float = field(
        default_factory=lambda: _env_float("REALTIME_PAPER_V4_SECTOR_WEAK_EXCESS", -0.003))
    paper_v4_sector_strong_excess: float = field(
        default_factory=lambda: _env_float("REALTIME_PAPER_V4_SECTOR_STRONG_EXCESS", 0.003))
    paper_v4_sector_mapping_min_confidence: float = field(
        default_factory=lambda: min(1.0, max(
            0.0, _env_float("REALTIME_PAPER_V4_SECTOR_MAPPING_MIN_CONFIDENCE", 0.8))))
    sector_meta_file: Path = field(
        default_factory=lambda: Path(os.environ.get(
            "REALTIME_SECTOR_META_FILE", "") or
            (Path(os.environ.get("SNAPSHOT_DIR", "snapshots")) / "all_a_stock_meta.parquet")))

    # ---- 预期收益历史校准（Top-N 展示重标定）------------------------------
    # ridge_pred 强正则收缩偏保守，按历史同档实际兑现(target_ret_{h}d)重标定展示值 + 胜率。
    # 只改展示、不改排序主序（排序仍按原始 ridge_pred）。样本不足/无 target 列则降级回退原始值。
    calib_enabled: bool = field(default_factory=lambda: _env_bool("REALTIME_CALIB", True))
    calib_bins: int = field(default_factory=lambda: _env_int("REALTIME_CALIB_BINS", 20))

    # ---- 账本 ----------------------------------------------------------------
    ledger_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("REALTIME_LEDGER_DIR", str(_realtime_dir()))))

    # ---- 推送（Bark / Server酱 / PushDeer）---------------------------------
    bark_key: str = field(default_factory=lambda: os.environ.get("BARK_KEY", "").strip())
    bark_endpoint: str = field(
        default_factory=lambda: os.environ.get("BARK_ENDPOINT", "https://api.day.app").strip())
    serverchan_key: str = field(default_factory=lambda: os.environ.get("SERVERCHAN_SCKEY", "").strip())
    pushdeer_key: str = field(default_factory=lambda: os.environ.get("PUSHDEER_KEY", "").strip())
    pushdeer_endpoint: str = field(
        default_factory=lambda: os.environ.get("PUSHDEER_ENDPOINT", "https://api2.pushdeer.com").strip())
    notify_cooldown_sec: int = field(default_factory=lambda: _env_int("REALTIME_NOTIFY_COOLDOWN", 300))
    notify_kinds: frozenset = field(default_factory=_parse_notify_kinds)

    # ---- 卖点纪律 ------------------------------------------------------------
    sell_horizon: int = field(default_factory=lambda: _env_int("REALTIME_SELL_HORIZON", 1))

    # ---- 交易时段 / 生命周期（本地时间，HHMM 整数）--------------------------
    session_start: int = field(default_factory=lambda: _env_int("REALTIME_SESSION_START", 925))
    session_end: int = field(default_factory=lambda: _env_int("REALTIME_SESSION_END", 1505))
    morning_close: int = field(default_factory=lambda: _env_int("REALTIME_MORNING_CLOSE", 1132))
    afternoon_open: int = field(default_factory=lambda: _env_int("REALTIME_AFTERNOON_OPEN", 1300))
    avoid_start: int = field(default_factory=lambda: _env_int("REALTIME_AVOID_START", 0))
    avoid_end: int = field(default_factory=lambda: _env_int("REALTIME_AVOID_END", 0))

    # ---- 运行参数 ------------------------------------------------------------
    dry_run: bool = field(default_factory=lambda: _env_bool("REALTIME_DRY_RUN", False))
    heartbeat_sec: int = field(default_factory=lambda: _env_int("REALTIME_HEARTBEAT", 60))
    # 策略信号先进入有界队列，再由单线程顺序执行通知与记账，避免网络 I/O 阻塞行情回调。
    effect_queue_size: int = field(
        default_factory=lambda: max(1, _env_int("REALTIME_EFFECT_QUEUE_SIZE", 1000)))
    effect_shutdown_grace_sec: float = field(
        default_factory=lambda: max(
            0.0, _env_float("REALTIME_EFFECT_SHUTDOWN_GRACE_SEC", 3.0)))

    def ensure_dirs(self) -> None:
        """确保账本目录存在（推送/状态/审计均落在此目录）。"""
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

    def paper_state_files(self) -> tuple[Path, ...]:
        """全部模拟盘账户状态文件；供订阅保护与热重载统一取用。"""
        base = Path(self.paper_state_file)
        files = [base]
        if getattr(self, "paper_v2_enabled", False):
            files.append(base.parent / f"{base.stem}_v2{base.suffix}")
        if getattr(self, "paper_v3_enabled", False):
            files.append(base.parent / f"{base.stem}_v3{base.suffix}")
        if getattr(self, "paper_v4_enabled", False):
            files.append(base.parent / f"{base.stem}_v4{base.suffix}")
        return tuple(files)


def load() -> RealtimeConfig:
    """构造一份运行配置（读环境变量）。

    先尝试从 REALTIME_ENV_FILE（默认 logs/realtime/notify.env）补齐未设置的环境变量，
    这样 BARK_KEY 等推送凭证可持久化在挂载盘上，无需每次 docker exec -e 传入。
    真实环境变量优先，不会被文件覆盖。
    """
    _load_env_file()
    return RealtimeConfig()
