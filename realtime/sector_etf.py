"""A股行业 ETF 实时信号：用 ETF 相对沪深300 ETF 的表现确认个股入场环境。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .snapshot import Snapshot
from .strategy import Signal, _digits


_MAPPING_VERSION = "industry_etf_exact_v2"

# 格式：展示名=ETF完整代码:主行业1|主行业2。这里只列能够被对应 ETF 较可靠代表的
# 申万三级主行业；宁可 unmapped 放行，也不使用概念或宽关键词误杀候选。
_DEFAULT_SPECS = (
    "半导体=512480.SH:数字芯片设计|模拟芯片设计|集成电路制造|集成电路封测|"
    "半导体材料|半导体设备|分立器件;"
    "证券=512880.SH:证券Ⅱ;"
    "银行=512800.SH:农商行Ⅲ|国有大型银行Ⅲ|城商行Ⅲ|股份制银行Ⅲ;"
    "医药=512010.SH:中药Ⅱ|体外诊断|其他医疗服务|其他生物制品|化学制剂|"
    "医疗研发外包|医疗耗材|医疗设备|医药流通|医院|原料药|疫苗|线下药店|"
    "血液制品|诊断服务;"
    "军工=512660.SH:军工电子Ⅱ|地面兵装Ⅱ|航天装备Ⅱ|航海装备Ⅱ|航空装备Ⅱ;"
    "新能源车=515030.SH:电池化学品|燃料电池|蓄电池及其他电池|锂电专用设备|锂电池;"
    "光伏=515790.SH:光伏主材|光伏加工设备|光伏电池组件|光伏辅材|硅料硅片|逆变器;"
    "有色=512400.SH:其他小金属|其他金属新材料|白银|磁性材料|稀土|钨|钴|钼|"
    "铅锌|铜|铝|锂|镍|黄金;"
    "食品饮料=159928.SZ:乳品|保健品|其他酒类|啤酒|烘焙食品|熟食|白酒Ⅱ|肉制品|"
    "调味发酵品Ⅱ|软饮料|零食|预加工食品;"
    "计算机=512720.SH:IT服务Ⅱ|其他计算机设备|垂直应用软件|安防设备|横向通用软件;"
    "通信=515880.SH:其他通信设备|电信运营商|通信工程及服务|通信应用增值服务|"
    "通信线缆及配套|通信终端及配件|通信网络设备及器件"
)


def _broker_code(code: str) -> str:
    return str(code or "").strip().upper()


@dataclass(frozen=True)
class SectorETFSpec:
    name: str
    code: str
    primary_industries: tuple[str, ...]


def parse_specs(raw: str = "") -> tuple[SectorETFSpec, ...]:
    """解析 ``名称=完整ETF代码:精确主行业1|精确主行业2`` 配置。"""
    text = (raw or _DEFAULT_SPECS).replace("；", ";")
    out: list[SectorETFSpec] = []
    seen_codes: set[str] = set()
    seen_industries: set[str] = set()
    for item in text.split(";"):
        identity, colon, industry_text = item.strip().partition(":")
        name, equal, raw_code = identity.partition("=")
        industries = tuple(
            x.strip() for x in industry_text.split("|") if x.strip())
        code = _broker_code(raw_code)
        if (not colon or not equal or not name.strip() or not industries or
                not code or "." not in code or code in seen_codes):
            continue
        # 同一个主行业只能归属一个 ETF；配置冲突时整条跳过，防顺序决定归属。
        if any(industry in seen_industries for industry in industries):
            continue
        seen_codes.add(code)
        seen_industries.update(industries)
        out.append(SectorETFSpec(name.strip(), code, industries))
    return tuple(out)


class SectorETFContext:
    """保存 ETF 快照并把股票行业映射为可审计的相对强弱信号。"""

    def __init__(self, cfg):
        self._enabled = bool(getattr(cfg, "sector_etf_enabled", True))
        self._benchmark = _broker_code(
            getattr(cfg, "sector_etf_benchmark", "510300.SH"))
        self._specs = parse_specs(getattr(cfg, "sector_etf_specs", ""))
        self._by_etf = {s.code: s for s in self._specs}
        self._by_primary_industry = {
            industry: spec
            for spec in self._specs
            for industry in spec.primary_industries
        }
        self._codes = frozenset({self._benchmark, *self._by_etf}) if self._enabled else frozenset()
        self._max_age = max(1.0, float(getattr(cfg, "sector_etf_quote_max_age_sec", 90.0)))
        self._weak = float(getattr(cfg, "paper_v4_sector_weak_excess", -0.003))
        self._strong = float(getattr(cfg, "paper_v4_sector_strong_excess", 0.003))
        self._mapping_min_confidence = min(1.0, max(0.0, float(
            getattr(cfg, "paper_v4_sector_mapping_min_confidence", 0.8))))
        self._snaps: dict[str, Snapshot] = {}
        self._recv_ts: dict[str, float] = {}
        self._last_status: dict[str, str] = {}
        self._stock_meta: dict[str, dict] = {}
        self._stock_map = self._load_stock_map(
            Path(getattr(cfg, "sector_meta_file", "snapshots/all_a_stock_meta.parquet")))

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def benchmark(self) -> str:
        return self._benchmark

    @property
    def weak_threshold(self) -> float:
        return self._weak

    @property
    def strong_threshold(self) -> float:
        return self._strong

    @property
    def mapping_version(self) -> str:
        return _MAPPING_VERSION

    @property
    def mapping_min_confidence(self) -> float:
        return self._mapping_min_confidence

    @property
    def mapped_stock_count(self) -> int:
        return len(self._stock_map)

    def mapped_count(self, codes) -> int:
        return sum(1 for code in codes if _digits(code) in self._stock_map)

    def subscription_codes(self) -> list[str]:
        return sorted(self._codes)

    def _resolve_code(self, code: str) -> str:
        value = _broker_code(code)
        if value in self._codes:
            return value
        digits = _digits(value)
        return next((c for c in self._codes if _digits(c) == digits), value)

    def is_sector_code(self, code: str) -> bool:
        return self._resolve_code(code) in self._codes

    @staticmethod
    def _clean_meta(value) -> str:
        text = "" if value is None else str(value).strip()
        return "" if text.lower() in {"nan", "<na>"} else text

    def _load_stock_map(self, path: Path) -> dict[str, SectorETFSpec]:
        if not self._enabled or not path.exists():
            return {}
        try:
            import pandas as pd
            columns = ["code", "a_industry", "a_industries"]
            try:
                df = pd.read_parquet(path, columns=columns)
            except Exception:
                df = pd.read_parquet(path)
            if "code" not in df.columns or "a_industry" not in df.columns:
                return {}
        except Exception as e:  # noqa: BLE001 - 元数据缺失只降级为无板块过滤
            print(f"[sector_etf] 行业映射加载失败(降级)：{type(e).__name__}", flush=True)
            return {}
        out: dict[str, SectorETFSpec] = {}
        for _, row in df.iterrows():
            code = _digits(row.get("code"))
            if not code:
                continue
            primary = self._clean_meta(row.get("a_industry"))
            hierarchy = self._clean_meta(row.get("a_industries"))
            self._stock_meta[code] = {
                "stock_industry": primary,
                "stock_industries": hierarchy,
            }
            spec = self._by_primary_industry.get(primary)
            if spec is not None:
                out[code] = spec
        return out

    def _age(self, code: str, now: Optional[float] = None) -> Optional[float]:
        ts = self._recv_ts.get(code)
        if ts is None:
            return None
        return max(0.0, (time.time() if now is None else now) - ts)

    @staticmethod
    def _ret(snap: Optional[Snapshot]) -> Optional[float]:
        if snap is None or snap.last is None or not snap.pre_close:
            return None
        try:
            return float(snap.last) / float(snap.pre_close) - 1.0
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _classify(self, excess: float) -> str:
        if excess <= self._weak:
            return "weak"
        if excess >= self._strong:
            return "strong"
        return "neutral"

    def update(self, snap: Snapshot, now: Optional[float] = None) -> list[Signal]:
        code = self._resolve_code(snap.code)
        if code not in self._codes:
            return []
        ts = time.time() if now is None else now
        snap.code = code
        self._snaps[code] = snap
        self._recv_ts[code] = ts
        if code == self._benchmark:
            return []
        assessment = self.assessment_for_etf(code, now=ts)
        status = assessment["status"]
        if status == "unavailable" or self._last_status.get(code) == status:
            return []
        self._last_status[code] = status
        spec = self._by_etf[code]
        excess = assessment["excess_return"]
        return [Signal(
            code=code,
            kind=f"sector_etf_{status}",
            level="warn" if status == "weak" else "info",
            reason=f"{spec.name}ETF相对沪深300 {excess:+.2%}，状态={status}",
            metrics=assessment,
            ts=ts,
        )]

    def assessment_for_etf(self, etf_code: str, now: Optional[float] = None) -> dict:
        code = _broker_code(etf_code)
        spec = self._by_etf.get(code)
        result = {
            "status": "unavailable",
            "sector": spec.name if spec else None,
            "etf_code": code if spec else None,
            "benchmark_code": self._benchmark,
            "etf_return": None,
            "benchmark_return": None,
            "excess_return": None,
            "etf_quote_age_sec": self._age(code, now),
            "benchmark_quote_age_sec": self._age(self._benchmark, now),
        }
        if not self._enabled or spec is None:
            return result
        etf_age = result["etf_quote_age_sec"]
        bench_age = result["benchmark_quote_age_sec"]
        if etf_age is None or bench_age is None or etf_age > self._max_age or bench_age > self._max_age:
            return result
        etf_ret = self._ret(self._snaps.get(code))
        bench_ret = self._ret(self._snaps.get(self._benchmark))
        if etf_ret is None or bench_ret is None:
            return result
        excess = etf_ret - bench_ret
        result.update({
            "status": self._classify(excess),
            "etf_return": etf_ret,
            "benchmark_return": bench_ret,
            "excess_return": excess,
        })
        return result

    def assessment_for_stock(self, code: str, now: Optional[float] = None) -> dict:
        digits = _digits(code)
        spec = self._stock_map.get(digits)
        meta = self._stock_meta.get(digits, {
            "stock_industry": "", "stock_industries": ""})
        mapping = {
            "mapping_version": self.mapping_version,
            "mapping_source": "exact_primary" if spec is not None else "unmapped",
            "mapping_confidence": 1.0 if spec is not None else 0.0,
            **meta,
        }
        if spec is None:
            return {
                "status": "unavailable", "sector": None, "etf_code": None,
                "benchmark_code": self._benchmark, "reason": "stock_sector_unmapped",
                **mapping,
            }
        result = self.assessment_for_etf(spec.code, now=now)
        result.update(mapping)
        result["reason"] = None if result["status"] != "unavailable" else "sector_quote_unavailable"
        return result

    def summary(self) -> str:
        counts = {"strong": 0, "neutral": 0, "weak": 0, "unavailable": 0}
        for code in self._by_etf:
            status = self.assessment_for_etf(code)["status"]
            counts[status] = counts.get(status, 0) + 1
        return (f"ETF强/中/弱/缺={counts['strong']}/{counts['neutral']}/"
                f"{counts['weak']}/{counts['unavailable']}")
