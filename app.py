"""A股股票分析 App —— Streamlit UI 入口（深色美化版）。

运行： python3 -m streamlit run app.py
颜色约定遵循 A 股习惯：涨=红，跌=绿。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stock_analyzer import advisor, amazingdata_source, backtest, broker_extra, candidate_eval, data, glossary, indicators, moneyflow, net, news, overseas, prediction, quant_signal, screener, sectors, sentiment_signal, snapshot, snapshot_batch, stock_meta, top10_eval

st.set_page_config(page_title="A股技术分析", layout="wide", page_icon="📈")

# 涨=红 跌=绿（A股习惯）
UP, DOWN, FLAT = "#ef232a", "#0aa869", "#e8b339"

# ------------------------- 全局样式 -------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1.2rem; max-width: 1200px;}
    .card {
        background: #131722; border: 1px solid #232a3a; border-radius: 14px;
        padding: 16px 20px; margin-bottom: 14px;
    }
    .hdr {display:flex; justify-content:space-between; align-items:flex-start;}
    .stock-name {font-size: 26px; font-weight: 700; color:#f2f4f8;}
    .stock-code {font-size: 15px; color:#8a93a6; margin-left: 10px;}
    .price {font-size: 34px; font-weight: 700; line-height: 1;}
    .chg {font-size: 16px; margin-left: 10px;}
    .sub-metrics {margin-top: 12px; color:#8a93a6; font-size: 14px;}
    .sub-metrics b {color:#c9d1e0; font-weight:600; margin-left:2px;}
    .sub-metrics span {margin-right: 22px;}
    .pill {padding: 3px 12px; border-radius: 12px; font-size: 13px;
           font-weight: 600; color:#fff; display:inline-block;}
    .strip {color:#8a93a6; font-size:14px;}
    .strip b {font-size:15px; margin-left:2px;}
    .tip-title {font-weight:700; color:#f2f4f8; margin-bottom:6px;}
    .tip-body {color:#aeb7c8; font-size:14px; line-height:1.7;}
    /* Tab 美化 */
    .stTabs [data-baseweb="tab-list"] {gap: 4px; background:#131722;
        border:1px solid #232a3a; border-radius:12px; padding:6px;}
    .stTabs [data-baseweb="tab"] {height:42px; border-radius:8px;
        padding:0 22px; color:#8a93a6; font-size:15px;}
    .stTabs [aria-selected="true"] {background:#1c2740 !important;
        color:#5b8cff !important; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 A股技术分析与加减仓建议")
st.caption("数据来源：AKShare（东财/新浪/腾讯自动切换）· 仅供学习参考，不构成投资建议")


def _default_analysis_symbols() -> str:
    fallback = "600707"
    try:
        codes: list[str] = []

        watch_frame = quant_signal.watchlist_frame(profile="short_stable", style="short_1_3")
        if watch_frame is not None and not watch_frame.empty and "code" in watch_frame.columns:
            sub = watch_frame.copy()
            if "watch_rank" in sub.columns:
                sub["watch_rank"] = pd.to_numeric(sub["watch_rank"], errors="coerce")
                sub = sub.sort_values("watch_rank", ascending=True, na_position="last")
            elif "pred" in sub.columns:
                sub["pred"] = pd.to_numeric(sub["pred"], errors="coerce")
                sub = sub.sort_values("pred", ascending=False)
            for raw in sub["code"].dropna().astype(str):
                code = data._normalize_symbol(raw)
                if code and code not in codes:
                    codes.append(code)
                if len(codes) >= 3:
                    break

        if len(codes) < 3:
            frame = quant_signal.latest_frame(profile="short_stable", style="short_1_3")
            if frame is not None and not frame.empty and "code" in frame.columns:
                sub = frame.copy()
                if "pred" in sub.columns:
                    sub["pred"] = pd.to_numeric(sub["pred"], errors="coerce")
                    sub = sub.sort_values("pred", ascending=False)
                for raw in sub["code"].dropna().astype(str):
                    code = data._normalize_symbol(raw)
                    if code and code not in codes:
                        codes.append(code)
                    if len(codes) >= 3:
                        break

        return " ".join(codes) if codes else fallback
    except Exception:  # noqa: BLE001
        return fallback


# ------------------------- 侧边栏 -------------------------
with st.sidebar:
    st.header("参数")
    symbols_input = st.text_input("股票代码(最多3个)", value=_default_analysis_symbols(), key="symbols_input",
                                  help="可输入1-3个，空格/逗号分隔；默认读取最新量化短线Top3。例：600519 000001 600707").strip()
    days = st.slider("回溯天数(自然日)", 120, 800, 400, step=20)
    adjust = st.selectbox("复权方式", ["qfq", "hfq", ""],
                          format_func=lambda x: {"qfq": "前复权", "hfq": "后复权", "": "不复权"}[x])
    proxy = st.text_input("代理(可选)", value="",
                          placeholder="http://127.0.0.1:7890",
                          help="填代理后：日韩/外围走该代理，且东财(A股行情/名称)请求也经代理路由，可尝试恢复被屏蔽的东财数据源；留空则直连(自动用新浪/腾讯兜底)。").strip()
    qwen_key = st.text_input("Qwen API Key(可选)", value="", type="password",
                             help="填入 DashScope(通义千问) API Key 后，新闻情绪用大模型分析；留空则用本地词典兜底。").strip()
    qwen_base = st.text_input(
        "Qwen接口地址(可选)",
        value="https://ws-yuvikqba21b4koic.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        help="专属工作空间的 openAiCompatible 地址；用公共 DashScope 则改为 https://dashscope.aliyuncs.com/compatible-mode/v1。").strip()
    _QWEN_MODELS = [
        "qwen3.7-plus", "qwen3.7-max", "qwen3.7-max-preview", "qwen3.7-max-2026-06-08",
        "qwen3.7-plus-2026-05-26", "qwen3.7-max-2026-05-20", "qwen3.7-max-2026-05-17",
        "qwen3.6-plus", "qwen3.6-flash", "qwen3.6-flash-2026-04-16", "qwen3.6-35b-a3b",
        "qwen3.5-plus", "qwen3.5-flash", "qwen3.5-plus-2026-04-20",
        "qwen3.5-plus-2026-02-15", "qwen3.5-flash-2026-02-23",
    ]
    qwen_model = st.selectbox("Qwen模型", _QWEN_MODELS, index=0,
                              help="选择通义千问模型；该工作空间需已授权对应模型。默认 qwen3.7-plus。")
    with st.expander("券商数据源 AmazingData(可选)"):
        st.caption("需先安装券商 SDK(tgw + AmazingData wheel)。可用时 A股行情优先走券商官方数据。")
        ad_user = st.text_input("账号", value="", key="ad_user").strip()
        ad_pwd = st.text_input("密码", value="", type="password", key="ad_pwd").strip()
        ad_host = st.text_input("服务器IP", value="101.230.159.234", key="ad_host",
                                help="星耀数智服务器，二选一：101.230.159.234 / 140.206.44.234").strip()
        ad_port = st.text_input("端口", value="8600", key="ad_port").strip()
        st.caption("状态：" + amazingdata_source.status())
    with st.expander("选股/回测 股票池(可选)"):
        pool_text = st.text_area("自选股代码(A)", value="",
                                 placeholder="600707 600519 000001\n或逗号/换行分隔",
                                 help="批量打分与回测用的股票池；与行业成分合并去重。")
        pool_industry = st.text_input("行业板块名(B, 可选)", value="",
                                      placeholder="如：半导体 / 银行 / 白酒",
                                      help="填东财行业板块名，取其成分股(最多50只)并入股票池。").strip()
    run = st.button("开始分析", type="primary", use_container_width=True)

# 代理变更时更新并清理外围/板块缓存，确保用新出口重新拉取
if proxy != st.session_state.get("_proxy"):
    st.session_state["_proxy"] = proxy
    _plist = [proxy] if proxy else []
    overseas.set_proxies(_plist)
    net.set_proxies(_plist)            # 同一代理也用于 akshare(东财)请求
    overseas.analyze.cache_clear()
    sectors.analyze_sectors.cache_clear()
    sectors._fetch_close.cache_clear()
    data.fetch_daily.cache_clear()     # 代理变化后重试东财数据源
    data.stock_name_map.cache_clear()

# 券商账号变更时重设登录并清理行情缓存
_ad_sig = (ad_user, ad_pwd, ad_host, ad_port)
if _ad_sig != st.session_state.get("_ad"):
    st.session_state["_ad"] = _ad_sig
    if ad_user and ad_pwd:
        amazingdata_source.set_credentials(ad_user, ad_pwd, ad_host,
                                           int(ad_port) if ad_port.isdigit() else 0)
        data.fetch_daily.cache_clear()
        broker_extra.clear_cache()

# Qwen 配置变更时刷新新闻分析缓存，使新 key/模型立即生效
_qwen_sig = (qwen_key, qwen_base, qwen_model)
if _qwen_sig != st.session_state.get("_qwen"):
    st.session_state["_qwen"] = _qwen_sig
    news.analyze_sector_news.cache_clear()


@st.cache_data(ttl=900, show_spinner=False)
def load(symbol: str, days: int, adjust: str):
    profile = {"steps": []}
    t0 = time.perf_counter()
    raw = data.fetch_daily(symbol, days=days, adjust=adjust)
    profile["steps"].append({"name": "K线", "sec": round(time.perf_counter() - t0, 3)})
    t1 = time.perf_counter()
    df = indicators.compute_all(raw)
    profile["steps"].append({"name": "指标", "sec": round(time.perf_counter() - t1, 3)})
    t2 = time.perf_counter()
    name = data.get_stock_name(symbol)
    profile["steps"].append({"name": "名称", "sec": round(time.perf_counter() - t2, 3)})
    profile["total_sec"] = round(time.perf_counter() - t0, 3)
    profile["kline"] = data.last_profile(symbol)
    return df, name, profile


@st.cache_data(ttl=900, show_spinner=False)
def load_overseas(proxy_key: str = ""):
    return overseas.analyze()


@st.cache_data(ttl=900, show_spinner=False)
def load_linkage(symbol: str, proxy_key: str = "", qwen_key: str = "",
                 qwen_base: str = "", qwen_model: str = ""):
    return sectors.analyze_linkage(symbol, key=qwen_key, model=qwen_model, base_url=qwen_base)


@st.cache_data(ttl=900, show_spinner=False)
def load_news(symbol: str, qwen_key: str = "", qwen_base: str = "", qwen_model: str = ""):
    return news.analyze(symbol, key=qwen_key, model=qwen_model, base_url=qwen_base)


@st.cache_data(ttl=120, show_spinner=False)
def load_broker(symbol: str, ad_sig: str = "", retry_token: int = 0):
    return broker_extra.analyze(symbol, retry=6, retry_interval=2.0)


@st.cache_data(ttl=900, show_spinner=False)
def load_money(symbol: str, price_df: pd.DataFrame):
    return moneyflow.analyze(symbol, price_df=price_df, prefer_remote=False)


def fmt_num(x, unit=""):
    """大数字转 亿/万，无效值返回 --。"""
    if x is None or (isinstance(x, float) and (np.isnan(x) or x == 0)):
        return "--"
    if abs(x) >= 1e8:
        return f"{x / 1e8:.2f}亿{unit}"
    if abs(x) >= 1e4:
        return f"{x / 1e4:.2f}万{unit}"
    return f"{x:.2f}{unit}"


def pct(v):
    return "--" if v is None or np.isnan(v) or v == 0 else f"{v:.2f}%"


def signal_pill(score: int):
    """返回 (文案, 颜色)。看多=红，看空=绿，中性=黄。"""
    if score > 0:
        return "偏多信号", UP
    if score < 0:
        return "偏空信号", DOWN
    return "观望信号", FLAT


def get_signal(advice, name: str):
    return next((s for s in advice.signals if s.name == name), None)


def tip_card(title: str, body: str):
    st.markdown(
        f'<div class="card"><div class="tip-title">💡 {title}</div>'
        f'<div class="tip-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def explain(name: str):
    """展开式「指标说明」，用大白话讲清术语的含义/算法/解读。"""
    html = glossary.explain_html(name)
    if not html:
        return
    with st.expander(f"📖 看不懂？点开了解「{name}」是什么意思"):
        st.markdown(html, unsafe_allow_html=True)


# ---- 图表样式 ----
def _style(fig, height):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), legend_orientation="h",
        legend=dict(y=1.08, x=0.5, xanchor="center"),
        xaxis=dict(gridcolor="#1e2536"), yaxis=dict(gridcolor="#1e2536"),
    )
    return fig


def volume_chart(df):
    colors = [UP if c >= o else DOWN for c, o in zip(df["close"], df["open"])]
    fig = go.Figure()
    fig.add_bar(x=df["date"], y=df["volume"], name="成交量", marker_color=colors)
    fig.add_scatter(x=df["date"], y=df["vol_ma5"], name="5日均量",
                    line=dict(color="#e8b339", width=1.5))
    return _style(fig, 420)


def kdj_chart(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("KDJ", "RSI", "BIAS 乖离率"))
    for col, c in zip(("kdj_k", "kdj_d", "kdj_j"), ("#e8b339", "#5b8cff", "#cc6bff")):
        fig.add_scatter(x=df["date"], y=df[col], name=col.split("_")[-1].upper(),
                        line=dict(width=1.4, color=c), row=1, col=1)
    for n, c in zip((6, 12, 24), ("#e8b339", "#5b8cff", "#cc6bff")):
        fig.add_scatter(x=df["date"], y=df[f"rsi{n}"], name=f"RSI{n}",
                        line=dict(width=1.2, color=c), row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color=UP, row=2, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color=DOWN, row=2, col=1)
    for n, c in zip((6, 12, 24), ("#e8b339", "#5b8cff", "#cc6bff")):
        fig.add_scatter(x=df["date"], y=df[f"bias{n}"], name=f"BIAS{n}",
                        line=dict(width=1.2, color=c), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#8a93a6", row=3, col=1)
    return _style(fig, 620)


def ma_chart(df, name, code):
    fig = go.Figure()
    fig.add_candlestick(x=df["date"], open=df["open"], high=df["high"],
                        low=df["low"], close=df["close"], name="K线",
                        increasing_line_color=UP, decreasing_line_color=DOWN)
    for n, c in zip((5, 10, 20, 60), ("#e8b339", "#5b8cff", "#cc6bff", "#8a93a6")):
        fig.add_scatter(x=df["date"], y=df[f"ma{n}"], name=f"MA{n}",
                        line=dict(width=1.2, color=c))
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _style(fig, 520)


# ------------------------- 主体 -------------------------
if run:
    st.session_state["started"] = True
if not st.session_state.get("started"):
    st.info("在左侧输入股票代码后点击「开始分析」。")
    st.stop()

# 解析最多3个代码，多个时提供选择器（各模块按所选股票渲染）
import re as _re_sym
_main_codes = [data._normalize_symbol(c) for c in _re_sym.split(r"[\s,，、;]+", symbols_input) if c.strip()][:3]
if not _main_codes:
    _main_codes = ["600707"]
if len(_main_codes) > 1:
    symbol = st.radio("选择要查看的股票（已一次分析 %d 只）" % len(_main_codes),
                      _main_codes, horizontal=True)
else:
    symbol = _main_codes[0]
_prefetch_codes = [c for c in _main_codes if c != symbol]

# 点击分析后立即给出可见反馈。后续量化文件和行情接口即使响应较慢，页面也不会像无响应。
_analysis_status = st.status("正在准备分析...", expanded=True)
_analysis_status.write("正在读取量化模型与白名单结果...")

# ---- 白名单量化总览 + 当前3只股票量化并排 ----
def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _fmt_pct(v, digits: int = 2):
    return f"{v * 100:+.{digits}f}%" if pd.notna(v) else "-"


def _fmt_pct_plain(v, digits: int = 1):
    return f"{v:.{digits}f}%" if pd.notna(v) else "-"


def _quant_eval_display(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df is None or eval_df.empty:
        return pd.DataFrame()
    cols = ["horizon", "annual_return", "sharpe", "max_drawdown", "win_rate", "direction_win_rate"]
    use = [c for c in cols if c in eval_df.columns]
    out = eval_df[use].copy().sort_values("horizon")
    out = out.rename(columns={
        "horizon": "周期", "annual_return": "年化", "sharpe": "Sharpe",
        "max_drawdown": "最大回撤", "win_rate": "组合胜率", "direction_win_rate": "单票方向胜率"})
    out["周期"] = out["周期"].map(lambda h: f"{int(h)}日")
    for c in ("年化", "最大回撤"):
        if c in out.columns:
            out[c] = out[c].map(lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "-")
    for c in ("组合胜率", "单票方向胜率"):
        if c in out.columns:
            out[c] = out[c].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) and v <= 1 else _fmt_pct_plain(v))
    if "Sharpe" in out.columns:
        out["Sharpe"] = out["Sharpe"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    return out


def _quant_advice_display(advice_df: pd.DataFrame) -> pd.DataFrame:
    if advice_df is None or advice_df.empty:
        return pd.DataFrame()
    out = advice_df.copy()
    out["训练目标"] = out.get("expected_return_horizon", pd.Series(dtype=float)).map(
        lambda h: f"{int(h)}日" if pd.notna(h) else "-"
    )
    out["Ridge预估收益"] = out.get("expected_return", pd.Series(dtype=float)).map(lambda v: _fmt_pct(v))
    out["白名单分位"] = out.get("watch_rank_pct", pd.Series(dtype=float)).map(
        lambda v: f"{v*100:.1f}%" if pd.notna(v) else "-")
    for c in ("quant_score", "target_price", "stop_loss", "risk_reward"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(3 if c == "quant_score" else 2)
    out = out.rename(columns={
        "style_label": "口径", "holding_days": "持有", "suggestion": "建议", "code": "代码",
        "name": "名称", "a_industry": "A股行业", "a_concepts": "A股概念",
        "overseas_sector": "外围映射", "business": "主营业务",
        "direction": "方向", "quant_score": "量化分", "entry_price": "参考价", "stop_loss": "止损价",
        "target_label": "目标类型", "target_price": "目标价", "risk_reward": "盈亏比",
        "watch_rank": "白名单排名", "rank": "全A排名", "validation_horizons": "验证周期",
        "reason": "理由", "date": "日期", "profile_label": "模型口径",
    })
    if "主营业务" in out.columns:
        out["主营业务"] = out["主营业务"].map(lambda v: (str(v)[:34] + "...") if isinstance(v, str) and len(v) > 36 else v)
    cols = ["口径", "持有", "训练目标", "建议", "代码", "名称", "A股行业", "A股概念", "外围映射", "方向", "Ridge预估收益", "量化分",
            "参考价", "止损价", "目标类型", "目标价", "盈亏比", "白名单排名", "白名单分位",
            "全A排名", "验证周期", "主营业务", "理由"]
    return out[[c for c in cols if c in out.columns]]


_QUANT_STYLE_TABLES = (
    ("short_1_3", "短线建议（1-3天）"),
    ("swing_7_15", "波段建议（7-15天）"),
)


def _render_quant_advice_tables(advice_df: pd.DataFrame, empty_text: str, height: int = 320) -> bool:
    if advice_df is None or advice_df.empty:
        st.caption(empty_text)
        return False
    rendered = False
    for style_key, title in _QUANT_STYLE_TABLES:
        if "style_key" in advice_df.columns:
            part = advice_df[advice_df["style_key"].astype(str) == style_key]
        else:
            cfg = quant_signal.trade_style_config(style_key)
            label = str(cfg.get("label") or "")
            part = advice_df[advice_df.get("style_label", pd.Series(dtype=str)).astype(str) == label]
        show = _quant_advice_display(part)
        if show.empty:
            continue
        st.markdown(f"**{title}**")
        st.dataframe(show, use_container_width=True, hide_index=True, height=height)
        rendered = True
    if not rendered:
        show = _quant_advice_display(advice_df)
        if show.empty:
            st.caption(empty_text)
            return False
        st.dataframe(show, use_container_width=True, hide_index=True, height=height)
        rendered = True
    return rendered


_profile_options = quant_signal.profile_options()
_quant_profile = st.selectbox(
    "量化档位",
    options=list(_profile_options.keys()),
    format_func=lambda k: _profile_options.get(k, k),
    index=list(_profile_options.keys()).index("short_stable") if "short_stable" in _profile_options else 0,
    help="用于下方单口径白名单表和历史验证；短线/波段双建议会按各自发布的 active 模型读取。",
)
_profile_cfg = quant_signal.profile_config(_quant_profile)
st.caption(f"当前单口径量化档位：**{_profile_cfg.get('label')}** ｜ {_profile_cfg.get('note', '')}")


with st.expander("🧮 白名单量化打分总览（全部自选股 · 打分/预估涨跌）", expanded=False):
    _dual_wl = _safe(lambda: quant_signal.watchlist_trade_advice())
    _render_quant_advice_tables(_dual_wl, "暂无白名单量化打分（需已发布 active 量化模型且自选池股票在预测范围内）。", height=320)
    st.caption("短线=1-3天，使用稳健短线口径和止盈1；波段=7-15天，使用V2进攻口径和止盈2。波段历史验证覆盖7/10/15日。")

    with st.expander("单口径明细与历史验证", expanded=False):
        _wl = _safe(lambda: quant_signal.watchlist_frame(profile=_quant_profile))
        if _wl is None or _wl.empty:
            st.info("暂无单口径白名单量化打分。")
        else:
            _asof = str(_wl["date"].iloc[0])
            _model_ver = str(_wl["model"].iloc[0])
            _pred_h = quant_signal.prediction_horizon()
            _pred_col = f"Ridge预估{_pred_h}日收益"
            st.caption(f"单口径打分日期 **{_asof}** ｜ 模型 **{_model_ver}** ｜ 池内股票 **{len(_wl)}** 只"
                       f"（量化分为横截面融合分，{_pred_col}为当前选择口径的回归预测，方向由融合分正负判定）")
            _wl_show = _wl.copy()
            _wl_show[_pred_col] = _wl_show["expected_return"].map(lambda v: _fmt_pct(v))
            _wl_show = _wl_show.rename(columns={
                "code": "代码", "name": "名称", "a_industry": "A股行业", "a_concepts": "A股概念",
                "overseas_sector": "外围映射", "business": "主营业务",
                "direction": "方向", "pred": "量化分",
                "watch_rank": "白名单排名", "watch_rank_pct": "白名单分位", "rank": "全A排名"})
            _wl_show["量化分"] = _wl_show["量化分"].round(3)
            _wl_show["白名单分位"] = (_wl_show["白名单分位"] * 100).round(1).map(lambda v: f"{v:.1f}%")
            _wl_show = _wl_show.rename(columns={
                "entry_price": "参考价", "stop_loss": "止损价", "take_profit_1": "止盈1",
                "take_profit_2": "止盈2", "risk_reward_1": "盈亏比1", "risk_reward_2": "盈亏比2"})
            if "主营业务" in _wl_show.columns:
                _wl_show["主营业务"] = _wl_show["主营业务"].map(lambda v: (str(v)[:34] + "...") if isinstance(v, str) and len(v) > 36 else v)
            _wl_cols = ["白名单排名", "代码", "名称", "A股行业", "A股概念", "外围映射", "方向", _pred_col, "量化分",
                        "参考价", "止损价", "止盈1", "止盈2", "盈亏比1", "白名单分位", "全A排名", "主营业务"]
            st.dataframe(
                _wl_show[[c for c in _wl_cols if c in _wl_show.columns]],
                use_container_width=True, hide_index=True, height=320)
            _bull = int((_wl["direction"] == "看多").sum())
            _bear = int((_wl["direction"] == "看空").sum())
            st.caption(f"方向分布：看多 **{_bull}** ｜ 看空 **{_bear}**")

            _eval = _safe(lambda: quant_signal.selected_evaluation(profile=_quant_profile))
            _eval_show = _quant_eval_display(_eval)
            if not _eval_show.empty:
                st.markdown("**白名单历史验证（动态周期）**")
                st.dataframe(_eval_show, use_container_width=True, hide_index=True)
                _ef = str(_eval["evaluation_file"].iloc[0]) if "evaluation_file" in _eval.columns and not _eval.empty else ""
                st.caption(f"验证周期来自评估文件：{_ef}；后续新增 10/15/30 日会自动扩展显示。")

    st.markdown("**本次分析的股票**")
    _q3_dual = _safe(lambda: quant_signal.trade_advice_for_codes(_main_codes))
    _render_quant_advice_tables(_q3_dual, "当前股票无量化打分。", height=220)

_analysis_status.write(f"正在加载 {symbol} 行情并计算技术指标...")
try:
    df, name, _load_profile = load(symbol, days, adjust)
except Exception as e:  # noqa: BLE001
    _analysis_status.update(label="分析失败", state="error", expanded=True)
    st.error(f"分析失败：{e}")
    st.stop()
_analysis_status.write("行情和技术指标已完成，正在启动外围、新闻、资金流和基本面接口...")

latest = df.iloc[-1]
prev_close = float(df["close"].iloc[-2])
change = float(latest["close"]) - prev_close
change_pct = change / prev_close * 100
cls_color = UP if change >= 0 else DOWN
amplitude = float(latest["amplitude"]) if "amplitude" in df.columns else \
    (float(latest["high"]) - float(latest["low"])) / prev_close * 100

advice = advisor.advise(df)

# 并发预取各重型数据源（外围/板块联动/新闻/资金流/基本面），使各栏各自就绪即显示，
# 而不是逐个串行等待。任务放入 session_state，同一组参数下 rerun 时复用正在跑的 future。
import threading as _threading
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx as _add_ctx, get_script_run_ctx as _get_ctx
except Exception:  # noqa: BLE001
    _add_ctx = _get_ctx = None

_ctx = _get_ctx() if _get_ctx else None
_PREFETCH_KEYS = {"overseas", "linkage", "news", "money", "broker"}


def _prefetch_init():
    if _add_ctx and _ctx:
        _add_ctx(_threading.current_thread(), _ctx)


_broker_retry_token = int(st.session_state.get("_broker_retry_token", 0) or 0)


def _task_signature() -> tuple:
    qwen_sig = (len(qwen_key or ""), (qwen_key or "")[-6:])
    money_sig = (len(df), str(df["date"].iloc[-1]) if "date" in df.columns and len(df) else "")
    return (symbol, proxy, qwen_sig, qwen_base, qwen_model, ad_user, ad_host, ad_port, money_sig, _broker_retry_token)


def _submit_task(pool, label: str, fn):
    task = {"label": label, "start": time.perf_counter(), "end": None, "future": None}

    def _wrapped():
        try:
            return fn()
        finally:
            task["end"] = time.perf_counter()

    task["future"] = pool.submit(_wrapped)
    return task


def _start_prefetch_tasks():
    pool = _ThreadPoolExecutor(max_workers=5, initializer=_prefetch_init)
    tasks = {
        "overseas": _submit_task(pool, "外围市场", lambda: load_overseas(proxy)),
        "linkage": _submit_task(pool, "美股板块联动", lambda: load_linkage(symbol, proxy, qwen_key, qwen_base, qwen_model)),
        "news": _submit_task(pool, "新闻资讯", lambda: load_news(symbol, qwen_key, qwen_base, qwen_model)),
        "money": _submit_task(pool, "资金流向", lambda: load_money(symbol, df)),
        "broker": _submit_task(pool, "基本面/资金面", lambda: load_broker(symbol, "|".join([ad_user, ad_host, ad_port]), _broker_retry_token)),
    }
    pool.shutdown(wait=False)
    return tasks


def _iter_task_futures(task: dict):
    fut = task.get("future") if isinstance(task, dict) else None
    if fut is not None:
        yield fut
    for item in (task.get("items") or []) if isinstance(task, dict) else []:
        item_fut = item.get("future") if isinstance(item, dict) else None
        if item_fut is not None:
            yield item_fut


def _start_warmup_task():
    pool = _ThreadPoolExecutor(max_workers=1, initializer=_prefetch_init)
    codes = [c for c in _main_codes if c != symbol][:2]

    def _warmup_all():
        out = []
        for code in codes:
            t0 = time.perf_counter()
            try:
                load(code, days, adjust)
                out.append({"code": code, "ok": True, "sec": round(time.perf_counter() - t0, 2), "note": ""})
            except Exception as e:  # noqa: BLE001
                out.append({"code": code, "ok": False, "sec": round(time.perf_counter() - t0, 2), "note": f"{type(e).__name__}: {e}"})
        return out

    task = _submit_task(pool, "其它股票行情预热", _warmup_all)
    pool.shutdown(wait=False)
    return task


_warmup_sig = (tuple(_main_codes), days, adjust)
_warmup_task = st.session_state.get("_warmup_task")
if st.session_state.get("_warmup_sig") != _warmup_sig or not isinstance(_warmup_task, dict):
    for fut in _iter_task_futures(st.session_state.get("_warmup_task") or {}):
        if not fut.done():
            fut.cancel()
    _warmup_task = _start_warmup_task()
    st.session_state["_warmup_sig"] = _warmup_sig
    st.session_state["_warmup_task"] = _warmup_task


# 固定 Top10 的大模型结果由 scheduler 在量化发布后统一计算并持久化；UI 只读缓存。
_cand_codes = candidate_eval.top_candidates(n=10, profile=_quant_profile, style="short_1_3")
_top10_cache = _safe(top10_eval.load) or {}


_task_sig = _task_signature()
_tasks = st.session_state.get("_prefetch_tasks")
if st.session_state.get("_prefetch_sig") != _task_sig or not isinstance(_tasks, dict) or set(_tasks) != _PREFETCH_KEYS:
    for task in (st.session_state.get("_prefetch_tasks") or {}).values():
        for fut in _iter_task_futures(task):
            if not fut.done():
                fut.cancel()
    _tasks = _start_prefetch_tasks()
    st.session_state["_prefetch_sig"] = _task_sig
    st.session_state["_prefetch_tasks"] = _tasks
_fut_overseas = _tasks["overseas"]["future"]
_fut_linkage = _tasks["linkage"]["future"]
_fut_news = _tasks["news"]["future"]
_fut_money = _tasks["money"]["future"]
_fut_broker = _tasks["broker"]["future"]


def _future_value(key: str):
    task = _tasks[key]
    fut = task["future"]
    elapsed = (task.get("end") or time.perf_counter()) - task["start"]
    if not fut.done():
        return None, f"{task['label']}仍在加载（已 {elapsed:.1f}s），不会阻塞其它栏目。"
    try:
        val = fut.result()
        return val, f"{task['label']}已完成，用时 {elapsed:.1f}s。"
    except Exception as e:  # noqa: BLE001
        return None, f"{task['label']}失败：{type(e).__name__}: {e}（{elapsed:.1f}s）"


def _done_value(key: str):
    val, _ = _future_value(key)
    return val


def _task_status_frame() -> pd.DataFrame:
    rows = []
    now = time.perf_counter()
    if _warmup_task:
        fut = _warmup_task["future"]
        sec = (_warmup_task.get("end") or now) - _warmup_task["start"]
        if not fut.done():
            state = "加载中"
        elif fut.exception():
            state = "失败"
        else:
            result = fut.result() or []
            done = sum(1 for x in result if x.get("ok"))
            state = f"完成 {done}/{len(result)}" if result else "无待预热股票"
        rows.append({"接口": _warmup_task["label"], "状态": state, "耗时": f"{sec:.1f}s"})
    for key, task in _tasks.items():
        fut = task["future"]
        sec = (task.get("end") or now) - task["start"]
        if not fut.done():
            state = "加载中"
        else:
            state = "失败" if fut.exception() else "完成"
        rows.append({"接口": task["label"], "状态": state, "耗时": f"{sec:.1f}s"})
    return pd.DataFrame(rows)


_pending_now = (
    any(not fut.done() for task in _tasks.values() for fut in _iter_task_futures(task))
    or any(not fut.done() for fut in _iter_task_futures(_warmup_task or {}))
)
if _pending_now:
    _analysis_status.update(
        label="首屏已就绪，后台数据继续加载",
        state="running",
        expanded=True,
    )
else:
    _analysis_status.update(label="分析数据已就绪", state="complete", expanded=False)

with st.expander("接口加载状态", expanded=_pending_now):
    st.dataframe(_task_status_frame(), use_container_width=True, hide_index=True)
    if _pending_now:
        st.caption("外围、新闻、资金流和基本面正在后台运行；已完成内容可先查看，刷新后会更新状态。")
    else:
        st.caption("后台接口已全部返回。")
    if st.button("刷新后台结果 / 重试券商", key="refresh_prefetch_results"):
        broker_task = _tasks.get("broker")
        if broker_task and broker_task["future"].done():
            try:
                broker_val = broker_task["future"].result()
            except Exception:  # noqa: BLE001
                broker_val = None
            if broker_val is None or not getattr(broker_val, "signals", None):
                st.session_state["_broker_retry_token"] = int(st.session_state.get("_broker_retry_token", 0) or 0) + 1
                load_broker.clear()
                broker_extra.clear_cache()
        st.rerun()


# ---- 头部卡片 ----
st.markdown(
    f"""
    <div class="card">
      <div class="hdr">
        <div><span class="stock-name">{name}</span>
             <span class="stock-code">{symbol}</span></div>
        <div style="text-align:right">
          <span class="price" style="color:{cls_color}">{latest['close']:.2f}</span>
          <span class="chg" style="color:{cls_color}">{change:+.2f}　{change_pct:+.2f}%</span>
        </div>
      </div>
      <div class="sub-metrics">
        <span>最高 <b>{latest['high']:.2f}</b></span>
        <span>最低 <b>{latest['low']:.2f}</b></span>
        <span>今开 <b>{latest['open']:.2f}</b></span>
        <span>昨收 <b>{prev_close:.2f}</b></span>
        <span>换手率 <b>{pct(float(latest['turnover']))}</b></span>
        <span>振幅 <b>{amplitude:.2f}%</b></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"📡 本次行情数据源：**{data.last_source(symbol)}**"
           + ("　（券商官方）" if data.last_source(symbol) == "银河AmazingData" else ""))
try:
    _step_txt = "　·　".join(f"{x['name']} {x['sec']:.2f}s" for x in _load_profile.get("steps", []))
    _src_txt = "　·　".join(
        f"{x.get('source')} {'✓' if x.get('ok') else '×'} {float(x.get('sec', 0)):.2f}s"
        for x in (_load_profile.get("kline", {}) or {}).get("sources", [])
    )
    if _step_txt or _src_txt:
        st.caption("⏱ 首屏耗时：" + (_step_txt or "-") + (f" ｜ K线源：{_src_txt}" if _src_txt else ""))
except Exception:  # noqa: BLE001
    pass


# ---- 候选栏 Top8（白名单量化优选 + 大模型综合评估排序）----
def _render_candidate_panel():
    if not _cand_codes:
        st.info("暂无候选（需已发布 active 量化模型且白名单股票在预测范围内）。")
        return

    # 各口径最优持有天数（来自白名单历史验证：按 Sharpe/方向胜率选最优 horizon）
    def _reco_line(style: str, label: str) -> str:
        r = _safe(lambda: quant_signal.recommended_horizon(style=style))
        if not r:
            return f"{label} 暂无验证"
        parts = [f"建议持有 **{r['horizon']}** 日"]
        if r.get("sharpe") is not None:
            parts.append(f"Sharpe {r['sharpe']:.2f}")
        if r.get("direction_win_rate") is not None:
            parts.append(f"方向胜率 {r['direction_win_rate'] * 100:.1f}%")
        elif r.get("win_rate") is not None:
            parts.append(f"胜率 {r['win_rate'] * 100:.1f}%")
        return f"{label} " + "，".join(parts)

    st.markdown("📌 **最优持有周期（历史验证）**：" + _reco_line("short_1_3", "短线") + " ｜ " + _reco_line("swing_7_15", "波段"))
    st.caption("最优 h 来自白名单历史验证(现口径含次日开盘成交/成本/涨停停牌过滤)，随每次训练重寻优更新；持有以此为参考，触及止盈/止损优先离场。")

    # 量化优选列表（秒级，先展示）
    _wf = _safe(lambda: quant_signal.watchlist_frame(profile=_quant_profile, style="short_1_3"))
    if _wf is not None and not _wf.empty:
        _wf = _wf[_wf["code"].astype(str).isin(set(_cand_codes))].copy()
        if "watch_rank" in _wf.columns:
            _wf["watch_rank"] = pd.to_numeric(_wf["watch_rank"], errors="coerce")
            _wf = _wf.sort_values("watch_rank", na_position="last")
        _q = _wf.rename(columns={
            "watch_rank": "量化排名", "code": "代码", "name": "名称",
            "a_industry": "A股行业", "a_concepts": "A股概念", "direction": "量化方向"})
        _q["Ridge预估"] = _wf["expected_return"].map(lambda v: _fmt_pct(v))
        _q_cols = [c for c in ["量化排名", "代码", "名称", "A股行业", "A股概念", "量化方向", "Ridge预估"] if c in _q.columns]
        st.markdown("**量化优选 Top8（按白名单池内排名）**")
        st.dataframe(_q[_q_cols], use_container_width=True, hide_index=True, height=210)

    # 同一短线口径的全 A 横截面排名，不调用大模型。
    _all_a = _safe(lambda: quant_signal.latest_frame(profile=_quant_profile, style="short_1_3"))
    if _all_a is not None and not _all_a.empty:
        # 仅展示普通账户可交易的沪深主板；保留原始全 A rank，不把筛选后顺序冒充全 A 名次。
        _main_board_prefixes = ("600", "601", "603", "605", "000", "001", "002", "003")
        _all_a = _all_a[_all_a["code"].astype(str).str.startswith(_main_board_prefixes)]
        _all_a = _all_a.sort_values("rank", na_position="last").head(10).copy()
        _all_a["expected_return"] = pd.to_numeric(_all_a.get("ridge_pred"), errors="coerce")
        _all_a["direction"] = pd.to_numeric(_all_a["pred"], errors="coerce").map(
            lambda score: "看多" if score > 0 else ("看空" if score < 0 else "中性"))
        _all_a = stock_meta.enrich_frame(
            _all_a,
            remote=False,
            use_all_a_meta=True,
        )
        if "meta_updated_at" in _all_a.columns:
            _meta_ready = _all_a["meta_updated_at"].fillna("").astype(str).ne("")
            _all_a.loc[~_meta_ready, ["a_industry", "a_concepts"]] = "映射更新中"
        else:
            _all_a["a_industry"] = "映射更新中"
            _all_a["a_concepts"] = "映射更新中"
        _all_a_names = _safe(data.stock_name_map) or {}
        if _all_a_names:
            _all_a["name"] = _all_a["code"].astype(str).map(_all_a_names).fillna(_all_a["name"])
        _a = _all_a.rename(columns={
            "date": "预测日期", "rank": "全A排名", "code": "代码", "name": "名称",
            "market_board": "交易板块", "a_industry": "东财行业", "a_concepts": "东财概念",
            "direction": "量化方向", "pred": "量化分"})
        _a["量化分"] = pd.to_numeric(_a["量化分"], errors="coerce").round(4)
        _a["Ridge预估"] = _all_a["expected_return"].map(lambda v: _fmt_pct(v))
        _a_cols = [c for c in [
            "预测日期", "全A排名", "代码", "名称", "交易板块", "东财行业", "东财概念",
            "量化方向", "量化分", "Ridge预估",
        ] if c in _a.columns]
        st.markdown("**全 A 量化排名 Top10（仅沪深主板，短线融合分）**")
        st.dataframe(_a[_a_cols], use_container_width=True, hide_index=True, height=385)

    # 固定 Top10 大模型综合排序：读取 scheduler 在量化发布后生成的持久化缓存。
    st.markdown("---")
    st.markdown("**🧠 大模型综合评估排序（模型发布后自动更新）**")
    _cache_tabs = st.tabs(["白名单 Top10", "全 A Top10", "创新药 Top10"])
    for _group, _cache_tab in zip(("白名单", "全A", "创新药"), _cache_tabs):
        with _cache_tab:
            _entry = _top10_cache.get(_group, {})
            _current_frame = (_safe(top10_eval.ranking_frames) or {}).get(_group, pd.DataFrame())
            _cache_valid = (
                not _current_frame.empty
                and _entry.get("cache_version") == top10_eval._CACHE_VERSION
                and _entry.get("date") == str(_current_frame["date"].iloc[0])
                and _entry.get("codes") == _current_frame["code"].astype(str).tolist()
                and _entry.get("fingerprint") == top10_eval._fingerprint(_current_frame)
            )
            if not _cache_valid:
                _entry = {}
            _ok = [r for r in (_entry.get("rows") or []) if r.get("available") and not r.get("stale")]
            if not _ok:
                st.info("缓存尚未生成；量化模型下次成功发布后将自动评估。")
                continue
            _tbl = pd.DataFrame([{
                "综合排名": r.get("llm_rank"),
                "代码": r.get("code"),
                "名称": r.get("name"),
                "大模型方向": r.get("direction"),
                "信心": r.get("confidence"),
                "综合分": round(float(r.get("composite") or 0.0), 2),
                "量化分": r.get("quant_score"),
                "核心逻辑": str(r.get("logic") or "")[:80],
            } for r in _ok])
            st.dataframe(_tbl, use_container_width=True, hide_index=True, height=385)
            st.caption(f"预测日期 {_entry.get('date', '-')} ｜ 模型 {_entry.get('model', '-')} ｜ 缓存 {_entry.get('updated_at', '-')}")
            with st.expander("查看核心逻辑/操作建议", expanded=False):
                for r in _ok:
                    st.markdown(f"**{r.get('llm_rank')}. {r.get('name')}（{r.get('code')}）** "
                                f"· {r.get('direction')}/{r.get('confidence')} · 综合分 {float(r.get('composite') or 0):+.2f}")
                    if r.get("logic"):
                        st.caption(f"逻辑：{r['logic']}")
                    if r.get("action"):
                        st.caption(f"操作：{r['action']}")


with st.expander("⭐ 候选栏 · Top8（量化优选 + 大模型综合排序）", expanded=True):
    _render_candidate_panel()

tab_pred, tab_vol, tab_kdj, tab_ma, tab_adv, tab_out, tab_news, tab_money, tab_fund, tab_screen = st.tabs(
    ["🔮 次日涨跌预估", "📊 成交量分析", "📈 KDJ指标", "〰 均线系统", "🎯 加减仓建议",
     "🌍 外围市场", "📰 新闻资讯", "💰 资金流向", "🏦 基本面/资金面", "🧮 选股/回测"]
)
with st.expander("基本面接口诊断", expanded=False):
    if st.button("基本面Top3自测", key="broker_top3_self_test"):
        rows = []
        broker_extra.clear_cache()
        for code in _main_codes:
            t0 = time.perf_counter()
            try:
                r = broker_extra.analyze(code, retry=2, retry_interval=0.5)
                rows.append({"代码": code, "可用": r.available, "信号数": len(r.signals),
                             "财务": "有" if r.revenue_text else "无", "耗时": f"{time.perf_counter() - t0:.1f}s",
                             "说明": (r.note or r.label)[:80]})
            except Exception as e:  # noqa: BLE001
                rows.append({"代码": code, "可用": False, "信号数": 0, "财务": "无",
                             "耗时": f"{time.perf_counter() - t0:.1f}s", "说明": f"{type(e).__name__}: {e}"[:80]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("仅在券商基本面数据长时间无结果时使用；普通分析无需执行。")

# ---- Tab 1 成交量 ----
with tab_vol:
    sig = get_signal(advice, "成交量/量能")
    label, color = signal_pill(sig.score)
    amt_ma5 = float(df["amount"].rolling(5).mean().iloc[-1])
    st.markdown(
        f'<div class="card"><span class="strip">当前量比：'
        f'<b style="color:{UP}">{latest["vol_ratio"]:.2f}</b>'
        f'　　5日均额：<b style="color:{UP}">{fmt_num(amt_ma5)}</b>'
        f'　　信号：<span class="pill" style="background:{color}">{label}</span></span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(volume_chart(df), use_container_width=True)
    tip_card("成交量分析", sig.detail)
    explain("成交量/量能")
    explain("换手率")

# ---- Tab 2 KDJ/RSI/BIAS ----
with tab_kdj:
    parts = []
    for nm in ("KDJ", "RSI", "BIAS乖离率"):
        s = get_signal(advice, nm)
        lb, cl = signal_pill(s.score)
        parts.append(f'{nm}：<span class="pill" style="background:{cl}">{lb}</span>')
    st.markdown(f'<div class="card"><span class="strip">'
                + "　　".join(parts) + "</span></div>", unsafe_allow_html=True)
    st.plotly_chart(kdj_chart(df), use_container_width=True)
    for nm in ("KDJ", "RSI", "BIAS乖离率"):
        s = get_signal(advice, nm)
        tip_card(nm, s.detail)
        explain(nm)

# ---- Tab 3 均线 ----
with tab_ma:
    sig = get_signal(advice, "均线系统")
    label, color = signal_pill(sig.score)
    st.markdown(
        f'<div class="card"><span class="strip">均线信号：'
        f'<span class="pill" style="background:{color}">{label}</span></span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(ma_chart(df, name, symbol), use_container_width=True)
    tip_card("均线系统", sig.detail)
    explain("均线系统")

# ---- Tab 4 综合建议 ----
with tab_adv:
    lvl_color = {"bullish": UP, "bearish": DOWN, "neutral": FLAT}[advice.level]
    st.markdown(
        f'<div class="card" style="text-align:center">'
        f'<div style="color:#8a93a6;font-size:14px">综合评分</div>'
        f'<div style="font-size:44px;font-weight:800;color:{lvl_color}">{advice.total_score:+d}</div>'
        f'<span class="pill" style="background:{lvl_color};font-size:16px;padding:6px 20px">'
        f'{advice.action}</span></div>',
        unsafe_allow_html=True,
    )
    rows = ""
    for s in advice.signals:
        _, c = signal_pill(s.score)
        dot = "🟢" if s.score > 0 else ("🔴" if s.score < 0 else "⚪")
        rows += (f'<div style="padding:8px 0;border-bottom:1px solid #1e2536">'
                 f'{dot} <b style="color:#e6e6e6">{s.name}</b> '
                 f'<span style="color:{c}">({s.score:+d})</span>'
                 f'<div class="tip-body">{s.detail}</div></div>')
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)
    st.caption("评分规则：各指标看多为正、看空为负；合计 ≥4 加仓，≤-4 减仓，其余持有观望。")
    explain("综合评分")

# ---- Tab 5 外围市场 ----
with tab_out:
    sent, _sent_status = _future_value("overseas")
    if sent is None:
        st.info(_sent_status)

    if sent is not None:
        lvl_color = {"bullish": UP, "bearish": DOWN, "neutral": FLAT}[sent.level]
        st.markdown(
            f'<div class="card" style="text-align:center">'
            f'<div style="color:#8a93a6;font-size:14px">外围情绪评分（对A股次日的参考）</div>'
            f'<div style="font-size:44px;font-weight:800;color:{lvl_color}">{sent.weighted_score:+.2f}</div>'
            f'<span class="pill" style="background:{lvl_color};font-size:16px;padding:6px 20px">'
            f'{sent.label}</span>'
            f'<div class="tip-body" style="margin-top:10px">{sent.summary}</div></div>',
            unsafe_allow_html=True,
        )

        rows = ""
        for m in sent.markets:
            if m.available:
                c = UP if m.score > 0 else (DOWN if m.score < 0 else FLAT)
                dot = "🟢" if m.score > 0 else ("🔴" if m.score < 0 else "⚪")
                rows += (
                    f'<div style="padding:10px 0;border-bottom:1px solid #1e2536;'
                    f'display:flex;justify-content:space-between">'
                    f'<span>{dot} <b style="color:#e6e6e6">{m.name}</b>'
                    f'<span style="color:#8a93a6;margin-left:10px">趋势 {m.trend}'
                    f'<span style="color:#5a6377">（{m.source}）</span></span></span>'
                    f'<span style="color:{c}">{m.last_close:,.2f}　'
                    f'{m.pct:+.2f}%（近3日 {m.cum3:+.2f}%）</span></div>'
                )
            else:
                rows += (
                    f'<div style="padding:10px 0;border-bottom:1px solid #1e2536;color:#8a93a6">'
                    f'⚪ <b>{m.name}</b>　数据不可用：{m.note}</div>'
                )
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)
        explain("外围市场")
        st.caption("说明：美股为隔夜收盘数据（只看美股三大指数）。外围情绪暂作独立参考，"
                   "下一步将与个股技术面、新闻情绪整合，形成次日涨跌预估。")

    # ---- 外围板块拆解 + 个股关联 ----
    st.markdown("---")
    st.markdown("#### 🧩 美股板块拆解与个股关联")
    link, _link_status = _future_value("linkage")
    if link is None:
        st.info(_link_status)

    if link is not None:
        # ============ 美股板块分析 ============
        st.markdown(
            f'<div class="card"><div class="tip-title">🌐 美股板块整体</div>'
            f'<div class="tip-body">{link.sector_summary}</div></div>',
            unsafe_allow_html=True,
        )
        us_cells = ""
        for s in link.all_sectors:
            r = s.regions.get("美股")
            if not (r and r.available):
                continue
            c = UP if r.score > 0 else (DOWN if r.score < 0 else FLAT)
            us_cells += (
                f'<div style="flex:0 0 31%;background:#0e1320;border:1px solid #1e2536;'
                f'border-radius:10px;padding:10px 12px;margin:4px 0.8%">'
                f'<div style="display:flex;justify-content:space-between">'
                f'<span style="color:#c9d1e0;font-weight:600">{s.name}</span>'
                f'<span style="color:{c};font-weight:700">{r.pct:+.2f}%</span></div>'
                f'<div style="color:#8a93a6;font-size:12px;margin-top:3px">趋势 {s.trend}</div></div>'
            )
        st.markdown(f'<div style="display:flex;flex-wrap:wrap" class="card">{us_cells}</div>',
                    unsafe_allow_html=True)
        with st.expander("📋 各板块代表龙头与主营业务（选取行业ETF前权重成分股）"):
            for s in link.all_sectors:
                us_qs = s.us_stocks()
                if not us_qs:
                    continue
                lines = ""
                for q in us_qs:
                    qc = UP if q.pct > 0 else (DOWN if q.pct < 0 else FLAT)
                    val = f'<span style="color:{qc}">{q.pct:+.2f}%</span>' if q.available else '<span style="color:#5a6377">—</span>'
                    lines += (f'<div style="padding:3px 0;font-size:13px">'
                              f'<b style="color:#e6e6e6">{q.name}</b> '
                              f'<span style="color:#8a93a6">{q.symbol} · {q.business}</span> {val}</div>')
                st.markdown(f'<div style="margin-bottom:8px"><b style="color:#c9d1e0">{s.name}</b>{lines}</div>',
                            unsafe_allow_html=True)

        # 个股主营 + 关联结论
        biz = link.business
        if biz:
            biz_html = (
                f'<b>主营业务：</b>{biz.get("主营业务","-")}<br>'
                f'<b>主要产品：</b>{biz.get("产品类型") or biz.get("产品名称") or "-"}'
            )
        else:
            biz_html = link.biz_note or "未获取到主营业务"

        matched_html = "".join(
            f'<span class="pill" style="background:#1c2740;color:#5b8cff;margin:2px 4px 2px 0">'
            f'{sec}</span>'
            for sec in link.matched
        ) or '<span style="color:#8a93a6">未匹配到明确板块</span>'
        engine_tag = ("🤖 AI识别" if link.map_engine == "AI" else "🔤 关键词识别")

        lvl_color = {"bullish": UP, "bearish": DOWN, "neutral": FLAT}[link.link_level]
        st.markdown(
            f'<div class="card">'
            f'<div class="tip-title">🔗 个股 → 美股板块 映射　'
            f'<span style="color:#5b8cff;font-size:13px">{engine_tag}</span></div>'
            f'<div class="tip-body">{biz_html}</div>'
            f'<div style="margin:10px 0">{matched_html}</div>'
            f'<div class="tip-body" style="border-top:1px solid #1e2536;padding-top:10px">'
            f'{link.link_conclusion}</div></div>',
            unsafe_allow_html=True,
        )
        explain("外围板块映射")
        st.caption("映射逻辑：提取个股主营/产品 → (有Qwen则AI识别概念/上下游，否则关键词) → "
                   "匹配美股行业板块 → 参考该板块隔夜涨跌评估情绪影响。仅供参考。")

# ---- Tab 6 新闻资讯 ----
with tab_news:
    nws, _news_status = _future_value("news")
    if nws is None:
        st.info(_news_status)

    if nws is not None:
        _hist_sentiment = _safe(lambda: sentiment_signal.score_at(symbol, latest["date"]))
        lvl_color = {"bullish": UP, "bearish": DOWN, "neutral": FLAT}[nws.overall_level]
        st.markdown(
            '<div class="card" style="padding:8px 16px">'
            '<span class="strip">📇 数据源：个股新闻/公告 <b>东方财富</b>　·　'
            '市场快讯 <b>新浪财经 + 财经早餐</b>　·　情绪引擎 '
            f'<b style="color:#5b8cff">{nws.engine}</b></span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card" style="text-align:center">'
            f'<div style="color:#8a93a6;font-size:14px">综合新闻情绪　'
            f'<span style="color:#5b8cff">引擎：{nws.engine}</span></div>'
            f'<div style="font-size:44px;font-weight:800;color:{lvl_color}">{nws.overall_score:+.2f}</div>'
            f'<span class="pill" style="background:{lvl_color};font-size:16px;padding:6px 20px">'
            f'{nws.overall_label}</span>'
            f'<div class="tip-body" style="margin-top:10px">{nws.conclusion}</div></div>',
            unsafe_allow_html=True,
        )

        # 板块新闻整体总结
        st.markdown(
            f'<div class="card"><div class="tip-title">📰 板块新闻整体总结</div>'
            f'<div class="tip-body">{nws.market_summary}</div></div>',
            unsafe_allow_html=True,
        )
        cells = ""
        for s in nws.sector_news:
            c = UP if s.score > 0.3 else (DOWN if s.score < -0.3 else FLAT)
            cells += (
                f'<div style="flex:0 0 31%;background:#0e1320;border:1px solid #1e2536;'
                f'border-radius:10px;padding:10px 12px;margin:4px 0.8%">'
                f'<div style="display:flex;justify-content:space-between">'
                f'<span style="color:#c9d1e0;font-weight:600">{s.name}</span>'
                f'<span style="color:{c};font-weight:700">{s.tone} {s.score:+.1f}</span></div>'
                f'<div style="color:#8a93a6;font-size:12px;margin-top:3px">{s.count} 条相关</div></div>'
            )
        if cells:
            st.markdown(f'<div style="display:flex;flex-wrap:wrap" class="card">{cells}</div>',
                        unsafe_allow_html=True)

        # 个股映射 A股行业/概念 新闻
        if nws.matched_sector_news:
            rows = ""
            for s in nws.matched_sector_news:
                c = UP if s.score > 0.3 else (DOWN if s.score < -0.3 else FLAT)
                samples = "".join(f'<div class="tip-body">· {t}</div>' for t in s.samples)
                rows += (f'<div style="padding:8px 0;border-bottom:1px solid #1e2536">'
                         f'<b style="color:#e6e6e6">{s.name}</b> '
                         f'<span style="color:{c}">{s.tone}({s.score:+.2f})</span>{samples}</div>')
            st.markdown(
                f'<div class="card"><div class="tip-title">🔗 个股对应A股行业/概念新闻</div>'
                f'<div class="tip-body">A股行业/概念：{"、".join(nws.matched_sectors) or "无"}</div>'
                f'{rows}</div>', unsafe_allow_html=True,
            )

        # 个股动作消息
        item_rows = ""
        for it in nws.stock_items:
            c = UP if it.sentiment > 0 else (DOWN if it.sentiment < 0 else FLAT)
            tag = "利好" if it.sentiment > 0 else ("利空" if it.sentiment < 0 else "中性")
            link_t = f'<a href="{it.url}" target="_blank" style="color:#c9d1e0;text-decoration:none">{it.title}</a>' if it.url else it.title
            item_rows += (
                f'<div style="padding:8px 0;border-bottom:1px solid #1e2536">'
                f'<span class="pill" style="background:{c};font-size:12px">{tag}</span> '
                f'{link_t}'
                f'<div style="color:#5a6377;font-size:12px">{it.time} · {it.source}</div></div>'
            )
        st.markdown(
            f'<div class="card"><div class="tip-title">📌 个股动作消息（新闻/公告）</div>'
            + (f'<div class="tip-body" style="margin-bottom:6px">{nws.stock_summary}</div>'
               if nws.stock_summary else "")
            + (item_rows or '<div class="tip-body">暂无个股新闻</div>')
            + '</div>', unsafe_allow_html=True,
        )
        explain("新闻资讯")
        _hist_note = getattr(_hist_sentiment, "note", "") if _hist_sentiment else ""
        _hist_state = "已纳入最终综合分" if (_hist_sentiment and _hist_sentiment.enabled) else "仅展示（留出验证未通过或尚未训练）"
        st.caption("实时数据源：个股新闻(东财) + 市场快讯(新浪/财经早餐)。")
        st.caption(f"历史舆情：{getattr(_hist_sentiment, 'model', '') or '未训练'}，"
                   f"衰减分 {getattr(_hist_sentiment, 'score', 0):+.3f}，"
                   f"文章 {getattr(_hist_sentiment, 'article_count', 0)} 条，{_hist_state}。 {_hist_note}")

# ---- Tab 7 基本面/资金面（券商 AmazingData）----
with tab_money:
    mflow, _money_status = _future_value("money")
    if mflow is None:
        st.info(_money_status)

    if mflow is not None and not mflow.available:
        st.warning(mflow.note)
    elif mflow is not None:
        lvl_color = {"bullish": UP, "bearish": DOWN, "neutral": FLAT}[mflow.level]
        st.markdown(
            f'<div class="card" style="text-align:center">'
            f'<div style="color:#8a93a6;font-size:14px">资金流向 综合（本地量价代理）</div>'
            f'<div style="font-size:44px;font-weight:800;color:{lvl_color}">{mflow.score:+.2f}</div>'
            f'<span class="pill" style="background:{lvl_color};font-size:16px;padding:6px 20px">'
            f'{mflow.label}</span></div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        _mf_proxy = bool(getattr(mflow, "is_proxy", False))
        c1.metric("当日净流入(量价代理)" if _mf_proxy else "当日主力净流入", fmt_num(mflow.net_main_1d, "元"))
        c2.metric("近5日净流入(量价代理)" if _mf_proxy else "近5日主力累计", fmt_num(mflow.net_main_5d, "元"))
        c3.metric("当日净占比(量价代理)" if _mf_proxy else "当日主力净占比", f"{mflow.ratio_main_1d:+.2f}%")
        rows = ""
        for s in mflow.signals:
            dot = "🟢" if s.score > 0 else ("🔴" if s.score < 0 else "⚪")
            c = UP if s.score > 0 else (DOWN if s.score < 0 else FLAT)
            rows += (f'<div style="padding:8px 0;border-bottom:1px solid #1e2536">'
                     f'{dot} <b style="color:#e6e6e6">{s.name}</b> '
                     f'<span style="color:{c}">({s.score:+d})</span>'
                     f'<div class="tip-body">{s.detail}</div></div>')
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)
        if mflow.note:
            st.caption(mflow.note)
        if mflow.recent is not None:
            with st.expander("📄 查看近5日资金流明细"):
                st.dataframe(mflow.recent, use_container_width=True, hide_index=True)
        st.caption("数据源：本地行情量价代理。按当日涨跌方向对成交额赋号近似资金净流入/流出（可正可负），"
                   "不请求东财主力资金流接口，且不等同于真实主力净流入。仅供参考。")

with tab_fund:
    fund, _fund_status = _future_value("broker")
    if fund is None:
        st.info(_fund_status)

    if fund is not None and not fund.available:
        st.warning(fund.note)
    elif fund is not None:
        lvl_color = {"bullish": UP, "bearish": DOWN, "neutral": FLAT}[fund.level]
        st.markdown(
            f'<div class="card" style="text-align:center">'
            f'<div style="color:#8a93a6;font-size:14px">基本面/资金面 综合（券商优先）</div>'
            f'<div style="font-size:44px;font-weight:800;color:{lvl_color}">{fund.score:+.2f}</div>'
            f'<span class="pill" style="background:{lvl_color};font-size:16px;padding:6px 20px">'
            f'{fund.label}</span></div>',
            unsafe_allow_html=True,
        )
        if fund.revenue_text:
            st.markdown(
                f'<div class="card"><div class="tip-title">💰 最新财务</div>'
                f'<div class="tip-body">{fund.revenue_text}</div></div>',
                unsafe_allow_html=True,
            )
        rows = ""
        for s in fund.signals:
            dot = "🟢" if s.score > 0 else ("🔴" if s.score < 0 else "⚪")
            c = UP if s.score > 0 else (DOWN if s.score < 0 else FLAT)
            rows += (f'<div style="padding:8px 0;border-bottom:1px solid #1e2536">'
                     f'{dot} <b style="color:#e6e6e6">{s.name}</b> '
                     f'<span style="color:{c}">({s.score:+d})</span>'
                     f'<div class="tip-body">{s.detail}</div></div>')
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)
        if fund.note:
            st.caption(fund.note)
        st.caption("数据源：银河 AmazingData 优先；券商空结果时最多重试 6 次、间隔 2s，仍为空则使用训练同源的东财数据中心快速兜底。"
                   "户数减少=筹码集中(偏多)；融资余额上升=杠杆加仓；龙虎榜净买入=资金流入。仅供参考。")

# ---- Tab 0 次日涨跌预估（多维整合 + Qwen 研判）----
with tab_pred:
    with st.spinner("正在汇总已就绪信号并研判…"):
        def _safe(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                return None
        p_sent = _done_value("overseas")
        p_link = _done_value("linkage")
        p_news = _done_value("news")
        p_mf = _done_value("money")
        p_fund = _done_value("broker")
        p_quant = _safe(lambda: quant_signal.get(symbol, profile=_quant_profile))
        p_sentiment = _safe(lambda: sentiment_signal.score_at(symbol, latest["date"]))
        _all_context_ready = all(x is not None for x in (p_sent, p_link, p_news, p_mf, p_fund))
        _pred_key = qwen_key if _all_context_ready else ""
        pred = prediction.predict(
            symbol, name, float(latest["close"]), change_pct,
            advice=advice, sent=p_sent, link=p_link, nws=p_news, fund=p_fund, mf=p_mf,
            quant=p_quant, tech_df=df, sentiment=p_sentiment,
            key=_pred_key, model=qwen_model, base_url=qwen_base)

    # 落盘当日多维快照（按交易日去重），为将来「完整预估」回测积累数据
    _snap_n = 0
    try:
        _record = {
            "date": latest["date"].strftime("%Y-%m-%d"),
            "close": round(float(latest["close"]), 2),
            "tech": advice.total_score,
            "overseas": round(p_sent.weighted_score, 2) if p_sent else None,
            "sector": round(p_link.link_score, 2) if p_link else None,
            "news": round(p_news.overall_score, 2) if p_news else None,
            "moneyflow": round(p_mf.score, 2) if (p_mf and getattr(p_mf, "available", False)) else None,
            "fund": round(p_fund.score, 2) if (p_fund and getattr(p_fund, "available", False)) else None,
            "quant_score": round(p_quant.score, 4) if p_quant else None,
            "quant_rank_pct": round(p_quant.rank_pct, 4) if p_quant else None,
            "quant_model": p_quant.model if p_quant else None,
            "sentiment_score": p_sentiment.score if (p_sentiment and p_sentiment.available) else None,
            "sentiment_model": p_sentiment.model if p_sentiment else None,
            "sentiment_count": p_sentiment.article_count if p_sentiment else 0,
            "pred_composite": pred.composite,
            "pred_level": pred.level,
            "engine": pred.engine,
        }
        _record.update(snapshot_batch._snapshot_news_fields(p_news, p_link))
        _snap_n = snapshot.save(symbol, _record)
    except Exception:  # noqa: BLE001
        pass

    lvl_color = {"bullish": UP, "bearish": DOWN, "neutral": FLAT}[pred.level]
    st.markdown(
        f'<div class="card" style="text-align:center">'
        f'<div style="color:#8a93a6;font-size:14px">次日涨跌研判　'
        f'<span style="color:#5b8cff">引擎：{pred.engine}</span></div>'
        f'<div style="font-size:40px;font-weight:800;color:{lvl_color}">{pred.direction}</div>'
        f'<span class="pill" style="background:{lvl_color};font-size:15px;padding:5px 18px">'
        f'信心 {pred.confidence}　·　综合分 {pred.composite:+.2f}</span></div>',
        unsafe_allow_html=True,
    )
    if pred.logic:
        st.markdown(f'<div class="card"><div class="tip-title">🧭 核心逻辑</div>'
                    f'<div class="tip-body">{pred.logic}</div></div>', unsafe_allow_html=True)
    if pred.risks:
        st.markdown(f'<div class="card"><div class="tip-title">⚠️ 主要风险</div>'
                    f'<div class="tip-body">{pred.risks}</div></div>', unsafe_allow_html=True)
    if pred.action:
        st.markdown(f'<div class="card"><div class="tip-title">🎯 操作建议</div>'
                    f'<div class="tip-body">{pred.action}</div></div>', unsafe_allow_html=True)
    with st.expander("📋 查看喂给模型的多维信号摘要"):
        st.text(pred.summary)
    st.caption("综合技术面 + 外围美股 + 板块联动 + 新闻情绪 + 资金流向 + 基本面/资金面。"
               "有 Qwen key 时由大模型研判，否则用规则法加权。仅为概率性参考，不构成投资建议。")
    st.caption(f"📦 已记录本股多维快照 {_snap_n} 天（每交易日一条，去重）。"
               "攒够后可在「选股/回测」用『完整预估快照回测』验证含消息面的胜率。")

# ---- Tab 8 选股/回测（多股票）----
with tab_screen:
    import re as _re
    import pandas as _pd

    # 量化预估（本次分析的3只股票，与下方技术面/Qwen回测并列参考）
    st.markdown("#### 🧮 量化模型预估")
    _bt_dual = _safe(lambda: quant_signal.trade_advice_for_codes(_main_codes))
    if _render_quant_advice_tables(_bt_dual, "当前股票无量化打分（需已发布 active 量化模型且股票在预测范围内）。", height=220):
        _asof = str(_bt_dual["date"].dropna().iloc[0]) if "date" in _bt_dual.columns and _bt_dual["date"].notna().any() else "-"
        st.caption(f"量化打分日期 **{_asof}**。短线=1-3天、波段=7-15天；两者使用不同量化口径和止盈目标，与下方技术面/Qwen规则回测供交叉参考。")
        _eval = _safe(lambda: quant_signal.selected_evaluation(profile=_quant_profile))
        _eval_show = _quant_eval_display(_eval)
        if not _eval_show.empty:
            st.markdown("#### 📊 当前量化档位白名单历史验证（动态周期）")
            st.dataframe(_eval_show, use_container_width=True, hide_index=True)
            st.caption("该表按评估文件中的 horizon 动态展示；后续加入 10/15/30 日，无需改 UI 列。")
    st.markdown("---")

    cand = [c for c in _re.split(r"[\s,，、;]+", pool_text or "") if c.strip()]
    if pool_industry:
        with st.spinner(f"正在获取「{pool_industry}」板块成分股…"):
            cand += screener.industry_constituents(pool_industry)
    cand = list(dict.fromkeys(data._normalize_symbol(c) for c in cand if c.strip()))
    st.markdown(f"**候选股票池：{len(cand)} 只**"
                + (f"（含行业「{pool_industry}」成分）" if pool_industry else ""))
    if not cand:
        st.info("在左侧「选股/回测 股票池」里粘贴代码(A)或填行业板块名(B)，再回到此页操作。")
    else:
        st.caption("提示：逐只拉取行情，股票多时较慢；结果缓存，重复查询会快。")
        _bt_hz = tuple(st.multiselect("回测周期", [1, 3, 5, 7, 10, 15, 30], default=[1, 3, 5, 7, 15],
                                      format_func=lambda h: f"{h}天")) or (1, 3, 5, 7, 15)
        _bt_hz_label = "/".join(str(h) for h in _bt_hz) + "天"
        c1, c2 = st.columns(2)
        do_score = c1.button("🧮 批量打分选股", use_container_width=True)
        do_bt = c2.button(f"📈 回测胜率(近3月·{_bt_hz_label})", use_container_width=True)
        bt_engine = st.radio("回测信号引擎", ["规则(技术面·快)", "Qwen(读技术面·慢·需key)"],
                             horizontal=True,
                             help="规则=加减仓评分；Qwen=大模型逐日读技术快照打分。Qwen 每个交易日一次调用，股票多会很慢且耗token，建议股票池小一些。")

        if do_score:
            with st.spinner(f"正在为 {len(cand)} 只股票打分…"):
                rows = screener.score_many(cand, profile=_quant_profile)
            ok = [r for r in rows if r.get("available")]
            if ok:
                dfp = _pd.DataFrame([{
                    "代码": r["code"], "名称": r["name"], "A股行业": r.get("a_industry") or r.get("industry", ""),
                    "A股概念": r.get("a_concepts") or r.get("sector", ""),
                    "外围映射": r.get("overseas_sector", ""),
                    "主营业务": ((str(r.get("business", ""))[:34] + "...") if len(str(r.get("business", ""))) > 36 else r.get("business", "")),
                    "融合评分": r.get("blended_score", r["score"]),
                    "技术评分": r["score"], "量化分位": r.get("quant_rank_pct"),
                    "量化排名": r.get("quant_rank"), "量化日期": r.get("quant_date"),
                    "参考价": r.get("quant_entry_price"), "止损价": r.get("quant_stop_loss"),
                    "止盈1": r.get("quant_take_profit_1"), "止盈2": r.get("quant_take_profit_2"),
                    "建议": r["action"], "最新价": r["close"], "涨跌%": r["pct"],
                } for r in ok])
                st.dataframe(dfp, use_container_width=True, hide_index=True)
                _cand_dual = _safe(lambda: quant_signal.trade_advice_for_codes(cand))
                st.markdown("**候选池量化建议**")
                _render_quant_advice_tables(_cand_dual, "候选池暂无量化打分。", height=260)
            bad = [r for r in rows if not r.get("available")]
            if bad:
                st.caption("未取到数据：" + "、".join(r["code"] for r in bad))
            st.caption("融合评分=技术面加减仓评分 + 量化模型截面分位加权；量化不可用时退化为技术评分。仅供参考，不构成投资建议。")

        if do_bt:
            _use_qwen = bt_engine.startswith("Qwen")
            _thr = 1 if _use_qwen else 3
            _eng = "qwen" if _use_qwen else "rule"
            _hz = _bt_hz
            with st.spinner(f"正在回测 {len(cand)} 只股票（{'Qwen' if _use_qwen else '规则'}，近3月，{_bt_hz_label}）…"):
                bts = backtest.run_many(cand, lookback=65, horizons=_hz, threshold=_thr,
                                        engine=_eng, key=qwen_key, model=qwen_model, base_url=qwen_base)
            valid = [b for b in bts if b.get("n_signals", 0) > 0]
            if valid:
                dfb_rows = []
                for b in valid:
                    row = {"代码": b["symbol"], "名称": b.get("name", ""), "信号数": b["n_signals"]}
                    row.update({f"胜率({h}天)%": b["winrate"].get(h) for h in _hz})
                    dfb_rows.append(row)
                dfb = _pd.DataFrame(dfb_rows)
                st.dataframe(dfb, use_container_width=True, hide_index=True)
                agg = {h: [b["winrate"][h] for b in valid if b["winrate"].get(h) is not None] for h in _hz}
                if any(agg[h] for h in _hz):
                    st.markdown(
                        f"**组合平均胜率**（{'Qwen' if _use_qwen else '规则'}）："
                        + "　·　".join(f"{h}天 {sum(agg[h])/len(agg[h]):.1f}%" for h in _hz if agg[h])
                        + f"（共 {sum(b['n_signals'] for b in valid)} 次信号）")
            else:
                st.warning("回测区间内无达到阈值的方向性信号。"
                           + ("（Qwen 引擎需先在侧边栏配置 API key）" if _use_qwen else ""))
            st.caption("回测信号：规则=加减仓评分（含MACD，阈值3、仅顺MA20方向）；Qwen=大模型读『截至当日』技术快照打分（因果、可回测）。"
                       f"检验信号后 {_bt_hz_label} 涨跌方向是否一致；无未来函数；新闻/外围因缺历史逐日快照未纳入。仅供参考。")

    # ---- 完整预估快照回测（含消息面，需已积累）----
    st.markdown("---")
    st.markdown(f"#### 📦 完整预估快照回测（当前股票 {symbol}）")
    if cand:
        if st.button("📸 批量记录股票池今日快照"):
            with st.spinner(f"正在为 {len(cand)} 只股票记录今日多维快照…"):
                sr = snapshot_batch.run(cand, key=qwen_key, model=qwen_model, base_url=qwen_base)
            st.success(f"已记录 {sr['count']} 只：{sr['ok']}"
                       + (f"；失败 {sr['fail']}" if sr["fail"] else ""))
    _snap_cnt = snapshot.count(symbol)
    st.caption(f"已积累 {_snap_cnt} 天多维快照（每次打开🔮次日预估自动记录一条）。"
               "含技术面+外围+新闻+基本面的综合研判，用历史快照 join 真实涨跌验证胜率——"
               "这是唯一能回测『消息面』的方式，需持续积累几天到几周才有样本。")
    _snap_hz = _bt_hz if "_bt_hz" in locals() else (1, 3, 5, 7)
    if st.button("运行快照回测（当前股票）"):
        sb = snapshot.backtest(symbol, horizons=_snap_hz, threshold=0.4)
        if sb["n_signals"] > 0:
            _snap_parts = [f"{h}天 {sb['winrate'].get(h)}%" for h in _snap_hz if sb["winrate"].get(h) is not None]
            st.markdown(f"**{symbol} 完整预估胜率**：" + "　·　".join(_snap_parts)
                        + f"（{sb['n_signals']} 次信号 / {sb['n_days']} 天快照）")
        else:
            st.info(sb["note"] or "样本不足")

_pending_tasks = [fut for task in list(_tasks.values()) + [_warmup_task or {}] for fut in _iter_task_futures(task) if not fut.done()]
if _pending_tasks:
    st.caption("后台接口仍在加载；页面不会等待慢接口。可在接口加载状态中点击刷新后台结果。")
