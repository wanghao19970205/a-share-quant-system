"""轻量移动端排名浏览页。

运行：streamlit run mobile_app.py --server.port 8502
仅展示白名单、全 A 主板、创新药三个量化 Top10；大模型结果由 scheduler 预先缓存。
"""
from __future__ import annotations

import html
import json
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_analyzer import advisor, data, indicators, top10_eval


st.set_page_config(page_title="A股 Top10", page_icon="📊", layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {max-width: 720px; padding: 1rem .8rem 3rem;}
    [data-testid="stAppViewContainer"] {background: #0b1018;}
    .hero {padding: .4rem .2rem 1rem;}
    .hero h1 {font-size: 1.65rem; margin: 0; color: #f4f7fb;}
    .hero p {color: #8793a7; margin: .35rem 0 0; font-size: .82rem;}
    .stock {background: #131b27; border: 1px solid #263246; border-radius: 10px;
            padding: .85rem .9rem; margin: .55rem 0;}
    .stock-head {display:flex; justify-content:space-between; gap:.6rem; align-items:center;}
    .identity {min-width:0;}
    .rank {color:#efb84b; font-size:1.15rem; font-weight:700; margin-right:.45rem;}
    .name {color:#f3f6fa; font-size:1.02rem; font-weight:700;}
    .code {color:#8b98ab; font-size:.78rem; margin-left:.35rem;}
    .tag {color:#a8c7ff; background:#1b3155; border-radius:5px; padding:.2rem .42rem;
          font-size:.72rem; white-space:nowrap;}
    .meta {color:#96a3b6; font-size:.78rem; line-height:1.55; margin-top:.55rem;}
    .meta b {color:#dce5f2; font-weight:600;}
    .logic {border-top:1px solid #263246; color:#c7d0dd; font-size:.8rem;
            line-height:1.55; margin-top:.6rem; padding-top:.55rem;}
    .logic strong {color:#efb84b;}
    .eval-grid {display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.35rem; margin-top:.65rem;}
    .eval-cell {background:#182231; border:1px solid #2b3a50; border-radius:6px; padding:.35rem .25rem;
                 text-align:center; min-width:0;}
    .eval-name {color:#9eabc0; font-size:.65rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .eval-score {font-size:.85rem; font-weight:700; margin-top:.15rem;}
    .eval-detail {color:#7f8da2; font-size:.62rem; line-height:1.3; margin-top:.15rem;}
    .chart-caption {color:#8f9caf; font-size:.7rem; margin:.25rem 0 .45rem;}
    @media (max-width: 420px) {.eval-grid {grid-template-columns:repeat(3, minmax(0, 1fr));}}
    .empty {color:#91a0b5; background:#131b27; border:1px solid #263246;
            border-radius:10px; padding:1rem;}
    .stButton button {border-radius:7px;}
    </style>
    """,
    unsafe_allow_html=True,
)

def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _text(value, fallback="-"):
    text = "" if value is None else str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "<na>"} else fallback


@st.cache_data(ttl=300, show_spinner=False)
def _mobile_snapshot() -> dict:
    path = top10_eval.mobile_snapshot_path()
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _top_quant_frames() -> dict[str, pd.DataFrame]:
    return top10_eval.ranking_frames()


@st.cache_data(ttl=300, show_spinner=False)
def _price_chart(code: str, freshness_bucket: int) -> pd.DataFrame:
    frame = data.fetch_daily(code, days=90, freshness_bucket=freshness_bucket)
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = indicators.add_ma(frame, windows=(5, 10, 20))
    return frame.tail(60).copy()


def _advice_details(code: str, llm_row: dict) -> tuple[object, list[dict]]:
    cached = llm_row.get("advice_signals") or []
    if cached:
        return llm_row.get("advice_total_score"), cached
    frame = _price_chart(code, int(time.time() // 300))
    if frame.empty:
        return None, []
    result = advisor.advise(indicators.compute_all(frame))
    return result.total_score, [
        {"name": signal.name, "score": signal.score, "detail": signal.detail}
        for signal in result.signals
    ]


def _render_eval_grid(signals: list[dict]) -> None:
    if not signals:
        st.caption("暂无技术子评估明细")
        return
    cells = []
    for signal in signals:
        score = pd.to_numeric(pd.Series([signal.get("score")]), errors="coerce").iloc[0]
        if pd.isna(score):
            score_text, color = "-", "#9eabc0"
        elif score > 0:
            score_text, color = f"+{int(score)}", "#ef6b72"
        elif score < 0:
            score_text, color = str(int(score)), "#55c79a"
        else:
            score_text, color = "0", "#9eabc0"
        cells.append(
            f'<div class="eval-cell" title="{html.escape(_text(signal.get("detail")))}">'
            f'<div class="eval-name">{html.escape(_text(signal.get("name")))}</div>'
            f'<div class="eval-score" style="color:{color}">{html.escape(score_text)}</div>'
            f'</div>'
        )
    st.markdown('<div class="eval-grid">' + "".join(cells) + '</div>', unsafe_allow_html=True)


def _render_moneyflow_chart(code: str, chart_key: str, chart_data: list[dict] | None) -> None:
    frame = pd.DataFrame(chart_data or [])
    if frame.empty:
        st.caption("暂无近14日资金流数据")
        return
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["net_amount"] = pd.to_numeric(frame["net_amount"], errors="coerce")
    frame = frame.dropna(subset=["date", "net_amount"]).sort_values("date").tail(14)
    if frame.empty:
        st.caption("暂无近14日资金流数据")
        return
    values = frame["net_amount"] / 1e8
    colors = ["#ef6b72" if value >= 0 else "#55c79a" for value in values]
    fig = go.Figure(go.Bar(
        x=frame["date"], y=values, name="资金净额代理",
        marker={"color": colors},
        hovertemplate="%{x|%m-%d}<br>%{y:+.2f}亿元<extra></extra>",
    ))
    fig.add_hline(y=0, line={"color": "#637086", "width": 1})
    fig.update_layout(
        height=210, margin={"l": 4, "r": 4, "t": 8, "b": 4},
        paper_bgcolor="#131b27", plot_bgcolor="#131b27",
        font={"color": "#aab6c8", "size": 10}, showlegend=False,
        xaxis={"showgrid": False, "tickformat": "%m-%d"},
        yaxis={"showgrid": True, "gridcolor": "#263246", "side": "right", "ticksuffix": "亿"},
        bargap=0.25,
    )
    st.plotly_chart(
        fig, use_container_width=True, config={"displayModeBar": False},
        key=f"moneyflow-chart-{chart_key}-{code}",
    )


def _render_price_chart(
    code: str,
    chart_key: str,
    chart_data: list[dict] | None = None,
) -> None:
    if chart_data:
        frame = pd.DataFrame(chart_data)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    else:
        bucket = int(time.time() // 300)
        frame = _price_chart(code, bucket)
    if frame.empty:
        st.caption("暂无曲线数据")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["close"], name="实盘曲线", line={"color": "#f2f5f8", "width": 2}))
    latest = frame.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[latest["date"]], y=[latest["close"]], name="今日实盘",
        mode="markers", marker={"color": "#ff6b6b", "size": 8},
    ))
    colors = {5: "#efb84b", 10: "#65a8ff", 20: "#d58cff"}
    for window, color in colors.items():
        fig.add_trace(go.Scatter(x=frame["date"], y=frame[f"ma{window}"], name=f"{window}日均线", line={"color": color, "width": 1.2}))
    fig.update_layout(
        height=230, margin={"l": 0, "r": 0, "t": 8, "b": 0},
        paper_bgcolor="#131b27", plot_bgcolor="#131b27",
        font={"color": "#aeb9ca", "size": 10},
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"size": 9}},
        xaxis={"showgrid": False, "showline": False},
        yaxis={"showgrid": True, "gridcolor": "#263246", "side": "right"},
        hovermode="x unified",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key=f"price-chart-{chart_key}-{code}",
    )


def _render_card(row: pd.Series, llm_row: dict | None, group_key: str = "default"):
    code = _text(row.get("code"))
    name = _text(row.get("name"), code)
    watch_rank = row.get("watch_rank")
    quant_rank = _text(watch_rank if watch_rank is not None and not pd.isna(watch_rank) else row.get("rank"))
    concepts = _text(row.get("a_concepts"))
    if concepts == "-":
        concepts = _text(row.get("sector"))
    industry = _text(row.get("a_industry"))
    if industry == "-":
        industry = _text(row.get("industry"))
    quant_score = row.get("pred")
    try:
        quant_score = f"{float(quant_score):+.3f}"
    except Exception:
        quant_score = "-"
    if llm_row:
        llm_rank = _text(llm_row.get("llm_rank"))
        llm_summary = f"{_text(llm_row.get('direction'))} / {_text(llm_row.get('confidence'))} / 综合分 {_text(llm_row.get('composite'))}"
        logic = _text(llm_row.get("logic"), "大模型未返回核心逻辑")
        entry = _text(llm_row.get("entry_price"))
        take_profit_1 = _text(llm_row.get("take_profit_1"))
        take_profit_2 = _text(llm_row.get("take_profit_2"))
        stop_loss = _text(llm_row.get("stop_loss"))
        action = _text(llm_row.get("action"), "-")
    else:
        llm_rank, llm_summary, logic = "-", "缓存暂未生成", "量化模型发布后，后台会自动生成大模型综合研判。"
        entry, take_profit_1, take_profit_2, stop_loss, action = "-", "-", "-", "-", "-"
    st.markdown(
        f'<article class="stock"><div class="stock-head"><div class="identity">'
        f'<span class="rank">量化 {html.escape(quant_rank)}</span>'
        f'<span class="name">{html.escape(name)}</span><span class="code">{html.escape(code)}</span>'
        f'</div><span class="tag">大模型 {html.escape(llm_rank)}</span></div>'
        f'<div class="meta"><b>概念</b> {html.escape(concepts)}<br>'
        f'<b>行业</b> {html.escape(industry)}　<b>量化分</b> {html.escape(quant_score)}<br>'
        f'<b>大模型</b> {html.escape(llm_summary)}<br>'
        f'<b>参考买入</b> {html.escape(entry)}　<b>止盈</b> {html.escape(take_profit_1)} / {html.escape(take_profit_2)}　'
        f'<b>止损</b> {html.escape(stop_loss)}</div>'
        f'<div class="logic"><strong>核心逻辑</strong>　{html.escape(logic)}<br>'
        f'<strong>操作建议</strong>　{html.escape(action)}</div></article>',
        unsafe_allow_html=True,
    )
    if llm_row:
        total, signals = _advice_details(code, llm_row)
        st.caption(f"加减仓子评估 · 综合分 {_text(total, '-')}")
        _render_eval_grid(signals)
        with st.expander("资金流与均线", expanded=False):
            st.caption("近14个交易日资金流量变化 · 本地量价代理，红色流入、绿色流出")
            _render_moneyflow_chart(code, group_key, llm_row.get("moneyflow_data"))
            st.caption("最近 60 个交易日 · 收盘价与简单移动平均线")
            _render_price_chart(code, group_key, llm_row.get("chart_data"))


st.markdown('<div class="hero"><h1>📊 A股 Top10</h1><p>量化结果浏览 · 仅供研究参考，不构成投资建议</p></div>', unsafe_allow_html=True)

snapshot = _mobile_snapshot()
if snapshot.get("groups"):
    frames = {
        label: pd.DataFrame(group.get("rows") or [])
        for label, group in snapshot["groups"].items()
    }
    llm_cache = {
        label: group.get("llm") or {}
        for label, group in snapshot["groups"].items()
    }
else:
    frames = _top_quant_frames()
    llm_cache = top10_eval.load()
labels = list(frames)
tab = st.tabs(labels)
for label, tab_item in zip(labels, tab):
    with tab_item:
        frame = frames[label]
        date = _text(frame["date"].iloc[0]) if not frame.empty and "date" in frame.columns else "暂无"
        model_meta = snapshot.get("model") or {}
        published_at = _text(model_meta.get("published_at"), "未知")
        source_job = _text(model_meta.get("job"), "top10-eval")
        job_label = {
            "daily-light": "日更任务",
            "intraday-light": "轻量日更任务",
            "top10-eval": "Top10评估",
        }.get(source_job, source_job)
        st.caption(
            f"预测日期：{date}　|　量化模型：active_quant / short_1_3　| "
            f"产出：{published_at}　|　来源：{job_label}"
        )
        if frame.empty:
            st.markdown('<div class="empty">暂无可用排名数据。请先完成量化训练或确认「创新药」板块映射可用。</div>', unsafe_allow_html=True)
            continue
        cache_entry = llm_cache.get(label, {})
        current_codes = frame["code"].astype(str).tolist()
        if snapshot.get("groups"):
            cache_valid = bool(cache_entry)
        else:
            quant_fingerprint = cache_entry.get("quant_fingerprint")
            cache_valid = (
                cache_entry.get("cache_version") == top10_eval._CACHE_VERSION
                and cache_entry.get("date") == date
                and cache_entry.get("codes") == current_codes
                and len(cache_entry.get("rows") or []) == len(current_codes)
                and (
                    quant_fingerprint is None
                    or quant_fingerprint == top10_eval._fingerprint(frame)
                )
            )
        if not cache_valid:
            cache_entry = {}
        llm_rows = {
            str(row.get("code")): row
            for row in cache_entry.get("rows", [])
            if row.get("available") and not row.get("stale")
        }
        if cache_entry:
            st.caption(f"大模型：{_text(cache_entry.get('model'))}　|　缓存：{_text(cache_entry.get('updated_at'))}")
        else:
            st.caption("大模型缓存尚未生成，量化模型下次发布后自动评估。")
        for _, row in frame.iterrows():
            _render_card(row, llm_rows.get(str(row.get("code"))), group_key=label)

st.caption("原始排序=量化融合分排序；大模型排序=方向×信心为主、综合分细分、量化分末位兜底。")
