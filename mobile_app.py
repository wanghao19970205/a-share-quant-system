"""轻量移动端排名浏览页。

运行：streamlit run mobile_app.py --server.port 8502
仅展示白名单、全 A 主板、创新药三个量化 Top10；大模型结果由 scheduler 预先缓存。
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from stock_analyzer import top10_eval


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
def _top_quant_frames() -> dict[str, pd.DataFrame]:
    return top10_eval.ranking_frames()


def _render_card(row: pd.Series, llm_row: dict | None):
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


st.markdown('<div class="hero"><h1>📊 A股 Top10</h1><p>量化结果浏览 · 仅供研究参考，不构成投资建议</p></div>', unsafe_allow_html=True)

frames = _top_quant_frames()
llm_cache = top10_eval.load()
labels = list(frames)
tab = st.tabs(labels)
for label, tab_item in zip(labels, tab):
    with tab_item:
        frame = frames[label]
        date = _text(frame["date"].iloc[0]) if not frame.empty and "date" in frame.columns else "暂无"
        st.caption(f"预测日期：{date}　|　量化模型：active_quant / short_1_3")
        if frame.empty:
            st.markdown('<div class="empty">暂无可用排名数据。请先完成量化训练或确认「创新药」板块映射可用。</div>', unsafe_allow_html=True)
            continue
        cache_entry = llm_cache.get(label, {})
        current_codes = frame["code"].astype(str).tolist()
        cache_valid = (
            cache_entry.get("cache_version") == top10_eval._CACHE_VERSION
            and cache_entry.get("date") == date
            and cache_entry.get("codes") == current_codes
            and cache_entry.get("fingerprint") == top10_eval._fingerprint(frame)
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
            _render_card(row, llm_rows.get(str(row.get("code"))))

st.caption("原始排序=量化融合分排序；大模型排序=方向×信心为主、综合分细分、量化分末位兜底。")
