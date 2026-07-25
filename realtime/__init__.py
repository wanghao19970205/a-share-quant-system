"""实时层：按选股清单订阅 Level-1 快照 → 异动/买卖点判定 → 推送 + 独立账本。

架构分层（先搭架子，策略后填）：
    config    运行配置（订阅来源、推送凭证、时段、规避窗）
    snapshot  Snapshot 数据模型 + 字段映射适配（吸收手册 vs 实际字段名差异）
    watchlist 订阅清单加载（选股清单 ∪ 持仓，去重夹紧）
    feed      订阅流封装（onSnapshot 真实流 / dry-run 假流，统一出 Snapshot）
    strategy  信号策略（可插拔；先放骨架 + 一个占位规则）
    notifier  推送通道（Server酱 / PushDeer，带节流）
    ledger    独立账本（JSONL，追加写，与 quant_data 隔离）
    engine    主循环 + 生命周期（时段/规避窗/心跳/优雅退出）

实际行情走 AmazingData onSnapshot 订阅流；query_snapshot 仅用于离线验证。
"""
from __future__ import annotations

__all__ = ["config", "snapshot", "watchlist", "feed", "strategy", "notifier", "ledger", "engine"]
