"""L2 個股趨勢模組（依「策略檢視」修正）。

- gate（必要條件）：收盤 > MA100（或完整均線多頭排列）、12 個月報酬 > 0、
  不在過去 12 個月報酬最極端的前後 3%、收盤 ÷ 52 週高 ≥ 85%
- 評分：52 週高點接近度、月營收 YoY、3 個月相對強度、12 個月報酬，
  以全市場百分位轉成 0–100 後加權 → 趨勢分（Trend Score）
- 不使用橫斷面 6 個月動能前 30%（台股是公認例外）；ADX、量能、法人買超先不納入（待驗證）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import L2Config
from .data.market import MarketData, asof_panel


def compute_features(md: MarketData, cfg: L2Config) -> dict[str, pd.DataFrame]:
    """預先算好整段期間的特徵寬表（回測時只算一次）。"""
    c = md.close
    f: dict[str, pd.DataFrame] = {}
    if cfg.use_full_ma_stack:
        m20, m60, m120 = ind.sma(c, 20), ind.sma(c, 60), ind.sma(c, 120)
        f["gate_ma"] = (c > m20) & (m20 > m60) & (m60 > m120) & (m60 > m60.shift(5))
    else:
        f["gate_ma"] = c > ind.sma(c, cfg.gate_ma)
    f["tsmom_12m"] = ind.total_return(c, cfg.tsmom_lookback)
    f["high_52w_ratio"] = ind.high_52w_ratio(c, 252)
    idx_ret = ind.total_return(md.index_close.reindex(c.index).ffill(), cfg.rel_strength_lookback)
    f["rel_strength"] = ind.total_return(c, cfg.rel_strength_lookback).sub(idx_ret, axis=0)
    f["revenue_yoy"] = asof_panel(md.revenue, "yoy", c.index, list(c.columns))
    f["tradable"] = md.tradable_mask()
    return f


def evaluate_trend(md: MarketData, date: pd.Timestamp, cfg: L2Config,
                   universe: list[str] | None = None,
                   features: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """在 date 對 universe（通常是 L3 通過的股票池）計算趨勢分與排名。"""
    date = pd.Timestamp(date)
    f = features or compute_features(md, cfg)
    row = {k: v.loc[:date].iloc[-1] for k, v in f.items() if len(v.loc[:date])}
    tradable = row["tradable"].fillna(False).astype(bool)
    df = pd.DataFrame({k: row[k] for k in ("tsmom_12m", "high_52w_ratio", "rel_strength", "revenue_yoy")})
    df["gate_ma"] = row["gate_ma"].fillna(False).astype(bool)

    # 極端報酬排除：全市場（可交易股）12 個月報酬的前後 3%
    r12 = df["tsmom_12m"].where(tradable)
    lo, hi = r12.quantile(cfg.extreme_return_pct), r12.quantile(1 - cfg.extreme_return_pct)
    df["extreme"] = (r12 < lo) | (r12 > hi)

    # 全市場百分位（0–100）
    pct = pd.DataFrame({k: df[k].where(tradable).rank(pct=True) * 100 for k in cfg.weights if k in df})
    w = pd.Series(cfg.weights)
    num = (pct * w).sum(axis=1, min_count=1)
    den = pct.notna().mul(w).sum(axis=1).replace(0, np.nan)
    df["raw_score"] = num / den

    df["gate"] = (tradable & df["gate_ma"] & (df["tsmom_12m"] > 0) & ~df["extreme"]
                  & (df["high_52w_ratio"] >= cfg.high_52w_min))
    if universe is not None:
        df["in_universe"] = df.index.isin(universe)
    else:
        df["in_universe"] = True
    eligible = df["gate"] & df["in_universe"]
    df["trend_score"] = df["raw_score"].where(eligible)
    df["rank"] = df["trend_score"].rank(ascending=False, method="first")
    n = eligible.sum()
    # 名次百分位：1.0 = 第一名
    df["rank_pct"] = 1 - (df["rank"] - 1) / max(n, 1)
    df["buy_zone"] = df["rank_pct"] > (1 - cfg.buy_top_pct)
    df["hold_zone"] = df["rank_pct"] > (1 - cfg.hold_top_pct)
    return df.sort_values("trend_score", ascending=False)
