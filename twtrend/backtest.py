"""事件驅動回測。

- 訊號在 t 日收盤後產生，t+1 日開盤執行（無未來函數）
- 漲跌停鎖死無法成交：買單遇漲停鎖死放棄、賣單遇跌停鎖死順延到隔日
- 極端保護停損以盤中觸價模擬：成交價 = min(開盤, 停損價)
- 限價買單：開盤 > 限價就放棄、不追價
- 成本：手續費（可折扣）+ 證交稅 0.3% + 滑價
- 漲跌幅制度：2015-06-01 以前 7%，之後 10%

提供三個基準比較：0050 買進持有、只用 L1 操作 0050、完整系統。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import Config
from .data.market import MarketData
from .engine import SignalEngine
from .l1_regime import compute_regime, regime_index, switches_per_year
from .broker import REGIME_CHANGE_2015, SimBroker, limit_pct, locked  # noqa: F401

@dataclass
class BacktestResult:
    equity: pd.Series
    exposure: pd.Series
    trades: pd.DataFrame
    regime: pd.DataFrame
    decisions: list[dict] = field(default_factory=list)

    def metrics(self) -> dict:
        return performance(self.equity, self.exposure, self.regime)


def performance(equity: pd.Series, exposure: pd.Series | None = None,
                regime: pd.DataFrame | None = None) -> dict:
    eq = equity.dropna()
    ret = eq.pct_change().dropna()
    mdd = ind.max_drawdown(eq)
    cagr = ind.cagr(eq)
    out = {
        "CAGR": cagr,
        "MDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Sortino": ind.sortino(ret),
        "Vol": float(ret.std() * np.sqrt(252)),
    }
    if exposure is not None and regime is not None and "state" in regime:
        st = regime["state"].reindex(exposure.index)
        bull = st == "BULL"
        out["多頭參與率"] = float(exposure[bull].gt(0.05).mean()) if bull.any() else np.nan
        bear = st == "BEAR"
        out["空頭平均持股"] = float(exposure[bear].mean()) if bear.any() else np.nan
        out["每年狀態切換"] = switches_per_year(regime["state"])
    return out


def buy_and_hold(close: pd.Series) -> pd.Series:
    c = close.dropna()
    return c / c.iloc[0]


def l1_only(md: MarketData, cfg: Config, asset: pd.Series, regime: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """只用 L1 操作單一資產（0050）：每日以前一日的目標曝險持有，含調整成本。"""
    reg = regime if regime is not None else compute_regime(md, cfg.l1, regime_index(md, cfg.l1.index_source))
    a = asset.dropna()
    w = reg["target_exposure"].reindex(a.index).ffill().fillna(0).shift(1).fillna(0)
    r = a.pct_change().fillna(0)
    turnover = w.diff().abs().fillna(w.abs())
    c = cfg.cost
    cost = turnover * (c.commission * c.commission_discount + c.slippage) + w.diff().clip(upper=0).abs() * c.tax
    eq = (1 + w * r - cost).cumprod()
    return eq, w


class Backtester:
    """事件驅動回測：逐日呼叫 SimBroker（與模擬盤共用撮合規則）。"""

    def __init__(self, md: MarketData, cfg: Config, engine: SignalEngine | None = None):
        self.md = md
        self.cfg = cfg
        self.engine = engine or SignalEngine(md, cfg)

    def run(self, start=None, end=None) -> BacktestResult:
        dates = self.md.dates
        if start:
            dates = dates[dates >= pd.Timestamp(start)]
        if end:
            dates = dates[dates <= pd.Timestamp(end)]
        broker = SimBroker(self.md, self.cfg, self.engine)
        for i, d in enumerate(dates):
            broker.step(d, dates[i + 1] if i + 1 < len(dates) else None)
        # 期末未平倉以收盤估值納入交易統計（不影響權益曲線）
        trades = pd.DataFrame(broker.trades + broker.mark_open_trades(dates[-1]))
        if not trades.empty:
            trades["open_at_end"] = trades["open_at_end"].fillna(False).astype(bool)
            trades.loc[trades["open_at_end"], "exit_reason"] = "期末未平倉（以收盤估值）"
            trades.loc[trades["open_at_end"], "exit_date"] = dates[-1]
        equity = pd.Series(broker.equity)
        self.broker = broker
        return BacktestResult(equity, pd.Series(broker.exposure), trades,
                              self.engine.regime.reindex(equity.index), broker.decisions)


def compare_benchmarks(md: MarketData, cfg: Config, result: BacktestResult,
                       benchmark_close: pd.Series) -> pd.DataFrame:
    """三個基準：0050 買進持有、只用 L1 操作 0050、完整系統。"""
    idx = result.equity.index
    bh = buy_and_hold(benchmark_close.reindex(idx).ffill())
    l1eq, l1w = l1_only(md, cfg, benchmark_close.reindex(idx).ffill(), result.regime)
    rows = {
        "0050 買進持有": performance(bh),
        "只用 L1 操作 0050": performance(l1eq, l1w, result.regime),
        "完整系統": result.metrics(),
    }
    return pd.DataFrame(rows).T


def period_report(equity: pd.Series, benchmark: pd.Series, periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """歷次空頭測試期：系統與基準在各期間的報酬與回撤。"""
    rows = {}
    for name, (s, e) in periods.items():
        eq = equity.loc[s:e]
        bm = benchmark.loc[s:e].dropna()
        if len(eq) < 5 or len(bm) < 5:
            continue
        rows[name] = {
            "系統報酬": eq.iloc[-1] / eq.iloc[0] - 1, "系統MDD": ind.max_drawdown(eq),
            "基準報酬": bm.iloc[-1] / bm.iloc[0] - 1, "基準MDD": ind.max_drawdown(bm),
        }
    return pd.DataFrame(rows).T


BEAR_PERIODS = {
    "2000–2001 網路泡沫": ("2000-02-01", "2001-10-31"),
    "2007–2008 金融海嘯": ("2007-10-01", "2008-12-31"),
    "2011 歐債危機": ("2011-02-01", "2011-12-31"),
    "2015 中國股災": ("2015-04-01", "2015-09-30"),
    "2018 Q4 貿易戰": ("2018-10-01", "2018-12-31"),
    "2020/3 新冠疫情": ("2020-01-15", "2020-06-30"),
    "2022 升息空頭": ("2022-01-01", "2022-10-31"),
    "2025/4 關稅衝擊": ("2025-03-01", "2025-06-30"),
}


def split_2015(equity: pd.Series) -> pd.DataFrame:
    """漲跌幅制度改變前後分段報告。"""
    pre, post = equity.loc[:REGIME_CHANGE_2015], equity.loc[REGIME_CHANGE_2015:]
    rows = {}
    for name, e in (("2015/6 前（7%）", pre), ("2015/6 後（10%）", post)):
        if len(e) > 20:
            rows[name] = performance(e)
    return pd.DataFrame(rows).T
