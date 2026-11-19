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
from .portfolio import Order, Position, trade_cost

REGIME_CHANGE_2015 = pd.Timestamp("2015-06-01")


def limit_pct(date: pd.Timestamp, cfg: Config) -> float:
    return cfg.cost.limit_pct_before_2015 if date < REGIME_CHANGE_2015 else cfg.cost.limit_pct_after_2015


def locked(md: MarketData, date: pd.Timestamp, sid: str, direction: str, cfg: Config) -> bool:
    """整日鎖在漲停（up）或跌停（down）：最高 = 最低，且漲跌幅接近上限。"""
    try:
        h, l, c = md.high.at[date, sid], md.low.at[date, sid], md.close.at[date, sid]
        prev = md.close.loc[:date, sid].iloc[-2]
    except (KeyError, IndexError):
        return False
    if np.isnan(h) or np.isnan(prev) or abs(h - l) > 1e-9:
        return False
    chg = c / prev - 1
    lp = limit_pct(date, cfg) - 0.005
    return chg >= lp if direction == "up" else chg <= -lp


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
    def __init__(self, md: MarketData, cfg: Config, engine: SignalEngine | None = None):
        self.md = md
        self.cfg = cfg
        self.engine = engine or SignalEngine(md, cfg)

    def run(self, start=None, end=None) -> BacktestResult:
        md, cfg, eng = self.md, self.cfg, self.engine
        dates = md.dates
        dates = dates[(dates >= pd.Timestamp(start)) if start else slice(None)]
        if end:
            dates = dates[dates <= pd.Timestamp(end)]
        c = cfg.cost
        cash = float(cfg.l4.total_capital)
        positions: dict[str, Position] = {}
        pending: list[Order] = []
        trades: list[dict] = []
        decisions: list[dict] = []
        eq_hist, exp_hist = {}, {}

        def mark(d):
            v = 0.0
            for s, p in positions.items():
                px = md.close.loc[:d, s].dropna()
                v += p.shares * (float(px.iloc[-1]) if len(px) else p.entry_price)
            return v

        for i, d in enumerate(dates):
            # ---- 1. 開盤：執行前一日產生的委託
            carry: list[Order] = []
            for o in pending:
                op = md.open.at[d, o.stock_id] if o.stock_id in md.open else np.nan
                if np.isnan(op):
                    if o.side == "sell":
                        carry.append(o)
                    continue
                if o.side == "sell":
                    if o.stock_id not in positions:
                        continue
                    if locked(md, d, o.stock_id, "down", cfg):
                        carry.append(o)       # 跌停鎖死：隔日繼續掛
                        continue
                    fill = op * (1 - c.slippage)
                    cash += self._close_position(positions.pop(o.stock_id), d, fill, o, trades)
                else:
                    if o.stock_id in positions or locked(md, d, o.stock_id, "up", cfg):
                        continue
                    if o.limit_price is not None and op > o.limit_price:
                        decisions.append({"date": d, "stock_id": o.stock_id, "action": "skip_buy",
                                          "reason": f"開盤 {op:.2f} > 限價 {o.limit_price:.2f}，不追價"})
                        continue
                    fill = op * (1 + c.slippage)
                    value = o.shares * fill
                    fee = trade_cost(value, "buy", c.commission, c.commission_discount, c.tax)
                    if value + fee > cash:
                        continue
                    cash -= value + fee
                    m = o.meta
                    stop = min(m["stop"], fill * 0.999)
                    positions[o.stock_id] = Position(
                        o.stock_id, o.shares, d, fill, stop, o.shares * (fill - stop), fill, stop,
                        fill * (1 - cfg.l4.catastrophic_stop), m.get("industry", ""), m.get("size_mult", 1.0),
                        m.get("state", ""), m.get("trend_score", np.nan), m.get("safety_score", np.nan),
                        o.signal_date, o.signal_price, None, fee)
            pending = carry

            # ---- 2. 盤中：極端保護停損條件單
            for sid in list(positions):
                p = positions[sid]
                lo = md.low.at[d, sid] if sid in md.low else np.nan
                if np.isnan(lo) or lo > p.catastrophic_stop:
                    continue
                if locked(md, d, sid, "down", cfg):
                    continue
                fill = min(float(md.open.at[d, sid]), p.catastrophic_stop) * (1 - c.slippage)
                o = Order(sid, "sell", p.shares, d, p.catastrophic_stop, None, "極端保護停損（盤中條件單）", 2)
                cash += self._close_position(positions.pop(sid), d, fill, o, trades)
                pending = [x for x in pending if x.stock_id != sid]

            # ---- 3. 收盤：更新停損、產生訊號
            eng.update_stops(d, positions)
            equity = cash + mark(d)
            eq_hist[d] = equity
            exp_hist[d] = (equity - cash) / equity if equity > 0 else 0.0
            weekly = self._is_rebalance_day(dates, i)
            sells = eng.sell_signals(d, positions, equity, weekly=weekly)
            sell_ids = {o.stock_id for o in sells} | {o.stock_id for o in pending if o.side == "sell"}
            pending.extend(o for o in sells if o.stock_id not in {x.stock_id for x in pending})
            for o in sells:
                decisions.append({"date": d, "stock_id": o.stock_id, "action": "sell", "reason": o.reason})
            if weekly:
                proceeds = sum(positions[s].shares * md.close.at[d, s] for s in sell_ids if s in positions)
                buys = eng.buy_signals(d, positions, equity, cash + proceeds * 0.99, sell_ids)
                pending.extend(buys)
                for o in buys:
                    decisions.append({"date": d, "stock_id": o.stock_id, "action": "buy", "reason": o.reason})

        # 期末平倉（只為統計交易，不計入權益曲線）
        last = dates[-1]
        for sid, p in list(positions.items()):
            px = float(md.close.loc[:last, sid].dropna().iloc[-1])
            o = Order(sid, "sell", p.shares, last, px, None, "期末未平倉（以收盤估值）", 99)
            self._close_position(p, last, px, o, trades, open_trade=True)

        equity = pd.Series(eq_hist)
        exposure = pd.Series(exp_hist)
        return BacktestResult(equity, exposure, pd.DataFrame(trades), eng.regime.reindex(equity.index), decisions)

    def _is_rebalance_day(self, dates: pd.DatetimeIndex, i: int) -> bool:
        """每週最後一個交易日（預設週五）收盤後重算排名。"""
        if i == len(dates) - 1:
            return True
        return dates[i + 1].isocalendar()[1] != dates[i].isocalendar()[1]

    def _close_position(self, p: Position, d, fill: float, o: Order, trades: list, open_trade: bool = False) -> float:
        c = self.cfg.cost
        value = p.shares * fill
        fee = trade_cost(value, "sell", c.commission, c.commission_discount, c.tax)
        pnl = value - fee - p.shares * p.entry_price - p.entry_cost
        trades.append({
            "stock_id": p.stock_id, "signal_date": p.signal_date, "signal_price": p.signal_price,
            "entry_date": p.entry_date, "entry_price": p.entry_price, "shares": p.shares,
            "initial_stop": p.initial_stop, "one_r": p.one_r, "exit_date": d, "exit_price": fill,
            "exit_signal_price": o.signal_price, "exit_reason": o.reason, "pnl": pnl,
            "r_multiple": pnl / p.one_r if p.one_r > 0 else np.nan,
            "entry_state": p.entry_state, "trend_score": p.entry_trend_score,
            "safety_score": p.entry_safety_score, "industry": p.industry, "open_at_end": open_trade,
            "holding_days": (pd.Timestamp(d) - pd.Timestamp(p.entry_date)).days,
        })
        return value - fee


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
