"""模擬券商：回測與模擬盤共用的逐日撮合邏輯。

每個交易日分三段：
  1 開盤：執行前一日委託（限價不追價、漲停鎖死買不到、跌停鎖死賣不掉順延）
  2 盤中：極端保護停損條件單（成交價 = min(開盤, 停損價)）
  3 收盤：更新追蹤停利、記錄權益、產生隔日委託（週排名日才產生買單）

狀態可序列化成 JSON（to_state / from_state），模擬盤靠它在每日之間保存帳戶。
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import Config
from .data.market import MarketData
from .engine import SignalEngine
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


def is_rebalance_day(d: pd.Timestamp, next_date: pd.Timestamp | None) -> bool:
    """每週最後一個交易日收盤後重算排名；不知道下一個交易日時（模擬盤的今天）以下一個營業日判斷。"""
    nxt = next_date if next_date is not None else d + pd.offsets.BDay(1)
    return nxt.isocalendar()[1] != d.isocalendar()[1]


class SimBroker:
    def __init__(self, md: MarketData, cfg: Config, engine: SignalEngine, capital: float | None = None,
                 auto_signals: bool = True):
        self.md, self.cfg, self.engine = md, cfg, engine
        self.cash = float(capital if capital is not None else cfg.l4.total_capital)
        self.positions: dict[str, Position] = {}
        self.pending: list[Order] = []
        self.trades: list[dict] = []
        self.decisions: list[dict] = []
        self.signals: list[dict] = []           # 每週候選與買 / 不買原因
        self.health_log: list[dict] = []        # 每週持股健檢結果
        self.equity: dict[pd.Timestamp, float] = {}
        self.exposure: dict[pd.Timestamp, float] = {}
        self.auto_signals = auto_signals         # False = 純手動帳戶（只撮合、不產生系統訊號）
        self.last_date: pd.Timestamp | None = None

    # ------------------------------------------------------------ 逐日
    def step(self, d: pd.Timestamp, next_date: pd.Timestamp | None = None) -> dict:
        d = pd.Timestamp(d)
        filled = self._open_phase(d)
        filled += self._intraday_phase(d)
        new_orders = self._close_phase(d, next_date)
        self.last_date = d
        return {"date": d, "filled": filled, "orders": new_orders,
                "equity": self.equity[d], "cash": self.cash, "positions": len(self.positions)}

    def _px(self, d, sid) -> float:
        s = self.md.close.loc[:d, sid].dropna() if sid in self.md.close else pd.Series(dtype=float)
        return float(s.iloc[-1]) if len(s) else np.nan

    def _open_phase(self, d) -> list[dict]:
        md, cfg, c = self.md, self.cfg, self.cfg.cost
        carry, filled = [], []
        for o in self.pending:
            op = md.open.at[d, o.stock_id] if (o.stock_id in md.open and d in md.open.index) else np.nan
            if np.isnan(op):
                if o.side == "sell":
                    carry.append(o)
                continue
            if o.side == "sell":
                if o.stock_id not in self.positions:
                    continue
                if locked(md, d, o.stock_id, "down", cfg):
                    carry.append(o)                    # 跌停鎖死：隔日繼續掛
                    self._decide(d, o, "carry_sell", "跌停鎖死未成交，隔日續掛")
                    continue
                if o.limit_price is not None and op < o.limit_price:
                    self._decide(d, o, "skip_sell", f"開盤 {op:.2f} < 限價 {o.limit_price:.2f}")
                    continue
                fill = op * (1 - c.slippage)
                self.cash += self._close_position(self.positions.pop(o.stock_id), d, fill, o)
                filled.append({"stock_id": o.stock_id, "side": "sell", "price": fill, "reason": o.reason})
            else:
                if o.stock_id in self.positions:
                    continue
                if locked(md, d, o.stock_id, "up", cfg):
                    self._decide(d, o, "skip_buy", "漲停鎖死買不到")
                    continue
                if o.limit_price is not None and op > o.limit_price:
                    self._decide(d, o, "skip_buy", f"開盤 {op:.2f} > 限價 {o.limit_price:.2f}，不追價")
                    continue
                fill = op * (1 + c.slippage)
                value = o.shares * fill
                fee = trade_cost(value, "buy", c.commission, c.commission_discount, c.tax)
                if value + fee > self.cash:
                    self._decide(d, o, "skip_buy", "現金不足")
                    continue
                self.cash -= value + fee
                m = o.meta
                stop = min(m.get("stop", fill * 0.92), fill * 0.999)
                self.positions[o.stock_id] = Position(
                    o.stock_id, o.shares, d, fill, stop, o.shares * (fill - stop), fill, stop,
                    fill * (1 - cfg.l4.catastrophic_stop), m.get("industry", ""), m.get("size_mult", 1.0),
                    m.get("state", ""), m.get("trend_score", np.nan), m.get("safety_score", np.nan),
                    o.signal_date, o.signal_price, None, fee)
                self.positions[o.stock_id].manual = bool(m.get("manual"))
                filled.append({"stock_id": o.stock_id, "side": "buy", "price": fill, "reason": o.reason})
        self.pending = carry
        return filled

    def _intraday_phase(self, d) -> list[dict]:
        md, cfg, c = self.md, self.cfg, self.cfg.cost
        filled = []
        for sid in list(self.positions):
            p = self.positions[sid]
            lo = md.low.at[d, sid] if (sid in md.low and d in md.low.index) else np.nan
            if np.isnan(lo) or lo > p.catastrophic_stop or locked(md, d, sid, "down", cfg):
                continue
            fill = min(float(md.open.at[d, sid]), p.catastrophic_stop) * (1 - c.slippage)
            o = Order(sid, "sell", p.shares, d, p.catastrophic_stop, None, "極端保護停損（盤中條件單）", 2)
            self.cash += self._close_position(self.positions.pop(sid), d, fill, o)
            self.pending = [x for x in self.pending if x.stock_id != sid]
            filled.append({"stock_id": sid, "side": "sell", "price": fill, "reason": o.reason})
        return filled

    def _close_phase(self, d, next_date) -> list[Order]:
        eng, md = self.engine, self.md
        eng.update_stops(d, self.positions)
        mv = sum(p.shares * (self._px(d, s) if not np.isnan(self._px(d, s)) else p.entry_price)
                 for s, p in self.positions.items())
        equity = self.cash + mv
        self.equity[d] = equity
        self.exposure[d] = mv / equity if equity > 0 else 0.0
        if not self.auto_signals:
            return []
        weekly = is_rebalance_day(d, next_date)
        sells = eng.sell_signals(d, self.positions, equity, weekly=weekly)
        pending_ids = {x.stock_id for x in self.pending}
        new = [o for o in sells if o.stock_id not in pending_ids]
        self.pending.extend(new)
        for o in new:
            self._decide(d, o, "sell", o.reason)
        if weekly:
            for hr in eng.last_health.values():
                self.health_log.append(hr.to_row())
        sell_ids = {o.stock_id for o in self.pending if o.side == "sell"}
        if weekly:
            proceeds = sum(self.positions[s].shares * self._px(d, s) for s in sell_ids if s in self.positions)
            log: list[dict] = []
            buys = eng.buy_signals(d, self.positions, equity, self.cash + proceeds * 0.99, sell_ids, log=log)
            self.signals.extend(log)
            self.pending.extend(buys)
            for o in buys:
                self._decide(d, o, "buy", o.reason)
            new += buys
        return new

    # ------------------------------------------------------------ 手動下單
    def place_manual(self, d, sid: str, side: str, shares: int | None = None,
                     limit: float | None = None, note: str = "") -> Order:
        """手動委託（股市大富翁式）：隔日開盤撮合，標記為人為干預，績效分開統計。"""
        d = pd.Timestamp(d)
        px = self._px(d, sid)
        if np.isnan(px):
            raise KeyError(f"{sid} 在 {d.date()} 沒有收盤價")
        if side == "sell":
            if sid not in self.positions:
                raise KeyError(f"模擬帳戶沒有 {sid} 持股")
            shares = shares or self.positions[sid].shares
            o = Order(sid, "sell", shares, d, px, limit, f"手動賣出 {note}".strip(), 0, {"manual": True})
        else:
            if not shares:
                raise ValueError("手動買進需指定股數")
            info = self.md.info
            o = Order(sid, "buy", int(shares), d, px, limit, f"手動買進 {note}".strip(), 0, {
                "manual": True, "stop": self.engine.manual_stop(d, sid),
                "industry": info.at[sid, "industry"] if sid in info.index else "",
                "state": self.engine.regime_at(d).get("state", ""),
            })
        self.pending = [x for x in self.pending if not (x.stock_id == sid and x.side == side)]
        self.pending.append(o)
        self._decide(d, o, f"manual_{side}", o.reason)
        return o

    # ------------------------------------------------------------ 帳務
    def _decide(self, d, o: Order, action: str, reason: str):
        self.decisions.append({"date": pd.Timestamp(d), "stock_id": o.stock_id, "action": action,
                               "reason": reason, "manual": bool(o.meta.get("manual"))})

    def _close_position(self, p: Position, d, fill: float, o: Order, open_trade: bool = False) -> float:
        c = self.cfg.cost
        value = p.shares * fill
        fee = trade_cost(value, "sell", c.commission, c.commission_discount, c.tax)
        pnl = value - fee - p.shares * p.entry_price - p.entry_cost
        self.trades.append({
            "stock_id": p.stock_id, "signal_date": p.signal_date, "signal_price": p.signal_price,
            "entry_date": p.entry_date, "entry_price": p.entry_price, "shares": p.shares,
            "initial_stop": p.initial_stop, "one_r": p.one_r, "exit_date": pd.Timestamp(d), "exit_price": fill,
            "exit_signal_price": o.signal_price, "exit_reason": o.reason, "pnl": pnl,
            "r_multiple": pnl / p.one_r if p.one_r > 0 else np.nan,
            "entry_state": p.entry_state, "trend_score": p.entry_trend_score,
            "safety_score": p.entry_safety_score, "industry": p.industry, "open_at_end": open_trade,
            "holding_days": (pd.Timestamp(d) - pd.Timestamp(p.entry_date)).days,
            "manual_entry": bool(p.manual), "manual_exit": bool(o.meta.get("manual")),
            "weak_seen": "、".join(p.weak_seen or []),
            "exit_health_items": "、".join(o.meta.get("health_items", [])),
        })
        return value - fee

    def mark_open_trades(self, d) -> list[dict]:
        """以 d 收盤估算未平倉部位（不改變帳戶狀態），供統計用。"""
        out = []
        c = self.cfg.cost
        for s, p in self.positions.items():
            px = self._px(d, s)
            value = p.shares * px
            pnl = value - trade_cost(value, "sell", c.commission, c.commission_discount, c.tax) \
                - p.shares * p.entry_price - p.entry_cost
            out.append({"stock_id": s, "signal_date": p.signal_date, "signal_price": p.signal_price,
                        "entry_date": p.entry_date, "entry_price": p.entry_price, "shares": p.shares,
                        "initial_stop": p.initial_stop, "one_r": p.one_r, "exit_date": pd.Timestamp(d),
                        "exit_price": px, "exit_signal_price": px, "exit_reason": "未平倉（以收盤估值）",
                        "pnl": pnl, "r_multiple": pnl / p.one_r if p.one_r > 0 else np.nan,
                        "entry_state": p.entry_state, "trend_score": p.entry_trend_score,
                        "safety_score": p.entry_safety_score, "industry": p.industry, "open_at_end": True,
                        "holding_days": (pd.Timestamp(d) - pd.Timestamp(p.entry_date)).days,
                        "manual_entry": bool(p.manual), "manual_exit": False,
                        "weak_seen": "、".join(p.weak_seen or []), "exit_health_items": "",
                        "close": px, "stop": p.stop})
        return out

    # ------------------------------------------------------------ 序列化
    def to_state(self) -> dict:
        def pos(p: Position):
            return _jsonable(p.to_dict())

        def order(o: Order):
            return _jsonable(asdict(o))

        return {"cash": self.cash, "positions": [pos(p) for p in self.positions.values()],
                "pending": [order(o) for o in self.pending],
                "last_date": _d(self.last_date), "auto_signals": self.auto_signals}

    def load_state(self, st: dict) -> None:
        self.cash = float(st["cash"])
        self.auto_signals = bool(st.get("auto_signals", True))
        self.last_date = pd.Timestamp(st["last_date"]) if st.get("last_date") else None
        self.positions = {}
        for d in st.get("positions", []):
            for k in ("entry_date", "signal_date", "pending_exit_since"):
                d[k] = pd.Timestamp(d[k]) if d.get(k) else None
            for k in ("entry_trend_score", "entry_safety_score", "signal_price"):
                if d.get(k) is None:
                    d[k] = np.nan
            p = Position(**d)
            self.positions[p.stock_id] = p
        self.pending = []
        for d in st.get("pending", []):
            d["signal_date"] = pd.Timestamp(d["signal_date"])
            self.pending.append(Order(**d))


def _d(x):
    return pd.Timestamp(x).strftime("%Y-%m-%d") if x is not None and not pd.isna(x) else None


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
