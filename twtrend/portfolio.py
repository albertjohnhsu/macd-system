"""L4 部位計算的基本元件：持股、委託單、停損與部位大小。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Position:
    stock_id: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float
    initial_stop: float
    one_r: float                       # 1R 金額 = 股數 × (進場價 − 初始停損)
    highest_close: float
    trailing_stop: float               # 只上移
    catastrophic_stop: float           # 極端保護停損價（條件單）
    industry: str = ""
    size_mult: float = 1.0
    entry_state: str = ""
    entry_trend_score: float = float("nan")
    entry_safety_score: float = float("nan")
    signal_date: pd.Timestamp | None = None
    signal_price: float = float("nan")
    pending_exit_since: pd.Timestamp | None = None
    entry_cost: float = 0.0
    manual: bool = False                                     # 手動下單（人為干預）
    health_weekly: dict = field(default_factory=dict)        # 最近一次每週健檢的每週項目 {項目: [狀態, 說明]}
    weak_seen: list = field(default_factory=list)            # 持有期間曾出現「減弱」的項目

    @property
    def stop(self) -> float:
        """有效停損 = max(初始停損, 追蹤停利)。"""
        return max(self.initial_stop, self.trailing_stop)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, pd.Timestamp):
                d[k] = v.strftime("%Y-%m-%d")
        return d


@dataclass
class Order:
    stock_id: str
    side: str                          # buy / sell
    shares: int
    signal_date: pd.Timestamp
    signal_price: float
    limit_price: float | None
    reason: str
    priority: int = 9
    meta: dict = field(default_factory=dict)


def initial_stop_price(close: pd.Series, low: pd.Series, atr_value: float,
                       atr_mult: float, swing_lookback: int) -> float:
    """初始停損：取「進場價 − k×ATR」與「近期波段低點下方」兩者中較遠（較低）者。"""
    px = float(close.iloc[-1])
    atr_stop = px - atr_mult * atr_value
    swing = float(low.iloc[-swing_lookback:].min()) * 0.995
    stop = min(atr_stop, swing)
    return max(stop, 0.01)


def chandelier(highest_close: float, atr_value: float, mult: float) -> float:
    return highest_close - mult * atr_value


def position_shares(equity: float, risk_per_trade: float, entry: float, stop: float,
                    max_value: float, size_mult: float = 1.0, adv: float | None = None,
                    max_adv_fraction: float = 0.05, available: float | None = None,
                    lot_size: int = 1) -> int:
    """風險平價部位：股數 = 每筆風險金額 ÷ 每股停損距離，再套各項上限。"""
    dist = entry - stop
    if dist <= 0 or entry <= 0:
        return 0
    risk_amount = equity * risk_per_trade * size_mult
    value = risk_amount / dist * entry
    caps = [value, max_value * size_mult]
    if adv is not None and not np.isnan(adv):
        caps.append(adv * max_adv_fraction)
    if available is not None:
        caps.append(max(available, 0.0))
    value = min(caps)
    shares = int(math.floor(value / entry / lot_size) * lot_size)
    return max(shares, 0)


def trade_cost(value: float, side: str, commission: float, discount: float, tax: float) -> float:
    fee = max(value * commission * discount, 20.0 if value > 0 else 0.0)   # 券商最低手續費 20 元
    return fee + (value * tax if side == "sell" else 0.0)
