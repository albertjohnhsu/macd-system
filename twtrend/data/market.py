"""MarketData：所有判斷層共用的資料容器。

日頻資料是寬表（index=日期、columns=股票代碼）；財報、營收、公告類是長表，
透過 `asof_panel` 以「公告日」對齊成寬表，保證 t 日只看得到 t 日以前已公告的資料。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)


@dataclass
class MarketData:
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    open: pd.DataFrame
    volume: pd.DataFrame
    turnover: pd.DataFrame
    index_close: pd.Series                     # 加權指數
    info: pd.DataFrame                         # index=stock_id: name, industry, market, listed_date, delisted_date, shares
    raw_close: pd.DataFrame | None = None      # 未還原收盤，判斷漲跌停
    revenue: pd.DataFrame = field(default_factory=lambda: _empty(["stock_id", "announce_date", "yoy"]))
    financials: pd.DataFrame = field(default_factory=lambda: _empty(["stock_id", "announce_date"]))
    governance: pd.DataFrame = field(default_factory=lambda: _empty(["stock_id", "announce_date", "pledge_ratio"]))
    events: pd.DataFrame = field(default_factory=lambda: _empty(["stock_id", "date", "event_type"]))
    audit: pd.DataFrame = field(default_factory=lambda: _empty(["stock_id", "announce_date", "opinion"]))
    flags: pd.DataFrame = field(default_factory=lambda: _empty(["stock_id", "flag_type", "start_date", "end_date"]))
    margin_market: pd.DataFrame = field(default_factory=lambda: _empty(["margin_balance", "maintenance_ratio"]))
    margin_usage: pd.DataFrame | None = None   # 個股融資使用率（寬表）
    institutional: pd.DataFrame | None = None  # 投信+外資淨買超（寬表）
    futures_net_oi: pd.Series | None = None    # 外資台指期淨未平倉
    macro: pd.DataFrame = field(default_factory=lambda: _empty(["announce_date", "leading_index", "m1b_yoy", "m2_yoy"]))

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def stocks(self) -> list[str]:
        return list(self.close.columns)

    def returns(self) -> pd.DataFrame:
        return self.close.pct_change(fill_method=None)

    def market_cap(self) -> pd.DataFrame:
        shares = self.info["shares"].reindex(self.close.columns).astype(float)
        px = self.raw_close if self.raw_close is not None else self.close
        return px.mul(shares, axis=1)

    def tradable_mask(self) -> pd.DataFrame:
        """已上市且未下市的日期為 True。"""
        mask = pd.DataFrame(True, index=self.dates, columns=self.close.columns)
        for sid in self.close.columns:
            row = self.info.loc[sid] if sid in self.info.index else None
            if row is None:
                continue
            if pd.notna(row.get("listed_date")):
                mask.loc[mask.index < pd.Timestamp(row["listed_date"]), sid] = False
            if pd.notna(row.get("delisted_date")):
                mask.loc[mask.index >= pd.Timestamp(row["delisted_date"]), sid] = False
        return mask & self.close.notna()

    def slice(self, end: pd.Timestamp) -> "MarketData":
        """只保留 end（含）以前的資料：用於逐日模擬時防止未來函數。"""
        end = pd.Timestamp(end)

        def cut(x):
            if x is None:
                return None
            return x.loc[:end]

        def cut_long(x, col="announce_date"):
            if x is None or x.empty or col not in x:
                return x
            return x[pd.to_datetime(x[col]) <= end]

        return MarketData(
            close=cut(self.close), high=cut(self.high), low=cut(self.low), open=cut(self.open),
            volume=cut(self.volume), turnover=cut(self.turnover), index_close=cut(self.index_close),
            info=self.info, raw_close=cut(self.raw_close),
            revenue=cut_long(self.revenue), financials=cut_long(self.financials),
            governance=cut_long(self.governance), events=cut_long(self.events, "date"),
            audit=cut_long(self.audit), flags=cut_long(self.flags, "start_date"),
            margin_market=cut(self.margin_market), margin_usage=cut(self.margin_usage),
            institutional=cut(self.institutional), futures_net_oi=cut(self.futures_net_oi),
            macro=cut_long(self.macro),
        )


def asof_panel(long_df: pd.DataFrame, value_col: str, dates: pd.DatetimeIndex,
               stocks: list[str], date_col: str = "announce_date") -> pd.DataFrame:
    """把長表依公告日向前填補成寬表：t 日的值 = t 日（含）以前最後一次公告的值。"""
    out = pd.DataFrame(np.nan, index=dates, columns=stocks)
    if long_df is None or long_df.empty or value_col not in long_df:
        return out
    df = long_df[["stock_id", date_col, value_col]].dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).drop_duplicates(["stock_id", date_col], keep="last")
    wide = df.pivot(index=date_col, columns="stock_id", values=value_col)
    wide = wide.reindex(columns=stocks)
    union = wide.index.union(dates)
    return wide.reindex(union).ffill().reindex(dates)


def active_flags(flags: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """date 當日仍生效的安全旗標。"""
    if flags is None or flags.empty:
        return flags
    d = pd.Timestamp(date)
    start = pd.to_datetime(flags["start_date"])
    end = pd.to_datetime(flags["end_date"])
    return flags[(start <= d) & (end.isna() | (end >= d))]
