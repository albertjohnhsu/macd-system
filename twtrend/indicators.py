"""技術指標（皆為向量化、無未來函數：t 日的值只用到 t 日以前的資料）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series | pd.DataFrame, n: int):
    return s.rolling(n, min_periods=n).mean()


def slope(s: pd.Series | pd.DataFrame, n: int):
    """n 日變化率，作為均線斜率的簡化代理。"""
    return s / s.shift(n) - 1.0


def true_range(high, low, close):
    prev = close.shift(1)
    if isinstance(close, pd.DataFrame):
        a, b, c = high - low, (high - prev).abs(), (low - prev).abs()
        return np.maximum(np.maximum(a, b), c)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high, low, close, n: int = 14):
    """Wilder ATR。"""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def total_return(close, n: int, skip: int = 0):
    """過去 n 日報酬，可跳過最近 skip 日（如 6 個月動能跳過 1 個月）。"""
    return close.shift(skip) / close.shift(n) - 1.0


def high_52w_ratio(close, n: int = 252):
    return close / close.rolling(n, min_periods=int(n * 0.8)).max()


def rolling_max_since(series: pd.Series, start_idx) -> float:
    return float(series.loc[start_idx:].max())


def idiosyncratic_vol(stock_ret: pd.DataFrame, mkt_ret: pd.Series, n: int = 60) -> pd.DataFrame:
    """對大盤迴歸後的殘差波動度（滾動 n 日）。"""
    mkt = mkt_ret.reindex(stock_ret.index)
    cov = stock_ret.rolling(n, min_periods=n).cov(mkt)
    var = mkt.rolling(n, min_periods=n).var()
    beta = cov.div(var, axis=0)
    resid = stock_ret - beta.mul(mkt, axis=0)
    return resid.rolling(n, min_periods=n).std() * np.sqrt(252)


def max_daily_return(ret: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Bali et al. (2011) MAX：近 n 日最大單日漲幅。"""
    return ret.rolling(n, min_periods=n).max()


def consecutive_limit_up(close: pd.DataFrame, limit_pct: float = 0.10, tol: float = 0.005) -> pd.DataFrame:
    """連續漲停天數（漲幅 ≥ 漲停 − 容差 視為漲停）。"""
    ret = close.pct_change()
    hit = ret >= (limit_pct - tol)
    grp = (~hit).cumsum()
    return hit.astype(int).groupby(grp).cumsum() if isinstance(hit, pd.Series) else \
        hit.apply(lambda c: c.astype(int).groupby((~c).cumsum()).cumsum())


def percentile_rank(df: pd.DataFrame) -> pd.DataFrame:
    """橫斷面（每一列）百分位，0–100。"""
    return df.rank(axis=1, pct=True) * 100.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / periods_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sortino(ret: pd.Series, periods_per_year: int = 252) -> float:
    downside = ret[ret < 0].std()
    if not downside or np.isnan(downside):
        return float("nan")
    return float(ret.mean() / downside * np.sqrt(periods_per_year))
