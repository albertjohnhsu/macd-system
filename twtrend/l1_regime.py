"""L1 大盤多空判斷模組：多指標合計分數 + 四狀態機（含遲滯、確認期、分批調整）。

L1 是總開關，輸出「持股上限」(cap) 與分批後的「目標曝險」(target_exposure)。
個股訊號再強，總曝險也不得超過 target_exposure。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import L1Config
from .data.market import MarketData

BULL, WARNING, BEAR, RECOVERY = "BULL", "WARNING", "BEAR", "RECOVERY"
STATE_ZH = {BULL: "多頭", WARNING: "轉弱警戒", BEAR: "空頭", RECOVERY: "打底轉強"}


def regime_index(md: MarketData, source: str = "taiex", exclude: tuple[str, ...] = ("2330",)) -> pd.Series:
    """L1 判斷用的指數。

    - taiex：加權指數
    - equal_weight：股票池等權重指數（失效情境優化 #1：避免被台積電扭曲）
    - ex_tsmc：扣除 exclude 後的市值加權指數
    """
    if source == "taiex":
        return md.index_close.dropna()
    ret = md.returns().where(md.tradable_mask())
    if source == "equal_weight":
        r = ret.mean(axis=1, skipna=True)
    elif source == "ex_tsmc":
        cap = md.market_cap().shift(1).where(md.tradable_mask())
        cap = cap.drop(columns=[c for c in exclude if c in cap.columns])
        r = (ret[cap.columns] * cap).sum(axis=1) / cap.sum(axis=1).replace(0, np.nan)
    else:
        raise ValueError(f"未知的 index source: {source}")
    return (1 + r.fillna(0)).cumprod() * 100


def breadth(md: MarketData, ma: int = 60) -> pd.Series:
    """站上 ma 日線的股票比例。"""
    mask = md.tradable_mask()
    m = ind.sma(md.close, ma)
    above = (md.close > m) & m.notna() & mask
    valid = m.notna() & mask
    return above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)


def new_high_minus_low(md: MarketData, n: int = 252, smooth: int = 10) -> pd.Series:
    mask = md.tradable_mask()
    hi = md.close.rolling(n, min_periods=int(n * 0.8)).max()
    lo = md.close.rolling(n, min_periods=int(n * 0.8)).min()
    nh = ((md.close >= hi) & mask).sum(axis=1)
    nl = ((md.close <= lo) & mask).sum(axis=1)
    return (nh - nl).rolling(smooth, min_periods=1).sum()


def compute_indicators(md: MarketData, cfg: L1Config, index: pd.Series | None = None) -> pd.DataFrame:
    """回傳每日各指標是否為多頭條件（1/0，資料不足為 NaN）。"""
    idx = (index if index is not None else regime_index(md, cfg.index_source)).dropna()
    dates = idx.index
    out = pd.DataFrame(index=dates)

    long_ma = ind.sma(idx, cfg.long_ma)
    above = (idx > long_ma).astype(float).where(long_ma.notna())
    # 確認期：連續 confirm_days 日在均線之上 / 之下才改變判斷
    n = cfg.confirm_days
    up_conf = above.rolling(n, min_periods=n).min() == 1
    dn_conf = above.rolling(n, min_periods=n).max() == 0
    confirmed = pd.Series(np.where(up_conf, 1.0, np.where(dn_conf, 0.0, np.nan)), index=dates)
    confirmed = confirmed.ffill().where(long_ma.notna())
    out["above_long_ma"] = confirmed

    mid = ind.sma(idx, cfg.mid_ma)
    out["mid_ma_slope"] = (ind.slope(mid, cfg.slope_lookback) > 0).astype(float).where(mid.shift(cfg.slope_lookback).notna())
    tsm = ind.total_return(idx, cfg.tsmom_lookback)
    out["tsmom_12m"] = (tsm > 0).astype(float).where(tsm.notna())

    br = breadth(md, cfg.breadth_ma).reindex(dates)
    # 背離：指數 60 日內創高，但寬度低於 20 日前
    idx_new_high = idx >= idx.rolling(60, min_periods=20).max()
    divergence = idx_new_high & (br < br.shift(20))
    out["breadth_above_ma"] = ((br > cfg.breadth_threshold) & ~divergence).astype(float).where(br.notna())
    out["breadth"] = br

    nhl = new_high_minus_low(md).reindex(dates)
    out["new_high_minus_low"] = (nhl > 0).astype(float).where(br.notna())

    if md.futures_net_oi is not None and not md.futures_net_oi.empty:
        oi = md.futures_net_oi.reindex(dates).ffill()
        chg = oi - oi.shift(20)
        # 淨空單未持續擴大：淨部位為多，或 20 日內淨空單沒有增加
        out["futures_net_short_not_rising"] = ((oi >= 0) | (chg >= 0)).astype(float).where(chg.notna())

    if md.margin_market is not None and not md.margin_market.empty:
        mm = md.margin_market.reindex(dates).ffill()
        ok = pd.Series(True, index=dates)
        valid = pd.Series(False, index=dates)
        if "margin_balance" in mm:
            g = mm["margin_balance"] / mm["margin_balance"].shift(20) - 1
            ok &= ~(g > 0.10)            # 融資 20 日暴增 10% 視為過熱
            valid |= g.notna()
        if "maintenance_ratio" in mm:
            mr = mm["maintenance_ratio"]
            ok &= ~(mr < 1.50)           # 大盤維持率 < 150% 進入斷頭警戒區
            valid |= mr.notna()
        out["margin_not_overheated"] = ok.astype(float).where(valid)

    if md.macro is not None and not md.macro.empty:
        mac = md.macro.copy()
        mac["announce_date"] = pd.to_datetime(mac["announce_date"])
        mac = mac.sort_values("announce_date").set_index("announce_date")
        mac = mac[~mac.index.duplicated(keep="last")]
        if "leading_index" in mac:
            li_up = (mac["leading_index"].diff() > 0).astype(float).where(mac["leading_index"].diff().notna())
            out["leading_indicator_up"] = li_up.reindex(li_up.index.union(dates)).ffill().reindex(dates)
        if {"m1b_yoy", "m2_yoy"} <= set(mac.columns):
            gold = (mac["m1b_yoy"] > mac["m2_yoy"]).astype(float).where(mac["m1b_yoy"].notna())
            out["m1b_above_m2"] = gold.reindex(gold.index.union(dates)).ffill().reindex(dates)

    out["index"] = idx
    out["long_ma"] = long_ma
    return out


def score(ind_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """加權分數 0–100，只計入當日有資料的指標。"""
    cols = [c for c in weights if c in ind_df]
    w = pd.Series({c: weights[c] for c in cols})
    vals = ind_df[cols]
    num = (vals * w).sum(axis=1, min_count=1)
    den = vals.notna().mul(w).sum(axis=1).replace(0, np.nan)
    return num / den * 100


def run_state_machine(ind_df: pd.DataFrame, sc: pd.Series, cfg: L1Config) -> pd.DataFrame:
    states: list[str | None] = []
    br = ind_df["breadth"]
    above = ind_df["above_long_ma"]
    br_min = br.rolling(cfg.recovery_window, min_periods=1).min()
    prev: str | None = None
    for i, d in enumerate(ind_df.index):
        s, a, b = sc.iloc[i], above.iloc[i], br.iloc[i]
        if np.isnan(s) or np.isnan(a):
            states.append(prev)
            continue
        is_above = a == 1
        thrust = (not np.isnan(b)) and br_min.iloc[i] < cfg.recovery_breadth_from and b > cfg.recovery_breadth_to
        if prev is None:
            if is_above and s >= cfg.enter_bull:
                st = BULL
            elif not is_above and s < cfg.bear_threshold:
                st = BEAR
            else:
                st = WARNING
        elif prev == BULL:
            if s < cfg.exit_bull or not is_above:
                st = BEAR if (not is_above and s < cfg.bear_threshold) else WARNING
            else:
                st = BULL
        elif prev == WARNING:
            if is_above and s >= cfg.enter_bull:
                st = BULL
            elif not is_above and s < cfg.bear_threshold:
                st = BEAR
            else:
                st = WARNING
        elif prev == BEAR:
            if is_above and s >= cfg.enter_bull:
                st = BULL
            elif thrust or s >= cfg.recovery_enter:
                st = RECOVERY
            else:
                st = BEAR
        else:  # RECOVERY
            if is_above and s >= cfg.enter_bull:
                st = BULL
            elif is_above:
                st = WARNING
            elif s < cfg.recovery_exit and not thrust:
                st = BEAR
            else:
                st = RECOVERY
        states.append(st)
        prev = st
    return pd.Series(states, index=ind_df.index, dtype=object)


def compute_regime(md: MarketData, cfg: L1Config, index: pd.Series | None = None,
                   step_interval: int = 5) -> pd.DataFrame:
    """完整 L1 輸出：指標、分數、狀態、持股上限、分批目標曝險。"""
    idx = index if index is not None else regime_index(md, cfg.index_source)
    ind_df = compute_indicators(md, cfg, idx)
    sc = score(ind_df, cfg.weights)
    state = run_state_machine(ind_df, sc, cfg)
    caps = {BULL: cfg.cap_bull, WARNING: cfg.cap_warning, BEAR: cfg.cap_bear, RECOVERY: cfg.cap_recovery}
    cap = state.map(caps).astype(float)

    if cfg.use_multi_horizon:
        idx_ = ind_df["index"]
        votes = pd.concat([(ind.total_return(idx_, h) > 0).astype(float) for h in cfg.multi_horizons], axis=1)
        mh = votes.mean(axis=1)                      # 0 / 1/3 / 2/3 / 1
        cap = pd.Series(np.round(mh * 3) / 3, index=ind_df.index).where(state.notna())

    # 分批調整：上限改變當日先調一批，之後每 step_interval 日再調一批（每批最多 max_step_per_rebalance）
    target = []
    cur, prev_cap, since = np.nan, np.nan, 0
    for c in cap.values:
        if np.isnan(c):
            target.append(np.nan)
            continue
        since += 1
        if np.isnan(cur):
            cur = c
        elif abs(c - cur) > 1e-9 and (c != prev_cap or since >= step_interval):
            step = cfg.max_step_per_rebalance
            cur = cur + float(np.clip(c - cur, -step, step))
            since = 0
        prev_cap = c
        target.append(cur)
    out = ind_df.copy()
    out["score"] = sc
    out["state"] = state
    out["cap"] = cap
    out["target_exposure"] = target
    return out


def switches_per_year(state: pd.Series) -> float:
    s = state.dropna()
    if len(s) < 2:
        return 0.0
    n = (s != s.shift()).sum() - 1
    years = len(s) / 252
    return float(n / years) if years > 0 else float(n)
