"""L3 安全性篩選：硬性排除 + 風險評分（0–100，越高越安全）。

在 L2 趨勢評分「之前」執行：硬性排除任一成立就不得買入；
安全分 ≥70 完整部位、50–69 部位減半、<50 不得買入（持有中者進入待出場）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import L3Config
from .data.market import MarketData

HARD_FLAG_TYPES = {
    "full_delivery": "全額交割",
    "suspended": "停止買賣",
    "delisting_warning": "下市警示",
    "disposal": "處置股",
}
ATTENTION = "attention"


# ---------------------------------------------------------------- 財報快照
def _ttm(g: pd.DataFrame, col: str, offset: int = 0) -> float:
    vals = g[col].iloc[len(g) - 4 - offset: len(g) - offset] if len(g) >= 4 + offset else None
    if vals is None or vals.isna().any():
        return np.nan
    return float(vals.sum())


def _latest(g: pd.DataFrame, col: str, offset: int = 0) -> float:
    if len(g) <= offset or col not in g:
        return np.nan
    v = g[col].iloc[-1 - offset]
    return float(v) if pd.notna(v) else np.nan


def _safe_div(a, b):
    return a / b if (b is not None and not np.isnan(b) and b != 0 and not np.isnan(a)) else np.nan


def piotroski(g: pd.DataFrame) -> float:
    """Piotroski F-Score（9 項）；需要至少 8 季資料，缺項不計分。"""
    if len(g) < 8:
        return np.nan
    ta, ta_p = _latest(g, "total_assets"), _latest(g, "total_assets", 4)
    ni, ni_p = _ttm(g, "net_income"), _ttm(g, "net_income", 4)
    ocf = _ttm(g, "operating_cf")
    roa, roa_p = _safe_div(ni, ta), _safe_div(ni_p, ta_p)
    lev = _safe_div(_latest(g, "long_term_debt"), ta)
    lev_p = _safe_div(_latest(g, "long_term_debt", 4), ta_p)
    cr = _safe_div(_latest(g, "current_assets"), _latest(g, "current_liabilities"))
    cr_p = _safe_div(_latest(g, "current_assets", 4), _latest(g, "current_liabilities", 4))
    sh, sh_p = _latest(g, "shares"), _latest(g, "shares", 4)
    rev, rev_p = _ttm(g, "revenue"), _ttm(g, "revenue", 4)
    gm = _safe_div(_ttm(g, "gross_profit"), rev)
    gm_p = _safe_div(_ttm(g, "gross_profit", 4), rev_p)
    at, at_p = _safe_div(rev, ta), _safe_div(rev_p, ta_p)
    tests = [
        roa > 0, ocf > 0, roa > roa_p, ocf > ni, lev <= lev_p, cr > cr_p,
        sh <= sh_p, gm > gm_p, at > at_p,
    ]
    return float(sum(bool(t) for t in tests if t is not np.nan))


def altman_z(g: pd.DataFrame, market_cap: float) -> float:
    ta = _latest(g, "total_assets")
    tl = _latest(g, "total_liabilities")
    wc = _latest(g, "current_assets") - _latest(g, "current_liabilities")
    re_ = _latest(g, "retained_earnings")
    ebit = _ttm(g, "ebit")
    sales = _ttm(g, "revenue")
    parts = [_safe_div(wc, ta), _safe_div(re_, ta), _safe_div(ebit, ta), _safe_div(market_cap, tl), _safe_div(sales, ta)]
    if any(np.isnan(p) for p in parts):
        return np.nan
    return 1.2 * parts[0] + 1.4 * parts[1] + 3.3 * parts[2] + 0.6 * parts[3] + 1.0 * parts[4]


def fundamental_snapshot(fin: pd.DataFrame, date: pd.Timestamp, mcap: pd.Series) -> pd.DataFrame:
    """以公告日 ≤ date 的財報計算各項財務指標。"""
    cols = ["fscore", "zscore", "current_ratio", "interest_cover", "accrual", "profit_neg_ocf_3y",
            "non_op_ratio", "eps_4q", "bvps"]
    if fin is None or fin.empty:
        return pd.DataFrame(columns=cols)
    f = fin.copy()
    f["announce_date"] = pd.to_datetime(f["announce_date"])
    f = f[f["announce_date"] <= date].sort_values(["stock_id", "period"])
    rows = {}
    for sid, g in f.groupby("stock_id"):
        g = g.tail(12).reset_index(drop=True)
        ta = _latest(g, "total_assets")
        ni, ocf = _ttm(g, "net_income"), _ttm(g, "operating_cf")
        pretax = _ttm(g, "pretax_income")
        non_op = _ttm(g, "non_operating_income")
        neg3 = np.nan
        if len(g) >= 12:
            yrs = [(g["net_income"].iloc[i:i + 4].sum(), g["operating_cf"].iloc[i:i + 4].sum()) for i in (0, 4, 8)]
            neg3 = float(all(n > 0 for n, _ in yrs) and all(o < 0 for _, o in yrs))
        rows[sid] = {
            "fscore": piotroski(g),
            "zscore": altman_z(g, float(mcap.get(sid, np.nan))) if len(g) >= 4 else np.nan,
            "current_ratio": _safe_div(_latest(g, "current_assets"), _latest(g, "current_liabilities")),
            "interest_cover": _safe_div(_ttm(g, "ebit"), _ttm(g, "interest_expense")),
            "accrual": _safe_div(ni - ocf, ta) if not (np.isnan(ni) or np.isnan(ocf)) else np.nan,
            "profit_neg_ocf_3y": neg3,
            "non_op_ratio": _safe_div(non_op, pretax) if pretax and pretax > 0 else np.nan,
            "eps_4q": _ttm(g, "eps"),
            "bvps": _latest(g, "bvps"),
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def _asof_long(df: pd.DataFrame, date: pd.Timestamp, cols: list[str], date_col: str = "announce_date") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d[d[date_col] <= date].sort_values(date_col)
    return d.groupby("stock_id").last().reindex(columns=cols)


def _event_counts(events: pd.DataFrame, date: pd.Timestamp, types: set[str], years: float) -> pd.Series:
    if events is None or events.empty:
        return pd.Series(dtype=float)
    e = events if pd.api.types.is_datetime64_any_dtype(events["date"]) else events.assign(date=pd.to_datetime(events["date"]))
    e = e[(e["date"] <= date) & (e["date"] > date - pd.Timedelta(days=int(365 * years))) & e["event_type"].isin(types)]
    return e.groupby("stock_id").size().astype(float)


# ---------------------------------------------------------------- 主函式
class SafetyEvaluator:
    """預先計算日頻面板，並快取「只在公告時才變動」的財報 / 治理快照，讓逐日回測可行。"""

    def __init__(self, md: MarketData, cfg: L3Config, price_features: dict[str, pd.DataFrame] | None = None):
        self.md, self.cfg = md, cfg
        self.stocks = md.stocks
        self.pf = price_features or compute_price_features(md)
        self.mcap = md.market_cap()
        self.to60 = md.turnover.rolling(60, min_periods=20).mean()
        self.info = md.info.reindex(self.stocks)
        self.listed = pd.to_datetime(self.info["listed_date"])
        self.exempt = self.info["industry"].isin(cfg.exempt_industries)
        self._fin_dates = _sorted_dates(md.financials)
        self._gov_dates = _sorted_dates(md.governance)
        self._aud_dates = _sorted_dates(md.audit)
        self._flags = None
        if md.flags is not None and not md.flags.empty:
            f = md.flags.copy()
            f["start"] = pd.to_datetime(f["start_date"])
            f["end"] = pd.to_datetime(f["end_date"])
            self._flags = f
        self._events = None
        if md.events is not None and not md.events.empty:
            self._events = md.events.assign(date=pd.to_datetime(md.events["date"]))
        self._fund_cache: dict = {}
        self._gov_cache: dict = {}
        self._aud_cache: dict = {}

    def _row(self, panel: pd.DataFrame | None, date) -> pd.Series:
        if panel is None or panel.empty:
            return pd.Series(np.nan, index=self.stocks)
        h = panel.loc[:date]
        return h.iloc[-1].reindex(self.stocks) if len(h) else pd.Series(np.nan, index=self.stocks)

    def _cached(self, cache, dates, date, fn):
        key = _last_le(dates, date)
        if key not in cache:
            cache[key] = fn(key if key is not None else date)
        return cache[key]

    def evaluate(self, date) -> pd.DataFrame:
        date = pd.Timestamp(date)
        cfg, md, stocks = self.cfg, self.md, self.stocks
        out = pd.DataFrame(index=pd.Index(stocks, name="stock_id"))
        reasons: dict[str, list[str]] = {s: [] for s in stocks}

        def add(mask: pd.Series, text: str):
            for s in mask[mask.fillna(False).astype(bool)].index:
                if s in reasons:
                    reasons[s].append(text)

        # ---- A. 硬性排除
        flag_r = self.flag_reasons(date)
        add(self.info["market"].isin(cfg.excluded_markets), "興櫃/創新板")
        add((date - self.listed).dt.days < cfg.min_listed_days, "上市未滿1年")


        to60 = self._row(self.to60, date)
        out["turnover_60d"] = to60
        add(to60 < cfg.min_turnover_60d, "60日均成交值不足")
        mcap = self._row(self.mcap, date)
        out["market_cap"] = mcap
        add(mcap < cfg.min_market_cap, "市值不足")

        # Z-Score 的市值取財報公告日當時的值（快照只在公告時更新）
        fund = self._cached(self._fund_cache, self._fin_dates, date,
                            lambda d: fundamental_snapshot(md.financials, d, self._row(self.mcap, d)).reindex(stocks))
        out = out.join(fund)
        add((fund["eps_4q"] < 0) & (fund["bvps"] < 10), "近4季虧損且淨值<10")

        # ---- B. 風險評分
        exempt = self.exempt
        pen = pd.DataFrame(index=stocks)

        def penalty(name: str, cond: pd.Series, pts: float):
            pen[name] = np.where(cond.reindex(stocks).fillna(False).astype(bool), pts, 0.0)

        penalty("fscore", fund["fscore"] <= cfg.fscore_low, 20)
        penalty("zscore", (fund["zscore"] < cfg.zscore_low) & ~exempt, 20)
        penalty("solvency", ((fund["current_ratio"] < cfg.current_ratio_low) |
                             (fund["interest_cover"] < cfg.interest_cover_low)) & ~exempt, 10)
        # 應計：全市場最高 10%；金融/建設業改用產業內百分位
        acc_rank = fund["accrual"].rank(pct=True)
        acc_rank_ind = fund["accrual"].groupby(self.info["industry"]).rank(pct=True)
        acc_rank = acc_rank.where(~exempt, acc_rank_ind)
        penalty("accrual", (acc_rank >= cfg.top_decile) | (fund["profit_neg_ocf_3y"] == 1), 15)
        penalty("non_operating", fund["non_op_ratio"] > cfg.non_operating_ratio_high, 10)

        gov = self._cached(self._gov_cache, self._gov_dates, date, lambda d: _asof_long(
            md.governance, d, ["pledge_ratio", "insider_holding_ratio", "transfer_declared"]).reindex(stocks))
        out = out.join(gov)
        penalty("pledge", gov["pledge_ratio"] > cfg.pledge_ratio_high, 15)
        penalty("insider_transfer", (gov["insider_holding_ratio"] < 0.10) & (gov["transfer_declared"] > 0.01), 10)

        chg = _event_counts(self._events, date, {"auditor_change", "cfo_change"}, 2).reindex(stocks)
        dil = _event_counts(self._events, date, {"cash_increase", "private_placement", "cb_issue"}, 3).reindex(stocks)
        out["mgmt_changes_2y"], out["dilution_3y"] = chg, dil
        penalty("mgmt_change", chg >= 1, 10)
        penalty("dilution", dil >= 2, 10)

        iv = self._row(self.pf["idio_vol"], date)
        mx = self._row(self.pf["max_ret"], date)
        ls = self.pf["limit_streak"].loc[:date].tail(20)
        lu = ls.max().reindex(stocks) if len(ls) else pd.Series(0.0, index=stocks)
        out["idio_vol"], out["max_ret_20d"], out["limit_up_streak"] = iv, mx, lu
        penalty("idio_vol", iv.rank(pct=True) >= cfg.top_decile, 20)
        penalty("lottery", (mx.rank(pct=True) >= cfg.top_decile) | (lu >= 2), 20)

        if md.margin_usage is not None and not md.margin_usage.empty:
            mu = self._row(md.margin_usage, date)
            out["margin_usage"] = mu
            penalty("margin_usage", mu > cfg.margin_usage_high, 10)

        out["score"] = (100 - pen.sum(axis=1)).clip(0, 100)
        out["penalties"] = pen.apply(lambda r: ",".join(k for k, v in r.items() if v > 0), axis=1)
        out["other_reasons"] = pd.Series({s: "；".join(r) for s, r in reasons.items()})
        out["flag_reasons"] = flag_r
        return self.finalize(out)

    def finalize(self, out: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        parts = out[["flag_reasons", "other_reasons"]].fillna("")
        out["reasons"] = parts.apply(lambda r: "；".join(x for x in r if x), axis=1)
        out["excluded"] = out["reasons"].str.len() > 0
        out["size_mult"] = np.select(
            [out["excluded"], out["score"] >= cfg.full_size_min, out["score"] >= cfg.half_size_min],
            [0.0, 1.0, 0.5], default=0.0)
        return out

    def flag_reasons(self, date) -> pd.Series:
        """每日更新的規則性旗標：全額交割、停止買賣、下市警示、處置、注意股次數、會計師意見。"""
        date = pd.Timestamp(date)
        cfg, md = self.cfg, self.md
        reasons: dict[str, list[str]] = {}

        def add(ids, text):
            for s in ids:
                reasons.setdefault(s, []).append(text)

        f = self._flags
        if f is not None and len(f):
            live = f[(f["start"] <= date) & (f["end"].isna() | (f["end"] >= date))]
            for ftype, zh in HARD_FLAG_TYPES.items():
                add(live.loc[live["flag_type"] == ftype, "stock_id"].unique(), zh)
            att = f[(f["flag_type"] == ATTENTION) & (f["start"] <= date) &
                    (f["start"] > date - pd.Timedelta(days=cfg.attention_lookback_days))]
            cnt = att.groupby("stock_id").size()
            add(cnt[cnt >= cfg.attention_max_count].index,
                f"近{cfg.attention_lookback_days}日注意股{cfg.attention_max_count}次以上")
        aud = self._cached(self._aud_cache, self._aud_dates, date,
                           lambda d: _asof_long(md.audit, d, ["opinion"]).reindex(self.stocks))
        bad = aud.index[aud["opinion"].notna() & (aud["opinion"] != "unqualified")]
        add(bad, "會計師非無保留意見")
        return pd.Series({s: "；".join(r) for s, r in reasons.items()}, dtype=object).reindex(self.stocks).fillna("")

    def refresh_flags(self, base: pd.DataFrame, date) -> pd.DataFrame:
        """以較早的完整評估為底，只更新當日規則性旗標（回測的日頻檢查用）。"""
        out = base.copy()
        out["flag_reasons"] = self.flag_reasons(date)
        return self.finalize(out)


def evaluate_safety(md: MarketData, date, cfg: L3Config,
                    price_features: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """單次評估（實盤每日用）；回測請直接使用 SafetyEvaluator 以利用快取。"""
    return SafetyEvaluator(md, cfg, price_features).evaluate(date)


def _sorted_dates(df: pd.DataFrame | None, col: str = "announce_date") -> np.ndarray:
    if df is None or df.empty or col not in df:
        return np.array([], dtype="datetime64[ns]")
    return np.sort(pd.to_datetime(df[col]).dropna().unique())


def _last_le(dates: np.ndarray, date: pd.Timestamp):
    if len(dates) == 0:
        return None
    i = np.searchsorted(dates, np.datetime64(date), side="right")
    return pd.Timestamp(dates[i - 1]) if i > 0 else None


def compute_price_features(md: MarketData, limit_pct: float = 0.10) -> dict[str, pd.DataFrame]:
    ret = md.returns()
    mkt = md.index_close.pct_change(fill_method=None)
    px = md.raw_close if md.raw_close is not None else md.close
    return {
        "idio_vol": ind.idiosyncratic_vol(ret, mkt, 60),
        "max_ret": ind.max_daily_return(ret, 20),
        "limit_streak": ind.consecutive_limit_up(px, limit_pct),
    }
