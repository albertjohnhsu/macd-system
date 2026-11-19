"""資料管線：外部來源 → SQLite → MarketData。"""
from __future__ import annotations

import logging
from datetime import date as _date

import numpy as np
import pandas as pd

from ..db import Database
from .finmind import FinMindClient, FinMindError
from .market import MarketData
from .twse import fetch_flags

log = logging.getLogger(__name__)


def update_from_finmind(db: Database, client: FinMindClient, start: str, end: str | None = None,
                        stocks: list[str] | None = None, benchmark: str = "0050",
                        with_fundamentals: bool = True) -> dict:
    """增量更新所有資料表。stocks=None 時抓全部上市櫃普通股（含已下市股以消除倖存者偏差）。"""
    report = {"errors": []}
    info = client.stock_info()
    try:
        dl = client.delisted()
        if not dl.empty:
            info = info.merge(dl[["stock_id", "delisted_date"]], on="stock_id", how="outer")
    except FinMindError as e:
        report["errors"].append(f"delisted: {e}")
    db.upsert_df("stock_info", info.reindex(columns=["stock_id", "name", "industry", "market",
                                                      "listed_date", "delisted_date"]))
    ids = stocks or info["stock_id"].tolist()
    ids = list(dict.fromkeys(ids + [benchmark]))

    db.upsert_df("index_daily", client.index_daily(start, end))
    for fn, table in ((client.market_margin, "margin_daily"), (client.futures_institutional, "futures_institutional")):
        try:
            db.upsert_df(table, fn(start, end))
        except (FinMindError, KeyError) as e:
            report["errors"].append(f"{table}: {e}")

    for i, sid in enumerate(ids):
        try:
            db.upsert_df("daily_price", client.daily_price(sid, start, end))
            db.upsert_df("margin_daily", client.stock_margin(sid, start, end))
            db.upsert_df("institutional_daily", client.institutional(sid, start, end))
            if with_fundamentals and sid != benchmark:
                db.upsert_df("monthly_revenue", client.monthly_revenue(sid, start, end))
                fin = client.financials(sid, start, end)
                if not fin.empty:
                    cols = [c for c in fin.columns if c in _FIN_COLS]
                    db.upsert_df("financials_quarterly", fin[cols])
        except (FinMindError, KeyError) as e:
            report["errors"].append(f"{sid}: {e}")
        if i % 50 == 0:
            log.info("FinMind 進度 %d/%d", i, len(ids))
    report["stocks"] = len(ids)
    return report


_FIN_COLS = {"stock_id", "period", "announce_date", "net_income", "operating_cf", "total_assets",
             "total_liabilities", "current_assets", "current_liabilities", "retained_earnings", "ebit",
             "interest_expense", "revenue", "gross_profit", "pretax_income", "non_operating_income",
             "equity", "eps", "long_term_debt", "shares", "bvps"}


def update_flags(db: Database, today: str | None = None) -> list[str]:
    today = today or _date.today().isoformat()
    df, errors = fetch_flags(today)
    db.upsert_df("safety_flags", df)
    return errors


def load_market_data(db: Database, start: str | None = None, end: str | None = None,
                     index_id: str = "TAIEX", exclude: tuple[str, ...] = ("0050",)) -> MarketData:
    where, params = [], []
    if start:
        where.append("date >= ?"); params.append(start)
    if end:
        where.append("date <= ?"); params.append(end)
    w = (" WHERE " + " AND ".join(where)) if where else ""
    px = db.query(f"SELECT * FROM daily_price{w}", tuple(params))
    if px.empty:
        raise ValueError("daily_price 沒有資料，請先執行 `twtrend update` 或 `twtrend demo`")
    px["date"] = pd.to_datetime(px["date"])
    px = px[~px["stock_id"].isin(exclude)]

    def wide(col):
        return px.pivot(index="date", columns="stock_id", values=col).sort_index()

    close = wide("close")
    idx = db.query(f"SELECT date, close FROM index_daily WHERE index_id = ?", (index_id,))
    idx["date"] = pd.to_datetime(idx["date"])
    index_close = idx.set_index("date")["close"].sort_index().reindex(close.index).ffill()

    info = db.query("SELECT * FROM stock_info").set_index("stock_id")
    shares = info["shares"] if "shares" in info else pd.Series(dtype=float)
    fin = db.query("SELECT stock_id, announce_date, shares FROM financials_quarterly WHERE shares IS NOT NULL")
    if not fin.empty:
        latest = fin.sort_values("announce_date").groupby("stock_id")["shares"].last()
        shares = shares.combine_first(latest) if len(shares) else latest
    info["shares"] = shares.reindex(info.index)

    mm = db.query("SELECT date, margin_balance, maintenance_ratio FROM margin_daily WHERE stock_id='MARKET'")
    mm["date"] = pd.to_datetime(mm["date"])
    sm = db.query("SELECT date, stock_id, margin_balance, margin_limit FROM margin_daily WHERE stock_id!='MARKET'")
    usage = None
    if not sm.empty:
        sm["date"] = pd.to_datetime(sm["date"])
        sm["usage"] = sm["margin_balance"] / sm["margin_limit"].replace(0, np.nan)
        usage = sm.pivot(index="date", columns="stock_id", values="usage").reindex(close.index).ffill()
    inst = db.query("SELECT date, stock_id, COALESCE(foreign_net,0)+COALESCE(trust_net,0) AS net FROM institutional_daily")
    inst_w = None
    if not inst.empty:
        inst["date"] = pd.to_datetime(inst["date"])
        inst_w = inst.pivot(index="date", columns="stock_id", values="net").reindex(close.index)
    fut = db.query("SELECT date, foreign_net_oi FROM futures_institutional")
    fut_s = None
    if not fut.empty:
        fut["date"] = pd.to_datetime(fut["date"])
        fut_s = fut.set_index("date")["foreign_net_oi"].sort_index()

    return MarketData(
        close=close, high=wide("high").reindex_like(close), low=wide("low").reindex_like(close),
        open=wide("open").reindex_like(close), volume=wide("volume").reindex_like(close),
        turnover=wide("turnover").reindex_like(close), raw_close=wide("raw_close").reindex_like(close),
        index_close=index_close, info=info.reindex(close.columns),
        revenue=db.query("SELECT stock_id, announce_date, yoy FROM monthly_revenue"),
        financials=db.query("SELECT * FROM financials_quarterly"),
        governance=db.query("SELECT * FROM governance_monthly"),
        events=db.query("SELECT * FROM corporate_events"),
        audit=db.query("SELECT * FROM audit_opinion"),
        flags=db.query("SELECT * FROM safety_flags"),
        margin_market=mm.set_index("date").sort_index() if not mm.empty else mm,
        margin_usage=usage, institutional=inst_w, futures_net_oi=fut_s,
        macro=db.query("SELECT * FROM macro_monthly"),
    )


def load_benchmark(db: Database, benchmark: str = "0050") -> pd.Series:
    df = db.query("SELECT date, close FROM daily_price WHERE stock_id = ?", (benchmark,))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()
