"""合成市場資料：在無法連上 FinMind / 證交所時，用來展示與測試整套流程。

模擬內容：多空循環的大盤、產業因子、趨勢股、樂透股（飆漲後崩落）、
地雷股（財報惡化 → 處置 / 全額交割 → 下市）、季財報、月營收、董監質押、總經指標。
注意：這只是用來驗證程式邏輯的假資料，不代表任何真實績效。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..db import Database

INDUSTRIES = ["半導體業", "電子零組件業", "電腦及週邊設備業", "航運業", "食品工業", "塑膠工業",
              "金融保險業", "生技醫療業", "建材營造業", "鋼鐵工業"]


def _regime_path(n: int, rng: np.random.Generator) -> np.ndarray:
    """產生多空循環的大盤日報酬（多頭 +18%/年、空頭 −35%/年、盤整 0）。"""
    out = np.empty(n)
    i = 0
    while i < n:
        kind = rng.choice(["bull", "bear", "flat"], p=[0.55, 0.2, 0.25])
        length = int(rng.integers(60, 400) if kind != "bear" else rng.integers(60, 200))
        mu = {"bull": 0.18, "bear": -0.35, "flat": 0.0}[kind] / 252
        vol = {"bull": 0.15, "bear": 0.30, "flat": 0.14}[kind] / np.sqrt(252)
        seg = rng.normal(mu, vol, min(length, n - i))
        out[i:i + len(seg)] = seg
        i += len(seg)
    return out


def generate(n_stocks: int = 60, start: str = "2014-01-01", end: str = "2024-12-31",
             seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    mkt = _regime_path(n, rng)
    ind_names = INDUSTRIES
    ind_ret = {k: rng.normal(0, 0.008, n) + 0.3 * mkt for k in ind_names}

    stock_ids = [str(1101 + i * 37) for i in range(n_stocks)]
    kinds = rng.choice(["normal", "trend", "lottery", "bomb"], n_stocks, p=[0.55, 0.25, 0.1, 0.1])
    info_rows, price_rows, fin_rows, rev_rows, gov_rows, flag_rows, event_rows, audit_rows = ([] for _ in range(8))

    for k, (sid, kind) in enumerate(zip(stock_ids, kinds)):
        industry = ind_names[k % len(ind_names)]
        beta = rng.uniform(0.6, 1.4)
        alpha = {"normal": 0.0, "trend": rng.uniform(0.10, 0.30), "lottery": 0.0, "bomb": 0.05}[kind] / 252
        idio = {"normal": 0.018, "trend": 0.02, "lottery": 0.035, "bomb": 0.025}[kind]
        # 趨勢股：alpha 有持續期（強勢期 / 普通期交替）
        alpha_path = np.full(n, alpha)
        if kind == "trend":
            phase = np.repeat(rng.choice([1.5, 0.0, -0.5], size=n // 120 + 1, p=[0.5, 0.35, 0.15]), 120)[:n]
            alpha_path = alpha * phase
        r = beta * mkt + 0.7 * (ind_ret[industry] - 0.3 * mkt) + alpha_path + rng.normal(0, idio, n)
        listed = dates[0] if rng.random() > 0.15 else dates[int(rng.integers(100, n // 2))]
        delisted = None
        if kind == "lottery":
            for _ in range(int(rng.integers(2, 5))):
                t = int(rng.integers(300, n - 60))
                r[t:t + 5] += 0.095                        # 連續漲停
                r[t + 20:t + 80] -= 0.012                  # 之後回落
        if kind == "bomb":
            t_bad = int(rng.integers(n // 3, n - 200))
            r[t_bad:t_bad + 120] -= 0.006
            if rng.random() < 0.6:
                delisted = dates[min(t_bad + 250, n - 1)]
                r[t_bad + 120:] -= 0.01
        r = np.clip(r, -0.095, 0.095)                       # 漲跌停
        close = 50 * np.exp(rng.uniform(-0.5, 1.0)) * np.cumprod(1 + r)
        live = (dates >= listed) & ((dates < delisted) if delisted is not None else True)
        intraday = np.abs(rng.normal(0, idio * 0.6, n))
        high = close * (1 + intraday)
        low = close * (1 - intraday)
        opn = np.clip(close / (1 + r) * (1 + rng.normal(0, idio * 0.3, n)), low, high)
        locked_up = r >= 0.094
        locked_dn = r <= -0.094
        high[locked_up | locked_dn] = close[locked_up | locked_dn]
        low[locked_up | locked_dn] = close[locked_up | locked_dn]
        opn[locked_up | locked_dn] = close[locked_up | locked_dn]
        shares = float(rng.uniform(1e8, 2e9)) if kind != "lottery" else float(rng.uniform(5e7, 3e8))
        vol = shares * rng.uniform(0.002, 0.01) * np.exp(rng.normal(0, 0.4, n))
        for i in np.where(live)[0]:
            price_rows.append((dates[i], sid, opn[i], high[i], low[i], close[i], close[i], vol[i], vol[i] * close[i]))
        info_rows.append((sid, f"合成{sid}", industry, "twse", listed.strftime("%Y-%m-%d"),
                          delisted.strftime("%Y-%m-%d") if delisted is not None else None, shares))

        # ---- 季財報（以公告日對齊）
        q_ends = pd.date_range(dates[0] - pd.DateOffset(years=3), dates[-1], freq="QE")
        bad_from = dates[t_bad] if kind == "bomb" else None
        ta = shares * 30
        for q in q_ends:
            ann = pd.Timestamp({3: f"{q.year}-05-15", 6: f"{q.year}-08-14", 9: f"{q.year}-11-14",
                                12: f"{q.year + 1}-03-31"}[q.month])
            bad = bad_from is not None and q >= bad_from - pd.DateOffset(months=9)
            margin = rng.uniform(0.05, 0.2) if not bad else rng.uniform(-0.1, 0.02)
            rev = ta * 0.25 * rng.uniform(0.9, 1.1)
            ni = rev * margin
            ocf = ni * rng.uniform(0.8, 1.3) if not bad else -abs(ni) - rev * 0.05
            fin_rows.append(dict(
                stock_id=sid, period=q.strftime("%Y-%m-%d"), announce_date=ann.strftime("%Y-%m-%d"),
                net_income=ni, operating_cf=ocf, total_assets=ta, total_liabilities=ta * (0.4 if not bad else 0.8),
                current_assets=ta * 0.4, current_liabilities=ta * (0.25 if not bad else 0.5),
                retained_earnings=ta * (0.2 if not bad else -0.05), ebit=ni * 1.25 + ta * 0.002,
                interest_expense=ta * 0.002, revenue=rev, gross_profit=rev * (0.25 if not bad else 0.08),
                pretax_income=ni * 1.2, non_operating_income=ni * (0.1 if not bad else 0.8),
                equity=ta * 0.6, eps=ni / shares * 10 / 10, long_term_debt=ta * (0.15 if not bad else 0.3),
                shares=shares, bvps=ta * (0.6 if not bad else 0.15) / shares))
            audit_rows.append((sid, q.strftime("%Y-%m-%d"), ann.strftime("%Y-%m-%d"),
                               "unqualified" if not (bad and q >= bad_from) else "qualified"))
        # ---- 月營收
        yoy_base = {"trend": 0.25, "normal": 0.03, "lottery": 0.0, "bomb": -0.05}[kind]
        for p in pd.date_range(dates[0], dates[-1], freq="MS"):
            ann = p + pd.DateOffset(months=1) + pd.Timedelta(days=9)
            rev_rows.append((sid, p.strftime("%Y-%m"), ann.strftime("%Y-%m-%d"), None,
                             yoy_base + rng.normal(0, 0.15)))
        # ---- 董監質押
        for p in pd.date_range(dates[0], dates[-1], freq="MS"):
            pledge = rng.uniform(0, 0.15) if kind != "bomb" else rng.uniform(0.3, 0.7)
            gov_rows.append((sid, p.strftime("%Y-%m"), (p + pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
                             rng.uniform(0.1, 0.4), pledge, 0.0))
        # ---- 安全旗標
        if kind == "lottery":
            for t in np.where(locked_up)[0][::3]:
                d0 = dates[t]
                flag_rows.append((sid, "attention", d0.strftime("%Y-%m-%d"), d0.strftime("%Y-%m-%d"), "synthetic"))
                flag_rows.append((sid, "attention", (d0 + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                                  (d0 + pd.Timedelta(days=3)).strftime("%Y-%m-%d"), "synthetic"))
                flag_rows.append((sid, "disposal", (d0 + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                                  (d0 + pd.Timedelta(days=17)).strftime("%Y-%m-%d"), "synthetic"))
        if kind == "bomb" and delisted is not None:
            fd = delisted - pd.Timedelta(days=120)
            flag_rows.append((sid, "full_delivery", fd.strftime("%Y-%m-%d"), None, "synthetic"))
            event_rows.append((sid, (fd - pd.Timedelta(days=200)).strftime("%Y-%m-%d"), "auditor_change", ""))
        if kind == "bomb":
            for j in range(3):
                event_rows.append((sid, (dates[t_bad] - pd.Timedelta(days=150 * j)).strftime("%Y-%m-%d"),
                                   "private_placement", ""))

    prices = pd.DataFrame(price_rows, columns=["date", "stock_id", "open", "high", "low", "close", "raw_close",
                                               "volume", "turnover"])
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    index_close = 8000 * np.cumprod(1 + mkt)
    idx = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "index_id": "TAIEX", "close": index_close})
    # 0050 ≈ 大盤 + 小幅追蹤誤差
    etf = 30 * np.cumprod(1 + mkt + rng.normal(0, 0.0005, n))
    etf_rows = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "stock_id": "0050", "open": etf, "high": etf * 1.005,
                             "low": etf * 0.995, "close": etf, "raw_close": etf, "volume": 1e7, "turnover": 1e7 * etf})
    prices = pd.concat([prices, etf_rows], ignore_index=True)

    margin_bal = 2.5e11 * np.exp(np.cumsum(0.6 * mkt + rng.normal(0, 0.002, n)))
    margin = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "stock_id": "MARKET", "margin_balance": margin_bal,
                           "maintenance_ratio": 1.65 + 3 * pd.Series(mkt).rolling(60, min_periods=1).sum().values})
    fut = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                        "foreign_net_oi": pd.Series(mkt * 4e5).rolling(20, min_periods=1).sum().values})
    months = pd.date_range(dates[0], dates[-1], freq="MS")
    m_ret = pd.Series(mkt, index=dates).resample("MS").sum().reindex(months).fillna(0).values
    macro = pd.DataFrame({
        "period": months.strftime("%Y-%m"),
        "announce_date": (months + pd.DateOffset(months=1) + pd.Timedelta(days=26)).strftime("%Y-%m-%d"),
        "leading_index": 100 + np.cumsum(m_ret * 10 + rng.normal(0, 0.2, len(months))),
        "m1b_yoy": 0.05 + pd.Series(m_ret).rolling(6, min_periods=1).sum().values,
        "m2_yoy": 0.05 + rng.normal(0, 0.005, len(months)),
    })
    return {
        "stock_info": pd.DataFrame(info_rows, columns=["stock_id", "name", "industry", "market", "listed_date",
                                                       "delisted_date", "shares"]),
        "daily_price": prices, "index_daily": idx,
        "financials_quarterly": pd.DataFrame(fin_rows),
        "monthly_revenue": pd.DataFrame(rev_rows, columns=["stock_id", "period", "announce_date", "revenue", "yoy"]),
        "governance_monthly": pd.DataFrame(gov_rows, columns=["stock_id", "period", "announce_date",
                                                              "insider_holding_ratio", "pledge_ratio", "transfer_declared"]),
        "safety_flags": pd.DataFrame(flag_rows, columns=["stock_id", "flag_type", "start_date", "end_date", "source"]),
        "corporate_events": pd.DataFrame(event_rows, columns=["stock_id", "date", "event_type", "detail"]),
        "audit_opinion": pd.DataFrame(audit_rows, columns=["stock_id", "period", "announce_date", "opinion"]),
        "margin_daily": margin, "futures_institutional": fut, "macro_monthly": macro,
        "_kinds": pd.DataFrame({"stock_id": stock_ids, "kind": kinds}),
    }


def write_to_db(db: Database, data: dict[str, pd.DataFrame]) -> None:
    for table, df in data.items():
        if table.startswith("_"):
            continue
        db.upsert_df(table, df.drop_duplicates())
