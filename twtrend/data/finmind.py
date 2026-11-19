"""FinMind API 用戶端與欄位轉換。

API：https://api.finmindtrade.com/api/v4/data
欄位對應標「待確認」者，實作上線前需對照 FinMind 官方文件驗證。
財報類資料 FinMind 的 date 是「財報期間」，這裡以法定公告期限推估公告日，避免未來函數：
  Q1 → 5/15、Q2 → 8/14、Q3 → 11/14、Q4 → 隔年 3/31；月營收 → 次月 10 日。
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"


class FinMindError(RuntimeError):
    pass


class FinMindClient:
    def __init__(self, token: str | None = None, session: requests.Session | None = None,
                 retries: int = 3, pause: float = 0.0):
        self.token = token
        self.session = session or requests.Session()
        self.retries = retries
        self.pause = pause

    def fetch(self, dataset: str, data_id: str | None = None, start: str | None = None,
              end: str | None = None, **extra: Any) -> pd.DataFrame:
        params = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start:
            params["start_date"] = start
        if end:
            params["end_date"] = end
        params.update(extra)
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        last = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(API_URL, params=params, headers=headers, timeout=60)
                payload = resp.json()
                if resp.status_code == 200 and payload.get("status", 200) == 200:
                    if self.pause:
                        time.sleep(self.pause)
                    return pd.DataFrame(payload.get("data", []))
                last = FinMindError(f"{dataset} {data_id}: {payload.get('msg', resp.status_code)}")
                if resp.status_code in (400, 401, 403):
                    break
            except (requests.RequestException, ValueError) as e:
                last = e
            time.sleep(2 ** attempt)
        raise FinMindError(str(last))

    # ------------------------------------------------------------ 各資料集
    def stock_info(self) -> pd.DataFrame:
        return transform_stock_info(self.fetch("TaiwanStockInfo"))

    def daily_price(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        adj = self.fetch("TaiwanStockPriceAdj", stock_id, start, end)
        raw = self.fetch("TaiwanStockPrice", stock_id, start, end)
        return transform_price(adj, raw)

    def index_daily(self, start: str, end: str | None = None, index_id: str = "TAIEX") -> pd.DataFrame:
        df = self.fetch("TaiwanStockPrice", index_id, start, end)
        if df.empty:
            return pd.DataFrame(columns=["date", "index_id", "close"])
        return pd.DataFrame({"date": df["date"], "index_id": index_id, "close": df["close"].astype(float)})

    def market_margin(self, start: str, end: str | None = None) -> pd.DataFrame:
        """大盤融資餘額（待確認：TaiwanStockTotalMarginPurchaseShortSale 的 name/TodayBalance 欄位）。"""
        df = self.fetch("TaiwanStockTotalMarginPurchaseShortSale", None, start, end)
        if df.empty:
            return df
        m = df[df["name"].astype(str).str.contains("MarginPurchaseMoney|融資金額", regex=True)]
        return pd.DataFrame({"date": m["date"], "stock_id": "MARKET",
                             "margin_balance": m["TodayBalance"].astype(float)})

    def stock_margin(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        df = self.fetch("TaiwanStockMarginPurchaseShortSale", stock_id, start, end)
        if df.empty:
            return df
        return pd.DataFrame({
            "date": df["date"], "stock_id": stock_id,
            "margin_balance": df["MarginPurchaseTodayBalance"].astype(float),
            "margin_limit": df["MarginPurchaseLimit"].astype(float),
            "short_balance": df["ShortSaleTodayBalance"].astype(float),
        })

    def institutional(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        df = self.fetch("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start, end)
        if df.empty:
            return df
        df["net"] = df["buy"].astype(float) - df["sell"].astype(float)
        name = df["name"].astype(str)
        piv = pd.DataFrame({
            "foreign_net": df["net"].where(name.str.startswith("Foreign")),
            "trust_net": df["net"].where(name == "Investment_Trust"),
            "dealer_net": df["net"].where(name.str.startswith("Dealer")),
            "date": df["date"],
        }).groupby("date").sum(min_count=1).reset_index()
        piv["stock_id"] = stock_id
        return piv

    def futures_institutional(self, start: str, end: str | None = None) -> pd.DataFrame:
        """外資台指期淨未平倉口數（待確認欄位：long_open_interest_balance_volume 等）。"""
        df = self.fetch("TaiwanFuturesInstitutionalInvestors", "TX", start, end)
        if df.empty:
            return pd.DataFrame(columns=["date", "foreign_net_oi"])
        f = df[df["institutional_investors"].astype(str).str.contains("外資|Foreign", regex=True)]
        net = f["long_open_interest_balance_volume"].astype(float) - f["short_open_interest_balance_volume"].astype(float)
        return pd.DataFrame({"date": f["date"], "foreign_net_oi": net}).groupby("date", as_index=False).sum()

    def monthly_revenue(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        return transform_revenue(self.fetch("TaiwanStockMonthRevenue", stock_id, start, end), stock_id)

    def financials(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        inc = self.fetch("TaiwanStockFinancialStatements", stock_id, start, end)
        bal = self.fetch("TaiwanStockBalanceSheet", stock_id, start, end)
        cf = self.fetch("TaiwanStockCashFlowsStatement", stock_id, start, end)
        return transform_financials(inc, bal, cf, stock_id)

    def delisted(self) -> pd.DataFrame:
        """已下市股票清單（待確認：TaiwanStockDelisting）。"""
        df = self.fetch("TaiwanStockDelisting")
        if df.empty:
            return df
        return pd.DataFrame({"stock_id": df["stock_id"], "name": df.get("stock_name"), "delisted_date": df["date"]})


# ---------------------------------------------------------------- 轉換函式
MARKET_MAP = {"twse": "twse", "tpex": "tpex", "emerging": "emerging", "上市": "twse", "上櫃": "tpex", "興櫃": "emerging"}


def transform_stock_info(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "name", "industry", "market", "listed_date"])
    out = pd.DataFrame({
        "stock_id": df["stock_id"].astype(str),
        "name": df.get("stock_name"),
        "industry": df.get("industry_category"),
        "market": df.get("type", pd.Series(dtype=str)).map(lambda x: MARKET_MAP.get(str(x), str(x))),
        "listed_date": df.get("date"),
    })
    # 只保留 4 碼普通股（排除權證、ETF 以外的商品；0050 等 ETF 另行指定）
    out = out[out["stock_id"].str.fullmatch(r"\d{4}")]
    return out.drop_duplicates("stock_id", keep="last")


def transform_price(adj: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "stock_id", "open", "high", "low", "close", "raw_close", "volume", "turnover"]
    if adj.empty and raw.empty:
        return pd.DataFrame(columns=cols)
    base = adj if not adj.empty else raw
    out = pd.DataFrame({
        "date": base["date"], "stock_id": base["stock_id"].astype(str),
        "open": base["open"].astype(float), "high": base["max"].astype(float),
        "low": base["min"].astype(float), "close": base["close"].astype(float),
        "volume": base["Trading_Volume"].astype(float), "turnover": base["Trading_money"].astype(float),
    })
    if not raw.empty:
        out = out.merge(raw[["date", "close"]].rename(columns={"close": "raw_close"}), on="date", how="left")
    else:
        out["raw_close"] = out["close"]
    out = out[(out["close"] > 0) & (out["volume"] > 0)]
    return out[cols]


def quarter_announce_date(period_end: pd.Timestamp) -> pd.Timestamp:
    p = pd.Timestamp(period_end)
    return {
        3: pd.Timestamp(p.year, 5, 15), 6: pd.Timestamp(p.year, 8, 14),
        9: pd.Timestamp(p.year, 11, 14), 12: pd.Timestamp(p.year + 1, 3, 31),
    }.get(p.month, p + pd.Timedelta(days=75))


def transform_revenue(df: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    cols = ["stock_id", "period", "announce_date", "revenue", "yoy"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    d["period"] = pd.to_datetime(dict(year=d["revenue_year"], month=d["revenue_month"], day=1))
    d = d.sort_values("period")
    d["revenue"] = d["revenue"].astype(float)
    prev = d.set_index("period")["revenue"]
    prev.index = prev.index + pd.DateOffset(years=1)
    d["yoy"] = d["revenue"].values / prev.reindex(d["period"]).values - 1
    d["announce_date"] = d["period"] + pd.DateOffset(months=1) + pd.Timedelta(days=9)
    d["stock_id"] = stock_id
    d["period"] = d["period"].dt.strftime("%Y-%m")
    d["announce_date"] = d["announce_date"].dt.strftime("%Y-%m-%d")
    return d[cols]


# FinMind type → 本系統欄位（待確認：各產業科目名稱不同，金融業需另外對應）
FIN_MAP = {
    "income": {
        "Revenue": "revenue", "GrossProfit": "gross_profit", "IncomeAfterTaxes": "net_income",
        "PreTaxIncome": "pretax_income", "EPS": "eps", "TotalNonoperatingIncomeAndExpense": "non_operating_income",
        "OperatingIncome": "ebit", "FinanceCosts": "interest_expense",
    },
    "balance": {
        "TotalAssets": "total_assets", "Liabilities": "total_liabilities", "CurrentAssets": "current_assets",
        "CurrentLiabilities": "current_liabilities", "RetainedEarnings": "retained_earnings",
        "Equity": "equity", "NoncurrentLiabilities": "long_term_debt", "OrdinaryShare": "share_capital",
    },
    "cashflow": {"CashFlowsFromOperatingActivities": "operating_cf",
                 "NetCashInflowFromOperatingActivities": "operating_cf"},
}


def _pivot(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df[df["type"].isin(mapping)].copy()
    d["field"] = d["type"].map(mapping)
    return d.pivot_table(index="date", columns="field", values="value", aggfunc="last")


def transform_financials(inc: pd.DataFrame, bal: pd.DataFrame, cf: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    parts = [_pivot(inc, FIN_MAP["income"]), _pivot(bal, FIN_MAP["balance"])]
    c = _pivot(cf, FIN_MAP["cashflow"])
    if not c.empty:
        # 台灣現金流量表為年初至今累計，轉成單季
        c.index = pd.to_datetime(c.index)
        c = c.sort_index()
        q = c.copy()
        for i in range(1, len(c)):
            if c.index[i].year == c.index[i - 1].year:
                q.iloc[i] = c.iloc[i] - c.iloc[i - 1]
        c = q
        c.index = c.index.strftime("%Y-%m-%d")
    parts.append(c)
    df = pd.concat([p for p in parts if not p.empty], axis=1)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if "share_capital" in df:
        df["shares"] = df["share_capital"] / 10.0      # 面額 10 元
        df["bvps"] = df.get("equity") / df["shares"]
        df = df.drop(columns=["share_capital"])
    df["announce_date"] = [quarter_announce_date(p).strftime("%Y-%m-%d") for p in df.index]
    df["period"] = df.index.strftime("%Y-%m-%d")
    df["stock_id"] = stock_id
    return df.reset_index(drop=True)
