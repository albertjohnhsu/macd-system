"""SQLite 資料表（依設計書「資料來源與資料表」）。

原始資料表（raw_*）以長表儲存；運算結果表與實盤表依設計書命名。
所有財報類資料以「公告日」(announce_date) 對齊，避免未來函數。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

SCHEMA = """
-- ===== 原始資料 =====
CREATE TABLE IF NOT EXISTS stock_info (
    stock_id TEXT PRIMARY KEY,
    name TEXT,
    industry TEXT,
    market TEXT,              -- twse / tpex / emerging / innovation
    listed_date TEXT,
    delisted_date TEXT,
    shares REAL               -- 流通股數
);
CREATE TABLE IF NOT EXISTS daily_price (
    date TEXT, stock_id TEXT,
    open REAL, high REAL, low REAL, close REAL,   -- 還原權息後價格
    raw_close REAL,                                -- 未還原收盤（漲跌停判斷用）
    volume REAL, turnover REAL,
    PRIMARY KEY (date, stock_id)
);
CREATE TABLE IF NOT EXISTS index_daily (
    date TEXT, index_id TEXT, close REAL,
    PRIMARY KEY (date, index_id)
);
CREATE TABLE IF NOT EXISTS margin_daily (
    date TEXT, stock_id TEXT,          -- stock_id='MARKET' 為大盤合計
    margin_balance REAL, margin_limit REAL, short_balance REAL,
    maintenance_ratio REAL,
    PRIMARY KEY (date, stock_id)
);
CREATE TABLE IF NOT EXISTS institutional_daily (
    date TEXT, stock_id TEXT, foreign_net REAL, trust_net REAL, dealer_net REAL,
    PRIMARY KEY (date, stock_id)
);
CREATE TABLE IF NOT EXISTS futures_institutional (
    date TEXT, foreign_net_oi REAL,
    PRIMARY KEY (date)
);
CREATE TABLE IF NOT EXISTS monthly_revenue (
    stock_id TEXT, period TEXT, announce_date TEXT, revenue REAL, yoy REAL,
    PRIMARY KEY (stock_id, period)
);
CREATE TABLE IF NOT EXISTS financials_quarterly (
    stock_id TEXT, period TEXT, announce_date TEXT,
    net_income REAL, operating_cf REAL, total_assets REAL, total_liabilities REAL,
    current_assets REAL, current_liabilities REAL, retained_earnings REAL,
    ebit REAL, interest_expense REAL, revenue REAL, gross_profit REAL,
    pretax_income REAL, non_operating_income REAL, equity REAL, eps REAL,
    long_term_debt REAL, shares REAL, bvps REAL,
    PRIMARY KEY (stock_id, period)
);
CREATE TABLE IF NOT EXISTS governance_monthly (
    stock_id TEXT, period TEXT, announce_date TEXT,
    insider_holding_ratio REAL, pledge_ratio REAL, transfer_declared REAL,
    PRIMARY KEY (stock_id, period)
);
CREATE TABLE IF NOT EXISTS corporate_events (
    stock_id TEXT, date TEXT, event_type TEXT,  -- auditor_change / cfo_change / cash_increase / private_placement / cb_issue
    detail TEXT,
    PRIMARY KEY (stock_id, date, event_type)
);
CREATE TABLE IF NOT EXISTS audit_opinion (
    stock_id TEXT, period TEXT, announce_date TEXT, opinion TEXT,  -- unqualified / qualified / disclaimer / adverse
    PRIMARY KEY (stock_id, period)
);
CREATE TABLE IF NOT EXISTS macro_monthly (
    period TEXT PRIMARY KEY, announce_date TEXT,
    leading_index REAL, m1b_yoy REAL, m2_yoy REAL
);

-- ===== 運算結果 =====
CREATE TABLE IF NOT EXISTS market_regime_daily (
    date TEXT PRIMARY KEY, indicators TEXT, score REAL, state TEXT, cap REAL, target_exposure REAL
);
CREATE TABLE IF NOT EXISTS trend_score_weekly (
    date TEXT, stock_id TEXT, components TEXT, trend_score REAL, rank INTEGER,
    PRIMARY KEY (date, stock_id)
);
CREATE TABLE IF NOT EXISTS safety_flags (
    stock_id TEXT, flag_type TEXT, start_date TEXT, end_date TEXT, source TEXT,
    PRIMARY KEY (stock_id, flag_type, start_date)
);
CREATE TABLE IF NOT EXISTS safety_score_quarterly (
    stock_id TEXT, announce_date TEXT, components TEXT, score REAL,
    PRIMARY KEY (stock_id, announce_date)
);

-- ===== 實盤 =====
CREATE TABLE IF NOT EXISTS positions (
    stock_id TEXT PRIMARY KEY, shares INTEGER, entry_date TEXT, entry_price REAL,
    initial_stop REAL, trailing_stop REAL, highest_close REAL, one_r REAL,
    pending_exit_since TEXT, catastrophic_stop REAL
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT, exec_date TEXT, stock_id TEXT, side TEXT, shares INTEGER,
    limit_price REAL, signal_price REAL, reason TEXT, status TEXT DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS trade_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id TEXT, signal_date TEXT, signal_price REAL, fill_date TEXT, fill_price REAL,
    shares INTEGER, initial_stop REAL, one_r REAL,
    entry_l1_state TEXT, entry_trend_score REAL, entry_safety_score REAL,
    exit_date TEXT, exit_price REAL, exit_reason TEXT, r_multiple REAL,
    entry_slippage REAL, exit_slippage REAL,
    followed_signal INTEGER DEFAULT 1, deviation_reason TEXT
);
CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, layer TEXT, stock_id TEXT, action TEXT, reason TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS health_checks (
    date TEXT, stock_id TEXT, status TEXT, triggers TEXT, weak_items TEXT, end_items TEXT, detail TEXT,
    PRIMARY KEY (date, stock_id)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, kind TEXT, message TEXT, sent INTEGER DEFAULT 0
);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL") if self.path != ":memory:" else None
        self._conn.executescript(SCHEMA)
        # 舊資料庫升級
        self.add_columns("positions", {"health_json": "TEXT", "weak_seen": "TEXT"})
        self.add_columns("trade_journal", {"weak_seen": "TEXT"})

    def add_columns(self, table: str, cols: dict[str, str]) -> None:
        have = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
        for c, typ in cols.items():
            if c not in have:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {c} {typ}")
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def upsert_df(self, table: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        cols = list(df.columns)
        placeholders = ",".join("?" * len(cols))
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        rows = [tuple(_py(v) for v in r) for r in df.itertuples(index=False, name=None)]
        with self.tx() as c:
            c.executemany(sql, rows)
        return len(rows)

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self._conn, params=params)

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.tx() as c:
            c.execute(sql, params)

    def close(self) -> None:
        self._conn.close()


def _py(v):
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v != v:  # NaN
        return None
    return v
