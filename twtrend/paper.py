"""模擬盤（類似股市大富翁）：虛擬帳戶依系統訊號自動下單、以真實行情撮合，逐日記錄績效。

- 每個帳戶綁定一份「規則版本」（完整參數快照 + 雜湊），可同時開多個帳戶做 A/B 比較
- 撮合規則與回測完全相同（SimBroker）：隔日開盤成交、限價不追價、漲跌停鎖死、成本與稅
- 可手動下單（標記為人為干預，績效分開統計）
- 帳戶建立日之後的交易日標記為 forward（真正的樣本外），之前的歷史重播標記為 replay
- 每週候選清單（買 / 不買及原因）完整記錄，供 review.py 做規則檢討
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import date as _date, datetime

import numpy as np
import pandas as pd
import yaml

from .broker import SimBroker
from .config import Config, _merge
from .data.market import MarketData
from .db import Database
from .engine import SignalEngine

PAPER_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_accounts (
    name TEXT PRIMARY KEY, created_at TEXT, start_date TEXT, forward_from TEXT,
    capital REAL, config_json TEXT, rule_version TEXT, overrides TEXT, note TEXT,
    state_json TEXT, last_date TEXT, active INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS paper_equity (
    name TEXT, date TEXT, equity REAL, cash REAL, exposure REAL, positions INTEGER, phase TEXT,
    PRIMARY KEY (name, date)
);
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, stock_id TEXT, signal_date TEXT, signal_price REAL,
    entry_date TEXT, entry_price REAL, shares INTEGER, initial_stop REAL, one_r REAL,
    exit_date TEXT, exit_price REAL, exit_signal_price REAL, exit_reason TEXT, pnl REAL, r_multiple REAL,
    entry_state TEXT, trend_score REAL, safety_score REAL, industry TEXT, holding_days INTEGER,
    manual_entry INTEGER, manual_exit INTEGER, phase TEXT, weak_seen TEXT, exit_health_items TEXT
);
CREATE TABLE IF NOT EXISTS paper_health (
    name TEXT, date TEXT, stock_id TEXT, status TEXT, triggers TEXT, weak_items TEXT, end_items TEXT,
    detail TEXT, phase TEXT,
    PRIMARY KEY (name, date, stock_id)
);
CREATE TABLE IF NOT EXISTS paper_signals (
    name TEXT, date TEXT, stock_id TEXT, rank INTEGER, trend_score REAL, safety_score REAL, size_mult REAL,
    state TEXT, price REAL, decision TEXT, reason TEXT, shares INTEGER, phase TEXT,
    PRIMARY KEY (name, date, stock_id)
);
CREATE TABLE IF NOT EXISTS paper_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, date TEXT, stock_id TEXT, action TEXT, reason TEXT,
    manual INTEGER, phase TEXT
);
"""

_VOLATILE_KEYS = {"db_path", "finmind_token"}


def ensure_schema(db: Database) -> None:
    db.conn.executescript(PAPER_SCHEMA)
    db.add_columns("paper_trades", {"weak_seen": "TEXT", "exit_health_items": "TEXT"})


def health_rows(rows: list[dict]) -> pd.DataFrame:
    """把 HealthResult.to_row() 轉成精簡欄位 + detail JSON。"""
    from .health import ITEMS
    out = []
    for r in rows:
        detail = {k: [r.get(k, ""), r.get(k + "_說明", "")] for k in ITEMS if r.get(k)}
        out.append({"date": r["date"], "stock_id": r["stock_id"], "status": r["status"], "triggers": r["triggers"],
                    "weak_items": r["weak_items"], "end_items": r["end_items"],
                    "detail": json.dumps(detail, ensure_ascii=False)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 規則版本
def config_from_dict(d: dict) -> Config:
    cfg = Config()
    _merge(cfg, {k: v for k, v in d.items() if k not in _VOLATILE_KEYS})
    return cfg


def rule_payload(cfg: Config) -> dict:
    d = json.loads(json.dumps(cfg.to_dict(), default=list))
    for k in _VOLATILE_KEYS:
        d.pop(k, None)
    return d


def rule_version(cfg: Config) -> str:
    raw = json.dumps(rule_payload(cfg), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def apply_overrides(cfg: Config, overrides: list[str]) -> dict:
    """`l4.use_ma60_exit=true`、`l1.index_source=equal_weight` 形式的參數覆寫。"""
    applied = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"覆寫格式應為 key=value：{item}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        obj = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        last = parts[-1]
        if is_dataclass(obj) and last not in {f.name for f in fields(obj)}:
            raise KeyError(f"未知參數：{key}")
        current = getattr(obj, last)
        if isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        else:
            if isinstance(current, float) and isinstance(value, int):
                value = float(value)
            setattr(obj, last, value)
        applied[key] = value
    return applied


def diff_rules(a: Config, b: Config) -> dict:
    """兩份規則的差異（只列不同的參數）。"""
    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flat(v, key + "."))
            else:
                out[key] = v
        return out
    fa, fb = flat(rule_payload(a)), flat(rule_payload(b))
    return {k: (fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}


# ---------------------------------------------------------------- 帳戶管理
def create_account(db: Database, name: str, cfg: Config, capital: float | None = None,
                   start_date: str | None = None, overrides: list[str] | None = None,
                   note: str = "", manual_only: bool = False) -> dict:
    ensure_schema(db)
    if not db.query("SELECT name FROM paper_accounts WHERE name=?", (name,)).empty:
        raise ValueError(f"帳戶 {name} 已存在")
    applied = apply_overrides(cfg, overrides or [])
    capital = float(capital or cfg.l4.total_capital)
    cfg.l4.total_capital = capital
    today = _date.today().isoformat()
    state = {"cash": capital, "positions": [], "pending": [], "last_date": None, "auto_signals": not manual_only}
    db.execute(
        "INSERT INTO paper_accounts (name, created_at, start_date, forward_from, capital, config_json, rule_version,"
        " overrides, note, state_json, last_date) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (name, datetime.now().isoformat(timespec="seconds"), start_date or today, today, capital,
         json.dumps(rule_payload(cfg), ensure_ascii=False), rule_version(cfg),
         json.dumps(applied, ensure_ascii=False), note, json.dumps(state), None))
    return {"name": name, "rule_version": rule_version(cfg), "overrides": applied, "start_date": start_date or today}


def list_accounts(db: Database) -> pd.DataFrame:
    ensure_schema(db)
    return db.query("SELECT name, rule_version, overrides, capital, start_date, forward_from, last_date, note, active"
                    " FROM paper_accounts ORDER BY created_at")


def _account(db: Database, name: str) -> pd.Series:
    acc = db.query("SELECT * FROM paper_accounts WHERE name=?", (name,))
    if acc.empty:
        raise KeyError(f"沒有模擬帳戶 {name}")
    return acc.iloc[0]


def account_config(db: Database, name: str) -> Config:
    return config_from_dict(json.loads(_account(db, name)["config_json"]))


def load_broker(db: Database, md: MarketData, name: str, engine: SignalEngine | None = None) -> tuple[SimBroker, pd.Series]:
    acc = _account(db, name)
    cfg = config_from_dict(json.loads(acc["config_json"]))
    eng = engine or SignalEngine(md, cfg)
    br = SimBroker(md, cfg, eng, capital=acc["capital"])
    br.load_state(json.loads(acc["state_json"]))
    return br, acc


def _phase(d, forward_from: str) -> str:
    return "forward" if pd.Timestamp(d) >= pd.Timestamp(forward_from) else "replay"


def run_account(db: Database, md: MarketData, name: str, until=None,
                engine_cache: dict | None = None) -> dict:
    """把帳戶推進到 until（預設為資料最後一天）：逐日撮合、產生訊號、寫入績效。"""
    ensure_schema(db)
    acc = _account(db, name)
    if not acc["active"]:
        return {"name": name, "days": 0, "note": "帳戶已停用"}
    cfg = config_from_dict(json.loads(acc["config_json"]))
    ver = acc["rule_version"]
    eng = (engine_cache or {}).get(ver)
    if eng is None:
        eng = SignalEngine(md, cfg)
        if engine_cache is not None:
            engine_cache[ver] = eng
    br, _ = load_broker(db, md, name, eng)
    start = br.last_date + pd.Timedelta(days=1) if br.last_date is not None else pd.Timestamp(acc["start_date"])
    end = pd.Timestamp(until) if until else md.dates[-1]
    dates = md.dates[(md.dates >= start) & (md.dates <= end)]
    if len(dates) == 0:
        return {"name": name, "days": 0, "equity": None}
    all_dates = md.dates
    for d in dates:
        pos = all_dates.get_loc(d)
        nxt = all_dates[pos + 1] if pos + 1 < len(all_dates) else None
        br.step(d, nxt)

    ff = acc["forward_from"]
    eq = pd.DataFrame({
        "name": name, "date": [d.strftime("%Y-%m-%d") for d in br.equity],
        "equity": list(br.equity.values()), "cash": np.nan, "exposure": list(br.exposure.values()),
        "positions": np.nan, "phase": [_phase(d, ff) for d in br.equity],
    })
    eq.loc[eq.index[-1], ["cash", "positions"]] = [br.cash, len(br.positions)]
    db.upsert_df("paper_equity", eq)
    if br.trades:
        t = pd.DataFrame(br.trades).drop(columns=["open_at_end"])
        t["name"] = name
        t["phase"] = [_phase(d, ff) for d in t["entry_date"]]
        db.upsert_df("paper_trades", t)
    if br.signals:
        s = pd.DataFrame(br.signals)
        s["name"] = name
        s["phase"] = [_phase(d, ff) for d in s["date"]]
        db.upsert_df("paper_signals", s)
    if br.health_log:
        h = health_rows(br.health_log)
        h["name"] = name
        h["phase"] = [_phase(d, ff) for d in h["date"]]
        db.upsert_df("paper_health", h)
    if br.decisions:
        dd = pd.DataFrame(br.decisions)
        dd["name"] = name
        dd["phase"] = [_phase(d, ff) for d in dd["date"]]
        db.upsert_df("paper_decisions", dd)
    db.execute("UPDATE paper_accounts SET state_json=?, last_date=? WHERE name=?",
               (json.dumps(br.to_state(), ensure_ascii=False), dates[-1].strftime("%Y-%m-%d"), name))
    return {"name": name, "days": len(dates), "equity": br.equity[dates[-1]], "cash": br.cash,
            "positions": len(br.positions), "pending": len(br.pending), "trades": len(br.trades)}


def run_all(db: Database, md: MarketData, until=None) -> list[dict]:
    cache: dict = {}
    return [run_account(db, md, n, until, cache) for n in list_accounts(db).query("active == 1")["name"]]


def place_order(db: Database, md: MarketData, name: str, side: str, sid: str,
                shares: int | None = None, limit: float | None = None, note: str = "") -> dict:
    br, acc = load_broker(db, md, name)
    d = br.last_date or md.dates[-1]
    o = br.place_manual(d, sid, side, shares, limit, note)
    db.execute("UPDATE paper_accounts SET state_json=? WHERE name=?", (json.dumps(br.to_state()), name))
    db.upsert_df("paper_decisions", pd.DataFrame([{
        "name": name, "date": d.strftime("%Y-%m-%d"), "stock_id": sid, "action": f"manual_{side}",
        "reason": o.reason, "manual": 1, "phase": _phase(d, acc["forward_from"])}]))
    return {"order": o, "exec": "下一個交易日開盤撮合"}


def set_active(db: Database, name: str, active: bool) -> None:
    db.execute("UPDATE paper_accounts SET active=? WHERE name=?", (int(active), name))


def delete_account(db: Database, name: str) -> None:
    with db.tx() as c:
        for t in ("paper_accounts", "paper_equity", "paper_trades", "paper_signals", "paper_decisions", "paper_health"):
            c.execute(f"DELETE FROM {t} WHERE name=?", (name,))


# ---------------------------------------------------------------- 查詢
def status(db: Database, md: MarketData, name: str) -> dict:
    br, acc = load_broker(db, md, name)
    d = br.last_date
    holdings = pd.DataFrame(br.mark_open_trades(d)) if d is not None and br.positions else pd.DataFrame()
    pending = pd.DataFrame([{"stock_id": o.stock_id, "side": o.side, "shares": o.shares,
                             "limit": o.limit_price, "reason": o.reason} for o in br.pending])
    eq = equity(db, name)
    return {"account": acc, "holdings": holdings, "pending": pending, "equity": eq,
            "cash": br.cash, "last_date": d}


def equity(db: Database, name: str) -> pd.DataFrame:
    df = db.query("SELECT date, equity, exposure, phase FROM paper_equity WHERE name=? ORDER BY date", (name,))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def trades(db: Database, name: str | None = None, phase: str | None = None) -> pd.DataFrame:
    q, p = "SELECT * FROM paper_trades WHERE 1=1", []
    if name:
        q += " AND name=?"; p.append(name)
    if phase:
        q += " AND phase=?"; p.append(phase)
    df = db.query(q + " ORDER BY exit_date", tuple(p))
    for c in ("signal_date", "entry_date", "exit_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def signals(db: Database, name: str | None = None) -> pd.DataFrame:
    q, p = "SELECT * FROM paper_signals", ()
    if name:
        q += " WHERE name=?"; p = (name,)
    df = db.query(q, p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def health(db: Database, name: str | None = None) -> pd.DataFrame:
    q, p = "SELECT * FROM paper_health", ()
    if name:
        q += " WHERE name=?"; p = (name,)
    df = db.query(q + " ORDER BY date, stock_id", p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def leaderboard(db: Database, benchmark: pd.Series | None = None, phase: str | None = None) -> pd.DataFrame:
    """排行榜（股市大富翁式）：報酬、回撤、Calmar、交易統計與相對 0050。"""
    from . import indicators as ind
    from .stats import trade_stats
    rows = []
    for acc in list_accounts(db).itertuples():
        eq = equity(db, acc.name)
        if phase:
            eq = eq[eq["phase"] == phase]
        if len(eq) < 2:
            continue
        e = eq["equity"]
        t = trades(db, acc.name, phase)
        st = trade_stats(t["r_multiple"]) if not t.empty else None
        ret = e.iloc[-1] / e.iloc[0] - 1
        mdd = ind.max_drawdown(e)
        row = {
            "帳戶": acc.name, "規則版本": acc.rule_version, "期間": f"{e.index[0].date()}~{e.index[-1].date()}",
            "權益": e.iloc[-1], "報酬": ret, "年化": ind.cagr(e), "MDD": mdd,
            "Calmar": ind.cagr(e) / abs(mdd) if mdd < 0 else np.nan, "平均曝險": eq["exposure"].mean(),
            "交易數": st.n if st else 0, "勝率": st.win_rate if st else np.nan,
            "期望值R": st.expectancy if st else np.nan,
            "手動交易": int((t["manual_entry"] == 1).sum() + (t["manual_exit"] == 1).sum()) if not t.empty else 0,
        }
        if benchmark is not None:
            b = benchmark.reindex(e.index).ffill().dropna()
            if len(b) > 1:
                row["0050報酬"] = b.iloc[-1] / b.iloc[0] - 1
                row["超額"] = ret - row["0050報酬"]
                row["0050 MDD"] = ind.max_drawdown(b)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("Calmar", ascending=False).reset_index(drop=True) if not df.empty else df
