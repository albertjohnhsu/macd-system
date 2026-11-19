"""實盤作業流程（設計書「實盤操作手冊 → 例行作業流程」）。

每日收盤後  run_daily ：更新 L1 狀態、L3 安全旗標、追蹤停利（只上移），產生隔日賣出清單並推播
每週五收盤後 run_weekly：重算 L2 排名，產生買入清單與股數（含產業與相關分群上限）
成交回報   record_fill：寫入持股與交易日誌（R 倍數、滑價、是否依訊號執行）
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from .config import Config
from .db import Database
from .data.loader import load_market_data
from .data.market import MarketData
from .engine import SignalEngine
from .l1_regime import STATE_ZH
from .portfolio import Order, Position


# ---------------------------------------------------------------- 持股存取
def load_positions(db: Database) -> dict[str, Position]:
    df = db.query("SELECT * FROM positions")
    out = {}
    jr = db.query("SELECT * FROM trade_journal WHERE exit_date IS NULL")
    jr = jr.set_index("stock_id") if not jr.empty else jr
    for r in df.itertuples(index=False):
        j = jr.loc[r.stock_id] if not jr.empty and r.stock_id in jr.index else None
        if isinstance(j, pd.DataFrame):
            j = j.iloc[-1]
        out[r.stock_id] = Position(
            stock_id=r.stock_id, shares=int(r.shares), entry_date=pd.Timestamp(r.entry_date),
            entry_price=float(r.entry_price), initial_stop=float(r.initial_stop), one_r=float(r.one_r),
            highest_close=float(r.highest_close), trailing_stop=float(r.trailing_stop),
            catastrophic_stop=float(r.catastrophic_stop),
            pending_exit_since=pd.Timestamp(r.pending_exit_since) if r.pending_exit_since else None,
            entry_state=j["entry_l1_state"] if j is not None else "",
            entry_trend_score=float(j["entry_trend_score"]) if j is not None and pd.notna(j["entry_trend_score"]) else np.nan,
            entry_safety_score=float(j["entry_safety_score"]) if j is not None and pd.notna(j["entry_safety_score"]) else np.nan,
            health_weekly=json.loads(r.health_json) if getattr(r, "health_json", None) else {},
            weak_seen=[x for x in (getattr(r, "weak_seen", "") or "").split("、") if x],
        )
    return out


def save_positions(db: Database, positions: dict[str, Position]) -> None:
    with db.tx() as c:
        c.execute("DELETE FROM positions")
        for p in positions.values():
            c.execute(
                "INSERT INTO positions (stock_id, shares, entry_date, entry_price, initial_stop, trailing_stop,"
                " highest_close, one_r, pending_exit_since, catastrophic_stop, health_json, weak_seen)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (p.stock_id, p.shares, _d(p.entry_date), p.entry_price, p.initial_stop, p.trailing_stop,
                 p.highest_close, p.one_r, _d(p.pending_exit_since), p.catastrophic_stop,
                 json.dumps(p.health_weekly or {}, ensure_ascii=False), "、".join(p.weak_seen or [])))


def _d(x) -> str | None:
    return pd.Timestamp(x).strftime("%Y-%m-%d") if x is not None and not pd.isna(x) else None


def account_equity(db: Database, cfg: Config, md: MarketData, date) -> tuple[float, float]:
    """權益 = 初始資金 + 已實現損益 + 持股市值 − 持股成本；回傳 (權益, 現金)。"""
    jr = db.query("SELECT * FROM trade_journal")
    realized = 0.0
    cost_open = 0.0
    c = cfg.cost
    for r in jr.itertuples(index=False):
        buy_val = r.fill_price * r.shares
        buy_fee = buy_val * c.commission * c.commission_discount
        if pd.notna(r.exit_date) and r.exit_date:
            sell_val = r.exit_price * r.shares
            realized += sell_val - sell_val * (c.commission * c.commission_discount + c.tax) - buy_val - buy_fee
        else:
            cost_open += buy_val + buy_fee
    positions = load_positions(db)
    mv = 0.0
    for s, p in positions.items():
        px = md.close.loc[:pd.Timestamp(date), s].dropna() if s in md.close else pd.Series(dtype=float)
        mv += p.shares * (float(px.iloc[-1]) if len(px) else p.entry_price)
    cash = cfg.l4.total_capital + realized - cost_open
    return cash + mv, cash


# ---------------------------------------------------------------- 例行作業
def _alert(db: Database, date, kind: str, msg: str) -> None:
    db.execute("INSERT INTO alerts (date, kind, message) VALUES (?,?,?)", (_d(date), kind, msg))


def _log(db: Database, date, layer: str, sid: str | None, action: str, reason: str, detail: dict | None = None):
    db.execute("INSERT INTO decision_log (date, layer, stock_id, action, reason, detail) VALUES (?,?,?,?,?,?)",
               (_d(date), layer, sid, action, reason, json.dumps(detail or {}, ensure_ascii=False, default=str)))


def save_regime(db: Database, regime: pd.DataFrame, since=None) -> None:
    r = regime.dropna(subset=["state"])
    if since is not None:
        r = r.loc[pd.Timestamp(since):]
    ind_cols = [c for c in r.columns if c not in ("score", "state", "cap", "target_exposure")]
    rows = pd.DataFrame({
        "date": r.index.strftime("%Y-%m-%d"),
        "indicators": [json.dumps({k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()})
                       for _, row in r[ind_cols].iterrows()],
        "score": r["score"].values, "state": r["state"].values, "cap": r["cap"].values,
        "target_exposure": r["target_exposure"].values,
    })
    db.upsert_df("market_regime_daily", rows)


def run_daily(db: Database, cfg: Config, date=None, md: MarketData | None = None,
              engine: SignalEngine | None = None) -> dict:
    md = md or load_market_data(db)
    date = pd.Timestamp(date) if date else md.dates[-1]
    eng = engine or SignalEngine(md.slice(date), cfg)
    positions = load_positions(db)

    # L1
    save_regime(db, eng.regime, since=date - pd.Timedelta(days=30))
    reg = eng.regime_at(date)
    prev = eng.regime.loc[:date, "state"].dropna()
    prev_state = prev.iloc[-2] if len(prev) > 1 else None
    if prev_state is not None and prev_state != reg["state"]:
        msg = f"大盤狀態 {STATE_ZH[prev_state]} → {STATE_ZH[reg['state']]}（分數 {reg['score']:.0f}，持股上限 {reg['cap']:.0%}）"
        _alert(db, date, "regime", msg)
        _log(db, date, "L1", None, "state_change", msg)

    # 停損維護
    before = {s: p.stop for s, p in positions.items()}
    eng.update_stops(date, positions)
    for s, p in positions.items():
        if p.stop > before[s] + 1e-9:
            _log(db, date, "L4", s, "raise_stop", f"停損上移 {before[s]:.2f} → {p.stop:.2f}")

    equity, cash = account_equity(db, cfg, md, date)
    saf = eng.safety(date, full=True)
    for s, p in positions.items():
        if s in saf.index and saf.at[s, "score"] < cfg.l3.half_size_min and p.pending_exit_since is None:
            _alert(db, date, "safety", f"{s} 安全分降至 {saf.at[s, 'score']:.0f}，進入待出場")
    weekly = date.weekday() == cfg.l4.rebalance_weekday
    sells = eng.sell_signals(date, positions, equity, weekly=weekly)
    if weekly:
        save_health(db, eng.last_health)
    next_day = _next_trading_day(date)
    # 過期未成交的買單作廢（不追價）；已有待執行賣單的股票不重複下單
    db.execute("UPDATE orders SET status='expired' WHERE status='pending' AND side='buy' AND exec_date <= ?",
               (_d(date),))
    open_sells = set(db.query("SELECT stock_id FROM orders WHERE status='pending' AND side='sell'")["stock_id"])
    sells = [o for o in sells if o.stock_id not in open_sells]
    for o in sells:
        _insert_order(db, o, next_day)
        _log(db, date, "L4", o.stock_id, "sell_signal", o.reason)
        _alert(db, date, "sell", f"賣出 {o.stock_id} {o.shares} 股：{o.reason}")
    save_positions(db, positions)
    return {"date": date, "regime": reg, "equity": equity, "cash": cash, "sells": sells,
            "positions": positions, "safety": saf}


def run_weekly(db: Database, cfg: Config, date=None, md: MarketData | None = None,
               engine: SignalEngine | None = None) -> dict:
    md = md or load_market_data(db)
    date = pd.Timestamp(date) if date else md.dates[-1]
    eng = engine or SignalEngine(md.slice(date), cfg)
    positions = load_positions(db)
    equity, cash = account_equity(db, cfg, md, date)
    saf = eng.safety(date, full=True)
    tr = eng.trend(date)
    scored = tr[tr["trend_score"].notna()]
    db.upsert_df("trend_score_weekly", pd.DataFrame({
        "date": date.strftime("%Y-%m-%d"), "stock_id": scored.index,
        "components": [json.dumps({k: (None if pd.isna(v) else round(float(v), 4)) for k, v in
                                   row[["tsmom_12m", "high_52w_ratio", "rel_strength", "revenue_yoy"]].items()})
                       for _, row in scored.iterrows()],
        "trend_score": scored["trend_score"].values, "rank": scored["rank"].astype(int).values,
    }))
    db.upsert_df("safety_score_quarterly", pd.DataFrame({
        "stock_id": saf.index, "announce_date": date.strftime("%Y-%m-%d"),
        "components": saf["penalties"].values, "score": saf["score"].values,
    }))
    health = eng.health_check(date, positions, weekly=True, saf=saf)
    save_health(db, health)
    save_positions(db, positions)
    for sid, hr in health.items():
        if hr.status != "健康":
            _log(db, date, "HEALTH", sid, hr.status, hr.summary(),
                 {k: list(v) for k, v in hr.items.items()})
    pending_sells = set(db.query("SELECT stock_id FROM orders WHERE side='sell' AND status='pending'")["stock_id"])
    buys = eng.buy_signals(date, positions, equity, cash, pending_sells)
    next_day = _next_trading_day(date)
    for o in buys:
        _insert_order(db, o, next_day)
        _log(db, date, "L4", o.stock_id, "buy_signal", o.reason, o.meta)
        _alert(db, date, "buy", f"買入 {o.stock_id} {o.shares} 股，限價 {o.limit_price}，"
                                f"停損 {o.meta['stop']:.2f}（{o.reason}）")
    return {"date": date, "buys": buys, "trend": tr, "safety": saf, "equity": equity, "health": health}


def save_health(db: Database, health: dict) -> None:
    """每週健檢結果寫入 health_checks（交易日誌的健檢紀錄）。"""
    if not health:
        return
    from .paper import health_rows
    db.upsert_df("health_checks", health_rows([hr.to_row() for hr in health.values()]))


def health_table(health: dict) -> pd.DataFrame:
    rows = []
    for sid, hr in health.items():
        r = {"代碼": sid, "狀態": hr.status, "觸發": "、".join(hr.triggers()) or "—"}
        for k, (st, note) in hr.items.items():
            r[k] = f"{st}｜{note}"
        rows.append(r)
    order = {"結束": 0, "減弱": 1, "健康": 2}
    df = pd.DataFrame(rows)
    return df.sort_values("狀態", key=lambda s: s.map(order)).reset_index(drop=True) if not df.empty else df


def _next_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    return d + pd.offsets.BDay(1)


def _insert_order(db: Database, o: Order, exec_date) -> None:
    db.execute(
        "INSERT INTO orders (created_at, exec_date, stock_id, side, shares, limit_price, signal_price, reason)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), _d(exec_date), o.stock_id, o.side, o.shares,
         o.limit_price, o.signal_price,
         json.dumps({"reason": o.reason, "signal_date": _d(o.signal_date), **o.meta}, ensure_ascii=False, default=str)))


# ---------------------------------------------------------------- 成交回報
def record_fill(db: Database, cfg: Config, order_id: int, fill_date: str, fill_price: float,
                shares: int | None = None, followed: bool = True, deviation: str | None = None) -> dict:
    """登錄成交。買單建立持股與日誌；賣單結算 R 倍數。"""
    od = db.query("SELECT * FROM orders WHERE id = ?", (order_id,))
    if od.empty:
        raise KeyError(f"找不到委託 #{order_id}")
    o = od.iloc[0]
    meta = json.loads(o["reason"]) if o["reason"] and o["reason"].startswith("{") else {"reason": o["reason"]}
    shares = int(shares or o["shares"])
    if o["side"] == "buy":
        stop = min(float(meta.get("stop", fill_price * 0.92)), fill_price * 0.999)
        one_r = shares * (fill_price - stop)
        p = Position(o["stock_id"], shares, pd.Timestamp(fill_date), fill_price, stop, one_r, fill_price, stop,
                     fill_price * (1 - cfg.l4.catastrophic_stop), meta.get("industry", ""),
                     float(meta.get("size_mult", 1.0)))
        positions = load_positions(db)
        positions[p.stock_id] = p
        save_positions(db, positions)
        db.execute(
            "INSERT INTO trade_journal (stock_id, signal_date, signal_price, fill_date, fill_price, shares, initial_stop,"
            " one_r, entry_l1_state, entry_trend_score, entry_safety_score, entry_slippage, followed_signal,"
            " deviation_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.stock_id, meta.get("signal_date") or o["created_at"][:10], o["signal_price"], fill_date, fill_price, shares, stop, one_r,
             meta.get("state"), meta.get("trend_score"), meta.get("safety_score"),
             fill_price / o["signal_price"] - 1 if o["signal_price"] else None, int(followed), deviation))
        result = {"position": p, "catastrophic_stop": p.catastrophic_stop}
    else:
        positions = load_positions(db)
        p = positions.pop(o["stock_id"], None)
        if p is None:
            raise KeyError(f"沒有 {o['stock_id']} 的持股")
        jr = db.query("SELECT id, fill_price, one_r, shares FROM trade_journal WHERE stock_id=? AND exit_date IS NULL"
                      " ORDER BY id DESC LIMIT 1", (p.stock_id,))
        c = cfg.cost
        gross = (fill_price - p.entry_price) * p.shares
        fees = (fill_price * p.shares) * (c.commission * c.commission_discount + c.tax) + \
               p.entry_price * p.shares * c.commission * c.commission_discount
        r_mult = (gross - fees) / p.one_r if p.one_r > 0 else None
        if not jr.empty:
            db.execute(
                "UPDATE trade_journal SET exit_date=?, exit_price=?, exit_reason=?, r_multiple=?, exit_slippage=?,"
                " followed_signal=MIN(followed_signal, ?), deviation_reason=COALESCE(?, deviation_reason),"
                " weak_seen=? WHERE id=?",
                (fill_date, fill_price, meta.get("reason"), r_mult,
                 1 - fill_price / o["signal_price"] if o["signal_price"] else None, int(followed), deviation,
                 "、".join(p.weak_seen or []), int(jr.iloc[0]["id"])))
        save_positions(db, positions)
        result = {"r_multiple": r_mult}
    db.execute("UPDATE orders SET status='filled' WHERE id=?", (order_id,))
    _log(db, fill_date, "EXEC", o["stock_id"], f"fill_{o['side']}", f"成交 {shares} 股 @ {fill_price}",
         {"order_id": int(order_id), "followed": followed, "deviation": deviation})
    return result


def cancel_order(db: Database, order_id: int, reason: str) -> None:
    db.execute("UPDATE orders SET status='cancelled' WHERE id=?", (order_id,))
    _log(db, datetime.now(), "EXEC", None, "cancel", reason, {"order_id": order_id})


def journal(db: Database) -> pd.DataFrame:
    return db.query("SELECT * FROM trade_journal ORDER BY id")


def equity_curve_from_journal(db: Database, cfg: Config) -> pd.Series:
    """以已實現 R 值近似實盤資金曲線（每筆結算時更新）。"""
    jr = db.query("SELECT exit_date, r_multiple, one_r FROM trade_journal WHERE exit_date IS NOT NULL ORDER BY exit_date")
    if jr.empty:
        return pd.Series(dtype=float)
    eq = cfg.l4.total_capital + (jr["r_multiple"] * jr["one_r"]).cumsum()
    return pd.Series(eq.values, index=pd.to_datetime(jr["exit_date"]))
