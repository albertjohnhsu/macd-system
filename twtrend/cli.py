"""命令列介面：`python -m twtrend <command>` 或安裝後 `twtrend <command>`。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import Config
from .db import Database

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def _ctx(args) -> tuple[Config, Database]:
    cfg = Config.load(args.config)
    if args.db:
        cfg.db_path = args.db
    cfg.finmind_token = cfg.finmind_token or os.getenv("FINMIND_TOKEN")
    return cfg, Database(cfg.db_path)


# ---------------------------------------------------------------- 資料
def cmd_demo(args):
    from .data import synthetic
    cfg, db = _ctx(args)
    data = synthetic.generate(n_stocks=args.stocks, start=args.start, end=args.end, seed=args.seed)
    synthetic.write_to_db(db, data)
    print(f"已寫入合成資料到 {cfg.db_path}：{args.stocks} 檔、{args.start} ~ {args.end}")
    print(data["_kinds"]["kind"].value_counts().rename("股票類型").to_string())


def cmd_update(args):
    from .data.finmind import FinMindClient
    from .data.loader import update_from_finmind
    cfg, db = _ctx(args)
    client = FinMindClient(cfg.finmind_token)
    stocks = args.stocks.split(",") if args.stocks else None
    rep = update_from_finmind(db, client, args.start, args.end, stocks, cfg.benchmark,
                              with_fundamentals=not args.prices_only)
    print(f"更新 {rep.get('stocks', 0)} 檔，錯誤 {len(rep['errors'])} 筆")
    for e in rep["errors"][:20]:
        print("  ", e)


def cmd_flags(args):
    from .data.loader import update_flags
    cfg, db = _ctx(args)
    errors = update_flags(db, args.date)
    print("安全旗標已更新" + (f"；{len(errors)} 個來源失敗：" if errors else ""))
    for e in errors:
        print("  ", e)


# ---------------------------------------------------------------- 實盤
def cmd_daily(args):
    from . import live, notify
    cfg, db = _ctx(args)
    out = live.run_daily(db, cfg, args.date)
    reg = out["regime"]
    from .l1_regime import STATE_ZH
    print(f"[{out['date'].date()}] 大盤：{STATE_ZH[reg['state']]}　分數 {reg['score']:.0f}　"
          f"上限 {reg['cap']:.0%}　目標曝險 {reg['target_exposure']:.0%}")
    print(f"權益 {out['equity']:,.0f}　現金 {out['cash']:,.0f}　持股 {len(out['positions'])} 檔")
    if out["sells"]:
        print("隔日賣出清單：")
        for o in out["sells"]:
            print(f"  {o.stock_id}  {o.shares:>7,} 股  {o.reason}")
    else:
        print("隔日賣出清單：無")
    notify.flush_alerts(db, dry_run=args.dry_run)


def cmd_weekly(args):
    from . import live, notify
    cfg, db = _ctx(args)
    out = live.run_weekly(db, cfg, args.date)
    tr = out["trend"]
    print(f"[{out['date'].date()}] 趨勢分排名（L3 通過者）前 15：")
    cols = ["trend_score", "rank", "high_52w_ratio", "tsmom_12m", "rel_strength", "revenue_yoy"]
    print(tr[tr["trend_score"].notna()][cols].head(15).round(3).to_string())
    if out.get("health"):
        _print_health(out["health"])
    if out["buys"]:
        print("下週一買入清單（限價、未成交不追價）：")
        for o in out["buys"]:
            m = o.meta
            print(f"  {o.stock_id}  {o.shares:>7,} 股  限價 {o.limit_price:>8.2f}  初始停損 {m['stop']:.2f}  "
                  f"極端停損 {o.signal_price * (1 - cfg.l4.catastrophic_stop):.2f}  安全分 {m['safety_score']:.0f}"
                  f"{'（部位減半）' if m['size_mult'] < 1 else ''}")
    else:
        print("買入清單：無（L1 不允許新增、或無合格候選 / 可用資金）")
    notify.flush_alerts(db, dry_run=args.dry_run)


def _print_health(health: dict) -> None:
    from . import live
    df = live.health_table(health)
    if df.empty:
        return
    n = df["狀態"].value_counts()
    print(f"持股趨勢健檢：健康 {n.get('健康', 0)}、減弱 {n.get('減弱', 0)}、結束 {n.get('結束', 0)}"
          "（只有「結束」才出場；「減弱」續抱、不加碼、減碼時優先賣）")
    for _, r in df.iterrows():
        print(f"  {r['代碼']}  {r['狀態']}  {r['觸發']}")
        for k in [c for c in df.columns if c not in ("代碼", "狀態", "觸發")]:
            v = r.get(k)
            if isinstance(v, str) and not v.startswith("健康"):
                print(f"      {k}：{v}")


def cmd_health(args):
    from . import live
    from .data.loader import load_market_data
    from .engine import SignalEngine
    cfg, db = _ctx(args)
    md = load_market_data(db)
    date = pd.Timestamp(args.date) if args.date else md.dates[-1]
    eng = SignalEngine(md.slice(date), cfg)
    positions = live.load_positions(db)
    if not positions:
        print("目前沒有持股")
        return
    health = eng.health_check(date, positions, weekly=True)
    live.save_health(db, health)
    live.save_positions(db, positions)
    print(f"[{date.date()}]")
    _print_health(health)


def cmd_orders(args):
    cfg, db = _ctx(args)
    df = db.query("SELECT id, exec_date, stock_id, side, shares, limit_price, signal_price, status, reason "
                  "FROM orders WHERE status = ? ORDER BY id", (args.status,))
    df["reason"] = df["reason"].map(lambda s: json.loads(s).get("reason", s) if s and s.startswith("{") else s)
    print(df.to_string(index=False) if not df.empty else "（無）")


def cmd_fill(args):
    from . import live
    cfg, db = _ctx(args)
    res = live.record_fill(db, cfg, args.order_id, args.date or pd.Timestamp.today().strftime("%Y-%m-%d"),
                           args.price, args.shares, followed=not args.deviation, deviation=args.deviation)
    if "position" in res:
        p = res["position"]
        print(f"已建立持股 {p.stock_id} {p.shares} 股 @ {p.entry_price}；初始停損 {p.initial_stop:.2f}；"
              f"1R = {p.one_r:,.0f} 元；請設定極端保護停損條件單 {res['catastrophic_stop']:.2f}")
    else:
        print(f"已平倉，R 倍數 {res['r_multiple']:+.2f}" if res["r_multiple"] is not None else "已平倉")


def cmd_cancel(args):
    from . import live
    cfg, db = _ctx(args)
    live.cancel_order(db, args.order_id, args.reason)
    print(f"委託 #{args.order_id} 已取消")


def cmd_journal(args):
    from . import live
    cfg, db = _ctx(args)
    df = live.journal(db)
    print(df.to_string(index=False) if not df.empty else "（無交易紀錄）")
    if args.csv:
        df.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"已輸出 {args.csv}")


def cmd_monitor(args):
    from . import live
    from .monitor import STAGES, evaluate_lights, lights_frame, next_stage, overall
    cfg, db = _ctx(args)
    jr = live.journal(db)
    eq = live.equity_curve_from_journal(db, cfg)
    lights = evaluate_lights(jr, eq, cfg.monitor)
    print(lights_frame(lights)[["level", "metric", "value", "yellow", "red", "note"]].round(3).to_string(index=False))
    print(f"\n整體：{overall(lights)}")
    n = int(jr["r_multiple"].notna().sum()) if not jr.empty else 0
    from .review import deviation_summary
    dev = deviation_summary(jr)
    if not dev.empty:
        print("\n偏離訊號檢討（每月）：")
        print(dev.round(2).to_string(index=False))
    stage, msg = next_stage(args.stage, lights, n)
    print(f"資金階段建議：{STAGES[stage]['name']}（每筆風險 {STAGES[stage]['risk']:.2%}）— {msg}")


def cmd_dashboard(args):
    from . import dashboard, live
    from .data.loader import load_market_data
    from .engine import SignalEngine
    from .monitor import evaluate_lights
    cfg, db = _ctx(args)
    md = load_market_data(db)
    date = pd.Timestamp(args.date) if args.date else md.dates[-1]
    eng = SignalEngine(md.slice(date), cfg)
    saf = eng.safety(date, full=True)
    tr = eng.trend(date)
    positions = live.load_positions(db)
    health = eng.health_check(date, positions, weekly=True, saf=saf)
    jr = live.journal(db)
    eq = live.equity_curve_from_journal(db, cfg)
    orders = db.query("SELECT exec_date AS 執行日, stock_id AS 代碼, side AS 買賣, shares AS 股數, "
                      "limit_price AS 限價 FROM orders WHERE status='pending' ORDER BY id")
    closed = jr[jr["r_multiple"].notna()].rename(columns={"exit_date": "exit_date"}) if not jr.empty else None
    page = dashboard.render(
        eng.regime, dashboard.candidates_frame(tr, saf), dashboard.holdings_frame(positions, md, saf, date, health),
        orders, closed, eq if len(eq) > 1 else None, evaluate_lights(jr, eq, cfg.monitor),
        {"p95": cfg.monitor.mdd_yellow, "p99": cfg.monitor.mdd_red}, title="台股趨勢追蹤 · 實盤監控")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"儀表板已輸出：{out}")


# ---------------------------------------------------------------- 回測與統計
def cmd_backtest(args):
    from . import dashboard, stats
    from .backtest import BEAR_PERIODS, Backtester, compare_benchmarks, period_report, split_2015
    from .data.loader import load_benchmark, load_market_data
    from .monitor import evaluate_lights
    cfg, db = _ctx(args)
    md = load_market_data(db)
    bench = load_benchmark(db, cfg.benchmark)
    res = Backtester(md, cfg).run(args.start, args.end)
    cmp_ = compare_benchmarks(md, cfg, res, bench)
    print("== 三個基準比較（以 Calmar 為主）==")
    print(cmp_.round(3).to_string())
    pr = period_report(res.equity, bench, BEAR_PERIODS)
    if not pr.empty:
        print("\n== 歷次空頭測試期 ==")
        print(pr.round(3).to_string())
    sp = split_2015(res.equity)
    if len(sp) > 1:
        print("\n== 2015/6 漲跌幅制度前後 ==")
        print(sp.round(3).to_string())
    closed = res.trades[~res.trades["open_at_end"]] if not res.trades.empty else res.trades
    if not closed.empty:
        st = stats.trade_stats(closed["r_multiple"])
        mean, lo, hi = stats.bootstrap_expectancy_ci(closed["r_multiple"], closed["exit_date"])
        print("\n== 交易統計（R）==")
        for k, v in st.to_dict().items():
            print(f"  {k:>18}: {v:.3f}" if isinstance(v, float) else f"  {k:>18}: {v}")
        print(f"  期望值 95% CI（區塊 bootstrap）：{lo:+.3f} ~ {hi:+.3f} → {'有優勢' if lo > 0 else '尚無法證明優勢'}")
        print(f"  95% 顯著所需筆數：{stats.required_trades(st.expectancy, st.std)}；t≥3 所需：{stats.required_trades(st.expectancy, st.std, 3)}")
        print(f"  期望值打對折後：{st.expectancy / 2:+.3f}R（{'仍為正' if st.expectancy > 0 else '為負'}）")
        mc = stats.mc_reorder(closed["r_multiple"], cfg.l4.risk_per_trade)
        print(f"  交易順序重排 MC：MDD 中位數 {-mc['mdd'].median():.1%}、95% {-mc['mdd'].quantile(0.05):.1%}；"
              f"最長連虧 95% {mc['max_streak'].quantile(0.95):.0f} 筆")
        print("\n== 出場原因 ==")
        print(closed["exit_reason"].str.split("：|（").str[0].value_counts().to_string())
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        res.trades.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"equity": res.equity, "exposure": res.exposure}).to_csv(out / "equity.csv", encoding="utf-8-sig")
        cmp_.to_csv(out / "benchmarks.csv", encoding="utf-8-sig")
        bh = bench.reindex(res.equity.index).ffill()
        bh = bh / bh.iloc[0] * res.equity.iloc[0]
        mc_b = None
        if not closed.empty:
            mc = stats.mc_reorder(closed["r_multiple"], cfg.l4.risk_per_trade)
            mc_b = {"p95": float(-mc["mdd"].quantile(0.05)), "p99": float(-mc["mdd"].quantile(0.01))}
        page = dashboard.render(res.regime, trades=closed, equity=res.equity,
                                lights=evaluate_lights(closed, res.equity, cfg.monitor), mc_bands=mc_b,
                                title="台股趨勢追蹤 · 回測報告",
                                note=f"{res.equity.index[0].date()} ~ {res.equity.index[-1].date()}")
        page = page.replace("<footer", "<section><h2>完整系統 vs 0050</h2>" + dashboard.line_chart(
            {"完整系統": res.equity, "0050 買進持有": bh}, fmt="{:,.0f}") +
            "<h3>三個基準</h3>" + dashboard.table(cmp_.reset_index().rename(columns={"index": "策略"}),
                                                {c: "pct" for c in ("CAGR", "MDD", "Vol", "多頭參與率", "空頭平均持股")}) +
            "</section><footer")
        (out / "report.html").write_text(page, encoding="utf-8")
        print(f"\n已輸出 {out}/trades.csv、equity.csv、benchmarks.csv、report.html")


def cmd_simulate(args):
    from . import stats
    mat = stats.simulate_paths(n_paths=args.paths, n_trades=args.trades, win_rate=args.win_rate, seed=args.seed)
    flat = mat.ravel()
    st = stats.trade_stats(flat[: min(len(flat), 400_000)])
    print("== 損益兩平勝率 ==")
    print(stats.breakeven_table(cost_r=0.1).round(3).to_string(index=False))
    print(f"\n== 模型期望值 ==\n  勝率 {st.win_rate:.1%}、盈虧比 {st.payoff:.2f}、期望值 {st.expectancy:+.2f}R、標準差 {st.std:.2f}R")
    print(f"  前 5% 交易占淨利 {st.top5_share:.0%}；去掉前 2% 後期望值 {st.exp_without_top2:+.2f}R")
    sd = stats.streak_distribution(mat)
    print(f"\n== 連虧（{args.trades} 筆）==\n  中位數 {sd['median']:.0f}、95% {sd['p95']:.0f}；"
          f"≥10 筆機率 {sd['prob_ge10']:.0%}、≥15 筆 {sd['prob_ge15']:.0%}")
    print("\n== 每筆風險 vs 回撤 ==")
    print(stats.risk_table(mat).round(3).to_string(index=False))
    print(f"\nKelly 最適比例約 {stats.kelly_fraction(flat[:200_000]):.1%}（實務用 1/10–1/15 Kelly）")
    print(f"95% 顯著所需筆數 {stats.required_trades(st.expectancy, st.std)}；t≥3 需 {stats.required_trades(st.expectancy, st.std, 3)}")


def cmd_config(args):
    cfg = Config()
    print(yaml.safe_dump(json.loads(json.dumps(cfg.to_dict(), default=list)), allow_unicode=True, sort_keys=False))


# ---------------------------------------------------------------- 模擬盤
def _paper_md(db):
    from .data.loader import load_market_data
    return load_market_data(db)


def cmd_paper_create(args):
    from . import paper
    cfg, db = _ctx(args)
    info = paper.create_account(db, args.name, cfg, args.capital, args.start, args.set, args.note, args.manual)
    print(f"已建立模擬帳戶 {info['name']}（規則版本 {info['rule_version']}），起始日 {info['start_date']}")
    if info["overrides"]:
        print("  參數覆寫：" + "、".join(f"{k}={v}" for k, v in info["overrides"].items()))
    print("  執行 `twtrend paper run` 開始撮合（起始日早於今天的部分會標記為 replay 歷史重播）")


def cmd_paper_list(args):
    from . import paper
    cfg, db = _ctx(args)
    df = paper.list_accounts(db)
    print(df.to_string(index=False) if not df.empty else "（尚無模擬帳戶，請用 `twtrend paper create`）")


def cmd_paper_run(args):
    from . import paper
    cfg, db = _ctx(args)
    md = _paper_md(db)
    if args.name:
        res = [paper.run_account(db, md, args.name, args.until)]
    else:
        res = paper.run_all(db, md, args.until)
    for r in res:
        if not r.get("days"):
            print(f"{r['name']}：已是最新")
            continue
        print(f"{r['name']}：推進 {r['days']} 日 → 權益 {r['equity']:,.0f}、現金 {r['cash']:,.0f}、"
              f"持股 {r['positions']} 檔、待撮合 {r['pending']} 筆、本次結算 {r['trades']} 筆")


def cmd_paper_order(args):
    from . import paper
    cfg, db = _ctx(args)
    out = paper.place_order(db, _paper_md(db), args.name, args.side, args.stock, args.shares, args.limit, args.note)
    o = out["order"]
    print(f"{args.name}：手動{'買進' if o.side == 'buy' else '賣出'} {o.stock_id} {o.shares} 股"
          f"{f'（限價 {o.limit_price}）' if o.limit_price else ''}，{out['exec']}；此筆會標記為人為干預")


def cmd_paper_status(args):
    from . import paper
    cfg, db = _ctx(args)
    st = paper.status(db, _paper_md(db), args.name)
    acc = st["account"]
    eq = st["equity"]
    print(f"帳戶 {acc['name']}（規則 {acc['rule_version']}）　最後撮合日 {st['last_date'].date() if st['last_date'] is not None else '—'}")
    if not eq.empty:
        e = eq["equity"]
        print(f"權益 {e.iloc[-1]:,.0f}（{e.iloc[-1] / acc['capital'] - 1:+.2%}）　現金 {st['cash']:,.0f}　"
              f"最大回撤 {(e / e.cummax() - 1).min():.2%}")
    if not st["holdings"].empty:
        h = st["holdings"][["stock_id", "shares", "entry_date", "entry_price", "close", "stop", "r_multiple", "manual_entry"]]
        print("持股：")
        print(h.round({"entry_price": 2, "close": 2, "stop": 2, "r_multiple": 2}).to_string(index=False))
    ph = paper.health(db, args.name)
    if not ph.empty and not st["holdings"].empty:
        last = ph[ph["stock_id"].isin(st["holdings"]["stock_id"])].sort_values("date").groupby("stock_id").last()
        print(f"最近一次持股健檢：")
        for sid, r in last.iterrows():
            print(f"  {sid}  {r['date'].date()}  {r['status']}  {r['triggers'] or ''}")
    if not st["pending"].empty:
        print("待撮合委託：")
        print(st["pending"].to_string(index=False))


def cmd_paper_board(args):
    from . import paper
    from .data.loader import load_benchmark
    cfg, db = _ctx(args)
    b = paper.leaderboard(db, load_benchmark(db, cfg.benchmark), args.phase)
    print(b.round(3).to_string(index=False) if not b.empty else "（尚無績效）")


def cmd_paper_review(args):
    from . import paper, review
    cfg, db = _ctx(args)
    md = _paper_md(db)
    t = paper.trades(db, args.name, args.phase)
    sig = paper.signals(db, args.name)
    hl = paper.health(db, args.name)
    if args.phase and not hl.empty:
        hl = hl[hl["phase"] == args.phase]
    print(f"== 規則檢討：{args.name}（{args.phase or '全部'}，{len(t)} 筆已結算）==")
    print(review.findings_frame(review.findings(t, md, sig, hl)).to_string(index=False))
    if not hl.empty:
        print("\n-- 持股健檢：減弱後的實際走勢（對照同期健康持股）--")
        print(review.weak_followup(hl, md).round(3).to_string(index=False))
        wo = review.weak_trade_outcome(t)
        if not wo.empty:
            print("\n-- 持股健檢：曾減弱 vs 未減弱的交易最終 R --")
            print(wo.round(3).to_string(index=False))
        he = review.health_exit_outcome(t, md)
        if not he.empty:
            print("\n-- 持股健檢：各「結束」項目出場結果 --")
            print(he.round(3).to_string(index=False))
    if not t.empty:
        for by in ("出場原因", "安全分組", "趨勢分組", "進場狀態"):
            df = review.attribution(t, by)
            if not df.empty:
                print(f"\n-- 分組：{by} --")
                print(df.round(3).to_string(index=False))
        print("\n-- 出場後走勢 --")
        print(review.exit_followup(t, md).round(3).to_string(index=False))
    if not sig.empty:
        print("\n-- 候選股事後報酬 --")
        print(review.candidate_followup(sig, md).round(3).to_string(index=False))


def cmd_paper_compare(args):
    from . import review
    cfg, db = _ctx(args)
    df = review.compare_accounts(db, args.base, args.others, args.phase)
    print(df.round(3).to_string(index=False))


def cmd_paper_report(args):
    from . import dashboard, paper, review
    from .data.loader import load_benchmark
    cfg, db = _ctx(args)
    md = _paper_md(db)
    bench = load_benchmark(db, cfg.benchmark)
    board = paper.leaderboard(db, bench, args.phase)
    accounts = paper.list_accounts(db)["name"].tolist()
    curves = {n: paper.equity(db, n)["equity"] for n in accounts}
    if curves:
        idx = max(curves.values(), key=len).index
        if len(idx):
            curves["0050 買進持有"] = bench.reindex(idx).ffill()
    focus = args.name or (board.iloc[0]["帳戶"] if not board.empty else None)
    fs, attr, efu, cfu, ab = pd.DataFrame(), {}, None, None, None
    if focus:
        t = paper.trades(db, focus, args.phase)
        sig = paper.signals(db, focus)
        hl = paper.health(db, focus)
        if args.phase and not hl.empty:
            hl = hl[hl["phase"] == args.phase]
        fs = review.findings_frame(review.findings(t, md, sig, hl))
        fs.insert(1, "帳戶", focus)
        if not t.empty:
            attr = {f"{focus} · {by}": review.attribution(t, by) for by in ("出場原因", "安全分組", "趨勢分組", "進場狀態", "手動")}
            efu = review.exit_followup(t, md)
            for title, df in ((f"{focus} · 健檢：減弱後走勢 vs 健康持股", review.weak_followup(hl, md)),
                              (f"{focus} · 健檢：曾減弱 vs 未減弱的交易 R", review.weak_trade_outcome(t)),
                              (f"{focus} · 健檢：各結束項目出場結果", review.health_exit_outcome(t, md))):
                if df is not None and not df.empty:
                    attr[title] = df
        cfu = review.candidate_followup(sig, md) if not sig.empty else None
        others = [n for n in accounts if n != focus]
        if others:
            ab = review.compare_accounts(db, focus, others, args.phase)
    page = dashboard.render_paper(board, curves, fs, attr, efu, cfu, ab,
                                  note=f"檢討帳戶：{focus}" if focus else "")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"模擬盤報告已輸出：{out}")


def cmd_paper_toggle(args):
    from . import paper
    cfg, db = _ctx(args)
    if args.action == "delete":
        paper.delete_account(db, args.name)
        print(f"已刪除模擬帳戶 {args.name} 與其紀錄")
    else:
        paper.set_active(db, args.name, args.action == "resume")
        print(f"{args.name} 已{'恢復' if args.action == 'resume' else '暫停'}")


def add_paper_parser(sub):
    pp = sub.add_parser("paper", help="模擬盤（類似股市大富翁）：虛擬帳戶、自動撮合、績效與規則檢討")
    ps = pp.add_subparsers(dest="paper_cmd", required=True)

    s = ps.add_parser("create", help="建立模擬帳戶（可用 --set 覆寫參數做 A/B 規則比較）")
    s.add_argument("name")
    s.add_argument("--capital", type=float)
    s.add_argument("--start", help="起始日；早於今天的部分視為 replay 歷史重播")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="參數覆寫，例如 --set l4.trailing_atr_mult=3.5，可重複")
    s.add_argument("--note", default="")
    s.add_argument("--manual", action="store_true", help="純手動帳戶：不產生系統訊號，只撮合你的委託")
    s.set_defaults(func=cmd_paper_create)

    s = ps.add_parser("list", help="列出模擬帳戶")
    s.set_defaults(func=cmd_paper_list)

    s = ps.add_parser("run", help="撮合到最新資料日（每日收盤資料更新後執行）")
    s.add_argument("name", nargs="?")
    s.add_argument("--until")
    s.set_defaults(func=cmd_paper_run)

    s = ps.add_parser("order", help="手動下單（下一個交易日開盤撮合，標記為人為干預）")
    s.add_argument("name")
    s.add_argument("side", choices=["buy", "sell"])
    s.add_argument("stock")
    s.add_argument("shares", type=int, nargs="?")
    s.add_argument("--limit", type=float)
    s.add_argument("--note", default="")
    s.set_defaults(func=cmd_paper_order)

    s = ps.add_parser("status", help="帳戶持股、委託與權益")
    s.add_argument("name")
    s.set_defaults(func=cmd_paper_status)

    s = ps.add_parser("board", help="排行榜")
    s.add_argument("--phase", choices=["forward", "replay"])
    s.set_defaults(func=cmd_paper_board)

    s = ps.add_parser("review", help="規則檢討：分組績效、出場後走勢、候選機會成本、修正建議")
    s.add_argument("name")
    s.add_argument("--phase", choices=["forward", "replay"])
    s.set_defaults(func=cmd_paper_review)

    s = ps.add_parser("compare", help="A/B 規則比較（B − A 期望值差與 CI）")
    s.add_argument("base")
    s.add_argument("others", nargs="+")
    s.add_argument("--phase", choices=["forward", "replay"])
    s.set_defaults(func=cmd_paper_compare)

    s = ps.add_parser("report", help="輸出模擬盤 HTML 報告")
    s.add_argument("--name", help="規則檢討的主帳戶（預設排行第一）")
    s.add_argument("--phase", choices=["forward", "replay"])
    s.add_argument("--out", default="output/paper.html")
    s.set_defaults(func=cmd_paper_report)

    for act in ("pause", "resume", "delete"):
        s = ps.add_parser(act, help={"pause": "暫停帳戶", "resume": "恢復帳戶", "delete": "刪除帳戶與紀錄"}[act])
        s.add_argument("name")
        s.set_defaults(func=cmd_paper_toggle, action=act)


# ---------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="twtrend", description="台股趨勢追蹤系統")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--db", default=None, help="SQLite 路徑（覆寫設定檔）")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo", help="產生合成資料（無法連線資料源時展示用）")
    s.add_argument("--stocks", type=int, default=60)
    s.add_argument("--start", default="2014-01-01")
    s.add_argument("--end", default="2024-12-31")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("update", help="從 FinMind 更新資料（需 FINMIND_TOKEN）")
    s.add_argument("--start", required=True)
    s.add_argument("--end")
    s.add_argument("--stocks", help="逗號分隔；省略則抓全部")
    s.add_argument("--prices-only", action="store_true")
    s.set_defaults(func=cmd_update)

    s = sub.add_parser("flags", help="從證交所/櫃買 OpenAPI 更新處置、注意、全額交割旗標")
    s.add_argument("--date")
    s.set_defaults(func=cmd_flags)

    for name, fn, h in (("daily", cmd_daily, "每日收盤後：L1、安全旗標、追蹤停利、賣出清單"),
                        ("weekly", cmd_weekly, "每週五收盤後：L2 排名與買入清單")):
        s = sub.add_parser(name, help=h)
        s.add_argument("--date")
        s.add_argument("--dry-run", action="store_true", help="只印出提醒，不推播")
        s.set_defaults(func=fn)

    s = sub.add_parser("health", help="持股趨勢健檢：每檔持股 健康 / 減弱 / 結束 與觸發項目")
    s.add_argument("--date")
    s.set_defaults(func=cmd_health)

    s = sub.add_parser("orders", help="列出委託")
    s.add_argument("--status", default="pending")
    s.set_defaults(func=cmd_orders)

    s = sub.add_parser("fill", help="登錄成交")
    s.add_argument("order_id", type=int)
    s.add_argument("price", type=float)
    s.add_argument("--date")
    s.add_argument("--shares", type=int)
    s.add_argument("--deviation", help="若未依訊號執行，寫下原因")
    s.set_defaults(func=cmd_fill)

    s = sub.add_parser("cancel", help="取消委託（如開盤超過限價未成交）")
    s.add_argument("order_id", type=int)
    s.add_argument("--reason", default="未成交，不追價")
    s.set_defaults(func=cmd_cancel)

    s = sub.add_parser("journal", help="交易日誌")
    s.add_argument("--csv")
    s.set_defaults(func=cmd_journal)

    s = sub.add_parser("monitor", help="紅黃燈與資金階段建議")
    s.add_argument("--stage", type=int, default=1, choices=[0, 1, 2])
    s.set_defaults(func=cmd_monitor)

    s = sub.add_parser("dashboard", help="輸出實盤監控儀表板 HTML")
    s.add_argument("--date")
    s.add_argument("--out", default="output/dashboard.html")
    s.set_defaults(func=cmd_dashboard)

    s = sub.add_parser("backtest", help="事件驅動回測與三基準比較")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--out", default="output/backtest")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("simulate", help="重現設計書的統計模擬（多次小虧、靠大趨勢回補）")
    s.add_argument("--paths", type=int, default=10_000)
    s.add_argument("--trades", type=int, default=200)
    s.add_argument("--win-rate", type=float, default=0.35)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_simulate)

    add_paper_parser(sub)

    s = sub.add_parser("config", help="印出預設參數（可存成 YAML 後修改）")
    s.set_defaults(func=cmd_config)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
