import pandas as pd

from twtrend import live
from twtrend.config import Config
from twtrend.data import synthetic
from twtrend.data.loader import load_market_data
from twtrend.db import Database


def test_live_cycle(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    synthetic.write_to_db(db, synthetic.generate(n_stocks=30, start="2016-01-01", end="2019-12-31", seed=3))
    cfg = Config()
    md = load_market_data(db)
    from twtrend.l1_regime import compute_regime
    reg = compute_regime(md, cfg.l1)
    fridays = reg[(reg["state"] == "BULL") & (reg.index.weekday == 4)].index
    buys = []
    for d in fridays[::-1]:
        out = live.run_weekly(db, cfg, d, md=md)
        if out["buys"]:
            buys = out["buys"]
            break
    assert buys, "應至少有一週產生買入清單"
    orders = db.query("SELECT * FROM orders WHERE status='pending' AND side='buy'")
    o = orders.iloc[0]
    fill_date = (pd.Timestamp(o["exec_date"])).strftime("%Y-%m-%d")
    res = live.record_fill(db, cfg, int(o["id"]), fill_date, float(o["signal_price"]) * 1.002)
    assert res["position"].one_r > 0
    pos = live.load_positions(db)
    assert o["stock_id"] in pos
    # 每日流程不會出錯，且停損只上移
    before = pos[o["stock_id"]].stop
    later = md.dates[md.dates > pd.Timestamp(fill_date)][:10]
    for d in later:
        live.run_daily(db, cfg, d, md=md)
    after = live.load_positions(db)
    if o["stock_id"] in after:
        assert after[o["stock_id"]].stop >= before - 1e-9
    j = live.journal(db)
    assert len(j) == 1 and j.iloc[0]["signal_date"] < fill_date
    eq, cash = live.account_equity(db, cfg, md, later[-1])
    assert eq == eq and eq > 0
