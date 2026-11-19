import numpy as np
import pandas as pd
import pytest

from twtrend.config import Config
from twtrend.engine import SignalEngine
from twtrend.health import END, HEALTHY, WEAK
from twtrend.portfolio import Position


def _find(md, eng, cond):
    """找一個 (日期, 股票) 符合條件的組合。"""
    hc = eng.health
    for d in md.dates[300::10]:
        for sid in md.stocks:
            c = md.close.at[d, sid]
            if np.isnan(c):
                continue
            if cond(c, hc.ma100.at[d, sid], hc.ret12.at[d, sid]):
                return d, sid
    pytest.skip("找不到符合條件的樣本")


def _pos(md, sid, d, weeks_ago=4, reached=True):
    entry_d = d - pd.Timedelta(weeks=weeks_ago)
    px = float(md.close.loc[:entry_d, sid].dropna().iloc[-1])
    hi = px * 1.5 if reached else px
    return Position(sid, 1000, entry_d, px, px * 0.5, 1000 * px * 0.1, hi, px * 0.5, px * 0.3)


@pytest.fixture
def eng(md):
    return SignalEngine(md, Config())


def test_long_trend_both_negative_is_end(md, eng):
    d, sid = _find(md, eng, lambda c, ma, r12: c < ma and r12 < 0)
    p = _pos(md, sid, d)
    res = eng.health.check(d, p, True, None, "BULL", None)
    assert res.items["長期趨勢"][0] == END
    assert "長期趨勢" in res.exit_items()
    # 非健檢日沿用上次結果，但每週項目的「結束」降為「減弱」，不在非健檢日出場
    res2 = eng.health.check(d + pd.Timedelta(days=1), p, False, None, "BULL", None)
    assert res2.items["長期趨勢"][0] == WEAK
    assert res2.exit_items() == []


def test_long_trend_one_negative_is_weak(md, eng):
    d, sid = _find(md, eng, lambda c, ma, r12: c < ma and r12 > 0)
    res = eng.health.check(d, _pos(md, sid, d), True, None, "BULL", None)
    assert res.items["長期趨勢"][0] == WEAK
    assert res.status in (WEAK, END)
    assert "長期趨勢" in res.triggers(WEAK)


def test_long_trend_exit_can_be_disabled(md):
    cfg = Config()
    cfg.health.long_trend_exit = False
    e = SignalEngine(md, cfg)
    d, sid = _find(md, e, lambda c, ma, r12: c < ma and r12 < 0)
    res = e.health.check(d, _pos(md, sid, d), True, None, "BULL", None)
    assert res.items["長期趨勢"][0] == WEAK


def test_stagnation_and_time_stop(md, eng):
    d, sid = _find(md, eng, lambda c, ma, r12: c > ma and r12 > 0)
    assert eng.health.check(d, _pos(md, sid, d, 4, False), True, None, "BULL", None).items["停滯檢查"][0] == HEALTHY
    assert eng.health.check(d, _pos(md, sid, d, 9, False), True, None, "BULL", None).items["停滯檢查"][0] == WEAK
    assert eng.health.check(d, _pos(md, sid, d, 13, False), True, None, "BULL", None).items["停滯檢查"][0] == END
    assert eng.health.check(d, _pos(md, sid, d, 13, True), True, None, "BULL", None).items["停滯檢查"][0] == HEALTHY


def test_l1_and_safety_items(md, eng):
    d, sid = _find(md, eng, lambda c, ma, r12: c > ma and r12 > 0)
    p = _pos(md, sid, d)
    row = pd.Series({"score": 60.0, "excluded": False, "reasons": ""})
    res = eng.health.check(d, p, False, row, "WARNING", None)
    assert res.items["安全性"][0] == WEAK and res.items["大盤L1"][0] == WEAK
    row = pd.Series({"score": 40.0, "excluded": False, "reasons": ""})
    assert eng.health.check(d, p, False, row, "BULL", None).items["安全性"][0] == END


def test_weak_is_sold_first_on_l1_reduction(md):
    cfg = Config()
    e = SignalEngine(md, cfg)
    reg = e.regime
    cfg.l4.portfolio_vol_target = None
    e = SignalEngine(md, cfg)
    d = reg.index[(reg["state"] == "BULL") & (reg["target_exposure"] >= 0.99)][50]
    live = [s for s in md.stocks if not np.isnan(md.close.at[d, s])][:2]
    a, b = live
    pa, pb = _pos(md, a, d), _pos(md, b, d)
    pa.health_weekly = {"長期趨勢": [WEAK, "跌破 MA100"]}   # a 減弱，b 健康
    pb.health_weekly = {}
    # 多頭日（大盤項目皆健康），權益小於持股市值 → 曝險超過上限，觸發 L1 減碼
    equity = (pa.shares * md.close.at[d, a] + pb.shares * md.close.at[d, b]) * 0.8
    orders = e.sell_signals(d, {b: pb, a: pa}, equity, weekly=False)
    l1 = [o for o in orders if o.reason.startswith("L1")]
    assert l1 and l1[0].stock_id == a


def test_weak_seen_recorded(md, eng):
    d, sid = _find(md, eng, lambda c, ma, r12: c < ma and r12 > 0)
    p = _pos(md, sid, d)
    eng.health.check(d, p, True, None, "BULL", None)
    assert "長期趨勢" in p.weak_seen
