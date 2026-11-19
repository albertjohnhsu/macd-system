import numpy as np
import pandas as pd
import pytest

from twtrend import paper, review
from twtrend.backtest import Backtester
from twtrend.config import Config
from twtrend.data import synthetic
from twtrend.data.loader import load_benchmark, load_market_data
from twtrend.db import Database


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    db = Database(tmp_path_factory.mktemp("p") / "p.sqlite")
    synthetic.write_to_db(db, synthetic.generate(n_stocks=25, start="2016-01-01", end="2019-06-30", seed=5))
    return db, load_market_data(db)


def test_segmented_run_equals_single_run(env):
    db, md = env
    paper.create_account(db, "one", Config(), start_date="2017-06-01")
    paper.create_account(db, "two", Config(), start_date="2017-06-01")
    paper.run_account(db, md, "one", "2019-06-30")
    for until in ("2017-12-29", "2018-07-13", "2019-06-30"):     # 分段推進（模擬每日 / 每週執行）
        paper.run_account(db, md, "two", until)
    e1, e2 = paper.equity(db, "one")["equity"], paper.equity(db, "two")["equity"]
    assert len(e1) == len(e2)
    np.testing.assert_allclose(e1.values, e2.values, rtol=1e-9)
    assert len(paper.trades(db, "one")) == len(paper.trades(db, "two"))


def test_paper_matches_backtest(env):
    db, md = env
    bt = Backtester(md, Config()).run("2017-06-01", "2019-06-30")
    e = paper.equity(db, "one")["equity"]
    np.testing.assert_allclose(e.values, bt.equity.values, rtol=1e-9)


def test_overrides_and_rule_version(env):
    db, md = env
    info = paper.create_account(db, "ab", Config(), start_date="2017-06-01",
                                overrides=["l4.trailing_atr_mult=3.5", "l1.index_source=equal_weight"])
    assert info["rule_version"] != paper.rule_version(Config())
    cfg = paper.account_config(db, "ab")
    assert cfg.l4.trailing_atr_mult == 3.5 and cfg.l1.index_source == "equal_weight"
    diff = paper.diff_rules(Config(), cfg)
    assert "l4.trailing_atr_mult" in diff
    with pytest.raises(KeyError):
        paper.apply_overrides(Config(), ["l4.no_such_param=1"])
    with pytest.raises(ValueError):
        paper.create_account(db, "ab", Config())


def test_manual_order_is_flagged(env):
    db, md = env
    paper.create_account(db, "manual", Config(), start_date="2018-01-02", manual_only=True)
    paper.run_account(db, md, "manual", "2018-03-01")
    sid = md.close.loc["2018-03-01"].dropna().index[0]
    paper.place_order(db, md, "manual", "buy", sid, 1000)
    paper.run_account(db, md, "manual", "2018-03-09")
    st = paper.status(db, md, "manual")
    assert sid in set(st["holdings"]["stock_id"])
    assert bool(st["holdings"].iloc[0]["manual_entry"])
    paper.place_order(db, md, "manual", "sell", sid)
    paper.run_account(db, md, "manual", "2018-03-16")
    t = paper.trades(db, "manual")
    assert len(t) == 1 and t.iloc[0]["manual_entry"] == 1 and t.iloc[0]["manual_exit"] == 1
    # 純手動帳戶不會產生系統訊號
    assert paper.signals(db, "manual").empty


def test_phase_labels(env):
    db, md = env
    eq = paper.equity(db, "one")
    assert set(eq["phase"]) == {"replay"}          # 起始日早於今天 → 全是歷史重播


def test_review_outputs(env):
    db, md = env
    t = paper.trades(db, "one")
    sig = paper.signals(db, "one")
    assert not sig.empty and set(sig["decision"]) <= {"buy", "skip"}
    fs = review.findings_frame(review.findings(t, md, sig))
    assert {"判定", "項目", "證據", "建議"} <= set(fs.columns)
    att = review.attribution(t, "出場原因")
    assert att["筆數"].sum() == t["r_multiple"].notna().sum()
    cf = review.candidate_followup(sig, md)
    assert "已買" in set(cf["類別"])
    board = paper.leaderboard(db, load_benchmark(db))
    assert {"one", "two"} <= set(board["帳戶"])
    cmp_ = review.compare_accounts(db, "one", ["two"])
    assert cmp_.iloc[0]["差"] == pytest.approx(0.0)


def test_compare_expectancy_detects_difference():
    a = pd.Series(np.r_[np.full(60, -1.0), np.full(40, 1.0)])
    b = pd.Series(np.r_[np.full(30, -1.0), np.full(70, 2.0)])
    assert review.compare_expectancy(a, b)["結論"] == "B 顯著較好"
    assert review.compare_expectancy(a[:10], b[:10])["結論"] == "樣本不足"
