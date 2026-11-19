import numpy as np
import pandas as pd
import pytest

from twtrend import indicators as ind
from twtrend import stats
from twtrend.backtest import Backtester, compare_benchmarks, l1_only
from twtrend.config import Config, L1Config
from twtrend.data.loader import load_benchmark
from twtrend.data.market import asof_panel
from twtrend.data.twse import parse_flags, roc_to_date
from twtrend.engine import SignalEngine
from twtrend.l1_regime import BEAR, BULL, WARNING, compute_regime, run_state_machine
from twtrend.l2_trend import evaluate_trend
from twtrend.l3_safety import SafetyEvaluator
from twtrend.monitor import GREEN, RED, evaluate_lights, overall
from twtrend.portfolio import chandelier, position_shares


# ---------------------------------------------------------------- 指標與時點
def test_asof_panel_has_no_lookahead():
    long = pd.DataFrame({"stock_id": ["A", "A"], "announce_date": ["2020-01-10", "2020-02-10"], "yoy": [0.1, 0.5]})
    dates = pd.bdate_range("2020-01-01", "2020-02-28")
    w = asof_panel(long, "yoy", dates, ["A"])
    assert np.isnan(w.loc["2020-01-09", "A"])
    assert w.loc["2020-01-10", "A"] == 0.1
    assert w.loc["2020-02-07", "A"] == 0.1
    assert w.loc["2020-02-10", "A"] == 0.5


def test_indicators_basic():
    s = pd.Series(np.arange(1, 301, dtype=float))
    assert ind.sma(s, 10).iloc[-1] == pytest.approx(295.5)
    assert ind.high_52w_ratio(s).iloc[-1] == 1.0
    eq = pd.Series([100, 120, 90, 130.0])
    assert ind.max_drawdown(eq) == pytest.approx(-0.25)


# ---------------------------------------------------------------- L1
def _ind_frame(scores, above, breadth=None):
    idx = pd.bdate_range("2020-01-01", periods=len(scores))
    df = pd.DataFrame({"above_long_ma": above, "breadth": breadth if breadth is not None else [0.5] * len(scores)},
                      index=idx)
    return df, pd.Series(scores, index=idx, dtype=float)


def test_state_machine_hysteresis():
    cfg = L1Config()
    # 70 → 多頭；55 仍在遲滯區 → 維持多頭；40 且在均線上 → 轉弱警戒；40 跌破均線 → 空頭
    df, sc = _ind_frame([70, 55, 50, 40, 40], [1, 1, 1, 1, 0])
    st = run_state_machine(df, sc, cfg)
    assert list(st) == [BULL, BULL, BULL, WARNING, BEAR]
    # 警戒中分數回到 60（未達 65）不轉多
    df, sc = _ind_frame([70, 40, 60, 66], [1, 1, 1, 1])
    assert list(run_state_machine(df, sc, cfg)) == [BULL, WARNING, WARNING, BULL]


def test_regime_outputs_and_gradual_exposure(md, cfg):
    reg = compute_regime(md, cfg.l1)
    valid = reg.dropna(subset=["state"])
    assert set(valid["state"]) <= {"BULL", "WARNING", "BEAR", "RECOVERY"}
    te = valid["target_exposure"]
    assert (te.diff().abs().dropna() <= cfg.l1.max_step_per_rebalance + 1e-9).all()
    assert te.between(0, 1).all()


def test_regime_index_sources(md, cfg):
    for src in ("taiex", "equal_weight", "ex_tsmc"):
        cfg.l1.index_source = src
        reg = compute_regime(md, cfg.l1)
        assert reg["state"].notna().any()


# ---------------------------------------------------------------- L3 / L2
def test_safety_excludes_flagged_stocks(md, cfg):
    flags = md.flags
    fd = flags[flags["flag_type"].isin(["disposal", "full_delivery"])]
    if fd.empty:
        pytest.skip("合成資料沒有處置/全額交割")
    row = fd.iloc[0]
    d = md.dates[md.dates >= pd.Timestamp(row["start_date"])][0]
    saf = SafetyEvaluator(md, cfg.l3).evaluate(d)
    assert saf.at[row["stock_id"], "excluded"]
    assert saf.at[row["stock_id"], "size_mult"] == 0
    assert saf["score"].between(0, 100).all()


def test_size_mult_rules(md, cfg):
    saf = SafetyEvaluator(md, cfg.l3).evaluate(md.dates[-1])
    ok = ~saf["excluded"]
    assert (saf.loc[ok & (saf["score"] >= 70), "size_mult"] == 1.0).all()
    assert (saf.loc[ok & saf["score"].between(50, 69.99), "size_mult"] == 0.5).all()
    assert (saf.loc[saf["score"] < 50, "size_mult"] == 0).all()


def test_trend_gate_and_buffer(md, cfg):
    d = md.dates[-1]
    tr = evaluate_trend(md, d, cfg.l2)
    scored = tr[tr["trend_score"].notna()]
    assert (scored["tsmom_12m"] > 0).all()
    assert (scored["high_52w_ratio"] >= cfg.l2.high_52w_min).all()
    assert (scored["gate_ma"]).all()
    # 買入區一定在持有區內（緩衝區）
    assert (~tr["buy_zone"] | tr["hold_zone"]).all()


# ---------------------------------------------------------------- L4
def test_position_sizing_caps():
    # 300 萬 × 0.5% = 1.5 萬風險；停損 8% → 18.75 萬
    sh = position_shares(3_000_000, 0.005, 100, 92, 300_000)
    assert sh * 100 == pytest.approx(187_500, abs=100)
    assert position_shares(3_000_000, 0.005, 100, 98, 300_000) * 100 <= 300_000
    assert position_shares(3_000_000, 0.005, 100, 92, 300_000, size_mult=0.5) * 100 == pytest.approx(93_700, abs=100)
    assert position_shares(3_000_000, 0.005, 100, 92, 300_000, adv=1_000_000) * 100 <= 50_000
    assert position_shares(3_000_000, 0.005, 100, 101, 300_000) == 0
    assert chandelier(120, 5, 3) == 105


def test_backtest_runs_and_respects_rules(md, cfg, db):
    res = Backtester(md, cfg).run()
    assert len(res.equity) == len(md.dates)
    assert res.exposure.max() <= 1.0 + 1e-6
    t = res.trades
    assert not t.empty
    assert (t["entry_date"] > t["signal_date"]).all()          # 隔日開盤才成交
    assert (pd.to_datetime(t["exit_date"]) >= pd.to_datetime(t["entry_date"])).all()
    # 空頭狀態中不新增部位
    st = res.regime["state"]
    entries_state = st.reindex(pd.to_datetime(t["signal_date"])).values
    assert not any(s in ("BEAR", "WARNING") for s in entries_state)
    cmp_ = compare_benchmarks(md, cfg, res, load_benchmark(db))
    assert {"0050 買進持有", "只用 L1 操作 0050", "完整系統"} <= set(cmp_.index)


def test_l1_only_zero_exposure_is_flat(md, cfg):
    cfg.l1.cap_bull = cfg.l1.cap_warning = cfg.l1.cap_bear = cfg.l1.cap_recovery = 0.0
    a = pd.Series(np.linspace(10, 20, len(md.dates)), index=md.dates)
    eq, w = l1_only(md, cfg, a)
    assert eq.iloc[-1] == pytest.approx(1.0)


def test_engine_sell_on_hard_flag(md, cfg):
    from twtrend.portfolio import Position
    fd = md.flags[md.flags["flag_type"] == "disposal"]
    if fd.empty:
        pytest.skip("無處置股")
    row = fd.iloc[0]
    d = md.dates[md.dates >= pd.Timestamp(row["start_date"])][0]
    sid = row["stock_id"]
    px = float(md.close.loc[:d, sid].dropna().iloc[-1])
    eng = SignalEngine(md, cfg)
    pos = {sid: Position(sid, 100, d, px, px * 0.5, 100 * px * 0.5, px, px * 0.5, px * 0.4)}
    orders = eng.sell_signals(d, pos, 3_000_000)
    assert orders and orders[0].priority == 1


# ---------------------------------------------------------------- 統計與監控
def test_breakeven_matches_design_doc():
    assert stats.breakeven_winrate(2, 0.1) == pytest.approx(0.366, abs=1e-3)
    assert stats.breakeven_winrate(3, 0) == pytest.approx(0.25)
    assert stats.round_trip_cost() == pytest.approx(0.00785)


def test_trade_stats_and_ci():
    r = [-1] * 65 + [3] * 35
    st = stats.trade_stats(r)
    assert st.win_rate == 0.35 and st.payoff == 3 and st.expectancy == pytest.approx(0.4)
    m, lo, hi = stats.bootstrap_expectancy_ci(r, n_boot=500)
    assert lo < m < hi
    assert stats.required_trades(0.2, 3.0) == 865


def test_cusum_detects_shift():
    r = np.r_[np.full(30, 0.3), np.full(30, -0.8)]
    cs = stats.cusum(r, target=0.2, std=1.0)
    assert cs["cusum"].iloc[29] == 0
    assert cs["cusum"].iloc[-1] > cs["h4"].iloc[-1]


def test_lights():
    good = pd.DataFrame({"r_multiple": [1.5, -1, -1, 2, -1] * 12, "followed_signal": 1,
                         "entry_slippage": 0.001, "exit_slippage": 0.001})
    eq = pd.Series(np.linspace(1, 1.2, 60))
    cfg = Config().monitor
    assert overall(evaluate_lights(good, eq, cfg)) == GREEN
    bad = good.assign(r_multiple=-1.8)
    assert overall(evaluate_lights(bad, eq, cfg)) == RED


def test_twse_parsing():
    assert roc_to_date("1131105") == "2024-11-05"
    assert roc_to_date("113/11/05") == "2024-11-05"
    df = parse_flags([{"Code": "2330", "DispositionPeriod": "113/11/05～113/11/18"}, {"Code": "abc"}],
                     "disposal", "twse", "2024-11-05")
    assert len(df) == 1 and df.iloc[0]["end_date"] == "2024-11-18"
