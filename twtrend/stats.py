"""交易統計（設計書「統計檢視」與「回測時必做的統計檢驗」）。

R 倍數為核心單位：1R = 每筆預設止損金額。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 基本統計
def breakeven_winrate(payoff: float, cost_r: float = 0.0) -> float:
    """損益兩平勝率 p* = (1 + 成本) ÷ (1 + 盈虧比)。"""
    return (1 + cost_r) / (1 + payoff)


def breakeven_table(payoffs=(1.5, 2, 3, 4, 5, 6), cost_r: float = 0.1) -> pd.DataFrame:
    return pd.DataFrame({
        "平均獲利÷平均虧損": payoffs,
        "損益兩平勝率（含成本）": [breakeven_winrate(p, cost_r) for p in payoffs],
        "不含成本": [breakeven_winrate(p, 0) for p in payoffs],
    })


def round_trip_cost(commission=0.001425, discount=1.0, tax=0.003, slippage=0.001) -> float:
    return commission * discount * 2 + tax + slippage * 2


def streaks(r: np.ndarray) -> int:
    best = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


@dataclass
class TradeStats:
    n: int
    win_rate: float
    avg_win: float
    avg_loss: float
    payoff: float
    expectancy: float
    std: float
    skew: float
    top5_share: float
    top10_share: float
    exp_without_top2: float
    max_losing_streak: int
    loss_vs_1r: float          # 虧損交易平均「實際虧損 ÷ 1R」

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def trade_stats(r_multiples) -> TradeStats:
    r = np.asarray(pd.Series(r_multiples).dropna(), dtype=float)
    n = len(r)
    if n == 0:
        nan = float("nan")
        return TradeStats(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0, nan)
    wins, losses = r[r > 0], r[r <= 0]
    total = r.sum()
    srt = np.sort(r)[::-1]

    def share(p):
        k = max(1, int(math.ceil(n * p)))
        return float(srt[:k].sum() / total) if total > 0 else float("nan")

    k2 = max(1, int(math.ceil(n * 0.02)))
    s = pd.Series(r)
    return TradeStats(
        n=n,
        win_rate=float(len(wins) / n),
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        payoff=float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else float("nan"),
        expectancy=float(r.mean()),
        std=float(r.std(ddof=1)) if n > 1 else float("nan"),
        skew=float(s.skew()) if n > 2 else float("nan"),
        top5_share=share(0.05),
        top10_share=share(0.10),
        exp_without_top2=float(srt[k2:].mean()) if n > k2 else float("nan"),
        max_losing_streak=streaks(r),
        loss_vs_1r=float(-losses.mean()) if len(losses) else float("nan"),
    )


def required_trades(expectancy: float, std: float, t: float = 1.96) -> int:
    """達到 t 值所需的交易筆數：n = (t × σ / μ)²。"""
    if expectancy <= 0:
        return 10**9
    return int(math.ceil((t * std / expectancy) ** 2))


# ---------------------------------------------------------------- 檢驗
def bootstrap_expectancy_ci(r_multiples, dates=None, n_boot: int = 5000, block: int = 10,
                            alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """期望值 bootstrap 信賴區間；有日期時依出場日排序做區塊抽樣，保留同期交易的相關性。"""
    r = pd.Series(r_multiples, dtype=float)
    if dates is not None:
        r = pd.Series(r.values, index=pd.to_datetime(pd.Series(dates).values)).sort_index()
    r = r.dropna().values
    n = len(r)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    n_blocks = int(math.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(r.mean()), float(lo), float(hi)


def mc_reorder(r_multiples, risk_per_trade: float = 0.005, n_paths: int = 5000, seed: int = 0) -> pd.DataFrame:
    """交易順序重排蒙地卡羅：固定比例複利下的 MDD、最長連虧、期末倍數分布。"""
    r = np.asarray(pd.Series(r_multiples).dropna(), dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_paths):
        x = rng.permutation(r)
        eq = np.cumprod(1 + risk_per_trade * x)
        peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
        rows.append((float((eq / peak - 1).min()), streaks(x), float(eq[-1])))
    return pd.DataFrame(rows, columns=["mdd", "max_streak", "multiple"])


def simulate_paths(n_paths: int = 10_000, n_trades: int = 200, win_rate: float = 0.35,
                   gap_loss_prob: float = 0.15, gap_loss_range=(1.2, 2.0),
                   win_median: float = 1.4, win_mean: float = 2.8, win_cap: float = 25.0,
                   cost_r: float = 0.1, seed: int = 42) -> np.ndarray:
    """設計書的示意模型：低勝率、右偏厚尾獲利。回傳 (n_paths, n_trades) 的 R 值矩陣。

    獲利以對數常態近似：中位數 win_median、平均 win_mean，上限 win_cap。
    """
    rng = np.random.default_rng(seed)
    mu = math.log(win_median)
    sigma = math.sqrt(max(2 * (math.log(win_mean) - mu), 1e-6))
    is_win = rng.random((n_paths, n_trades)) < win_rate
    wins = np.minimum(rng.lognormal(mu, sigma, (n_paths, n_trades)), win_cap)
    gap = rng.random((n_paths, n_trades)) < gap_loss_prob
    losses = np.where(gap, -rng.uniform(*gap_loss_range, (n_paths, n_trades)), -1.0)
    return np.where(is_win, wins, losses) - cost_r


def risk_table(r_matrix: np.ndarray, risks=(0.005, 0.01, 0.02, 0.03)) -> pd.DataFrame:
    rows = []
    for f in risks:
        eq = np.cumprod(1 + f * r_matrix, axis=1)
        peak = np.maximum.accumulate(np.concatenate([np.ones((eq.shape[0], 1)), eq], axis=1), axis=1)[:, 1:]
        mdd = (eq / peak - 1).min(axis=1)
        rows.append({
            "每筆風險": f, "MDD中位數": float(np.median(-mdd)), "MDD95百分位": float(np.quantile(-mdd, 0.95)),
            "MDD99百分位": float(np.quantile(-mdd, 0.99)),
            "資金倍數中位數": float(np.median(eq[:, -1])), "期末虧損機率": float((eq[:, -1] < 1).mean()),
        })
    return pd.DataFrame(rows)


def streak_distribution(r_matrix: np.ndarray) -> dict:
    s = np.array([streaks(row) for row in r_matrix])
    return {"median": float(np.median(s)), "p95": float(np.quantile(s, 0.95)),
            "p99": float(np.quantile(s, 0.99)), "prob_ge10": float((s >= 10).mean()),
            "prob_ge15": float((s >= 15).mean())}


def kelly_fraction(r_multiples) -> float:
    """以 R 分布估計 Kelly 最適比例（數值搜尋 max E[log(1+f·R)]）。"""
    r = np.asarray(pd.Series(r_multiples).dropna(), dtype=float)
    if len(r) == 0 or r.mean() <= 0:
        return 0.0
    grid = np.linspace(0.001, min(0.99, 0.99 / max(1e-9, -r.min())), 400)
    g = [np.mean(np.log1p(f * r)) for f in grid]
    return float(grid[int(np.argmax(g))])


def cusum(r_multiples, target: float, k: float = 0.5, std: float | None = None) -> pd.DataFrame:
    """下行 CUSUM：偵測每筆 R 值的平均是否低於目標期望值。
    S_t = max(0, S_{t-1} + (target − r_t) − k·σ)；超過 h·σ（常用 h=4~5）即警示。"""
    r = pd.Series(r_multiples, dtype=float).dropna().reset_index(drop=True)
    sd = std if std is not None else (r.std(ddof=1) if len(r) > 1 else 1.0)
    s, out = 0.0, []
    for x in r:
        s = max(0.0, s + (target - x) - k * sd)
        out.append(s)
    return pd.DataFrame({"r": r, "cusum": out, "h4": 4 * sd, "h5": 5 * sd})


def deflated_sharpe(sharpe: float, n_obs: int, n_trials: int, skew: float = 0.0,
                    kurt: float = 3.0, sharpe_std_trials: float = 0.5) -> float:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio：
    多次參數試驗後，觀測 Sharpe 超過「純雜訊最大 Sharpe」的機率。輸入為非年化（每期）Sharpe。"""
    from statistics import NormalDist
    nd = NormalDist()
    if n_trials <= 1:
        sr0 = 0.0
    else:
        emc = 0.5772156649
        sr0 = sharpe_std_trials * ((1 - emc) * nd.inv_cdf(1 - 1 / n_trials) +
                                   emc * nd.inv_cdf(1 - 1 / (n_trials * math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2))
    z = (sharpe - sr0) * math.sqrt(max(n_obs - 1, 1)) / denom
    return nd.cdf(z)


def pbo(perf_matrix: pd.DataFrame, n_splits: int = 8) -> float:
    """回測過度擬合機率（CSCV）。perf_matrix：index=期間報酬、columns=參數組合。
    回傳樣本內最佳參數在樣本外排名落於中位數以下的比例。"""
    from itertools import combinations
    m = perf_matrix.dropna(how="all")
    T = len(m) - len(m) % n_splits
    if T < n_splits or m.shape[1] < 2:
        return float("nan")
    chunks = np.array_split(m.iloc[:T], n_splits)
    logits = []
    for ins in combinations(range(n_splits), n_splits // 2):
        is_df = pd.concat([chunks[i] for i in ins])
        oos_df = pd.concat([chunks[i] for i in range(n_splits) if i not in ins])
        is_sr = is_df.mean() / is_df.std().replace(0, np.nan)
        oos_sr = oos_df.mean() / oos_df.std().replace(0, np.nan)
        best = is_sr.idxmax()
        rank = oos_sr.rank(pct=True)[best]
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))
    return float(np.mean(np.array(logits) <= 0))
