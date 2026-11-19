"""實盤監控：黃燈 / 紅燈判定（事前寫死，不在虧損時憑感覺決定）。

黃燈 → 檢討、不改規則；紅燈 → 暫停新部位、重新驗證（並依操作手冊降級）。
初期沒有自己的分布時，用設計書「統計檢視」的模擬值；累積 100 筆以上改用實盤 R 分布。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import MonitorConfig
from .stats import cusum, mc_reorder, simulate_paths, streaks, trade_stats

GREEN, YELLOW, RED = "綠燈", "黃燈", "紅燈"


@dataclass
class Light:
    metric: str
    value: float
    yellow: float
    red: float
    level: str
    note: str = ""


def reference_distribution(r_ref: np.ndarray | None, window: int, n: int = 5000, seed: int = 0) -> dict:
    """參考分布：滾動 window 筆期望值的 5/10 百分位、MDD 與連虧的 95/99 百分位。"""
    rng = np.random.default_rng(seed)
    if r_ref is None or len(r_ref) < window:
        mat = simulate_paths(n_paths=2000, n_trades=200, seed=seed)
        sample_means = mat[:, :window].mean(axis=1)
        mdd = None
        st = np.array([streaks(row) for row in mat])
    else:
        r_ref = np.asarray(r_ref, dtype=float)
        sample_means = np.array([rng.choice(r_ref, window).mean() for _ in range(n)])
        mc = mc_reorder(r_ref, n_paths=2000, seed=seed)
        mdd = (-mc["mdd"]).quantile([0.95, 0.99]).values
        st = mc["max_streak"].values
    return {
        "exp_p10": float(np.quantile(sample_means, 0.10)),
        "exp_p05": float(np.quantile(sample_means, 0.05)),
        "streak_p95": float(np.quantile(st, 0.95)),
        "streak_p99": float(np.quantile(st, 0.99)),
        "mdd_p95": float(mdd[0]) if mdd is not None else None,
        "mdd_p99": float(mdd[1]) if mdd is not None else None,
    }


def evaluate_lights(trades: pd.DataFrame, equity: pd.Series | None, cfg: MonitorConfig,
                    r_reference: np.ndarray | None = None) -> list[Light]:
    closed = trades.dropna(subset=["r_multiple"]) if not trades.empty else trades
    r = closed["r_multiple"].astype(float).values if not closed.empty else np.array([])
    use_own = len(r) >= cfg.min_trades_for_own_dist
    ref = reference_distribution(r if use_own else r_reference, cfg.rolling_window)
    lights: list[Light] = []

    def light(metric, value, y, rd, higher_is_bad=True, note=""):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            lvl = GREEN
        elif higher_is_bad:
            lvl = RED if value > rd else YELLOW if value > y else GREEN
        else:
            lvl = RED if value < rd else YELLOW if value < y else GREEN
        lights.append(Light(metric, value, y, rd, lvl, note))

    # 1. 實盤最大回撤
    mdd = -ind.max_drawdown(equity) if equity is not None and len(equity) > 1 else float("nan")
    y = ref["mdd_p95"] or cfg.mdd_yellow
    rd = ref["mdd_p99"] or cfg.mdd_red
    light("實盤最大回撤", mdd, y, rd)

    # 2. 滾動 50 筆期望值（紅燈需連續 2 次檢查都低於 5 百分位）
    w = cfg.rolling_window
    if len(r) >= w:
        roll = pd.Series(r).rolling(w).mean().dropna()
        cur = float(roll.iloc[-1])
        prev = float(roll.iloc[-2]) if len(roll) > 1 else cur
        lvl_val = cur if prev < ref["exp_p05"] else max(cur, ref["exp_p05"])
        light(f"滾動 {w} 筆期望值 (R)", lvl_val, ref["exp_p10"], ref["exp_p05"], higher_is_bad=False,
              note=f"目前 {cur:+.2f}R")
    else:
        light(f"滾動 {w} 筆期望值 (R)", float("nan"), ref["exp_p10"], ref["exp_p05"], False,
              note=f"樣本不足（{len(r)}/{w} 筆）")

    # 3. 最長連虧
    light("最長連虧筆數", float(streaks(r)) if len(r) else float("nan"),
          ref["streak_p95"] if use_own else cfg.streak_yellow,
          ref["streak_p99"] if use_own else cfg.streak_red)

    # 4. 實際虧損 ÷ 1R
    st = trade_stats(r)
    light("實際虧損 ÷ 預設 1R", st.loss_vs_1r, cfg.loss_ratio_yellow, cfg.loss_ratio_red)

    # 5. 實際滑價（相對假設）
    if not closed.empty and "entry_slippage" in closed:
        slip = pd.concat([closed["entry_slippage"], closed.get("exit_slippage", pd.Series(dtype=float))]).dropna()
        ratio = float(slip.abs().mean() / cfg.slippage_assumed) if len(slip) else float("nan")
        light("實際滑價 ÷ 假設", ratio, 1.5, 2.0)

    # 6. 遵守訊號率（操作手冊升級條件 ≥ 95%）
    if not closed.empty and "followed_signal" in closed:
        rate = float(closed["followed_signal"].astype(float).mean())
        light("遵守訊號率", rate, 0.95, 0.90, higher_is_bad=False)

    # 7. CUSUM（輔助）
    if len(r) >= 10:
        target = st.expectancy if use_own else 0.20
        cs = cusum(r, target)
        val = float(cs["cusum"].iloc[-1])
        light("R 值 CUSUM", val, float(cs["h4"].iloc[-1]), float(cs["h5"].iloc[-1]),
              note=f"目標期望值 {target:+.2f}R")
    return lights


def overall(lights: list[Light]) -> str:
    levels = [l.level for l in lights]
    return RED if RED in levels else YELLOW if YELLOW in levels else GREEN


def lights_frame(lights: list[Light]) -> pd.DataFrame:
    return pd.DataFrame([l.__dict__ for l in lights])


STAGES = [
    {"stage": 0, "name": "流程試跑", "capital": 0, "risk": 0.0, "note": "紙上交易 2–4 週"},
    {"stage": 1, "name": "小額實盤", "capital": 1_000_000, "risk": 0.0025, "note": "累積前 30 筆，量測滑價與實際虧損÷1R"},
    {"stage": 2, "name": "全額實盤", "capital": 3_000_000, "risk": 0.005, "note": "不再加大每筆風險"},
]


def next_stage(current: int, lights: list[Light], n_trades: int, process_errors: int = 0) -> tuple[int, str]:
    """分階段投入資金的升降級建議。"""
    status = overall(lights)
    if status == RED:
        return max(0, current - 1), "觸發紅燈：退回上一階段、每筆風險減半，處理完原因再升回"
    by = {l.metric: l for l in lights}
    if current == 0:
        return (1, "流程無錯誤，可進入小額實盤") if process_errors == 0 else (0, "流程仍有錯誤")
    if current == 1:
        slip_ok = by.get("實際滑價 ÷ 假設") is None or by["實際滑價 ÷ 假設"].level == GREEN
        r_ok = by["實際虧損 ÷ 預設 1R"].level == GREEN
        follow = by.get("遵守訊號率")
        follow_ok = follow is not None and follow.value >= 0.95
        if n_trades >= 30 and slip_ok and r_ok and follow_ok:
            return 2, "滑價與 1R 執行在黃燈以內且遵守率 ≥ 95%：可升為全額實盤"
        return 1, f"維持小額實盤（{n_trades}/30 筆）"
    return current, "維持"
