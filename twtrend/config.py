"""系統參數。

所有數字都是設計書中的「起始值」，實際值應由回測（walk-forward）決定。
可用 YAML 檔覆寫：`Config.load("config/default.yaml")`。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class L1Config:
    """L1 大盤多空判斷。"""

    # 判斷用指數：taiex / equal_weight / ex_tsmc（優化 #1：避免被台積電權重扭曲）
    index_source: str = "taiex"
    long_ma: int = 200                 # 主要趨勢均線
    mid_ma: int = 60                   # 中期均線（看斜率）
    slope_lookback: int = 20           # 斜率回看天數
    tsmom_lookback: int = 252          # 12 個月時間序列動能
    breadth_ma: int = 60               # 寬度：站上 60 日線比例
    breadth_threshold: float = 0.5
    # 各指標權重（高=3、中=2、低=1）
    weights: dict[str, float] = field(default_factory=lambda: {
        "above_long_ma": 3.0,
        "mid_ma_slope": 2.0,
        "tsmom_12m": 3.0,
        "breadth_above_ma": 2.0,
        "new_high_minus_low": 1.0,
        "futures_net_short_not_rising": 2.0,
        "margin_not_overheated": 2.0,
        "leading_indicator_up": 1.0,
        "m1b_above_m2": 1.0,
    })
    # 遲滯（hysteresis）：高於 enter_bull 才轉多、低於 exit_bull 才離開多頭
    enter_bull: float = 65.0
    exit_bull: float = 45.0
    bear_threshold: float = 45.0       # 空頭：跌破 200 日線且分數 < 此值
    confirm_days: int = 3              # 穿越均線需連續 N 日才生效
    # 各狀態持股上限
    cap_bull: float = 1.00
    cap_warning: float = 0.50
    cap_bear: float = 0.0              # 設計書：0–20%，由回測決定
    cap_recovery: float = 0.40         # 設計書：30–50%
    # 打底轉強：寬度急升（不對稱回補）
    recovery_breadth_from: float = 0.20
    recovery_breadth_to: float = 0.60
    recovery_window: int = 10
    # 打底轉強也需遲滯：分數 ≥ recovery_enter 才進入、< recovery_exit 才退回空頭
    recovery_enter: float = 55.0
    recovery_exit: float = 30.0
    # 分批調整：狀態變化時每次最多調整的持股比例
    max_step_per_rebalance: float = 0.34
    # 多週期訊號平均（失效情境優化）：3/6/12 個月各 1/3，持股上限分四檔
    use_multi_horizon: bool = False
    multi_horizons: tuple[int, ...] = (63, 126, 252)


@dataclass
class L2Config:
    """L2 個股趨勢評分（採「策略檢視」修正後版本）。"""

    gate_ma: int = 100                 # 簡化 gate：收盤 > MA100
    use_full_ma_stack: bool = False    # True 時改用 收盤>MA20>MA60>MA120 且 MA60 上揚
    tsmom_lookback: int = 252          # 時間序列條件：12 個月報酬 > 0
    extreme_return_pct: float = 0.03   # 排除過去 12 個月報酬最極端的前後 3%
    high_52w_min: float = 0.85         # 收盤 ÷ 52 週高 ≥ 85%
    rel_strength_lookback: int = 63    # 3 個月相對強度
    weights: dict[str, float] = field(default_factory=lambda: {
        "high_52w_ratio": 0.40,
        "revenue_yoy": 0.30,
        "rel_strength": 0.20,
        "tsmom_12m": 0.10,
    })
    buy_top_pct: float = 0.30          # 前 30% 才買
    hold_top_pct: float = 0.50         # 跌出前 50% 才換（緩衝區）


@dataclass
class L3Config:
    """L3 安全性篩選。"""

    min_turnover_60d: float = 30_000_000      # 60 日平均成交值 NT$3,000 萬
    min_market_cap: float = 5_000_000_000     # 市值 NT$50 億
    min_listed_days: int = 365
    attention_lookback_days: int = 30
    attention_max_count: int = 2               # 近 30 日注意股次數 ≥ 此值即排除
    excluded_markets: tuple[str, ...] = ("emerging", "innovation")
    full_size_min: float = 70.0
    half_size_min: float = 50.0
    # 風險評分各維度起始門檻
    fscore_low: int = 3
    zscore_low: float = 1.8
    current_ratio_low: float = 1.0
    interest_cover_low: float = 3.0
    non_operating_ratio_high: float = 0.5
    pledge_ratio_high: float = 0.30
    margin_usage_high: float = 0.40
    top_decile: float = 0.90                   # 全市場最高 10%
    exempt_industries: tuple[str, ...] = ("金融保險業", "建材營造業")


@dataclass
class L4Config:
    """L4 部位與出場（出場精簡為 4 層）。"""

    total_capital: float = 3_000_000
    risk_per_trade: float = 0.005              # 每筆風險 0.5%
    max_position_value: float = 300_000        # 單筆上限
    max_industry_weight: float = 0.30
    min_positions: int = 6
    max_positions: int = 12
    max_adv_fraction: float = 0.05             # 不超過 20 日均成交值 5%
    atr_period: int = 20
    initial_stop_atr_mult: float = 2.5         # 初始停損 2–3 × ATR(20)
    swing_low_lookback: int = 20               # 與近期波段低點比較，取較遠者
    trailing_atr_period: int = 22
    trailing_atr_mult: float = 3.0             # Chandelier exit
    catastrophic_stop: float = 0.15            # 極端保護停損
    pending_exit_days: int = 20                # 安全分 < 50 的待出場期限
    # 設計書建議出場精簡為 4 層；以下兩層預設關閉，供回測比較
    use_ma60_exit: bool = False                # 週收盤跌破 MA60 且 MA60 下彎（與 ATR 停利二擇一）
    rotate_on_rank: bool = False               # 趨勢分跌出前 50% 輪動替換
    rebalance_weekday: int = 4                 # 週五收盤後算排名
    corr_lookback: int = 60
    corr_cluster_threshold: float = 0.70
    max_cluster_risk_r: float = 1.5
    portfolio_vol_target: float | None = 0.20  # 組合年化波動上限，只用來減碼
    buy_slippage_allow: float = 0.01           # 限價 = 訊號價 × (1 + 1%)
    lot_size: int = 1                          # 以股數計，允許零股


@dataclass
class HealthConfig:
    """持股趨勢健檢（實盤手冊第 6 點）。每個「結束」條件都可關閉，方便用模擬盤 A/B 驗證。"""

    enabled: bool = True
    long_trend_exit: bool = True       # 收盤 < MA100 且 12M 報酬 < 0 → 結束
    rank_exit: bool = True             # 趨勢分跌出前 50% → 結束（輪換為新候選）
    rank_min_universe: int = 10        # 合格候選少於此數時不做排名出場（樣本太小）
    revenue_exit: bool = True          # 營收連 3 月負成長且長期趨勢減弱 → 結束
    stagnation_weeks: int = 8          # 8 週未達 +1R → 減弱
    time_stop: bool = True             # 12 週仍未達 +1R → 結束（待驗證）
    time_stop_weeks: int = 12


@dataclass
class CostConfig:
    commission: float = 0.001425
    commission_discount: float = 1.0
    tax: float = 0.003
    slippage: float = 0.001
    limit_pct_after_2015: float = 0.10
    limit_pct_before_2015: float = 0.07


@dataclass
class MonitorConfig:
    """黃燈 / 紅燈（暫定值取自設計書模擬，累積 100 筆後改用實盤分布）。"""

    rolling_window: int = 50
    mdd_yellow: float = 0.237      # 0.5% 風險下 MC MDD 95 百分位
    mdd_red: float = 0.30
    streak_yellow: int = 16
    streak_red: int = 20
    loss_ratio_yellow: float = 1.3
    loss_ratio_red: float = 1.5
    slippage_assumed: float = 0.001
    min_trades_for_own_dist: int = 100


@dataclass
class Config:
    db_path: str = "data/twtrend.sqlite"
    benchmark: str = "0050"
    index_id: str = "TAIEX"
    finmind_token: str | None = None          # 建議改用環境變數 FINMIND_TOKEN
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    l3: L3Config = field(default_factory=L3Config)
    l4: L4Config = field(default_factory=L4Config)
    health: HealthConfig = field(default_factory=HealthConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            _merge(cfg, data)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(obj: Any, data: dict[str, Any]) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"未知的設定欄位：{type(obj).__name__}.{key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
