"""持股趨勢健檢（設計書「實盤操作手冊 6. 持股趨勢健檢」）。

每檔持股定期判定為三種狀態之一：
  健康：續抱
  減弱：續抱、不加碼、L1 減碼時優先賣
  結束：出場
只有「結束」才出場；「減弱」不是賣出理由。

| 檢查項目                      | 頻率      | 健康              | 減弱              | 結束（出場）                              |
|------------------------------|-----------|-------------------|-------------------|------------------------------------------|
| 追蹤停利線（3×ATR）            | 每日      | 收盤在線上          | —                 | 收盤跌破，隔日開盤賣出                       |
| 長期趨勢：收盤 vs MA100、12M 報酬 | 每週      | 兩者皆正向          | 只有一項轉負        | 兩項皆轉負                                 |
| 相對強度：3M 報酬 vs 等權重指數   | 每週      | 贏過指數            | 輸給指數           | 趨勢分跌出前 50%，輪換為新候選                |
| 52 週高點接近度                 | 每週      | ≥ 85%             | < 85%             | 不單獨觸發出場                             |
| 營收動能：月營收 YoY            | 每月      | 正成長              | 連續 2 個月轉弱     | 連續 3 個月負成長，且長期趨勢已減弱            |
| 停滯檢查：進場後是否達 +1R       | 每週      | 8 週內達到          | 8 週未達           | 12 週仍未達（時間止損，待驗證）               |
| 安全性：硬性排除、風險分          | 每日/每季 | 無旗標、分數 ≥ 70    | 分數 50–69         | 硬性排除成立，或分數 < 50 後 20 個交易日內      |
| 大盤 L1                       | 每日      | 多頭               | 轉弱警戒           | 依上限減碼，先賣「減弱」狀態的持股             |

追蹤停利、安全性、L1 三項的「結束」動作由 engine.sell_signals 原有的出場層執行；
本模組負責每項的狀態判定，以及每週項目（長期趨勢、相對強度排名、營收、時間止損）的「結束」。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HEALTHY, WEAK, END = "健康", "減弱", "結束"
_RANK = {HEALTHY: 0, WEAK: 1, END: 2}

ITEMS = ["追蹤停利", "長期趨勢", "相對強度", "52週高", "營收動能", "停滯檢查", "安全性", "大盤L1"]
WEEKLY_ITEMS = {"長期趨勢", "相對強度", "52週高", "營收動能", "停滯檢查"}


@dataclass
class HealthResult:
    stock_id: str
    date: pd.Timestamp
    items: dict[str, tuple[str, str]] = field(default_factory=dict)   # 項目 → (狀態, 說明)

    @property
    def status(self) -> str:
        if not self.items:
            return HEALTHY
        return max((s for s, _ in self.items.values()), key=lambda s: _RANK[s])

    def triggers(self, level: str | None = None) -> list[str]:
        lv = level or self.status
        return [k for k, (s, _) in self.items.items() if s == lv and lv != HEALTHY]

    def exit_items(self) -> list[str]:
        """由本模組負責出場的「結束」項目（每日類由原出場層處理）。"""
        return [k for k, (s, _) in self.items.items() if s == END and k in WEEKLY_ITEMS]

    def summary(self) -> str:
        t = self.triggers()
        return self.status + (f"（{'、'.join(t)}）" if t else "")

    def to_row(self) -> dict:
        row = {"date": self.date, "stock_id": self.stock_id, "status": self.status,
               "triggers": "、".join(self.triggers()),
               "weak_items": "、".join(self.triggers(WEAK)), "end_items": "、".join(self.triggers(END))}
        for k in ITEMS:
            s, note = self.items.get(k, ("", ""))
            row[k] = s
            row[k + "_說明"] = note
        return row


class HealthChecker:
    """依附於 SignalEngine，共用其預先計算的面板資料。"""

    def __init__(self, engine):
        from . import indicators as ind
        from .l1_regime import regime_index
        self.eng = engine
        md = engine.md
        self.cfg = engine.cfg
        self.ma100 = ind.sma(md.close, 100)
        self.ret12 = ind.total_return(md.close, 252)
        self.ret63 = ind.total_return(md.close, 63)
        self.hi52 = ind.high_52w_ratio(md.close, 252)
        ew = regime_index(md, "equal_weight")
        self.ew_ret63 = ind.total_return(ew, 63)
        self._rev = self._revenue_history(md.revenue)

    @staticmethod
    def _revenue_history(rev: pd.DataFrame) -> dict[str, pd.Series]:
        if rev is None or rev.empty or "yoy" not in rev:
            return {}
        r = rev.dropna(subset=["yoy", "announce_date"]).copy()
        r["announce_date"] = pd.to_datetime(r["announce_date"])
        r = r.sort_values("announce_date")
        return {sid: g.set_index("announce_date")["yoy"] for sid, g in r.groupby("stock_id")}

    def _yoy_last(self, sid: str, d: pd.Timestamp, n: int) -> list[float]:
        s = self._rev.get(sid)
        if s is None:
            return []
        return list(s.loc[:d].tail(n).values)

    def _val(self, panel, d, sid) -> float:
        try:
            v = panel.at[d, sid] if isinstance(panel, pd.DataFrame) else panel.at[d]
        except KeyError:
            return np.nan
        return float(v) if pd.notna(v) else np.nan

    # ------------------------------------------------------------ 主函式
    def check(self, d, p, weekly: bool, safety_row: pd.Series | None, regime_state: str | None,
              trend: pd.DataFrame | None = None) -> HealthResult:
        d = pd.Timestamp(d)
        sid = p.stock_id
        hc = self.cfg.health
        res = HealthResult(sid, d)
        close = self._val(self.eng.md.close, d, sid)

        # 1. 追蹤停利線（每日）
        if not np.isnan(close):
            res.items["追蹤停利"] = (END, f"收盤 {close:.2f} < {p.stop:.2f}") if close < p.stop \
                else (HEALTHY, f"距停損 {(close - p.stop) / close:.1%}")

        # 7. 安全性（每日旗標 / 每週風險分）
        if safety_row is not None:
            sc = float(safety_row["score"])
            if bool(safety_row["excluded"]) or sc < self.cfg.l3.half_size_min:
                why = safety_row["reasons"] or f"分數 {sc:.0f} < {self.cfg.l3.half_size_min:.0f}"
                res.items["安全性"] = (END, why)
            elif sc < self.cfg.l3.full_size_min:
                res.items["安全性"] = (WEAK, f"分數 {sc:.0f}")
            else:
                res.items["安全性"] = (HEALTHY, f"分數 {sc:.0f}")

        # 8. 大盤 L1（每日）：空頭的「結束」動作是 L1 減碼，由出場層依上限處理
        if regime_state:
            res.items["大盤L1"] = {"BULL": (HEALTHY, "多頭"), "WARNING": (WEAK, "轉弱警戒"),
                                 "RECOVERY": (WEAK, "打底轉強"), "BEAR": (WEAK, "空頭：依上限減碼")}.get(
                regime_state, (HEALTHY, regime_state))

        if not weekly:
            # 非健檢日沿用上次每週項目結果（存在持股上，各帳戶獨立）
            for k, (s, note) in (p.health_weekly or {}).items():
                res.items[k] = (WEAK if s == END else s, note)   # 每週項目的「結束」只在健檢日執行
            return res

        # 2. 長期趨勢（每週）
        ma = self._val(self.ma100, d, sid)
        r12 = self._val(self.ret12, d, sid)
        neg = [x for x, bad in (("跌破 MA100", not np.isnan(ma) and close < ma),
                                ("12 個月報酬為負", not np.isnan(r12) and r12 < 0)) if bad]
        long_state = END if len(neg) == 2 else WEAK if len(neg) == 1 else HEALTHY
        if neg:
            long_note = "、".join(neg)
        elif np.isnan(ma) or np.isnan(r12):
            long_note = "資料不足"
        else:
            long_note = f"MA100 之上、12M {r12:+.0%}"
        if long_state == END and not hc.long_trend_exit:
            long_state = WEAK
        res.items["長期趨勢"] = (long_state, long_note)

        # 3. 相對強度（每週）：輸給等權重指數 = 減弱；趨勢分跌出前 50% = 結束
        rs = self._val(self.ret63, d, sid) - self._val(self.ew_ret63, d, None) \
            if not np.isnan(self._val(self.ew_ret63, d, None)) else np.nan
        state, note = (HEALTHY, f"贏指數 {rs:+.1%}") if (not np.isnan(rs) and rs >= 0) else \
            (WEAK, f"輸指數 {rs:+.1%}" if not np.isnan(rs) else "資料不足")
        if hc.rank_exit and trend is not None and sid in trend.index:
            elig = trend["trend_score"].dropna()
            raw = trend.at[sid, "raw_score"]
            if len(elig) >= hc.rank_min_universe and pd.notna(raw):
                cutoff = float(elig.quantile(1 - self.cfg.l2.hold_top_pct))
                if raw < cutoff:
                    state, note = END, f"趨勢分 {raw:.0f} 跌出前 {self.cfg.l2.hold_top_pct:.0%}（門檻 {cutoff:.0f}）"
        res.items["相對強度"] = (state, note)

        # 4. 52 週高點接近度（每週，不單獨觸發出場）
        h = self._val(self.hi52, d, sid)
        if not np.isnan(h):
            res.items["52週高"] = (HEALTHY, f"{h:.0%}") if h >= self.cfg.l2.high_52w_min else (WEAK, f"{h:.0%}")

        # 5. 營收動能（每月公告，於每週健檢時以最新已公告資料判定）
        yoy = self._yoy_last(sid, d, 3)
        if yoy:
            if len(yoy) >= 3 and all(y < 0 for y in yoy) and long_state in (WEAK, END) and hc.revenue_exit:
                res.items["營收動能"] = (END, f"連 3 月 YoY 為負（{yoy[-1]:+.0%}）且長期趨勢減弱")
            elif len(yoy) >= 3 and yoy[-1] < yoy[-2] < yoy[-3]:
                res.items["營收動能"] = (WEAK, f"YoY 連 2 月轉弱（{yoy[-3]:+.0%}→{yoy[-2]:+.0%}→{yoy[-1]:+.0%}）")
            elif yoy[-1] < 0:
                res.items["營收動能"] = (WEAK, f"YoY {yoy[-1]:+.0%}")
            else:
                res.items["營收動能"] = (HEALTHY, f"YoY {yoy[-1]:+.0%}")

        # 6. 停滯檢查：進場後是否曾達 +1R
        weeks = (d - pd.Timestamp(p.entry_date)).days / 7
        reached = p.shares > 0 and (p.highest_close - p.entry_price) * p.shares >= p.one_r > 0
        if reached:
            res.items["停滯檢查"] = (HEALTHY, "已達 +1R")
        elif weeks >= hc.time_stop_weeks and hc.time_stop:
            res.items["停滯檢查"] = (END, f"{weeks:.0f} 週仍未達 +1R（時間止損）")
        elif weeks >= hc.stagnation_weeks:
            res.items["停滯檢查"] = (WEAK, f"{weeks:.0f} 週未達 +1R")
        else:
            res.items["停滯檢查"] = (HEALTHY, f"進場 {weeks:.0f} 週")

        p.health_weekly = {k: list(v) for k, v in res.items.items() if k in WEEKLY_ITEMS}
        p.weak_seen = sorted(set(p.weak_seen or []) | set(res.triggers(WEAK)))
        return res
