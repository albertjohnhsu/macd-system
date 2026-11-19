"""訊號引擎：整合 L1/L2/L3，產生每日賣出清單與每週買入清單（L4 規則）。

同一套程式同時給實盤（live.py）與回測（backtest.py）使用，確保兩者規則一致。
進場：L1 允許 + L3 通過 + L2 趨勢分前段，三層同時放行。
出場（優先順序）：
  1 L3 硬性排除          → 全數賣出
  2 極端保護停損 15%      → 全數賣出（實盤以條件單盤中執行）
  3 追蹤停利 / 初始停損    → 收盤跌破 max(初始停損, 最高收盤 − 3×ATR22)，隔日開盤賣出
  4 L1 減碼              → 曝險超過分批目標時，先賣安全分最低、再賣趨勢分最低者
  另：安全分 < 50 進入待出場（最長 20 個交易日）；MA60 破線、排名輪動為可選層。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import Config
from .data.market import MarketData
from .l1_regime import STATE_ZH, compute_regime, regime_index
from .l2_trend import compute_features, evaluate_trend
from .l3_safety import SafetyEvaluator
from .portfolio import Order, Position, chandelier, initial_stop_price, position_shares


class SignalEngine:
    def __init__(self, md: MarketData, cfg: Config, regime: pd.DataFrame | None = None):
        self.md = md
        self.cfg = cfg
        self.regime = regime if regime is not None else compute_regime(md, cfg.l1, regime_index(md, cfg.l1.index_source))
        self.l2_features = compute_features(md, cfg.l2)
        self.l3 = SafetyEvaluator(md, cfg.l3)
        c4 = cfg.l4
        self.atr_entry = ind.atr(md.high, md.low, md.close, c4.atr_period)
        self.atr_trail = ind.atr(md.high, md.low, md.close, c4.trailing_atr_period)
        self.ma60 = ind.sma(md.close, 60)
        self.adv20 = md.turnover.rolling(20, min_periods=10).mean()
        self.ret = md.returns()
        self._safety_cache: dict[pd.Timestamp, pd.DataFrame] = {}
        self._safety_base: tuple | None = None
        self._trend_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    # ------------------------------------------------------------ 各層查詢
    def regime_at(self, date) -> pd.Series:
        r = self.regime.loc[:pd.Timestamp(date)]
        return r.iloc[-1] if len(r) else pd.Series(dtype=object)

    def safety(self, date, full: bool = False) -> pd.DataFrame:
        """L3 評估。完整風險評分只在 full=True（每週排名日）重算；規則性旗標每日更新。"""
        d = pd.Timestamp(date)
        base_date = self._safety_base[2] if self._safety_base is not None else None
        if d in self._safety_cache and (not full or base_date == d):
            return self._safety_cache[d]
        if full or self._safety_base is None:
            res = self.l3.evaluate(d)
            self._safety_base = (None, res, d)
        else:
            res = self.l3.refresh_flags(self._safety_base[1], d)
        if len(self._safety_cache) > 8:
            self._safety_cache.clear()
        self._safety_cache[d] = res
        self._trend_cache.pop(d, None)
        return res

    def trend(self, date) -> pd.DataFrame:
        d = pd.Timestamp(date)
        if d not in self._trend_cache:
            if len(self._trend_cache) > 8:
                self._trend_cache.clear()
            saf = self.safety(d)
            universe = list(saf.index[(~saf["excluded"]) & (saf["size_mult"] > 0)])
            self._trend_cache[d] = evaluate_trend(self.md, d, self.cfg.l2, universe, self.l2_features)
        return self._trend_cache[d]

    def _px(self, date, sid) -> float:
        return float(self.md.close.at[pd.Timestamp(date), sid])

    # ------------------------------------------------------------ 停損維護
    def update_stops(self, date, positions: dict[str, Position]) -> None:
        """每日收盤後：更新持有期間最高收盤與追蹤停利（只上移）。"""
        d = pd.Timestamp(date)
        for p in positions.values():
            c = self.md.close.at[d, p.stock_id] if p.stock_id in self.md.close else np.nan
            if np.isnan(c):
                continue
            p.highest_close = max(p.highest_close, float(c))
            a = self.atr_trail.at[d, p.stock_id]
            if not np.isnan(a):
                p.trailing_stop = max(p.trailing_stop, chandelier(p.highest_close, float(a), self.cfg.l4.trailing_atr_mult))

    # ------------------------------------------------------------ 賣出
    def sell_signals(self, date, positions: dict[str, Position], equity: float,
                     weekly: bool = False) -> list[Order]:
        d = pd.Timestamp(date)
        c4 = self.cfg.l4
        saf = self.safety(d, full=weekly)
        orders: dict[str, Order] = {}

        def sell(p: Position, reason: str, prio: int, **meta):
            if p.stock_id not in orders or orders[p.stock_id].priority > prio:
                orders[p.stock_id] = Order(p.stock_id, "sell", p.shares, d, self._px(d, p.stock_id),
                                           None, reason, prio, meta)

        for sid, p in positions.items():
            close = self.md.close.at[d, sid]
            if np.isnan(close):
                continue
            row = saf.loc[sid] if sid in saf.index else None
            if row is not None and row["excluded"] and _is_hard(row["reasons"]):
                sell(p, f"L3 硬性排除：{row['reasons']}", 1)
                continue
            if close <= p.catastrophic_stop:
                sell(p, f"極端保護停損（-{c4.catastrophic_stop:.0%}）", 2)
                continue
            if close < p.stop:
                kind = "追蹤停利" if p.trailing_stop > p.initial_stop else "初始停損"
                sell(p, f"{kind}：收盤 {close:.2f} < {p.stop:.2f}", 4)
                continue
            if row is not None and (row["score"] < self.cfg.l3.half_size_min or row["excluded"]):
                if p.pending_exit_since is None:
                    p.pending_exit_since = d
                held = self.md.close.loc[p.pending_exit_since:d].shape[0] - 1
                if held >= c4.pending_exit_days:
                    sell(p, f"安全分降級待出場滿 {c4.pending_exit_days} 日", 6)
                    continue
            elif row is not None:
                p.pending_exit_since = None
            if weekly and c4.use_ma60_exit:
                m = self.ma60.loc[:d, sid]
                if len(m) > 5 and close < m.iloc[-1] and m.iloc[-1] < m.iloc[-6]:
                    sell(p, "週收盤跌破 MA60 且 MA60 下彎", 5)
                    continue
            if weekly and c4.rotate_on_rank:
                tr = self.trend(d)
                if sid not in tr.index or not bool(tr.at[sid, "hold_zone"]):
                    sell(p, "趨勢分跌出前 50%，輪動替換", 7)

        # L1 減碼（含組合波動上限）
        remaining = {s: p for s, p in positions.items() if s not in orders}
        reg = self.regime_at(d)
        target = float(reg.get("target_exposure", np.nan)) if len(reg) else np.nan
        if not np.isnan(target):
            target = min(target, self.vol_scaled_cap(d, positions, equity))
            value = {s: p.shares * self._px(d, s) for s, p in remaining.items()}
            exposure = sum(value.values())
            limit = target * equity
            if exposure > limit * 1.02 + 1:
                tr = self.trend(d)
                order = sorted(remaining, key=lambda s: (
                    saf.at[s, "score"] if s in saf.index else 0,
                    tr.at[s, "trend_score"] if s in tr.index and not np.isnan(tr.at[s, "trend_score"]) else -1))
                state = STATE_ZH.get(reg.get("state"), reg.get("state"))
                for s in order:
                    if exposure <= limit * 1.02 + 1:
                        break
                    sell(remaining[s], f"L1 減碼（{state}，目標曝險 {target:.0%}）", 3)
                    exposure -= value[s]
        return sorted(orders.values(), key=lambda o: o.priority)

    def vol_scaled_cap(self, date, positions: dict[str, Position], equity: float) -> float:
        """組合層級波動上限：20 日年化波動超過目標時，等比例降低曝險上限（只減不加）。"""
        tgt = self.cfg.l4.portfolio_vol_target
        if not tgt or not positions or equity <= 0:
            return 1.0
        d = pd.Timestamp(date)
        w = pd.Series({s: p.shares * self._px(d, s) / equity for s, p in positions.items()})
        r = self.ret.loc[:d, list(w.index)].tail(20).fillna(0)
        if len(r) < 10:
            return 1.0
        port_vol = float((r @ w).std() * np.sqrt(252))
        exposure = float(w.sum())
        if port_vol <= tgt or port_vol == 0:
            return 1.0
        return exposure * tgt / port_vol

    # ------------------------------------------------------------ 買入
    def buy_signals(self, date, positions: dict[str, Position], equity: float, cash: float,
                    pending_sells: set[str] | None = None, log: list | None = None) -> list[Order]:
        """產生買入清單。傳入 log（list）時，會記錄每個候選「買 / 不買及原因」，供模擬盤做規則檢討。"""
        d = pd.Timestamp(date)
        c4 = self.cfg.l4
        pending_sells = pending_sells or set()
        reg = self.regime_at(d)
        if not len(reg) or pd.isna(reg.get("target_exposure")):
            return []
        held = {s: p for s, p in positions.items() if s not in pending_sells}
        target = min(float(reg["target_exposure"]), self.vol_scaled_cap(d, held, equity))
        exposure = sum(p.shares * self._px(d, s) for s, p in held.items())
        available = min(equity * target - exposure, cash)
        gate_reason = None
        if reg["state"] in ("BEAR", "WARNING"):          # 空頭 / 轉弱警戒：停止新增部位
            gate_reason = f"L1 {STATE_ZH.get(reg['state'])}：停止新增"
        elif available <= 0:
            gate_reason = "已達 L1 曝險上限或現金不足"
        elif len(held) >= c4.max_positions:
            gate_reason = "持股檔數已滿"
        if gate_reason and log is None:
            return []

        saf = self.safety(d, full=True)
        tr = self.trend(d)
        cands = tr[tr["buy_zone"] & tr["trend_score"].notna()]
        cands = cands[~cands.index.isin(list(positions))]

        def record(sid, row, decision, reason, shares=0):
            if log is not None:
                log.append({
                    "date": d, "stock_id": sid, "rank": int(row["rank"]), "trend_score": float(row["trend_score"]),
                    "safety_score": float(saf.at[sid, "score"]) if sid in saf.index else np.nan,
                    "size_mult": float(saf.at[sid, "size_mult"]) if sid in saf.index else 0.0,
                    "state": reg["state"], "price": self._px(d, sid), "decision": decision,
                    "reason": reason, "shares": shares,
                })

        if gate_reason:
            for sid, row in cands.iterrows():
                record(sid, row, "skip", gate_reason)
            return []

        info = self.md.info
        ind_value: dict[str, float] = {}
        for s, p in held.items():
            k = info.at[s, "industry"] if s in info.index else ""
            ind_value[k] = ind_value.get(k, 0.0) + p.shares * self._px(d, s)
        clusters_r = self._cluster_risk(d, held)

        orders: list[Order] = []
        n_pos = len(held)
        for sid, row in cands.iterrows():
            if n_pos >= c4.max_positions or available <= 0:
                record(sid, row, "skip", "持股檔數已滿" if n_pos >= c4.max_positions else "可用資金用完")
                continue
            mult = float(saf.at[sid, "size_mult"]) if sid in saf.index else 0.0
            if mult <= 0:
                record(sid, row, "skip", "安全分不足")
                continue
            hist_c = self.md.close.loc[:d, sid].dropna()
            hist_l = self.md.low.loc[:d, sid].dropna()
            a = self.atr_entry.at[d, sid]
            if len(hist_c) < c4.swing_low_lookback or np.isnan(a):
                record(sid, row, "skip", "歷史資料不足")
                continue
            px = float(hist_c.iloc[-1])
            stop = initial_stop_price(hist_c, hist_l, float(a), c4.initial_stop_atr_mult, c4.swing_low_lookback)
            industry = info.at[sid, "industry"] if sid in info.index else ""
            ind_room = equity * c4.max_industry_weight - ind_value.get(industry, 0.0)
            shares = position_shares(equity, c4.risk_per_trade, px * (1 + c4.buy_slippage_allow), stop,
                                     c4.max_position_value, mult, self.adv20.at[d, sid], c4.max_adv_fraction,
                                     min(available, ind_room), c4.lot_size)
            if shares <= 0:
                record(sid, row, "skip", "產業上限" if ind_room <= 0 else "部位計算為 0")
                continue
            # 相關分群上限：與既有持股 60 日相關 > 0.7 視為同群，每群風險合計 ≤ 1.5R
            cluster = self._cluster_of(d, sid, held)
            if cluster is not None and clusters_r.get(cluster, 0.0) + mult > c4.max_cluster_risk_r + 1e-9:
                record(sid, row, "skip", "相關分群上限")
                continue
            value = shares * px
            orders.append(Order(sid, "buy", shares, d, px, round(px * (1 + c4.buy_slippage_allow), 2),
                                f"L2 趨勢分第 {int(row['rank'])} 名", 8, {
                                    "stop": stop, "trend_score": float(row["trend_score"]),
                                    "safety_score": float(saf.at[sid, "score"]), "size_mult": mult,
                                    "state": reg["state"], "industry": industry,
                                }))
            record(sid, row, "buy", orders[-1].reason, shares)
            available -= value
            ind_value[industry] = ind_value.get(industry, 0.0) + value
            key = cluster if cluster is not None else sid
            clusters_r[key] = clusters_r.get(key, 0.0) + mult
            held = {**held, sid: Position(sid, shares, d, px, stop, 0, px, stop, 0, industry, mult)}
            n_pos += 1
        return orders

    def manual_stop(self, date, sid: str) -> float:
        """手動下單時套用同一套初始停損公式。"""
        d = pd.Timestamp(date)
        c4 = self.cfg.l4
        hist_c = self.md.close.loc[:d, sid].dropna()
        hist_l = self.md.low.loc[:d, sid].dropna()
        a = self.atr_entry.at[d, sid]
        if len(hist_c) < c4.swing_low_lookback or np.isnan(a):
            return float(hist_c.iloc[-1]) * 0.92
        return initial_stop_price(hist_c, hist_l, float(a), c4.initial_stop_atr_mult, c4.swing_low_lookback)

    def _corr(self, d: pd.Timestamp, cols: list[str]) -> pd.DataFrame:
        r = self.ret.loc[:d, cols].tail(self.cfg.l4.corr_lookback)
        return r.corr(min_periods=20)

    def _cluster_of(self, d: pd.Timestamp, sid: str, held: dict[str, Position]) -> str | None:
        if not held:
            return None
        cols = list(held) + [sid]
        corr = self._corr(d, cols)[sid].drop(sid)
        hit = corr[corr > self.cfg.l4.corr_cluster_threshold]
        if hit.empty:
            return None
        return self._cluster_map(d, held).get(hit.idxmax())

    def _cluster_map(self, d: pd.Timestamp, held: dict[str, Position]) -> dict[str, str]:
        """把持股依相關係數 > 門檻做連通分群，群名取群內第一檔。"""
        names = list(held)
        if not names:
            return {}
        corr = self._corr(d, names)
        parent = {n: n for n in names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(names):
            for b in names[i + 1:]:
                v = corr.at[a, b] if a in corr.index and b in corr.columns else np.nan
                if not np.isnan(v) and v > self.cfg.l4.corr_cluster_threshold:
                    parent[find(a)] = find(b)
        return {n: find(n) for n in names}

    def _cluster_risk(self, d: pd.Timestamp, held: dict[str, Position]) -> dict[str, float]:
        cmap = self._cluster_map(d, held)
        out: dict[str, float] = {}
        for s, p in held.items():
            k = cmap.get(s, s)
            out[k] = out.get(k, 0.0) + p.size_mult
        return out


def _is_hard(reasons: str) -> bool:
    """硬性排除中「規則性」項目（交易狀態、會計師意見）觸發立即賣出；
    流動性/市值/虧損等慢變數改走待出場流程，避免一次跌破門檻就砍單。"""
    hard = ("全額交割", "停止買賣", "下市警示", "處置股", "注意股", "會計師")
    return any(h in reasons for h in hard)
