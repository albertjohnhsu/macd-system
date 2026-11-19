"""規則檢討：把模擬盤 / 回測的紀錄轉成「是否該修正規則」的證據。

原則（依設計書）：
- 每個發現都附樣本數與信賴區間；樣本不足只列為「觀察」，不建議修改
- 「正常的痛」不改規則：只有分組差異在統計上顯著、且方向與設計假設相反時才列為「候選修正」
- 候選修正一律走：新增 A/B 模擬帳戶（--set 覆寫參數）→ 前進測試 → 回測兩段（2015 前後）都穩定 → 才換版
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.market import MarketData
from .stats import trade_stats

MIN_GROUP = 20          # 分組至少筆數才比較
MIN_DIFF_TRADES = 30    # A/B 比較每邊至少筆數


# ---------------------------------------------------------------- 分組績效
def _ci(r: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(r) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(r, (n_boot, len(r))).mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def exit_category(reason: str) -> str:
    s = str(reason)
    for key, cat in (("手動", "手動"), ("L3", "L3 排除"), ("極端", "極端停損"), ("初始停損", "初始停損"),
                     ("追蹤停利", "追蹤停利"), ("L1", "L1 減碼"), ("安全分", "安全分待出場"),
                     ("MA60", "MA60 破線"), ("輪動", "排名輪動"), ("未平倉", "未平倉")):
        if key in s:
            return cat
    return "其他"


def add_buckets(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["出場原因"] = t["exit_reason"].map(exit_category)
    t["安全分組"] = pd.cut(t["safety_score"], [-1, 49.9, 69.9, 84.9, 101], labels=["<50", "50–69", "70–84", "85+"])
    if t["trend_score"].notna().sum() >= 8:
        t["趨勢分組"] = pd.qcut(t["trend_score"], 4, labels=["Q1 低", "Q2", "Q3", "Q4 高"], duplicates="drop")
    t["進場狀態"] = t["entry_state"].fillna("—")
    t["持有天數"] = pd.cut(t["holding_days"], [-1, 10, 30, 60, 120, 10_000],
                       labels=["≤10", "11–30", "31–60", "61–120", ">120"])
    t["手動"] = np.where((t.get("manual_entry", 0) == 1) | (t.get("manual_exit", 0) == 1), "手動介入", "依訊號")
    if "entry_date" in t:
        t["進場月份"] = pd.to_datetime(t["entry_date"]).dt.month.map(lambda m: "1 月" if m == 1 else "其他月")
    return t


def attribution(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    t = add_buckets(trades)
    rows = []
    for k, g in t.groupby(by, observed=True):
        r = g["r_multiple"].dropna().values
        if len(r) == 0:
            continue
        lo, hi = _ci(r)
        st = trade_stats(r)
        rows.append({by: k, "筆數": len(r), "勝率": st.win_rate, "平均R": st.expectancy, "CI下": lo, "CI上": hi,
                     "總R": float(r.sum()), "R占比": np.nan, "樣本足夠": len(r) >= MIN_GROUP})
    df = pd.DataFrame(rows)
    if not df.empty:
        tot = t["r_multiple"].sum()
        df["R占比"] = df["總R"] / tot if tot else np.nan
    return df


# ---------------------------------------------------------------- 事後追蹤
def forward_return(md: MarketData, sid: str, d, h: int) -> float:
    if sid not in md.close:
        return np.nan
    s = md.close[sid]
    after = s.loc[pd.Timestamp(d):].dropna()
    if len(after) <= h:
        return np.nan
    return float(after.iloc[h] / after.iloc[0] - 1)


def forward_max(md: MarketData, sid: str, d, h: int) -> float:
    if sid not in md.close:
        return np.nan
    after = md.close[sid].loc[pd.Timestamp(d):].dropna()
    if len(after) <= h:
        return np.nan
    return float(after.iloc[1:h + 1].max() / after.iloc[0] - 1)


def exit_followup(trades: pd.DataFrame, md: MarketData, horizons=(20, 60)) -> pd.DataFrame:
    """出場後走勢：停損 / 停利後股價又大漲的比例高，代表出場可能太緊。"""
    t = add_buckets(trades[trades["exit_reason"].map(exit_category) != "未平倉"])
    for h in horizons:
        t[f"出場後{h}日"] = [forward_return(md, s, d, h) for s, d in zip(t["stock_id"], t["exit_date"])]
    t["出場後60日最高"] = [forward_max(md, s, d, 60) for s, d in zip(t["stock_id"], t["exit_date"])]
    rows = []
    for k, g in t.groupby("出場原因"):
        valid = g[f"出場後{horizons[-1]}日"].dropna()
        rows.append({
            "出場原因": k, "筆數": len(g), "平均R": g["r_multiple"].mean(),
            **{f"出場後{h}日平均": g[f"出場後{h}日"].mean() for h in horizons},
            "60日內再漲>20%比例": (g["出場後60日最高"] > 0.20).mean() if g["出場後60日最高"].notna().any() else np.nan,
            "可追蹤筆數": len(valid),
        })
    return pd.DataFrame(rows)


def candidate_followup(sig: pd.DataFrame, md: MarketData, horizons=(20, 60)) -> pd.DataFrame:
    """候選股事後報酬：已買 vs 各種原因沒買。檢驗各項限制（L1、安全分、分群、產業上限）的機會成本。"""
    if sig.empty:
        return pd.DataFrame()
    s = sig.copy()
    for h in horizons:
        s[f"後{h}日"] = [forward_return(md, a, d, h) for a, d in zip(s["stock_id"], s["date"])]
    s["類別"] = np.where(s["decision"] == "buy", "已買", "未買：" + s["reason"].astype(str).str.replace(r"：.*", "", regex=True))
    rows = []
    for k, g in s.groupby("類別"):
        r = g[f"後{horizons[-1]}日"].dropna().values
        lo, hi = _ci(r) if len(r) >= 3 else (np.nan, np.nan)
        rows.append({"類別": k, "筆數": len(g), **{f"後{h}日平均": g[f"後{h}日"].mean() for h in horizons},
                     f"後{horizons[-1]}日勝率": float((r > 0).mean()) if len(r) else np.nan,
                     "CI下": lo, "CI上": hi})
    return pd.DataFrame(rows).sort_values("筆數", ascending=False)


# ---------------------------------------------------------------- A/B 比較
def compare_expectancy(a: pd.Series, b: pd.Series, n_boot: int = 4000, seed: int = 0) -> dict:
    """兩組 R 值期望值差（B − A）的 bootstrap 95% CI。"""
    ra, rb = np.asarray(a.dropna(), float), np.asarray(b.dropna(), float)
    out = {"A筆數": len(ra), "B筆數": len(rb), "A期望值": ra.mean() if len(ra) else np.nan,
           "B期望值": rb.mean() if len(rb) else np.nan}
    if len(ra) < 3 or len(rb) < 3:
        out.update({"差": np.nan, "CI下": np.nan, "CI上": np.nan, "結論": "樣本不足"})
        return out
    rng = np.random.default_rng(seed)
    diff = rng.choice(rb, (n_boot, len(rb))).mean(1) - rng.choice(ra, (n_boot, len(ra))).mean(1)
    lo, hi = np.quantile(diff, [0.025, 0.975])
    enough = min(len(ra), len(rb)) >= MIN_DIFF_TRADES
    concl = ("樣本不足" if not enough else "B 顯著較好" if lo > 0 else "B 顯著較差" if hi < 0 else "差異不顯著")
    out.update({"差": rb.mean() - ra.mean(), "CI下": float(lo), "CI上": float(hi), "結論": concl})
    return out


def compare_accounts(db, base: str, others: list[str], phase: str | None = None) -> pd.DataFrame:
    from . import paper
    from . import indicators as ind
    ta = paper.trades(db, base, phase)
    ea = paper.equity(db, base)
    rows = []
    for o in others:
        tb = paper.trades(db, o, phase)
        eb = paper.equity(db, o)
        common = ea.index.intersection(eb.index)
        if phase:
            common = common[ea.loc[common, "phase"] == phase]
        res = compare_expectancy(ta["r_multiple"], tb["r_multiple"])
        if len(common) > 1:
            a, b = ea.loc[common, "equity"], eb.loc[common, "equity"]
            res.update({"A報酬": a.iloc[-1] / a.iloc[0] - 1, "B報酬": b.iloc[-1] / b.iloc[0] - 1,
                        "A MDD": ind.max_drawdown(a), "B MDD": ind.max_drawdown(b)})
        cfg_a, cfg_b = paper.account_config(db, base), paper.account_config(db, o)
        res["規則差異"] = "; ".join(f"{k}: {v[0]}→{v[1]}" for k, v in paper.diff_rules(cfg_a, cfg_b).items()
                                if not k.startswith("l4.total_capital"))
        rows.append({"A": base, "B": o, **res})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 建議
@dataclass
class Finding:
    level: str          # 候選修正 / 觀察 / 正常
    topic: str
    evidence: str
    action: str


def _sig_lower(df: pd.DataFrame, key: str, low: str, high: str) -> tuple[bool, str]:
    if df.empty or key not in df:
        return False, ""
    a = df[df[key] == low]
    b = df[df[key] == high]
    if a.empty or b.empty:
        return False, ""
    a, b = a.iloc[0], b.iloc[0]
    txt = f"{low}：{a['平均R']:+.2f}R（{a['筆數']} 筆，CI {a['CI下']:+.2f}~{a['CI上']:+.2f}）；" \
          f"{high}：{b['平均R']:+.2f}R（{b['筆數']} 筆）"
    ok = a["樣本足夠"] and b["樣本足夠"] and a["CI上"] < b["CI下"]
    return bool(ok), txt


def findings(trades: pd.DataFrame, md: MarketData, sig: pd.DataFrame | None = None) -> list[Finding]:
    out: list[Finding] = []
    t = trades.dropna(subset=["r_multiple"])
    n = len(t)
    if n == 0:
        return [Finding("觀察", "樣本", "尚無已結算交易", "繼續累積交易，不做任何修改")]
    st = trade_stats(t["r_multiple"])
    lo, hi = _ci(t["r_multiple"].values)

    # 1. 整體期望值與兩個關鍵數字（勝率 30%、平均獲利 2.5R）
    level = "觀察" if n < 100 else ("候選修正" if hi < 0 else "正常")
    out.append(Finding(level, "整體期望值",
                       f"{n} 筆，期望值 {st.expectancy:+.2f}R（CI {lo:+.2f}~{hi:+.2f}），勝率 {st.win_rate:.0%}、"
                       f"平均獲利 {st.avg_win:.2f}R、盈虧比 {st.payoff:.2f}",
                       "樣本 < 100 筆前只檢查執行品質" if n < 100 else
                       ("期望值 CI 上緣 < 0：暫停新部位、重新驗證 L2 / 出場規則" if hi < 0 else "維持規則")))
    if n >= 50 and (st.win_rate < 0.30 or st.avg_win < 2.5):
        out.append(Finding("觀察", "結構條件",
                           f"勝率 {st.win_rate:.0%}（門檻 30%）、平均獲利 {st.avg_win:.2f}R（門檻 2.5R）",
                           "設計書指出兩者任一不足時結構難以獲利；先查是出場太早（看出場後走勢）還是選股品質"))

    # 2. 獲利集中度
    if n >= 50:
        out.append(Finding("正常" if st.skew > 0 else "觀察", "獲利集中度",
                           f"前 5% 交易占淨利 {st.top5_share:.0%}，偏態 {st.skew:.2f}",
                           "偏態為正屬預期，不可跳過訊號或設固定停利" if st.skew > 0 else
                           "偏態非正：大贏家被截斷，檢查追蹤停利倍數與是否過早移到成本價"))

    # 3. 安全分組
    ok, txt = _sig_lower(attribution(t, "安全分組"), "安全分組", "50–69", "85+")
    if txt:
        out.append(Finding("候選修正" if ok else "觀察", "L3 安全分門檻", txt,
                           "建立 A/B 帳戶：--set l3.half_size_min=60 或 l3.full_size_min=80" if ok else "差異不顯著，維持"))

    # 4. 趨勢分組（高分組應較好）
    at = attribution(t, "趨勢分組") if "trend_score" in t else pd.DataFrame()
    ok, txt = _sig_lower(at, "趨勢分組", "Q1 低", "Q4 高")
    if txt:
        out.append(Finding("候選修正" if ok else "觀察", "L2 買入區寬度", txt,
                           "建立 A/B 帳戶：--set l2.buy_top_pct=0.2" if ok else "差異不顯著，維持"))

    # 5. 出場後走勢：停利 / 停損後又大漲
    fu = exit_followup(t, md)
    for _, r in fu.iterrows():
        if r["出場原因"] in ("追蹤停利", "初始停損") and r["可追蹤筆數"] >= MIN_GROUP:
            p = r["60日內再漲>20%比例"]
            if p > 0.35:
                knob = "l4.trailing_atr_mult=3.5" if r["出場原因"] == "追蹤停利" else "l4.initial_stop_atr_mult=3.0"
                out.append(Finding("候選修正", f"{r['出場原因']}可能太緊",
                                   f"{r['可追蹤筆數']} 筆中 {p:.0%} 在出場後 60 日內再漲 > 20%",
                                   f"建立 A/B 帳戶：--set {knob}（停損放寬後，風險平價會自動縮小部位，每筆 1R 金額不變）"))
            else:
                out.append(Finding("正常", f"{r['出場原因']}出場後走勢",
                                   f"60 日內再漲 > 20% 的比例 {p:.0%}", "維持"))

    # 6. 手動介入
    if "manual_entry" in t:
        man = t[(t["manual_entry"] == 1) | (t.get("manual_exit", 0) == 1)]
        sysd = t.drop(man.index)
        if len(man) >= 5:
            cmp_ = compare_expectancy(sysd["r_multiple"], man["r_multiple"])
            out.append(Finding("觀察", "人為干預",
                               f"手動 {len(man)} 筆 {cmp_['B期望值']:+.2f}R vs 依訊號 {cmp_['A期望值']:+.2f}R（{cmp_['結論']}）",
                               "手動績效未顯著較好前，維持只依訊號執行"))

    # 7. 1 月季節性（52 週高點策略在台股 1 月偏弱）
    am = attribution(t, "進場月份")
    if not am.empty and set(am["進場月份"]) == {"1 月", "其他月"}:
        ok, txt = _sig_lower(am, "進場月份", "1 月", "其他月")
        out.append(Finding("候選修正" if ok else "觀察", "1 月季節性", txt,
                           "考慮 1 月降低新部位或提高 52 週高門檻（需回測兩段驗證）" if ok else "維持"))

    # 8. 候選機會成本
    if sig is not None and not sig.empty:
        cf = candidate_followup(sig, md)
        bought = cf[cf["類別"] == "已買"]
        for _, r in cf[cf["類別"].str.startswith("未買")].iterrows():
            if r["筆數"] >= MIN_GROUP * 2 and not bought.empty:
                b = bought.iloc[0]
                if pd.notna(r["CI下"]) and pd.notna(b["CI上"]) and r["CI下"] > b["CI上"]:
                    out.append(Finding("候選修正", f"限制的機會成本：{r['類別']}",
                                       f"未買組後 60 日平均 {r['後60日平均']:+.1%}，顯著高於已買組 {b['後60日平均']:+.1%}",
                                       "檢查該限制是否過嚴（L1 狀態門檻、分群上限、產業上限）；以 A/B 帳戶驗證"))
    return out


def findings_frame(fs: list[Finding]) -> pd.DataFrame:
    order = {"候選修正": 0, "觀察": 1, "正常": 2}
    df = pd.DataFrame([f.__dict__ for f in fs]).rename(
        columns={"level": "判定", "topic": "項目", "evidence": "證據", "action": "建議"})
    return df.sort_values("判定", key=lambda s: s.map(order)).reset_index(drop=True)
