"""實盤監控儀表板：輸出單一 HTML 檔（內嵌 SVG，無外部相依）。

內容：大盤狀態燈號、候選清單、持股安全分與停損距離、待執行委託、
累積與滾動 50 筆統計（含信賴區間）、R 倍數分布、資金曲線與回撤、CUSUM、紅黃燈。
"""
from __future__ import annotations

import html
import math
from datetime import datetime

import numpy as np
import pandas as pd

from .l1_regime import STATE_ZH
from .monitor import GREEN, RED, YELLOW, Light, overall
from .stats import bootstrap_expectancy_ci, cusum, trade_stats

STATE_COLOR = {"BULL": "var(--good)", "WARNING": "var(--warn)", "BEAR": "var(--bad)", "RECOVERY": "var(--info)"}
LIGHT_COLOR = {GREEN: "var(--good)", YELLOW: "var(--warn)", RED: "var(--bad)"}
IND_ZH = {
    "above_long_ma": "指數 vs 200 日線", "mid_ma_slope": "60 日線斜率", "tsmom_12m": "12 個月報酬",
    "breadth_above_ma": "站上 60 日線比例", "new_high_minus_low": "新高 − 新低家數",
    "futures_net_short_not_rising": "外資期貨淨空單", "margin_not_overheated": "融資 / 維持率",
    "leading_indicator_up": "景氣領先指標", "m1b_above_m2": "M1B > M2",
}


def _e(x) -> str:
    return html.escape(str(x))


def _fmt(v, kind="num") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "r":
        return f"{v:+.2f}R"
    if kind == "int":
        return f"{int(v):,}"
    return f"{v:,.2f}"


# ---------------------------------------------------------------- SVG
def line_chart(series: dict[str, pd.Series], height: int = 180, fmt="{:.0f}",
               colors=("var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"), bands: pd.Series | None = None,
               band_colors: dict | None = None, hlines: list[tuple[float, str]] | None = None,
               xaxis: bool = True) -> str:
    series = {k: v.dropna() for k, v in series.items() if v is not None and len(v.dropna()) > 1}
    if not series:
        return '<p class="muted">資料不足</p>'
    W, H, L, R, T, B = 720, height, 72, 10, 10, 22
    allv = pd.concat(series.values())
    lo, hi = float(allv.min()), float(allv.max())
    for y, _ in hlines or []:
        lo, hi = min(lo, y), max(hi, y)
    if hi == lo:
        hi = lo + 1
    x0 = min(s.index.min() for s in series.values())
    x1 = max(s.index.max() for s in series.values())
    span = (x1 - x0).total_seconds() or 1

    def X(t):
        return L + (W - L - R) * (t - x0).total_seconds() / span

    def Y(v):
        return T + (H - T - B) * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    if bands is not None and band_colors:
        b = bands.dropna()
        seg_start = None
        for i, (t, v) in enumerate(b.items()):
            nxt = b.index[i + 1] if i + 1 < len(b) else x1
            if seg_start is None:
                seg_start, cur = t, v
            if i + 1 == len(b) or b.iloc[i + 1] != cur:
                parts.append(f'<rect x="{X(seg_start):.1f}" y="{T}" width="{max(X(nxt) - X(seg_start), 0.5):.1f}" '
                             f'height="{H - T - B}" fill="{band_colors.get(cur, "transparent")}" opacity="0.13"/>')
                seg_start = None
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                     f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{_e(fmt.format(v))}</text>')
    for y, label in hlines or []:
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(y):.1f}" y2="{Y(y):.1f}" class="hline"/>'
                     f'<text x="{W - R}" y="{Y(y) - 3:.1f}" class="tick" text-anchor="end">{_e(label)}</text>')
    years = pd.date_range(x0, x1, freq="YS")
    ticks = years if 2 <= len(years) <= 14 else pd.date_range(x0, x1, periods=5)
    for t in (ticks if xaxis else []):
        parts.append(f'<text x="{X(t):.1f}" y="{H - 5}" class="tick" text-anchor="middle">'
                     f'{t.year if len(years) >= 2 else t.strftime("%m/%d")}</text>')
    legend = []
    for i, (name, s) in enumerate(series.items()):
        step = max(1, len(s) // 600)
        s2 = s.iloc[::step]
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in s2.items())
        col = colors[i % len(colors)]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6"/>')
        legend.append(f'<span><i style="background:{col}"></i>{_e(name)}</span>')
    parts.append("</svg>")
    return f'<div class="legend">{"".join(legend)}</div>' + "".join(parts) if len(series) > 1 else "".join(parts)


def histogram(values, bins: int = 24, height: int = 160) -> str:
    v = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(v) < 3:
        return '<p class="muted">交易筆數不足</p>'
    v = np.clip(v, -3, 10)
    counts, edges = np.histogram(v, bins=bins)
    W, H, L, B = 720, height, 30, 22
    bw = (W - L - 10) / bins
    mx = counts.max() or 1
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for i, c in enumerate(counts):
        h = (H - B - 8) * c / mx
        col = "var(--good)" if edges[i] >= 0 else "var(--bad)"
        out.append(f'<rect x="{L + i * bw + 1:.1f}" y="{H - B - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" '
                   f'fill="{col}" opacity="0.8"><title>{edges[i]:.1f}~{edges[i + 1]:.1f}R：{c} 筆</title></rect>')
    for x in (-2, -1, 0, 1, 2, 5, 10):
        if edges[0] <= x <= edges[-1]:
            px = L + (x - edges[0]) / (edges[-1] - edges[0]) * (W - L - 10)
            out.append(f'<text x="{px:.1f}" y="{H - 5}" class="tick" text-anchor="middle">{x}R</text>')
    out.append("</svg>")
    return "".join(out)


def table(df: pd.DataFrame, formats: dict[str, str] | None = None, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return '<p class="muted">（無）</p>'
    formats = formats or {}
    head = "".join(f"<th>{_e(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.head(max_rows).iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            kind = formats.get(c)
            if isinstance(v, (int, float, np.floating, np.integer)) and kind != "raw":
                cells.append(f'<td class="num">{_fmt(float(v), kind or "num")}</td>')
            else:
                cells.append(f"<td>{v if kind == 'html' else _e('' if pd.isna(v) else v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


# ---------------------------------------------------------------- 頁面
def render(regime: pd.DataFrame, candidates: pd.DataFrame | None = None, holdings: pd.DataFrame | None = None,
           orders: pd.DataFrame | None = None, trades: pd.DataFrame | None = None,
           equity: pd.Series | None = None, lights: list[Light] | None = None,
           mc_bands: dict[str, float] | None = None, title: str = "台股趨勢追蹤", note: str = "") -> str:
    reg = regime.dropna(subset=["state"])
    last = reg.iloc[-1]
    state = last["state"]
    as_of = reg.index[-1].strftime("%Y-%m-%d")
    ind_rows = []
    for k, zh in IND_ZH.items():
        if k in reg.columns:
            v = last[k]
            mark = "—" if pd.isna(v) else ("✓ 多" if v == 1 else "✗ 空")
            cls = "" if pd.isna(v) else ("ok" if v == 1 else "ng")
            ind_rows.append(f'<li class="{cls}"><span>{zh}</span><b>{mark}</b></li>')
    hero = f"""
<section class="hero">
  <div class="lamp" style="--c:{STATE_COLOR.get(state, 'var(--muted)')}"><span></span></div>
  <div>
    <div class="muted">大盤狀態 · {as_of}</div>
    <div class="state">{_e(STATE_ZH.get(state, state))}</div>
    <div class="kpis">
      <div><small>合計分</small><b>{last['score']:.0f}</b></div>
      <div><small>持股上限</small><b>{last['cap']:.0%}</b></div>
      <div><small>分批目標曝險</small><b>{last['target_exposure']:.0%}</b></div>
      <div><small>市場寬度</small><b>{_fmt(last.get('breadth'), 'pct')}</b></div>
    </div>
  </div>
  <ul class="inds">{''.join(ind_rows)}</ul>
</section>"""
    recent = reg.loc[reg.index[-1] - pd.Timedelta(days=365 * 3):]
    band_c = {"BULL": "var(--good)", "WARNING": "var(--warn)", "BEAR": "var(--bad)", "RECOVERY": "var(--info)"}
    regime_chart = line_chart({"判斷用指數": recent["index"], "200 日線": recent["long_ma"]},
                              bands=recent["state"], band_colors=band_c)
    score_chart = line_chart({"L1 合計分": recent["score"]}, height=120)

    sections = [hero, f'<section><h2>大盤指數與狀態（近 3 年）</h2>{regime_chart}'
                      f'<p class="muted">底色：綠 多頭、黃 轉弱警戒、紅 空頭、藍 打底轉強</p>{score_chart}</section>']

    if holdings is not None:
        sections.append(f"<section><h2>持股：安全分與停損距離</h2>{table(holdings, {'停損距離': 'pct', '損益': 'pct', '股數': 'int', '安全分': 'int'})}</section>")
    if orders is not None:
        sections.append(f"<section><h2>待執行委託</h2>{table(orders, {'股數': 'int'})}</section>")
    if candidates is not None:
        sections.append(f"<section><h2>候選清單（L3 通過、L2 前 30%）</h2>"
                        f"{table(candidates, {'排名': 'int', '安全分': 'int', '52週高比': 'pct', '12月報酬': 'pct', '營收YoY': 'pct'})}</section>")

    if trades is not None and not trades.empty and "r_multiple" in trades:
        r = trades["r_multiple"].dropna()
        st = trade_stats(r)
        mean, lo, hi = bootstrap_expectancy_ci(r, trades.get("exit_date"), n_boot=2000)
        roll = r.tail(50)
        st50 = trade_stats(roll)
        m50, lo50, hi50 = bootstrap_expectancy_ci(roll, n_boot=2000)
        stats_tbl = pd.DataFrame({
            "": ["交易筆數", "勝率", "盈虧比", "期望值", "期望值 95% CI", "前 5% 交易獲利占比", "最長連虧", "實際虧損÷1R"],
            "累積": [st.n, _fmt(st.win_rate, "pct"), _fmt(st.payoff), _fmt(st.expectancy, "r"),
                   f"{_fmt(lo, 'r')} ~ {_fmt(hi, 'r')}", _fmt(st.top5_share, "pct"), st.max_losing_streak,
                   _fmt(st.loss_vs_1r)],
            "滾動 50 筆": [st50.n, _fmt(st50.win_rate, "pct"), _fmt(st50.payoff), _fmt(st50.expectancy, "r"),
                       f"{_fmt(lo50, 'r')} ~ {_fmt(hi50, 'r')}", _fmt(st50.top5_share, "pct"),
                       st50.max_losing_streak, _fmt(st50.loss_vs_1r)],
        })
        sections.append(f"<section><h2>交易統計</h2>{table(stats_tbl, {'累積': 'raw', '滾動 50 筆': 'raw'})}"
                        f"<h3>R 倍數分布</h3>{histogram(r)}</section>")
        cs = cusum(r.values, target=0.20 if st.n < 100 else st.expectancy)
        cs_series = pd.Series(cs["cusum"].values, index=pd.date_range("2000-01-01", periods=len(cs), freq="D"))
        sections.append(f"<section><h2>R 值 CUSUM 控制圖（橫軸為交易序號）</h2>"
                        f"{line_chart({'CUSUM': cs_series}, height=140, fmt='{:.1f}', hlines=[(float(cs['h4'].iloc[-1]), 'h=4σ'), (float(cs['h5'].iloc[-1]), 'h=5σ')], xaxis=False)}</section>")

    if equity is not None and len(equity) > 1:
        dd = equity / equity.cummax() - 1
        hl = []
        if mc_bands:
            hl = [(-mc_bands["p95"], "MC MDD 95%"), (-mc_bands["p99"], "MC MDD 99%")]
        sections.append(f"<section><h2>資金曲線與回撤</h2>{line_chart({'權益': equity}, fmt='{:,.0f}')}"
                        f"{line_chart({'回撤': dd}, height=130, fmt='{:.0%}', hlines=hl)}</section>")

    if lights:
        st_all = overall(lights)
        rows = "".join(
            f'<tr><td><span class="dot" style="background:{LIGHT_COLOR[l.level]}"></span>{_e(l.level)}</td>'
            f'<td>{_e(l.metric)}</td><td class="num">{_fmt(l.value)}</td><td class="num">{_fmt(l.yellow)}</td>'
            f'<td class="num">{_fmt(l.red)}</td><td>{_e(l.note)}</td></tr>' for l in lights)
        sections.append(f'<section><h2>紅黃燈 <span class="pill" style="background:{LIGHT_COLOR[st_all]}">{_e(st_all)}</span></h2>'
                        f'<div class="tw"><table><thead><tr><th>燈號</th><th>指標</th><th>目前</th><th>黃燈</th><th>紅燈</th><th>備註</th></tr></thead>'
                        f'<tbody>{rows}</tbody></table></div>'
                        f'<p class="muted">黃燈：檢討、不改規則。紅燈：暫停新部位、重新驗證，並退回上一資金階段。</p></section>')

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title>
<style>{CSS}</style></head><body><main>
<header><h1>{_e(title)}</h1><div class="muted">產生時間 {gen}{(' · ' + _e(note)) if note else ''}</div></header>
{''.join(sections)}
<footer class="muted">本頁由程式自動產生。訊號只依事前寫定的規則；偏離訊號請記錄在交易日誌。</footer>
</main></body></html>"""


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;--good:#1f9d55;--warn:#d69e2e;
--bad:#d64545;--info:#3b7dd8;--s1:#3b7dd8;--s2:#c06a2b;--s3:#7a5cc4;--s4:#2a9d8f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141416;--card:#1d1d20;--fg:#ececec;
--muted:#9a9aa0;--line:#2e2e33;--s1:#6ea2ef;--s2:#e08a4c}}
:root[data-theme="dark"]{--bg:#141416;--card:#1d1d20;--fg:#ececec;--muted:#9a9aa0;--line:#2e2e33;--s1:#6ea2ef;--s2:#e08a4c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif}
main{max-width:1000px;margin:0 auto;padding:20px 16px 60px}h1{font-size:22px;margin:0}
h2{font-size:17px;margin:0 0 10px}h3{font-size:15px;margin:16px 0 6px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}
.muted{color:var(--muted);font-size:13px}.hero{display:grid;grid-template-columns:auto 1fr;gap:16px;align-items:center}
.hero .inds{grid-column:1/-1;list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px}
.inds li{display:flex;justify-content:space-between;border:1px solid var(--line);border-radius:8px;padding:4px 10px;font-size:13px}
.inds li.ok b{color:var(--good)}.inds li.ng b{color:var(--bad)}
.lamp span{display:block;width:64px;height:64px;border-radius:50%;background:var(--c);box-shadow:0 0 24px var(--c)}
.state{font-size:30px;font-weight:700}.kpis{display:flex;flex-wrap:wrap;gap:18px;margin-top:4px}
.kpis small{display:block;color:var(--muted);font-size:12px}.kpis b{font-size:18px}
.chart{width:100%;height:auto;display:block}.chart .grid{stroke:var(--line)}.chart .hline{stroke:var(--bad);stroke-dasharray:4 3}
.chart .tick{fill:var(--muted);font-size:11px}.legend{display:flex;gap:14px;font-size:12px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:3px;margin-right:5px;vertical-align:middle}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums}th{color:var(--muted);font-weight:600}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.pill{color:#fff;border-radius:99px;padding:1px 10px;font-size:13px;margin-left:6px}
footer{margin-top:20px}
"""


def holdings_frame(positions, md, safety: pd.DataFrame, as_of) -> pd.DataFrame:
    rows = []
    d = pd.Timestamp(as_of)
    for s, p in positions.items():
        px = md.close.loc[:d, s].dropna()
        c = float(px.iloc[-1]) if len(px) else p.entry_price
        rows.append({
            "代碼": s, "股數": p.shares, "成本": p.entry_price, "收盤": c, "損益": c / p.entry_price - 1,
            "停損價": p.stop, "停損距離": (c - p.stop) / c, "極端停損": p.catastrophic_stop,
            "安全分": float(safety.at[s, "score"]) if s in safety.index else np.nan,
            "狀態": "待出場" if p.pending_exit_since is not None else "",
        })
    return pd.DataFrame(rows)


def candidates_frame(trend: pd.DataFrame, safety: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    t = trend[trend["buy_zone"] & trend["trend_score"].notna()].head(n)
    return pd.DataFrame({
        "排名": t["rank"].astype(int).values, "代碼": list(t.index), "趨勢分": t["trend_score"].values,
        "安全分": safety["score"].reindex(t.index).values, "52週高比": t["high_52w_ratio"].values,
        "12月報酬": t["tsmom_12m"].values, "營收YoY": t["revenue_yoy"].values,
        "部位": safety["size_mult"].reindex(t.index).map({1.0: "完整", 0.5: "減半"}).values,
    })
