"""證交所 / 櫃買中心 OpenAPI：處置股、注意股、全額交割（變更交易方法）。

端點與欄位標為「待確認」，上線前請對照
https://openapi.twse.com.tw/ 與 https://www.tpex.org.tw/openapi/ 的 Swagger 文件。
OpenAPI 只提供「當日」資料；歷史旗標需每日收盤後抓取並累積進 safety_flags。
"""
from __future__ import annotations

import re

import pandas as pd
import requests

# (來源, URL, 旗標類型)
ENDPOINTS = [
    ("twse", "https://openapi.twse.com.tw/v1/announcement/punish", "disposal"),          # 處置有價證券（待確認）
    ("twse", "https://openapi.twse.com.tw/v1/announcement/notetrans", "attention"),      # 注意交易資訊（待確認）
    ("twse", "https://openapi.twse.com.tw/v1/exchangeReport/TWTAWU", "full_delivery"),   # 變更交易（待確認）
    ("tpex", "https://www.tpex.org.tw/openapi/v1/tpex_disposal_information", "disposal"),  # 待確認
    ("tpex", "https://www.tpex.org.tw/openapi/v1/tpex_trading_warning_information", "attention"),  # 待確認
]

_CODE_KEYS = ("Code", "SecuritiesCompanyCode", "證券代號", "SecurityCode", "StockNo")
_DATE_KEYS = ("Date", "DispositionStartDate", "公布日期", "日期", "PeriodStart")
_END_KEYS = ("DispositionEndDate", "PeriodEnd", "處置迄日")


def roc_to_date(s: str) -> str | None:
    """民國日期（1131105 / 113/11/05）→ 2024-11-05。"""
    if s is None:
        return None
    digits = re.sub(r"\D", "", str(s))
    if len(digits) == 7:
        y, m, d = int(digits[:3]) + 1911, int(digits[3:5]), int(digits[5:])
    elif len(digits) == 8:
        y, m, d = int(digits[:4]), int(digits[4:6]), int(digits[6:])
    else:
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def parse_flags(records: list[dict], flag_type: str, source: str, today: str) -> pd.DataFrame:
    rows = []
    for r in records:
        code = next((str(r[k]).strip() for k in _CODE_KEYS if k in r and r[k]), None)
        if not code or not re.fullmatch(r"\d{4,6}", code):
            continue
        start = next((roc_to_date(r[k]) for k in _DATE_KEYS if k in r and r[k]), None) or today
        end = next((roc_to_date(r[k]) for k in _END_KEYS if k in r and r[k]), None)
        # 處置期間有時寫在同一欄：「113/11/05～113/11/18」
        for v in r.values():
            m = re.search(r"(\d{2,3}/\d{1,2}/\d{1,2})\s*[～~至-]\s*(\d{2,3}/\d{1,2}/\d{1,2})", str(v))
            if m:
                start, end = roc_to_date(m.group(1)), roc_to_date(m.group(2))
                break
        if flag_type == "attention":
            end = start
        rows.append({"stock_id": code, "flag_type": flag_type, "start_date": start, "end_date": end, "source": source})
    return pd.DataFrame(rows, columns=["stock_id", "flag_type", "start_date", "end_date", "source"])


def fetch_flags(today: str, session: requests.Session | None = None) -> tuple[pd.DataFrame, list[str]]:
    s = session or requests.Session()
    frames, errors = [], []
    for source, url, ftype in ENDPOINTS:
        try:
            resp = s.get(url, timeout=30, headers={"Accept": "application/json"})
            resp.raise_for_status()
            frames.append(parse_flags(resp.json(), ftype, source, today))
        except Exception as e:  # noqa: BLE001 — 單一來源失敗不應中斷其他來源
            errors.append(f"{source} {ftype}: {e}")
    df = pd.concat(frames, ignore_index=True) if frames else parse_flags([], "", "", today)
    return df.drop_duplicates(["stock_id", "flag_type", "start_date"]), errors
