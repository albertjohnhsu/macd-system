"""推播：Telegram Bot、LINE Messaging API。

LINE Notify 已於 2025-03-31 停止服務，因此改用 LINE Messaging API 的 push message。
權杖請放在環境變數，不要寫進設定檔：
  TWTREND_TELEGRAM_TOKEN / TWTREND_TELEGRAM_CHAT_ID
  TWTREND_LINE_CHANNEL_TOKEN / TWTREND_LINE_USER_ID
"""
from __future__ import annotations

import os

import requests

from .db import Database


def send_telegram(text: str, token: str, chat_id: str) -> None:
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat_id, "text": text[:4000]}, timeout=20)
    r.raise_for_status()


def send_line(text: str, channel_token: str, user_id: str) -> None:
    r = requests.post("https://api.line.me/v2/bot/message/push",
                      headers={"Authorization": f"Bearer {channel_token}"},
                      json={"to": user_id, "messages": [{"type": "text", "text": text[:4900]}]}, timeout=20)
    r.raise_for_status()


def channels() -> list[tuple[str, callable]]:
    out = []
    tg_t, tg_c = os.getenv("TWTREND_TELEGRAM_TOKEN"), os.getenv("TWTREND_TELEGRAM_CHAT_ID")
    if tg_t and tg_c:
        out.append(("telegram", lambda m: send_telegram(m, tg_t, tg_c)))
    ln_t, ln_u = os.getenv("TWTREND_LINE_CHANNEL_TOKEN"), os.getenv("TWTREND_LINE_USER_ID")
    if ln_t and ln_u:
        out.append(("line", lambda m: send_line(m, ln_t, ln_u)))
    return out


def flush_alerts(db: Database, dry_run: bool = False) -> list[str]:
    """把尚未送出的提醒合併成一則訊息推播；未設定任何通道時只印出。"""
    pending = db.query("SELECT id, date, kind, message FROM alerts WHERE sent = 0 ORDER BY id")
    if pending.empty:
        return []
    lines = [f"[{r.date}] {r.message}" for r in pending.itertuples()]
    text = "台股趨勢追蹤系統\n" + "\n".join(lines)
    chs = channels()
    if dry_run or not chs:
        print(text)
    else:
        for name, fn in chs:
            fn(text)
    db.execute(f"UPDATE alerts SET sent = 1 WHERE id IN ({','.join(str(i) for i in pending['id'])})")
    return lines
