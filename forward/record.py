# -*- coding: utf-8 -*-
"""前向記錄：每天把「轉折 F0」與「審計系統二（原版）」的訊號記下來，純做紀錄。

- 只增不改：用「在檔案尾端追加」的方式寫入，已經記下的列永遠不會被改動。
- 只記「今天以前」的交易日（當天的台指期結算價可能還沒定案）；所以每天早上跑，記下前一個交易日。
- 記錄方式欄：
    即時＝在訊號日隔天以內記下（早於下一個交易日開盤）
    補記＝隔天以後才記（例如排程停了幾天）。補記的訊號只用到當天以前的資料，沒有偷看未來。
- 訊號算法直接用 bt/f0.py 與 bt/audit2.py（與網頁相同），不另寫一份。

執行：  FINMIND_TOKEN=你的token python forward/record.py
"""
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bt import audit2 as A2   # noqa: E402
from bt import data as D      # noqa: E402
from bt import f0 as F        # noqa: E402

LOG_PATH = ROOT / "data" / "forward_log.csv"
TICKER = "00631L"
START_DATE = pd.Timestamp("2026-11-24")   # 前向記錄起算：2026-11-24 收盤
COLUMNS = ["date", "tx_close", "tx_turning_n", "f0_hold", "f0_action",
           "a2_hold", "a2_action", "etf_close", "recorded_on", "kind"]


def _action(now, prev):
    """持有狀態（1／0）與前一天的差 → 動作文字。"""
    if now > prev:
        return "買進"
    if now < prev:
        return "賣出"
    return "持有" if now > 0 else "空手"


def build_rows(sig, a2, start, today, recorded_on):
    """由 F0 訊號表（bt.f0.compute_f0）與審計系統二訊號表（bt.audit2.compute_signals）組出記錄列。

    只取 start 以後、today 以前（不含 today）的日期；兩邊都有資料才記。
    回傳 DataFrame（欄位 COLUMNS，依日期排序）。
    """
    sig = sig.dropna(subset=["N"])
    days = sig.index.intersection(a2.index)
    days = days[(days >= start) & (days < pd.Timestamp(today))]
    rows = []
    for d in days:
        i_f, i_a = sig.index.get_loc(d), a2.index.get_loc(d)
        f0_prev = sig["Target"].iat[i_f - 1] if i_f > 0 else 0.0
        a2_prev = a2["Position"].iat[i_a - 1] if i_a > 0 else 0.0
        f0_now, a2_now = float(sig["Target"].iat[i_f]), float(a2["Position"].iat[i_a])
        late = (pd.Timestamp(recorded_on) - d) > timedelta(days=1)
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "tx_close": round(float(sig["Close"].iat[i_f]), 2),
            "tx_turning_n": int(sig["N"].iat[i_f]),
            "f0_hold": int(f0_now),
            "f0_action": _action(f0_now, f0_prev),
            "a2_hold": int(a2_now),
            "a2_action": _action(a2_now, a2_prev),
            "etf_close": round(float(a2["Close"].iat[i_a]), 2),
            "recorded_on": pd.Timestamp(recorded_on).strftime("%Y-%m-%d"),
            "kind": "補記" if late else "即時",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def append_new(rows, path=LOG_PATH):
    """只追加還沒記過的日期；已存在的列一律不動。回傳新增的列數。"""
    path = Path(path)
    have = set()
    if path.exists() and path.stat().st_size > 0:
        have = set(pd.read_csv(path, usecols=["date"], dtype=str)["date"])
    new = rows[~rows["date"].isin(have)]
    if new.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        new.to_csv(f, index=False, header=write_header, lineterminator="\n")
    return len(new)


def main():
    token = os.environ.get("FINMIND_TOKEN", "")
    if not token:
        sys.exit("沒有 FINMIND_TOKEN：請設定環境變數（GitHub 上是 Settings → Secrets → Actions）")
    today = D.today_tw()
    price, _ = D.load_price(TICKER, token)
    sig = F.compute_f0(F.fetch_tx(token))
    a2 = A2.compute_signals(price)
    rows = build_rows(sig, a2, START_DATE, today, today)
    n = append_new(rows)
    last = rows["date"].max() if len(rows) else "—"
    print(f"今天 {today}；可記錄 {len(rows)} 天（最新 {last}）；新增 {n} 列 → {LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
