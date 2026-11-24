# -*- coding: utf-8 -*-
"""00631L Signal 頁面下方兩張指標圖的資料。

風險指標：加權指數的日 MACD(200, 209, 210) 的 DIF（價格用 (H+L+2C)/4），
          另算近 750 個交易日的 95%／5% 位置線。
風控指標：加權指數 60 分 K 的 RSI60，每天取最後一根的值與當天最低值。
          60 分 K 在公開網站拿不到，資料由私人庫算好後放進 data/g5_daily.csv：

    date,g5_close,g5_min
    2026-11-24,47.81,41.20

    date      交易日（YYYY-MM-DD）
    g5_close  當天最後一根 60 分 K 的 RSI60
    g5_min    當天所有 60 分 K 的 RSI60 最低值（盤中觸及用）
"""
from pathlib import Path

import pandas as pd

from . import indicators as ti

G5_PATH = Path(__file__).resolve().parent.parent / "data" / "g5_daily.csv"
DIF_FAST, DIF_SLOW = 200, 209
RANK_WINDOW = 750


def risk_gauge(taiex):
    """加權指數日 K → 含 DIF、DIF_HI（95% 位置）、DIF_LO（5% 位置）的表。"""
    p = (taiex["High"] + taiex["Low"] + 2 * taiex["Close"]) / 4
    dif = ti.ema(p, DIF_FAST) - ti.ema(p, DIF_SLOW)
    out = pd.DataFrame({"DIF": dif})
    out["DIF_HI"] = dif.rolling(RANK_WINDOW, min_periods=250).quantile(0.95)
    out["DIF_LO"] = dif.rolling(RANK_WINDOW, min_periods=250).quantile(0.05)
    return out


def load_g5(path=G5_PATH):
    """讀取風控指標資料。沒有檔案或沒有資料時回傳空表。"""
    path = Path(path)
    cols = ["g5_close", "g5_min"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
    df = pd.read_csv(path, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"])
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()[cols]
