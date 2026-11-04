# -*- coding: utf-8 -*-
"""資料模組：從 FinMind 抓日K，處理分割還原，並可轉成週線。

FinMind 免費帳號不能使用「還原股價」資料集（TaiwanStockPriceAdj 只限付費會員），
所以這裡用原始日K（TaiwanStockPrice）＋程式內建的分割表自行還原。
- 00631L 不配息，分割還原後就是正確的還原價。
- 其他會配息的股票，沒有還原除權息，報酬會被低估（頁面會顯示提醒）。
"""
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TW_TZ = timezone(timedelta(hours=8))
DATA_START = date(2010, 1, 1)  # 一律抓完整歷史，回測區間在之後才切，指標才有足夠的暖機資料

# 已知分割事件：{代號: [(生效日, 分割倍數), ...]}
SPLIT_EVENTS = {
    "00631L": [("2026-03-31", 22)],
}
# 不配息的標的（分割還原後即為完整還原價）
NO_DIVIDEND = {"00631L", "TAIEX"}

# 單日變動超過 30% 才警告（2 倍槓桿 ETF 單日最多約 20%，分割約 95%）
JUMP_WARN_LIMIT = 0.30


def today_tw():
    return datetime.now(TW_TZ).date()


def _finmind_request(dataset, data_id, start_date_obj, token):
    params = {
        "dataset": dataset,
        "data_id": data_id,
        "start_date": start_date_obj.strftime("%Y-%m-%d"),
        "end_date": today_tw().strftime("%Y-%m-%d"),
    }
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(FINMIND_URL, params=params, headers=headers, timeout=30)
    res.raise_for_status()
    payload = res.json()
    if payload.get("status", 200) != 200:
        raise RuntimeError(f"FinMind 回應錯誤（{dataset} {data_id}）：{payload.get('msg', payload.get('status'))}")
    return pd.DataFrame(payload.get("data", []))


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_daily(symbol, token):
    """抓取原始日K。失敗會直接丟出錯誤（不會被快取）。"""
    df = _finmind_request("TaiwanStockPrice", symbol, DATA_START, token)
    if df.empty:
        raise ValueError(f"FinMind 沒有回傳 {symbol} 的日K資料，請確認代號是否正確")

    df = df.rename(columns={"date": "Date", "open": "Open", "max": "High",
                            "min": "Low", "close": "Close", "Trading_Volume": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"])
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df.reindex(columns=["Date"] + cols)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 沒有公告成交價的日子，FinMind 會把開高低收填 0，必須排除
    df = df[(df["Close"] > 0) & (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0)]
    df = df.drop_duplicates(subset="Date", keep="last").set_index("Date").sort_index()
    if df.empty:
        raise ValueError(f"{symbol} 沒有有效的價格資料")
    return df


def apply_split_adjustment(df, symbol):
    """依 SPLIT_EVENTS 還原分割前的價格與成交量。資料已還原過時會自動略過。回傳 (資料, 說明)。"""
    notes = []
    df = df.copy()
    for date_str, factor in SPLIT_EVENTS.get(symbol, []):
        eff = pd.Timestamp(date_str)
        before = df[df.index < eff]
        after = df[df.index >= eff]
        if before.empty or after.empty:
            continue
        ratio = before["Close"].iloc[-1] / after["Close"].iloc[0]
        if factor * 0.8 <= ratio <= factor * 1.2:
            mask = df.index < eff
            for c in ["Open", "High", "Low", "Close"]:
                df.loc[mask, c] = df.loc[mask, c] / factor
            df.loc[mask, "Volume"] = df.loc[mask, "Volume"] * factor
            notes.append(("info", f"已依 1 拆 {factor} 還原：{date_str} 之前的價格 ÷ {factor}"))
    return df, notes


def load_price(symbol, token):
    """讀取標的日K並完成還原。回傳 (日K, 提示訊息清單)。"""
    df, notes = apply_split_adjustment(fetch_daily(symbol, token), symbol)
    chg = df["Close"].pct_change().abs()
    for dt, v in chg[chg > JUMP_WARN_LIMIT].items():
        notes.append(("warn", f"{dt:%Y-%m-%d} 單日價格變動 {v*100:.1f}%，可能有尚未還原的分割或減資，"
                              f"請在 bt/data.py 的 SPLIT_EVENTS 補上。"))
    if symbol not in NO_DIVIDEND:
        notes.append(("warn", f"{symbol} 未還原除權息（FinMind 免費帳號無還原股價），配息股的報酬會被低估。"))
    return df, notes


def load_benchmark(bench_id, token):
    """讀取大盤（預設 TAIEX 加權指數）日K。"""
    df, _ = apply_split_adjustment(fetch_daily(bench_id, token), bench_id)
    return df


def resample_bars(df, timeframe="D"):
    """日線直接回傳；週線以「週一到週日」為一週合併，日期標在該週最後一個交易日。"""
    if timeframe != "W":
        return df.copy()
    week = df.index.to_period("W-SUN")
    g = df.groupby(week)
    out = pd.DataFrame({
        "Open": g["Open"].first(),
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].last(),
        "Volume": g["Volume"].sum(),
    })
    out.index = pd.DatetimeIndex(df.index.to_series().groupby(week).max().values, name="Date")
    return out
