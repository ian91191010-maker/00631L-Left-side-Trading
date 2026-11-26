# -*- coding: utf-8 -*-
"""網頁的快取層：所有 st.cache_data / st.cache_resource 都集中在這裡。

TTL 規則
- 日K、今年的台指期、計算結果：一律 TTL_DATA（2 小時），內外層相同，畫面寫的更新頻率才是真的。
- 已結束的年份（去年以前）台指期：30 天。資料不會再變，平常只會抓「今年」1 次請求。

「重新取得最新資料」只清這一頁用到的快取，而且全站 10 分鐘內只能按一次，
避免公開網站被一直按而用光 FinMind 額度。
"""
import math
import time

import pandas as pd
import streamlit as st

from bt import data as D
from bt import f0 as F
from bt import gauges as G

TICKER = "00631L"
BENCH = "TAIEX"
TTL_DATA = 2 * 3600
TTL_CLOSED_YEAR = 30 * 24 * 3600
REFRESH_COOLDOWN = 10 * 60
CLOSED_YEAR_GRACE_DAYS = 7      # 新年頭 7 天，去年仍當「今年」處理，確保最後幾天資料已補齊


# ---------- 原始資料 ----------
@st.cache_data(show_spinner=False, ttl=TTL_DATA)
def price_raw(symbol, token):
    return D.fetch_daily(symbol, token)


@st.cache_data(show_spinner=False, ttl=TTL_DATA)
def tx_year_open(year, token):
    return F.fetch_tx_year(year, token)


@st.cache_data(show_spinner=False, ttl=TTL_CLOSED_YEAR, max_entries=30)
def tx_year_closed(year, token):
    return F.fetch_tx_year(year, token)


def is_closed_year(year, today=None):
    today = pd.Timestamp(today or D.today_tw())
    return year < today.year and (today - pd.Timestamp(year=today.year, month=1, day=1)).days >= CLOSED_YEAR_GRACE_DAYS


def tx_year(year, token):
    return (tx_year_closed if is_closed_year(year) else tx_year_open)(year, token)


# ---------- 計算結果 ----------
@st.cache_data(show_spinner=False, ttl=TTL_DATA)
def signal_core(token):
    """00631L 日K（已還原）、F0 訊號表、提示訊息。這兩份資料失敗時整頁無法顯示。"""
    price, notes = D.prepare_price(price_raw(TICKER, token), TICKER)
    sig = F.compute_f0(F.fetch_tx(token, fetch_year=tx_year))
    return price, sig, notes


@st.cache_data(show_spinner=False, ttl=TTL_DATA)
def risk_gauge(token):
    """風險指標（加權指數 DIF）。只供參考，失敗不影響訊號。"""
    taiex = D.prepare_price(price_raw(BENCH, token), BENCH)[0]
    return G.risk_gauge(taiex)


@st.cache_data(show_spinner=False, ttl=1800)
def g5_table(mtime):
    """風控指標 CSV（檔案修改時間當快取 key，檔案一更新就重讀）。回傳 (表, 錯誤訊息或 None)。"""
    return G.load_g5()


# ---------- 重新取得 ----------
@st.cache_resource
def _refresh_state():
    return {"last": 0.0}


def refresh(token):
    """清掉這一頁用到的快取（內層原始資料＋外層計算結果）。回傳 (是否已清除, 提示文字)。"""
    state = _refresh_state()
    wait = REFRESH_COOLDOWN - (time.time() - state["last"])
    if wait > 0:
        return False, f"資料剛在 {math.ceil((REFRESH_COOLDOWN - wait) / 60)} 分鐘內更新過，請 {math.ceil(wait / 60)} 分鐘後再試。"
    state["last"] = time.time()
    price_raw.clear(TICKER, token)
    price_raw.clear(BENCH, token)
    tx_year_open.clear()            # 只有今年（新年頭幾天含去年），已結束年份保留
    signal_core.clear(token)
    risk_gauge.clear(token)
    return True, "已重新取得最新資料。"
