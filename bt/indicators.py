# -*- coding: utf-8 -*-
"""技術指標模組。

所有函式都只用「當根K棒以前」的資料計算，沒有未來函數。
EMA / RMA 採用 TradingView 的算法：前 n 根先用簡單平均當起點，之後才遞迴計算，
所以數值會和 TradingView 圖表上的指標一致（資料夠長時）。

新增指標的方法：在這裡寫一個函式，輸入 pandas Series，回傳 pandas Series 即可。
"""
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 基礎均線
# ------------------------------------------------------------
def _seeded_recursive(series, n, alpha):
    """TradingView 式遞迴平均：第一個值 = 前 n 個有效值的簡單平均，之後 = alpha*今天 + (1-alpha)*昨天。"""
    x = np.asarray(series, dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < n:
        return pd.Series(out, index=getattr(series, "index", None))
    start = valid[0]
    seed_end = start + n - 1
    if seed_end >= len(x):
        return pd.Series(out, index=getattr(series, "index", None))
    out[seed_end] = np.nanmean(x[start:seed_end + 1])
    prev = out[seed_end]
    for i in range(seed_end + 1, len(x)):
        v = x[i]
        if np.isnan(v):          # 中間缺值時沿用前一個數值
            out[i] = prev
            continue
        prev = alpha * v + (1 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=getattr(series, "index", None))


def sma(series, n):
    """簡單移動平均（= TradingView ta.sma）。"""
    return series.rolling(window=int(n), min_periods=int(n)).mean()


def ema(series, n):
    """指數移動平均（= TradingView ta.ema）。"""
    n = int(n)
    return _seeded_recursive(series, n, 2.0 / (n + 1))


def rma(series, n):
    """Wilder 平滑（= TradingView ta.rma），ATR、ADX 使用。"""
    n = int(n)
    return _seeded_recursive(series, n, 1.0 / n)


def moving_average(series, n, ma_type="SMA"):
    """依設定選擇 SMA 或 EMA。"""
    return ema(series, n) if str(ma_type).upper() == "EMA" else sma(series, n)


# ------------------------------------------------------------
# TSI 真實強弱指數
# ------------------------------------------------------------
def tsi(close, r=25, s=13, signal=13):
    """TSI。
    動量 = close − close[1]
    TSI = 100 × EMA(EMA(動量, r), s) ÷ EMA(EMA(|動量|, r), s)
    信號線 = EMA(TSI, signal)
    回傳 (tsi, 信號線)。
    """
    mom = close.diff()
    num = ema(ema(mom, r), s)
    den = ema(ema(mom.abs(), r), s)
    t = 100 * num / den.replace(0, np.nan)
    return t, ema(t, signal)


# ------------------------------------------------------------
# ATR / ADX
# ------------------------------------------------------------
def true_range(high, low, close, handle_na=True):
    """真實波幅。handle_na=True 時第一根用 high−low（= TradingView ta.tr(true)）。"""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    if handle_na:
        tr.iloc[0] = high.iloc[0] - low.iloc[0]
    else:
        tr[prev_close.isna()] = np.nan
    return tr


def atr(high, low, close, n=14):
    """平均真實波幅（= TradingView ta.atr）。"""
    return rma(true_range(high, low, close, handle_na=True), n)


def adx(high, low, close, n=14, smoothing=14):
    """ADX（= TradingView ta.dmi 的 adx）。回傳 (adx, +DI, −DI)。"""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    plus_dm[up.isna()] = np.nan
    minus_dm[down.isna()] = np.nan

    tr_rma = rma(true_range(high, low, close, handle_na=False), n)
    plus_di = 100 * rma(plus_dm, n) / tr_rma
    minus_di = 100 * rma(minus_dm, n) / tr_rma
    total = plus_di + minus_di
    dx = (plus_di - minus_di).abs() / total.where(total != 0, 1.0)
    return 100 * rma(dx, smoothing), plus_di, minus_di


# ------------------------------------------------------------
# 交叉判斷小工具
# ------------------------------------------------------------
def cross_over(a, b):
    """a 由下往上穿越 b（今天 a>b 且昨天 a<=b）。b 可以是數字。"""
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift(1) <= b_prev)


def cross_under(a, b):
    """a 由上往下跌破 b（今天 a<b 且昨天 a>=b）。b 可以是數字。"""
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return (a < b) & (a.shift(1) >= b_prev)
