# -*- coding: utf-8 -*-
"""基準策略：主頁正在使用的「左側交易」審計策略。

判斷邏輯與主頁 app - 00631L-Left-side-Trading.py 的 compute_positions() 逐行相同，
指標也用同一個 ta 套件計算，確保兩邊訊號完全一致。
⚠ 修改主頁的策略邏輯時，這裡也要一起改。

成交規則與回測頁相同：
- 收盤決定目標部位，下一個交易日開盤成交（含滑價、手續費、證交稅）。
- 部位依策略本身的 30% / 60% / 100% 分段，不套用側邊欄的「倉位比例」。
- 加碼時買進「增加的部位 × 當時總資產」；目標部位歸零時全部賣出。
- 一律用日線計算（策略本來就是日線設計），均價與冷卻狀態從資料第一天開始累積，與主頁一致。
"""
import math

import numpy as np
import pandas as pd
import ta

NAME = "審計策略（基準）"


def compute_indicators(daily):
    df = daily[["Open", "High", "Low", "Close"]].copy()
    df["Adj_Close"] = df["Close"]
    df["RSI"] = ta.momentum.rsi(df["Adj_Close"], window=14)
    bb = ta.volatility.BollingerBands(df["Adj_Close"], window=20, window_dev=2.2)
    df["BB_Lower"] = bb.bollinger_lband()
    df["MA20"] = bb.bollinger_mavg()
    df["BIAS"] = (df["Adj_Close"] - df["MA20"]) / df["MA20"] * 100
    df["MA60"] = df["Adj_Close"].rolling(window=60).mean()
    return df


def compute_positions(df):
    """逐日計算目標部位（與主頁相同）。回傳 (目標部位, 均價, 冷卻狀態)。"""
    rsi_a = df["RSI"].to_numpy()
    bias_a = df["BIAS"].to_numpy()
    close_a = df["Adj_Close"].to_numpy()
    lower_a = df["BB_Lower"].to_numpy()
    ma20_a = df["MA20"].to_numpy()
    ma60_a = df["MA60"].to_numpy()

    n = len(df)
    positions = np.zeros(n)
    avg_costs = np.zeros(n)
    cooldowns = np.zeros(n, dtype=bool)
    current_pos = 0.0
    avg_cost = 0.0
    is_cooldown = False

    for i in range(1, n):
        rsi = rsi_a[i]
        bias = bias_a[i]
        close = close_a[i]
        lower_bb = lower_a[i]
        ma20 = ma20_a[i]
        ma60 = ma60_a[i]

        target_pos = current_pos

        if target_pos == 0:
            avg_cost = 0.0
            if rsi > 50 or close > ma20:
                is_cooldown = False

        is_bull_trend = close > ma60

        if is_bull_trend:
            if close > ma20:
                if not is_cooldown:
                    target_pos = 1.0
            else:
                target_pos = 0.0
        else:
            if close >= ma20 or rsi > 50:
                target_pos = 0.0
            if not is_cooldown:
                if rsi < 20 or bias < -12:
                    target_pos = max(target_pos, 1.0)
                elif rsi < 25 or close < lower_bb:
                    target_pos = max(target_pos, 0.6)
                elif rsi < 32 or bias < -6:
                    target_pos = max(target_pos, 0.3)

        if target_pos > current_pos:
            if current_pos == 0:
                avg_cost = close
            else:
                avg_cost = (avg_cost * current_pos + close * (target_pos - current_pos)) / target_pos
        elif target_pos == 0:
            avg_cost = 0.0

        if current_pos > 0 and avg_cost > 0:
            if close <= avg_cost * 0.85:
                target_pos = 0.0
                is_cooldown = True
                avg_cost = 0.0

        positions[i] = target_pos
        avg_costs[i] = avg_cost
        cooldowns[i] = is_cooldown
        current_pos = target_pos

    return positions, avg_costs, cooldowns


def run_baseline(daily, settings, start, end):
    """模擬基準策略的資金變化。回傳格式與 strategy.run_strategy 相同（equity、shares、trades、fills）。"""
    ind = compute_indicators(daily)
    pos, _, cooldown = compute_positions(ind)
    idx = ind.index
    O = ind["Open"].to_numpy(float)
    C = ind["Close"].to_numpy(float)
    in_win = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))

    fee = settings.fee_rate / 100 * settings.fee_discount
    tax = settings.tax_rate / 100
    slip = settings.slippage / 100
    cash = float(settings.initial_cash)
    shares = 0
    held_target = 0.0          # 目前實際持有的目標部位層級
    trade = None
    trades, fills = [], []
    equity = np.full(len(ind), np.nan)
    shares_arr = np.zeros(len(ind))
    last_i = None

    def close_trade(i, is_open=False):
        nonlocal trade
        sell_net, sell_value, sell_qty = trade["sell_net"], trade["sell_value"], trade["sell_qty"]
        if is_open:
            sell_net += C[i] * shares * (1 - fee - tax)
            sell_value += C[i] * shares
            sell_qty += shares
        pnl = sell_net - trade["buy_cost"]
        trades.append({
            "entry_date": idx[trade["entry_i"]], "exit_date": pd.NaT if is_open else idx[i],
            "bars": i - trade["entry_i"], "entry_price": trade["buy_value"] / trade["buy_qty"],
            "exit_price": sell_value / sell_qty if sell_qty else math.nan,
            "pnl": pnl, "ret": pnl / trade["buy_cost"] * 100,
            "reason": "持有中" if is_open else trade["reason"], "is_open": is_open,
        })
        trade = None

    for i in range(1, len(ind)):
        if not in_win[i]:
            continue
        last_i = i
        target = pos[i - 1]                      # 前一天收盤決定的目標部位，今天開盤執行
        if target > held_target:                 # 加碼
            px = O[i] * (1 + slip)
            equity_now = cash + shares * O[i]
            budget = min((target - held_target) * equity_now, cash)
            qty = int(budget // (px * (1 + fee)))
            if qty > 0:
                cost = px * qty * (1 + fee)
                cash -= cost
                shares += qty
                if trade is None:
                    trade = {"entry_i": i, "buy_cost": 0.0, "buy_value": 0.0, "buy_qty": 0,
                             "sell_net": 0.0, "sell_value": 0.0, "sell_qty": 0, "reason": ""}
                trade["buy_cost"] += cost
                trade["buy_value"] += px * qty
                trade["buy_qty"] += qty
                fills.append({"Date": idx[i], "side": "買進", "qty": qty, "price": px,
                              "reason": f"加碼至{target*100:.0f}%" if held_target > 0 else f"進場{target*100:.0f}%",
                              "full": np.nan})
            held_target = target
        elif target == 0 and shares > 0:         # 出場
            px = O[i] * (1 - slip)
            net = px * shares * (1 - fee - tax)
            cash += net
            reason = "停損（均價−15%）" if cooldown[i - 1] else "訊號出場"
            trade["sell_net"] += net
            trade["sell_value"] += px * shares
            trade["sell_qty"] += shares
            trade["reason"] = reason
            fills.append({"Date": idx[i], "side": "賣出", "qty": shares, "price": px, "reason": reason, "full": True})
            shares = 0
            held_target = 0.0
            close_trade(i)
        elif target == 0:
            held_target = 0.0

        equity[i] = cash + shares * C[i]
        shares_arr[i] = shares

    if shares > 0 and last_i is not None:
        close_trade(last_i, is_open=True)

    mask = pd.Series(in_win, index=idx)
    mask.iloc[0] = False
    tr = pd.DataFrame(trades, columns=["entry_date", "exit_date", "bars", "entry_price", "exit_price",
                                       "pnl", "ret", "reason", "is_open"])
    tr["is_open"] = tr["is_open"].astype(bool)
    return {
        "equity": pd.Series(equity, index=idx)[mask],
        "shares": pd.Series(shares_arr, index=idx)[mask],
        "trades": tr,
        "fills": pd.DataFrame(fills, columns=["Date", "side", "qty", "price", "reason", "full"]),
        "positions": pd.Series(pos, index=idx),
    }
