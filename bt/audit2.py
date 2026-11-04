# -*- coding: utf-8 -*-
"""審計系統二：TSI 進場、SMA 死亡交叉出場（照字面規則）。

- 空手時，收盤 TSI 藍線 > 信號線 → 進場（部位 100%）
- 持有時，收盤 SMA5 由上往下跌破 SMA55 → 全部出場
- 進場當天不檢查出場；沒有停損
- 訊號在收盤判斷，下一個交易日執行

注意（照字面規則的特性）：
- 出場是「跌破」事件。若進場時 SMA5 已在 SMA55 之下，要等 SMA5 先回到上方再跌破才會出場。
- 出場當天若 TSI 藍線仍在信號線上方，隔天收盤會再次進場。
"""
import numpy as np
import pandas as pd

from . import indicators as ti

TSI_R, TSI_S, TSI_SIG = 25, 13, 13
SMA_FAST, SMA_SLOW = 5, 55


def compute_signals(daily):
    """回傳含指標、目標部位、動作的日K表。"""
    df = daily[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["TSI"], df["TSI_SIG"] = ti.tsi(df["Close"], TSI_R, TSI_S, TSI_SIG)
    df["SMA_F"] = ti.sma(df["Close"], SMA_FAST)
    df["SMA_S"] = ti.sma(df["Close"], SMA_SLOW)

    entry = (df["TSI"] > df["TSI_SIG"]).fillna(False).to_numpy()
    exit_ = ti.cross_under(df["SMA_F"], df["SMA_S"]).fillna(False).to_numpy()

    pos = np.zeros(len(df))
    current = 0.0
    for i in range(len(df)):
        if current == 0 and entry[i]:
            current = 1.0
        elif current == 1 and exit_[i]:
            current = 0.0
        pos[i] = current

    df["Position"] = pos
    df["Position_Shift"] = df["Position"].diff().fillna(df["Position"])
    df["Action"] = np.where(df["Position_Shift"] > 0, "BUY (+100%)",
                            np.where(df["Position_Shift"] < 0, "SELL ALL", ""))
    df["Chg_Pct"] = df["Close"].pct_change() * 100
    return df


def build_trades(df):
    """整理成完整進出場循環。漲跌幅以訊號日收盤價計算，未計手續費與交易稅。"""
    pos = df["Position"].to_numpy()
    close = df["Close"].to_numpy()
    idx = df.index
    rows = []
    entry_i = 0 if pos[0] > 0 else None
    for i in range(1, len(df)):
        if pos[i - 1] == 0 and pos[i] > 0:
            entry_i = i
        elif pos[i - 1] > 0 and pos[i] == 0 and entry_i is not None:
            rows.append({"entry_date": idx[entry_i], "exit_date": idx[i], "days": i - entry_i,
                         "entry_price": close[entry_i], "exit_price": close[i],
                         "ret": (close[i] / close[entry_i] - 1) * 100, "is_open": False})
            entry_i = None
    if entry_i is not None:
        rows.append({"entry_date": idx[entry_i], "exit_date": pd.NaT, "days": len(df) - 1 - entry_i,
                     "entry_price": close[entry_i], "exit_price": close[-1],
                     "ret": (close[-1] / close[entry_i] - 1) * 100, "is_open": True})
    out = pd.DataFrame(rows, columns=["entry_date", "exit_date", "days", "entry_price", "exit_price",
                                      "ret", "is_open"])
    out["is_open"] = out["is_open"].astype(bool)
    return out
