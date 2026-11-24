# -*- coding: utf-8 -*-
"""轉折 F0：用台指期的「當月轉折」當 00631L 的進出場指標（新增頁面用；不影響原本三個頁面）。

規則（與研究回測 dna-audit futures/etf_apply.py 的 F0 相同）
- 訊號序列＝台指期近月連續合約（一般時段、不調整價差）。
    收盤＝逐日結算價（沒有就用收盤，並夾在當天最高最低之間）；最高、最低＝盤中。
- 當月轉折 N ＝ 前 2 個「完整月」的（最高 ＋ 最低）÷ 2，四捨五入到整數（0.5 進位）。
- 收盤（四捨五入到小數 2 位）> N → 持有 00631L；< N → 空手；等於 N → 維持前一天。
- 台指期收盤判定 → 下一個交易日 00631L 開盤成交。只做多，空手拿現金。
- 近月：到期日 > 當天的最小月份契約。到期日＝資料裡該契約最後出現的交易日；
  還在交易的契約用交割月第三個星期三（遇假日順延的情況無法事先得知，最多差 1 天）。

資料：FinMind TaiwanFuturesDaily（TX）。
"""
import numpy as np
import pandas as pd
import requests
import streamlit as st

from . import data as D

VERSION = "1"
TX_START = "2014-06-01"      # 00631L 2014-11 上市；N 需要前 2 個完整月，所以往前多抓
BUY_FEE = 0.001425 * 0.3     # 手續費 3 折
SELL_FEE = 0.001425 * 0.3 + 0.001   # ＋ETF 證交稅 0.1%


# ----------------------------------------------------------------------
# 資料
# ----------------------------------------------------------------------
def normalize_tx(rows):
    """FinMind TaiwanFuturesDaily 原始列 → date, contract, session, Open, High, Low, Close, Volume, Settle"""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["contract_date"].astype(str).str.fullmatch(r"\d{6}")]     # 排除價差單（例如 202612/202701）
    df = df.rename(columns={"contract_date": "contract", "trading_session": "session", "open": "Open",
                            "max": "High", "min": "Low", "close": "Close", "volume": "Volume",
                            "settlement_price": "Settle"})
    df["date"] = pd.to_datetime(df["date"])
    df["contract"] = df["contract"].astype(str)
    for c in ["Open", "High", "Low", "Close", "Settle"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df = df[["date", "contract", "session", "Open", "High", "Low", "Close", "Volume", "Settle"]]
    return df.drop_duplicates(["date", "contract", "session"], keep="last").sort_values(["date", "contract"])


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def fetch_tx(token):
    """抓台指期日資料（依年份分段，避免單次要求過長）。失敗會丟出錯誤，不會被快取。"""
    rows = []
    s = pd.Timestamp(TX_START)
    today = pd.Timestamp(D.today_tw())
    while s <= today:
        e = min(pd.Timestamp(year=s.year, month=12, day=31), today)
        res = requests.get(D.FINMIND_URL, headers={"Authorization": f"Bearer {token}"}, timeout=60,
                           params={"dataset": "TaiwanFuturesDaily", "data_id": "TX",
                                   "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d")})
        res.raise_for_status()
        payload = res.json()
        if payload.get("status", 200) != 200:
            raise RuntimeError(f"FinMind 回應錯誤（台指期 {s:%Y}）：{payload.get('msg', payload.get('status'))}")
        rows += payload.get("data", [])
        s = e + pd.Timedelta(days=1)
    tx = normalize_tx(rows)
    if tx.empty:
        raise ValueError("FinMind 沒有回傳台指期資料")
    return tx


# ----------------------------------------------------------------------
# 訊號
# ----------------------------------------------------------------------
def round_half_up(x, decimals=0):
    v = pd.Series(x, dtype="float64")
    f = 10.0 ** decimals
    return np.sign(v) * np.floor(v.abs() * f + 0.5) / f


def third_wednesday(y, m):
    d = pd.Timestamp(year=y, month=m, day=1)
    d += pd.Timedelta(days=(2 - d.weekday()) % 7)
    return d + pd.Timedelta(days=14)


def front_series(tx):
    """近月連續合約日 K（一般時段、不調整）。回傳含 Open/High/Low/Close（結算價）/Contract 的表。"""
    pos = tx[tx["session"] == "position"].copy()
    pos = pos[(pos["Close"] > 0) | (pos["Settle"] > 0)]
    pos.loc[~(pos["Close"] > 0), "Close"] = pos["Settle"]
    for c in ["Open", "High", "Low"]:
        bad = ~(pos[c] > 0)
        pos.loc[bad, c] = pos.loc[bad, "Close"]
    exp = pos.groupby("contract")["date"].max().sort_index()
    last = pos["date"].max()
    for c in exp.index[exp == last]:                       # 還在交易的契約
        exp[c] = third_wednesday(int(c[:4]), int(c[4:]))
    cs, ex = exp.index.to_numpy(), exp.to_numpy()
    dates = pd.DatetimeIndex(sorted(pos["date"].unique()))
    front = []
    for d in dates:
        ok = np.nonzero(ex > np.datetime64(d))[0]
        front.append(cs[ok[0]] if len(ok) else None)
    front = pd.Series(front, index=dates, dtype=object)
    front = front[front.notna()]
    bars = pos.set_index(["date", "contract"])
    rows = bars.reindex(list(zip(front.index, front.to_numpy())))
    out = pd.DataFrame({"Open": rows["Open"].to_numpy(), "High": rows["High"].to_numpy(),
                        "Low": rows["Low"].to_numpy(), "Close_raw": rows["Close"].to_numpy(),
                        "Settle": rows["Settle"].to_numpy(), "Contract": front.to_numpy()}, index=front.index)
    st_ = np.where(np.isnan(out["Settle"]) | (out["Settle"] <= 0), out["Close_raw"], out["Settle"])
    out["Close"] = pd.Series(st_, index=out.index).clip(lower=out["Low"], upper=out["High"])
    return out.dropna(subset=["Close", "High", "Low"])


def turning_month(daily):
    """每天的當月轉折：前 2 個完整月最高＋最低 ÷ 2"""
    per = daily.index.to_period("M")
    mh = daily["High"].groupby(per).max()
    ml = daily["Low"].groupby(per).min()
    n = (pd.concat([mh.shift(1), mh.shift(2)], axis=1).max(axis=1, skipna=False)
         + pd.concat([ml.shift(1), ml.shift(2)], axis=1).min(axis=1, skipna=False)) / 2
    return pd.Series(per.map(n), index=daily.index, dtype=float)


def compute_f0(tx):
    """每個台指期交易日收盤後的 F0 狀態。回傳含 Close、N、Gap、Target（1＝持有、0＝空手）。"""
    d = front_series(tx)
    out = pd.DataFrame(index=d.index)
    out["Close"] = d["Close"]
    out["N"] = round_half_up(turning_month(d), 0)
    c = round_half_up(d["Close"], 2).to_numpy()
    n = out["N"].to_numpy()
    tgt, cur = np.zeros(len(d)), 0.0
    for i in range(len(d)):
        if np.isnan(n[i]):
            cur = 0.0
        elif c[i] > n[i]:
            cur = 1.0
        elif c[i] < n[i]:
            cur = 0.0
        tgt[i] = cur
    out["Target"] = tgt
    out["Gap"] = (out["Close"] / out["N"] - 1) * 100
    out["Contract"] = d["Contract"]
    return out


# ----------------------------------------------------------------------
# 套用到 00631L
# ----------------------------------------------------------------------
def align(sig, price):
    """00631L 每個交易日的持有狀態＝前一個台指期交易日收盤後的 F0 狀態（隔日開盤成交）"""
    w = sig["Target"].reindex(price.index.union(sig.index)).ffill()
    return w.shift(1).reindex(price.index).fillna(0.0)


def simulate(price, hold):
    """全額投入、隔日開盤成交。回傳 (每日報酬, 交易表)。price 需有 Open、Close。"""
    o, c = price["Open"].to_numpy(float), price["Close"].to_numpy(float)
    h = hold.to_numpy(float)
    r = np.zeros(len(price))
    trades, cur, eq = [], None, 1.0
    for i in range(len(price)):
        prev = h[i - 1] if i else 0.0
        if prev == 0 and h[i] == 1:
            r[i] = c[i] / o[i] * (1 - BUY_FEE) - 1
            cur = {"entry_date": price.index[i], "entry_price": o[i], "v0": eq}
        elif prev == 1 and h[i] == 0:
            r[i] = o[i] / c[i - 1] * (1 - SELL_FEE) - 1
        elif h[i] == 1:
            r[i] = c[i] / c[i - 1] - 1
        eq *= 1 + r[i]
        if prev == 1 and h[i] == 0 and cur is not None:
            trades.append({"entry_date": cur["entry_date"], "exit_date": price.index[i],
                           "entry_price": cur["entry_price"], "exit_price": o[i],
                           "ret": (eq / cur["v0"] - 1) * 100, "is_open": False})
            cur = None
    if cur is not None:
        trades.append({"entry_date": cur["entry_date"], "exit_date": pd.NaT, "entry_price": cur["entry_price"],
                       "exit_price": c[-1], "ret": (eq / cur["v0"] - 1) * 100, "is_open": True})
    tr = pd.DataFrame(trades, columns=["entry_date", "exit_date", "entry_price", "exit_price", "ret", "is_open"])
    tr["is_open"] = tr["is_open"].astype(bool)
    return pd.Series(r, index=price.index), tr


def buy_hold(price):
    r = price["Close"].pct_change()
    r.iloc[0] = price["Close"].iat[0] / price["Open"].iat[0] * (1 - BUY_FEE) - 1
    return r


def stats(r, hold=None):
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iat[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    return {"總報酬": (eq.iat[-1] - 1) * 100, "年化報酬": cagr * 100, "最大回撤": mdd * 100,
            "報酬回撤比": cagr / abs(mdd) if mdd < 0 else np.nan,
            "持有比例": np.nan if hold is None else float(np.mean(hold)) * 100}
