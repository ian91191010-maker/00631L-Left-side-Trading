# -*- coding: utf-8 -*-
"""
00631L 交易訊號（Streamlit）

說明：
- 策略判斷邏輯與舊版完全相同（RSI / 布林 / BIAS / MA20 / MA60 / 階梯加碼 / -15% 停損 / 冷卻）。
- 已移除「資金回測」功能。
- 資料取得、還原、錯誤處理與介面皆已整理。
"""
import inspect
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import ta
from plotly.subplots import make_subplots

# ==========================================
# 基本設定
# ==========================================
TICKER = "00631L"
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TW_TZ = timezone(timedelta(hours=8))  # 台灣時間（無日光節約，固定 UTC+8）

# 已知需要還原的分割事件：{代號: [(生效日, 分割倍數), ...]}
# 00631L 於 2026-03-31 以 1 拆 22 恢復交易；FinMind 的原始日K不會自動還原。
SPLIT_EVENTS = {
    "00631L": [("2026-03-31", 22)],
}

# 槓桿 ETF 單日漲跌幅上限約 10%，超過 12% 幾乎一定是分割或資料異常
JUMP_WARN_LIMIT = 0.12

CHART_RANGES = {
    "近 3 個月": 63,
    "近 6 個月": 126,
    "近 1 年": 252,
    "近 2 年": 504,
    "近 5 年": 1260,
    "全部": 0,
}


def today_tw():
    return datetime.now(TW_TZ).date()


# ==========================================
# 資料取得模組
# （失敗時直接丟出錯誤：Streamlit 不會快取「失敗」的結果，下次會重新抓）
# ==========================================
def _finmind_request(dataset, data_id, start_date_obj, token):
    params = {
        "dataset": dataset,
        "data_id": data_id,
        "start_date": start_date_obj.strftime("%Y-%m-%d"),
        "end_date": today_tw().strftime("%Y-%m-%d"),
        "token": token,
    }
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(FINMIND_URL, params=params, headers=headers, timeout=20)
    res.raise_for_status()
    payload = res.json()
    if payload.get("status", 200) != 200:
        raise RuntimeError(f"FinMind 回應錯誤（{dataset} {data_id}）：{payload.get('msg', payload.get('status'))}")
    return pd.DataFrame(payload.get("data", []))


@st.cache_data(show_spinner=False, ttl=7200)
def fetch_stock_data(symbol, start_date_obj, token):
    df = _finmind_request("TaiwanStockPrice", symbol, start_date_obj, token)
    if df.empty:
        raise ValueError(f"FinMind 沒有回傳 {symbol} 的日K資料")

    df = df.rename(columns={"date": "Date", "open": "Open", "max": "High",
                            "min": "Low", "close": "Close", "Trading_Volume": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"])
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df.reindex(columns=["Date"] + cols)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df.drop_duplicates(subset="Date", keep="last").set_index("Date").sort_index()
    if df.empty:
        raise ValueError(f"{symbol} 沒有有效的收盤價資料")
    return df


@st.cache_data(show_spinner=False, ttl=7200)
def fetch_futures_data(start_date_obj, token):
    df = _finmind_request("TaiwanFuturesDaily", "TX", start_date_obj, token)
    if df.empty:
        raise ValueError("FinMind 沒有回傳台指期資料")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.sort_values(["date", "volume"], ascending=[True, False])
    df = df.drop_duplicates(subset=["date"], keep="first")

    df = df.rename(columns={"date": "Date", "close": "Futures_Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df["Futures_Close"] = pd.to_numeric(df["Futures_Close"], errors="coerce")
    return df[["Futures_Close"]]


def apply_split_adjustment(df, symbol):
    """依 SPLIT_EVENTS 還原分割前的價格與成交量。回傳 (資料, 說明清單)。"""
    notes = []
    df = df.copy()
    for date_str, factor in SPLIT_EVENTS.get(symbol, []):
        eff = pd.Timestamp(date_str)
        before = df[df.index < eff]
        after = df[df.index >= eff]
        if before.empty or after.empty:
            continue  # 資料區間沒有跨過這次分割
        ratio = before["Close"].iloc[-1] / after["Close"].iloc[0]
        if factor * 0.8 <= ratio <= factor * 1.2:
            mask = df.index < eff
            for c in ["Open", "High", "Low", "Close"]:
                df.loc[mask, c] = df.loc[mask, c] / factor
            df.loc[mask, "Volume"] = df.loc[mask, "Volume"] * factor
            notes.append(("info", f"已依 1 拆 {factor} 還原：{date_str} 之前的價格 ÷ {factor}"))
        # 若比例接近 1，代表資料來源已經還原過，不再重複處理
    return df, notes


# ==========================================
# 核心策略模組（邏輯與舊版相同）
# ==========================================
def categorize_basis(row):
    b = row["Smooth_Basis"]
    if pd.isna(b):
        return "數據不足"
    if row["Is_Dividend_Season"]:
        if b >= 20:
            return "極端正價差"
        elif b >= 0:
            return "微幅正價差"
        else:
            return "假性逆價差"
    else:
        if b >= 40:
            return "極端正價差"
        elif b >= 5:
            return "微幅正價差"
        elif b < 0:
            return "實質逆價差"
        else:
            return "平水雜訊"


def map_action(shift, position):
    # 舊版用「變動幅度 ≤ -99%」判斷全賣，導致 60%→0%、30%→0% 的完整出場被標成「部分」。
    # 這裡改看賣出後部位是否歸零；只影響標籤文字，不影響部位計算。
    if shift > 0:
        return f"BUY (+{shift*100:.0f}%)"
    elif shift < 0:
        return "SELL ALL" if position == 0 else "SELL (部分)"
    return ""


def compute_positions(df):
    """逐日計算目標部位。判斷規則與舊版逐行相同；另外記錄每日均價與冷卻狀態供介面顯示。"""
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


@st.cache_data(show_spinner=False, ttl=7200)
def get_strategy_results(ticker, data_start_date, token):
    """回傳 (策略結果 DataFrame, 提示訊息清單)。目標標的取不到資料時會丟出錯誤。"""
    notes = []

    df, split_notes = apply_split_adjustment(fetch_stock_data(ticker, data_start_date, token), ticker)
    notes += split_notes
    # 00631L 不配息，還原價 = 分割還原後的價格
    df["Adj_Open"] = df["Open"]
    df["Adj_Close"] = df["Close"]

    chg = df["Close"].pct_change().abs()
    for dt, v in chg[chg > JUMP_WARN_LIMIT].items():
        notes.append(("warn", f"{dt:%Y-%m-%d} 單日價格變動 {v*100:.1f}%，可能有尚未還原的分割或資料異常，"
                              f"請確認程式上方的 SPLIT_EVENTS 設定。"))

    # --- 技術指標 ---
    df["RSI"] = ta.momentum.rsi(df["Adj_Close"], window=14)
    bb = ta.volatility.BollingerBands(df["Adj_Close"], window=20, window_dev=2.2)
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Upper"] = bb.bollinger_hband()
    df["MA20"] = bb.bollinger_mavg()
    df["BIAS"] = (df["Adj_Close"] - df["MA20"]) / df["MA20"] * 100
    df["MA60"] = df["Adj_Close"].rolling(window=60).mean()

    # --- 期現貨價差（僅供參考，不參與進出場判斷；取不到資料時不影響策略）---
    try:
        taiex_close = fetch_stock_data("TAIEX", data_start_date, token)["Close"].reindex(df.index).ffill()
        fut_close = fetch_futures_data(data_start_date, token)["Futures_Close"].reindex(df.index).ffill()
        df["Basis"] = fut_close - taiex_close
    except Exception as e:
        df["Basis"] = np.nan
        notes.append(("warn", f"期現貨價差資料取得失敗（不影響進出場訊號）：{e}"))
    df["Smooth_Basis"] = df["Basis"].rolling(window=5).mean()
    df["Month"] = df.index.month
    df["Is_Dividend_Season"] = df["Month"].isin([6, 7, 8])
    df["Basis_State"] = df.apply(categorize_basis, axis=1)

    # --- 進出場 ---
    positions, avg_costs, cooldowns = compute_positions(df)
    df["Position"] = positions
    df["Avg_Cost"] = avg_costs
    df["Cooldown"] = cooldowns
    df["Position_Shift"] = df["Position"].diff().fillna(0)
    df["Action"] = [map_action(s, p) for s, p in zip(df["Position_Shift"], df["Position"])]
    return df, notes


# ==========================================
# 介面輔助函式
# ==========================================
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show_chart(fig):
    cfg = {"displaylogo": False, "scrollZoom": True}
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", config=cfg)
    else:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def fmt(v, spec=".2f", suffix=""):
    return "—" if pd.isna(v) else f"{v:{spec}}{suffix}"


def describe_state(latest):
    """依最新一天的資料，回傳 (主標題, 說明, 色碼)。"""
    pos_pct = latest["Position"] * 100
    action = latest["Action"]
    if action:
        if latest["Position_Shift"] > 0:
            return "買進", f"{action}，目標部位 {pos_pct:.0f}%", "#2E7D32"
        return "賣出", f"{action}，目標部位 {pos_pct:.0f}%", "#C62828"
    if latest["Position"] > 0:
        return "持有", f"目標部位 {pos_pct:.0f}%", "#2E7D32"
    if latest["Cooldown"]:
        return "空手", "停損後冷卻中，暫不進場", "#F9A825"
    return "空手", "等待進場條件", "#F9A825"


def build_chart(plot_df, show_ma, show_bb, show_rsi):
    rows = 2 if show_rsi else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.72, 0.28] if show_rsi else [1.0])

    idx = plot_df.index
    fig.add_trace(go.Candlestick(
        x=idx, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
        name="K線", increasing_line_color="#ef5350", decreasing_line_color="#26a69a"), row=1, col=1)

    if show_ma:
        fig.add_trace(go.Scatter(x=idx, y=plot_df["MA20"], name="MA20", mode="lines",
                                 line=dict(color="#F9A825", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=idx, y=plot_df["MA60"], name="MA60", mode="lines",
                                 line=dict(color="#7E57C2", width=1.2)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=idx, y=plot_df["BB_Lower"], name="布林下軌", mode="lines",
                                 line=dict(color="#90A4AE", width=1, dash="dot")), row=1, col=1)

    buys = plot_df[plot_df["Position_Shift"] > 0]
    sells = plot_df[plot_df["Position_Shift"] < 0]
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Low"] * 0.985, mode="markers", name="買進",
        marker=dict(symbol="triangle-up", size=11, color="#FFD700", line=dict(width=1, color="#555")),
        hovertext=[f"{a}　收盤 {c:.2f}" for a, c in zip(buys["Action"], buys["Close"])], hoverinfo="text"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["High"] * 1.015, mode="markers", name="賣出",
        marker=dict(symbol="triangle-down", size=11, color="#1E90FF", line=dict(width=1, color="#555")),
        hovertext=[f"{a}　收盤 {c:.2f}" for a, c in zip(sells["Action"], sells["Close"])], hoverinfo="text"),
        row=1, col=1)

    if show_rsi:
        fig.add_trace(go.Scatter(x=idx, y=plot_df["RSI"], name="RSI(14)", mode="lines",
                                 line=dict(color="#42A5F5", width=1.3)), row=2, col=1)
        for level, label in [(20, "20"), (32, "32"), (50, "50")]:
            fig.add_hline(y=level, line_dash="dot", line_width=0.8, line_color="#9E9E9E",
                          annotation_text=label, annotation_position="right", row=2, col=1)
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=2, col=1)

    # 持倉期間底色（部位越重顏色越深）。必須在圖層都建立之後才加入，否則子圖不會顯示。
    pos = plot_df["Position"].to_numpy()
    start = 0
    for i in range(1, len(pos) + 1):
        if i == len(pos) or pos[i] != pos[start]:
            if pos[start] > 0:
                x1 = idx[i] if i < len(pos) else idx[-1]
                alpha = 0.08 + 0.12 * pos[start]
                fig.add_vrect(x0=idx[start], x1=x1, fillcolor=f"rgba(255, 165, 0, {alpha:.3f})",
                              layer="below", line_width=0, row=1, col=1)
            start = i

    # 隱藏週末與休市日，避免K線出現空隙
    all_days = pd.date_range(idx.min(), idx.max(), freq="D")
    missing = [d.strftime("%Y-%m-%d") for d in all_days.difference(idx)]
    fig.update_xaxes(rangebreaks=[dict(values=missing)])
    fig.update_yaxes(title_text="價格 (NTD)", row=1, col=1)
    fig.update_layout(height=680 if show_rsi else 520, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10), dragmode="pan", hovermode="x",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    return fig


# ==========================================
# ===== UI =====
# ==========================================
st.set_page_config(page_title="00631L 交易訊號", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
.signal-card {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 12px 40px; padding: 18px 26px; margin: 4px 0 18px 0;
    border-radius: 10px; border-left: 8px solid var(--accent); background: var(--tint);
}
.signal-card .state {font-size: 2.6rem; font-weight: 700; line-height: 1.1;}
.signal-card .detail {font-size: 1.05rem; opacity: 0.85; margin-top: 4px;}
.signal-card .price {font-size: 2.6rem; font-weight: 700; text-align: right;
                     font-variant-numeric: tabular-nums; line-height: 1.1;}
.signal-card .asof {font-size: 0.95rem; opacity: 0.75; text-align: right; margin-top: 4px;}
@media (max-width: 640px) {
    .signal-card .state, .signal-card .price {font-size: 2rem;}
    .signal-card .price, .signal-card .asof {text-align: left;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 00631L 交易訊號")
st.caption("元大台灣50正2。訊號依每日收盤價計算，資料來自 FinMind。")

try:
    finmind_token = st.secrets["FINMIND_TOKEN"]
except Exception:
    finmind_token = ""

# ----- 側邊欄 -----
st.sidebar.subheader("資料")
data_start_date = st.sidebar.date_input(
    "資料起始日",
    datetime.strptime("2014-01-01", "%Y-%m-%d").date(),
    min_value=datetime.strptime("2010-01-01", "%Y-%m-%d").date(),
    help="策略的均價與冷卻狀態會從這天開始逐日累積，起始日不同可能讓今天的訊號不同。"
         "建議維持預設值（00631L 於 2014 年底上市）。",
)
if st.sidebar.button("🔄 重新取得最新資料", help="資料預設每 2 小時自動更新；按這裡可立即更新。"):
    st.cache_data.clear()

st.sidebar.subheader("圖表")
range_label = st.sidebar.select_slider("顯示區間", options=list(CHART_RANGES.keys()), value="近 1 年")
show_ma = st.sidebar.checkbox("均線 MA20 / MA60", value=True)
show_bb = st.sidebar.checkbox("布林通道下軌", value=False)
show_rsi = st.sidebar.checkbox("RSI 副圖", value=True)

# ----- 取得資料 -----
if not finmind_token:
    st.error("尚未設定 FinMind API Token。")
    st.markdown(
        "請到 Streamlit Community Cloud 的 **App 設定 → Secrets**，貼上這一行後儲存：\n\n"
        '`FINMIND_TOKEN = "你的token"`'
    )
    st.stop()

try:
    with st.spinner("正在取得資料並計算策略…"):
        result_df, notes = get_strategy_results(TICKER, data_start_date, finmind_token)
except Exception as e:
    st.error(f"資料取得失敗：{e}")
    st.caption("可能是網路、FinMind 服務或 Token 額度問題。請稍後按側邊欄的「重新取得最新資料」。")
    st.stop()

for level, msg in notes:
    if level == "warn":
        st.warning(msg)

# ----- 最新訊號 -----
latest = result_df.iloc[-1]
prev_close = result_df["Close"].iloc[-2] if len(result_df) > 1 else np.nan
title, detail, accent = describe_state(latest)
r, g, b = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)

st.markdown(f"""
<div class="signal-card" style="--accent: {accent}; --tint: rgba({r}, {g}, {b}, 0.16);">
  <div>
    <div class="state">{title}</div>
    <div class="detail">{detail}</div>
  </div>
  <div>
    <div class="price">{latest['Close']:.2f}</div>
    <div class="asof">{latest.name:%Y-%m-%d} 收盤，訊號於下一個交易日執行</div>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
day_chg = (latest["Close"] / prev_close - 1) * 100 if pd.notna(prev_close) else np.nan
c1.metric("收盤價", f"{latest['Close']:.2f}", None if pd.isna(day_chg) else f"{day_chg:+.2f}%",
          delta_color="inverse")  # 台股慣例：漲紅跌綠
c2.metric("RSI(14)", fmt(latest["RSI"], ".1f"))
c3.metric("乖離率 BIAS", fmt(latest["BIAS"], "+.2f", "%"), help="(收盤 − MA20) / MA20")
if pd.isna(latest["MA60"]):
    zone = "資料不足"
else:
    zone = "多頭區" if latest["Adj_Close"] > latest["MA60"] else "空頭區"
c4.metric("趨勢區（MA60）", zone, help="收盤 > MA60 為多頭區，否則為空頭區")
if latest["Position"] > 0 and latest["Avg_Cost"] > 0:
    stop_price = latest["Avg_Cost"] * 0.85
    c5.metric("停損價", f"{stop_price:.2f}",
              f"收盤高於 {(latest['Close'] / stop_price - 1) * 100:.1f}%", delta_color="off",
              help="加權均價 × 0.85，收盤跌破即全數出場並進入冷卻")
else:
    c5.metric("停損價", "—", help="空手時沒有停損價")

# ----- 圖表 -----
days = CHART_RANGES[range_label]
plot_df = result_df.tail(days) if days > 0 else result_df
show_chart(build_chart(plot_df, show_ma, show_bb, show_rsi))
for level, msg in notes:
    if level == "info":
        st.caption(msg)

# ----- 下方分頁 -----
tab_log, tab_rules, tab_basis = st.tabs(["近期交易紀錄", "策略規則", "期現貨價差（參考）"])

with tab_log:
    trades = result_df[result_df["Position_Shift"] != 0].tail(10).iloc[::-1]
    if trades.empty:
        st.info("目前沒有任何進出場紀錄。")
    else:
        log = pd.DataFrame({
            "日期": trades.index.strftime("%Y-%m-%d"),
            "方向": np.where(trades["Position_Shift"] > 0, "買進", "賣出"),
            "動作": trades["Action"].values,
            "收盤價": trades["Close"].round(2).values,
            "目標部位": (trades["Position"] * 100).round(0).astype(int).astype(str).values + "%",
        })
        st.dataframe(log, hide_index=True)
        st.caption("最近 10 筆，最新在最上方。價格為分割還原後的收盤價。")

with tab_rules:
    st.markdown("""
**多頭區（收盤 > MA60）**
- 收盤 > MA20：目標部位 100%
- 收盤 ≤ MA20：全數出場

**空頭區（收盤 ≤ MA60）**
- 出場：收盤 ≥ MA20，或 RSI > 50
- 進場（非冷卻期間，只加碼不減碼）：

| 條件 | 目標部位 |
|---|---|
| RSI < 20 或 BIAS < −12% | 100% |
| RSI < 25 或 收盤跌破布林下軌（20 日、2.2 倍標準差） | 60% |
| RSI < 32 或 BIAS < −6% | 30% |

**停損與冷卻**
- 持倉中，收盤 ≤ 加權均價 × 0.85：全數出場並進入冷卻
- 冷卻期間不進場；空手且（RSI > 50 或收盤 > MA20）時解除
""")

with tab_basis:
    st.caption("僅供參考，不參與進出場判斷。")
    if result_df["Smooth_Basis"].notna().any():
        b1, b2 = st.columns(2)
        b1.metric("5 日平滑價差（點）", fmt(latest["Smooth_Basis"], ".1f"))
        b2.metric("價差狀態", latest["Basis_State"])
        st.line_chart(plot_df["Smooth_Basis"].dropna())
    else:
        st.info("目前沒有期現貨價差資料。")
