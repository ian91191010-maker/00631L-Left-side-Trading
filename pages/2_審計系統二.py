# -*- coding: utf-8 -*-
"""審計系統二頁面：TSI 藍線 > 信號線進場、SMA5 跌破 SMA55 出場。

判斷邏輯在 bt/audit2.py；這個檔案只負責介面。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bt import audit2 as A2          # noqa: E402
from bt import data as D             # noqa: E402

TICKER = "00631L"
CHART_RANGES = {"近 3 個月": 63, "近 6 個月": 126, "近 1 年": 252, "近 2 年": 504, "近 5 年": 1260, "全部": 0}
PROFIT_FILL = "rgba(239, 83, 80, 0.14)"   # 獲利：淺紅
LOSS_FILL = "rgba(38, 166, 154, 0.16)"    # 虧損：淺綠

st.set_page_config(page_title="00631L 審計系統二", page_icon="📊", layout="wide",
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


@st.cache_data(show_spinner=False, ttl=7200)
def get_results(token):
    daily, notes = D.load_price(TICKER, token)
    return A2.compute_signals(daily), notes


def fmt(v, spec=".2f", suffix=""):
    return "—" if pd.isna(v) else f"{v:{spec}}{suffix}"


def describe_state(latest):
    if latest["Action"]:
        if latest["Position_Shift"] > 0:
            return "買進", "TSI 藍線 > 信號線，目標部位 100%", "#2E7D32"
        return "賣出", "SMA5 跌破 SMA55，全部出場", "#C62828"
    if latest["Position"] > 0:
        return "持有", "目標部位 100%，等待 SMA5 跌破 SMA55", "#2E7D32"
    return "空手", "等待 TSI 藍線 > 信號線", "#F9A825"


def build_chart(plot_df, trades, show_tsi):
    rows = 2 if show_tsi else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.72, 0.28] if show_tsi else [1.0])
    idx = plot_df.index

    hover = [f"開 {o:.2f}<br>高 {h:.2f}<br>低 {l:.2f}<br>收 {c:.2f}" + ("" if pd.isna(g) else f"（{g:+.2f}%）")
             for o, h, l, c, g in zip(plot_df["Open"], plot_df["High"], plot_df["Low"],
                                      plot_df["Close"], plot_df["Chg_Pct"])]
    fig.add_trace(go.Candlestick(
        x=idx, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
        name="K線", whiskerwidth=0, hovertext=hover, hoverinfo="text",
        increasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
        decreasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a")), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=plot_df["SMA_F"], name=f"SMA{A2.SMA_FAST}", mode="lines",
                             hoverinfo="skip", line=dict(color="#F9A825", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=plot_df["SMA_S"], name=f"SMA{A2.SMA_SLOW}", mode="lines",
                             hoverinfo="skip", line=dict(color="#7E57C2", width=1.2)), row=1, col=1)

    buys = plot_df[plot_df["Position_Shift"] > 0]
    sells = plot_df[plot_df["Position_Shift"] < 0]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Low"] * 0.985, mode="markers", name="買進", hoverinfo="skip",
                             marker=dict(symbol="triangle-up", size=10, color="#FFD700",
                                         line=dict(width=1, color="#555"))), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["High"] * 1.015, mode="markers", name="賣出",
                             hoverinfo="skip",
                             marker=dict(symbol="triangle-down", size=10, color="#1E90FF",
                                         line=dict(width=1, color="#555"))), row=1, col=1)

    if show_tsi:
        fig.add_trace(go.Scatter(x=idx, y=plot_df["TSI"], name="TSI", mode="lines", hoverinfo="skip",
                                 line=dict(color="#2962FF", width=1.3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=idx, y=plot_df["TSI_SIG"], name="信號線", mode="lines", hoverinfo="skip",
                                 line=dict(color="#FF5252", width=1.1)), row=2, col=1)
        fig.add_hline(y=0, line_width=0.8, line_color="#9E9E9E", row=2, col=1)
        fig.update_yaxes(title_text="TSI", row=2, col=1)

    # 持倉底色：獲利淺紅、虧損淺綠（必須在所有圖層建立後才加）
    lo, hi = idx[0], idx[-1]
    for t in trades.itertuples():
        x0 = t.entry_date
        x1 = hi if pd.isna(t.exit_date) else t.exit_date
        if x1 < lo or x0 > hi:
            continue
        fig.add_vrect(x0=max(x0, lo), x1=min(x1, hi), fillcolor=PROFIT_FILL if t.ret > 0 else LOSS_FILL,
                      layer="below", line_width=0, row=1, col=1)

    all_days = pd.date_range(lo, hi, freq="D")
    missing = [d.strftime("%Y-%m-%d") for d in all_days.difference(idx)]
    fig.update_xaxes(rangebreaks=[dict(values=missing)], showspikes=True, spikemode="across",
                     spikesnap="cursor", spikethickness=1, spikedash="dot", spikecolor="#888")
    fig.update_yaxes(showspikes=True, spikethickness=1, spikedash="dot", spikecolor="#888", row=1, col=1)
    fig.update_yaxes(title_text="價格 (NTD)", row=1, col=1)
    fig.update_layout(height=680 if show_tsi else 520, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10), dragmode="pan", hovermode="x",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    return fig


# ===== 介面 =====
st.markdown("## 00631L 審計系統二")
st.caption(f"TSI({A2.TSI_R},{A2.TSI_S},{A2.TSI_SIG}) 藍線 > 信號線進場，"
           f"SMA{A2.SMA_FAST} 跌破 SMA{A2.SMA_SLOW} 出場。訊號依每日收盤價計算，資料來自 FinMind。")

try:
    token = st.secrets["FINMIND_TOKEN"]
except Exception:
    token = ""

st.sidebar.subheader("資料")
if st.sidebar.button("🔄 重新取得最新資料", help="資料預設每 2 小時自動更新；按這裡可立即更新。"):
    st.cache_data.clear()
st.sidebar.subheader("圖表")
range_label = st.sidebar.select_slider("顯示區間", options=list(CHART_RANGES), value="近 3 個月")
show_tsi = st.sidebar.checkbox("TSI 副圖", value=True)

if not token:
    st.error("尚未設定 FinMind API Token。請到 App 設定 → Secrets 貼上：`FINMIND_TOKEN = \"你的token\"`")
    st.stop()

try:
    with st.spinner("正在取得資料並計算訊號…"):
        df, notes = get_results(token)
except Exception as e:
    st.error(f"資料取得失敗：{e}")
    st.caption("可能是網路、FinMind 服務或 Token 額度問題。請稍後按側邊欄的「重新取得最新資料」。")
    st.stop()

for level, msg in notes:
    if level == "warn":
        st.warning(msg)

latest = df.iloc[-1]
prev_close = df["Close"].iloc[-2] if len(df) > 1 else np.nan
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

# 照字面規則的提醒：持有中但 SMA5 已在 SMA55 之下，必須先回到上方再跌破才會出場
if latest["Position"] > 0 and latest["SMA_F"] < latest["SMA_S"]:
    st.info(f"目前持有中，但 SMA{A2.SMA_FAST} 已在 SMA{A2.SMA_SLOW} 之下。"
            f"依規則要等 SMA{A2.SMA_FAST} 先回到上方、再次跌破才會出場。")

c1, c2, c3, c4 = st.columns(4)
day_chg = (latest["Close"] / prev_close - 1) * 100 if pd.notna(prev_close) else np.nan
c1.metric("收盤價", f"{latest['Close']:.2f}", None if pd.isna(day_chg) else f"{day_chg:+.2f}%",
          delta_color="inverse")
c2.metric("TSI", fmt(latest["TSI"], ".2f"))
c3.metric("信號線", fmt(latest["TSI_SIG"], ".2f"),
          None if pd.isna(latest["TSI"]) else ("藍線在上" if latest["TSI"] > latest["TSI_SIG"] else "藍線在下"),
          delta_color="off")
if pd.isna(latest["SMA_S"]):
    c4.metric(f"SMA{A2.SMA_FAST} / SMA{A2.SMA_SLOW}", "資料不足")
else:
    gap = (latest["SMA_F"] / latest["SMA_S"] - 1) * 100
    c4.metric(f"SMA{A2.SMA_FAST} vs SMA{A2.SMA_SLOW}", "在上方" if gap > 0 else "在下方",
              f"差距 {gap:+.2f}%", delta_color="off")

trades = A2.build_trades(df)
days = CHART_RANGES[range_label]
plot_df = df.tail(days) if days > 0 else df
st.plotly_chart(build_chart(plot_df, trades, show_tsi), width="stretch",
                config={"displaylogo": False, "scrollZoom": True})
st.caption("底色：淺紅＝該次交易獲利，淺綠＝虧損（持有中的交易以最新收盤價計算）。")
for level, msg in notes:
    if level == "info":
        st.caption(msg)

tab_log, tab_rules = st.tabs(["近期交易紀錄", "策略規則"])
with tab_log:
    if trades.empty:
        st.info("目前沒有任何進出場紀錄。")
    else:
        recent = trades.tail(10).iloc[::-1]
        log = pd.DataFrame({
            "進場日": recent["entry_date"].dt.strftime("%Y-%m-%d").values,
            "出場日": recent["exit_date"].dt.strftime("%Y-%m-%d").fillna("持有中").values,
            "持有交易日": recent["days"].astype(int).values,
            "進場價": recent["entry_price"].values,
            "出場價": recent["exit_price"].values,
            "漲跌幅": recent["ret"].values,
            "結果": np.where(recent["is_open"], "持有中", np.where(recent["ret"] > 0, "獲利", "虧損")),
        })

        def row_style(row):
            css = ("color:#ef5350;font-weight:600;background-color:rgba(239,83,80,0.14)" if row["漲跌幅"] > 0 else
                   "color:#26a69a;font-weight:600;background-color:rgba(38,166,154,0.16)")
            return [css if c in ("漲跌幅", "結果") else "" for c in row.index]

        st.dataframe(log.style.format({"進場價": "{:.2f}", "出場價": "{:.2f}", "漲跌幅": "{:+.2f}%"})
                     .apply(row_style, axis=1), hide_index=True)
        st.caption("最近 10 次，最新在最上方。漲跌幅以訊號日收盤價計算，未計手續費與交易稅；持有中的交易為未實現漲跌幅。")

with tab_rules:
    st.markdown(f"""
**進場**：空手時，收盤 TSI({A2.TSI_R},{A2.TSI_S},{A2.TSI_SIG}) 藍線 > 信號線 → 目標部位 100%

**出場**：持有時，收盤 SMA{A2.SMA_FAST} 由上往下跌破 SMA{A2.SMA_SLOW} → 全部出場

**其他**
- 沒有停損；進場當天不檢查出場。
- 訊號依收盤判斷，下一個交易日執行。
- TSI = 100 × EMA(EMA(動量, {A2.TSI_R}), {A2.TSI_S}) ÷ EMA(EMA(|動量|, {A2.TSI_R}), {A2.TSI_S})，動量 = 收盤 − 前一日收盤；信號線 = EMA(TSI, {A2.TSI_SIG})。
- 規則照字面執行：進場時若 SMA{A2.SMA_FAST} 已在 SMA{A2.SMA_SLOW} 之下，要等它先回到上方再跌破才會出場；出場隔天若 TSI 藍線仍在上方會再次進場。
- 回測頁有「審計系統二」預設組合，可一鍵載入相同規則測試績效。
""")
