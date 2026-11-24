# -*- coding: utf-8 -*-
"""轉折 F0 頁面：台指期當月轉折 → 00631L 進出場（只做多）。

判斷邏輯在 bt/f0.py；這個檔案只負責介面。原本的主頁、策略回測、審計系統二都沒有改動。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bt import data as D          # noqa: E402
from bt import f0 as F            # noqa: E402

TICKER = "00631L"
CHART_RANGES = {"近 3 個月": 63, "近 6 個月": 126, "近 1 年": 252, "近 2 年": 504, "近 5 年": 1260, "全部": 0}
PROFIT_FILL = "rgba(239, 83, 80, 0.14)"   # 獲利：淺紅
LOSS_FILL = "rgba(38, 166, 154, 0.16)"    # 虧損：淺綠
SPLIT_DATE = "2020-10-30"                 # 績效對照的前後段分界

st.set_page_config(page_title="00631L 轉折 F0", page_icon="📐", layout="wide", initial_sidebar_state="expanded")
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


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def get_results(token):
    price, notes = D.load_price(TICKER, token)
    sig = F.compute_f0(F.fetch_tx(token))
    return price, sig, notes


def describe_state(sig):
    now, prev = sig["Target"].iat[-1], sig["Target"].iat[-2] if len(sig) > 1 else 0.0
    if now > prev:
        return "買進", "台指期收盤站上當月轉折，下一個交易日開盤買進 00631L", "#2E7D32"
    if now < prev:
        return "賣出", "台指期收盤跌破當月轉折，下一個交易日開盤賣出 00631L", "#C62828"
    if now > 0:
        return "持有", "台指期收盤在當月轉折之上，繼續持有", "#2E7D32"
    return "空手", "台指期收盤在當月轉折之下，等待站上轉折", "#F9A825"


def build_chart(p, tx, trades):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.62, 0.38])
    idx = p.index
    hover = [f"開 {o:.2f}<br>高 {h:.2f}<br>低 {l:.2f}<br>收 {c:.2f}"
             for o, h, l, c in zip(p["Open"], p["High"], p["Low"], p["Close"])]
    fig.add_trace(go.Candlestick(
        x=idx, open=p["Open"], high=p["High"], low=p["Low"], close=p["Close"], name="00631L",
        whiskerwidth=0, hovertext=hover, hoverinfo="text",
        increasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
        decreasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a")), row=1, col=1)
    buys = p[p["Entry"]]
    sells = p[p["Exit"]]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Low"] * 0.985, mode="markers", name="買進", hoverinfo="skip",
                             marker=dict(symbol="triangle-up", size=10, color="#FFD700",
                                         line=dict(width=1, color="#555"))), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["High"] * 1.015, mode="markers", name="賣出", hoverinfo="skip",
                             marker=dict(symbol="triangle-down", size=10, color="#1E90FF",
                                         line=dict(width=1, color="#555"))), row=1, col=1)
    t = tx.reindex(idx)
    fig.add_trace(go.Scatter(x=idx, y=t["Close"], name="台指期收盤（結算價）", mode="lines", hoverinfo="skip",
                             line=dict(color="#2962FF", width=1.3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=t["N"], name="當月轉折", mode="lines", hoverinfo="skip",
                             line=dict(color="#FF5252", width=1.3, shape="hv")), row=2, col=1)
    fig.update_yaxes(title_text="台指期", row=2, col=1)

    lo, hi = idx[0], idx[-1]
    for tr in trades.itertuples():
        x1 = hi if pd.isna(tr.exit_date) else tr.exit_date
        if x1 < lo or tr.entry_date > hi:
            continue
        fig.add_vrect(x0=max(tr.entry_date, lo), x1=min(x1, hi), fillcolor=PROFIT_FILL if tr.ret > 0 else LOSS_FILL,
                      layer="below", line_width=0, row=1, col=1)
    missing = [d.strftime("%Y-%m-%d") for d in pd.date_range(lo, hi, freq="D").difference(idx)]
    fig.update_xaxes(rangebreaks=[dict(values=missing)], showspikes=True, spikemode="across",
                     spikesnap="cursor", spikethickness=1, spikedash="dot", spikecolor="#888")
    fig.update_yaxes(title_text="價格 (NTD)", row=1, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      dragmode="pan", hovermode="x",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    return fig


# ===== 介面 =====
st.markdown("## 00631L 轉折 F0")
st.caption("台指期收盤（逐日結算價）站上「當月轉折」→ 持有 00631L，跌破 → 空手。只做多、全額進出，"
           "收盤判定、下一個交易日開盤成交。資料來自 FinMind。")

try:
    token = st.secrets["FINMIND_TOKEN"]
except Exception:
    token = ""

st.sidebar.subheader("資料")
if st.sidebar.button("🔄 重新取得最新資料", help="資料預設每 3 小時自動更新；按這裡可立即更新。"):
    st.cache_data.clear()
st.sidebar.subheader("圖表")
range_label = st.sidebar.select_slider("顯示區間", options=list(CHART_RANGES), value="近 1 年")

if not token:
    st.error("尚未設定 FinMind API Token。請到 App 設定 → Secrets 貼上：`FINMIND_TOKEN = \"你的token\"`")
    st.stop()

try:
    with st.spinner("正在取得台指期與 00631L 資料並計算訊號…"):
        price, sig, notes = get_results(token)
except Exception as e:
    st.error(f"資料取得失敗：{e}")
    st.caption("可能是網路、FinMind 服務或 Token 額度問題。請稍後按側邊欄的「重新取得最新資料」。")
    st.stop()

for level, msg in notes:
    if level == "warn":
        st.warning(msg)

sig = sig.dropna(subset=["N"])
start = max(price.index[0], sig.index[0])
p = price.loc[start:].copy()
hold = F.align(sig, p)
ret, trades = F.simulate(p, hold)
p["Entry"] = (hold.diff().fillna(hold) > 0)
p["Exit"] = (hold.diff().fillna(0) < 0)

title, detail, accent = describe_state(sig)
r, g, b = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
last_tx = sig.index[-1]
st.markdown(f"""
<div class="signal-card" style="--accent: {accent}; --tint: rgba({r}, {g}, {b}, 0.16);">
  <div>
    <div class="state">{title}</div>
    <div class="detail">{detail}</div>
  </div>
  <div>
    <div class="price">{p['Close'].iat[-1]:.2f}</div>
    <div class="asof">00631L {p.index[-1]:%Y-%m-%d} 收盤｜台指期訊號日 {last_tx:%Y-%m-%d}</div>
  </div>
</div>
""", unsafe_allow_html=True)
if last_tx < p.index[-1]:
    st.info(f"台指期資料只到 {last_tx:%Y-%m-%d}，比 00631L 舊；今天的訊號等 FinMind 更新後再按「重新取得最新資料」。")

c1, c2, c3, c4 = st.columns(4)
c1.metric("台指期收盤（結算價）", f"{sig['Close'].iat[-1]:,.0f}")
c2.metric("當月轉折", f"{sig['N'].iat[-1]:,.0f}")
c3.metric("距離轉折", f"{sig['Gap'].iat[-1]:+.2f}%", "在轉折之上" if sig["Target"].iat[-1] > 0 else "在轉折之下",
          delta_color="off")
c4.metric("近月契約", str(sig["Contract"].iat[-1]))

days = CHART_RANGES[range_label]
view = p.tail(days) if days > 0 else p
st.plotly_chart(build_chart(view, sig, trades), width="stretch",
                config={"displaylogo": False, "scrollZoom": True})
st.caption("底色：淺紅＝該次交易獲利，淺綠＝虧損（持有中的交易以最新收盤價計算）。下圖紅線＝當月轉折（每月換一次）。")

tab_log, tab_perf, tab_rules = st.tabs(["近期交易紀錄", "績效對照", "策略規則"])
with tab_log:
    if trades.empty:
        st.info("目前沒有任何進出場紀錄。")
    else:
        recent = trades.tail(10).iloc[::-1]
        log = pd.DataFrame({
            "進場日": recent["entry_date"].dt.strftime("%Y-%m-%d").values,
            "出場日": recent["exit_date"].dt.strftime("%Y-%m-%d").fillna("持有中").values,
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
        st.caption("最近 10 次，最新在最上方。以開盤成交價計算，已扣手續費（3 折）與證交稅 0.1%；持有中的交易為未實現。")

with tab_perf:
    bh = F.buy_hold(p)
    split = pd.Timestamp(SPLIT_DATE)
    rows = {}
    for label, sl in (("全段", slice(None)), ("前段（至 2020-10）", slice(None, split)),
                      ("後段（2020-11 起）", slice(split + pd.Timedelta(days=1), None))):
        for name, rr, hh in (("轉折 F0", ret, hold), ("買進持有", bh, None)):
            x = rr.loc[sl]
            if len(x) < 30:
                continue
            s = F.stats(x, None if hh is None else hh.loc[sl])
            rows[f"{label}｜{name}"] = s
    perf = pd.DataFrame(rows).T
    st.dataframe(perf.style.format({"總報酬": "{:,.0f}%", "年化報酬": "{:.1f}%", "最大回撤": "{:.1f}%",
                                    "報酬回撤比": "{:.2f}", "持有比例": "{:.0f}%"}, na_rep="—"))
    eq = pd.DataFrame({"轉折 F0": (1 + ret).cumprod(), "買進持有": (1 + bh).cumprod()})
    fig = go.Figure()
    for col, color in (("轉折 F0", "#2962FF"), ("買進持有", "#9E9E9E")):
        fig.add_trace(go.Scatter(x=eq.index, y=eq[col], name=col, mode="lines", line=dict(color=color, width=1.4)))
    fig.update_yaxes(type="log", title_text="資金倍數（對數）")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption("兩條線都是 00631L、全額投入、開盤成交、已扣成本；轉折 F0 空手時現金不計利息。報酬回撤比＝年化報酬 ÷ 最大回撤。")
    st.warning("提醒：F0 在台灣（2014 年起的 00631L、2006 年起的模擬）能明顯降低最大回撤，但在美股（SSO、QLD）"
               "沒有通過樣本外測試：回撤變小、報酬少更多。它比較像「避開長期空頭的保險」，"
               "遇到急跌後馬上漲回來（V 型反轉）的行情會少賺。")

with tab_rules:
    st.markdown("""
**訊號序列**：台指期近月連續合約（一般時段 08:45～13:45、不調整價差）。收盤用逐日結算價（沒有就用收盤），最高最低用盤中。

**當月轉折 N**：前 2 個「完整月」的（最高 ＋ 最低）÷ 2，四捨五入到整數；每月第一個交易日換新的 N。

**判斷**：收盤（四捨五入到小數 2 位）> N → 持有；< N → 空手；等於 N → 維持前一天的狀態。

**成交**：台指期收盤判定，下一個交易日 00631L 開盤成交；只做多；全額進出；沒有停損。

**成本**：手續費 0.1425% × 3 折，賣出另加證交稅 0.1%。

**其他**
- 近月＝到期日 > 當天的最小月份契約；到期日用資料裡該契約最後出現的交易日，還在交易的契約用交割月第三個星期三（遇假日順延時，換月那一天可能差 1 天）。
- 00631L 價格只做分割還原（不配息），與主頁相同。
- 這個頁面的訊號與研究回測（dna-audit `futures/etf_apply.py` 的 F0）逐日一致。
""")
