# -*- coding: utf-8 -*-
"""前向記錄頁面：每天記下「轉折 F0」與「審計系統二（原版）」的訊號，純做紀錄。

記錄檔：data/forward_log.csv（由 forward/record.py 每天追加，只增不改）。
這個頁面只負責顯示；績效用記錄檔裡的持有狀態，配 00631L 價格重算（隔日開盤成交、含成本）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bt import audit2 as A2          # noqa: E402
from bt import data as D             # noqa: E402
from bt import f0 as F               # noqa: E402

TICKER = "00631L"
LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "forward_log.csv"
STATE_COLOR = {"買進": "#2E7D32", "持有": "#2E7D32", "賣出": "#C62828", "空手": "#F9A825"}

st.set_page_config(page_title="00631L 前向記錄", page_icon="🗓️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
.sig-grid {display: flex; flex-wrap: wrap; gap: 14px; margin: 4px 0 18px 0;}
.sig-card {flex: 1 1 320px; padding: 16px 22px; border-radius: 10px;
           border-left: 8px solid var(--accent); background: var(--tint);}
.sig-card .name {font-size: 0.95rem; opacity: 0.75;}
.sig-card .state {font-size: 2.2rem; font-weight: 700; line-height: 1.15;}
.sig-card .detail {font-size: 0.95rem; opacity: 0.8; margin-top: 2px;}
</style>
""", unsafe_allow_html=True)


def card(name, state, detail):
    accent = STATE_COLOR.get(state, "#888888")
    r, g, b = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
    return (f'<div class="sig-card" style="--accent:{accent};--tint:rgba({r},{g},{b},0.16);">'
            f'<div class="name">{name}</div><div class="state">{state}</div>'
            f'<div class="detail">{detail}</div></div>')


@st.cache_data(show_spinner=False, ttl=1800)
def load_log(mtime):
    df = pd.read_csv(LOG_PATH, dtype={"date": str, "recorded_on": str})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def recompute(token):
    """用現在的資料重算全部訊號，拿來和記錄檔對照（偵測資料被事後修正）。"""
    price, _ = D.load_price(TICKER, token)
    return price, F.compute_f0(F.fetch_tx(token)), A2.compute_signals(price)


st.markdown("## 00631L 前向記錄")
st.caption("每天記下兩個策略的訊號，純做紀錄、不改過去。記錄檔只在檔案尾端新增；"
           "「即時」＝訊號日隔天以內記下，「補記」＝之後才補（只用到當天以前的資料）。")

if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
    st.info("記錄檔還沒有資料。排程啟動後，每個交易日的隔天早上會記下前一個交易日的訊號。")
    st.stop()

log = load_log(LOG_PATH.stat().st_mtime)
last = log.iloc[-1]
agree = last["f0_hold"] == last["a2_hold"]
st.markdown(
    '<div class="sig-grid">'
    + card("轉折 F0（正式訊號）", last["f0_action"],
           f"{last['date']:%Y-%m-%d} 台指期收 {last['tx_close']:,.0f}｜轉折價 {last['tx_turning_n']:,}")
    + card("審計系統二 原版（參考）", last["a2_action"],
           f"{last['date']:%Y-%m-%d} 00631L 收 {last['etf_close']:.2f}")
    + "</div>", unsafe_allow_html=True)
st.caption(f"最新一筆：{last['date']:%Y-%m-%d}（{last['kind']}，{last['recorded_on']} 記下）｜"
           f"兩個策略{'方向一致' if agree else '方向不一致'}｜共 {len(log)} 個交易日")

tab_log, tab_perf, tab_check = st.tabs(["每日記錄", "記錄後績效", "核對與說明"])

with tab_log:
    show = pd.DataFrame({
        "日期": log["date"].dt.strftime("%Y-%m-%d"),
        "台指期收盤": log["tx_close"], "轉折價": log["tx_turning_n"],
        "F0 動作": log["f0_action"], "系統二 動作": log["a2_action"],
        "00631L 收盤": log["etf_close"], "記錄日": log["recorded_on"], "方式": log["kind"],
    }).iloc[::-1]

    def color(v):
        c = STATE_COLOR.get(v)
        return f"color:{c};font-weight:600" if c else ""

    st.dataframe(show.style.map(color, subset=["F0 動作", "系統二 動作"])
                 .format({"台指期收盤": "{:,.0f}", "轉折價": "{:,}", "00631L 收盤": "{:.2f}"}),
                 hide_index=True, height=480)
    st.caption("最新在最上方。動作＝這一天收盤判定的結果，下一個交易日 00631L 開盤執行。")

with tab_perf:
    try:
        token = st.secrets["FINMIND_TOKEN"]
    except Exception:
        token = ""
    if not token:
        st.info("需要 FinMind Token 才能算績效（App 設定 → Secrets：`FINMIND_TOKEN = \"你的token\"`）。")
    else:
        try:
            price, _, _ = recompute(token)
        except Exception as e:
            st.error(f"資料取得失敗：{e}")
            price = None
        if price is not None:
            p = price.loc[log["date"].iloc[0]:]
            after = p.index[p.index > log["date"].iloc[-1]]      # 只算到「最新記錄日的下一個交易日」開盤成交為止
            if len(after):
                p = p.loc[:after[0]]
            if len(p) < 3:
                st.info("記錄還太短，累積幾個交易日後再看。")
            else:
                holds = {}
                for name, col in (("轉折 F0", "f0_hold"), ("審計系統二 原版", "a2_hold")):
                    s = log.set_index("date")[col].astype(float)
                    # 起算日前一天的狀態＝空手（記錄從 2026-11-24 收盤起算）
                    holds[name] = F.align(pd.DataFrame({"Target": s}), p)
                rets = {n: F.simulate(p, h)[0] for n, h in holds.items()}
                rets["買進持有"] = F.buy_hold(p)
                eq = pd.DataFrame({n: (1 + r).cumprod() for n, r in rets.items()})
                fig = go.Figure()
                for col, color_ in zip(eq.columns, ("#2962FF", "#FF9800", "#9E9E9E")):
                    fig.add_trace(go.Scatter(x=eq.index, y=eq[col], name=col, mode="lines+markers",
                                             line=dict(color=color_, width=1.6)))
                fig.update_yaxes(title_text="資金倍數（起算日＝1）")
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
                tbl = pd.DataFrame({"累計報酬": (eq.iloc[-1] - 1) * 100,
                                    "持有天數": [int(holds[n].sum()) if n in holds else len(p) for n in eq.columns]})
                st.dataframe(tbl.style.format({"累計報酬": "{:+.2f}%"}))
                st.caption("用記錄檔裡的持有狀態配 00631L 價格重算：隔日開盤成交、手續費 3 折、賣出加證交稅 0.1%、全額投入。"
                           "起算日前視為空手，所以起算當天若是「持有」，累計報酬會從隔日開盤才開始算。"
                           "天數很少時只是記錄，不能拿來判斷策略好壞。")

with tab_check:
    st.markdown("""
**這個頁面在記什麼**
- 轉折 F0：台指期收盤（逐日結算價）站上當月轉折 → 持有 00631L，跌破 → 空手。00631L 的正式進出場指標。
- 審計系統二（原版）：TSI 藍線 > 信號線進場、SMA5 跌破 SMA55 出場。只做參考對照，不是進出場依據。
- 兩者都在收盤後判定、下一個交易日開盤成交；規則與「轉折F0」、「審計系統二」頁面相同。

**記錄的規則**
- 只在檔案尾端新增，過去的列不改（GitHub 的提交紀錄可以查驗）。
- 只記「今天以前」的交易日，因為當天的台指期結算價可能還沒定案。
- 「補記」的列也只用當天以前的資料算，沒有用到未來資料；但記錄日在訊號日之後，不算即時。
""")
    try:
        token = st.secrets["FINMIND_TOKEN"]
    except Exception:
        token = ""
    if token:
        try:
            _, sig_now, a2_now = recompute(token)
            chk = log.set_index("date")
            f_now = sig_now["Target"].reindex(chk.index)
            a_now = a2_now["Position"].reindex(chk.index)
            bad = chk[(f_now != chk["f0_hold"]) | (a_now != chk["a2_hold"])]
            if bad.empty:
                st.success(f"核對：記錄檔 {len(chk)} 天的持有狀態，與現在重算的結果完全一致。")
            else:
                st.warning("核對：以下日期的記錄與現在重算不同（可能是資料來源事後修正）。記錄檔不會被改動，以記錄為準。")
                st.dataframe(bad[["f0_hold", "a2_hold"]].assign(重算F0=f_now[bad.index], 重算系統二=a_now[bad.index]))
        except Exception as e:
            st.caption(f"核對略過（資料取得失敗：{e}）")
