# -*- coding: utf-8 -*-
"""00631L Signal 頁面：00631L 的進出場訊號、K 線與兩個指標圖。

判斷邏輯在 bt/f0.py；指標資料在 bt/gauges.py；這個檔案只負責介面。
這一頁沒有側邊欄，導覽放在頁面最下方。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bt import data as D          # noqa: E402
from bt import f0 as F            # noqa: E402
from bt import gauges as G        # noqa: E402

TICKER = "00631L"
CHART_RANGES = {"近 3 個月": 63, "近 6 個月": 126, "近 1 年": 252, "近 2 年": 504, "近 5 年": 1260, "上市以來": 0}
UP, DOWN = "#ef5350", "#26a69a"           # 紅漲綠跌
PROFIT_FILL = "rgba(239, 83, 80, 0.14)"   # 獲利：淺紅
LOSS_FILL = "rgba(38, 166, 154, 0.16)"    # 虧損：淺綠
SPLIT_DATE = "2020-10-30"                 # 績效對照的前後段分界
ROW_HEIGHTS = [0.46, 0.18, 0.18, 0.18]

st.set_page_config(page_title="00631L Signal", page_icon="📐", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stSidebarNav"] {display: none !important;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
.signal-card {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 12px 40px; padding: 18px 26px; margin: 4px 0 18px 0;
    border-radius: 10px; border-left: 8px solid var(--accent); background: var(--tint);
}
.signal-card .state {font-size: 2.6rem; font-weight: 700; line-height: 1.1;}
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
    taiex = D.load_benchmark("TAIEX", token)
    return price, sig, G.risk_gauge(taiex), notes


@st.cache_data(show_spinner=False, ttl=1800)
def get_g5(mtime):
    return G.load_g5()


def describe_state(sig):
    now, prev = sig["Target"].iat[-1], sig["Target"].iat[-2] if len(sig) > 1 else 0.0
    if now > prev:
        return "買進", "#2E7D32"
    if now < prev:
        return "賣出", "#C62828"
    if now > 0:
        return "持有", "#2E7D32"
    return "空手", "#F9A825"


def theme_colors():
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = True
    return ("#ece9e4", "rgba(255,255,255,0.10)") if dark else ("#31333f", "rgba(0,0,0,0.08)")


def _clean(s, nd=2):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


def build_chart(p, tx, gauge, g5, trades):
    """四張圖共用同一條橫軸：K 線、台指期與轉折、風險指標、風控指標。"""
    txt, grid = theme_colors()
    idx = p.index
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055, row_heights=ROW_HEIGHTS)
    nohover = dict(hoverinfo="none")
    fig.add_trace(go.Candlestick(
        x=idx, open=p["Open"], high=p["High"], low=p["Low"], close=p["Close"], name="00631L",
        whiskerwidth=0, showlegend=False, **nohover,
        increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN)), row=1, col=1)
    buys, sells = p[p["Entry"]], p[p["Exit"]]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Low"] * 0.985, mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="triangle-up", size=10, color="#FFD700",
                                         line=dict(width=1, color="#555"))), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["High"] * 1.015, mode="markers", hoverinfo="skip",
                             showlegend=False, marker=dict(symbol="triangle-down", size=10, color="#1E90FF",
                                                           line=dict(width=1, color="#555"))), row=1, col=1)
    t = tx.reindex(idx)
    fig.add_trace(go.Scatter(x=idx, y=t["Close"], mode="lines", showlegend=False, line=dict(color="#4f7cff", width=1.3),
                             **nohover), row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=t["N"], mode="lines", showlegend=False,
                             line=dict(color="#FF5252", width=1.3, shape="hv"), **nohover), row=2, col=1)
    g = gauge.reindex(idx)
    fig.add_trace(go.Scatter(x=idx, y=g["DIF"], mode="lines", showlegend=False, line=dict(color="#e0b84a", width=1.3),
                             **nohover), row=3, col=1)
    for col in ("DIF_HI", "DIF_LO"):
        fig.add_trace(go.Scatter(x=idx, y=g[col], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="#8d887f", width=0.9, dash="dash")), row=3, col=1)
    r = g5.reindex(idx)
    fig.add_trace(go.Scatter(x=idx, y=r["g5_close"], mode="lines", showlegend=False, line=dict(color="#b98bf0", width=1.3),
                             connectgaps=False, **nohover), row=4, col=1)
    hit = r[r["g5_min"] < 34]
    fig.add_trace(go.Scatter(x=hit.index, y=hit["g5_close"], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=5, color=UP)), row=4, col=1)
    for lv in (67, 50, 34):
        fig.add_hline(y=lv, row=4, col=1, line=dict(color="#8d887f" if lv != 50 else "#555", width=0.8, dash="dash"))
    fig.update_yaxes(range=[10, 90], row=4, col=1, tickvals=[34, 50, 67])
    if not r["g5_close"].notna().any():
        fig.add_annotation(text="資料尚未送入", xref="x4 domain", yref="y4 domain", x=0.5, y=0.5, showarrow=False,
                           font=dict(color="#8d887f", size=13))

    lo, hi = idx[0], idx[-1]
    for tr in trades.itertuples():
        x1 = hi if pd.isna(tr.exit_date) else tr.exit_date
        if x1 < lo or tr.entry_date > hi:
            continue
        fig.add_vrect(x0=max(tr.entry_date, lo), x1=min(x1, hi), fillcolor=PROFIT_FILL if tr.ret > 0 else LOSS_FILL,
                      layer="below", line_width=0, row=1, col=1)
    cross_k = len(fig.layout.shapes)      # 十字線：垂直虛線橫跨四張圖，由 JS 隨滑鼠移動
    fig.add_shape(type="line", xref="x", yref="paper", x0=lo, x1=lo, y0=0, y1=1, visible=False,
                  line=dict(color=txt, width=1, dash="dot"))
    labels = {2: "台指期收盤（藍）／當月轉折（紅）", 3: "風險指標", 4: "風控指標"}
    for row, label in labels.items():
        dom = fig.layout[f"yaxis{row}"].domain
        fig.add_annotation(text=label, xref="paper", yref="paper", x=0, y=dom[1] + 0.004, xanchor="left",
                           yanchor="bottom", showarrow=False, font=dict(color=txt, size=12))
    n_note = len(fig.layout.annotations) - 3            # 前面的「資料尚未送入」不算
    missing = [d.strftime("%Y-%m-%d") for d in pd.date_range(lo, hi, freq="D").difference(idx)]
    fig.update_xaxes(rangebreaks=[dict(values=missing)], gridcolor=grid, linecolor=grid, tickfont=dict(color=txt),
                     tickformatstops=[dict(dtickrange=[None, "M1"], value="%m-%d"),
                                      dict(dtickrange=["M1", "M12"], value="%Y-%m"),
                                      dict(dtickrange=["M12", None], value="%Y")])
    fig.update_yaxes(gridcolor=grid, zeroline=False, tickfont=dict(color=txt), automargin=True)
    fig.update_layout(height=940, xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=8, b=44),
                      dragmode="pan", hovermode="x", showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=txt))
    data = {
        "d": [d.strftime("%Y-%m-%d") for d in idx],
        "o": _clean(p["Open"]), "h": _clean(p["High"]), "l": _clean(p["Low"]), "c": _clean(p["Close"]),
        "tx": _clean(t["Close"], 0), "n": _clean(t["N"], 0), "dif": _clean(g["DIF"], 0), "g5": _clean(r["g5_close"]),
        "cross": cross_k, "note": n_note,
    }
    return fig, data, txt


CHART_HTML = """
<meta charset="utf-8">
<div id="strip" style="font:14px/1.5 'Source Sans Pro',sans-serif;color:__TXT__;min-height:24px;
     font-variant-numeric:tabular-nums;margin:0 0 4px 4px"></div>
__PLOT__
<script>
var D = __DATA__, UPC = "__UP__", DNC = "__DOWN__";
var gd = document.getElementById("pc"), ix = {};
D.d.forEach(function (s, i) { ix[s] = i; });
function f(v, n) { return v === null ? "—" : Number(v).toLocaleString("en-US", {minimumFractionDigits: n, maximumFractionDigits: n}); }
function show(i) {
  var prev = i > 0 ? D.c[i - 1] : D.o[i], chg = (D.c[i] / prev - 1) * 100, col = D.c[i] >= prev ? UPC : DNC;
  document.getElementById("strip").innerHTML =
    '<span style="opacity:.65">' + D.d[i] + '</span>　開 ' + f(D.o[i], 2) + '　高 ' + f(D.h[i], 2) + '　低 ' + f(D.l[i], 2) +
    '　收 <span style="color:' + col + '">' + f(D.c[i], 2) + '（' + (chg >= 0 ? "+" : "") + f(chg, 2) + '%）</span>';
  var a = D.note, u = {};
  u["annotations[" + a + "].text"] = "台指期收盤（藍）／當月轉折（紅）　" + f(D.tx[i], 0) + " / " + f(D.n[i], 0);
  u["annotations[" + (a + 1) + "].text"] = "風險指標　" + f(D.dif[i], 0);
  u["annotations[" + (a + 2) + "].text"] = "風控指標　" + (D.g5[i] === null ? "—" : f(D.g5[i], 2));
  Plotly.relayout(gd, u);
}
function cross(x) {
  var u = {}; u["shapes[" + D.cross + "].x0"] = x; u["shapes[" + D.cross + "].x1"] = x; u["shapes[" + D.cross + "].visible"] = true;
  Plotly.relayout(gd, u);
}
gd.on("plotly_hover", function (e) {
  var x = String(e.points[0].x).slice(0, 10), i = ix[x];
  if (i === undefined) return;
  cross(x); show(i);
});
gd.on("plotly_unhover", function () {
  var u = {}; u["shapes[" + D.cross + "].visible"] = false; Plotly.relayout(gd, u); show(D.d.length - 1);
});
function applyTheme() {
  var col = "#31333f";
  try { col = getComputedStyle(window.parent.document.querySelector(".stApp")).color; }
  catch (e) { if (window.matchMedia("(prefers-color-scheme: dark)").matches) col = "#ece9e4"; }
  var m = col.match(/[0-9]+/g).map(Number), lum = 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2];
  var grid = lum > 128 ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.08)", u = {"font.color": col};
  document.getElementById("strip").style.color = col;
  ["", "2", "3", "4"].forEach(function (k) {
    u["xaxis" + k + ".tickfont.color"] = col; u["xaxis" + k + ".gridcolor"] = grid;
    u["yaxis" + k + ".tickfont.color"] = col; u["yaxis" + k + ".gridcolor"] = grid;
  });
  for (var j = 0; j < 3; j++) u["annotations[" + (D.note + j) + "].font.color"] = col;
  u["shapes[" + D.cross + "].line.color"] = col;
  Plotly.relayout(gd, u);
}
applyTheme();
show(D.d.length - 1);
</script>
"""


# ===== 介面 =====
try:
    token = st.secrets["FINMIND_TOKEN"]
except Exception:
    token = ""

head_l, head_r = st.columns([5, 2], vertical_alignment="center")
head_l.markdown("## 00631L Signal")
if head_r.button("🔄 重新取得最新資料", help="資料預設每 3 小時自動更新；按這裡可立即更新。"):
    st.cache_data.clear()

if not token:
    st.error("尚未設定 FinMind API Token。請到 App 設定 → Secrets 貼上：`FINMIND_TOKEN = \"你的token\"`")
    st.stop()

try:
    with st.spinner("正在取得資料並計算訊號…"):
        price, sig, gauge, notes = get_results(token)
except Exception as e:
    st.error(f"資料取得失敗：{e}")
    st.caption("可能是網路、FinMind 服務或 Token 額度問題。請稍後按右上角的「重新取得最新資料」。")
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

title, accent = describe_state(sig)
r_, g_, b_ = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
last_tx = sig.index[-1]
st.markdown(f"""
<div class="signal-card" style="--accent: {accent}; --tint: rgba({r_}, {g_}, {b_}, 0.16);">
  <div class="state">{title}</div>
  <div>
    <div class="price">{p['Close'].iat[-1]:.2f}</div>
    <div class="asof">{p.index[-1]:%Y-%m-%d}｜{last_tx:%Y-%m-%d}</div>
  </div>
</div>
""", unsafe_allow_html=True)
if last_tx == pd.Timestamp(D.today_tw()):
    st.caption(f"台指期 {last_tx:%Y-%m-%d} 的結算價是今天剛取得的，可能還沒定案，正式訊號以隔天早上的資料為準。")

c1, c2, c3 = st.columns(3)
c1.metric("台指期收盤（結算價）", f"{sig['Close'].iat[-1]:,.0f}")
c2.metric("當月轉折", f"{sig['N'].iat[-1]:,.0f}")
c3.metric("距離轉折", f"{sig['Gap'].iat[-1]:+.2f}%", "在轉折之上" if sig["Target"].iat[-1] > 0 else "在轉折之下",
          delta_color="off")

range_label = st.segmented_control("顯示區間", list(CHART_RANGES), default="近 1 年", label_visibility="collapsed")
days = CHART_RANGES[range_label or "近 1 年"]
view = p.tail(days) if days > 0 else p
g5 = get_g5(G.G5_PATH.stat().st_mtime if G.G5_PATH.exists() else 0)
fig, chart_data, txt_color = build_chart(view, sig, gauge, g5, trades)
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="pc",
                        config={"displaylogo": False, "scrollZoom": True, "displayModeBar": False})
html = (CHART_HTML.replace("__TXT__", txt_color).replace("__PLOT__", plot_html)
        .replace("__DATA__", json.dumps(chart_data, allow_nan=False)).replace("__UP__", UP).replace("__DOWN__", DOWN))
components.html(html, height=1000)

tab_log, tab_perf = st.tabs(["近期交易紀錄", "績效對照"])
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

with tab_perf:
    bh = F.buy_hold(p)
    split = pd.Timestamp(SPLIT_DATE)
    rows = {}
    for label, sl in (("全段", slice(None)), ("前段（至 2020-10）", slice(None, split)),
                      ("後段（2020-11 起）", slice(split + pd.Timedelta(days=1), None))):
        for name, rr, hh in (("策略", ret, hold), ("買進持有", bh, None)):
            x = rr.loc[sl]
            if len(x) < 30:
                continue
            rows[f"{label}｜{name}"] = F.stats(x, None if hh is None else hh.loc[sl])
    perf = pd.DataFrame(rows).T
    st.dataframe(perf.style.format({"總報酬": "{:,.0f}%", "年化報酬": "{:.1f}%", "最大回撤": "{:.1f}%",
                                    "報酬回撤比": "{:.2f}", "持有比例": "{:.0f}%"}, na_rep="—"))
    eq = pd.DataFrame({"策略": (1 + ret).cumprod(), "買進持有": (1 + bh).cumprod()})
    fig2 = go.Figure()
    for col, color in (("策略", "#2962FF"), ("買進持有", "#9E9E9E")):
        fig2.add_trace(go.Scatter(x=eq.index, y=eq[col], name=col, mode="lines", line=dict(color=color, width=1.4)))
    fig2.update_yaxes(type="log", title_text="資金倍數（對數）")
    fig2.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                       legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    st.plotly_chart(fig2, width="stretch", config={"displaylogo": False})

st.divider()
nav = st.columns(4)
nav[0].page_link("app - 00631L-Left-side-Trading.py", label="主頁")
nav[1].page_link("pages/1_策略回測.py", label="策略回測")
nav[2].page_link("pages/2_審計系統二.py", label="審計系統二")
nav[3].page_link("pages/4_前向記錄.py", label="前向記錄")
with st.expander("說明"):
    st.markdown("""
- 資料來源：FinMind（台指期逐日結算價、00631L 與加權指數日 K）。00631L 只做分割還原，不配息。
- 訊號在台指期收盤後判定，下一個交易日 00631L 開盤成交；只做多、全額進出。
- 成本：手續費 0.1425% × 3 折，賣出另加證交稅 0.1%。績效已扣成本，空手時現金不計利息。
- K 線底色：淺紅＝該次交易獲利，淺綠＝虧損（持有中以最新收盤價計）。黃色三角＝買進，藍色三角＝賣出。
- 圖表：四張圖共用橫軸，滑鼠移上去時垂直虛線會同時出現在四張圖，K 線資訊顯示在 K 線圖上方。
- 風險指標、風控指標為參考指標，僅供對照，不構成投資建議。
""")
