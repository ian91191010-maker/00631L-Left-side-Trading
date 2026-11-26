# -*- coding: utf-8 -*-
"""00631L Signal：00631L 的進出場訊號、K 線與兩個指標圖（整個網站只有這一頁）。

判斷邏輯在 bt/f0.py；指標資料在 bt/gauges.py；快取在 ui/cache.py；這個檔案只負責介面。
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from bt import costs
from bt import data as D
from bt import f0 as F
from bt import gauges as G
from bt.finmind import FinMindError
from ui import cache as UC

CHART_RANGES = {"近 3 個月": 63, "近 6 個月": 126, "近 1 年": 252, "近 2 年": 504, "近 5 年": 1260, "上市以來": 0}
UP, DOWN = "#ef5350", "#26a69a"           # 紅漲綠跌
PROFIT_FILL = "rgba(239, 83, 80, 0.14)"   # 獲利：淺紅
LOSS_FILL = "rgba(38, 166, 154, 0.16)"    # 虧損：淺綠
SPLIT_DATE = "2020-10-30"                 # 績效對照的前後段分界
ROW_HEIGHTS = [0.46, 0.18, 0.18, 0.18]

PAGE_CSS = """
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
.signal-card .tag {font-size: 1rem; font-weight: 600; margin-left: 10px; padding: 2px 8px;
                   border-radius: 6px; border: 1px solid currentColor; opacity: 0.7; vertical-align: middle;}
.signal-card .asof {font-size: 0.95rem; opacity: 0.75; text-align: right; margin-top: 4px;}
@media (max-width: 640px) {
    .signal-card .state, .signal-card .price {font-size: 2rem;}
    .signal-card .price, .signal-card .asof {text-align: left;}
}
</style>
"""


def safe_message(e):
    """可以公開顯示的錯誤文字：已知錯誤照原文（不含 Token），其他只顯示類型。"""
    if isinstance(e, (FinMindError, ValueError)):
        return str(e)
    return f"資料處理發生錯誤（{type(e).__name__}）"


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


def _ds(x):
    """日期 → 'YYYY-MM-DD'（類別軸的格子名稱）。"""
    return pd.Timestamp(x).strftime("%Y-%m-%d")


LOD_DAY_MAX, LOD_WEEK_MAX = 600, 2000      # 看得到的交易日 ≤600：日K；≤2000：週K；更多：月K


def _agg(p, key):
    """把日K合併成週K或月K（照日曆週、月），K 棒放在該期間最後一個交易日的位置。"""
    g = p.groupby(key)
    return {"x": [_ds(d) for d in g.apply(lambda x: x.index[-1])], "o": _clean(g["Open"].first()),
            "h": _clean(g["High"].max()), "l": _clean(g["Low"].min()), "c": _clean(g["Close"].last())}


def lod_level(n_bars):
    return "D" if n_bars <= LOD_DAY_MAX else ("W" if n_bars <= LOD_WEEK_MAX else "M")


def build_chart(p, tx, gauge, g5, trades):
    """四張圖共用同一條橫軸：K 線、台指期與轉折、風險指標、風控指標。"""
    txt, grid = theme_colors()
    idx = p.index
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055, row_heights=ROW_HEIGHTS)
    nohover = dict(hoverinfo="none")
    # 日K、週K、月K 三層都先畫好，只顯示一層；縮放時切換「顯示哪一層」，不替換資料
    # （替換資料會讓 Plotly 把類別軸重新判斷成日期軸）。K 棒太多時看週K／月K：瀏覽器畫 3,000 根很慢，也看不清楚。
    lods = {"D": {"x": list(idx), "o": p["Open"], "h": p["High"], "l": p["Low"], "c": p["Close"]},
            "W": _agg(p, p.index.to_period("W-SUN")), "M": _agg(p, p.index.to_period("M"))}
    level = lod_level(len(idx))
    for lv in ("D", "W", "M"):
        src = lods[lv]
        fig.add_trace(go.Candlestick(
            x=src["x"], open=src["o"], high=src["h"], low=src["l"], close=src["c"], name=f"00631L {lv}",
            whiskerwidth=0, showlegend=False, visible=(lv == level), **nohover,
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
        fig.add_vrect(x0=_ds(max(tr.entry_date, lo)), x1=_ds(min(x1, hi)),
                      fillcolor=PROFIT_FILL if tr.ret > 0 else LOSS_FILL, layer="below", line_width=0, row=1, col=1)
    # 十字線與三張副圖的標籤由 JS 畫在圖上方的文字層，滑鼠移動時不重畫 Plotly 圖（避免 lag）
    doms = {row: list(fig.layout[f"yaxis{row}"].domain) for row in (2, 3, 4)}
    # 橫軸用「交易日」當刻度（類別軸）：每個交易日一格，週末與休市日本來就不存在，
    # 不需要 Plotly 的 rangebreaks（它在拖曳、縮放時會重算上千個缺口，造成 lag、跳動、位置偏移）。
    for trace in fig.data:
        trace.x = [_ds(x) for x in trace.x]
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=[_ds(d) for d in idx],
                     range=[-0.5, len(idx) - 0.5],
                     nticks=8, tickangle=0, gridcolor=grid, linecolor=grid, tickfont=dict(color=txt))
    fig.update_yaxes(gridcolor=grid, zeroline=False, tickfont=dict(color=txt), automargin=True)
    fig.update_layout(height=940, xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=8, b=44),
                      dragmode="pan", hovermode=False, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=txt))
    data = {
        "d": [d.strftime("%Y-%m-%d") for d in idx],
        "o": _clean(p["Open"]), "h": _clean(p["High"]), "l": _clean(p["Low"]), "c": _clean(p["Close"]),
        "tx": _clean(t["Close"], 0), "n": _clean(t["N"], 0), "dif": _clean(g["DIF"], 0), "g5": _clean(r["g5_close"]),
        "lv": level, "dmax": LOD_DAY_MAX, "wmax": LOD_WEEK_MAX,
        "dom": doms, "hi": _clean(g["DIF_HI"], 0), "lo": _clean(g["DIF_LO"], 0),
    }
    return fig, data, txt


CHART_HTML = """
<meta charset="utf-8">
<style>
#strip {font:14px/1.5 'Source Sans Pro',sans-serif;min-height:24px;font-variant-numeric:tabular-nums;margin:0 0 4px 4px}
#box {position:relative}
.ov {position:absolute;pointer-events:none;z-index:5}
.lbl {font:12px 'Source Sans Pro',sans-serif;white-space:nowrap;font-variant-numeric:tabular-nums}
#vline {top:0;width:0;border-left:1px dotted;display:none}
</style>
<div id="strip" style="color:__TXT__"></div>
<div id="box">
__PLOT__
<div id="vline" class="ov"></div>
<div id="l1" class="ov lbl"></div><div id="l2" class="ov lbl"></div><div id="l3" class="ov lbl"></div><div id="l4" class="ov lbl"></div>
</div>
<script>
// 滑鼠移動時只改文字層（十字線、K 線資訊、三張副圖的數值），不呼叫 Plotly.relayout，圖表不重畫。
var D = __DATA__, UPC = "__UP__", DNC = "__DOWN__";
var gd = document.getElementById("pc"), box = document.getElementById("box"), ix = {};
var vline = document.getElementById("vline"), L = {1: document.getElementById("l1"), 2: document.getElementById("l2"), 3: document.getElementById("l3"), 4: document.getElementById("l4")};
D.d.forEach(function (s, i) { ix[s] = i; });
function f(v, n) { return v === null ? "—" : Number(v).toLocaleString("en-US", {minimumFractionDigits: n, maximumFractionDigits: n}); }
function size() {
  var s = gd._fullLayout && gd._fullLayout._size;
  return s || {l: 8, t: 8, w: gd.clientWidth - 16, h: gd.clientHeight - 52};
}
function place() {            // 依目前圖表尺寸擺好標籤與十字線的位置（縮放視窗時也會重算）
  var s = size();
  [2, 3, 4].forEach(function (r) {
    L[r].style.left = s.l + "px";
    L[r].style.top = (s.t + (1 - D.dom[r][1]) * s.h - 18) + "px";
  });
  L[1].style.left = (s.l + 6) + "px"; L[1].style.top = (s.t + 4) + "px";
  vline.style.top = s.t + "px"; vline.style.height = s.h + "px";
}
var last = -1;
function show(i) {
  if (i === last) return;
  last = i;
  var prev = i > 0 ? D.c[i - 1] : D.o[i], chg = (D.c[i] / prev - 1) * 100, col = D.c[i] >= prev ? UPC : DNC;
  document.getElementById("strip").innerHTML =
    '<span style="opacity:.65">' + D.d[i] + '</span>　開 ' + f(D.o[i], 2) + '　高 ' + f(D.h[i], 2) + '　低 ' + f(D.l[i], 2) +
    '　收 <span style="color:' + col + '">' + f(D.c[i], 2) + '（' + (chg >= 0 ? "+" : "") + f(chg, 2) + '%）</span>';
  L[2].textContent = "台指期收盤（藍）／當月轉折（紅）　" + f(D.tx[i], 0) + " / " + f(D.n[i], 0);
  L[3].textContent = "風險指標　" + f(D.dif[i], 0);
  L[4].textContent = "風控指標　" + (D.g5[i] === null ? "—" : f(D.g5[i], 2));
}
// 滑鼠位置 → 第幾根 K 棒：直接用橫軸換算（Plotly 內建的滑鼠提示已關閉，它對 3,000 根 K 棒非常耗時）
var mouse = null, queued = false;
function xa() { return gd._fullLayout.xaxis; }
function frame() {             // 每個畫面更新週期最多處理一次滑鼠移動
  queued = false;
  if (!mouse) return;
  var m = mouse; mouse = null;
  if (m.hide) { vline.style.display = "none"; show(D.d.length - 1); return; }
  var s = size(), px = m.x - s.l, py = m.y - s.t;
  if (px < 0 || px > s.w || py < 0 || py > s.h) { vline.style.display = "none"; show(D.d.length - 1); return; }
  var i = Math.round(xa().p2l(px));
  if (i < 0 || i >= D.d.length) return;
  vline.style.left = (s.l + xa().l2p(i)) + "px"; vline.style.display = "block";
  show(i);
}
function queue(m) { mouse = m; if (!queued) { queued = true; requestAnimationFrame(frame); } }
box.addEventListener("mousemove", function (e) {
  if (e.buttons) { vline.style.display = "none"; return; }     // 拖曳中先藏十字線
  var r = box.getBoundingClientRect(); queue({x: e.clientX - r.left, y: e.clientY - r.top});
});
box.addEventListener("mouseleave", function () { queue({hide: true}); });
gd.on("plotly_afterplot", place);
// 拖曳、縮放停下來 0.15 秒後，上面三張圖的縱軸自動貼合目前看得到的資料（只重畫一次）
var fitTimer = null, fitting = false;
function span(arrs, a, b) {
  var lo = Infinity, hi = -Infinity;
  arrs.forEach(function (arr) { for (var i = a; i <= b; i++) { var v = arr[i]; if (v !== null) { if (v < lo) lo = v; if (v > hi) hi = v; } } });
  if (lo === Infinity) return null;
  var pad = (hi - lo) * 0.06 || Math.abs(hi) * 0.02 || 1;
  return [lo - pad, hi + pad];
}
function yFit(r0, r1) {        // 回傳三張圖貼合 [r0, r1] 這段可見資料的縱軸範圍
  var n = D.d.length, a = Math.max(0, Math.ceil(r0)), b = Math.min(n - 1, Math.floor(r1)), u = {};
  if (b < a) return u;
  var y1 = span([D.l, D.h], a, b), y2 = span([D.tx, D.n], a, b), y3 = span([D.dif, D.hi, D.lo], a, b);
  if (y1) { y1[0] = Math.min(y1[0], y1[0] * 0.985); u["yaxis.range"] = y1; }   // 留位置給買賣三角形
  if (y2) u["yaxis2.range"] = y2;
  if (y3) u["yaxis3.range"] = y3;
  return u;
}
var LV_NAME = {D: "日K", W: "週K", M: "月K"}, curLv = D.lv;
function lvTag() { L[1].textContent = "K 線：" + LV_NAME[curLv] + (curLv === "D" ? "" : "（放大後自動切換成日K）"); }
function fitY() {               // 縱軸貼合可見資料；K 棒數量變了就切換日K／週K／月K，全部在同一次重畫完成
  var r = xa().range, nVis = r[1] - r[0], lv = nVis <= D.dmax ? "D" : (nVis <= D.wmax ? "W" : "M");
  var data = {}, traces = [0, 1, 2];                // 前三條線＝日K、週K、月K
  if (lv !== curLv) { data = {visible: [lv === "D", lv === "W", lv === "M"]}; curLv = lv; lvTag(); }
  else { traces = []; }
  fitting = true;
  Plotly.update(gd, data, yFit(r[0], r[1]), traces).then(function () { fitting = false; }, function () { fitting = false; });
}
gd.on("plotly_relayout", function (e) {
  if (fitting) return;
  var keys = Object.keys(e || {});
  if (!keys.some(function (k) { return k.indexOf("xaxis") === 0; })) return;
  clearTimeout(fitTimer); fitTimer = setTimeout(fitY, 150);
});
function applyTheme() {
  var col = "#31333f";
  try { col = getComputedStyle(window.parent.document.querySelector(".stApp")).color; }
  catch (e) { if (window.matchMedia("(prefers-color-scheme: dark)").matches) col = "#ece9e4"; }
  var m = col.match(/[0-9]+/g).map(Number), lum = 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2];
  var grid = lum > 128 ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.08)", u = {"font.color": col};
  document.getElementById("strip").style.color = col;
  [1, 2, 3, 4].forEach(function (r) { L[r].style.color = col; });
  L[1].style.opacity = 0.7;
  vline.style.borderColor = col;
  ["", "2", "3", "4"].forEach(function (k) {
    u["xaxis" + k + ".tickfont.color"] = col; u["xaxis" + k + ".gridcolor"] = grid;
    u["yaxis" + k + ".tickfont.color"] = col; u["yaxis" + k + ".gridcolor"] = grid;
  });
  Plotly.relayout(gd, u);       // 只在載入時執行一次
}
applyTheme();
lvTag();
place();
show(D.d.length - 1);
</script>
"""


# ===== 介面 =====
st.set_page_config(page_title="00631L Signal", page_icon="📐", layout="wide", initial_sidebar_state="collapsed")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

try:
    token = st.secrets["FINMIND_TOKEN"]
except Exception:
    token = ""

head_l, head_r = st.columns([5, 2], vertical_alignment="center")
head_l.markdown("## 00631L Signal")
if head_r.button("🔄 重新取得最新資料", help="資料每 2 小時自動更新；按這裡可立即更新（全站 10 分鐘內限按一次）。"):
    if token:
        ok, msg = UC.refresh(token)
        (st.toast if ok else st.info)(msg)

if not token:
    st.error("尚未設定 FinMind API Token。請到 App 設定 → Secrets 貼上：`FINMIND_TOKEN = \"你的token\"`")
    st.stop()

# ---------- 主要資料（失敗就無法顯示訊號）----------
try:
    with st.spinner("正在取得資料並計算訊號…"):
        price, sig, notes = UC.signal_core(token)
except Exception as e:
    st.error(f"資料取得失敗：{safe_message(e)}")
    st.caption("可能是網路、FinMind 服務或 Token 額度問題。請稍後按右上角的「重新取得最新資料」。")
    st.stop()

for level, msg in notes:
    if level == "warn":
        st.warning(msg)

# ---------- 參考資料（失敗只顯示提醒，不影響訊號）----------
try:
    gauge = UC.risk_gauge(token)
except Exception as e:
    gauge = pd.DataFrame(columns=["DIF", "DIF_HI", "DIF_LO"], dtype=float)
    st.warning(f"風險指標暫時無法取得（不影響進出場訊號）：{safe_message(e)}")
g5, g5_err = UC.g5_table(G.G5_PATH.stat().st_mtime if G.G5_PATH.exists() else 0)
if g5_err:
    st.warning(f"{g5_err}（不影響進出場訊號）")

sig = sig.dropna(subset=["N"])
if sig.empty or price.empty:
    st.error("資料不足，無法計算訊號（台指期或 00631L 沒有有效資料）。")
    st.stop()
start = max(price.index[0], sig.index[0])
p = price.loc[start:].copy()
if p.empty:
    st.error("00631L 與台指期沒有重疊的交易日，無法計算訊號。")
    st.stop()
hold = F.align(sig, p)
ret, trades = F.simulate(p, hold)
p["Entry"] = (hold.diff().fillna(hold) > 0)
p["Exit"] = (hold.diff().fillna(0) < 0)

title, accent = describe_state(sig)
r_, g_, b_ = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
last_tx = sig.index[-1]
provisional = last_tx == pd.Timestamp(D.today_tw())
state_html = f'{title}<span class="tag">暫定</span>' if provisional else title
st.markdown(f"""
<div class="signal-card" style="--accent: {accent}; --tint: rgba({r_}, {g_}, {b_}, 0.16);">
  <div class="state">{state_html}</div>
  <div>
    <div class="price">{p['Close'].iat[-1]:.2f}</div>
    <div class="asof">{p.index[-1]:%Y-%m-%d}｜{last_tx:%Y-%m-%d}</div>
  </div>
</div>
""", unsafe_allow_html=True)
if provisional:
    st.caption(f"暫定：台指期 {last_tx:%Y-%m-%d} 的結算價是今天剛取得的，可能還沒定案，正式訊號以隔天早上的資料為準。")

c1, c2, c3 = st.columns(3)
c1.metric("台指期收盤（結算價）", f"{sig['Close'].iat[-1]:,.0f}")
c2.metric("當月轉折", f"{sig['N'].iat[-1]:,.0f}")
c3.metric("距離轉折", f"{sig['Gap'].iat[-1]:+.2f}%", "在轉折之上" if sig["Target"].iat[-1] > 0 else "在轉折之下",
          delta_color="off")


@st.fragment
def chart_section(p, sig, gauge, g5, trades):
    """切換顯示區間時只重畫這一區，不重算整頁。"""
    range_label = st.segmented_control("顯示區間", list(CHART_RANGES), default="近 1 年",
                                       label_visibility="collapsed")
    days = CHART_RANGES[range_label or "近 1 年"]
    view = p.tail(days) if days > 0 else p
    fig, chart_data, txt_color = build_chart(view, sig, gauge, g5, trades)
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="pc",
                            config={"displaylogo": False, "scrollZoom": True, "displayModeBar": False})
    html = (CHART_HTML.replace("__TXT__", txt_color).replace("__PLOT__", plot_html)
            .replace("__DATA__", json.dumps(chart_data, allow_nan=False)).replace("__UP__", UP)
            .replace("__DOWN__", DOWN))
    components.html(html, height=1000)


chart_section(p, sig, gauge, g5, trades)

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
        st.caption("漲跌幅已扣成本；持有中的交易以最新收盤價計算、尚未扣賣出成本。")

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
    if rows:
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
with st.expander("說明"):
    st.markdown(f"""
- 資料來源：FinMind（台指期逐日結算價、00631L 與加權指數日 K）。00631L 只做分割還原，不配息。
- 訊號在台指期收盤後判定，下一個交易日 00631L 開盤成交；只做多、全額進出。
- 成本：{costs.describe()}。績效已扣成本，空手時現金不計利息。
- 資料每 2 小時自動更新；已結束年份的台指期資料長期保存，不重複下載。
- K 線底色：淺紅＝該次交易獲利，淺綠＝虧損（持有中以最新收盤價計）。黃色三角＝買進，藍色三角＝賣出。
- 圖表：四張圖共用橫軸，滑鼠移上去時垂直虛線會同時出現在四張圖，K 線資訊顯示在 K 線圖上方。
- 風險指標、風控指標為參考指標，僅供對照，不構成投資建議。
""")
