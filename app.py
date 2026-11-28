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
ROW_HEIGHTS = [0.38, 0.15, 0.28, 0.19]   # 00631L K 線、風控指標、台指期 K 線、風險指標
CHART_HEIGHT = 1120

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


TF_OPTIONS = {"日線": "D", "週線": "W", "月線": "M"}
TF_NAME = {"D": "日K", "W": "週K", "M": "月K"}
# 看得到的 K 棒太多時自動合併成較粗的週期（瀏覽器畫上千根 K 棒很慢，也看不清楚）；放大後回到選的週期。
# 每一項：(週期, 這個週期最多顯示幾根「選定週期」的 K 棒；None＝不限)
LOD_CHAIN = {"D": [("D", 600), ("W", 2000), ("M", None)],
             "W": [("W", 800), ("M", None)],
             "M": [("M", None)]}
TX_NAME_COLOR = "#FF9800"                 # 當月轉折：橘色階梯線


def _key(index, lv):
    """日期 → 所屬 K 棒（日：當天；週：該週；月：該月）。"""
    if lv == "D":
        return pd.Index(index.normalize())
    return pd.Index(index.to_period("W-SUN" if lv == "W" else "M"))


def _ohlc(df, key):
    g = df.groupby(key)
    return pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(),
                         "Low": g["Low"].min(), "Close": g["Close"].last()})


def _layers(bars, base, labels):
    """選定週期的 K 棒（bars）再合併成較粗週期的 K 棒，放在該期間最後一根的位置。"""
    out = {}
    ends = pd.to_datetime(pd.Series(labels, index=bars.index))
    for lv, _ in LOD_CHAIN[base]:
        if lv == base:
            out[lv] = {"x": list(labels), "o": bars["Open"], "h": bars["High"], "l": bars["Low"], "c": bars["Close"],
                       "pos": list(range(len(labels)))}
            continue
        key = _key(pd.DatetimeIndex(ends.values), lv)
        g = bars.reset_index(drop=True).groupby(key)
        last = pd.Series(list(labels)).groupby(key).last()
        pos = pd.Series(range(len(labels))).groupby(key).last()
        out[lv] = {"x": list(last), "o": g["Open"].first(), "h": g["High"].max(), "l": g["Low"].min(),
                   "c": g["Close"].last(), "pos": list(pos)}
    return out


def build_chart(p, tx, gauge, g5, trades, tf="D"):
    """四張圖共用同一條橫軸：00631L K 線、風控指標、台指期 K 線＋當月轉折、風險指標。tf：D／W／M。"""
    txt, grid = theme_colors()
    key = _key(p.index, tf)
    bars = _ohlc(p, key)
    end = pd.Series(p.index, index=key).groupby(level=0).last()          # 每根 K 棒＝該期間最後一個交易日
    labels = [_ds(d) for d in end.reindex(bars.index)]
    n = len(bars)
    entry = p["Entry"].groupby(key).any().reindex(bars.index, fill_value=False)
    exit_ = p["Exit"].groupby(key).any().reindex(bars.index, fill_value=False)

    t = tx.loc[tx.index >= p.index[0] - pd.Timedelta(days=40)]
    tkey = _key(t.index, tf)
    tbars = _ohlc(t, tkey).reindex(bars.index)
    tn = t["N"].groupby(tkey).last().reindex(bars.index)
    g = gauge.groupby(_key(gauge.index, tf)).last().reindex(bars.index) if len(gauge) else \
        pd.DataFrame(index=bars.index, columns=["DIF", "DIF_HI", "DIF_LO"], dtype=float)
    if len(g5):
        g5k = _key(g5.index, tf)
        r = pd.DataFrame({"g5_close": g5["g5_close"].groupby(g5k).last(),
                          "g5_min": g5["g5_min"].groupby(g5k).min()}).reindex(bars.index)
    else:
        r = pd.DataFrame(index=bars.index, columns=["g5_close", "g5_min"], dtype=float)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.075, row_heights=ROW_HEIGHTS)
    nohover = dict(hoverinfo="none")
    chain = LOD_CHAIN[tf]
    level = next(i for i, (_, mx) in enumerate(chain) if mx is None or n <= mx)
    # K 線：選定週期＋較粗週期都先畫好，只顯示一層；縮放時切換「顯示哪一層」，不替換資料
    # （替換資料會讓 Plotly 把類別軸重新判斷成日期軸）。
    k1 = []
    for i, (lv, src) in enumerate(_layers(bars, tf, labels).items()):
        k1.append(len(fig.data))
        fig.add_trace(go.Candlestick(
            x=src["x"], open=src["o"], high=src["h"], low=src["l"], close=src["c"], name=f"00631L {lv}",
            whiskerwidth=0, showlegend=False, visible=(i == level), **nohover,
            increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN)), row=1, col=1)
    lab = pd.Series(labels, index=bars.index)
    fig.add_trace(go.Scatter(x=lab[entry].tolist(), y=(bars["Low"][entry] * 0.985).tolist(), mode="markers",
                             hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="triangle-up", size=10, color="#FFD700",
                                         line=dict(width=1, color="#555"))), row=1, col=1)
    fig.add_trace(go.Scatter(x=lab[exit_].tolist(), y=(bars["High"][exit_] * 1.015).tolist(), mode="markers",
                             hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="triangle-down", size=10, color="#1E90FF",
                                         line=dict(width=1, color="#555"))), row=1, col=1)
    lay1 = _layers(bars, tf, labels)
    groups = [[] for _ in chain]                      # 每個合併層級的所有線（K 線＋指標線），一起切換顯示

    def lines(series, row, **kw):
        """同一條線每個合併層級各畫一條（取該期間最後一個值）；K 棒合併時線也跟著變少點，重畫比較快。"""
        vals = list(series)
        for i, (lv, src) in enumerate(lay1.items()):
            groups[i].append(len(fig.data))
            fig.add_trace(go.Scatter(x=src["x"], y=[vals[q] for q in src["pos"]], mode="lines", showlegend=False,
                                     visible=(i == level), **nohover, **kw), row=row, col=1)

    # 風控指標（第 2 張）
    lines(r["g5_close"], 2, line=dict(color="#b98bf0", width=1.3), connectgaps=False)
    hit = r["g5_min"] < 34
    fig.add_trace(go.Scatter(x=lab[hit].tolist(), y=r["g5_close"][hit].tolist(), mode="markers", hoverinfo="skip",
                             showlegend=False, marker=dict(size=5, color=UP)), row=2, col=1)
    for lv in (67, 50, 34):
        fig.add_hline(y=lv, row=2, col=1, line=dict(color="#8d887f" if lv != 50 else "#555", width=0.8, dash="dash"))
    fig.update_yaxes(range=[10, 90], row=2, col=1, tickvals=[34, 50, 67])
    if not r["g5_close"].notna().any():
        fig.add_annotation(text="資料尚未送入", xref="x2 domain", yref="y2 domain", x=0.5, y=0.5, showarrow=False,
                           font=dict(color="#8d887f", size=13))
    # 台指期 K 線＋當月轉折（第 3 張）：黑白 K 棒（漲空心、跌實心），顏色跟著深淺色主題
    k3 = []
    for i, (lv, src) in enumerate(_layers(tbars, tf, labels).items()):
        k3.append(len(fig.data))
        fig.add_trace(go.Candlestick(
            x=src["x"], open=src["o"], high=src["h"], low=src["l"], close=src["c"], name=f"TX {lv}",
            whiskerwidth=0, showlegend=False, visible=(i == level), **nohover,
            increasing=dict(line=dict(color=txt, width=1), fillcolor="rgba(0,0,0,0)"),
            decreasing=dict(line=dict(color=txt, width=1), fillcolor=txt)), row=3, col=1)
    lines(tn, 3, line=dict(color=TX_NAME_COLOR, width=1.5, shape="hv"))
    # 風險指標（第 4 張）
    lines(g["DIF"], 4, line=dict(color="#e0b84a", width=1.3))
    for col in ("DIF_HI", "DIF_LO"):
        lines(g[col], 4, line=dict(color="#8d887f", width=0.9, dash="dash"))
    # 持有期間底色
    pos = {d: i for i, d in enumerate(bars.index)}
    lo, hi = p.index[0], p.index[-1]
    for trd in trades.itertuples():
        x1 = hi if pd.isna(trd.exit_date) else trd.exit_date
        if x1 < lo or trd.entry_date > hi:
            continue
        a = pos.get(_key(pd.DatetimeIndex([max(trd.entry_date, lo)]), tf)[0])
        b = pos.get(_key(pd.DatetimeIndex([min(x1, hi)]), tf)[0])
        if a is None or b is None:
            continue
        fig.add_vrect(x0=labels[a], x1=labels[b], fillcolor=PROFIT_FILL if trd.ret > 0 else LOSS_FILL,
                      layer="below", line_width=0, row=1, col=1)
    doms = {row: list(fig.layout["yaxis" + ("" if row == 1 else str(row))].domain) for row in (1, 2, 3, 4)}
    # 橫軸用「K 棒序號」當刻度（類別軸）：週末與休市日本來就不存在，不需要 Plotly 的 rangebreaks。
    # 每張圖都顯示日期刻度。
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=labels, range=[-0.5, n - 0.5],
                     showticklabels=True, nticks=8, tickangle=0, gridcolor=grid, linecolor=grid,
                     tickfont=dict(color=txt, size=11))
    fig.update_yaxes(gridcolor=grid, zeroline=False, tickfont=dict(color=txt), automargin=True)
    fig.update_layout(height=CHART_HEIGHT, margin=dict(l=8, r=8, t=8, b=30),
                      dragmode="pan", hovermode=False, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=txt))
    for k in ("", "2", "3", "4"):
        fig.layout["xaxis" + k].rangeslider = dict(visible=False)
    data = {
        "d": labels,
        "o": _clean(bars["Open"]), "h": _clean(bars["High"]), "l": _clean(bars["Low"]), "c": _clean(bars["Close"]),
        "to": _clean(tbars["Open"], 0), "th": _clean(tbars["High"], 0), "tl": _clean(tbars["Low"], 0),
        "tc": _clean(tbars["Close"], 0), "n": _clean(tn, 0),
        "dif": _clean(g["DIF"], 0), "hi": _clean(g["DIF_HI"], 0), "lo": _clean(g["DIF_LO"], 0),
        "g5": _clean(r["g5_close"]),
        "lay": {"names": [TF_NAME[lv] for lv, _ in chain], "max": [mx for _, mx in chain], "k3": k3, "cur": level,
                "groups": [[k1[i], k3[i]] + groups[i] for i in range(len(chain))]},
        "dom": doms,
    }
    return fig, data, txt


CHART_HTML = """
<meta charset="utf-8">
<style>
#strip {font:14px/1.5 'Source Sans Pro',sans-serif;min-height:24px;font-variant-numeric:tabular-nums;margin:0 0 4px 4px}
#box {position:relative}
#box.touch, #box.touch * {touch-action: pan-y !important}
.ov {position:absolute;pointer-events:none;z-index:5}
.lbl {font:12px 'Source Sans Pro',sans-serif;white-space:nowrap;font-variant-numeric:tabular-nums}
#vline {top:0;width:0;border-left:1px dotted;display:none}
.dtag {font:11px 'Source Sans Pro',sans-serif;padding:1px 5px;border-radius:3px;display:none;transform:translateX(-50%);white-space:nowrap;font-variant-numeric:tabular-nums}
</style>
<div id="strip" style="color:__TXT__"></div>
<div id="box">
__PLOT__
<div id="vline" class="ov"></div>
<div id="l1" class="ov lbl"></div><div id="t1" class="ov dtag"></div><div id="t2" class="ov dtag"></div><div id="t3" class="ov dtag"></div><div id="t4" class="ov dtag"></div><div id="l2" class="ov lbl"></div><div id="l3" class="ov lbl"></div><div id="l4" class="ov lbl"></div>
</div>
<script>
// 滑鼠移動時只改文字層（十字線、K 線資訊、三張副圖的數值），不呼叫 Plotly.relayout，圖表不重畫。
var D = __DATA__, UPC = "__UP__", DNC = "__DOWN__";
var gd = document.getElementById("pc"), box = document.getElementById("box"), ix = {};
var vline = document.getElementById("vline"), L = {1: document.getElementById("l1"), 2: document.getElementById("l2"), 3: document.getElementById("l3"), 4: document.getElementById("l4")};
var T = {1: document.getElementById("t1"), 2: document.getElementById("t2"), 3: document.getElementById("t3"), 4: document.getElementById("t4")};
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
  [1, 2, 3, 4].forEach(function (r) { T[r].style.top = (s.t + (1 - D.dom[r][0]) * s.h + 3) + "px"; });
}
function tags(px, i) {          // 每張圖下方的日期刻度上，顯示對齊線所在的日期
  [1, 2, 3, 4].forEach(function (r) {
    if (px === null) { T[r].style.display = "none"; return; }
    T[r].textContent = D.d[i]; T[r].style.left = px + "px"; T[r].style.display = "block";
  });
}
var last = -1;
function show(i) {
  if (i === last) return;
  last = i;
  var prev = i > 0 ? D.c[i - 1] : D.o[i], chg = (D.c[i] / prev - 1) * 100, col = D.c[i] >= prev ? UPC : DNC;
  document.getElementById("strip").innerHTML =
    '<span style="opacity:.65">' + D.d[i] + '</span>　開 ' + f(D.o[i], 2) + '　高 ' + f(D.h[i], 2) + '　低 ' + f(D.l[i], 2) +
    '　收 <span style="color:' + col + '">' + f(D.c[i], 2) + '（' + (chg >= 0 ? "+" : "") + f(chg, 2) + '%）</span>';
  L[2].textContent = "風控指標　" + (D.g5[i] === null ? "—" : f(D.g5[i], 2));
  L[3].textContent = "台指期　開 " + f(D.to[i], 0) + "　高 " + f(D.th[i], 0) + "　低 " + f(D.tl[i], 0) +
                     "　收 " + f(D.tc[i], 0) + "　｜　當月轉折（橘）" + f(D.n[i], 0);
  L[4].textContent = "風險指標　" + f(D.dif[i], 0);
}
// 滑鼠位置 → 第幾根 K 棒：直接用橫軸換算（Plotly 內建的滑鼠提示已關閉，它對 3,000 根 K 棒非常耗時）
var mouse = null, queued = false;
function xa() { return gd._fullLayout.xaxis; }
function frame() {             // 每個畫面更新週期最多處理一次滑鼠移動
  queued = false;
  if (!mouse) return;
  var m = mouse; mouse = null;
  if (m.hide) { hideCross(); return; }
  var s = size(), px = m.x - s.l, py = m.y - s.t;
  if (px < 0 || px > s.w || py < 0 || py > s.h) { hideCross(); return; }
  var i = Math.round(xa().p2l(px));
  if (i < 0 || i >= D.d.length) return;
  var x = s.l + xa().l2p(i);
  vline.style.left = x + "px"; vline.style.display = "block"; tags(x, i);
  show(i);
}
function hideCross() { vline.style.display = "none"; tags(null); show(D.d.length - 1); }
function queue(m) { mouse = m; if (!queued) { queued = true; requestAnimationFrame(frame); } }
box.addEventListener("mousemove", function (e) {
  if (e.buttons) { vline.style.display = "none"; tags(null); return; }     // 拖曳中先藏十字線
  var r = box.getBoundingClientRect(); queue({x: e.clientX - r.left, y: e.clientY - r.top});
});
box.addEventListener("mouseleave", function () { queue({hide: true}); });

// ---------- 手機觸控 ----------
// 一指左右滑＝對齊線跟著手指（上下滑交給瀏覽器捲動網頁）；一指點一下＝對齊線跳到那一天；
// 兩指捏合／張開＝縮放，兩指一起移動＝平移。
var TOUCH = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
if (TOUCH) box.classList.add("touch");
function tpos(t) { var r = box.getBoundingClientRect(); return {x: t.clientX - r.left, y: t.clientY - r.top}; }
var swipe = null, pinch = null, pinchPending = null, pinchBusy = false, pinchQueued = false;
function pinchStep() {
  pinchQueued = false;
  if (pinchBusy || !pinchPending) return;
  var r = pinchPending; pinchPending = null; pinchBusy = true;
  Plotly.relayout(gd, {"xaxis.range": r}).then(done, done);
  function done() { pinchBusy = false; if (pinchPending) schedPinch(); }
}
function schedPinch() { if (!pinchQueued) { pinchQueued = true; requestAnimationFrame(pinchStep); } }
// 觸控事件全部由這段程式處理、不交給 Plotly（Plotly 收到觸控會擋住網頁捲動）
box.addEventListener("touchstart", function (e) {
  e.stopPropagation();
  if (e.touches.length >= 2) {
    e.preventDefault(); e.stopPropagation();
    var a = tpos(e.touches[0]), b = tpos(e.touches[1]), s = size(), r = xa().range.map(Number);
    pinch = {d0: Math.max(30, Math.abs(a.x - b.x)), c0: xa().p2l((a.x + b.x) / 2 - s.l), r0: r};
    vline.style.display = "none"; tags(null);
  } else if (e.touches.length === 1 && !pinch) {
    var p0 = tpos(e.touches[0]); swipe = {x: p0.x, y: p0.y, dir: null};
  }
}, {passive: false, capture: true});
box.addEventListener("touchmove", function (e) {
  e.stopPropagation();
  if (pinch && e.touches.length >= 2) {
    e.preventDefault(); e.stopPropagation();
    var a = tpos(e.touches[0]), b = tpos(e.touches[1]), s = size(), n = D.d.length;
    var d = Math.max(30, Math.abs(a.x - b.x)), cx = (a.x + b.x) / 2 - s.l;
    var span = Math.max(15, Math.min(n, (pinch.r0[1] - pinch.r0[0]) * pinch.d0 / d));
    var lo = pinch.c0 - cx / s.w * span, hi = lo + span;          // 兩指中間那一天保持在兩指中間
    if (lo < -0.5) { hi += -0.5 - lo; lo = -0.5; }
    if (hi > n - 0.5) { lo -= hi - (n - 0.5); hi = n - 0.5; }
    pinchPending = [lo, hi]; schedPinch();
  } else if (!pinch && e.touches.length === 1 && swipe) {
    var p = tpos(e.touches[0]);
    if (!swipe.dir) {                                              // 先看手指往哪個方向動，決定後整個手勢都照這個方向
      var dx = Math.abs(p.x - swipe.x), dy = Math.abs(p.y - swipe.y);
      if (dx < 6 && dy < 6) return;
      swipe.dir = dx > dy ? "h" : "v";
    }
    if (swipe.dir === "h") { e.preventDefault(); queue(p); }        // 左右：移動對齊線；上下：不攔截，網頁照常捲動
  }
}, {passive: false, capture: true});
function touchEnd(e) {
  e.stopPropagation();
  if (pinch && e.touches.length < 2) pinch = null;
  if (!e.touches.length) {
    if (swipe && !swipe.dir && e.type === "touchend") queue({x: swipe.x, y: swipe.y});   // 點一下（沒有移動）：對齊線跳到那一天
    swipe = null;
  }
}
box.addEventListener("touchend", touchEnd, {capture: true});
box.addEventListener("touchcancel", touchEnd, {capture: true});
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
  var y1 = span([D.l, D.h], a, b), y3 = span([D.tl, D.th, D.n], a, b), y4 = span([D.dif, D.hi, D.lo], a, b);
  if (y1) { y1[0] = Math.min(y1[0], y1[0] * 0.985); u["yaxis.range"] = y1; }   // 留位置給買賣三角形
  if (y3) u["yaxis3.range"] = y3;                                                  // 風控指標（第 2 張）固定 10～90
  if (y4) u["yaxis4.range"] = y4;
  return u;
}
var LAY = D.lay, curLv = LAY.cur;
function lvTag() {
  L[1].textContent = "K 線：" + LAY.names[curLv] + (curLv === 0 ? "" : "（K 棒太多，自動合併；放大後回到" + LAY.names[0] + "）");
}
function fitY() {               // 縱軸貼合可見資料；K 棒數量變了就切換合併層級（00631L 與台指期一起），同一次重畫完成
  var r = xa().range, nVis = r[1] - r[0], lv = 0;
  while (LAY.max[lv] !== null && nVis > LAY.max[lv]) lv++;
  var data = {}, traces = [];
  if (lv !== curLv) {               // 舊層級全部隱藏、新層級全部顯示（K 線與指標線一起）
    var off = LAY.groups[curLv], on = LAY.groups[lv];
    traces = off.concat(on); data = {visible: off.map(function () { return false; }).concat(on.map(function () { return true; }))};
    curLv = lv; lvTag();
  }
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
  var m = /^#([0-9a-f]{6})$/i.test(col) ? [0, 2, 4].map(function (k) { return parseInt(col.substr(1 + k, 2), 16); })
                                          : col.match(/[0-9.]+/g).map(Number);
  var lum = 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2];
  var grid = lum > 128 ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.08)", u = {"font.color": col};
  document.getElementById("strip").style.color = col;
  [1, 2, 3, 4].forEach(function (r) { L[r].style.color = col; });
  L[1].style.opacity = 0.7;
  vline.style.borderColor = col;
  ["", "2", "3", "4"].forEach(function (k) {
    u["xaxis" + k + ".tickfont.color"] = col; u["xaxis" + k + ".gridcolor"] = grid;
    u["yaxis" + k + ".tickfont.color"] = col; u["yaxis" + k + ".gridcolor"] = grid;
  });
  if (TOUCH) u["dragmode"] = false;   // 手機：一指不拖曳圖表（才能上下捲動網頁），改用兩指平移
  var panel = lum > 128 ? "rgba(30,30,30,0.92)" : "rgba(255,255,255,0.95)";
  [1, 2, 3, 4].forEach(function (r) { T[r].style.color = col; T[r].style.background = panel; T[r].style.border = "1px solid " + grid; });
  // 台指期黑白 K 棒：漲空心、跌實心，顏色＝文字色（深色主題白、淺色主題黑）
  Plotly.update(gd, {"increasing.line.color": col, "decreasing.line.color": col, "decreasing.fillcolor": col,
                     "increasing.fillcolor": "rgba(0,0,0,0)"}, u, LAY.k3);   // 只在載入時執行一次
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
    c_range, c_tf = st.columns([3, 1.3], vertical_alignment="center")
    with c_range:
        range_label = st.segmented_control("顯示區間", list(CHART_RANGES), default="近 1 年",
                                           label_visibility="collapsed")
    with c_tf:
        tf_label = st.segmented_control("K 線週期", list(TF_OPTIONS), default="日線", label_visibility="collapsed",
                                        help="所有圖表一起切換。看得到的 K 棒太多時會自動合併成較粗的週期，放大後回到這裡選的週期。")
    days = CHART_RANGES[range_label or "近 1 年"]
    view = p.tail(days) if days > 0 else p
    fig, chart_data, txt_color = build_chart(view, sig, gauge, g5, trades, TF_OPTIONS[tf_label or "日線"])
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="pc",
                            config={"displaylogo": False, "scrollZoom": True, "displayModeBar": False})
    html = (CHART_HTML.replace("__TXT__", txt_color).replace("__PLOT__", plot_html)
            .replace("__DATA__", json.dumps(chart_data, allow_nan=False)).replace("__UP__", UP)
            .replace("__DOWN__", DOWN))
    components.html(html, height=CHART_HEIGHT + 60)


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
- 圖表由上而下：00631L K 線、風控指標、台指期 K 線（漲空心、跌實心）＋當月轉折（橘色階梯線）、風險指標。四張圖共用橫軸，可切換日線／週線／月線；滑鼠移上去時，四張圖會同時出現垂直虛線與日期。
- 風險指標、風控指標為參考指標，僅供對照，不構成投資建議。
""")
