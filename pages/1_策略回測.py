# -*- coding: utf-8 -*-
"""策略回測頁面（介面）。

側邊欄勾選進出場機制與參數，主畫面顯示績效、圖表、交易明細與參數比較。
計算邏輯都在 bt/ 資料夾，這個檔案只負責介面。
"""
import inspect
import sys
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# 讓 pages/ 底下的頁面找得到 bt/ 模組
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bt import backtest as B          # noqa: E402
from bt import baseline as BL         # noqa: E402
from bt import charts as C            # noqa: E402
from bt import data as D              # noqa: E402
from bt.strategy import (StrategyConfig, TradeSettings, build_indicators,  # noqa: E402
                         enabled_entry_labels, run_strategy)

st.set_page_config(page_title="策略回測", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1400px;}</style>",
            unsafe_allow_html=True)

# ============================================================
# 預設值與預設組合
# ============================================================
CFG_FIELDS = [f.name for f in fields(StrategyConfig)]
BASE_CFG = asdict(StrategyConfig())
FLAG_FIELDS = [f for f in CFG_FIELDS if f.startswith(("use_", "exit_"))]


def _preset(**on):
    """從預設參數出發，先把所有機制關掉，再打開指定的機制。"""
    p = dict(BASE_CFG)
    for f in FLAG_FIELDS:
        p[f] = False
    p.update(on)
    return p


PRESETS = {
    "組合A": _preset(use_ma_entry=True, use_tsi_pos=True, use_tsi_above_sig=True,
                    use_atr_stop=True, atr_mult=2.5, atr_buffer=5.0,
                    exit_mkt=True, exit_dead_cross=True, exit_tsi_zero=True),
    "組合B（00631L）": _preset(use_trend=True, trend_n=120, use_adx=True, adx_th=20.0,
                             use_tsi_pos=True, use_tsi_above_sig=True,
                             use_atr_stop=True, atr_mult=2.5, atr_buffer=5.0, exit_trend=True),
    "對照組": _preset(use_ma_entry=True, exit_dead_cross=True),
    # 與「審計系統二」頁面相同：TSI 藍>紅進場、SMA5 跌破 SMA55 出場、沒有停損
    "審計系統二": _preset(use_tsi_above_sig=True, tsi_r=25, tsi_s=13, tsi_sig=13,
                     exit_dead_cross=True, ma_type="SMA", ma_short=5, ma_long=55),
}

SETTING_DEFAULTS = {
    "set_ticker": "00631L",
    "set_bench": "TAIEX",
    "set_start": date(2016, 1, 1),
    "set_end": D.today_tw(),
    "set_cash": 1_000_000,
    "set_pos": 100,
    "set_fee": 0.1425,
    "set_disc": 1.0,
    "set_tax": 0.1,
    "set_slip": 0.05,
    "set_tf": "D",
    "set_exec": "next_open",
}


def init_state():
    for f, v in PRESETS["組合A"].items():
        st.session_state.setdefault("cfg_" + f, v)
    for k, v in SETTING_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("loaded_preset", "組合A")


def load_preset(name):
    for f, v in PRESETS[name].items():
        st.session_state["cfg_" + f] = v
    st.session_state["loaded_preset"] = name


init_state()


def K(f):
    return "cfg_" + f


def on(f):
    return bool(st.session_state[K(f)])


# ============================================================
# 側邊欄
# ============================================================
sb = st.sidebar
sb.subheader("預設組合")
pc = sb.columns(2)
for k, name in enumerate(PRESETS):
    pc[k % 2].button(name, on_click=load_preset, args=(name,), width="stretch")
sb.caption(f"最後載入：{st.session_state['loaded_preset']}（之後的修改不會改變這個標示）")

with sb.expander("回測設定", expanded=False):
    st.text_input("標的代號", key="set_ticker")
    st.text_input("大盤代號", key="set_bench", help="FinMind 的加權指數代號是 TAIEX（Yahoo 的 ^TWII 不適用）")
    c1, c2 = st.columns(2)
    c1.date_input("開始日", key="set_start", min_value=date(2014, 1, 1))
    c2.date_input("結束日", key="set_end", min_value=date(2014, 1, 1))
    st.number_input("初始資金", key="set_cash", min_value=10_000, step=100_000)
    st.radio("倉位比例", [30, 50, 100], key="set_pos", horizontal=True, format_func=lambda v: f"{v}%")
    c1, c2 = st.columns(2)
    c1.number_input("手續費 %", key="set_fee", min_value=0.0, step=0.01, format="%.4f")
    c2.number_input("手續費折數", key="set_disc", min_value=0.05, max_value=1.0, step=0.05,
                    help="1.0 = 不打折，0.6 = 6 折")
    st.radio("證交稅", [0.1, 0.3], key="set_tax", horizontal=True,
             format_func=lambda v: "ETF 0.1%" if v == 0.1 else "個股 0.3%")
    st.number_input("滑價 %", key="set_slip", min_value=0.0, step=0.05, format="%.2f")
    st.radio("時間框架", ["D", "W"], key="set_tf", horizontal=True,
             format_func=lambda v: "日線" if v == "D" else "週線")
    st.radio("訊號成交價", ["next_open", "close"], key="set_exec",
             format_func=lambda v: "訊號隔日開盤價（建議）" if v == "next_open" else "訊號當日收盤價（有前視偏差）")

with sb.expander("1. 進場條件（勾選者需同時成立）", expanded=True):
    st.checkbox("均線多頭", key=K("use_ma_entry"))
    st.radio("均線判斷方式", ["state", "cross"], key=K("ma_entry_mode"), horizontal=True,
             disabled=not on("use_ma_entry"),
             format_func=lambda v: "短均在長均之上" if v == "state" else "黃金交叉當天")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("均線", ["SMA", "EMA"], key=K("ma_type"))
    c2.number_input("短", key=K("ma_short"), min_value=1, step=1)
    c3.number_input("長", key=K("ma_long"), min_value=2, step=1)
    st.checkbox("TSI > 0", key=K("use_tsi_pos"))
    st.checkbox("TSI 藍線 > 信號線", key=K("use_tsi_above_sig"))
    c1, c2, c3 = st.columns(3)
    c1.number_input("TSI r", key=K("tsi_r"), min_value=1, step=1)
    c2.number_input("TSI s", key=K("tsi_s"), min_value=1, step=1)
    c3.number_input("信號線", key=K("tsi_sig"), min_value=1, step=1)
    st.checkbox("ADX > 門檻", key=K("use_adx"))
    c1, c2 = st.columns(2)
    c1.number_input("ADX 門檻", key=K("adx_th"), min_value=0.0, step=5.0, disabled=not on("use_adx"))
    c2.number_input("ADX 週期", key=K("adx_len"), min_value=2, step=1)
    st.checkbox("趨勢濾網：收盤 > N 日均線", key=K("use_trend"))
    st.number_input("趨勢均線 N", key=K("trend_n"), min_value=2, step=10,
                    help="同時用於「跌破趨勢均線」出場")
    st.checkbox("大盤多頭才進場（大盤收盤 > 大盤 EMA）", key=K("use_mkt_filter"))
    st.number_input("大盤 EMA", key=K("mkt_ema"), min_value=2, step=10,
                    help="同時用於「大盤惡化」出場")

with sb.expander("2. 停損", expanded=True):
    st.checkbox("ATR 停損", key=K("use_atr_stop"))
    c1, c2, c3 = st.columns(3)
    c1.number_input("ATR 倍數", key=K("atr_mult"), min_value=0.5, step=0.5,
                    disabled=not on("use_atr_stop") or on("use_scale_stop"))
    c2.number_input("緩衝 %", key=K("atr_buffer"), min_value=0.0, max_value=50.0, step=5.0)
    c3.number_input("ATR 週期", key=K("atr_len"), min_value=2, step=1)
    st.caption("停損價 = (進場價 − ATR × 倍數) × (1 − 緩衝%)，ATR 在進場當下固定")
    st.checkbox("固定百分比停損", key=K("use_fixed_stop"))
    st.number_input("固定停損 %", key=K("fixed_stop_pct"), min_value=0.5, step=1.0,
                    disabled=not on("use_fixed_stop"))
    st.checkbox("分批停損（取代單一 ATR 停損）", key=K("use_scale_stop"))
    c1, c2 = st.columns(2)
    c1.number_input("第一段 ATR×（賣 50%）", key=K("scale_mult1"), min_value=0.5, step=0.5,
                    disabled=not on("use_scale_stop"))
    c2.number_input("第二段 ATR×（賣剩餘）", key=K("scale_mult2"), min_value=0.5, step=0.5,
                    disabled=not on("use_scale_stop"))
    st.checkbox("TSI 藍穿紅且均線多頭時回補", key=K("use_readd"), disabled=not on("use_scale_stop"))

with sb.expander("3. 停利", expanded=False):
    st.checkbox("移動停損：從最高點回落", key=K("use_trail_stop"))
    st.number_input("回落 %", key=K("trail_pct"), min_value=1.0, step=1.0, disabled=not on("use_trail_stop"))
    st.checkbox("移動停利：獲利達標後啟動", key=K("use_trail_tp"))
    c1, c2 = st.columns(2)
    c1.number_input("啟動獲利 %", key=K("trail_tp_trigger"), min_value=1.0, step=5.0,
                    disabled=not on("use_trail_tp"))
    c2.number_input("回落 % ", key=K("trail_tp_pct"), min_value=1.0, step=1.0, disabled=not on("use_trail_tp"))
    st.checkbox("ATR 移動停利：最高點 − ATR × 倍數", key=K("use_atr_trail"))
    st.number_input("ATR 倍數 ", key=K("atr_trail_mult"), min_value=0.5, step=0.5,
                    disabled=not on("use_atr_trail"))

with sb.expander("4. 訊號出場（可複選）", expanded=True):
    st.checkbox("大盤惡化：大盤收盤下穿其 EMA", key=K("exit_mkt"))
    st.checkbox("跌破趨勢均線", key=K("exit_trend"))
    st.checkbox("TSI 破零", key=K("exit_tsi_zero"))
    st.checkbox("均線死亡交叉", key=K("exit_dead_cross"))
    st.checkbox("TSI 下穿信號線減半倉", key=K("exit_tsi_half"))
    st.caption("同一根K棒多個條件成立時：停損 → 移動停利／停損 → 大盤惡化 → 跌破趨勢均線 → TSI破零 → 死叉 → 減半")

# ============================================================
# 讀取設定
# ============================================================
cfg = StrategyConfig(**{f: st.session_state[K(f)] for f in CFG_FIELDS})
for f in ("ma_short", "ma_long", "tsi_r", "tsi_s", "tsi_sig", "adx_len", "trend_n", "mkt_ema", "atr_len"):
    setattr(cfg, f, int(getattr(cfg, f)))
ss = st.session_state
settings = TradeSettings(initial_cash=float(ss.set_cash), position_pct=float(ss.set_pos),
                         fee_rate=float(ss.set_fee), fee_discount=float(ss.set_disc),
                         tax_rate=float(ss.set_tax), slippage=float(ss.set_slip), exec_price=ss.set_exec)
ticker = ss.set_ticker.strip().upper()
bench_id = ss.set_bench.strip().upper()
start, end = pd.Timestamp(ss.set_start), pd.Timestamp(ss.set_end)
timeframe = ss.set_tf

# ============================================================
# 主畫面
# ============================================================
st.markdown(f"## 策略回測　{ticker}")
st.caption("資料來自 FinMind 原始日K（已依分割表還原）。預設訊號於下一根K棒開盤成交，價格停損以盤中觸價成交。")

try:
    token = st.secrets["FINMIND_TOKEN"]
except Exception:
    token = ""
if not token:
    st.error("尚未設定 FinMind API Token。請到 App 設定 → Secrets 貼上：`FINMIND_TOKEN = \"你的token\"`")
    st.stop()

problems = []
if not enabled_entry_labels(cfg):
    problems.append("請至少勾選一個進場條件。")
if cfg.ma_short >= cfg.ma_long:
    problems.append("短均線必須小於長均線。")
if start >= end:
    problems.append("開始日必須早於結束日。")
if problems:
    for p in problems:
        st.warning(p)
    st.stop()

try:
    with st.spinner("正在取得資料…"):
        daily, notes = D.load_price(ticker, token)
        try:
            bench_daily = D.load_benchmark(bench_id, token)
        except Exception as e:
            bench_daily = None
            notes.append(("warn", f"大盤 {bench_id} 資料取得失敗，「大盤多頭」與「大盤惡化」不會觸發：{e}"))
except Exception as e:
    st.error(f"資料取得失敗：{e}")
    st.caption("可能是代號錯誤、網路、FinMind 服務或 Token 額度問題。")
    st.stop()

for level, msg in notes:
    (st.warning if level == "warn" else st.caption)(msg)

bars = D.resample_bars(daily, timeframe)
bench_bars = D.resample_bars(bench_daily, timeframe) if bench_daily is not None else None
ind = build_indicators(bars, bench_bars, cfg)
if ind[(ind.index >= start) & (ind.index <= end)].empty:
    st.warning("回測區間內沒有資料，請調整日期。")
    st.stop()

res = run_strategy(ind, cfg, settings, start, end)
equity, trades, fills = res["equity"], res["trades"], res["fills"]
bh = B.buy_and_hold(ind, settings, start, end)
perf = B.performance(equity, settings.initial_cash, trades, res["shares"])
perf_bh = B.performance(bh, settings.initial_cash)
# 基準：主頁的審計策略，一律用日線、同樣的日期與費用
base = BL.run_baseline(daily, settings, start, end)
perf_base = B.performance(base["equity"], settings.initial_cash, base["trades"], base["shares"])
real_start, real_end = equity.index[0], equity.index[-1]

if settings.exec_price == "close":
    st.warning("目前使用「訊號當日收盤價」成交：收盤才知道的訊號用收盤價買賣，實際做不到，績效會偏樂觀。")


# ---------- 績效摘要 ----------
def pct(v, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.2f}%" if sign else f"{v:.2f}%"


def num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return "∞" if np.isinf(v) else f"{v:.2f}"


st.caption(f"回測期間 {real_start:%Y-%m-%d} ～ {real_end:%Y-%m-%d}　·　"
           f"{'日線' if timeframe == 'D' else '週線'}　·　倉位 {settings.position_pct:.0f}%　·　"
           f"與「{BL.NAME}」比較（基準一律日線、依自身 30/60/100% 分段）")


def diff_text(key, unit="%"):
    a, b = perf[key], perf_base[key]
    if any(isinstance(v, float) and (np.isnan(v) or np.isinf(v)) for v in (a, b)):
        return None
    return f"{a - b:+.2f}{unit} vs 基準"


# 差異：比基準好顯示紅色、比較差顯示綠色（台股習慣）
m = st.columns(4)
m[0].metric("總報酬", pct(perf["總報酬%"]), diff_text("總報酬%"), delta_color="inverse")
m[1].metric("年化報酬", pct(perf["年化報酬%"]), diff_text("年化報酬%"), delta_color="inverse")
m[2].metric("最大回撤", pct(perf["最大回撤%"]), diff_text("最大回撤%"), delta_color="inverse",
            help="回撤是負數，差異為正代表回撤比基準小")
m[3].metric("Calmar", num(perf["Calmar"]), diff_text("Calmar", ""), delta_color="inverse")

ROWS = [("總報酬%", "總報酬", "pct"), ("年化報酬%", "年化報酬", "pct"), ("最大回撤%", "最大回撤", "pct"),
        ("Calmar", "Calmar", "num"), ("勝率%", "勝率", "pct0"), ("交易次數", "交易次數", "int"),
        ("獲利因子", "獲利因子", "num"), ("持倉時間%", "持倉時間", "pct0")]


def fmt_cell(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if kind == "pct":
        return f"{v:+.2f}%"
    if kind == "pct0":
        return f"{v:.1f}%"
    if kind == "int":
        return f"{int(v)}"
    return num(v)


def diff_cell(key, kind):
    a, b = perf[key], perf_base[key]
    if any(isinstance(v, float) and (np.isnan(v) or np.isinf(v)) for v in (a, b)):
        return "—"
    d = a - b
    if kind == "int":
        return f"{int(d):+d}"
    if kind in ("pct", "pct0"):
        return f"{d:+.2f}%"
    return f"{d:+.2f}"


cmp_df = pd.DataFrame({
    "指標": [r[1] for r in ROWS],
    "目前測試": [fmt_cell(perf[k], t) for k, _, t in ROWS],
    "審計策略（基準）": [fmt_cell(perf_base[k], t) for k, _, t in ROWS],
    "差異（測試−基準）": [diff_cell(k, t) for k, _, t in ROWS],
    "買進持有": [fmt_cell(perf_bh.get(k), t) if k in ("總報酬%", "年化報酬%", "最大回撤%", "Calmar") else "—"
             for k, _, t in ROWS],
})
BETTER_HIGH = {"總報酬", "年化報酬", "最大回撤", "Calmar", "勝率", "獲利因子"}


def cmp_style(row):
    out = [""] * len(row)
    v = row["差異（測試−基準）"]
    if row["指標"] in BETTER_HIGH and v not in ("—",) and v.lstrip("+-").replace("%", "").replace(".", "").isdigit():
        d = float(v.replace("%", ""))
        if d != 0:
            css = ("color:#ef5350;font-weight:600" if d > 0 else "color:#26a69a;font-weight:600")
            out[list(row.index).index("差異（測試−基準）")] = css
    return out


st.dataframe(cmp_df.style.apply(cmp_style, axis=1), hide_index=True)

tab_chart, tab_eq, tab_trades, tab_grid, tab_rules = st.tabs(
    ["K線圖", "資金曲線", "交易明細", "參數比較", "規則說明"])

# ---------- K線圖 ----------
RANGES = {"近 3 個月": 3, "近 6 個月": 6, "近 1 年": 12, "近 2 年": 24, "全部": 0}
with tab_chart:
    c1, c2, c3 = st.columns([3, 1, 1])
    rng = c1.select_slider("顯示區間", options=list(RANGES), value="近 3 個月")
    show_tsi = c2.checkbox("TSI", value=True)
    show_adx = c3.checkbox("ADX", value=True)
    view = ind[(ind.index >= real_start) & (ind.index <= real_end)]
    if RANGES[rng]:
        view = view[view.index >= real_end - pd.DateOffset(months=RANGES[rng])]
    fig = C.price_chart(view, res["stop_line"], fills, trades, cfg, show_tsi, show_adx, daily=(timeframe == "D"))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    st.caption("底色：淺紅＝該次交易獲利，淺綠＝虧損。紅色點線＝當下最近的停損價。空心三角＝部分賣出。")

# ---------- 資金曲線 ----------
with tab_eq:
    fig = C.equity_chart(equity, bh, B.drawdown(equity), B.drawdown(bh),
                         base["equity"], B.drawdown(base["equity"]))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

# ---------- 交易明細 ----------
with tab_trades:
    which = st.radio("顯示", ["目前測試", BL.NAME], horizontal=True, label_visibility="collapsed")
    show_trades = trades if which == "目前測試" else base["trades"]
    if show_trades.empty:
        st.info("這段期間沒有任何交易。")
    else:
        t = show_trades.iloc[::-1]
        log = pd.DataFrame({
            "進場日": t["entry_date"].dt.strftime("%Y-%m-%d").values,
            "出場日": t["exit_date"].dt.strftime("%Y-%m-%d").fillna("持有中").values,
            "持有K棒": t["bars"].astype(int).values,
            "進場價": t["entry_price"].values,
            "出場價": t["exit_price"].values,
            "損益%": t["ret"].values,
            "損益金額": t["pnl"].values,
            "出場原因": t["reason"].values,
        })

        def row_style(row):
            v = row["損益%"]
            css = ("color:#ef5350;font-weight:600;background-color:rgba(239,83,80,0.14)" if v > 0 else
                   "color:#26a69a;font-weight:600;background-color:rgba(38,166,154,0.16)")
            return [css if c in ("損益%", "損益金額") else "" for c in row.index]

        styled = (log.style.format({"進場價": "{:.2f}", "出場價": "{:.2f}", "損益%": "{:+.2f}%",
                                    "損益金額": "{:+,.0f}"}).apply(row_style, axis=1))
        st.dataframe(styled, hide_index=True, height=min(38 * (len(log) + 1), 600))
        st.caption("進場價、出場價為含滑價的加權平均成交價；損益已扣手續費、證交稅。"
                   "出場原因有多段時以「→」串接（例如：分批停損① → 均線死叉）。")
        st.download_button("下載交易明細 CSV", log.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{ticker}_{'test' if which == '目前測試' else 'baseline'}_trades.csv",
                           mime="text/csv")

# ---------- 參數比較 ----------
with tab_grid:
    st.caption("以目前側邊欄的設定為基準，最多選兩個參數，各自輸入要測試的數值（用逗號分隔）。"
               f"組合數上限 {B.MAX_COMBOS}。")
    names = list(B.PARAM_REGISTRY)
    with st.form("grid_form"):
        c1, c2 = st.columns(2)
        p1 = c1.selectbox("參數一", names, index=names.index("ATR停損倍數"))
        v1 = c1.text_input("數值", "2.0, 2.5, 3.0", key="grid_v1")
        p2 = c2.selectbox("參數二", ["（不使用）"] + names, index=1 + names.index("ATR停損緩衝%"))
        v2 = c2.text_input("數值 ", "0, 5, 10", key="grid_v2")
        go_grid = st.form_submit_button("開始比較", type="primary")

    if go_grid:
        try:
            spec = {p1: B.parse_values(v1, B.PARAM_REGISTRY[p1][2])}
            if p2 != "（不使用）":
                if p2 == p1:
                    raise ValueError("兩個參數不能相同")
                spec[p2] = B.parse_values(v2, B.PARAM_REGISTRY[p2][2])
            if any(len(v) == 0 for v in spec.values()):
                raise ValueError("請輸入數值")
            with st.spinner("正在跑參數組合…"):
                table = B.run_grid(bars, bench_bars, cfg, settings, start, end, spec)
            st.session_state["grid_result"] = (table, list(spec), B.disabled_params(list(spec), cfg))
        except ValueError as e:
            st.error(f"無法執行：{e}")

    if "grid_result" in st.session_state:
        table, pnames, disabled = st.session_state["grid_result"]
        for d in disabled:
            st.warning(f"「{d}」對應的機制目前沒有勾選，改變這個參數不會影響結果。")
        show = table[pnames + ["總報酬%", "年化報酬%", "最大回撤%", "Calmar", "勝率%", "交易次數", "獲利因子"]].copy()
        base_calmar = perf_base["Calmar"]
        show["Calmar勝過基準"] = ["✓" if (not np.isnan(v) and not np.isnan(base_calmar) and v > base_calmar) else ""
                              for v in show["Calmar"]]
        show = show.sort_values("Calmar", ascending=False, na_position="last")
        st.caption(f"{BL.NAME}：總報酬 {pct(perf_base['總報酬%'])}、最大回撤 {pct(perf_base['最大回撤%'])}、"
                   f"Calmar {num(perf_base['Calmar'])}　·　{int((show['Calmar勝過基準'] == '✓').sum())} / {len(show)} 組勝過基準")
        best = show["Calmar"].max()
        styled = (show.style
                  .format({"總報酬%": "{:+.2f}", "年化報酬%": "{:+.2f}", "最大回撤%": "{:.2f}",
                           "Calmar": "{:.2f}", "勝率%": "{:.1f}", "獲利因子": "{:.2f}"}, na_rep="—")
                  .apply(lambda r: ["background-color:rgba(239,83,80,0.14);font-weight:600"
                                    if r["Calmar"] == best else "" for _ in r], axis=1))
        st.dataframe(styled, hide_index=True)
        st.caption("依 Calmar（年化報酬 ÷ 最大回撤）由高到低排列，最高者以淺紅標示。")
        st.download_button("匯出比較表 CSV", show.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{ticker}_param_compare.csv", mime="text/csv")

# ---------- 規則說明 ----------
with tab_rules:
    st.markdown("""
**成交方式**
- 進場、訊號出場、減半倉、回補：K棒收盤判斷，下一根K棒開盤成交（週線＝下週第一個交易日開盤）。
- 價格停損（ATR、固定、分批、移動停損、移動停利）：盤中最低價碰到就以停損價成交；開盤已跳空低於停損價則以開盤價成交。
- 移動停損的「最高點」算到前一根K棒為止；ATR 停損的 ATR 在進場當下固定。
- 每次進場用「當時總資產 × 倉位比例」買進，零股計算；沒用到的現金不計息。

**出場優先順序（同一根K棒）**
ATR／固定／分批停損 → 移動停利／移動停損 → 大盤惡化 → 跌破趨勢均線 → TSI破零 → 均線死叉 → TSI減半倉

**機制細節**
- 多個停損同時勾選：取較高（較近）的停損價。分批停損會取代單一 ATR 停損，緩衝 % 同樣套用。
- 分批停損：第一段賣 50%，第二段賣剩餘。勾選回補時，只在第一段之後、第二段之前，TSI 藍線上穿信號線且短均在長均之上，才回補到原倉位，並以回補價與當下 ATR 重算停損。
- TSI 減半倉：每次持倉只減一次，之後不加回。
- 大盤惡化：只在持倉中出場，不阻擋進場（要阻擋進場請勾「大盤多頭才進場」）。
- 訊號出場都以「穿越當天」判斷（例如死叉＝短均線今天跌破長均線）。
- 週線模式下所有指標都用週K計算，包含大盤 EMA。
- 部位歸零時，進場價、停損價、最高點、分批與減半狀態全部重置。

**審計策略（基準）**
- 與主頁的交易訊號邏輯完全相同：RSI／BIAS／布林下軌分 30%、60%、100% 加碼，MA60 分多空區，均價 −15% 停損並冷卻。
- 同樣的日期區間、手續費、證交稅、滑價，訊號隔日開盤成交；倉位依自身分段，不套用側邊欄的倉位比例。
- 一律用日線計算；均價與冷卻狀態從資料第一天開始累積，與主頁一致。
- 加碼時買進「增加的部位 × 當時總資產」，目標歸零時全部賣出。

**資料**
- FinMind 免費帳號無還原股價，程式以原始日K＋分割表還原；00631L 不配息，結果正確。其他配息股未還原除權息。
- 買進持有：回測第一根K棒開盤全額買進並持有到最後，同樣扣手續費與滑價。
""")
