# -*- coding: utf-8 -*-
"""Plotly 圖表。顏色依台股習慣：漲／獲利為紅，跌／虧損為綠。"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP = "#ef5350"
DOWN = "#26a69a"
PROFIT_FILL = "rgba(239, 83, 80, 0.14)"   # 獲利：淺紅
LOSS_FILL = "rgba(38, 166, 154, 0.16)"    # 虧損：淺綠
TSI_BLUE = "#2962FF"
SIGNAL_RED = "#FF5252"


def _no_hover(**kw):
    kw["hoverinfo"] = "skip"
    return kw


def price_chart(view, stop_line, fills, trades, cfg, show_tsi=True, show_adx=True, daily=True):
    """K棒 + 均線 + 進出場標記 + 停損線 + 持倉底色，下方可加 TSI、ADX 子圖。"""
    rows = 1 + int(show_tsi) + int(show_adx)
    heights = {1: [1.0], 2: [0.72, 0.28], 3: [0.6, 0.2, 0.2]}[rows]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=heights)
    idx = view.index

    # K棒：細邊框、實心；滑過只顯示K棒資訊
    chg = view["Close"].pct_change() * 100
    hover = [f"開 {o:.2f}<br>高 {h:.2f}<br>低 {l:.2f}<br>收 {c:.2f}" + ("" if pd.isna(g) else f"（{g:+.2f}%）")
             for o, h, l, c, g in zip(view["Open"], view["High"], view["Low"], view["Close"], chg)]
    fig.add_trace(go.Candlestick(
        x=idx, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
        name="K線", whiskerwidth=0, hovertext=hover, hoverinfo="text",
        increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN)), row=1, col=1)

    fig.add_trace(go.Scatter(x=idx, y=view["MA_S"], name=f"{cfg.ma_type}{cfg.ma_short}", mode="lines",
                             line=dict(color="#F9A825", width=1.2), **_no_hover()), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=view["MA_L"], name=f"{cfg.ma_type}{cfg.ma_long}", mode="lines",
                             line=dict(color="#7E57C2", width=1.2), **_no_hover()), row=1, col=1)
    if cfg.use_trend or cfg.exit_trend:
        fig.add_trace(go.Scatter(x=idx, y=view["TREND_MA"], name=f"{cfg.ma_type}{cfg.trend_n}", mode="lines",
                                 line=dict(color="#8D6E63", width=1.2, dash="dash"), **_no_hover()), row=1, col=1)

    sl = stop_line.reindex(idx)
    if sl.notna().any():
        fig.add_trace(go.Scatter(x=idx, y=sl, name="停損線", mode="lines", connectgaps=False,
                                 line=dict(color="#E53935", width=1, dash="dot", shape="hv"), **_no_hover()),
                      row=1, col=1)

    if not fills.empty:
        f = fills[(fills["Date"] >= idx[0]) & (fills["Date"] <= idx[-1])]
        lows = view["Low"].reindex(f["Date"]).to_numpy()
        highs = view["High"].reindex(f["Date"]).to_numpy()
        buy = (f["side"] == "買進").to_numpy()
        full = f["full"].fillna(False).astype(bool).to_numpy()
        sell_full = (~buy) & full
        sell_part = (~buy) & (~full)
        fig.add_trace(go.Scatter(x=f["Date"][buy], y=lows[buy] * 0.985, mode="markers", name="買進",
                                 marker=dict(symbol="triangle-up", size=10, color="#FFD700",
                                             line=dict(width=1, color="#555")), **_no_hover()), row=1, col=1)
        fig.add_trace(go.Scatter(x=f["Date"][sell_full], y=highs[sell_full] * 1.015, mode="markers", name="出場",
                                 marker=dict(symbol="triangle-down", size=10, color="#1E90FF",
                                             line=dict(width=1, color="#555")), **_no_hover()), row=1, col=1)
        if sell_part.any():
            fig.add_trace(go.Scatter(x=f["Date"][sell_part], y=highs[sell_part] * 1.015, mode="markers",
                                     name="部分賣出",
                                     marker=dict(symbol="triangle-down-open", size=10, color="#1E90FF",
                                                 line=dict(width=1.5)), **_no_hover()), row=1, col=1)

    r = 2
    if show_tsi:
        fig.add_trace(go.Scatter(x=idx, y=view["TSI"], name="TSI", mode="lines",
                                 line=dict(color=TSI_BLUE, width=1.3), **_no_hover()), row=r, col=1)
        fig.add_trace(go.Scatter(x=idx, y=view["TSI_SIG"], name="信號線", mode="lines",
                                 line=dict(color=SIGNAL_RED, width=1.1), **_no_hover()), row=r, col=1)
        fig.add_hline(y=0, line_width=0.8, line_color="#9E9E9E", row=r, col=1)
        fig.update_yaxes(title_text="TSI", row=r, col=1)
        r += 1
    if show_adx:
        fig.add_trace(go.Scatter(x=idx, y=view["ADX"], name="ADX", mode="lines",
                                 line=dict(color="#FB8C00", width=1.3), **_no_hover()), row=r, col=1)
        fig.add_hline(y=cfg.adx_th, line_dash="dot", line_width=0.8, line_color="#9E9E9E",
                      annotation_text=f"{cfg.adx_th:g}", annotation_position="right", row=r, col=1)
        fig.update_yaxes(title_text="ADX", row=r, col=1)

    # 持倉底色：獲利淺紅、虧損淺綠（必須在所有圖層建立後才加，子圖才會顯示）
    lo, hi = idx[0], idx[-1]
    for t in trades.itertuples():
        x0 = t.entry_date
        x1 = hi if pd.isna(t.exit_date) else t.exit_date
        if x1 < lo or x0 > hi:
            continue
        fig.add_vrect(x0=max(x0, lo), x1=min(x1, hi), fillcolor=PROFIT_FILL if t.ret > 0 else LOSS_FILL,
                      layer="below", line_width=0, row=1, col=1)

    if daily:  # 日線隱藏週末與休市日；週線不需要
        all_days = pd.date_range(lo, hi, freq="D")
        missing = [d.strftime("%Y-%m-%d") for d in all_days.difference(idx)]
        fig.update_xaxes(rangebreaks=[dict(values=missing)])
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor="#888")
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_layout(height={1: 520, 2: 660, 3: 780}[rows], xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10), dragmode="pan", hovermode="x",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    return fig


BASE_BLUE = "#1E88E5"


def equity_chart(equity, bh_equity, dd, bh_dd, base_equity=None, base_dd=None):
    """資金曲線（目前測試 vs 審計策略基準 vs 買進持有）＋回撤曲線。"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.65, 0.35])
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="目前測試", mode="lines",
                             line=dict(color=UP, width=1.6)), row=1, col=1)
    if base_equity is not None:
        fig.add_trace(go.Scatter(x=base_equity.index, y=base_equity, name="審計策略（基準）", mode="lines",
                                 line=dict(color=BASE_BLUE, width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name="買進持有", mode="lines",
                             line=dict(color="#9E9E9E", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="目前測試回撤", mode="lines", fill="tozeroy",
                             line=dict(color=UP, width=1), fillcolor="rgba(239, 83, 80, 0.18)"), row=2, col=1)
    if base_dd is not None:
        fig.add_trace(go.Scatter(x=base_dd.index, y=base_dd, name="基準回撤", mode="lines",
                                 line=dict(color=BASE_BLUE, width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=bh_dd.index, y=bh_dd, name="買進持有回撤", mode="lines",
                             line=dict(color="#9E9E9E", width=1)), row=2, col=1)
    fig.update_yaxes(title_text="總資產", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="回撤 %", ticksuffix="%", row=2, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    fig.update_traces(hovertemplate="%{y:,.2f}")
    return fig
