# -*- coding: utf-8 -*-
"""績效計算、買進持有對照、參數比較。"""
import copy
import itertools
import math

import numpy as np
import pandas as pd

from .strategy import build_indicators, run_strategy


# ============================================================
# 績效指標
# ============================================================
def performance(equity, initial_cash, trades=None, shares=None):
    """回傳績效字典。equity 為每根K棒的總資產。"""
    eq = equity.dropna()
    out = {"總報酬%": math.nan, "年化報酬%": math.nan, "最大回撤%": math.nan, "Calmar": math.nan,
           "勝率%": math.nan, "交易次數": 0, "獲利因子": math.nan, "持倉時間%": math.nan}
    if eq.empty:
        return out
    final = eq.iloc[-1]
    out["總報酬%"] = (final / initial_cash - 1) * 100
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if years > 0 and final > 0:
        out["年化報酬%"] = ((final / initial_cash) ** (1 / years) - 1) * 100
    dd = eq / eq.cummax() - 1
    out["最大回撤%"] = dd.min() * 100
    if out["最大回撤%"] < 0 and not math.isnan(out["年化報酬%"]):
        out["Calmar"] = out["年化報酬%"] / abs(out["最大回撤%"])

    if trades is not None:
        closed = trades[~trades["is_open"]]
        out["交易次數"] = int(len(closed))
        if len(closed):
            out["勝率%"] = (closed["pnl"] > 0).mean() * 100
            gain = closed.loc[closed["pnl"] > 0, "pnl"].sum()
            loss = -closed.loc[closed["pnl"] < 0, "pnl"].sum()
            out["獲利因子"] = gain / loss if loss > 0 else (math.inf if gain > 0 else math.nan)
    if shares is not None and len(shares):
        out["持倉時間%"] = (shares > 0).mean() * 100
    return out


def drawdown(equity):
    eq = equity.dropna()
    return (eq / eq.cummax() - 1) * 100


def buy_and_hold(ind, settings, start, end):
    """第一根K棒開盤全額買進並持有到最後（同樣扣手續費與滑價）。"""
    win = ind[(ind.index >= pd.Timestamp(start)) & (ind.index <= pd.Timestamp(end))]
    if win.empty:
        return pd.Series(dtype=float)
    fee = settings.fee_rate / 100 * settings.fee_discount
    px = win["Open"].iloc[0] * (1 + settings.slippage / 100)
    qty = int(settings.initial_cash // (px * (1 + fee)))
    cash = settings.initial_cash - qty * px * (1 + fee)
    return cash + qty * win["Close"]


# ============================================================
# 參數比較
# ============================================================
# 可比較的參數：顯示名稱 → (所屬設定, 欄位名稱, 型別, 需要的機制開關)
PARAM_REGISTRY = {
    "ATR停損倍數": ("cfg", "atr_mult", float, "use_atr_stop"),
    "ATR停損緩衝%": ("cfg", "atr_buffer", float, None),
    "固定停損%": ("cfg", "fixed_stop_pct", float, "use_fixed_stop"),
    "分批停損①倍數": ("cfg", "scale_mult1", float, "use_scale_stop"),
    "分批停損②倍數": ("cfg", "scale_mult2", float, "use_scale_stop"),
    "移動停損回落%": ("cfg", "trail_pct", float, "use_trail_stop"),
    "移動停利啟動%": ("cfg", "trail_tp_trigger", float, "use_trail_tp"),
    "移動停利回落%": ("cfg", "trail_tp_pct", float, "use_trail_tp"),
    "ATR移動停利倍數": ("cfg", "atr_trail_mult", float, "use_atr_trail"),
    "短均線": ("cfg", "ma_short", int, None),
    "長均線": ("cfg", "ma_long", int, None),
    "ADX門檻": ("cfg", "adx_th", float, "use_adx"),
    "趨勢均線N": ("cfg", "trend_n", int, None),
    "倉位比例%": ("settings", "position_pct", float, None),
}

# 會影響指標計算的欄位（變動時需要重算指標）
_INDICATOR_FIELDS = ("ma_type", "ma_short", "ma_long", "trend_n", "tsi_r", "tsi_s", "tsi_sig",
                     "adx_len", "atr_len", "mkt_ema")

MAX_COMBOS = 100


def parse_values(text, typ):
    """把「2.0, 2.5, 3.0」這類文字轉成數值清單。"""
    vals = []
    for part in str(text).replace("，", ",").replace("、", ",").split(","):
        part = part.strip()
        if part:
            try:
                vals.append(typ(float(part)))
            except ValueError:
                raise ValueError(f"「{part}」不是數字") from None
    return list(dict.fromkeys(vals))  # 去除重複、保留順序


def param_has_effect(name, cfg):
    """這個參數在目前勾選的機制下，改了會不會影響結果。"""
    if name == "ATR停損倍數":
        return cfg.use_atr_stop and not cfg.use_scale_stop   # 勾分批停損時，單一 ATR 停損不使用
    if name == "ATR停損緩衝%":
        return cfg.use_atr_stop or cfg.use_scale_stop
    if name in ("短均線", "長均線"):
        return cfg.use_ma_entry or cfg.exit_dead_cross or (cfg.use_scale_stop and cfg.use_readd)
    if name == "趨勢均線N":
        return cfg.use_trend or cfg.exit_trend
    need = PARAM_REGISTRY[name][3]
    return True if need is None else bool(getattr(cfg, need))


def disabled_params(names, cfg):
    """回傳「機制沒有勾選、所以改了也不會影響結果」的參數名稱。"""
    return [n for n in names if not param_has_effect(n, cfg)]


def run_grid(bars, bench_bars, cfg, settings, start, end, param_values):
    """param_values：{參數名稱: [數值, ...]}。回傳比較表 DataFrame。"""
    names = list(param_values.keys())
    combos = list(itertools.product(*[param_values[n] for n in names]))
    if len(combos) > MAX_COMBOS:
        raise ValueError(f"組合數 {len(combos)} 超過上限 {MAX_COMBOS}，請減少數值")

    ind_cache = {}
    rows = []
    for combo in combos:
        c = copy.deepcopy(cfg)
        s = copy.deepcopy(settings)
        for name, val in zip(names, combo):
            target, attr, _, _ = PARAM_REGISTRY[name]
            setattr(c if target == "cfg" else s, attr, val)
        key = tuple(getattr(c, f) for f in _INDICATOR_FIELDS)
        if key not in ind_cache:
            ind_cache[key] = build_indicators(bars, bench_bars, c)
        res = run_strategy(ind_cache[key], c, s, start, end)
        perf = performance(res["equity"], s.initial_cash, res["trades"], res["shares"])
        row = dict(zip(names, combo))
        row.update(perf)
        rows.append(row)
    return pd.DataFrame(rows)
