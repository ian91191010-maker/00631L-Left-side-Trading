# -*- coding: utf-8 -*-
"""策略引擎。

時間規則（避免前視偏差）：
- 所有進場、訊號出場都在「K棒收盤後」判斷，預設在「下一根K棒開盤價」成交。
  週線模式下 = 下週第一個交易日開盤。
- 價格類停損（ATR、固定 %、分批、移動停損、移動停利）在K棒盤中觸價成交：
  最低價碰到停損價就以停損價成交；若開盤已跳空低於停損價，則以開盤價成交。
- 移動停損使用的「最高點」只算到前一根K棒為止，避免同一根K棒「先創高又觸價」的偷看。

同一根K棒多個出場條件同時成立時的優先順序：
  ATR／固定／分批停損 → 移動停利／移動停損 → 大盤惡化 → 跌破趨勢均線 → TSI破零 → 均線死亡交叉 → TSI減半倉

新增機制的方法：
- 進場條件：寫一個 cond_xxx(ind, cfg) 回傳 True/False 序列，加進 ENTRY_RULES。
- 訊號出場：寫一個 exit_xxx(ind, cfg) 回傳 True/False 序列，依優先順序加進 SIGNAL_EXITS。
- 停損停利：寫一個回傳價格的函式，在 stop_levels() 裡加入。
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ti


# ============================================================
# 參數設定
# ============================================================
@dataclass
class StrategyConfig:
    # ---- 1. 進場條件（勾選者需同時成立）----
    use_ma_entry: bool = True          # 均線多頭／黃金交叉
    ma_entry_mode: str = "state"       # "state"=短均線在長均線之上；"cross"=黃金交叉當天
    ma_type: str = "SMA"               # SMA 或 EMA
    ma_short: int = 21
    ma_long: int = 55
    use_tsi_pos: bool = True           # TSI > 0
    use_tsi_above_sig: bool = True     # TSI 藍線 > 信號線
    tsi_r: int = 25
    tsi_s: int = 13
    tsi_sig: int = 13
    use_adx: bool = False              # ADX > 門檻
    adx_len: int = 14
    adx_th: float = 20.0
    use_trend: bool = False            # 收盤 > N 日均線
    trend_n: int = 120
    use_mkt_filter: bool = False       # 大盤收盤 > 大盤 EMA 才進場
    mkt_ema: int = 200

    # ---- 2. 停損 ----
    atr_len: int = 14
    use_atr_stop: bool = True          # 停損價 = (進場價 − ATR × 倍數) × (1 − 緩衝%)
    atr_mult: float = 2.5
    atr_buffer: float = 5.0            # %
    use_fixed_stop: bool = False       # 停損價 = 進場價 × (1 − X%)
    fixed_stop_pct: float = 10.0
    use_scale_stop: bool = False       # 分批停損（勾選時取代單一 ATR 停損）
    scale_mult1: float = 2.5           # 第一段：賣 50%
    scale_mult2: float = 4.0           # 第二段：賣剩餘
    use_readd: bool = False            # 第一段停損後，TSI 藍穿紅且均線多頭時回補

    # ---- 3. 停利 ----
    use_trail_stop: bool = False       # 從持倉最高點回落 X% 出場
    trail_pct: float = 15.0
    use_trail_tp: bool = False         # 獲利達 A% 後啟動，從最高點回落 B% 出場
    trail_tp_trigger: float = 30.0
    trail_tp_pct: float = 10.0
    use_atr_trail: bool = False        # 最高點 − ATR × 倍數
    atr_trail_mult: float = 3.0

    # ---- 4. 訊號出場（可複選）----
    exit_mkt: bool = True              # 大盤收盤下穿其 EMA（只出場，不擋進場）
    exit_trend: bool = False           # 收盤跌破趨勢均線（組合 B 使用）
    exit_tsi_zero: bool = True         # TSI 藍線跌破 0
    exit_dead_cross: bool = True       # 均線死亡交叉
    exit_tsi_half: bool = False        # TSI 下穿信號線 → 減半倉（每次持倉只減一次）


@dataclass
class TradeSettings:
    initial_cash: float = 1_000_000.0
    position_pct: float = 100.0        # 每次進場使用總資產的比例 %
    fee_rate: float = 0.1425           # 手續費 %
    fee_discount: float = 1.0          # 折數，0.6 = 6 折
    tax_rate: float = 0.1              # 證交稅 %（ETF 0.1、個股 0.3），只在賣出時收
    slippage: float = 0.05             # 滑價 %
    exec_price: str = "next_open"      # "next_open"=訊號隔日開盤；"close"=訊號當日收盤（有前視偏差）


# ============================================================
# 指標計算
# ============================================================
def _align(series, index):
    """把大盤資料對齊到標的的日期（沒有的日子沿用前一天）。"""
    return series.reindex(series.index.union(index)).ffill().reindex(index)


def build_indicators(bars, bench_bars, cfg):
    """依設定計算所有指標，回傳一張表。bench_bars 可以是 None。"""
    df = bars[["Open", "High", "Low", "Close", "Volume"]].copy()
    c = df["Close"]
    df["MA_S"] = ti.moving_average(c, cfg.ma_short, cfg.ma_type)
    df["MA_L"] = ti.moving_average(c, cfg.ma_long, cfg.ma_type)
    df["TREND_MA"] = ti.moving_average(c, cfg.trend_n, cfg.ma_type)
    df["TSI"], df["TSI_SIG"] = ti.tsi(c, cfg.tsi_r, cfg.tsi_s, cfg.tsi_sig)
    df["ADX"], df["PDI"], df["MDI"] = ti.adx(df["High"], df["Low"], c, cfg.adx_len, cfg.adx_len)
    df["ATR"] = ti.atr(df["High"], df["Low"], c, cfg.atr_len)
    if bench_bars is not None and not bench_bars.empty:
        bc = bench_bars["Close"]
        df["MKT"] = _align(bc, df.index)
        df["MKT_EMA"] = _align(ti.ema(bc, cfg.mkt_ema), df.index)
    else:
        df["MKT"] = np.nan
        df["MKT_EMA"] = np.nan
    return df


# ============================================================
# 1. 進場條件（每個都是獨立函式）
# ============================================================
def cond_ma(ind, cfg):
    above = ind["MA_S"] > ind["MA_L"]
    if cfg.ma_entry_mode == "cross":
        return ti.cross_over(ind["MA_S"], ind["MA_L"])
    return above


def cond_tsi_pos(ind, cfg):
    return ind["TSI"] > 0


def cond_tsi_above_signal(ind, cfg):
    return ind["TSI"] > ind["TSI_SIG"]


def cond_adx(ind, cfg):
    return ind["ADX"] > cfg.adx_th


def cond_trend(ind, cfg):
    return ind["Close"] > ind["TREND_MA"]


def cond_market(ind, cfg):
    return ind["MKT"] > ind["MKT_EMA"]


ENTRY_RULES = [
    ("use_ma_entry", cond_ma, "均線"),
    ("use_tsi_pos", cond_tsi_pos, "TSI>0"),
    ("use_tsi_above_sig", cond_tsi_above_signal, "TSI藍>紅"),
    ("use_adx", cond_adx, "ADX門檻"),
    ("use_trend", cond_trend, "收盤>趨勢均線"),
    ("use_mkt_filter", cond_market, "大盤多頭"),
]


def entry_signal(ind, cfg):
    """所有勾選的進場條件同時成立才為 True。沒有勾任何條件時永遠不進場。"""
    sig = None
    for flag, func, _ in ENTRY_RULES:
        if getattr(cfg, flag):
            s = func(ind, cfg).fillna(False).astype(bool)
            sig = s if sig is None else (sig & s)
    if sig is None:
        return pd.Series(False, index=ind.index)
    return sig


def enabled_entry_labels(cfg):
    return [label for flag, _, label in ENTRY_RULES if getattr(cfg, flag)]


# ============================================================
# 4. 訊號出場（依優先順序排列）
# ============================================================
def exit_market(ind, cfg):
    """大盤收盤價下穿其 EMA。"""
    return ti.cross_under(ind["MKT"], ind["MKT_EMA"])


def exit_trend_ma(ind, cfg):
    """收盤價跌破趨勢均線。"""
    return ti.cross_under(ind["Close"], ind["TREND_MA"])


def exit_tsi_zero(ind, cfg):
    """TSI 藍線跌破 0。"""
    return ti.cross_under(ind["TSI"], 0)


def exit_dead_cross(ind, cfg):
    """短均線跌破長均線。"""
    return ti.cross_under(ind["MA_S"], ind["MA_L"])


SIGNAL_EXITS = [
    ("exit_mkt", exit_market, "大盤惡化"),
    ("exit_trend", exit_trend_ma, "跌破趨勢均線"),
    ("exit_tsi_zero", exit_tsi_zero, "TSI破零"),
    ("exit_dead_cross", exit_dead_cross, "均線死叉"),
]


def signal_tsi_half(ind, cfg):
    """TSI 下穿信號線 → 減半倉。"""
    return ti.cross_under(ind["TSI"], ind["TSI_SIG"])


def signal_readd(ind, cfg):
    """TSI 藍線上穿信號線，且短均線在長均線之上 → 回補。"""
    return ti.cross_over(ind["TSI"], ind["TSI_SIG"]) & (ind["MA_S"] > ind["MA_L"])


# ============================================================
# 2. / 3. 停損停利價格（每個都是獨立函式）
# ============================================================
def stop_atr(ref_price, atr_value, mult, buffer_pct):
    """ATR 停損：(進場價 − ATR × 倍數) × (1 − 緩衝%)。ATR 在進場當下固定。"""
    return (ref_price - atr_value * mult) * (1 - buffer_pct / 100)


def stop_fixed(ref_price, pct):
    """固定百分比停損。"""
    return ref_price * (1 - pct / 100)


def stop_trailing(peak, pct):
    """移動停損：從持倉最高點回落 X%。"""
    return peak * (1 - pct / 100)


def stop_trailing_tp(peak, avg_cost, trigger_pct, pct):
    """移動停利：最高點達均價 +A% 後啟動，從最高點回落 B%。未啟動時回傳 NaN。"""
    if peak >= avg_cost * (1 + trigger_pct / 100):
        return peak * (1 - pct / 100)
    return math.nan


def stop_atr_trailing(peak, atr_value, mult):
    """ATR 移動停利：最高點 − ATR × 倍數（ATR 用前一根K棒的值）。"""
    return peak - atr_value * mult


# ============================================================
# 引擎
# ============================================================
class _Engine:
    """逐根K棒模擬。所有持倉狀態集中在這裡，部位歸零時一律呼叫 _reset()。"""

    def __init__(self, ind, cfg, settings):
        self.ind = ind
        self.cfg = cfg
        self.s = settings
        self.idx = ind.index
        self.O = ind["Open"].to_numpy(float)
        self.H = ind["High"].to_numpy(float)
        self.L = ind["Low"].to_numpy(float)
        self.C = ind["Close"].to_numpy(float)
        self.ATR = ind["ATR"].to_numpy(float)

        self.fee = settings.fee_rate / 100 * settings.fee_discount
        self.tax = settings.tax_rate / 100
        self.slip = settings.slippage / 100

        self.cash = float(settings.initial_cash)
        self.shares = 0
        self.trades = []
        self.fills = []
        self.trade = None
        self._reset()

    # ---------- 狀態 ----------
    def _reset(self):
        """部位歸零時強制重置所有狀態變數，避免狀態卡住導致不再進場。"""
        self.ref = math.nan          # 計算停損用的進場價（回補後更新）
        self.atr0 = math.nan         # 進場當下固定的 ATR
        self.peak = math.nan         # 持倉期間最高價（到前一根K棒為止）
        self.avg_cost = 0.0          # 加權平均成本（不含手續費）
        self.stage = 0               # 分批停損：0=未觸發，1=已賣出第一段
        self.halved = False          # TSI 減半倉是否已執行
        self.half_sold = 0           # 第一段停損賣出的股數（回補用）

    # ---------- 成交 ----------
    def _buy(self, i, raw_price, qty, reason):
        px = raw_price * (1 + self.slip)
        cost = px * qty * (1 + self.fee)
        self.cash -= cost
        self.shares += qty
        if self.trade is None:
            self.trade = {"entry_i": i, "buy_cost": 0.0, "buy_value": 0.0, "buy_qty": 0,
                          "sell_net": 0.0, "sell_value": 0.0, "sell_qty": 0, "reasons": []}
        t = self.trade
        t["buy_cost"] += cost
        t["buy_value"] += px * qty
        t["buy_qty"] += qty
        if t["buy_qty"] > qty:
            t["reasons"].append(reason)
        self.fills.append({"Date": self.idx[i], "side": "買進", "qty": qty, "price": px, "reason": reason})
        return px

    def _sell(self, i, raw_price, qty, reason):
        qty = min(qty, self.shares)
        if qty <= 0:
            return
        px = raw_price * (1 - self.slip)
        net = px * qty * (1 - self.fee - self.tax)
        self.cash += net
        self.shares -= qty
        t = self.trade
        t["sell_net"] += net
        t["sell_value"] += px * qty
        t["sell_qty"] += qty
        t["reasons"].append(reason)
        self.fills.append({"Date": self.idx[i], "side": "賣出", "qty": qty, "price": px, "reason": reason,
                           "full": self.shares == 0})
        if self.shares == 0:
            self._close_trade(i)
            self._reset()

    def _close_trade(self, i, is_open=False, mark_price=None):
        t = self.trade
        sell_net, sell_value, sell_qty = t["sell_net"], t["sell_value"], t["sell_qty"]
        if is_open:  # 期末尚未出場：以最後收盤價（扣賣出成本）計算未實現損益
            net = mark_price * self.shares * (1 - self.fee - self.tax)
            sell_net += net
            sell_value += mark_price * self.shares
            sell_qty += self.shares
        pnl = sell_net - t["buy_cost"]
        self.trades.append({
            "entry_date": self.idx[t["entry_i"]],
            "exit_date": pd.NaT if is_open else self.idx[i],
            "bars": i - t["entry_i"],
            "entry_price": t["buy_value"] / t["buy_qty"],
            "exit_price": sell_value / sell_qty if sell_qty else math.nan,
            "pnl": pnl,
            "ret": pnl / t["buy_cost"] * 100,
            "reason": "持有中" if is_open else " → ".join(t["reasons"]),
            "is_open": is_open,
        })
        self.trade = None

    # ---------- 動作 ----------
    def enter(self, i, raw_price, atr_value):
        budget = self.cash * self.s.position_pct / 100
        px = raw_price * (1 + self.slip)
        qty = int(budget // (px * (1 + self.fee)))
        if qty < 1:
            return
        fill = self._buy(i, raw_price, qty, "進場")
        self.ref = fill
        self.atr0 = atr_value
        self.peak = fill
        self.avg_cost = fill
        self.stage = 0
        self.halved = False

    def readd(self, i, raw_price, atr_value):
        px = raw_price * (1 + self.slip)
        qty = min(self.half_sold, int(self.cash // (px * (1 + self.fee))))
        if qty < 1:
            return
        old = self.shares
        fill = self._buy(i, raw_price, qty, "回補")
        self.avg_cost = (self.avg_cost * old + fill * qty) / (old + qty)
        self.ref = fill
        self.atr0 = atr_value
        self.peak = max(self.peak, fill)
        self.stage = 0
        self.half_sold = 0

    def sell_half(self, i, raw_price, reason):
        qty = self.shares // 2 or self.shares
        self._sell(i, raw_price, qty, reason)
        return qty

    # ---------- 停損停利價位 ----------
    def stop_levels(self, i):
        """回傳 (第一組, 第二組)，每組是 [(價位, 動作, 名稱)]，由高到低排列。"""
        cfg = self.cfg
        g1, g2 = [], []
        if cfg.use_scale_stop:
            if self.stage == 0:
                g1.append((stop_atr(self.ref, self.atr0, cfg.scale_mult1, cfg.atr_buffer), "half", "分批停損①"))
            g1.append((stop_atr(self.ref, self.atr0, cfg.scale_mult2, cfg.atr_buffer), "all", "分批停損②"))
        elif cfg.use_atr_stop:
            g1.append((stop_atr(self.ref, self.atr0, cfg.atr_mult, cfg.atr_buffer), "all", "ATR停損"))
        if cfg.use_fixed_stop:
            g1.append((stop_fixed(self.ref, cfg.fixed_stop_pct), "all", "固定停損"))

        if cfg.use_trail_tp:
            g2.append((stop_trailing_tp(self.peak, self.avg_cost, cfg.trail_tp_trigger, cfg.trail_tp_pct),
                       "all", "移動停利"))
        if cfg.use_atr_trail and i > 0:
            g2.append((stop_atr_trailing(self.peak, self.ATR[i - 1], cfg.atr_trail_mult), "all", "ATR移動停利"))
        if cfg.use_trail_stop:
            g2.append((stop_trailing(self.peak, cfg.trail_pct), "all", "移動停損"))

        def clean(g):
            return sorted([x for x in g if not math.isnan(x[0])], key=lambda x: -x[0])
        return clean(g1), clean(g2)

    def nearest_stop(self, i):
        g1, g2 = self.stop_levels(i)
        levels = [x[0] for x in g1 + g2]
        return max(levels) if levels else math.nan

    def process_stops(self, i, at_open):
        """依優先順序檢查停損。at_open=True 只檢查開盤跳空；False 檢查盤中最低價。"""
        trigger = self.O[i] if at_open else self.L[i]
        for group in (0, 1):
            while self.shares > 0:
                hit = None
                for level, action, label in self.stop_levels(i)[group]:
                    if trigger <= level:
                        hit = (level, action, label)
                        break
                if hit is None:
                    break
                level, action, label = hit
                fill = self.O[i] if self.O[i] <= level else level
                if action == "half" and self.stage == 0:
                    sold = self.sell_half(i, fill, label)
                    if self.shares > 0:          # 只剩 1 股時會直接全部賣出，此時狀態已重置
                        self.half_sold = sold
                        self.stage = 1
                else:
                    self._sell(i, fill, self.shares, label)
            if self.shares == 0:
                return


def run_strategy(ind, cfg, settings, start=None, end=None):
    """執行回測。回傳 dict：equity、shares、stop_line（Series）、trades、fills（DataFrame）。"""
    eng = _Engine(ind, cfg, settings)
    idx = ind.index
    n = len(ind)
    start = pd.Timestamp(start) if start is not None else idx[0]
    end = pd.Timestamp(end) if end is not None else idx[-1]
    in_win = ((idx >= start) & (idx <= end))

    entry = entry_signal(ind, cfg).to_numpy()
    exits = [(label, func(ind, cfg).fillna(False).to_numpy())
             for flag, func, label in SIGNAL_EXITS if getattr(cfg, flag)]
    half_sig = signal_tsi_half(ind, cfg).fillna(False).to_numpy() if cfg.exit_tsi_half else None
    readd_sig = (signal_readd(ind, cfg).fillna(False).to_numpy()
                 if (cfg.use_scale_stop and cfg.use_readd) else None)
    needs_atr = cfg.use_atr_stop or cfg.use_scale_stop or cfg.use_atr_trail
    at_close = settings.exec_price == "close"

    equity = np.full(n, np.nan)
    shares_arr = np.zeros(n)
    stop_line = np.full(n, np.nan)
    pending = None
    last_i = None

    def execute(order, i, price):
        kind = order["kind"]
        if kind == "entry" and eng.shares == 0:
            eng.enter(i, price, order["atr"])
        elif kind == "readd" and eng.shares > 0 and eng.stage == 1:
            eng.readd(i, price, order["atr"])
        elif kind == "exit" and eng.shares > 0:
            eng._sell(i, price, eng.shares, order["reason"])
        elif kind == "half" and eng.shares > 0:
            eng.sell_half(i, price, order["reason"])
            if eng.shares > 0:
                eng.halved = True

    for i in range(n):
        if not in_win[i]:
            continue
        last_i = i

        # A. 開盤：先執行前一根K棒收盤時決定的「買進」類單
        if pending and pending["kind"] in ("entry", "readd"):
            execute(pending, i, eng.O[i])
            pending = None
        # B. 開盤跳空觸發停損（優先於訊號出場）
        if eng.shares > 0:
            eng.process_stops(i, at_open=True)
        # C. 開盤執行訊號出場／減半
        if pending:
            execute(pending, i, eng.O[i])
            pending = None
        # D. 盤中觸價停損停利
        if eng.shares > 0:
            stop_line[i] = eng.nearest_stop(i)
            eng.process_stops(i, at_open=False)
        # E. 更新持倉最高價（供下一根K棒的移動停損使用）
        if eng.shares > 0:
            eng.peak = max(eng.peak, eng.H[i])

        # F. 收盤：判斷下一步
        order = None
        if eng.shares > 0:
            for label, arr in exits:
                if arr[i]:
                    order = {"kind": "exit", "reason": label}
                    break
            if order is None and half_sig is not None and not eng.halved and half_sig[i]:
                order = {"kind": "half", "reason": "TSI減半"}
            if order is None and readd_sig is not None and eng.stage == 1 and readd_sig[i]:
                order = {"kind": "readd", "atr": eng.ATR[i]}
        else:
            can_fill = at_close or (i + 1 < n and in_win[i + 1])
            atr_ok = (not needs_atr) or not math.isnan(eng.ATR[i])
            if entry[i] and can_fill and atr_ok:
                order = {"kind": "entry", "atr": eng.ATR[i]}

        if order is not None:
            if at_close:
                execute(order, i, eng.C[i])
            else:
                pending = order

        equity[i] = eng.cash + eng.shares * eng.C[i]
        shares_arr[i] = eng.shares

    if eng.shares > 0 and last_i is not None:
        eng._close_trade(last_i, is_open=True, mark_price=eng.C[last_i])

    sl = pd.Series(in_win, index=idx)
    trades = pd.DataFrame(eng.trades, columns=["entry_date", "exit_date", "bars", "entry_price", "exit_price",
                                               "pnl", "ret", "reason", "is_open"])
    trades["is_open"] = trades["is_open"].astype(bool)
    fills = pd.DataFrame(eng.fills, columns=["Date", "side", "qty", "price", "reason", "full"])
    return {
        "equity": pd.Series(equity, index=idx)[sl],
        "shares": pd.Series(shares_arr, index=idx)[sl],
        "stop_line": pd.Series(stop_line, index=idx)[sl],
        "trades": trades,
        "fills": fills,
    }
