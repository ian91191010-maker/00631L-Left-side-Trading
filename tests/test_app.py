# -*- coding: utf-8 -*-
"""自動測試：不需要 FinMind Token、不連網路（用假資料代替 FinMind）。

執行方式（在專案根目錄）：
    pip install -r requirements.txt -r requirements-dev.txt
    python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bt import costs, f0 as F, finmind, gauges as G  # noqa: E402

TOKEN = "TEST_SECRET_TOKEN_123"


# ---------------------------------------------------------------- 假 FinMind
def _bdays(start, end):
    return pd.bdate_range(start, end)


def _walk(n, start, seed):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


def fake_rows(dataset, data_id, start, end):
    days = _bdays("2014-06-01", pd.Timestamp.today().normalize())
    if dataset == "TaiwanStockPrice":
        seed, base = (1, 20.0) if data_id == "00631L" else (2, 9000.0)
        d = days[days >= "2014-10-31"] if data_id == "00631L" else days
        c = _walk(len(d), base, seed)
        rows = [{"date": x.strftime("%Y-%m-%d"), "open": v, "max": v * 1.01, "min": v * 0.99, "close": v,
                 "Trading_Volume": 1000} for x, v in zip(d, c)]
    else:  # 台指期：只放近月一般時段
        c = _walk(len(days), 9000.0, 3)
        rows = []
        for x, v in zip(days, c):
            exp = F.third_wednesday(x.year, x.month)
            y, m = (x.year, x.month) if x <= exp else ((x + pd.offsets.MonthBegin(1)).year, (x + pd.offsets.MonthBegin(1)).month)
            rows.append({"date": x.strftime("%Y-%m-%d"), "contract_date": f"{y}{m:02d}", "trading_session": "position",
                         "open": v, "max": v * 1.01, "min": v * 0.99, "close": v, "volume": 100, "settlement_price": v})
    return [r for r in rows if str(start)[:10] <= r["date"] <= str(end)[:10]]


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p


class FakeSession:
    calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        FakeSession.calls.append(params)
        return FakeResponse(200, {"status": 200, "data": fake_rows(params["dataset"], params["data_id"],
                                                                   params["start_date"], params["end_date"])})


@pytest.fixture
def fake_finmind(monkeypatch):
    FakeSession.calls = []
    monkeypatch.setattr(finmind, "_get_session", lambda: FakeSession())
    return FakeSession


# ---------------------------------------------------------------- Token 不外洩
def test_http_error_message_has_no_token(monkeypatch):
    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            return FakeResponse(402, {"msg": f"quota exceeded for {TOKEN}", "status": 402})
    monkeypatch.setattr(finmind, "_get_session", lambda: S())
    with pytest.raises(finmind.FinMindError) as e:
        finmind.request("TaiwanStockPrice", "00631L", "2026-01-01", "2026-01-02", TOKEN)
    assert TOKEN not in str(e.value) and "402" in str(e.value)


def test_connection_error_message_has_no_token(monkeypatch):
    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            raise requests.exceptions.ConnectionError(f"{url}?token={TOKEN}")
    monkeypatch.setattr(finmind, "_get_session", lambda: S())
    with pytest.raises(finmind.FinMindError) as e:
        finmind.request("TaiwanStockPrice", "00631L", "2026-01-01", "2026-01-02", TOKEN)
    assert TOKEN not in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__  # 原始例外（含網址）不會被串在後面


def test_payload_error_message_masks_token(monkeypatch):
    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            return FakeResponse(200, {"msg": f"bad token {TOKEN}", "status": 400})
    monkeypatch.setattr(finmind, "_get_session", lambda: S())
    with pytest.raises(finmind.FinMindError) as e:
        finmind.request("TaiwanStockPrice", "00631L", "2026-01-01", "2026-01-02", TOKEN)
    assert TOKEN not in str(e.value)


def test_token_only_in_header(fake_finmind):
    finmind.request("TaiwanStockPrice", "00631L", "2026-01-01", "2026-01-05", TOKEN)
    assert all(TOKEN not in str(v) for v in fake_finmind.calls[-1].values())


# ---------------------------------------------------------------- 計算與資料
def test_costs_are_30pct_discount_no_slippage():
    assert costs.SLIPPAGE == 0
    assert costs.BUY_COST == pytest.approx(0.001425 * 0.3)
    assert costs.SELL_COST == pytest.approx(0.001425 * 0.3 + 0.001)
    assert F.BUY_FEE == costs.BUY_COST and F.SELL_FEE == costs.SELL_COST


def test_bt_does_not_import_streamlit():
    for name in ("data", "f0", "finmind", "gauges", "costs", "audit2", "indicators"):
        src = (ROOT / "bt" / f"{name}.py").read_text(encoding="utf-8")
        assert "import streamlit" not in src, name


@pytest.mark.parametrize("content", ["", "date,g5_close,g5_min\n", "foo,bar\n1,2\n", "date,g5_close,g5_min\nxx,1,2\n"])
def test_load_g5_never_crashes(tmp_path, content):
    p = tmp_path / "g5.csv"
    p.write_text(content, encoding="utf-8")
    df, err = G.load_g5(p)
    assert list(df.columns) == G.G5_COLS


def test_turning_month_uses_two_previous_complete_months():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    d = pd.DataFrame({"High": 100.0, "Low": 100.0}, index=idx)
    d.loc["2024-01-10", "High"] = 130.0
    d.loc["2024-02-15", "Low"] = 80.0
    n = F.turning_month(d)
    assert np.isnan(n.loc["2024-02-01"])                 # 2 月只有 1 個完整前月
    assert n.loc["2024-03-01"] == pytest.approx((130 + 80) / 2)


def test_closed_year_rule():
    from ui.cache import is_closed_year
    assert is_closed_year(2025, "2026-06-01")
    assert not is_closed_year(2025, "2026-01-03")       # 新年頭 7 天，去年仍當今年處理
    assert not is_closed_year(2026, "2026-06-01")


def test_fetch_tx_one_request_per_year(fake_finmind):
    tx = F.fetch_tx(TOKEN)
    assert len(fake_finmind.calls) == len(F.tx_years())
    assert not tx.empty


# ---------------------------------------------------------------- 網頁（Streamlit 官方測試工具）
def test_app_runs(fake_finmind):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.secrets["FINMIND_TOKEN"] = TOKEN
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert len(at.metric) == 3
    for v in ["近 3 個月", "上市以來"]:                  # 切換圖表區間
        at.button_group[0].set_value(v).run()
        assert not at.exception


def test_app_without_token_shows_setup_message():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert at.error and "Token" in at.error[0].value
