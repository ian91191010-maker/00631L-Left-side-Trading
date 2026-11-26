# -*- coding: utf-8 -*-
"""FinMind API 的唯一出口（全專案所有 HTTP 請求都走這裡）。

- Token 只放在 Authorization header，不放網址。
- 暫時性錯誤（429、5xx、連線逾時）自動重試 3 次，間隔 1、2、4 秒。
  402（額度用完）不重試，重試也沒用。
- 失敗時丟出 FinMindError，訊息只有「資料集、代號、狀態碼」，絕不含網址或 Token，
  可以直接顯示在公開網頁上。
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TIMEOUT = 60

_session = None


class FinMindError(RuntimeError):
    """可以安全顯示給訪客看的錯誤（不含網址、不含 Token）。"""


def _get_session():
    global _session
    if _session is None:
        retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET"]), raise_on_status=False)
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=retry))
        _session = s
    return _session


def request(dataset, data_id, start, end, token):
    """抓一段資料，回傳 list[dict]（FinMind 的 data 欄位）。start、end 可以是日期或 'YYYY-MM-DD' 字串。"""
    label = f"{dataset} {data_id} {str(start)[:10]}～{str(end)[:10]}"
    params = {"dataset": dataset, "data_id": data_id,
              "start_date": str(start)[:10], "end_date": str(end)[:10]}
    try:
        res = _get_session().get(FINMIND_URL, params=params, timeout=TIMEOUT,
                                 headers={"Authorization": f"Bearer {token}"})
    except requests.exceptions.Timeout:
        raise FinMindError(f"FinMind 連線逾時（{label}）") from None
    except requests.exceptions.RequestException as e:
        raise FinMindError(f"FinMind 連線失敗（{label}：{type(e).__name__}）") from None

    if res.status_code != 200:
        hint = {402: "，可能是 Token 額度用完", 401: "，Token 無效", 400: "，可能是 Token 無效"}.get(res.status_code, "")
        raise FinMindError(f"FinMind 回應 HTTP {res.status_code}（{label}{hint}）")
    try:
        payload = res.json()
    except ValueError:
        raise FinMindError(f"FinMind 回傳的不是 JSON（{label}）") from None
    if payload.get("status", 200) != 200:
        msg = str(payload.get("msg", payload.get("status")))
        if token:
            msg = msg.replace(token, "***")
        msg = msg[:120]
        raise FinMindError(f"FinMind 回應錯誤（{label}）：{msg}")
    return payload.get("data", [])
