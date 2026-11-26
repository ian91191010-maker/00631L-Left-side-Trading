# 00631L Signal

用台指期的「當月轉折（F0）」判斷 00631L 的進出場，網站只有一頁：訊號卡、K 線、台指期與轉折、風險指標、風控指標、交易紀錄與績效對照。

另外每天早上由 GitHub Actions 記下前一個交易日的 F0 與「審計系統二」訊號（`data/forward_log.csv`，只增不改），純做紀錄，網頁不顯示。

## 檔案結構

```
你的倉庫/
├─ app.py                  網站唯一的頁面（Streamlit 的 Main file 指向它）
├─ ui/cache.py             網頁快取：TTL、按年保存台指期、「重新取得」與 10 分鐘冷卻
├─ bt/                     計算模組（純 Python，不依賴 Streamlit；網頁與排程共用）
│  ├─ finmind.py           FinMind API 唯一出口：重試、Token 只放 header、錯誤訊息不含 Token
│  ├─ data.py              00631L／加權指數日K、分割還原
│  ├─ f0.py                轉折 F0（台指期近月、當月轉折、套用到 00631L、績效）
│  ├─ costs.py             交易成本：手續費 3 折、無滑價、ETF 證交稅 0.1%
│  ├─ gauges.py            風險指標、風控指標（data/g5_daily.csv）
│  ├─ audit2.py            審計系統二（只給前向記錄做對照）
│  └─ indicators.py        技術指標
├─ forward/record.py       每日前向記錄腳本
├─ data/g5_daily.csv       風控指標資料（由私人庫算好送入）
├─ data/forward_log.csv    前向記錄（排程產生，不要手動改）
├─ tests/test_app.py       自動測試（不需要 Token）
├─ requirements.txt        網站需要的套件（已鎖主版本）
├─ requirements-dev.txt    測試用套件
└─ .github/workflows/      keep_alive.yml（防休眠）、forward_record.yml（每日記錄）
```

## 資料更新與 FinMind 額度

- 日K與今年的台指期每 2 小時更新一次；去年以前的台指期保存 30 天（資料不會再變）。
- 「重新取得最新資料」只清這一頁的快取，全站 10 分鐘內只能按一次，避免額度被用光。
- 第一次開站（或伺服器重開）約打 15 次 API，之後每 2 小時約 3 次。

## 本機執行與測試

```
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q          # 自動測試，不需要 Token
streamlit run app.py         # 需要 .streamlit/secrets.toml 裡有 FINMIND_TOKEN
```

`.streamlit/secrets.toml` 已列在 `.gitignore`，不會被推上 GitHub。

## 分割事件

`bt/data.py` 的 `SPLIT_EVENTS` 記錄分割，目前有 00631L 2026-03-31 一拆二十二。
若頁面出現「單日價格變動超過 30%」的警告，代表有新的分割或減資需要補上。

## 前向記錄排程

- GitHub 儲存庫 Settings → Secrets and variables → Actions 需要有 `FINMIND_TOKEN`。
- 每天台北 08:00 記下前一個交易日；2026-12-01 之前自動排程會略過，可到 Actions 頁手動 Run workflow 測試。
- 記錄檔只在尾端新增；排程裡有檢查，若已存在的行被改動會停止提交。
