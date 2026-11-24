# 00631L 交易訊號 ＋ 策略回測

一個網址、五個頁面：
- **主頁**：00631L 每日交易訊號（左側交易審計系統）
- **策略回測**：勾選進出場、停損停利機制，測試不同策略組合的績效，並與審計系統比較
- **審計系統二**：TSI 藍線 > 信號線進場、SMA5 跌破 SMA55 出場
- **00631L Signal**：00631L 的進出場訊號（正式訊號）、K 線，加上風險指標、風控指標兩張圖；沒有側邊欄，導覽在頁面最下方
- **前向記錄**：每天記下 F0 與審計系統二原版的訊號，純做紀錄（`data/forward_log.csv`，只增不改）

## 檔案結構

```
你的倉庫/
├─ app - 00631L-Left-side-Trading.py   主頁（交易訊號），Streamlit 的 Main file 指向它
├─ pages/
│  ├─ 1_策略回測.py                    回測頁面介面（等於 app.py）
│  ├─ 2_審計系統二.py                  審計系統二頁面
│  ├─ 3_00631L_Signal.py               00631L Signal 頁面
│  └─ 4_前向記錄.py                    前向記錄頁面
├─ forward/record.py                   每日記錄腳本（只在檔案尾端追加）
├─ data/g5_daily.csv                   風控指標資料（60 分 RSI60），由私人庫算好送入；格式見 bt/gauges.py
├─ data/forward_log.csv                前向記錄（由排程產生，不要手動改）
├─ bt/                                 回測模組
│  ├─ __init__.py
│  ├─ data.py                          抓資料、分割還原、週線轉換
│  ├─ indicators.py                    SMA／EMA／TSI／ATR／ADX（TradingView 算法）
│  ├─ strategy.py                      策略引擎（每個機制都是獨立函式）
│  ├─ backtest.py                      績效、買進持有、參數比較
│  ├─ baseline.py                      審計系統（主頁）的回測基準，邏輯與主頁相同
│  ├─ audit2.py                        審計系統二的判斷邏輯
│  ├─ gauges.py                        風險指標（日 MACD 的 DIF）與風控指標（60 分 RSI60）資料
│  ├─ f0.py                            轉折 F0 的判斷邏輯與套用到 00631L
│  └─ charts.py                        Plotly 圖表
├─ requirements.txt                    需要安裝的套件
├─ .github/workflows/keep_alive.yml    防休眠排程
└─ .github/workflows/forward_record.yml  每日前向記錄排程（2026-12-01 起自動執行）
```

## 新增指標或機制

- 指標：在 `bt/indicators.py` 寫一個函式，再到 `bt/strategy.py` 的 `build_indicators()` 加一行。
- 進場條件：在 `bt/strategy.py` 寫 `cond_xxx(ind, cfg)`，加進 `ENTRY_RULES`。
- 訊號出場：寫 `exit_xxx(ind, cfg)`，依優先順序加進 `SIGNAL_EXITS`。
- 停損停利：寫一個回傳價格的函式，加進 `_Engine.stop_levels()`。
- 參數要出現在側邊欄：在 `StrategyConfig` 加欄位，並在 `pages/1_策略回測.py` 加一個輸入框。

## 分割事件

`bt/data.py` 的 `SPLIT_EVENTS` 記錄分割，目前有 00631L 2026-03-31 一拆二十二。
若頁面出現「單日價格變動超過 30%」的警告，代表有新的分割或減資需要補上。

## 前向記錄排程

- 需要在 GitHub 儲存庫的 Settings → Secrets and variables → Actions 新增 `FINMIND_TOKEN`。
- 每天台北 08:00 記下前一個交易日；2026-12-01 之前自動排程會略過，想先測試可到 Actions 頁手動 Run workflow。
- 記錄檔只在尾端新增；排程裡有檢查，若已存在的行被改動會停止提交。
