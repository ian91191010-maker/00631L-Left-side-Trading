# 00631L 交易訊號 ＋ 策略回測

一個網址、三個頁面：
- **主頁**：00631L 每日交易訊號（左側交易審計系統）
- **策略回測**：勾選進出場、停損停利機制，測試不同策略組合的績效，並與審計系統比較
- **審計系統二**：TSI 藍線 > 信號線進場、SMA5 跌破 SMA55 出場

## 檔案結構

```
你的倉庫/
├─ app - 00631L-Left-side-Trading.py   主頁（交易訊號），Streamlit 的 Main file 指向它
├─ pages/
│  ├─ 1_策略回測.py                    回測頁面介面（等於 app.py）
│  └─ 2_審計系統二.py                  審計系統二頁面
├─ bt/                                 回測模組
│  ├─ __init__.py
│  ├─ data.py                          抓資料、分割還原、週線轉換
│  ├─ indicators.py                    SMA／EMA／TSI／ATR／ADX（TradingView 算法）
│  ├─ strategy.py                      策略引擎（每個機制都是獨立函式）
│  ├─ backtest.py                      績效、買進持有、參數比較
│  ├─ baseline.py                      審計系統（主頁）的回測基準，邏輯與主頁相同
│  ├─ audit2.py                        審計系統二的判斷邏輯
│  └─ charts.py                        Plotly 圖表
├─ requirements.txt                    需要安裝的套件
└─ .github/workflows/keep_alive.yml    防休眠排程
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
