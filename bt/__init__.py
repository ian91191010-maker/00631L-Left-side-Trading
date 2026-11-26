# -*- coding: utf-8 -*-
"""計算模組（純 Python，不依賴 Streamlit；網頁與每日排程共用）。

- finmind.py     FinMind API 唯一出口（重試、錯誤訊息不含 Token）
- data.py        00631L／加權指數日K、分割還原
- f0.py          轉折 F0：台指期近月序列、當月轉折、套用到 00631L、績效
- costs.py       交易成本（手續費 3 折、無滑價、ETF 證交稅 0.1%）
- gauges.py      風險指標（加權指數 DIF）與風控指標（g5 CSV）
- audit2.py      審計系統二（只給前向記錄做對照）
- indicators.py  技術指標（與 TradingView 算法一致）
"""
