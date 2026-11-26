# -*- coding: utf-8 -*-
"""交易成本：全專案唯一一份（網頁績效、前向記錄都用這裡）。

口徑：手續費 0.1425% × 3 折、無滑價；賣出另加 ETF 證交稅 0.1%。
"""
FEE_RATE = 0.001425        # 手續費（牌價）
FEE_DISCOUNT = 0.3         # 3 折
ETF_TAX = 0.001            # ETF 證交稅 0.1%，只在賣出時收
SLIPPAGE = 0.0             # 無滑價

BUY_COST = FEE_RATE * FEE_DISCOUNT + SLIPPAGE
SELL_COST = FEE_RATE * FEE_DISCOUNT + ETF_TAX + SLIPPAGE


def describe():
    return f"手續費 {FEE_RATE*100:.4g}% × {FEE_DISCOUNT*10:.0f} 折，賣出另加證交稅 {ETF_TAX*100:.1f}%，無滑價"
