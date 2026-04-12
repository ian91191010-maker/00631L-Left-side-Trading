import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# ==========================================
# 網頁 UI 設定
# ==========================================
st.set_page_config(page_title="左側極值網格防禦策略", layout="wide")
st.title("00631L.TW 左側均值回歸 (網格建倉)")

st.sidebar.subheader("資料源設定")
finmind_token = st.sidebar.text_input("FinMind API Token", type="password")
lookback_years = st.sidebar.number_input("回測年數", min_value=1, max_value=10, value=5)
plot_days = st.sidebar.slider("圖表顯示天數 (0為顯示全部)", 0, 1500, 0, step=50)

ticker = "00631L"

st.sidebar.markdown("---")
st.sidebar.subheader("資金回測與摩擦成本設定")
bt_start_date = st.sidebar.date_input("回測起始日", datetime.strptime('2024-01-01', '%Y-%m-%d'))
initial_capital = st.sidebar.number_input("起始投入資金 (NTD)", min_value=10000, value=100000, step=10000)
fee_rate = st.sidebar.number_input("券商單邊手續費率 (%)", value=0.1425, format="%.4f") / 100
tax_rate = st.sidebar.number_input("ETF 賣出交易稅率 (%)", value=0.1, format="%.3f") / 100

# ==========================================
# 資料獲取模組 
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_stock_data(symbol, years, token):
    if not token: return pd.DataFrame()
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {"dataset": "TaiwanStockPrice", "data_id": symbol, "start_date": start_date, "end_date": end_date, "token": token}
    try:
        res = requests.get(url, params=parameter, timeout=15)
        if res.status_code != 200: return pd.DataFrame()
        json_data = res.json()
        if json_data.get("msg") != "success": return pd.DataFrame()
        df = pd.DataFrame(json_data.get("data", []))
        if df.empty: return df
            
        df = df.rename(columns={"date": "Date", "open": "Open", "max": "High", "min": "Low", "close": "Close"})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Close'])
        
        df['Adj_Open'] = df['Open']
        df['Adj_High'] = df['High']
        df['Adj_Low'] = df['Low']
        df['Adj_Close'] = df['Close']
        
        split_date = pd.to_datetime('2026-03-31') 
        split_ratio = 22.0
        mask = df.index < split_date
        
        df.loc[mask, 'Adj_Open'] = df.loc[mask, 'Open'] / split_ratio
        df.loc[mask, 'Adj_High'] = df.loc[mask, 'High'] / split_ratio
        df.loc[mask, 'Adj_Low'] = df.loc[mask, 'Low'] / split_ratio
        df.loc[mask, 'Adj_Close'] = df.loc[mask, 'Close'] / split_ratio
            
        return df
    except:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_futures_data(years, token):
    if not token: return pd.DataFrame()
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {"dataset": "TaiwanFuturesDaily", "data_id": "TX", "start_date": start_date, "end_date": end_date, "token": token}
    
    try:
        res = requests.get(url, params=parameter, timeout=15)
        if res.status_code != 200: return pd.DataFrame()
        df = pd.DataFrame(res.json().get("data", []))
        if df.empty: return df
        
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df.sort_values(['date', 'volume'], ascending=[True, False])
        df = df.drop_duplicates(subset=['date'], keep='first') 
        
        df = df.rename(columns={"date": "Date", "close": "Futures_Close"})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Futures_Close'] = pd.to_numeric(df['Futures_Close'], errors='coerce')
        return df[['Futures_Close']]
    except:
        return pd.DataFrame()

# ==========================================
# 核心策略模組 (左側均值回歸與網格建倉)
# ==========================================
def run_left_side_strategy(df_target, df_taiex, df_futures):
    df = df_target.copy()
    df_taiex = df_taiex.reindex(df.index).ffill()
    
    # --- 左側極端指標 ---
    # 1. RSI (過度恐慌偵測)
    df['RSI'] = ta.momentum.rsi(df['Adj_Close'], window=14)
    
    # 2. 布林通道 (價格乖離偵測)
    bb = ta.volatility.BollingerBands(df['Adj_Close'], window=20, window_dev=2.2)
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Upper'] = bb.bollinger_hband()
    df['MA20'] = bb.bollinger_mavg()
    
    # 3. 負乖離率 (BIAS)
    df['BIAS'] = (df['Adj_Close'] - df['MA20']) / df['MA20'] * 100

    # 4. 期現貨籌碼觀測 (作為輔助)
    df_futures = df_futures.reindex(df.index).ffill()
    df['Basis'] = df_futures['Futures_Close'] - df_taiex['Close']
    df['Smooth_Basis'] = df['Basis'].rolling(window=5).mean()
    df['Month'] = df.index.month
    df['Is_Dividend_Season'] = df['Month'].isin([6, 7, 8])
    
    def categorize_basis(row):
        b = row['Smooth_Basis']
        if pd.isna(b): return "數據不足"
        if row['Is_Dividend_Season']:
            if b >= 20: return "極端正價差"
            elif b >= 0: return "微幅正價差"
            else: return "假性逆價差"
        else:
            if b >= 40: return "極端正價差"
            elif b >= 5: return "微幅正價差"
            elif b < 0: return "實質逆價差"
            else: return "平水雜訊"
    df['Basis_State'] = df.apply(categorize_basis, axis=1)
    
    # --- 倉位狀態機 (分批網格建倉) ---
    positions = np.zeros(len(df))
    current_pos = 0.0
    
    for i in range(1, len(df)):
        rsi = df['RSI'].iloc[i]
        bias = df['BIAS'].iloc[i]
        close = df['Adj_Close'].iloc[i]
        lower_bb = df['BB_Lower'].iloc[i]
        ma20 = df['MA20'].iloc[i]
        
        target_pos = current_pos
        
        # 【出場條件】均值回歸：只要價格反彈碰到 20日線，或 RSI 回溫至 50，無條件獲利了結
        if close >= ma20 or rsi > 50:
            target_pos = 0.0
        else:
            # 【進場條件】網格向下加碼 (只增不減，直到觸發出場)
            # Level 3: 極度恐慌 (打滿 100%)
            if rsi < 20 or bias < -12:
                target_pos = max(target_pos, 1.0)
            # Level 2: 嚴重超賣 (建倉至 60%)
            elif rsi < 25 or close < lower_bb:
                target_pos = max(target_pos, 0.6)
            # Level 1: 恐慌初現 (試單 30%)
            elif rsi < 32 or bias < -6:
                target_pos = max(target_pos, 0.3)
                
        positions[i] = target_pos
        current_pos = target_pos

    df['Position'] = positions
    df['Position_Shift'] = df['Position'].diff().fillna(0)
    
    # 動態產生買賣 Action 標籤
    def map_action(shift):
        if shift > 0: return f"BUY (+{shift*100:.0f}%)"
        elif shift < 0: 
            return "SELL ALL" if shift <= -0.99 else f"SELL (部分)"
        return ""
    
    df['Action'] = df['Position_Shift'].apply(map_action)
    return df

# ==========================================
# 資金回測模組 (支援網格分批買賣)
# ==========================================
def calculate_equity_curve(df, start_date, initial_capital, fee_rate, tax_rate):
    mask = df.index >= pd.to_datetime(start_date)
    if not mask.any(): return pd.DataFrame()
    
    btest_df = df.loc[mask].copy()
    cash = initial_capital
    shares = 0
    equity = []
    
    for i in range(len(btest_df)):
        today_open = btest_df['Adj_Open'].iloc[i]
        today_close = btest_df['Adj_Close'].iloc[i]
        
        if i > 0:
            target_exposure = btest_df['Position'].iloc[i-1]
            current_exposure = btest_df['Position'].iloc[i-2] if i > 1 else 0.0
            shift = target_exposure - current_exposure
            
            # 分批買進
            if shift > 0 and cash > 0:
                current_equity = cash + (shares * today_open)
                target_buy_value = current_equity * shift
                max_buy_value = min(target_buy_value, cash)
                add_shares = np.floor(max_buy_value / (today_open * (1 + fee_rate)))
                
                if add_shares > 0:
                    cost = add_shares * today_open
                    fee = cost * fee_rate
                    cash = cash - cost - fee
                    shares += add_shares
                    
            # 分批/全數賣出
            elif shift < 0 and shares > 0:
                if target_exposure == 0:
                    sell_shares = shares # 全賣
                else:
                    proportion_to_sell = abs(shift) / current_exposure
                    sell_shares = np.floor(shares * proportion_to_sell)
                    
                if sell_shares > 0:
                    gross_proceeds = sell_shares * today_open
                    fee = gross_proceeds * fee_rate
                    tax = gross_proceeds * tax_rate
                    cash = cash + gross_proceeds - fee - tax
                    shares -= sell_shares
                    
        current_value = cash + (shares * today_close)
        equity.append(current_value)
        
    btest_df['Equity'] = equity
    btest_df['Drawdown'] = (btest_df['Equity'] / btest_df['Equity'].cummax()) - 1
    return btest_df

# ==========================================
# 執行與圖表渲染
# ==========================================
if st.sidebar.button("執行左側網格策略運算"):
    if not finmind_token:
        st.error("執行中止：尚未輸入 FinMind API Token。")
    else:
        with st.spinner('正在獲取多維度市場資料並計算網格矩陣...'):
            df_target = fetch_stock_data(ticker, lookback_years, finmind_token)
            df_taiex = fetch_stock_data("TAIEX", lookback_years, finmind_token)
            df_futures = fetch_futures_data(lookback_years, finmind_token)
            
            error_msgs = []
            if df_target.empty: error_msgs.append(f"缺失 {ticker} 資料")
            if df_taiex.empty: error_msgs.append("缺失 TAIEX 資料")
            
            if not error_msgs:
                result_df = run_left_side_strategy(df_target, df_taiex, df_futures)
                
                # ==========================================
                # [第一區] 戰略全貌：價格行為與網格觸發圖
                # ==========================================
                st.header("區域一：左側網格建倉軌跡")
                plot_df = result_df.tail(plot_days) if plot_days > 0 else result_df
                
                fig = go.Figure()
                
                # 繪製持倉區間底色 (依據部位大小給予不同透明度)
                in_position = False
                start_date = None
                for date, row in plot_df.iterrows():
                    if row['Position'] > 0 and not in_position:
                        start_date = date
                        in_position = True
                    elif row['Position'] == 0 and in_position:
                        fig.add_vrect(x0=start_date, x1=date, fillcolor="rgba(255, 165, 0, 0.15)", layer="below", line_width=0)
                        in_position = False
                if in_position:
                    fig.add_vrect(x0=start_date, x1=plot_df.index[-1], fillcolor="rgba(255, 165, 0, 0.15)", layer="below", line_width=0)

                # K線與布林通道下軌
                fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                             low=plot_df['Low'], close=plot_df['Close'], name='K線'))
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Lower'], mode='lines', line=dict(color='rgba(0,191,255,0.6)', dash='dot'), name='布林下軌 (接刀線)'))
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], mode='lines', line=dict(color='rgba(255,255,255,0.4)'), name='20日線 (停利線)'))
                
                # 買賣標記 (分批網格)
                buys = plot_df[plot_df['Position_Shift'] > 0]
                sells = plot_df[plot_df['Position_Shift'] < 0]
                
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low'], mode='markers',
                                         marker=dict(symbol='triangle-up', size=14, color='orange', line=dict(color='darkred', width=2)), 
                                         text=buys['Action'], name='向下攤平加碼'))
                fig.add_trace(go.Scatter(x=sells.index, y=sells['High'], mode='markers',
                                         marker=dict(symbol='triangle-down', size=16, color='cyan', line=dict(color='blue', width=2)), 
                                         text=sells['Action'], name='均值回歸停利'))
                
                fig.update_layout(title=f"{ticker} 左側逆勢網格圖", xaxis_title="日期", yaxis_title="價格", height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # ==========================================
                # [第二區] 執行指令：次日操作監控表
                # ==========================================
                st.header("區域二：左側極值監控表")
                view_df = result_df.tail(15).copy()
                view_df.index = view_df.index.strftime('%Y-%m-%d')
                view_df.index.name = '日期'

                def map_basis_ui(state):
                    if state == "極端正價差": return "⚠️ 情緒過熱"
                    elif state == "微幅正價差": return "🔥 軋空起手"
                    elif state == "實質逆價差": return "🛡️ 轉倉紅利"
                    elif state == "假性逆價差": return "❄️ 除息干擾"
                    else: return "⚖️ 價差平水"
                    
                view_df['籌碼觀測'] = view_df['Basis_State'].apply(map_basis_ui)
                view_df['RSI(恐慌度)'] = view_df['RSI'].round(1).astype(str) + " (低於30警戒)"
                view_df['負乖離率'] = view_df['BIAS'].round(2).astype(str) + "%"

                display_cols = ['Close', 'RSI(恐慌度)', '負乖離率', 'BB_Lower', 'MA20', '籌碼觀測', 'Position', 'Action']
                view_df = view_df[display_cols]
                view_df = view_df.rename(columns={'Close': '實際報價', 'BB_Lower': '布林下軌', 'MA20': '月均線(停利點)', 'Position': '目標倉位(%)'})
                view_df['目標倉位(%)'] = (view_df['目標倉位(%)'] * 100).astype(int).astype(str) + "%"

                st.dataframe(view_df.style.map(
                    lambda x: 'background-color: #ffcccc' if x == "0%" else ('background-color: #ffe4b5' if x != "100%" else 'background-color: #ccffcc'),
                    subset=['目標倉位(%)']
                ), use_container_width=True)
                
                st.divider()

                # ==========================================
                # [第三區] 戰略評估：實盤資金權益曲線
                # ==========================================
                st.header("區域三：資金績效模擬 (分批網格實戰)")
                btest_df = calculate_equity_curve(result_df, start_date=bt_start_date, 
                                                  initial_capital=initial_capital, 
                                                  fee_rate=fee_rate, tax_rate=tax_rate)
                
                if not btest_df.empty:
                    final_equity = btest_df['Equity'].iloc[-1]
                    max_dd = btest_df['Drawdown'].min() * 100
                    total_return = ((final_equity / initial_capital) - 1) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("最終帳戶淨值 (NTD)", f"${final_equity:,.0f}")
                    col2.metric("區間總報酬率", f"{total_return:.2f}%")
                    col3.metric("最大歷史回檔 (MDD)", f"{max_dd:.2f}%")
                    
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(x=btest_df.index, y=btest_df['Equity'], mode='lines', name='帳戶總淨值', line=dict(color='gold', width=2)))
                    
                    b_points = btest_df[btest_df['Position_Shift'] > 0]
                    s_points = btest_df[btest_df['Position_Shift'] < 0]
                    
                    fig_eq.add_trace(go.Scatter(x=b_points.index, y=b_points['Equity'], mode='markers',
                                             marker=dict(symbol='triangle-up', size=10, color='orange'), name='向下攤平加碼'))
                    fig_eq.add_trace(go.Scatter(x=s_points.index, y=s_points['Equity'], mode='markers',
                                             marker=dict(symbol='triangle-down', size=10, color='cyan'), name='強制停利出局'))
                    
                    fig_eq.update_layout(title=f"網格系統權益曲線 (起始本金: {initial_capital:,.0f})", xaxis_title="日期", yaxis_title="淨值 (NTD)", height=400)
                    st.plotly_chart(fig_eq, use_container_width=True)
            else:
                st.error("🚨 核心資料鏈斷裂，策略中止執行。")
                for msg in error_msgs: st.error(f"-> {msg}")
