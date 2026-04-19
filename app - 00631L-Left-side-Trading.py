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
st.set_page_config(page_title="00631L 交易策略", layout="wide")

st.markdown("<h2 style='text-align: center;'>[00631L 交易策略]</h2>", unsafe_allow_html=True)

try:
    finmind_token = st.secrets["FINMIND_TOKEN"]
except:
    finmind_token = ""  

# ==========================================
# 側邊欄：1. 資料源設定 (決定下載多少資料)
# ==========================================
st.sidebar.subheader("1. 資料下載設定")
# 設定資料下載的起點，預設為 2014 年
data_default_start = datetime.strptime('2014-01-01', '%Y-%m-%d').date()
data_start_date = st.sidebar.date_input("資料下載起始日", data_default_start, 
                                        min_value=datetime.strptime('2010-01-01', '%Y-%m-%d').date(),
                                        help="建議比回測開始日早 3-6 個月，以利技術指標計算")

plot_days = st.sidebar.slider("圖表顯示天數 (0為顯示全部)", 0, 1500, 0, step=50)
btn_run_strategy = st.sidebar.button("▶️ 執行策略運算")

st.sidebar.markdown("---")

# ==========================================
# 側邊欄：2. 資金回測設定 (決定模擬哪段區間)
# ==========================================
st.sidebar.subheader("2. 資金回測與成本設定")

default_bt_start = datetime.strptime('2024-01-01', '%Y-%m-%d').date()
default_bt_end = datetime.today().date()

# 將起訖日期拆分為兩個獨立欄位
col_bt1, col_bt2 = st.sidebar.columns(2)
with col_bt1:
    bt_start_date = st.date_input("回測開始日", default_bt_start)
with col_bt2:
    bt_end_date = st.date_input("回測結束日", default_bt_end)

initial_capital = st.sidebar.number_input("起始投入資金 (NTD)", min_value=10000, value=100000, step=10000)
fee_rate = st.sidebar.number_input("券商單邊手續費率 (%)", value=0.1425, format="%.4f") / 100
tax_rate = st.sidebar.number_input("ETF 賣出交易稅率 (%)", value=0.1, format="%.3f") / 100

btn_run_backtest = st.sidebar.button("📊 執行資金回測")

if "result_df" not in st.session_state:
    st.session_state.result_df = pd.DataFrame()
if "show_backtest" not in st.session_state:
    st.session_state.show_backtest = False

# ==========================================
# 資料獲取模組 
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_futures_data(start_date_obj, token): # 參數改為 start_date_obj
    if not token: return pd.DataFrame()
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = start_date_obj.strftime('%Y-%m-%d') # 轉為字串
    
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
# 核心策略模組
# ==========================================
@st.cache_data(show_spinner=False)
def get_strategy_results(ticker, data_start_date, token): # 參數名同步修改
    df_target = fetch_stock_data(ticker, data_start_date, token)
    df_taiex = fetch_stock_data("TAIEX", data_start_date, token)
    df_futures = fetch_futures_data(data_start_date, token)
    
    if df_target.empty or df_taiex.empty:
        return pd.DataFrame()
        
    df = df_target.copy()
    df_taiex = df_taiex.reindex(df.index).ffill()
    
    df['RSI'] = ta.momentum.rsi(df['Adj_Close'], window=14)
    bb = ta.volatility.BollingerBands(df['Adj_Close'], window=20, window_dev=2.2)
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Upper'] = bb.bollinger_hband()
    df['MA20'] = bb.bollinger_mavg()
    df['BIAS'] = (df['Adj_Close'] - df['MA20']) / df['MA20'] * 100
    df['MA60'] = df['Adj_Close'].rolling(window=60).mean()

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
    
    positions = np.zeros(len(df))
    current_pos = 0.0
    avg_cost = 0.0
    is_cooldown = False

    for i in range(1, len(df)):
        rsi = df['RSI'].iloc[i]
        bias = df['BIAS'].iloc[i]
        close = df['Adj_Close'].iloc[i]
        lower_bb = df['BB_Lower'].iloc[i]
        ma20 = df['MA20'].iloc[i]
        ma60 = df['MA60'].iloc[i] 
        
        target_pos = current_pos
        
        if target_pos == 0:
            avg_cost = 0.0 
            if rsi > 50 or close > ma20:
                is_cooldown = False

        is_bull_trend = close > ma60  
        
        if is_bull_trend:
            if close > ma20:
                if not is_cooldown: target_pos = 1.0
            else:
                target_pos = 0.0
        else:
            if close >= ma20 or rsi > 50: target_pos = 0.0
            if not is_cooldown:
                if rsi < 20 or bias < -12: target_pos = max(target_pos, 1.0)
                elif rsi < 25 or close < lower_bb: target_pos = max(target_pos, 0.6)
                elif rsi < 32 or bias < -6: target_pos = max(target_pos, 0.3)

        if target_pos > current_pos:
            if current_pos == 0: avg_cost = close
            else: avg_cost = (avg_cost * current_pos + close * (target_pos - current_pos)) / target_pos
        elif target_pos == 0:
            avg_cost = 0.0

        if current_pos > 0 and avg_cost > 0:
            if close <= avg_cost * 0.85:
                target_pos = 0.0
                is_cooldown = True 
                avg_cost = 0.0      

        positions[i] = target_pos
        current_pos = target_pos

    df['Position'] = positions
    df['Position_Shift'] = df['Position'].diff().fillna(0)
    
    def map_action(shift):
        if shift > 0: return f"BUY (+{shift*100:.0f}%)"
        elif shift < 0: return "SELL ALL" if shift <= -0.99 else f"SELL (部分)"
        return ""
    
    df['Action'] = df['Position_Shift'].apply(map_action)
    return df

# ==========================================
# 資金回測模組 
# ==========================================
def calculate_equity_curve(df, start_date, end_date, initial_capital, fee_rate, tax_rate):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # 裁切回測區間
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    if not mask.any(): return pd.DataFrame()
    
    btest_df = df.loc[mask].copy()
    cash = initial_capital
    shares = 0
    equity = []
    
    # 紀錄當下持有的曝險比例
    current_exposure = 0.0
    
    for i in range(len(btest_df)):
        today_open = btest_df['Adj_Open'].iloc[i]
        today_close = btest_df['Adj_Close'].iloc[i]
        current_date = btest_df.index[i]
        
        # 取得「前一天」的目標部位：從完整的 df 中尋找絕對位置
        idx_in_df = df.index.get_loc(current_date)
        if idx_in_df > 0:
            target_exposure = df['Position'].iloc[idx_in_df - 1]
        else:
            target_exposure = 0.0
            
        # 計算應調整的部位差額
        shift = target_exposure - current_exposure
        
        # 執行買進邏輯 (當 shift > 0 且有剩餘資金)
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
                
        # 執行賣出邏輯 (當 shift < 0 且持有股數)
        elif shift < 0 and shares > 0:
            if target_exposure == 0: 
                sell_shares = shares # 全部賣出
            else:
                proportion_to_sell = abs(shift) / current_exposure
                sell_shares = np.floor(shares * proportion_to_sell) # 依比例賣出
                
            if sell_shares > 0:
                gross_proceeds = sell_shares * today_open
                fee = gross_proceeds * fee_rate
                tax = gross_proceeds * tax_rate
                cash = cash + gross_proceeds - fee - tax
                shares -= sell_shares
        
        # 更新當前持有的曝險水位，並記錄今日收盤後的總淨值
        current_exposure = target_exposure
        current_value = cash + (shares * today_close)
        equity.append(current_value)
        
    btest_df['Equity'] = equity
    btest_df['Drawdown'] = (btest_df['Equity'] / btest_df['Equity'].cummax()) - 1
    return btest_df

# ==========================================
# 按鈕觸發邏輯與圖表渲染
# ==========================================
ticker = "00631L" # 定義目標 ETF 代號

if btn_run_strategy:
    if not finmind_token:
        st.error("🚨 尚未在 secrets.toml 設定 FinMind API Token。")
    else:
        with st.spinner('正在獲取資料並運算策略...'):
            # 將 lookback_years 改為 data_start_date
            st.session_state.result_df = get_strategy_results(ticker, data_start_date, finmind_token) 
            st.session_state.show_backtest = False

if btn_run_backtest:
    if st.session_state.result_df.empty:
        with st.spinner('正在獲取底層資料...'):
            # 將 lookback_years 改為 data_start_date
            st.session_state.result_df = get_strategy_results(ticker, data_start_date, finmind_token)
    st.session_state.show_backtest = True

if not st.session_state.result_df.empty:
    result_df = st.session_state.result_df

    latest_row = result_df.iloc[-1]
    last_date = latest_row.name.strftime('%Y-%m-%d')
    last_price = f"{latest_row['Close']:.2f}"
    
    # 1. 判斷精確的 Action 狀態 (加入判斷持有或空手)
    if latest_row['Action']:
        last_action = latest_row['Action']
    else:
        if latest_row['Position'] > 0:
            last_action = "HOLD (持有)"
        else:
            last_action = "EMPTY (空手)"
    
    # 2. 依據 Action 判定整體橫幅底色 (透明綠/紅/黃)
    if "BUY" in last_action or "HOLD" in last_action: 
        box_bg = "rgba(46, 125, 50, 0.5)"  # 透明綠色
    elif "SELL" in last_action: 
        box_bg = "rgba(198, 40, 40, 0.5)"  # 透明紅色
    elif "EMPTY" in last_action or "空手" in last_action: 
        box_bg = "rgba(249, 168, 37, 0.5)" # 透明黃色
    else:
        box_bg = "rgba(69, 69, 69, 0.5)"   # 預設透明灰
    
    # 3. 移除圖示，並將字體放大一倍 (設定為 36px)
    banner_html = f"""
    <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 25px;">
        <div style="background-color: {box_bg}; color: white; padding: 10px 25px; border-radius: 6px; font-size: 36px; font-weight: bold; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);">{last_date}</div>
        <div style="background-color: {box_bg}; color: white; padding: 10px 25px; border-radius: 6px; font-size: 36px; font-weight: bold; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);">{last_price}</div>
        <div style="background-color: {box_bg}; color: white; padding: 10px 25px; border-radius: 6px; font-size: 36px; font-weight: bold; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);">{last_action}</div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

    plot_df = result_df.tail(plot_days) if plot_days > 0 else result_df
    
    fig = go.Figure()
    
    in_position = False
    start_dt = None
    for date, row in plot_df.iterrows():
        if row['Position'] > 0 and not in_position:
            start_dt = date
            in_position = True
        elif row['Position'] == 0 and in_position:
            fig.add_vrect(x0=start_dt, x1=date, fillcolor="rgba(255, 165, 0, 0.15)", layer="below", line_width=0)
            in_position = False
    if in_position:
        fig.add_vrect(x0=start_dt, x1=plot_df.index[-1], fillcolor="rgba(255, 165, 0, 0.15)", layer="below", line_width=0)

    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], 
        name='K線',
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ))
    
    buys = plot_df[plot_df['Position_Shift'] > 0]
    sells = plot_df[plot_df['Position_Shift'] < 0]
    
    for dt in buys.index:
        fig.add_vline(x=dt, line_dash="dash", line_color="#FFD700", opacity=0.8, line_width=0.8) 
    for dt in sells.index:
        fig.add_vline(x=dt, line_dash="dash", line_color="#1E90FF", opacity=0.8, line_width=0.8) 
    
    fig.update_layout(title="", xaxis_title="日期", yaxis_title="價格", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 進出場紀錄 (最近3次)
    # ==========================================
    st.markdown("<h4 style='text-align: left; margin-top: 20px;'>近期交易紀錄 (最近3次)</h4>", unsafe_allow_html=True)
    
    # 4. 取得所有有動作的紀錄，並只保留最後 3 筆
    recent_actions = result_df[result_df['Position_Shift'] != 0].tail(3).copy()
    
    if recent_actions.empty:
        st.info("目前無任何進出場動作。")
    else:
        recent_actions = recent_actions.iloc[::-1] # 反轉讓最新的一筆在最上方
        for idx, row in recent_actions.iterrows():
            d_str = idx.strftime('%Y-%m-%d')
            p_str = f"{row['Close']:.2f}"
            a_str = row['Action']
            
            if row['Position_Shift'] > 0:
                triangle = "<span style='color: #2E7D32; font-size: 1.2em;'>►</span>"
            else:
                triangle = "<span style='color: #C62828; font-size: 1.2em;'>◄</span>"
                
            st.markdown(f"{triangle} &nbsp; **{d_str}** &nbsp;&nbsp;|&nbsp;&nbsp; 收盤價：**{p_str}** &nbsp;&nbsp;|&nbsp;&nbsp; 動作：**{a_str}**", unsafe_allow_html=True)

    # ==========================================
    # 績效模擬
    # ==========================================
    if st.session_state.show_backtest:
        st.markdown("<hr style='border: 1px solid #555; margin-top: 30px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: left;'>績效模擬</h4>", unsafe_allow_html=True)
        
        btest_df = calculate_equity_curve(result_df, start_date=bt_start_date, end_date=bt_end_date,
                                          initial_capital=initial_capital, fee_rate=fee_rate, tax_rate=tax_rate)
        
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
            
            fig_eq.update_layout(title=f"起始本金: {initial_capital:,.0f}", xaxis_title="日期", yaxis_title="淨值 (NTD)", height=400)
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.warning("⚠️ 該回測區間內無有效交易資料，請調整左側的回測日期區間。")
