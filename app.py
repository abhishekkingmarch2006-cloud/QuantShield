import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
from datetime import datetime

# Page styling config
st.set_page_config(page_title="QuantShield Stock Analyzer", layout="wide")
st.title("🛡️ QuantShield: Indian Stock Fundamental & Technical Hub")
st.write("A professional dashboard delivering comprehensive Screener-style fundamentals and TradingView-style technical indicators.")

# Sidebar Ticker Setup
st.sidebar.header("Stock Selection")
ticker_input = st.sidebar.text_input("Enter NSE Ticker (e.g., RELIANCE, TCS, HDFCBANK):", value="RELIANCE").upper().strip()

# Format for Yahoo Finance Indian Market
if ticker_input and not ticker_input.endswith(".NS"):
    ticker = f"{ticker_input}.NS"
else:
    ticker = ticker_input

if ticker:
    st.markdown(f"## Data Overview for **{ticker_input}**")
    
    with st.spinner("Fetching market data..."):
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        hist_data = ticker_obj.history(period="1y")

    if not info or len(hist_data) == 0:
        st.error(f"Could not retrieve complete metrics for '{ticker_input}'. Verify the NSE ticker symbol.")
    else:
        # TIER 1: FUNDAMENTALS & RATIOS
        st.markdown("---")
        st.header("📊 Tier 1: Screener-Style Fundamental Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        market_cap = info.get('marketCap', 0) / 10000000 
        pe_ratio = info.get('trailingPE', 'N/A')
        pb_ratio = info.get('priceToBook', 'N/A')
        roe = info.get('returnOnEquity', 0) * 100
        debt_to_equity = info.get('debtToEquity', 'N/A')
        dividend_yield = info.get('dividendYield', 0) * 100
        
        col1.metric("Market Cap (Cr)", f"₹ {market_cap:,.2f}")
        col2.metric("P/E Ratio", f"{pe_ratio}" if isinstance(pe_ratio, str) else f"{pe_ratio:.2f}")
        col3.metric("Return on Equity (ROE)", f"{roe:.2f}%" if roe else "N/A")
        col4.metric("Debt to Equity", f"{debt_to_equity / 100:.2f}" if isinstance(debt_to_equity, (int, float)) else f"{debt_to_equity}")

        # TIER 2: TECHNICAL INDICATORS
        st.markdown("---")
        st.header("📈 Tier 2: TradingView-Style Technical Indicator Analysis")
        
        df = hist_data.copy()
        
        # Calculate Technical Indicators via ta library
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        latest_close = df['Close'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1]
        latest_sma50 = df['SMA_50'].iloc[-1]

        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("Current Close Price", f"₹ {latest_close:.2f}")
        
        rsi_status = "Neutral"
        if latest_rsi >= 70: rsi_status = "🔴 Overbought"
        elif latest_rsi <= 30: rsi_status = "🟢 Oversold"
        t_col2.metric("RSI (14-Day)", f"{latest_rsi:.2f}", delta=rsi_status, delta_color="off")
        
        trend_status = "Bullish Track" if latest_close > latest_sma50 else "Bearish Track"
        t_col3.metric("50-Day Moving Average", f"₹ {latest_sma50:.2f}", delta=trend_status)

        # Plot charts
        st.subheader("📉 Technical Charts Tracking Window")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        ax1.plot(df.index, df['Close'], label="Close Price", color="black", linewidth=1.5)
        ax1.plot(df.index, df['EMA_20'], label="20 EMA", color="orange", linestyle="--")
        ax1.plot(df.index, df['SMA_50'], label="50 SMA", color="blue", linestyle="-.")
        ax1.set_title(f"{ticker_input} Daily Closing Technical Chart Structure", fontsize=12)
        ax1.set_ylabel("Price (INR)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(df.index, df['RSI'], label="RSI Line", color="purple")
        ax2.axhline(70, color="red", linestyle=":", alpha=0.7)
        ax2.axhline(30, color="green", linestyle=":", alpha=0.7)
        ax2.fill_between(df.index, 30, 70, color='purple', alpha=0.05)
        ax2.set_ylabel("RSI Range")
        ax2.set_xlabel("Date")
        ax2.set_ylim(10, 90)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close(fig)
