import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime

# Page styling config
st.set_page_config(page_title="QuantShield Stock Analyzer", layout="wide")
st.title("🛡️ QuantShield: Indian Stock Fundamental & Technical Hub")
st.write("A professional dashboard delivering comprehensive Screener-style fundamentals and TradingView-style technical indicators.")

# 1. Sidebar Ticker Setup
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
        
        # Historical prices for technical analysis (1 Year of daily data)
        hist_data = ticker_obj.history(period="1y")

    if not info or len(hist_data) == 0:
        st.error(f"Could not retrieve complete metrics for '{ticker_input}'. Verify the NSE ticker symbol.")
    else:
        # -------------------------------------------------------------
        # TIER 1: THE SCREENER ENGINE (FUNDAMENTALS & RATIOS)
        # -------------------------------------------------------------
        st.markdown("---")
        st.header("📊 Tier 1: Screener-Style Fundamental Analysis")
        
        # Primary Ratios Display Matrix
        col1, col2, col3, col4 = st.columns(4)
        
        # Safely extract ratios with fallbacks if missing
        market_cap = info.get('marketCap', 0) / 10000000 # Convert to Crores
        pe_ratio = info.get('trailingPE', 'N/A')
        pb_ratio = info.get('priceToBook', 'N/A')
        roe = info.get('returnOnEquity', 0) * 100
        debt_to_equity = info.get('debtToEquity', 'N/A')
        dividend_yield = info.get('dividendYield', 0) * 100
        
        col1.metric("Market Cap (Cr)", f"₹ {market_cap:,.2f}")
        col2.metric("P/E Ratio", f"{pe_ratio}" if isinstance(pe_ratio, str) else f"{pe_ratio:.2f}")
        col3.metric("Return on Equity (ROE)", f"{roe:.2f}%" if roe else "N/A")
        col4.metric("Debt to Equity", f"{debt_to_equity / 100:.2f}" if isinstance(debt_to_equity, (int, float)) else f"{debt_to_equity}")

        # Expanded Ratios Table
        st.subheader("💡 Key Financial Ratios")
        ratios_data = {
            "Metric Name": ["Price to Book (P/B)", "Dividend Yield", "Profit Margin", "EBITDA Margin", "Trailing EPS"],
            "Value": [
                f"{pb_ratio}", 
                f"{dividend_yield:.2f}%", 
                f"{info.get('profitMargins', 0)*100:.2f}%", 
                f"{info.get('ebitdaMargins', 0)*100:.2f}%", 
                f"₹ {info.get('trailingEps', 'N/A')}"
            ]
        }
        st.table(pd.DataFrame(ratios_data).set_index("Metric Name"))

        # Financial Statements Dropdowns
        with st.expander("📅 View Annual Income Statement (Profit & Loss)"):
            financials = ticker_obj.financials
            if not financials.empty:
                st.dataframe(financials.dropna(how='all'))
            else:
                st.info("Annual income statement details are currently unavailable for this stock asset.")

        with st.expander("🏛️ View Annual Balance Sheet"):
            balance_sheet = ticker_obj.balance_sheet
            if not balance_sheet.empty:
                st.dataframe(balance_sheet.dropna(how='all'))
            else:
                st.info("Balance sheet statement details are currently unavailable for this stock asset.")


        # -------------------------------------------------------------
        # TIER 2: THE TRADINGVIEW ENGINE (TECHNICAL INDICATORS)
        # -------------------------------------------------------------
        st.markdown("---")
        st.header("📈 Tier 2: TradingView-Style Technical Indicator Analysis")
        
        # Calculate Technical Indicators via pandas_ta
        df = hist_data.copy()
        
        # Moving Averages
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        
        # RSI (Relative Strength Index)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd_df is not None:
            df = pd.concat([df, macd_df], axis=1)

        # Get latest current indicators for summary text box cards
        latest_close = df['Close'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1]
        latest_sma50 = df['SMA_50'].iloc[-1]
        latest_ema20 = df['EMA_20'].iloc[-1]

        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("Current Close Price", f"₹ {latest_close:.2f}")
        
        # Format RSI with clear market status definitions
        rsi_status = "Neutral"
        if latest_rsi >= 70: rsi_status = "🔴 Overbought"
        elif latest_rsi <= 30: rsi_status = "🟢 Oversold"
        t_col2.metric("RSI (14-Day)", f"{latest_rsi:.2f}", delta=rsi_status, delta_color="off")
        
        # Trend status via moving average cross positions
        trend_status = "Bullish Track" if latest_close > latest_sma50 else "Bearish Track"
        t_col3.metric("50-Day Moving Average", f"₹ {latest_sma50:.2f}", delta=trend_status)

        # Plot charts visually
        st.subheader("📉 Technical Charts Tracking Window")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Closing Prices along with Moving Averages
        ax1.plot(df.index, df['Close'], label="Close Price", color="black", linewidth=1.5)
        ax1.plot(df.index, df['EMA_20'], label="20 EMA (Short Term)", color="orange", linestyle="--")
        ax1.plot(df.index, df['SMA_50'], label="50 SMA (Medium Term)", color="blue", linestyle="-.")
        ax1.set_title(f"{ticker_input} Daily Closing Technical Chart Structure", fontsize=12)
        ax1.set_ylabel("Price (INR)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI Panel with boundary bands
        ax2.plot(df.index, df['RSI'], label="RSI Threshold Line", color="purple")
        ax2.axhline(70, color="red", linestyle=":", alpha=0.7)
        ax2.axhline(30, color="green", linestyle=":", alpha=0.7)
        ax2.fill_between(df.index, 30, 70, color='purple', alpha=0.05)
        ax2.set_ylabel("RSI Range")
        ax2.set_xlabel("Timeline Date")
        ax2.set_ylim(10, 90)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close(fig)
