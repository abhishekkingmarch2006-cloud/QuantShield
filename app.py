import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Global Financial Command Center", layout="wide")
st.title("🌐 Global Stock Fundamental & Technical Analysis Hub")
st.write("Analyze any stock worldwide. Enter the exact global ticker symbol to pull company branding, descriptions, core statements, and technical charts.")

# 1. Global Ticker Search Setup
st.sidebar.header("Global Search Configuration")
ticker_input = st.sidebar.text_input(
    "Enter Global Stock Ticker:", 
    value="RELIANCE.NS"
).strip()

st.sidebar.markdown("""
**💡 Global Ticker Quick Guide:**
* **Indian Stocks (NSE):** Add `.NS` (e.g., `TCS.NS`, `RELIANCE.NS`)
* **US Stocks (NASDAQ/NYSE):** Type directly (e.g., `AAPL`, `TSLA`)
* **UK Stocks (London):** Add `.L` (e.g., `VOD.L`)
""")

if ticker_input:
    ticker_upper = ticker_input.upper()
    
    with st.spinner("Fetching global market data and profile metrics from Yahoo Finance..."):
        ticker_obj = yf.Ticker(ticker_input)
        hist_data = ticker_obj.history(period="1y")
        
        # Pull company profile details Safely
        try:
            info = ticker_obj.info
        except Exception:
            info = {}

    if hist_data.empty:
        st.error(f"❌ Could not retrieve market data for '{ticker_input}'. Please check the ticker formatting suffix rules on the sidebar.")
    else:
        # Clean up multi-level columns if returned by yfinance
        if isinstance(hist_data.columns, pd.MultiIndex):
            hist_data.columns = hist_data.columns.get_level_values(0)
            
        # -------------------------------------------------------------
        # BRANDING HEADER: LOGO & DESCRIPTION
        # -------------------------------------------------------------
        st.markdown("---")
        
        # Setup columns for Logo + Title
        head_col1, head_col2 = st.columns([1, 5])
        
        company_name = info.get('longName', ticker_upper)
        website = info.get('website', '')
        
        with head_col1:
            # Check if a website exists to safely fetch the logo
            if website:
                clean_domain = website.replace("https://", "").replace("http://", "").replace("www.", "").split('/')[0]
                logo_url = f"https://logo.clearbit.com/{clean_domain}?size=120"
                
                # Wrap in a try-except block so an invalid URL won't crash the app
                try:
                    st.image(logo_url, width=120)
                except Exception:
                    st.markdown("## 🏢") # Fallback to emoji if the image download fails
            else:
                st.markdown("## 🏢") # Fallback to emoji if no website exists
                
        with head_col2:
            st.title(company_name)
            if website:
                st.markdown(f"🔗 [Visit Official Website]({website})")
                
        # Render Company Description
        st.subheader("📋 Company Profile & Business Summary")
        business_summary = info.get('longBusinessSummary', 'No company description profile summary found for this asset index tier.')
        st.write(business_summary)
        
        # -------------------------------------------------------------
        # PART 1: GLOBAL TECH-TRACKING ENGINE (TECHNICAL ANALYSIS)
        # -------------------------------------------------------------
        st.markdown("---")
        st.header("📈 Technical Analysis")
        
        df = hist_data.copy()
        
        # Calculate Technical Indicators cleanly ensuring data structures match
        df['Close_Clean'] = df['Close'].astype(float)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close_Clean'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close_Clean'], window=20)
        df['RSI'] = ta.momentum.rsi(df['Close_Clean'], window=14)

        # Extract latest metrics for indicators
        latest_close = float(df['Close_Clean'].iloc[-1])
        latest_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0
        latest_sma50 = df['SMA_50'].iloc[-1]
        
        t_col1, t_col2, t_col3 = st.columns(3)
        currency_label = info.get('currency', 'Units')
        t_col1.metric("Current Closing Price", f"{latest_close:.2f} {currency_label}")
        
        rsi_status = "Neutral"
        if latest_rsi >= 70: rsi_status = "🔴 Overbought"
        elif latest_rsi <= 30: rsi_status = "🟢 Oversold"
        t_col2.metric("RSI (14-Day)", f"{latest_rsi:.2f}", delta=rsi_status, delta_color="off")
        
        if not pd.isna(latest_sma50):
            trend_status = "Bullish Track" if latest_close > latest_sma50 else "Bearish Track"
            t_col3.metric("50-Day Moving Average", f"{latest_sma50:.2f} {currency_label}", delta=trend_status)
        else:
            t_col3.metric("50-Day Moving Average", "Calculating...")

        # Plot Visual Technical Graphs
        st.subheader("📉 Technical Charts Tracking Window")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Close prices and Moving Averages
        ax1.plot(df.index, df['Close_Clean'].values, label="Close Price", color="black", linewidth=1.5)
        ax1.plot(df.index, df['EMA_20'].values, label="20 EMA (Short Term)", color="orange", linestyle="--")
        
        if not pd.isna(latest_sma50):
            ax1.plot(df.index, df['SMA_50'].values, label="50 SMA (Medium Term)", color="blue", linestyle="-.")
            
        ax1.set_title(f"{ticker_upper} Historical Technical Trajectory Chart", fontsize=12)
        ax1.set_ylabel(f"Price ({currency_label})")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI
        ax2.plot(df.index, df['RSI'].values, label="RSI Line", color="purple")
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

        # -------------------------------------------------------------
        # PART 2: GLOBAL SCREENER ENGINE (FUNDAMENTALS & STATEMENTS)
        # -------------------------------------------------------------
        st.markdown("---")
        st.header("📊Valuation & Financial Statements")
        
        # Metrics Matrix Summary Display
        if info:
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            
            raw_market_cap = info.get('marketCap', 0)
            display_cap = f"{raw_market_cap:,}" if raw_market_cap else "N/A"
            
            pe_ratio = info.get('trailingPE', 'N/A')
            pe_display = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
            
            pb_ratio = info.get('priceToBook', 'N/A')
            pb_display = f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else "N/A"
            
            roe = info.get('returnOnEquity', None)
            roe_display = f"{roe * 100:.2f}%" if roe is not None else "N/A"
            
            f_col1.metric("Market Cap", f"{display_cap} {currency_label}")
            f_col2.metric("P/E Ratio", pe_display)
            f_col3.metric("P/B Ratio", pb_display)
            f_col4.metric("Return on Equity (ROE)", roe_display)

        # Complete Financial Statements Render tabs
        st.subheader("📋 Core Financial Statements")
        tab1, tab2, tab3 = st.tabs(["Income Statement (P&L)", "Balance Sheet", "Cash Flow Statement"])
        
        with tab1:
            income_statement = ticker_obj.financials
            if income_statement is not None and not income_statement.empty:
                st.dataframe(income_statement.dropna(how='all'), use_container_width=True)
            else:
                st.warning("Income Statement details are unavailable for this asset identifier.")
                
        with tab2:
            balance_sheet = ticker_obj.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty:
                st.dataframe(balance_sheet.dropna(how='all'), use_container_width=True)
            else:
                st.warning("Balance Sheet details are unavailable for this asset identifier.")
                
        with tab3:
            cash_flow = ticker_obj.cashflow
            if cash_flow is not None and not cash_flow.empty:
                st.dataframe(cash_flow.dropna(how='all'), use_container_width=True)
            else:
                st.warning("Cash Flow Statement details are unavailable for this asset identifier.")
