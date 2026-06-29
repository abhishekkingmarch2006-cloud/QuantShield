import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Global Finance Hub", layout="wide")
st.title("🌐 Integrated Global Stock Analysis & Forecasting Dashboard")
st.write("Track any stock worldwide. Enter the exact ticker symbol to view profile details, clean technical analysis indicators, and 5-year data projections.")

# 1. Sidebar Config
st.sidebar.header("Search Parameters")
ticker_input = st.sidebar.text_input("Enter Global Ticker Symbol:", value="RELIANCE.NS").strip()

st.sidebar.markdown("""
**💡 Quick Suffix Guide:**
* **Indian NSE:** Add `.NS` (e.g., `RELIANCE.NS`, `TCS.NS`)
* **US NASDAQ/NYSE:** Type directly (e.g., `AAPL`, `TSLA`)
* **UK London Market:** Add `.L` (e.g., `VOD.L`)
""")

if ticker_input:
    ticker_upper = ticker_input.upper()
    
    with st.spinner("Fetching market telemetry from data streams..."):
        ticker_obj = yf.Ticker(ticker_input)
        # 5-Year data download covers our historical baseline and long term ARIMA limits
        hist_data = ticker_obj.history(period="5y")
        
        try:
            info = ticker_obj.info
        except Exception:
            info = {}

    if hist_data.empty:
        st.error(f"❌ No asset data found for symbol '{ticker_input}'. Ensure the ticker layout match market rules.")
    else:
        # -------------------------------------------------------------
        # ⚡ CRITICAL DATA EXTRACTION ENGINE (FIX FOR INDIAN MULTIINDEX)
        # -------------------------------------------------------------
        df = hist_data.copy()
        
        # If columns contain MultiIndex tiers (Common with Indian markets via yfinance), extract the 'Close' slice explicitly
        if isinstance(df.columns, pd.MultiIndex):
            close_series = df['Close'][ticker_upper].dropna() if ticker_upper in df['Close'].columns else df['Close'].iloc[:, 0].dropna()
        else:
            close_series = df['Close'].dropna()
            
        # Convert index safely to clean timezone-naive datetimes
        close_series.index = pd.to_datetime(close_series.index).tz_localize(None)
        close_series = close_series.astype(float)

        # -------------------------------------------------------------
        # CORPORATE BRANDING EXTRACTION
        # -------------------------------------------------------------
        with head_col1:
            if website:
                # Advanced domain cleaning to ensure Clearbit API can read the URL format
                clean_domain = website.lower().strip()
                for prefix in ["https://", "http://", "www."]:
                    if clean_domain.startswith(prefix):
                        clean_domain = clean_domain[len(prefix):]
                # Split at the first slash to isolate just the root domain name (e.g., relianceindustries.com)
                clean_domain = clean_domain.split('/')[0]
                
                logo_url = f"https://logo.clearbit.com/{clean_domain}?size=120"
                try: 
                    st.image(logo_url, width=120)
                except Exception: 
                    st.markdown("## 🏢")
            else:
                st.markdown("## 🏢")
        # -------------------------------------------------------------
        # TECHNICAL TRACKING ENGINE
        # -------------------------------------------------------------
        st.markdown("---")
        st.header("📈 Technical Analysis & Price Forecasting Engine")
        
        # Pull 1 Year subset out of the cleaned series for the layout tracking view
        close_1y = close_series.tail(252)
        
        # Calculate clean indicator paths
        sma_50 = ta.trend.sma_indicator(close_1y, window=50)
        ema_20 = ta.trend.ema_indicator(close_1y, window=20)
        rsi_14 = ta.momentum.rsi(close_1y, window=14)
        
        # Metric indicators layout cards row
        latest_close = close_1y.iloc[-1]
        latest_rsi = rsi_14.iloc[-1] if not pd.isna(rsi_14.iloc[-1]) else 50.0
        latest_sma = sma_50.iloc[-1]
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Current Market Price", f"{latest_close:.2f} {currency_label}")
        
        rsi_delta = "🔴 Overbought" if latest_rsi >= 70 else "🟢 Oversold" if latest_rsi <= 30 else "Neutral"
        m_col2.metric("RSI (14-Day Baseline)", f"{latest_rsi:.2f}", delta=rsi_delta, delta_color="off")
        
        if not pd.isna(latest_sma):
            trend_delta = "Bullish Track" if latest_close > latest_sma else "Bearish Track"
            m_col3.metric("50-Day Moving Average", f"{latest_sma:.2f} {currency_label}", delta=trend_delta)
        else:
            m_col3.metric("50-Day Moving Average", "Processing Setup...")

        # DE-CLUTTER TABS: Split current active charting and long term prediction engines
        tech_tab1, tech_tab2 = st.tabs(["📊 1-Year Active Tracking Charts", "🔮 5-Year Statistical Projections"])
        
        with tech_tab1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            
            ax1.plot(close_1y.index, close_1y.values, label="Closing Price", color="black", linewidth=1.5)
            ax1.plot(close_1y.index, ema_20.values, label="20 EMA", color="orange", linestyle="--")
            if not sma_50.dropna().empty:
                ax1.plot(close_1y.index, sma_50.values, label="50 SMA", color="blue", linestyle="-.")
            ax1.set_title(f"{ticker_upper} Technical Trajectory Chart")
            ax1.set_ylabel(f"Price ({currency_label})")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.2)
            
            ax2.plot(close_1y.index, rsi_14.values, label="RSI Line", color="purple")
            ax2.axhline(70, color="red", linestyle=":", alpha=0.5)
            ax2.axhline(30, color="green", linestyle=":", alpha=0.5)
            ax2.set_ylabel("RSI Value")
            ax2.set_ylim(10, 90)
            ax2.grid(True, alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)
            
        with tech_tab2:
            with st.spinner("Executing Auto-ARIMA math structures across weekly historical aggregates..."):
                try:
                    # Sample weekly to keep long projections visually clean and fast
                    weekly_series = close_series.resample('W-MON').mean().dropna()
                    model = auto_arima(weekly_series, seasonal=False, error_action='ignore', suppress_warnings=True)
                    fitted = model.fit(weekly_series)
                    
                    # Project 260 weeks out (~5 Years)
                    forecast_val, conf_int = fitted.predict(n_periods=260, return_conf_int=True)
                    forecast_idx = pd.date_range(start=weekly_series.index[-1] + pd.DateOffset(weeks=1), periods=260, freq='W-MON')
                    f_series = pd.Series(forecast_val, index=forecast_idx)
                    
                    p_col1, p_col2 = st.columns([2, 1])
                    with p_col1:
                        fig_arima, ax_arima = plt.subplots(figsize=(10, 5))
                        ax_arima.plot(weekly_series.index, weekly_series.values, label="Historical Data", color="blue")
                        ax_arima.plot(f_series.index, f_series.values, label="ARIMA Path Prediction", color="red", linestyle="--")
                        ax_arima.fill_between(f_series.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='Confidence Frame')
                        ax_arima.set_ylabel(f"Price ({currency_label})")
                        ax_arima.legend(loc="upper left")
                        ax_arima.grid(True, alpha=0.2)
                        st.pyplot(fig_arima)
                        plt.close(fig_arima)
                    with p_col2:
                        fig_hist, ax_hist = plt.subplots(figsize=(5, 5))
                        sns.histplot(f_series, kde=True, color="teal", ax=ax_hist)
                        ax_hist.set_title("Forecast Value Density Distribution")
                        st.pyplot(fig_hist)
                        plt.close(fig_hist)
                        
                    # Target checkpoints breakdown
                    st.markdown("**Predicted Price Checkpoints (Year-End Targets):**")
                    checkpoints = f_series.resample('YE').last()
                    c_cols = st.columns(len(checkpoints))
                    for idx, yr in enumerate(checkpoints.index):
                        c_cols[idx].metric(f"{yr.strftime('%Y')} Target", f"{checkpoints.iloc[idx]:.2f} {currency_label}")
                except Exception as e:
                    st.warning(f"Forecasting model execution skipped: {e}")

        # -------------------------------------------------------------
        # FUNDAMENTALS ENGINE
        # -------------------------------------------------------------
        st.markdown("---")
        st.header("📊 Part 2: Fundamental Statements & Corporate Growth Forecast")
        
        income_statement = ticker_obj.financials
        balance_sheet = ticker_obj.balance_sheet
        cash_flow = ticker_obj.cashflow
        
        if info:
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            raw_mcap = info.get('marketCap', 0)
            if raw_mcap:
                formatted_mcap = f"₹ {raw_mcap / 10000000:.2f} Cr" if currency_label == "INR" else f"{raw_mcap / 1000000000:.2f} B {currency_label}"
            else:
                formatted_mcap = "N/A"
            f_col1.metric("Market Capitalization", formatted_mcap)
            f_col2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            f_col3.metric("P/B Ratio", f"{info.get('priceToBook', 'N/A')}")
            f_col4.metric("Return on Equity (ROE)", f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A")

        # Projections Sub-Engine
        st.subheader("📈 5-Year Statement Value Trajectory Projections")
        if income_statement is not None and not income_statement.empty and income_statement.shape[1] >= 2:
            try:
                rev_label = 'Total Revenue' if 'Total Revenue' in income_statement.index else 'Revenue' if 'Revenue' in income_statement.index else None
                if rev_label:
                    rev_vals = income_statement.loc[rev_label].dropna().values[::-1]
                    n = min(3, len(rev_vals)-1)
                    cagr = ((rev_vals[-1] / rev_vals[-1-n]) ** (1/n)) - 1
                    cagr = max(min(cagr, 0.25), -0.15) # Cap boundaries to keep visualization logical
                    
                    proj_years = [str(datetime.today().year + i) for i in range(1, 6)]
                    proj_rev = []
                    curr_rev = rev_vals[-1]
                    for _ in range(5):
                        curr_rev *= (1 + cagr)
                        proj_rev.append(curr_rev)
                        
                    scale_factor = 10000000 if currency_label == "INR" else 1000000000
                    scale_lbl = "Crores (Cr)" if currency_label == "INR" else "Billions (B)"
                    
                    st.write(f"Estimated revenue trajectory using a calculated 3-year historical baseline CAGR of **{cagr*100:.2f}%** (Values in **{scale_lbl}**):")
                    proj_df = pd.DataFrame({'Year': proj_years, 'Projected Total Revenue': proj_rev}).set_index('Year')
                    st.dataframe((proj_df / scale_factor).round(2).T, use_container_width=True)
            except Exception:
                st.info("Statement row variances are too high to compile reliable forward growth metrics.")

        # Data sheets tabs view
        st.subheader("📋 Core Financial Statements Grid")
        stmt_tab1, stmt_tab2, stmt_tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
        with stmt_tab1: st.dataframe(income_statement.dropna(how='all') if income_statement is not None else pd.DataFrame(), use_container_width=True)
        with stmt_tab2: st.dataframe(balance_sheet.dropna(how='all') if balance_sheet is not None else pd.DataFrame(), use_container_width=True)
        with stmt_tab3: st.dataframe(cash_flow.dropna(how='all') if cash_flow is not None else pd.DataFrame(), use_container_width=True)
