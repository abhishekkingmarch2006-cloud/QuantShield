import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Global Financial Command Center", layout="wide")
st.title("🌐 Global Stock Fundamental, Technical & 5-Year Forecasting Hub")
st.write("Analyze any stock worldwide. Enter the exact global ticker symbol to pull company branding, descriptions, core statements, technical charts, and 5-year predictive engines.")

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
        # Pull 5 years of daily historical data to support both technical charts and long-term ARIMA forecasting models
        hist_data = ticker_obj.history(period="5y")
        
        # Pull company profile details Safely
        try:
            info = ticker_obj.info
        except Exception:
            info = {}

    if hist_data.empty:
        st.error(f"❌ Could not retrieve market data for '{ticker_input}'. Please check the ticker formatting suffix rules on the sidebar.")
    else:
        # Flatten MultiIndex columns if returned by yfinance
        if isinstance(hist_data.columns, pd.MultiIndex):
            hist_data.columns = [col[0] for col in hist_data.columns]
        
        hist_data = hist_data.loc[:, ~hist_data.columns.duplicated()].copy()

        # -------------------------------------------------------------
        # BRANDING HEADER: LOGO & DESCRIPTION
        # -------------------------------------------------------------
        st.markdown("---")
        head_col1, head_col2 = st.columns([1, 5])
        
        company_name = info.get('longName', ticker_upper)
        website = info.get('website', '')
        currency_label = info.get('currency', 'Units')
        
        with head_col1:
            if website:
                clean_domain = website.replace("https://", "").replace("http://", "").replace("www.", "").split('/')[0]
                logo_url = f"https://logo.clearbit.com/{clean_domain}?size=120"
                try:
                    st.image(logo_url, width=120)
                except Exception:
                    st.markdown("## 🏢")
            else:
                st.markdown("## 🏢")
                
        with head_col2:
            st.title(company_name)
            if website:
                st.markdown(f"🔗 [Visit Official Website]({website})")
                
        st.subheader("📋 Company Profile & Business Summary")
        business_summary = info.get('longBusinessSummary', 'No company description profile summary found.')
        st.write(business_summary)
        
        # -------------------------------------------------------------
        # PART 1: TECHNICAL ANALYSIS ENGINE & 5-YEAR PRICE FORECAST
        # -------------------------------------------------------------
        st.markdown("---")
        st.header("📈 Part 1: TradingView Technical Analysis & 5-Year Price Forecast")
        
        df = hist_data.copy()
        close_series = pd.Series(df['Close'].values, index=df.index).astype(float)
        
        # Calculate Technical Indicators (using the last 1 year of data for clear visualization)
        df_1y = df.tail(252).copy()
        close_1y = pd.Series(df_1y['Close'].values, index=df_1y.index).astype(float)
        df_1y['SMA_50'] = ta.trend.sma_indicator(close_1y, window=50)
        df_1y['EMA_20'] = ta.trend.ema_indicator(close_1y, window=20)
        df_1y['RSI'] = ta.momentum.rsi(close_1y, window=14)

        latest_close = float(close_series.iloc[-1])
        latest_rsi = float(df_1y['RSI'].iloc[-1]) if not pd.isna(df_1y['RSI'].iloc[-1]) else 50.0
        latest_sma50 = df_1y['SMA_50'].iloc[-1]
        
        t_col1, t_col2, t_col3 = st.columns(3)
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

        # Technical Charts
        st.subheader("📉 1-Year Technical Indicators Window")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(df_1y.index, close_1y.values, label="Close Price", color="black", linewidth=1.5)
        ax1.plot(df_1y.index, df_1y['EMA_20'].values, label="20 EMA", color="orange", linestyle="--")
        if not pd.isna(latest_sma50):
            ax1.plot(df_1y.index, df_1y['SMA_50'].values, label="50 SMA", color="blue", linestyle="-.")
        ax1.set_ylabel(f"Price ({currency_label})")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(df_1y.index, df_1y['RSI'].values, label="RSI", color="purple")
        ax2.axhline(70, color="red", linestyle=":")
        ax2.axhline(30, color="green", linestyle=":")
        ax2.set_ylabel("RSI Range")
        ax2.set_ylim(10, 90)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        # 🚀 5-YEAR ARIMA PRICE FORECASTING SUB-ENGINE
        st.subheader("🔮 5-Year Technical Price Projection (ARIMA)")
        with st.spinner("Running Auto-ARIMA optimization models across 5 years of timeline history..."):
            try:
                # Resample historical data to weekly frequencies to handle long horizon limits safely
                df_weekly = close_series.resample('W-MON').mean().dropna()
                model = auto_arima(df_weekly, seasonal=False, error_action='ignore', suppress_warnings=True)
                fitted_model = model.fit(df_weekly)
                
                # Forecast 260 weeks forward (~5 Years into 2031)
                forecast_periods = 260
                forecast_values, conf_int = fitted_model.predict(n_periods=forecast_periods, return_conf_int=True)
                forecast_index = pd.date_range(start=df_weekly.index[-1] + pd.DateOffset(weeks=1), periods=forecast_periods, freq='W-MON')
                
                forecast_series = pd.Series(forecast_values, index=forecast_index)
                
                f_col1, f_col2 = st.columns([2, 1])
                with f_col1:
                    fig_arima, ax_arima = plt.subplots(figsize=(10, 5))
                    ax_arima.plot(df_weekly.index, df_weekly.values, label="Historical Close", color="blue")
                    ax_arima.plot(forecast_series.index, forecast_series.values, label="ARIMA Forecast", color="red", linestyle="--")
                    ax_arima.fill_between(forecast_series.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
                    ax_arima.set_title(f"5-Year Price Trajectory Model (ARIMA Order: {model.order})")
                    ax_arima.set_ylabel(f"Price ({currency_label})")
                    ax_arima.legend(loc="upper left")
                    ax_arima.grid(True, alpha=0.3)
                    st.pyplot(fig_arima)
                    plt.close(fig_arima)
                
                with f_col2:
                    fig_hist, ax_hist = plt.subplots(figsize=(5, 5))
                    sns.histplot(forecast_series, kde=True, ax=ax_hist, color="crimson", bins=15)
                    ax_hist.set_title("Forecast Price Distribution Density")
                    ax_hist.set_xlabel(f"Projected Price ({currency_label})")
                    st.pyplot(fig_hist)
                    plt.close(fig_hist)
                    
                # Show annual checkpoint text targets
                annual_targets = forecast_series.resample('YE').last()
                st.markdown("**Predicted Long-Term Price Checkpoints:**")
                checkpoint_cols = st.columns(len(annual_targets.index))
                for idx, year_date in enumerate(annual_targets.index):
                    checkpoint_cols[idx].metric(f"Year {year_date.strftime('%Y')} Target", f"{annual_targets.iloc[idx]:.2f} {currency_label}")
            except Exception as e:
                st.warning(f"Technical forecasting model computation timed out: {e}")

        # -------------------------------------------------------------
        # PART 2: FUNDAMENTALS ENGINE & 5-YEAR GROWTH PROJECTIONS
        # -------------------------------------------------------------
        st.markdown("---")
        st.header("📊 Part 2: Screener Valuation, Statements & 5-Year Fundamental Growth Forecast")
        
        income_statement = ticker_obj.financials
        balance_sheet = ticker_obj.balance_sheet
        cash_flow = ticker_obj.cashflow

        # Valuation Metric Row
        if info:
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            raw_market_cap = info.get('marketCap', 0)
            
            if raw_market_cap:
                if currency_label == "INR":
                    formatted_mcap = f"₹ {raw_market_cap / 10000000:.2f} Cr"
                else:
                    formatted_mcap = f"{raw_market_cap / 1000000000:.2f} B {currency_label}"
            else:
                formatted_mcap = "N/A"
                
            f_col1.metric("Market Cap (Local Currency)", formatted_mcap)
            f_col2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            f_col3.metric("P/B Ratio", f"{info.get('priceToBook', 'N/A')}")
            f_col4.metric("Return on Equity (ROE)", f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A")

        # 🚀 5-YEAR FUNDAMENTAL FORECAST SUB-ENGINE (CAGR PROJECTIONS)
        st.subheader("📈 5-Year Corporate Revenue & Profit Projections")
        if income_statement is not None and not income_statement.empty and income_statement.shape[1] >= 2:
            try:
                # Extract Revenue and Net Income rows safely matching modern yfinance layouts
                revenue_row = income_statement.loc['Total Revenue'] if 'Total Revenue' in income_statement.index else income_statement.loc['Revenue'] if 'Revenue' in income_statement.index else pd.Series()
                net_income_row = income_statement.loc['Net Income'] if 'Net Income' in income_statement.index else pd.Series()
                
                if not revenue_row.empty and len(revenue_row) >= 2:
                    # Clean arrays ordered chronological (oldest to newest)
                    rev_values = revenue_row.dropna().values[::-1]
                    net_values = net_income_row.dropna().values[::-1] if not net_income_row.empty else []
                    
                    # Calculate basic 3-Year CAGR trend
                    n_years = min(3, len(rev_values) - 1)
                    cagr_rev = ((rev_values[-1] / rev_values[-1 - n_years]) ** (1 / n_years)) - 1
                    
                    # Prevent extreme CAGR distortions from skewed volatile data years
                    cagr_rev = max(min(cagr_rev, 0.25), -0.15) 
                    
                    # Build 5-Year projections data frames
                    base_year = datetime.today().year
                    projection_years = [str(base_year + i) for i in range(1, 6)]
                    
                    projected_revenue = []
                    current_rev = rev_values[-1]
                    for i in range(5):
                        current_rev *= (1 + cagr_rev)
                        projected_revenue.append(current_rev)
                        
                    fund_df = pd.DataFrame({
                        'Year': projection_years,
                        'Projected Total Revenue': projected_revenue
                    }).set_index('Year')
                    
                    # Scale display labels cleanly based on local currency denominations
                    scale_factor = 10000000 if currency_label == "INR" else 1000000000
                    scale_text = "Crores (Cr)" if currency_label == "INR" else "Billions (B)"
                    
                    st.write(f"Projections calculated using a baseline historical revenue trend CAGR of **{cagr_rev*100:.2f}%**. Figures are adjusted in **{scale_text}**:")
                    
                    # Display metrics row
                    st.dataframe((fund_df / scale_factor).round(2).T, use_container_width=True)
            except Exception as fe:
                st.info("Fundamental metrics structure is too volatile to compute standardized CAGR projections automatically.")
        else:
            st.info("Insufficient statement history rows to compile 5-year CAGR growth trajectories.")

        # Core Data Sheets Table Tabs
        st.subheader("📋 Core Financial Statements Data Grid")
        tab1, tab2, tab3 = st.tabs(["Income Statement (P&L)", "Balance Sheet", "Cash Flow Statement"])
        with tab1:
            st.dataframe(income_statement.dropna(how='all') if income_statement is not None else pd.DataFrame(), use_container_width=True)
        with tab2:
            st.dataframe(balance_sheet.dropna(how='all') if balance_sheet is not None else pd.DataFrame(), use_container_width=True)
        with tab3:
            st.dataframe(cash_flow.dropna(how='all') if cash_flow is not None else pd.DataFrame(), use_container_width=True)
