# app.py
import streamlit as st
# Import the custom analysis method from our engine.py file
from engine import process_stock_data

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📈 Stock Technical Analysis Platform")
st.write("Enter a stock ticker below to pull real-time data and generate a comprehensive technical health snapshot.")

# Input layout block form
with st.form(key="ticker_form"):
    ticker_input = st.text_input("Stock Ticker (e.g. AAPL, AMD, NVDA):", value="AAPL")
    submit_button = st.form_submit_button(label="Run Analysis")

if submit_button or ticker_input:
    with st.spinner(f"Requesting data matrix for {ticker_input.upper()}..."):
        # Call our processing script file engine
        result = process_stock_data(ticker_input)
        
        if result is None:
            st.error(f"Error: No trading data found for ticker '{ticker_input.upper()}'. Please check the token name.")
        else:
            # Unpack the figure and data payload dictionaries
            fig, metrics = result
            
            # ── Draw High-Level Metric Tiles ────────────────────────────────
            st.subheader(f"Summary Report for {metrics['ticker']}")
            
            trend = "bullish" if metrics['current_price'] > metrics['ma50'] else "bearish"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${metrics['current_price']:.2f}")
            with col2:
                st.metric(
                    label="RSI (14d)", 
                    value=f"{metrics['rsi_now']:.1f}", 
                    delta="Overbought" if metrics['rsi_now'] > 70 else "Oversold" if metrics['rsi_now'] < 30 else "Neutral", 
                    delta_color="inverse" if metrics['rsi_now'] > 70 else "normal"
                )
            with col3:
                st.metric("Max Drawdown (1Y)", f"{metrics['max_dd']:.1%}")

            # ── Context Insights Summary ──────────────────────────────────
            st.markdown(f"• Short-term trend is **{trend}** (price is {'above' if trend == 'bullish' else 'below'} the 50-day MA).")
            
            if isinstance(metrics['target_mean'], (int, float)):
                upside = (metrics['target_mean'] - metrics['current_price']) / metrics['current_price'] * 100
                st.markdown(f"• Wall Street projects **{upside:+.1f}%** upside to the average analyst target (${metrics['target_mean']:.2f}).")

            # ── Main Graphic Chart Render ─────────────────────────────────
            st.pyplot(fig)