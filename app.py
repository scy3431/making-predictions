import streamlit as st
import matplotlib.pyplot as plt
from engine import process_stock_data

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Technical Analysis Platform")
st.write("Enter a stock ticker below to pull real-time data and generate a comprehensive technical health snapshot.")

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form(key="ticker_form"):
    ticker_input  = st.text_input("Stock Ticker (e.g. AAPL, AMD, NVDA):", value="AAPL")
    submit_button = st.form_submit_button(label="Run Analysis")

if submit_button and ticker_input:
    with st.spinner(f"Fetching data for {ticker_input.upper()}..."):
        result = process_stock_data(ticker_input)

    if result is None:
        st.error(
            f"No trading data found for **'{ticker_input.upper()}'**. "
            "Please double-check the ticker symbol."
        )
        st.stop()

    fig, m = result

    # ── Helpers ───────────────────────────────────────────────────────────
    def fmt(val):
        return f"${val:.2f}" if isinstance(val, (int, float)) else "N/A"

    def fmt_cap(val):
        if val is None: return "N/A"
        if val >= 1e12: return f"${val/1e12:.2f}T"
        if val >= 1e9:  return f"${val/1e9:.2f}B"
        if val >= 1e6:  return f"${val/1e6:.2f}M"
        return f"${val:,.0f}"

    trend       = "bullish 📈" if m["current_price"] > m["ma50"] else "bearish 📉"
    trend_word  = "above" if "bullish" in trend else "below"

    # ── Company header ─────────────────────────────────────────────────────
    st.subheader(f"{m['company_name']}  ({m['ticker']})")
    st.caption(
        f"Short-term trend is **{trend}** — price is {trend_word} the 50-day MA."
    )
    if isinstance(m["target_mean"], (int, float)):
        upside = (m["target_mean"] - m["current_price"]) / m["current_price"] * 100
        st.caption(
            f"Wall Street consensus: **{upside:+.1f}%** upside to average analyst "
            f"target ({fmt(m['target_mean'])})."
        )

    st.divider()

    # ── Metric tiles — row 1 ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Current Price", f"${m['current_price']:.2f}")
    with c2:
        rsi = m["rsi_now"]
        rsi_delta = "Overbought ⚠" if rsi > 70 else "Oversold ⚠" if rsi < 30 else "Neutral ✓"
        rsi_color = "inverse" if rsi > 70 else "normal" if rsi < 30 else "off"
        st.metric("RSI (14d)", f"{rsi:.1f}", delta=rsi_delta, delta_color=rsi_color)
    with c3:
        st.metric("Max Drawdown (1Y)", f"{m['max_dd']:.1%}")
    with c4:
        st.metric("Market Cap", fmt_cap(m["market_cap"]))

    # ── Metric tiles — row 2 ───────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        pe = m["pe_ratio"]
        st.metric("Trailing P/E", f"{round(pe, 1)}x" if isinstance(pe, float) else "N/A")
    with c6:
        fpe = m["forward_pe"]
        st.metric("Forward P/E", f"{round(fpe, 1)}x" if isinstance(fpe, float) else "N/A")
    with c7:
        b = m["beta"]
        st.metric("Beta", f"{round(b, 2)}" if isinstance(b, float) else "N/A")
    with c8:
        dy = m["div_yield"]
        st.metric("Dividend Yield", f"{dy:.2%}" if dy else "—")

    st.divider()

    # ── Earnings beat / miss ──────────────────────────────────────────────
    st.subheader("🧾 Most Recent Earnings Report")
    earn = m.get("earnings")

    if earn:
        e1, e2, e3, e4 = st.columns(4)

        # Format the date
        date_val = earn["date"]
        try:
            date_str = date_val.strftime("%b %d, %Y")
        except AttributeError:
            date_str = str(date_val)[:10]

        with e1:
            st.metric("Report Date", date_str)

        with e2:
            est = earn["eps_estimate"]
            st.metric("EPS Estimate", f"${est:.2f}" if est is not None else "N/A")

        with e3:
            actual = earn["eps_actual"]
            if est is not None and actual is not None:
                delta_val   = actual - est
                delta_str   = f"{delta_val:+.2f} vs est."
                delta_color = "normal" if actual >= est else "inverse"
            else:
                delta_str, delta_color = None, "off"
            st.metric(
                "Reported EPS",
                f"${actual:.2f}" if actual is not None else "N/A",
                delta=delta_str,
                delta_color=delta_color,
            )

        with e4:
            beat = earn["beat"]
            surp = earn["surprise_pct"]
            surp_str = f"{surp:+.1f}% surprise" if surp is not None else ""
            if beat is True:
                st.success(f"✅ Beat  {surp_str}")
            elif beat is False:
                st.error(f"❌ Missed  {surp_str}")
            else:
                st.info("Result unavailable")
    else:
        st.info("Earnings data is unavailable for this ticker.")

    st.divider()

    # ── Analyst price targets ─────────────────────────────────────────────
    target_keys = ["target_low", "target_median", "target_mean", "target_high"]
    if all(isinstance(m.get(k), (int, float)) for k in target_keys):
        st.subheader("🎯 Analyst Price Targets")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.metric("Low",    fmt(m["target_low"]))
        with t2:
            st.metric("Median", fmt(m["target_median"]))
        with t3:
            st.metric("Mean",   fmt(m["target_mean"]))
        with t4:
            st.metric("High",   fmt(m["target_high"]))
        st.divider()

    # ── Main chart ────────────────────────────────────────────────────────
    st.subheader("📊 Technical Analysis Charts")
    st.pyplot(fig)
    plt.close(fig)  # free memory after render
