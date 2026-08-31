import streamlit as st
import matplotlib.pyplot as plt

from engine import process_stock_data
from rl_agent import train_rl_agent

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
)


st.title("📈 Stock Technical Analysis Platform")

st.write(
    "Enter a stock ticker to pull historical market data and generate a "
    "technical and fundamental analysis dashboard."
)

with st.form(key="ticker_form"):
    ticker_input = st.text_input(
        "Stock Ticker (e.g. AAPL, AMD, NVDA):",
        value="AAPL",
    )

    submit_button = st.form_submit_button(label="Run Analysis")


if submit_button and ticker_input:

    ticker_symbol = ticker_input.strip().upper()

    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        result = process_stock_data(ticker_symbol)

    if result is None:
        st.error(
            f"No trading data found for **'{ticker_symbol}'**. "
            "Please double-check the ticker symbol."
        )
        st.stop()

    fig, metrics = result

    def fmt_price(value):
        if isinstance(value, (int, float)):
            return f"${value:.2f}"
        return "N/A"

    def fmt_market_cap(value):
        if value is None:
            return "N/A"

        if value >= 1e12:
            return f"${value / 1e12:.2f}T"

        if value >= 1e9:
            return f"${value / 1e9:.2f}B"

        if value >= 1e6:
            return f"${value / 1e6:.2f}M"

        return f"${value:,.0f}"

    current_price = metrics["current_price"]
    ma50 = metrics["ma50"]

    trend = "bullish " if current_price > ma50 else "bearish 📉"
    trend_word = "above" if current_price > ma50 else "below"

    st.subheader(f"{metrics['company_name']} ({metrics['ticker']})")

    st.caption(
        f"Short-term trend is **{trend}** — "
        f"price is {trend_word} the 50-day moving average."
    )

    target_mean = metrics["target_mean"]

    if isinstance(target_mean, (int, float)):
        upside = (target_mean - current_price) / current_price * 100

        st.caption(
            f"Analyst consensus target: **{upside:+.1f}%** "
            f"relative to the current price "
            f"({fmt_price(target_mean)} average target)."
        )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
        )

    with c2:
        rsi = metrics["rsi_now"]

        rsi_status = (
            "Overbought ⚠" if rsi > 70 else "Oversold ⚠" if rsi < 30 else "Neutral ✓"
        )

        rsi_color = "inverse" if rsi > 70 else "normal" if rsi < 30 else "off"

        st.metric(
            "RSI (14d)",
            f"{rsi:.1f}",
            delta=rsi_status,
            delta_color=rsi_color,
        )

    with c3:
        st.metric(
            "Max Drawdown (1Y)",
            f"{metrics['max_dd']:.1%}",
        )

    with c4:
        st.metric(
            "Market Cap",
            fmt_market_cap(metrics["market_cap"]),
        )

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        pe = metrics["pe_ratio"]

        st.metric(
            "Trailing P/E",
            f"{round(pe, 1)}x" if isinstance(pe, (int, float)) else "N/A",
        )

    with c6:
        forward_pe = metrics["forward_pe"]

        st.metric(
            "Forward P/E",
            f"{round(forward_pe, 1)}x"
            if isinstance(forward_pe, (int, float))
            else "N/A",
        )

    with c7:
        beta = metrics["beta"]

        st.metric(
            "Beta",
            f"{round(beta, 2)}" if isinstance(beta, (int, float)) else "N/A",
        )

    with c8:
        dividend_yield = metrics["div_yield"]

        st.metric(
            "Dividend Yield",
            f"{dividend_yield:.2%}" if dividend_yield else "—",
        )

    st.divider()

    st.subheader("Most Recent Earnings Report")

    earnings = metrics.get("earnings")

    if earnings:

        e1, e2, e3, e4 = st.columns(4)

        date_value = earnings["date"]

        try:
            date_string = date_value.strftime("%b %d, %Y")
        except AttributeError:
            date_string = str(date_value)[:10]

        with e1:
            st.metric(
                "Report Date",
                date_string,
            )

        with e2:
            estimate = earnings["eps_estimate"]

            st.metric(
                "EPS Estimate",
                f"${estimate:.2f}" if estimate is not None else "N/A",
            )

        with e3:
            actual = earnings["eps_actual"]

            if estimate is not None and actual is not None:

                difference = actual - estimate

                delta_string = f"{difference:+.2f} vs est."

                delta_color = "normal" if actual >= estimate else "inverse"

            else:
                delta_string = None
                delta_color = "off"

            st.metric(
                "Reported EPS",
                f"${actual:.2f}" if actual is not None else "N/A",
                delta=delta_string,
                delta_color=delta_color,
            )

        with e4:

            beat = earnings["beat"]
            surprise = earnings["surprise_pct"]

            surprise_string = (
                f"{surprise:+.1f}% surprise" if surprise is not None else ""
            )

            if beat is True:

                st.success(f"Beat {surprise_string}")

            elif beat is False:

                st.error(f"Missed {surprise_string}")

            else:

                st.info("Result unavailable")

    else:

        st.info("Earnings data is unavailable for this ticker.")

    st.divider()

    target_keys = [
        "target_low",
        "target_median",
        "target_mean",
        "target_high",
    ]

    targets_available = all(
        isinstance(metrics.get(key), (int, float)) for key in target_keys
    )

    if targets_available:

        st.subheader("Analyst Price Targets")

        t1, t2, t3, t4 = st.columns(4)

        with t1:
            st.metric(
                "Low",
                fmt_price(metrics["target_low"]),
            )

        with t2:
            st.metric(
                "Median",
                fmt_price(metrics["target_median"]),
            )

        with t3:
            st.metric(
                "Mean",
                fmt_price(metrics["target_mean"]),
            )

        with t4:
            st.metric(
                "High",
                fmt_price(metrics["target_high"]),
            )

        st.divider()

    st.subheader("📊 Technical Analysis Charts")
    st.pyplot(fig)
    plt.close(fig)

    # Reinforcement Learning Agent
    st.divider()

    st.subheader("Q-Learning Trading Agent")

    st.write(
        "The reinforcement-learning agent uses the technical indicators "
        "already calculated by the analysis engine to learn a simple "
        "HOLD / BUY / SELL policy."
    )

    st.caption(
        "The agent uses RSI, Bollinger Band position, MACD histogram, "
        "and current position as its state."
    )

    with st.expander(
        "Train the Q-learning agent",
        expanded=False,
    ):

        st.warning(
            "Training is performed in-sample using the historical data "
            "displayed above. The resulting signal should be treated as "
            "an experimental model output, not an investment recommendation."
        )

        episodes = st.slider(
            "Training episodes",
            min_value=25,
            max_value=250,
            value=100,
            step=25,
        )

        train_button = st.button(
            "Train Agent",
            type="secondary",
        )

        if train_button:

            with st.spinner(f"Training agent for {episodes} episodes..."):

                rl_result = train_rl_agent(
                    metrics["hist"],
                    n_episodes=episodes,
                )

            training = rl_result["training"]
            recommendation = rl_result["recommendation"]

            # Results
            st.success("Agent training completed.")

            r1, r2, r3, r4 = st.columns(4)

            with r1:
                st.metric(
                    "Recommendation",
                    recommendation["action"],
                )

            with r2:
                st.metric(
                    "Confidence",
                    f"{recommendation['confidence']:.1%}",
                )

            with r3:
                st.metric(
                    "States Learned",
                    training["q_table_states"],
                )

            with r4:
                st.metric(
                    "Final ε",
                    f"{training['final_epsilon']:.3f}",
                )

            st.markdown("### Current RL State")

            st.write("The state is represented as:")

            st.code(recommendation["state"])

            st.markdown("### Q-Values")

            q_values = recommendation["q_values"]

            q1, q2, q3 = st.columns(3)

            with q1:
                st.metric(
                    "HOLD",
                    f"{q_values['HOLD']:.4f}",
                )

            with q2:
                st.metric(
                    "BUY",
                    f"{q_values['BUY']:.4f}",
                )

            with q3:
                st.metric(
                    "SELL",
                    f"{q_values['SELL']:.4f}",
                )

            st.markdown("### Training Performance")

            p1, p2 = st.columns(2)

            with p1:
                st.metric(
                    "Avg. Return — Last 20 Episodes",
                    f"{training['avg_return_last20']:.2%}",
                )

            with p2:
                st.metric(
                    "Avg. Sharpe-like Score — Last 20",
                    f"{training['avg_sharpe_last20']:.3f}",
                )

            st.info(
                "A higher Q-value means the trained agent assigned a "
                "higher expected reward to that action for the current "
                "state. This does not mean the action will be profitable."
            )
