# Making Predictions

A Python-based stock analysis platform that combines **technical analysis, fundamental data, market statistics, and a Q-learning reinforcement-learning agent** into an interactive Streamlit dashboard.

The project is designed as an educational/data-science project for exploring how different types of market information can be combined into a single analysis workflow.

> **Disclaimer:** This project is for educational purposes only. It is not financial advice, and its outputs should not be interpreted as predictions or recommendations to buy or sell securities.

---

## What Does This Project Do?

The application takes a stock ticker such as:

AAPL
NVDA
AMD
MSFT
TSLA

and retrieves approximately one year of historical market data using Yahoo Finance.

It then calculates technical indicators, retrieves available company fundamentals and analyst estimates, creates a visual dashboard, and optionally trains a simple Q-learning trading agent.

The workflow looks like this:

                    Stock Ticker
                         │
                         ▼
                  ┌─────────────┐
                  │  Yahoo       │
                  │  Finance     │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  engine.py  │
                  └──────┬──────┘
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      Technical/Fundamental      Historical Data
          Analysis                    │
             │                       ▼
             ▼                ┌─────────────┐
      Streamlit Dashboard     │ rl_agent.py │
             │                └──────┬──────┘
             │                       │
             │                       ▼
             │                 HOLD / BUY / SELL
             │
             └───────────┬───────────────┘
                         │
                         ▼
                       User

---

# Project Structure

making-predictions/
│
├── app.py
├── engine.py
├── rl_agent.py
├── requirements.txt
└── README.md

## `app.py`

The main user interface.

It:

* Creates the Streamlit dashboard
* Accepts a stock ticker
* Calls `engine.py`
* Displays technical and fundamental metrics
* Displays the generated charts
* Provides an optional interface for training the Q-learning agent
* Displays the agent's resulting HOLD / BUY / SELL signal

You normally do **not** need to modify this file to analyze a different stock.

---

## `engine.py`

The main analysis engine.

It retrieves market data and calculates the features used throughout the application.

### Technical indicators

The engine calculates:

* 50-day moving average
* 200-day moving average
* Daily returns
* Bollinger Bands
* Bollinger Band percentage
* RSI
* MACD
* MACD signal line
* MACD histogram
* Maximum drawdown

### Fundamental information

When Yahoo Finance provides the information, the engine also retrieves:

* Market capitalization
* Trailing P/E
* Forward P/E
* Beta
* Dividend yield
* 52-week high
* 52-week low
* Analyst price targets
* Company name
* Recent earnings information

The engine ultimately returns:

fig, metrics

The `metrics` dictionary also contains the processed historical DataFrame under:

metrics["hist"]

This allows the reinforcement-learning agent to use the same processed data as the rest of the application.

---

## `rl_agent.py`

This file contains the reinforcement-learning component.

**This file is intentionally separate from the main analysis engine.**

The agent uses a Q-learning approach with a discretized state.

### State

The state consists of:

RSI
Bollinger Band position
MACD histogram
Current position

These continuous indicators are converted into discrete categories.

Conceptually:

┌─────────────┐
│     RSI     │
└──────┬──────┘
       │
       ├── Oversold
       ├── Neutral
       └── Overbought

┌─────────────┐
│  BB Position│
└──────┬──────┘
       │
       ├── Lower
       ├── Middle
       └── Upper

┌─────────────┐
│ MACD Signal │
└──────┬──────┘
       │
       ├── Negative
       └── Positive

┌─────────────┐
│   Position  │
└──────┬──────┘
       │
       ├── Not invested
       └── Invested

The agent then chooses between:

0 = HOLD
1 = BUY
2 = SELL

The Q-table is updated using the Bellman equation during training.

### Training

The agent walks through the historical price series repeatedly.

Exploration starts high and gradually decreases:

ε = 1.00
   ↓
   ↓
   ↓
ε = 0.01

This allows the agent to explore different actions early in training and rely more heavily on learned Q-values later.

---

# Installation

## 1. Clone the repository

git clone https://github.com/scy3431/making-predictions.git

Move into the project directory:

cd making-predictions

---

## 2. Create a virtual environment

### macOS / Linux

python3 -m venv .venv

Activate it:

source .venv/bin/activate

### Windows

python -m venv .venv

Activate it:

.venv\Scripts\activate

---

## 3. Install dependencies

Run:

pip install -r requirements.txt

The project currently relies on:

* Streamlit
* yfinance
* pandas
* matplotlib
* NumPy
* SciPy

---

# Running the Application

Once the dependencies are installed, run:

streamlit run app.py

Streamlit will provide a local address, usually similar to:

http://localhost:8501

Open that address in your browser.

---

# Using the Dashboard

## Step 1 — Enter a ticker

Enter a valid ticker symbol.

For example:

AAPL

Then click:

**Run Analysis**

---

## Step 2 — Review the overview

The dashboard displays:

* Current price
* RSI
* Maximum drawdown
* Market capitalization
* Trailing P/E
* Forward P/E
* Beta
* Dividend yield

It also gives a simple comparison between the current price and the 50-day moving average.

---

## Step 3 — Review earnings

If Yahoo Finance provides the information, the dashboard displays the most recent earnings report, including:

* Report date
* EPS estimate
* Reported EPS
* Earnings surprise
* Beat/miss status

---

## Step 4 — Review analyst targets

When available, the dashboard displays:

* Low target
* Median target
* Mean target
* High target

These are presented alongside the current stock price.

---

## Step 5 — Review technical charts

The dashboard generates several charts covering:

### Price

The primary chart contains:

* Closing price
* 50-day moving average
* 200-day moving average
* Bollinger Bands

### Volume

Shows daily trading volume and a smoothed volume line.

### Daily returns

Shows the distribution of daily percentage returns.

### RSI

Displays the 14-day RSI and highlights the traditional:

RSI > 70  → potentially overbought
RSI < 30  → potentially oversold

These thresholds should not be interpreted as automatic buy/sell signals.

### Analyst targets

Compares analyst price targets with the current price.

### Valuation snapshot

Provides a compact summary of fundamental and technical metrics.

---

# Using the Q-Learning Agent

The reinforcement-learning section is optional.

Expand:

**Train the Q-learning agent**

Choose the number of training episodes and click:

**Train Agent**

The application then passes the processed historical data from `engine.py` into:

train_rl_agent()

The trained agent produces:

HOLD
BUY
SELL

along with:

* Confidence
* Number of learned states
* Final exploration rate
* Current state
* Q-values
* Average return over the last 20 episodes
* Average Sharpe-like score over the last 20 episodes

---

# Understanding the Q-Values

Suppose the agent produces:

HOLD    0.12
BUY     0.38
SELL    0.05

The agent would choose:

BUY

because BUY has the highest Q-value.

The Q-value represents the reward the agent has learned to associate with an action in the current state.

It is **not** a probability that the stock will increase.

For example:

BUY Q-value = 0.38

does not mean:

38% chance the stock goes up

---

# Important Limitations

This project is intentionally relatively simple and should not be treated as a production trading system.

## 1. Historical data

The application relies on historical market data retrieved through Yahoo Finance.

Data availability and quality can vary by ticker.

---

## 2. In-sample reinforcement learning

The current RL implementation trains on the same historical dataset used for analysis.

That means its performance metrics are **not an out-of-sample test**.

A model can perform well on the data it trained on without being useful on future data.

---

## 3. Simplified trading environment

The reinforcement-learning environment is intentionally simplified.

It does not model every real-world trading consideration such as:

* Transaction costs
* Bid/ask spreads
* Slippage
* Taxes
* Market impact
* Position sizing
* Short selling
* Portfolio allocation

---

## 4. Limited state representation

The agent only uses a small set of discretized technical features.

Real financial markets depend on many other variables, including:

* Market regime
* Interest rates
* Earnings
* Economic conditions
* News
* Sector behavior
* Volatility
* Liquidity
* Investor sentiment

---

## 5. Technical indicators are not predictions

Indicators such as RSI, MACD, and moving averages describe characteristics of historical price behavior.

They do not guarantee future price movements.

---

# Troubleshooting

## `ModuleNotFoundError`

If you see an error such as:

ModuleNotFoundError: No module named 'streamlit'

make sure your virtual environment is activated and run:

pip install -r requirements.txt

---

## Yahoo Finance returns no data

If the application says:

No trading data found

check that:

1. The ticker is correct.
2. You have an internet connection.
3. Yahoo Finance is currently accessible.
4. The ticker actually has available historical market data.

---

## Some metrics display `N/A`

This is expected.

Yahoo Finance does not provide every fundamental field for every security.

The application is designed to continue running when optional information is unavailable.

---

## RL training takes a while

Increasing the number of training episodes increases the amount of work performed by the Q-learning agent.

For quick experiments, start with:

25–100 episodes

For longer experiments, increase the number gradually.

---

# Future Improvements

Potential next steps for the project include:

* Out-of-sample backtesting
* Transaction costs and slippage
* Portfolio-level reinforcement learning
* Position sizing
* Additional technical indicators
* Feature normalization
* Hyperparameter optimization
* Comparison against buy-and-hold
* Walk-forward validation
* Sharpe ratio and maximum drawdown comparisons
* Visualization of the agent's trades
* Persistent model saving/loading
* More advanced RL algorithms such as DQN

A particularly useful next step would be building a **proper backtesting environment**. That would make it possible to compare the learned strategy against a simple buy-and-hold benchmark rather than looking only at in-sample training performance.

---

# Technologies

This project uses:

| Technology | Purpose                           |
| ---------- | --------------------------------- |
| Python     | Core programming language         |
| Streamlit  | Interactive web application       |
| yfinance   | Market and company data           |
| pandas     | Data manipulation                 |
| NumPy      | Numerical computation             |
| SciPy      | Data smoothing                    |
| Matplotlib | Visualization                     |
| Q-learning | Reinforcement-learning experiment |

---

# Disclaimer

This project is an educational demonstration of financial data analysis and reinforcement learning.

It is **not financial advice**.

The technical indicators, analyst estimates, and reinforcement-learning outputs are not guaranteed to predict future market performance. Always perform independent research and consider the risks associated with investing before making financial decisions.

