import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
from scipy.ndimage import gaussian_filter1d


def analyze_stock(ticker_symbol):
    ticker_symbol = ticker_symbol.upper()
    print(f"\n=== {ticker_symbol} ===")

    ticker = yf.Ticker(ticker_symbol)
    hist   = ticker.history(period="1y")

    if hist.empty:
        print("Error: No data found. Check the ticker symbol.")
        return

    # ── Technical indicators ──────────────────────────────────────────────
    hist["MA50"]  = hist["Close"].rolling(50).mean()
    hist["MA200"] = hist["Close"].rolling(200).mean()
    hist["Daily_Return"] = hist["Close"].pct_change()

    # Bollinger Bands (20-day, 2σ)
    hist["BB_Mid"]   = hist["Close"].rolling(20).mean()
    hist["BB_Std"]   = hist["Close"].rolling(20).std()
    hist["BB_Upper"] = hist["BB_Mid"] + 2 * hist["BB_Std"]
    hist["BB_Lower"] = hist["BB_Mid"] - 2 * hist["BB_Std"]

    # RSI (14-day)
    delta         = hist["Close"].diff()
    gain          = delta.clip(lower=0).rolling(14).mean()
    loss          = -delta.clip(upper=0).rolling(14).mean()
    hist["RSI"]   = 100 - (100 / (1 + gain / loss))

    # Max drawdown
    roll_max = hist["Close"].cummax()
    max_dd   = ((hist["Close"] - roll_max) / roll_max).min()

    current_price = hist["Close"].iloc[-1]
    ma50          = hist["MA50"].iloc[-1]
    ma200         = hist["MA200"].iloc[-1]
    rsi_now       = hist["RSI"].iloc[-1]

    info          = ticker.info
    pe_ratio      = info.get("trailingPE",      "N/A")
    forward_pe    = info.get("forwardPE",       "N/A")
    target_mean   = info.get("targetMeanPrice", "N/A")
    target_high   = info.get("targetHighPrice", "N/A")
    target_low    = info.get("targetLowPrice",  "N/A")
    target_median = info.get("targetMedianPrice","N/A")
    market_cap    = info.get("marketCap",        None)
    beta          = info.get("beta",             "N/A")
    div_yield     = info.get("dividendYield",    None)
    week52_high   = info.get("fiftyTwoWeekHigh", "N/A")
    week52_low    = info.get("fiftyTwoWeekLow",  "N/A")

    def fmt(val):
        return f"${val:.2f}" if isinstance(val, (int, float)) else str(val)

    def fmt_cap(val):
        if val is None: return "N/A"
        if val >= 1e12: return f"${val/1e12:.2f}T"
        if val >= 1e9:  return f"${val/1e9:.2f}B"
        if val >= 1e6:  return f"${val/1e6:.2f}M"
        return f"${val:,.0f}"

    # ── Console report ───────────────────────────────────────────────────
    print(f"Current Price:       ${current_price:.2f}")
    print(f"50-Day MA:           ${ma50:.2f}")
    print(f"200-Day MA:          ${ma200:.2f}")
    print(f"RSI (14d):           {rsi_now:.1f}  {'⚠ Overbought' if rsi_now > 70 else '⚠ Oversold' if rsi_now < 30 else '✓ Neutral'}")
    print(f"Max Drawdown (1Y):   {max_dd:.1%}")
    print(f"Market Cap:          {fmt_cap(market_cap)}")
    print(f"Trailing P/E:        {pe_ratio}")
    print(f"Forward P/E:         {forward_pe}")
    print(f"Beta:                {round(beta, 2) if isinstance(beta, float) else beta}")
    print(f"Dividend Yield:      {f'{div_yield:.2%}' if div_yield else 'N/A'}")
    print(f"52-Week High / Low:  {fmt(week52_high)} / {fmt(week52_low)}")
    print("\n--- Wall Street Price Targets ---")
    print(f"  Average: {fmt(target_mean)}   Median: {fmt(target_median)}")
    print(f"  High:    {fmt(target_high)}   Low:    {fmt(target_low)}")
    print("\n--- Summary ---")
    trend = "bullish" if current_price > ma50 else "bearish"
    print(f"• Short-term {trend} (price {'above' if trend == 'bullish' else 'below'} 50-day MA)")
    if isinstance(target_mean, (int, float)):
        upside = (target_mean - current_price) / current_price * 100
        print(f"• Wall Street projects {upside:+.1f}% upside to average target")
    print(f"• Max 1Y drawdown: {max_dd:.1%}")

    # ── Smoothing helpers ─────────────────────────────────────────────────
    def smooth(series, sigma=2.5):
        arr  = series.to_numpy(dtype=float)
        mask = ~np.isnan(arr)
        out  = np.full_like(arr, np.nan)
        if mask.sum() > 5:
            out[mask] = gaussian_filter1d(arr[mask], sigma=sigma)
        return out

    close_s    = smooth(hist["Close"],                              sigma=2.5)
    ma50_s     = smooth(hist["MA50"].reindex(hist.index),           sigma=1.5)
    ma200_s    = smooth(hist["MA200"].reindex(hist.index),          sigma=1.5)
    bb_up_s    = smooth(hist["BB_Upper"].reindex(hist.index),       sigma=1.5)
    bb_lo_s    = smooth(hist["BB_Lower"].reindex(hist.index),       sigma=1.5)
    rsi_s      = smooth(hist["RSI"].reindex(hist.index),            sigma=1.5)
    vol_smooth = hist["Volume"].rolling(5, center=True).mean() / 1e6

    # ── Theme ─────────────────────────────────────────────────────────────
    DARK   = "#0d1117"
    MID    = "#161b22"
    GRID   = "#21262d"
    TEXT   = "#e6edf3"
    BLUE   = "#58a6ff"
    AMBER  = "#e3b341"
    RED    = "#f85149"
    GREEN  = "#3fb950"
    MUTED  = "#8b949e"
    PURPLE = "#bc8cff"

    def style_ax(ax):
        ax.set_facecolor(MID)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 14), facecolor=DARK, dpi=140)
    gs  = gridspec.GridSpec(
        4, 2, figure=fig,
        height_ratios=[2.5, 1.0, 1.0, 1.3],
        hspace=0.52, wspace=0.35,
    )

    # 1. Price + MAs + Bollinger Bands ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    ax1.fill_between(hist.index, bb_up_s, bb_lo_s,
                     alpha=0.07, color=PURPLE)
    ax1.plot(hist.index, bb_up_s, color=PURPLE, lw=0.75, alpha=0.55, label="Bollinger Bands")
    ax1.plot(hist.index, bb_lo_s, color=PURPLE, lw=0.75, alpha=0.55)
    ax1.fill_between(hist.index, close_s, alpha=0.08, color=BLUE)
    ax1.plot(hist.index, close_s,  color=BLUE,  lw=2.0, label="Close",      zorder=3)
    ax1.plot(hist.index, ma50_s,   color=AMBER, lw=1.2, linestyle="--",
             label="50-day MA",  zorder=2, alpha=0.9)
    ax1.plot(hist.index, ma200_s,  color=RED,   lw=1.2, linestyle=":",
             label="200-day MA", zorder=2, alpha=0.9)
    ax1.axhline(current_price, color=GREEN, lw=0.8, linestyle="-.", alpha=0.55, zorder=1)

    ax1.set_title(f"{ticker_symbol} — Price, Moving Averages & Bollinger Bands (1 Year)",
                  fontsize=11, pad=10)
    ax1.legend(facecolor=MID, edgecolor=GRID, labelcolor=TEXT,
               fontsize=8, framealpha=0.8, loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    style_ax(ax1)

    # 2. Volume ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    bar_colors = [GREEN if c >= o else RED
                  for c, o in zip(hist["Close"], hist["Open"])]
    ax2.bar(hist.index, hist["Volume"] / 1e6, color=bar_colors, width=1, alpha=0.45)
    ax2.plot(hist.index, vol_smooth, color=BLUE, lw=1.2, alpha=0.9)
    ax2.set_title("Volume (M shares)", fontsize=10)
    ax2.set_ylabel("Millions", fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    style_ax(ax2)

    # 3. Daily returns distribution ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    returns = hist["Daily_Return"].dropna() * 100
    n, bins, patches = ax3.hist(returns, bins=40, edgecolor=MID, alpha=0.85, linewidth=0.4)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(GREEN if left >= 0 else RED)
    ax3.axvline(0, color=TEXT, lw=0.9, linestyle="--")
    ax3.axvline(returns.mean(), color=AMBER, lw=1.0, linestyle="--", alpha=0.8,
                label=f"Mean {returns.mean():.2f}%")
    ax3.set_title("Daily Returns Distribution (%)", fontsize=10)
    ax3.set_xlabel("Return %", fontsize=8)
    ax3.legend(facecolor=MID, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
    style_ax(ax3)

    # 4. RSI ───────────────────────────────────────────────────────────────
    ax_rsi = fig.add_subplot(gs[2, :])

    rsi_safe = np.nan_to_num(rsi_s, nan=50.0)
    ax_rsi.plot(hist.index, rsi_s, color=BLUE, lw=1.5, zorder=3)
    ax_rsi.fill_between(hist.index, rsi_safe, 70,
                        where=rsi_safe > 70, color=RED,   alpha=0.22, interpolate=True)
    ax_rsi.fill_between(hist.index, rsi_safe, 30,
                        where=rsi_safe < 30, color=GREEN, alpha=0.22, interpolate=True)

    ax_rsi.axhline(70, color=RED,   lw=0.8, linestyle="--", alpha=0.6, label="Overbought (70)")
    ax_rsi.axhline(50, color=MUTED, lw=0.5, linestyle=":",  alpha=0.4)
    ax_rsi.axhline(30, color=GREEN, lw=0.8, linestyle="--", alpha=0.6, label="Oversold (30)")
    ax_rsi.axhline(rsi_now, color=AMBER, lw=0.8, linestyle="-.", alpha=0.5)
    ax_rsi.text(hist.index[-1], rsi_now + 1.5, f"  {rsi_now:.1f}",
                color=AMBER, fontsize=8, va="bottom")

    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title("RSI (14-day)", fontsize=10)
    ax_rsi.set_ylabel("RSI", fontsize=8)
    ax_rsi.legend(facecolor=MID, edgecolor=GRID, labelcolor=TEXT,
                  fontsize=7, loc="upper left")
    style_ax(ax_rsi)

    # 5. Analyst price targets ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3, 0])
    target_vals = [target_low, target_median, target_mean, target_high, current_price]
    if all(isinstance(v, (int, float)) for v in target_vals):
        labels     = ["Low", "Median", "Mean", "High", "Current"]
        tgt_colors = [RED, AMBER, AMBER, GREEN, BLUE]
        bars = ax4.bar(labels, target_vals, color=tgt_colors, width=0.5,
                       edgecolor=DARK, linewidth=0.5)
        y_min = min(target_vals) * 0.95
        ax4.set_ylim(bottom=y_min)
        for bar, val in zip(bars, target_vals):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (max(target_vals) - y_min) * 0.01,
                     f"${val:.0f}", ha="center", va="bottom", color=TEXT, fontsize=8)
        ax4.set_title("Analyst Price Targets vs Current", fontsize=10)
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    else:
        ax4.text(0.5, 0.5, "Targets unavailable", ha="center", va="center",
                 transform=ax4.transAxes, color=MUTED, fontsize=10)
    style_ax(ax4)

    # 6. Valuation snapshot ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis("off")

    upside_str = (f"{((target_mean - current_price) / current_price * 100):+.1f}%"
                  if isinstance(target_mean, float) else "N/A")
    rsi_tag    = "OB ⚠" if rsi_now > 70 else "OS ⚠" if rsi_now < 30 else "Neutral"
    w52_str    = f"{fmt(week52_high)} / {fmt(week52_low)}"

    rows = [
        ("Current Price",  f"${current_price:.2f}"),
        ("Market Cap",     fmt_cap(market_cap)),
        ("Trailing P/E",   f"{round(pe_ratio, 1)}x"   if isinstance(pe_ratio,   float) else "N/A"),
        ("Forward P/E",    f"{round(forward_pe, 1)}x"  if isinstance(forward_pe, float) else "N/A"),
        ("Beta",           f"{round(beta, 2)}"          if isinstance(beta,       float) else "N/A"),
        ("Div. Yield",     f"{div_yield:.2%}"           if div_yield else "—"),
        ("52W High / Low", w52_str),
        ("Max Drawdown",   f"{max_dd:.1%}"),
        ("Upside (mean)",  upside_str),
        ("RSI (14d)",      f"{rsi_now:.1f} — {rsi_tag}"),
        ("50d Signal",     "▲ Bullish" if current_price > ma50  else "▼ Bearish"),
        ("200d Signal",    "▲ Bullish" if current_price > ma200 else "▼ Bearish"),
    ]

    top_y = 0.97
    step  = top_y / len(rows)

    for i, (label, value) in enumerate(rows):
        y = top_y - i * step

        # Pick value colour
        if "▲" in value:
            vcol = GREEN
        elif "▼" in value:
            vcol = RED
        elif label == "RSI (14d)":
            vcol = RED if rsi_now > 70 else GREEN if rsi_now < 30 else AMBER
        elif label == "Max Drawdown":
            vcol = RED
        elif label == "Market Cap":
            vcol = AMBER
        elif "%" in value:
            vcol = GREEN if value.startswith("+") else RED if value.startswith("-") else TEXT
        else:
            vcol = TEXT

        ax5.text(0.04, y, label, transform=ax5.transAxes,
                 color=MUTED, fontsize=8, va="top")
        ax5.text(0.57, y, value, transform=ax5.transAxes,
                 color=vcol, fontsize=8, va="top", fontweight="bold")

        if i < len(rows) - 1:
            ax5.axhline(y - step * 0.28, xmin=0.03, xmax=0.97,
                        color=GRID, linewidth=0.4)

    ax5.set_title("Snapshot", fontsize=10, color=TEXT, pad=8)

    fig.suptitle(f"{ticker_symbol} — Analysis Dashboard",
                 fontsize=14, color=TEXT, y=0.999, fontweight="500")
    plt.show()

if __name__ == "__main__":
    while (t := "".join(input("\nEnter stock ticker ('leave' to exit): ").lower().split())) != "leave":
        analyze_stock(t)
