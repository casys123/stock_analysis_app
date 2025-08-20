import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# ----------------------------
# Page / Styles
# ----------------------------
st.set_page_config(page_title="StockInsight Pro", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #1E88E5; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #0D47A1; border-bottom: 2px solid #64B5F6;
                  padding-bottom: 0.5rem; margin-top: 1.5rem; margin-bottom: 1rem; }
    .positive { color: #2E7D32; font-weight: bold; }
    .negative { color: #C62828; font-weight: bold; }
    .metric-card { background-color: #BBDEFB; border-radius: 8px; padding: 1rem; text-align: center; margin: 0.5rem; }
    .recommendation-buy { background-color: #E8F5E9; color: #2E7D32; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; text-align: center; }
    .recommendation-hold { background-color: #FFF8E1; color: #F57C00; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; text-align: center; }
    .recommendation-sell { background-color: #FFEBEE; color: #C62828; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; text-align: center; }
    .news-card { background-color: #FFF3E0; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 StockInsight Pro</h1>', unsafe_allow_html=True)
st.markdown("### Your All-in-One Stock Analysis Dashboard")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Stock Selection")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", "AAPL").strip().upper()
analysis_type = st.sidebar.selectbox("Select Analysis Type:", ["Comprehensive Analysis", "Technical Analysis", "Fundamental Analysis", "Options Analysis"])
time_frame = st.sidebar.selectbox("Select Time Frame:", ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "5Y"])
st.sidebar.markdown("---")
st.sidebar.info("💡 Aggregates Yahoo data with technicals and a simple rules-based view.")

# ----------------------------
# Helpers
# ----------------------------
def map_timeframe(tf: str):
    """
    Map UI time frame to (period, interval) acceptable by yfinance.
    Use slightly longer period for indicator stability.
    """
    tf = tf.upper()
    if tf == "1D":
        return ("5d", "5m")      # intraday needs interval
    if tf == "1W":
        return ("1mo", "30m")
    if tf == "1M":
        return ("3mo", "1d")
    if tf == "3M":
        return ("6mo", "1d")
    if tf == "6M":
        return ("1y", "1d")
    if tf == "YTD":
        # compute from Jan 1 to today with 1d bars
        return ("ytd", "1d")
    if tf == "1Y":
        return ("1y", "1d")
    if tf == "5Y":
        return ("5y", "1d")
    return ("1y", "1d")

@st.cache_data(show_spinner=False, ttl=300)
def fetch_yahoo_data(ticker: str, tf: str):
    period, interval = map_timeframe(tf)
    t = yf.Ticker(ticker)

    # History
    hist = pd.DataFrame()
    try:
        hist = t.history(period=period, interval=interval, auto_adjust=False)
        # Drop empty columns if any
        if not hist.empty and "Close" in hist:
            hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            hist = hist[~hist.index.duplicated(keep="last")]
    except Exception as e:
        st.error(f"Yahoo history error: {e}")

    # Fast info first; fall back to .info and history
    current_price = None
    previous_close = None
    mcap = None
    pe = None
    eps = None
    profit_margins = None
    roe = None
    dividend_yield = None
    week52_high = None
    sector = "N/A"
    industry = "N/A"

    try:
        fi = t.fast_info  # lightweight
        current_price = fi.get("last_price")
        previous_close = fi.get("previous_close")
        mcap = fi.get("market_cap")
        week52_high = fi.get("year_high")
    except Exception:
        pass

    # Fallback to info (can be slow / partial)
    try:
        info = t.info or {}
        sector = info.get("sector", sector)
        industry = info.get("industry", industry)
        mcap = mcap if mcap is not None else info.get("marketCap")
        pe = info.get("trailingPE")
        eps = info.get("trailingEps")
        profit_margins = info.get("profitMargins")
        roe = info.get("returnOnEquity")
        dividend_yield = info.get("dividendYield")
        if current_price is None and not hist.empty:
            current_price = float(hist["Close"].iloc[-1])
        if previous_close is None and len(hist) >= 2:
            previous_close = float(hist["Close"].iloc[-2])
    except Exception:
        info = {}

    # Recommendations (safe)
    try:
        recs = t.recommendations
    except Exception:
        recs = None

    return {
        "history": hist,
        "info": info,
        "fast_info": fi if 'fi' in locals() else {},
        "current_price": current_price,
        "previous_close": previous_close if previous_close is not None else current_price,
        "market_cap": mcap,
        "pe": pe,
        "eps": eps,
        "profit_margins": profit_margins,
        "roe": roe,
        "dividend_yield": dividend_yield,
        "week52_high": week52_high,
        "sector": sector,
        "industry": industry,
        "recommendations": recs,
    }

def fetch_financial_news(ticker: str):
    # Placeholder; wire a real news API later
    return [
        {"title": f"{ticker} — Sample headline", "source": "Demo", "date": "2025-01-01",
         "summary": "Connect a real news API here (e.g., Polygon/NewsAPI) and filter by the ticker."}
    ]

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Close" not in df:
        return df

    out = df.copy()

    # MAs
    out["MA20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["MA50"] = out["Close"].rolling(50, min_periods=50).mean()

    # RSI (Wilder-ish)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.ewm(alpha=1/14, adjust=False).mean()
    roll_down = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    out["RSI"] = out["RSI"].clip(0, 100)

    # MACD
    exp12 = out["Close"].ewm(span=12, adjust=False).mean()
    exp26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = exp12 - exp26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Histogram"] = out["MACD"] - out["MACD_Signal"]

    return out

def generate_recommendation(data: dict, tech: pd.DataFrame) -> str:
    if not data or tech.empty:
        return "Insufficient data"

    cp = data.get("current_price")
    pc = data.get("previous_close") or cp
    if cp in (None, 0) or pc in (None, 0):
        return "Hold"

    price_change_pct = (cp - pc) / pc * 100 if pc else 0
    rsi = float(tech["RSI"].dropna().iloc[-1]) if "RSI" in tech and not tech["RSI"].dropna().empty else 50

    score = 0
    # Momentum
    if price_change_pct > 2: score += 1
    elif price_change_pct < -2: score -= 1
    # RSI
    if rsi < 30: score += 1
    elif rsi > 70: score -= 1

    if score >= 2: return "Strong Buy"
    if score == 1: return "Buy"
    if score == 0: return "Hold"
    if score == -1: return "Sell"
    return "Strong Sell"

def create_price_chart(df: pd.DataFrame, ticker: str):
    if df.empty: return None
    rcParams["figure.figsize"] = (12, 6)
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label="Close", linewidth=2, color="#1E88E5")
    if "MA20" in df: ax.plot(df.index, df["MA20"], "--", label="20-Day MA", color="#FF9800")
    if "MA50" in df: ax.plot(df.index, df["MA50"], "--", label="50-Day MA", color="#D81B60")
    ax.set_title(f"{ticker} Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    return fig

def create_technical_indicators_chart(df: pd.DataFrame, ticker: str):
    if df.empty or "RSI" not in df or "MACD" not in df: return None
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # RSI
    ax1.plot(df.index, df["RSI"], label="RSI", linewidth=2, color="#26A69A")
    ax1.axhline(70, color="r", linestyle="--", alpha=0.7, label="Overbought (70)")
    ax1.axhline(30, color="g", linestyle="--", alpha=0.7, label="Oversold (30)")
    ax1.set_title(f"{ticker} RSI (14)")
    ax1.set_ylabel("RSI")
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MACD
    ax2.plot(df.index, df["MACD"], label="MACD", linewidth=2, color="#7B1FA2")
    ax2.plot(df.index, df["MACD_Signal"], label="Signal", linewidth=2, color="#FF9800")
    ax2.bar(df.index, df["MACD_Histogram"], label="Histogram", alpha=0.3, color="#26A69A")
    ax2.set_title(f"{ticker} MACD")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    return fig

# ----------------------------
# Main
# ----------------------------
def main():
    with st.spinner("Fetching data…"):
        data = fetch_yahoo_data(ticker_symbol, time_frame)
        hist = data["history"] if data else pd.DataFrame()
        tech = calculate_technical_indicators(hist) if not hist.empty else pd.DataFrame()
        news = fetch_financial_news(ticker_symbol)
        reco = generate_recommendation(data, tech)

    # Header KPIs
    if data:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.markdown(f"### {ticker_symbol}")
            st.markdown(f"**Sector:** {data.get('sector','N/A')}")
            st.markdown(f"**Industry:** {data.get('industry','N/A')}")
        with c2:
            cp = data.get("current_price")
            pc = data.get("previous_close") or cp
            if cp is not None and pc:
                delta = cp - pc
                pct = (delta/pc*100) if pc else 0
                st.markdown(f"### ${cp:,.2f}")
                if delta >= 0:
                    st.markdown(f"<p class='positive'>+{delta:,.2f} ({pct:.2f}%)</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='negative'>{delta:,.2f} ({pct:.2f}%)</p>", unsafe_allow_html=True)
            else:
                st.markdown("### —")
        with c3:
            if reco in ("Strong Buy", "Buy"):
                st.markdown(f'<div class="recommendation-buy">Recommendation: {reco}</div>', unsafe_allow_html=True)
            elif reco == "Hold":
                st.markdown(f'<div class="recommendation-hold">Recommendation: {reco}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="recommendation-sell">Recommendation: {reco}</div>', unsafe_allow_html=True)
            mc = data.get("market_cap")
            st.markdown(f"**Market Cap:** {'$' + format(mc, ',') if mc else 'N/A'}")
            st.markdown(f"**P/E Ratio:** {data.get('pe','N/A')}")

    # Charts
    st.markdown('<h2 class="sub-header">Price Chart & Technical Analysis</h2>', unsafe_allow_html=True)
    if not tech.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = create_price_chart(tech, ticker_symbol)
            if fig1: st.pyplot(fig1)
        with col2:
            fig2 = create_technical_indicators_chart(tech, ticker_symbol)
            if fig2: st.pyplot(fig2)
    else:
        st.warning("No historical data available for technical analysis.")

    # Fundamentals
    st.markdown('<h2 class="sub-header">Fundamental Analysis</h2>', unsafe_allow_html=True)
    if data:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-card">**P/E Ratio:** ' + str(data.get("pe","N/A")) + "</div>", unsafe_allow_html=True)
            st.markdown('<div class="metric-card">**EPS:** ' + str(data.get("eps","N/A")) + "</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card">**Profit Margin:** ' + str(data.get("profit_margins","N/A")) + "</div>", unsafe_allow_html=True)
            st.markdown('<div class="metric-card">**ROE:** ' + str(data.get("roe","N/A")) + "</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card">**Dividend Yield:** ' + str(data.get("dividend_yield","N/A")) + "</div>", unsafe_allow_html=True)
            st.markdown('<div class="metric-card">**52 Week High:** ' + str(data.get("week52_high","N/A")) + "</div>", unsafe_allow_html=True)
    else:
        st.info("No fundamental data available for this stock.")

    # News
    st.markdown('<h2 class="sub-header">Recent News</h2>', unsafe_allow_html=True)
    if news:
        for n in news:
            with st.expander(f"{n['title']} — {n['source']} ({n['date']})"):
                st.markdown(f"<div class='news-card'>{n['summary']}</div>", unsafe_allow_html=True)
    else:
        st.info("No recent news available for this stock.")

    # Investment Thesis (simple)
    st.markdown('<h2 class="sub-header">Investment Thesis</h2>', unsafe_allow_html=True)
    if data and not tech.empty:
        cp = data.get("current_price")
        pc = data.get("previous_close") or cp
        change_pct = ((cp - pc) / pc * 100) if (cp and pc) else 0
        rsi = float(tech["RSI"].dropna().iloc[-1]) if "RSI" in tech and not tech["RSI"].dropna().empty else 50

        st.markdown("### Summary Analysis")
        if change_pct > 5:
            st.markdown(f"- The stock has shown <span class='positive'>strong positive momentum</span>, gaining {change_pct:.2f}%.", unsafe_allow_html=True)
        elif change_pct < -5:
            st.markdown(f"- The stock has seen <span class='negative'>notable weakness</span>, down {abs(change_pct):.2f}%.", unsafe_allow_html=True)
        else:
            st.markdown(f"- The stock is relatively stable recently (±5%), change {change_pct:.2f}%.", unsafe_allow_html=True)

        if rsi < 30:
            st.markdown(f"- RSI {rsi:.2f} suggests **oversold** conditions (potential opportunity).", unsafe_allow_html=True)
        elif rsi > 70:
            st.markdown(f"- RSI {rsi:.2f} suggests **overbought** conditions (risk of pullback).", unsafe_allow_html=True)
        else:
            st.markdown(f"- RSI {rsi:.2f} is neutral.", unsafe_allow_html=True)

        st.markdown("### Potential Risks")
        st.markdown("- Market volatility can impact short-term performance\n- Regulatory/competitive pressures\n- Macro conditions affecting demand")

        st.markdown("### Potential Opportunities")
        st.markdown("- Durable fundamentals/brand moat\n- Product/segment innovation\n- New market expansion")

if __name__ == "__main__":
    main()
