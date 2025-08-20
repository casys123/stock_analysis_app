import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(
    page_title="StockInsight Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('<h1 style="text-align:center;color:#1E88E5;">📊 StockInsight Pro</h1>', unsafe_allow_html=True)
st.markdown("### Compare Yahoo Finance vs Finnhub Data")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").upper()
time_frame = st.sidebar.selectbox(
    "Select Time Frame:",
    ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "5Y"]
)

# ----------------------------
# Finnhub API Setup
# ----------------------------
FINNHUB_URL = "https://finnhub.io/api/v1"
API_KEY = st.secrets["finnhub"]["api_key"]


def map_timeframe(tf: str):
    now = int(datetime.now().timestamp())
    if tf == "1D": return "5", now - 60*60*24*5
    if tf == "1W": return "30", now - 60*60*24*30
    if tf == "1M": return "D", now - 60*60*24*90
    if tf == "3M": return "D", now - 60*60*24*180
    if tf == "6M": return "D", now - 60*60*24*365
    if tf == "YTD": return "D", int(datetime(datetime.now().year, 1, 1).timestamp())
    if tf == "1Y": return "D", now - 60*60*24*365
    if tf == "5Y": return "W", now - 60*60*24*365*5
    return "D", now - 60*60*24*365


# ----------------------------
# Fetch Yahoo Finance Data
# ----------------------------
def fetch_yahoo_data(ticker: str, period="1y"):
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    info = t.info

    return {
        "history": hist,
        "current_price": info.get("currentPrice"),
        "previous_close": info.get("previousClose"),
        "market_cap": info.get("marketCap"),
        "pe": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "dividend_yield": info.get("dividendYield"),
        "week52_high": info.get("fiftyTwoWeekHigh"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


# ----------------------------
# Fetch Finnhub Data
# ----------------------------
def fetch_finnhub_data(ticker: str, tf: str):
    res, start = map_timeframe(tf)
    now = int(datetime.now().timestamp())

    candles = requests.get(
        f"{FINNHUB_URL}/stock/candle",
        params={"symbol": ticker, "resolution": res, "from": start, "to": now, "token": API_KEY}
    ).json()

    if candles.get("s") == "ok":
        df = pd.DataFrame({
            "Date": pd.to_datetime(candles["t"], unit="s"),
            "Open": candles["o"],
            "High": candles["h"],
            "Low": candles["l"],
            "Close": candles["c"],
            "Volume": candles["v"]
        }).set_index("Date")
    else:
        df = pd.DataFrame()

    profile = requests.get(f"{FINNHUB_URL}/stock/profile2", params={"symbol": ticker, "token": API_KEY}).json()
    quote = requests.get(f"{FINNHUB_URL}/quote", params={"symbol": ticker, "token": API_KEY}).json()

    return {
        "history": df,
        "current_price": quote.get("c"),
        "previous_close": quote.get("pc"),
        "market_cap": profile.get("marketCapitalization"),
        "pe": profile.get("peBasicExclExtraTTM"),
        "eps": profile.get("epsBasicExclExtraItemsTTM"),
        "dividend_yield": profile.get("dividendYieldIndicatedAnnual"),
        "week52_high": profile.get("52WeekHigh"),
        "sector": profile.get("finnhubIndustry"),
        "industry": profile.get("ipo"),  # Finnhub has no industry field, using IPO date as placeholder
    }


# ----------------------------
# Charts
# ----------------------------
def create_price_chart(df, ticker, source):
    if df.empty: return None
    rcParams["figure.figsize"] = (10, 4)
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label=f"{source} Close")
    ax.set_title(f"{ticker} Closing Prices ({source})")
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    return fig


# ----------------------------
# Main App
# ----------------------------
st.spinner("Fetching data...")

yahoo_data = fetch_yahoo_data(symbol, "1y")
finnhub_data = fetch_finnhub_data(symbol, time_frame)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Yahoo Finance")
    st.write(f"Sector: {yahoo_data['sector']}")
    st.write(f"Industry: {yahoo_data['industry']}")
    st.metric("Current Price", yahoo_data["current_price"], yahoo_data["current_price"]-yahoo_data["previous_close"] if yahoo_data["previous_close"] else 0)

    fig1 = create_price_chart(yahoo_data["history"], symbol, "Yahoo")
    if fig1: st.pyplot(fig1)

with col2:
    st.subheader("📊 Finnhub")
    st.write(f"Sector: {finnhub_data['sector']}")
    st.write(f"Industry (IPO date placeholder): {finnhub_data['industry']}")
    st.metric("Current Price", finnhub_data["current_price"], finnhub_data["current_price"]-finnhub_data["previous_close"] if finnhub_data["previous_close"] else 0)

    fig2 = create_price_chart(finnhub_data["history"], symbol, "Finnhub")
    if fig2: st.pyplot(fig2)

# Comparison Table
st.markdown("## 🔍 Comparison Table")
compare_df = pd.DataFrame({
    "Yahoo": {
        "Current Price": yahoo_data["current_price"],
        "Market Cap": yahoo_data["market_cap"],
        "P/E": yahoo_data["pe"],
        "EPS": yahoo_data["eps"],
        "Dividend Yield": yahoo_data["dividend_yield"],
        "52W High": yahoo_data["week52_high"],
    },
    "Finnhub": {
        "Current Price": finnhub_data["current_price"],
        "Market Cap": finnhub_data["market_cap"],
        "P/E": finnhub_data["pe"],
        "EPS": finnhub_data["eps"],
        "Dividend Yield": finnhub_data["dividend_yield"],
        "52W High": finnhub_data["week52_high"],
    }
})
st.dataframe(compare_df)
