import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# =========================================
# Streamlit Setup & Styles
# =========================================
st.set_page_config(
    page_title="StockInsight Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
st.markdown("### Compare Yahoo Finance and Finnhub — quotes, charts, fundamentals, recommendations, and news")

# =========================================
# Sidebar
# =========================================
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").strip().upper()
time_frame = st.sidebar.selectbox("Select Time Frame:", ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "5Y"])

# =========================================
# Finnhub config (with secrets fallback)
# =========================================
FINNHUB_URL = "https://finnhub.io/api/v1"
try:
    FINNHUB_API_KEY = st.secrets["finnhub"]["api_key"]
except Exception:
    FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY"  # quick local test fallback (replace for testing)

# =========================================
# Helpers
# =========================================
def map_timeframe(tf: str):
    """Return (resolution, start_ts, end_ts) for Finnhub /stock/candle."""
    now = int(datetime.now().timestamp())
    if tf == "1D":
        return "5", now - 60*60*24*5, now
    if tf == "1W":
        return "30", now - 60*60*24*30, now
    if tf == "1M":
        return "D", now - 60*60*24*90, now
    if tf == "3M":
        return "D", now - 60*60*24*180, now
    if tf == "6M":
        return "D", now - 60*60*24*365, now
    if tf == "YTD":
        start = int(datetime(datetime.now().year, 1, 1).timestamp())
        return "D", start, now
    if tf == "1Y":
        return "D", now - 60*60*24*365, now
    if tf == "5Y":
        return "W", now - 60*60*24*365*5, now
    return "D", now - 60*60*24*365, now

def yahoo_period_for(tf: str) -> str:
    """Map UI timeframe to yfinance history period."""
    if tf == "1D": return "5d"   # get enough bars for display
    if tf == "1W": return "1mo"
    if tf == "1M": return "3mo"
    if tf == "3M": return "6mo"
    if tf == "6M": return "1y"
    if tf == "YTD": return "ytd"
    if tf == "1Y": return "1y"
    if tf == "5Y": return "5y"
    return "1y"

# =========================================
# Data Fetchers (cache-safe)
# =========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo_data(ticker: str, tf: str):
    """Fetch history + key stats + recommendations + news from Yahoo."""
    period = yahoo_period_for(tf)
    t = yf.Ticker(ticker)

    # History
    try:
        hist = t.history(period=period)
        if not hist.empty and "Close" in hist:
            hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            hist = hist[~hist.index.duplicated(keep="last")]
    except Exception:
        hist = pd.DataFrame()

    # Info (can be partial)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # Recommendations (DataFrame or None) → sanitize to DataFrame
    try:
        recs_df = t.recommendations
        if recs_df is not None and not recs_df.empty:
            # keep recent subset with minimal columns if present
            use_cols = [c for c in recs_df.columns if c.lower() in {"to grade", "from grade", "firm", "action"}]
            recs_df = recs_df[use_cols] if use_cols else recs_df.copy()
            recs_df = recs_df.tail(20)  # last 20 entries
        else:
            recs_df = pd.DataFrame()
    except Exception:
        recs_df = pd.DataFrame()

    # News (list of dicts)
    try:
        news_list = t.news or []
        # normalize to a small table
        news_rows = []
        for n in news_list[:20]:
            news_rows.append({
                "provider": n.get("publisher", n.get("provider", "")),
                "title": n.get("title", ""),
                "link": n.get("link", n.get("url", "")),
                "published": datetime.fromtimestamp(n["providerPublishTime"]).strftime("%Y-%m-%d %H:%M:%S")
                    if n.get("providerPublishTime") else ""
            })
        yahoo_news_df = pd.DataFrame(news_rows)
    except Exception:
        yahoo_news_df = pd.DataFrame()

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
        "recommendations": recs_df,    # DataFrame
        "news": yahoo_news_df          # DataFrame
    }

@st.cache_data(ttl=300, show_spinner=False)
def fetch_finnhub_data(ticker: str, tf: str):
    """Fetch candles + profile + quote + recommendations + news from Finnhub."""
    res, start, end = map_timeframe(tf)

    # Candles
    try:
        candles = requests.get(
            f"{FINNHUB_URL}/stock/candle",
            params={"symbol": ticker, "resolution": res, "from": start, "to": end, "token": FINNHUB_API_KEY},
            timeout=15
        ).json()
        if candles.get("s") == "ok":
            hist = pd.DataFrame({
                "Date": pd.to_datetime(candles["t"], unit="s"),
                "Open": candles["o"],
                "High": candles["h"],
                "Low": candles["l"],
                "Close": candles["c"],
                "Volume": candles["v"],
            }).set_index("Date")
        else:
            hist = pd.DataFrame()
    except Exception:
        hist = pd.DataFrame()

    # Profile
    try:
        profile = requests.get(
            f"{FINNHUB_URL}/stock/profile2",
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
            timeout=15
        ).json() or {}
    except Exception:
        profile = {}

    # Quote (current price, previous close, etc.)
    try:
        quote = requests.get(
            f"{FINNHUB_URL}/quote",
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
            timeout=15
        ).json() or {}
    except Exception:
        quote = {}

    # Recommendation trends
    try:
        recs = requests.get(
            f"{FINNHUB_URL}/stock/recommendation",
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
            timeout=15
        ).json() or []
        finnhub_recs_df = pd.DataFrame(recs)
        # keep last 12 entries max
        if not finnhub_recs_df.empty:
            finnhub_recs_df = finnhub_recs_df.head(12)
    except Exception:
        finnhub_recs_df = pd.DataFrame()

    # Company news (last 30 days)
    try:
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=30)
        news = requests.get(
            f"{FINNHUB_URL}/company-news",
            params={"symbol": ticker, "from": str(from_date), "to": str(to_date), "token": FINNHUB_API_KEY},
            timeout=15
        ).json() or []
        news_rows = []
        for n in news[:30]:
            dt = n.get("datetime")
            news_rows.append({
                "provider": n.get("source", ""),
                "title": n.get("headline", ""),
                "link": n.get("url", ""),
                "published": datetime.utcfromtimestamp(dt).strftime("%Y-%m-%d %H:%M:%S") if dt else ""
            })
        finnhub_news_df = pd.DataFrame(news_rows)
    except Exception:
        finnhub_news_df = pd.DataFrame()

    # Collect
    return {
        "history": hist,
        "current_price": quote.get("c"),
        "previous_close": quote.get("pc"),
        "market_cap": profile.get("marketCapitalization"),
        "pe": profile.get("peBasicExclExtraTTM"),
        "eps": profile.get("epsBasicExclExtraItemsTTM"),
        "dividend_yield": profile.get("dividendYieldIndicatedAnnual"),
        "week52_high": profile.get("52WeekHigh") if "52WeekHigh" in profile else profile.get("52WeekHigh", None),
        "sector": profile.get("finnhubIndustry"),
        "industry": profile.get("ipo"),  # Finnhub doesn't offer industry detail beyond 'finnhubIndustry'
        "recommendations": finnhub_recs_df,  # DataFrame
        "news": finnhub_news_df              # DataFrame
    }

# =========================================
# Chart Helpers (Matplotlib only)
# =========================================
def plot_close(df: pd.DataFrame, ticker: str, source_label: str):
    if df is None or df.empty or "Close" not in df:
        return None
    rcParams["figure.figsize"] = (11, 4)
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label=f"{source_label} Close")
    ax.set_title(f"{ticker} — {source_label} Closing Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    return fig

# =========================================
# Main App
# =========================================
with st.spinner("Fetching data from Yahoo & Finnhub…"):
    y = fetch_yahoo_data(symbol, time_frame)
    f = fetch_finnhub_data(symbol, time_frame)

# KPIs
c1, c2 = st.columns(2)
with c1:
    st.subheader("📊 Yahoo Finance")
    st.write(f"Sector: {y.get('sector') or 'N/A'}")
    st.write(f"Industry: {y.get('industry') or 'N/A'}")
    cp, pc = y.get("current_price"), y.get("previous_close")
    if cp is not None and pc:
        delta = cp - pc
        pct = (delta / pc) * 100 if pc else 0
        st.metric("Current Price", f"${cp:,.2f}", f"{delta:,.2f} ({pct:.2f}%)")
    fig_y = plot_close(y.get("history"), symbol, "Yahoo")
    if fig_y: st.pyplot(fig_y)

with c2:
    st.subheader("📊 Finnhub")
    st.write(f"Sector: {f.get('sector') or 'N/A'}")
    st.write(f"Industry (IPO date shown as placeholder): {f.get('industry') or 'N/A'}")
    cp, pc = f.get("current_price"), f.get("previous_close")
    if cp is not None and pc:
        delta = cp - pc
        pct = (delta / pc) * 100 if pc else 0
        st.metric("Current Price", f"${cp:,.2f}", f"{delta:,.2f} ({pct:.2f}%)")
    fig_f = plot_close(f.get("history"), symbol, "Finnhub")
    if fig_f: st.pyplot(fig_f)

# Comparison Table
st.markdown('<h2 class="sub-header">Key Metrics — Yahoo vs Finnhub</h2>', unsafe_allow_html=True)
compare_df = pd.DataFrame({
    "Yahoo": {
        "Current Price": y.get("current_price"),
        "Market Cap": y.get("market_cap"),
        "P/E (trailing)": y.get("pe"),
        "EPS (trailing)": y.get("eps"),
        "Dividend Yield": y.get("dividend_yield"),
        "52W High": y.get("week52_high"),
    },
    "Finnhub": {
        "Current Price": f.get("current_price"),
        "Market Cap": f.get("market_cap"),
        "P/E (basic excl. extra TTM)": f.get("pe"),
        "EPS (basic excl. extra TTM)": f.get("eps"),
        "Dividend Yield (indicated annual)": f.get("dividend_yield"),
        "52W High": f.get("week52_high"),
    }
})
st.dataframe(compare_df)

# Analyst Recommendations
st.markdown('<h2 class="sub-header">Analyst Recommendations</h2>', unsafe_allow_html=True)
rc1, rc2 = st.columns(2)
with rc1:
    st.markdown("**Yahoo Finance (recent)**")
    yr = y.get("recommendations")
    if isinstance(yr, pd.DataFrame) and not yr.empty:
        st.dataframe(yr)
    else:
        st.info("No recent recommendations from Yahoo.")

with rc2:
    st.markdown("**Finnhub (latest)**")
    fr = f.get("recommendations")
    if isinstance(fr, pd.DataFrame) and not fr.empty:
        # If Finnhub returns columns like: buy, hold, sell, strongBuy, strongSell, period
        show_cols = [c for c in ["period", "strongBuy", "buy", "hold", "sell", "strongSell"] if c in fr.columns]
        st.dataframe(fr[show_cols] if show_cols else fr)
    else:
        st.info("No recent recommendation trends from Finnhub.")

# News Sections
st.markdown('<h2 class="sub-header">Recent News</h2>', unsafe_allow_html=True)
nc1, nc2 = st.columns(2)
with nc1:
    st.markdown("**Yahoo Finance News**")
    yn = y.get("news")
    if isinstance(yn, pd.DataFrame) and not yn.empty:
        for _, row in yn.head(10).iterrows():
            with st.expander(f"{row.get('title','(no title)')} — {row.get('provider','')} ({row.get('published','')})"):
                link = row.get("link", "")
                if link:
                    st.write(f"[Open article]({link})")
    else:
        st.info("No Yahoo news available.")

with nc2:
    st.markdown("**Finnhub Company News (30d)**")
    fn = f.get("news")
    if isinstance(fn, pd.DataFrame) and not fn.empty:
        for _, row in fn.head(10).iterrows():
            with st.expander(f"{row.get('title','(no title)')} — {row.get('provider','')} ({row.get('published','')})"):
                link = row.get("link", "")
                if link:
                    st.write(f"[Open article]({link})")
    else:
        st.info("No Finnhub news available (or missing API key).")

# Simple Rules-Based Recommendation (from price change only, as a demo)
st.markdown('<h2 class="sub-header">Quick Take</h2>', unsafe_allow_html=True)
y_cp, y_pc = y.get("current_price"), y.get("previous_close")
if y_cp and y_pc:
    change_pct = (y_cp - y_pc) / y_pc * 100
    score = 0
    if change_pct > 2: score += 1
    elif change_pct < -2: score -= 1
    label = "Hold"
    if score >= 1: label = "Buy"
    if score <= -1: label = "Sell"
    box_class = "recommendation-hold"
    if label == "Buy": box_class = "recommendation-buy"
    if label == "Sell": box_class = "recommendation-sell"
    st.markdown(f'<div class="{box_class}">Yahoo momentum signal: <b>{label}</b> (Δ {change_pct:.2f}%)</div>', unsafe_allow_html=True)
else:
    st.info("Not enough data to compute a quick momentum signal.")
