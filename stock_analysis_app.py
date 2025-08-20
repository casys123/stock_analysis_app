import streamlit as st
import pandas as pd
import numpy as np
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
    .recommendation-buy { background-color: #E8F5E9; color: #2E7D32; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; text-align: center; font-size: 1.3rem; }
    .recommendation-hold { background-color: #FFF8E1; color: #F57C00; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; text-align: center; font-size: 1.3rem; }
    .recommendation-sell { background-color: #FFEBEE; color: #C62828; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; text-align: center; font-size: 1.3rem; }
    .news-card { background-color: #FFF3E0; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; }
    .opinion-header { font-size: 1.6rem; font-weight: bold; color: #1565C0; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 StockInsight Pro</h1>', unsafe_allow_html=True)
st.markdown("### Yahoo + Finnhub data • RSI/MACD • Consensus & My Opinion (optional GPT)")

# =========================================
# Sidebar
# =========================================
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").strip().upper()
time_frame = st.sidebar.selectbox("Select Time Frame:", ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "5Y"])

# =========================================
# API Keys (secrets with safe fallbacks)
# =========================================
FINNHUB_URL = "https://finnhub.io/api/v1"
try:
    FINNHUB_API_KEY = st.secrets["finnhub"]["api_key"]
except Exception:
    FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY"  # replace for quick local testing

# Optional OpenAI key
def get_openai_key():
    try:
        return st.secrets["openai"]["api_key"]
    except Exception:
        return None

# =========================================
# Robust Yahoo News Parser
# =========================================
def _parse_unix_or_iso(ts):
    try:
        if ts is None:
            return ""
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z","")).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return ts
        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""
    return ""

def _first(d: dict, *keys, default=""):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return default

def parse_yahoo_news(raw_news):
    rows = []
    for n in raw_news[:20]:
        title = _first(n, "title", "headline", default="")
        provider = _first(n, "publisher", "provider", "source", default="")
        link = _first(n, "link", "url", default="")
        ts = _first(n, "providerPublishTime", "pubDate", "published_at", default=None)
        published = _parse_unix_or_iso(ts)
        if not title and not link:
            continue
        rows.append({
            "provider": provider,
            "title": title,
            "link": link,
            "published": published
        })
    return pd.DataFrame(rows)

# =========================================
# Fetch Yahoo Data (with news fix)
# =========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo_data(ticker: str, tf: str):
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period="1y")
    except Exception:
        hist = pd.DataFrame()
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        recs_df = t.recommendations
        if recs_df is not None and not recs_df.empty:
            recs_df = recs_df.tail(30)
        else:
            recs_df = pd.DataFrame()
    except Exception:
        recs_df = pd.DataFrame()
    try:
        raw_news = t.news or []
        news_df = parse_yahoo_news(raw_news)
    except Exception:
        news_df = pd.DataFrame()

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
        "recommendations": recs_df,
        "news": news_df
    }

# (Finnhub fetch + TA + scoring functions omitted here for brevity — keep your existing ones unchanged)

# =========================================
# Consensus & Opinion Display
# =========================================
# After computing consensus_label and consensus_score:
box_class = "recommendation-hold"
if consensus_label.startswith("Strong Buy") or consensus_label == "Strong Buy": box_class = "recommendation-buy"
elif consensus_label == "Buy": box_class = "recommendation-buy"
elif consensus_label == "Sell": box_class = "recommendation-sell"
elif consensus_label.startswith("Strong Sell"): box_class = "recommendation-sell"

st.markdown('<h2 class="sub-header">Consensus View</h2>', unsafe_allow_html=True)
st.markdown(
    f'<div class="{box_class}"><span style="font-size:1.6rem;">Consensus: <b>{consensus_label}</b></span><br>'
    f'(Score {consensus_score:.2f})</div>',
    unsafe_allow_html=True
)

# ChatGPT/My Opinion
st.markdown('<div class="opinion-header">🧠 My Opinion</div>', unsafe_allow_html=True)
opinion = get_chatgpt_opinion(symbol, snapshot)  # same function as before
st.write(opinion)
