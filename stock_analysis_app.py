from typing import Optional, Dict

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
    .recommendation-buy { background-color: #E8F5E9; color: #2E7D32; padding: 0.7rem 1rem; border-radius: 20px; font-weight: 700; text-align: center; font-size: 1.35rem; }
    .recommendation-hold { background-color: #FFF8E1; color: #F57C00; padding: 0.7rem 1rem; border-radius: 20px; font-weight: 700; text-align: center; font-size: 1.35rem; }
    .recommendation-sell { background-color: #FFEBEE; color: #C62828; padding: 0.7rem 1rem; border-radius: 20px; font-weight: 700; text-align: center; font-size: 1.35rem; }
    .news-card { background-color: #FFF3E0; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; }
    .opinion-header { font-size: 1.7rem; font-weight: 800; color: #1565C0; margin-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 StockInsight Pro</h1>', unsafe_allow_html=True)
st.markdown("### Yahoo + Finnhub • RSI/MACD • Fibonacci • Consensus & My Opinion")

# =========================================
# Sidebar
# =========================================
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").strip().upper()
time_frame = st.sidebar.selectbox("Select Time Frame:", ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "5Y"])

# --- Technical settings ---
st.sidebar.subheader("Technical Settings")
fib_window = st.sidebar.selectbox(
    "Fibonacci Lookback",
    ["3M", "6M", "1Y", "2Y", "Max"],
    index=2,  # default 1Y
    help="Choose the window used to detect swing high/low for Fibonacci retracement."
)

# =========================================
# API Keys (secrets + in-app override)
# =========================================
FINNHUB_URL = "https://finnhub.io/api/v1"
try:
    FINNHUB_API_KEY_DEFAULT = st.secrets["finnhub"]["api_key"]
except Exception:
    FINNHUB_API_KEY_DEFAULT = "YOUR_FINNHUB_API_KEY"

def get_openai_key():
    try:
        return st.secrets["openai"]["api_key"]
    except Exception:
        return None

# session-persisted Finnhub key with in-app override
if "finnhub_api_key" not in st.session_state:
    st.session_state.finnhub_api_key = FINNHUB_API_KEY_DEFAULT

with st.sidebar.expander("API Keys", expanded=False):
    st.caption("Paste a valid Finnhub key here to enable Finnhub data.")
    key_input = st.text_input(
        "Finnhub API Key",
        value="" if st.session_state.finnhub_api_key in (None, "", "YOUR_FINNHUB_API_KEY") else st.session_state.finnhub_api_key,
        type="password"
    )
    if key_input:
        st.session_state.finnhub_api_key = key_input

FINNHUB_API_KEY = st.session_state.finnhub_api_key

@st.cache_data(ttl=120, show_spinner=False)
def validate_finnhub(api_key: Optional[str]) -> bool:
    if not api_key or api_key == "YOUR_FINNHUB_API_KEY":
        return False
    try:
        r = requests.get(f"{FINNHUB_URL}/quote", params={"symbol":"AAPL", "token": api_key}, timeout=10)
        if r.status_code != 200:
            return False
        j = r.json()
        return isinstance(j, dict) and "c" in j
    except Exception:
        return False

FINNHUB_OK = validate_finnhub(FINNHUB_API_KEY)
if not FINNHUB_OK:
    st.warning("⚠️ Finnhub API key invalid or missing. Enter a valid key in the sidebar to enable Finnhub news & metrics.")

# =========================================
# Helper: timeframe mapping
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
    return {
        "1D":"5d","1W":"1mo","1M":"3mo","3M":"6mo","6M":"1y","YTD":"ytd","1Y":"1y","5Y":"5y"
    }.get(tf, "1y")

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
        rows.append({"provider": provider, "title": title, "link": link, "published": published})
    return pd.DataFrame(rows)

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
            hist = hist[["Open","High","Low","Close","Volume"]].copy()
            hist = hist[~hist.index.duplicated(keep="last")]
    except Exception:
        hist = pd.DataFrame()

    # Info
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # Recommendations
    try:
        recs_df = t.recommendations
        recs_df = recs_df.tail(30).copy() if recs_df is not None and not recs_df.empty else pd.DataFrame()
    except Exception:
        recs_df = pd.DataFrame()

    # News (robust + visibility)
    try:
        raw_news = t.news or []
        if len(raw_news) == 0:
            st.info("Yahoo returned 0 news items for this ticker (common on hosted environments).")
        news_df = parse_yahoo_news(raw_news)
    except Exception as e:
        st.warning(f"Yahoo news parsing failed: {e}")
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_finnhub_data(ticker: str, tf: str, api_key: Optional[str]):
    """Fetch candles + profile + quote + recommendations + news + metrics from Finnhub."""
    if not api_key:
        return {
            "history": pd.DataFrame(), "current_price": None, "previous_close": None,
            "market_cap": None, "pe": None, "eps": None, "dividend_yield": None,
            "week52_high": None, "sector": None, "industry": None,
            "recommendations": pd.DataFrame(), "news": pd.DataFrame()
        }

    res, start, end = map_timeframe(tf)

    # Candles
    try:
        candles = requests.get(
            f"{FINNHUB_URL}/stock/candle",
            params={"symbol": ticker, "resolution": res, "from": start, "to": end, "token": api_key},
            timeout=15
        ).json()
        if candles.get("s") == "ok":
            hist = pd.DataFrame({
                "Date": pd.to_datetime(candles["t"], unit="s"),
                "Open": candles["o"], "High": candles["h"], "Low": candles["l"],
                "Close": candles["c"], "Volume": candles["v"],
            }).set_index("Date")
        else:
            hist = pd.DataFrame()
    except Exception:
        hist = pd.DataFrame()

    # Profile
    try:
        profile = requests.get(f"{FINNHUB_URL}/stock/profile2",
                               params={"symbol": ticker, "token": api_key}, timeout=15).json() or {}
    except Exception:
        profile = {}

    # Quote
    try:
        quote = requests.get(f"{FINNHUB_URL}/quote",
                             params={"symbol": ticker, "token": api_key}, timeout=15).json() or {}
    except Exception:
        quote = {}

    # Recommendation trends
    try:
        recs = requests.get(f"{FINNHUB_URL}/stock/recommendation",
                            params={"symbol": ticker, "token": api_key}, timeout=15).json() or []
        finnhub_recs_df = pd.DataFrame(recs)
        if not finnhub_recs_df.empty:
            finnhub_recs_df = finnhub_recs_df.head(12).copy()
    except Exception:
        finnhub_recs_df = pd.DataFrame()

    # Company news with error surfacing + 30→90 day fallback
    def _fetch_company_news(symbol: str, days_back: int) -> pd.DataFrame:
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=days_back)
        resp = requests.get(f"{FINNHUB_URL}/company-news",
                            params={"symbol": symbol, "from": str(from_date), "to": str(to_date), "token": api_key},
                            timeout=15)
        status = resp.status_code
        text = resp.text
        try:
            data = resp.json()
        except Exception:
            data = None

        if status != 200:
            st.error(f"Finnhub news HTTP {status}. Response head: {text[:200]}")
            return pd.DataFrame()

        if isinstance(data, dict) and data.get("error"):
            st.error(f"Finnhub news error: {data.get('error')}")
            return pd.DataFrame()

        if not isinstance(data, list):
            st.warning("Finnhub news returned unexpected payload.")
            return pd.DataFrame()

        rows = []
        for n in data[:60]:
            dt = n.get("datetime")
            rows.append({
                "provider": n.get("source", ""),
                "title": n.get("headline", ""),
                "link": n.get("url", ""),
                "published": datetime.utcfromtimestamp(dt).strftime("%Y-%m-%d %H:%M:%S") if dt else ""
            })
        return pd.DataFrame(rows)

    finnhub_news_df = _fetch_company_news(ticker, 30)
    if finnhub_news_df.empty:
        finnhub_news_df = _fetch_company_news(ticker, 90)

    # 52w high via metrics
    try:
        metrics = requests.get(f"{FINNHUB_URL}/stock/metric",
                               params={"symbol": ticker, "metric": "price", "token": api_key}, timeout=15).json() or {}
        m = metrics.get("metric", {}) if isinstance(metrics, dict) else {}
        week52_high = m.get("52WeekHigh")
    except Exception:
        week52_high = None

    return {
        "history": hist,
        "current_price": quote.get("c"),
        "previous_close": quote.get("pc"),
        "market_cap": profile.get("marketCapitalization"),
        "pe": profile.get("peBasicExclExtraTTM"),
        "eps": profile.get("epsBasicExclExtraItemsTTM"),
        "dividend_yield": profile.get("dividendYieldIndicatedAnnual"),
        "week52_high": week52_high,
        "sector": profile.get("finnhubIndustry"),
        "industry": profile.get("ipo"),  # placeholder
        "recommendations": finnhub_recs_df,
        "news": finnhub_news_df
    }

# =========================================
# Technical Indicators
# =========================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Close" not in df:
        return df
    out = df.copy()
    out["MA20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["MA50"] = out["Close"].rolling(50, min_periods=50).mean()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.ewm(alpha=1/14, adjust=False).mean()
    roll_down = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    out["RSI"] = out["RSI"].clip(0, 100)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    return out

# =========================================
# Fibonacci Retracement (fixed window control)
# =========================================
def _fib_rows_from_choice(choice: str, df_len: int) -> int:
    """Map UI choice to approximate trading-day rows."""
    mapping = {
        "3M": 63,
        "6M": 126,
        "1Y": 252,
        "2Y": 504,
        "Max": df_len or 0,
    }
    rows = mapping.get(choice, 252)
    return min(rows, df_len or 0)

def compute_fib(df: pd.DataFrame, rows: int) -> Optional[Dict]:
    """Compute Fib using the last `rows` bars (fixed window)."""
    if df is None or df.empty:
        return None
    if not set(["High", "Low", "Close"]).issubset(df.columns):
        return None
    rows = max(2, min(rows, len(df)))  # need at least 2 points
    window = df.tail(rows)

    swing_high = float(window["High"].max())
    swing_low = float(window["Low"].min())
    if swing_high <= swing_low:
        return None

    last_close = float(window["Close"].iloc[-1])
    mid = (swing_high + swing_low) / 2.0
    uptrend = last_close >= mid

    diff = swing_high - swing_low
    if uptrend:
        levels = {
            "23.6%": swing_high - 0.236*diff,
            "38.2%": swing_high - 0.382*diff,
            "50.0%": swing_high - 0.500*diff,
            "61.8%": swing_high - 0.618*diff,
            "78.6%": swing_high - 0.786*diff,
        }
        basis = "low→high (support levels)"
    else:
        levels = {
            "23.6%": swing_low + 0.236*diff,
            "38.2%": swing_low + 0.382*diff,
            "50.0%": swing_low + 0.500*diff,
            "61.8%": swing_low + 0.618*diff,
            "78.6%": swing_low + 0.786*diff,
        }
        basis = "high→low (resistance levels)"

    nearest_name, nearest_val, nearest_dist = None, None, float("inf")
    for name, val in levels.items():
        d = abs(last_close - val)
        if d < nearest_dist:
            nearest_name, nearest_val, nearest_dist = name, float(val), float(d)
    nearest_pct = (nearest_dist / last_close * 100.0) if last_close else None

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "levels": levels,
        "uptrend": uptrend,
        "last_close": last_close,
        "nearest_level_name": nearest_name,
        "nearest_level_value": nearest_val,
        "nearest_level_distance_pct": nearest_pct,
        "basis": basis,
        "rows": rows,
    }

def plot_price_with_fib(df: pd.DataFrame, fib: Dict, ticker: str):
    try:
        rcParams["figure.figsize"] = (11, 5)
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close")
        # Draw fib levels
        for name, val in fib["levels"].items():
            ax.axhline(val, linestyle="--", linewidth=1, alpha=0.5)
            ax.text(df.index[0], val, f"  {name}: {val:,.2f}", va="bottom", fontsize=8)
        ax.set_title(f"{ticker} — Fibonacci Retracement ({fib['basis']})")
        ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        fig.autofmt_xdate()
        return fig
    except Exception:
        return None

# =========================================
# Charts (single-plot, default colors)
# =========================================
def plot_close(df: pd.DataFrame, ticker: str, source_label: str):
    if df is None or df.empty or "Close" not in df:
        return None
    rcParams["figure.figsize"] = (11, 4)
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label=f"{source_label} Close")
    if "MA20" in df.columns:
        ax.plot(df.index, df["MA20"], linestyle="--", label="MA20")
    if "MA50" in df.columns:
        ax.plot(df.index, df["MA50"], linestyle="--", label="MA50")
    ax.set_title(f"{ticker} — {source_label} Price")
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    return fig

def plot_rsi(df: pd.DataFrame, ticker: str, source_label: str):
    if df is None or df.empty or "RSI" not in df:
        return None
    rcParams["figure.figsize"] = (11, 3.8)
    fig, ax = plt.subplots()
    ax.plot(df.index, df["RSI"], label="RSI")
    ax.axhline(70, linestyle="--"); ax.axhline(30, linestyle="--")
    ax.set_ylim(0, 100)
    ax.set_title(f"{ticker} — RSI (14) [{source_label}]")
    ax.set_xlabel("Date"); ax.set_ylabel("RSI")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    return fig

def plot_macd(df: pd.DataFrame, ticker: str, source_label: str):
    if df is None or df.empty or "MACD" not in df or "MACD_Signal" not in df:
        return None
    rcParams["figure.figsize"] = (11, 3.8)
    fig, ax = plt.subplots()
    ax.plot(df.index, df["MACD"], label="MACD")
    ax.plot(df.index, df["MACD_Signal"], label="Signal")
    if "MACD_Hist" in df.columns:
        ax.bar(df.index, df["MACD_Hist"], alpha=0.3, label="Histogram")
    ax.set_title(f"{ticker} — MACD [{source_label}]")
    ax.set_xlabel("Date"); ax.set_ylabel("Value")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    return fig

# =========================================
# Scoring / Consensus
# =========================================
def score_yahoo_recs(df: pd.DataFrame) -> float:
    """Score Yahoo recommendations using To Grade/Action if present (≈ −1..+1)."""
    if df is None or df.empty:
        return 0.0
    score, n = 0.0, 0

    def map_grade(g: str):
        if not isinstance(g, str):
            return 0
        g = g.lower()
        buys = ["buy","strong buy","overweight","outperform"]
        sells = ["sell","strong sell","underperform"]
        holds = ["hold","market perform","neutral"]
        if any(k in g for k in buys): return 1
        if any(k in g for k in sells): return -1
        if any(k in g for k in holds): return 0
        return 0

    lower_cols = [c.lower() for c in df.columns]
    for _, row in df.tail(15).iterrows():
        if "to grade" in lower_cols:
            score += map_grade(row[df.columns[lower_cols.index("to grade")]])
            n += 1
        elif "action" in lower_cols:
            act = str(row[df.columns[lower_cols.index("action")]]).lower()
            if "up" in act or "initiate" in act or "reiterate buy" in act:
                score += 1; n += 1
            elif "down" in act or "reduce" in act:
                score -= 1; n += 1
    return score / n if n else 0.0

def score_finnhub_recs(df: pd.DataFrame) -> float:
    """Use latest row: weighted by strongBuy/buy/hold/sell/strongSell (≈ −2..+2)."""
    if df is None or df.empty:
        return 0.0
    row = df.iloc[0]
    sb = int(row.get("strongBuy", 0) or 0)
    b  = int(row.get("buy", 0) or 0)
    h  = int(row.get("hold", 0) or 0)
    s  = int(row.get("sell", 0) or 0)
    ss = int(row.get("strongSell", 0) or 0)
    total = sb + b + h + s + ss
    if total == 0:
        return 0.0
    raw = (2*sb + 1*b + 0*h - 1*s - 2*ss)
    return raw / max(total, 1)

def score_momentum(change_pct: Optional[float], rsi: Optional[float], macd: Optional[float], macd_sig: Optional[float]) -> float:
    """Combine price change, RSI, MACD into a momentum score (≈ −2..+2)."""
    score = 0.0
    if change_pct is not None:
        if change_pct > 2: score += 0.7
        elif change_pct < -2: score -= 0.7
    if rsi is not None:
        if rsi < 30: score += 0.6
        elif rsi > 70: score -= 0.6
    if macd is not None and macd_sig is not None:
        score += 0.5 if (macd - macd_sig) > 0 else -0.5
    return score

def score_fib(fib: Optional[Dict]) -> float:
    """Lightweight Fib signal:
       +0.6 if price is above 61.8% (uptrend) or below 38.2% (downtrend),
       -0.6 if below 61.8% (uptrend) or above 38.2% (downtrend)."""
    if not fib: return 0.0
    lv = fib["levels"]; price = fib["last_close"]; up = fib["uptrend"]
    l382, l618 = lv.get("38.2%"), lv.get("61.8%")
    if any(v is None for v in [l382, l618, price]): return 0.0
    if up:
        if price >= l618: return 0.6
        if price <= l382: return -0.6
    else:
        if price <= l382: return 0.6
        if price >= l618: return -0.6
    return 0.0

def label_from_score(x: float) -> str:
    if x >= 1.5: return "Strong Buy"
    if x >= 0.5: return "Buy"
    if x > -0.5: return "Hold"
    if x > -1.5: return "Sell"
    return "Strong Sell"

# =========================================
# ChatGPT Opinion (optional; falls back to rules) — includes Fib context
# =========================================
def get_chatgpt_opinion(symbol: str, snapshot: dict) -> str:
    y_cp = snapshot.get("y_cp"); y_pc = snapshot.get("y_pc")
    change_pct = ((y_cp - y_pc)/y_pc*100) if (y_cp and y_pc) else None
    rsi = snapshot.get("rsi")
    macd = snapshot.get("macd"); macd_sig = snapshot.get("macd_sig")
    yahoo_s = snapshot.get("yahoo_rec_score", 0.0)
    finn_s  = snapshot.get("finnhub_rec_score", 0.0)
    consensus = snapshot.get("consensus_label", "Hold")
    fib = snapshot.get("fib") or {}
    fib_basis = fib.get("basis", "N/A")
    fib_near = fib.get("nearest_level_name", "N/A")
    fib_near_val = fib.get("nearest_level_value", None)
    fib_near_pct = fib.get("nearest_level_distance_pct", None)
    fib_trend = "uptrend" if fib.get("uptrend") else "downtrend" if fib else "N/A"

    api_key = get_openai_key()
    if not api_key:
        macd_rel = ">" if (macd is not None and macd_sig is not None and macd > macd_sig) else "<="
        near_str = f"{fib_near} @ {fib_near_val:,.2f} ({fib_near_pct:.2f}% away)" if (fib_near_val and fib_near_pct is not None) else "N/A"
        return (f"My take (rules-based): **{consensus}**. "
                f"Momentum Δ={change_pct:.2f}% | RSI={rsi:.1f} | MACD {macd_rel} Signal | "
                f"Fib: {fib_trend}, nearest {near_str} [{fib_basis}] | "
                f"Yahoo recs={yahoo_s:.2f} | Finnhub recs={finn_s:.2f}") if change_pct is not None and rsi is not None else \
               (f"My take (rules-based): **{consensus}**. Fib: {fib_trend}, nearest {near_str} [{fib_basis}] | "
                f"Yahoo recs={yahoo_s:.2f} | Finnhub recs={finn_s:.2f}")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        macd_state = ("bullish" if (macd is not None and macd_sig is not None and macd > macd_sig)
                      else "bearish" if (macd is not None and macd_sig is not None and macd < macd_sig)
                      else "N/A")
        chg_str  = f"{change_pct:.2f}%" if change_pct is not None else "N/A"
        rsi_str  = f"{rsi:.1f}" if rsi is not None else "N/A"
        near_str = f"{fib_near} @ {fib_near_val:,.2f} ({fib_near_pct:.2f}% away)" if (fib_near_val and fib_near_pct is not None) else "N/A"

        prompt = f"""
You are an equity analyst. Give a concise Buy/Sell/Hold view for {symbol} in 3 short bullets.
Inputs:
- Price change vs prev close: {chg_str}
- RSI(14): {rsi_str}
- MACD vs Signal: {macd_state}
- Yahoo analyst score (−1..+1): {yahoo_s:.2f}
- Finnhub analyst score (−2..+2 normalized): {finn_s:.2f}
- Preliminary consensus: {consensus}
- Fibonacci: trend={fib_trend}, nearest_level={near_str}, basis={fib_basis}

Output:
1) Overall rating: **Buy**, **Hold**, or **Sell**
2) One-line rationale using inputs (mention Fib if relevant)
3) One key risk
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=180,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        macd_rel = ">" if (macd is not None and macd_sig is not None and macd > macd_sig) else "<="
        near_str = f"{fib_near} @ {fib_near_val:,.2f} ({fib_near_pct:.2f}% away)" if (fib_near_val and fib_near_pct is not None) else "N/A"
        return (f"My take (rules-based): **{consensus}**. "
                f"Momentum Δ={change_pct:.2f}% | RSI={rsi:.1f} | MACD {macd_rel} Signal | "
                f"Fib: {fib_trend}, nearest {near_str} [{fib_basis}] | "
                f"Yahoo recs={yahoo_s:.2f} | Finnhub recs={finn_s:.2f}") if change_pct is not None and rsi is not None else \
               (f"My take (rules-based): **{consensus}**. Fib: {fib_trend}, nearest {near_str} [{fib_basis}] | "
                f"Yahoo recs={yahoo_s:.2f} | Finnhub recs={finn_s:.2f}")

# =========================================
# MAIN APP FLOW
# =========================================
with st.spinner("Fetching data from Yahoo & Finnhub…"):
    y = fetch_yahoo_data(symbol, time_frame)
    f = fetch_finnhub_data(symbol, time_frame, FINNHUB_API_KEY if FINNHUB_OK else None)

# Pick a base history for technicals (prefer Yahoo)
base_hist = y.get("history")
if base_hist is None or base_hist.empty:
    base_hist = f.get("history")
ta = add_indicators(base_hist)

# KPI & Charts
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
    fig_y = plot_close(add_indicators(y.get("history")), symbol, "Yahoo")
    if fig_y: st.pyplot(fig_y)

with c2:
    st.subheader("📊 Finnhub")
    st.write(f"Sector: {f.get('sector') or 'N/A'}")
    st.write(f"Industry (IPO as placeholder): {f.get('industry') or 'N/A'}")
    cp2, pc2 = f.get("current_price"), f.get("previous_close")
    if cp2 is not None and pc2:
        delta2 = cp2 - pc2
        pct2 = (delta2 / pc2) * 100 if pc2 else 0
        st.metric("Current Price", f"${cp2:,.2f}", f"{delta2:,.2f} ({pct2:.2f}%)")
    fig_f = plot_close(add_indicators(f.get("history")), symbol, "Finnhub")
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

# Technicals
st.markdown('<h2 class="sub-header">Technical Indicators</h2>', unsafe_allow_html=True)
if ta is not None and not ta.empty:
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        rsi_fig = plot_rsi(ta, symbol, "Base")
        if rsi_fig: st.pyplot(rsi_fig)
    with tcol2:
        macd_fig = plot_macd(ta, symbol, "Base")
        if macd_fig: st.pyplot(macd_fig)
else:
    st.info("No historical data available for RSI/MACD.")

# Fibonacci
st.markdown('<h2 class="sub-header">Fibonacci Retracement</h2>', unsafe_allow_html=True)
fib_rows = _fib_rows_from_choice(fib_window, len(base_hist) if base_hist is not None else 0)
fib = compute_fib(base_hist, fib_rows)
if fib:
    fib_fig = plot_price_with_fib(base_hist.tail(fib_rows), fib, symbol)
    if fib_fig: st.pyplot(fib_fig)
    nearest_desc = (f"{fib['nearest_level_name']} @ {fib['nearest_level_value']:,.2f} "
                    f"({fib['nearest_level_distance_pct']:.2f}% away)" if fib.get("nearest_level_value") else "N/A")
    st.caption(
        f"Window: {fib_window} ({fib['rows']} bars). {fib['basis']}. "
        f"Swing High: {fib['swing_high']:,.2f} • Swing Low: {fib['swing_low']:,.2f} • Nearest: {nearest_desc}"
    )
else:
    st.info("Not enough data to compute Fibonacci levels for the selected window.")

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
    st.markdown("**Finnhub (latest trends)**")
    fr = f.get("recommendations")
    if isinstance(fr, pd.DataFrame) and not fr.empty:
        show_cols = [c for c in ["period","strongBuy","buy","hold","sell","strongSell"] if c in fr.columns]
        st.dataframe(fr[show_cols] if show_cols else fr)
    else:
        st.info("No recent recommendation trends from Finnhub.")

# News
st.markdown('<h2 class="sub-header">Recent News</h2>', unsafe_allow_html=True)
nc1, nc2 = st.columns(2)
with nc1:
    st.markdown("**Yahoo Finance News**")
    yn = y.get("news")
    if isinstance(yn, pd.DataFrame) and not yn.empty:
        for _, row in yn.head(10).iterrows():
            title = row.get("title", "(no title)") or "(no title)"
            provider = row.get("provider", "")
            published = row.get("published", "")
            with st.expander(f"{title} — {provider} ({published})"):
                link = row.get("link", "")
                if link: st.write(f"[Open article]({link})")
    else:
        st.info("No Yahoo news available. This is often due to Yahoo blocking hosted traffic (HTTP 999) or empty feeds.")

with nc2:
    st.markdown("**Finnhub Company News (last 30–90 days)**")
    fn = f.get("news")
    if isinstance(fn, pd.DataFrame) and not fn.empty:
        for _, row in fn.head(10).iterrows():
            title = row.get("title", "(no title)") or "(no title)"
            provider = row.get("provider", "")
            published = row.get("published", "")
            with st.expander(f"{title} — {provider} ({published})"):
                link = row.get("link", "")
                if link: st.write(f"[Open article]({link})")
    else:
        if FINNHUB_OK:
            st.info("No Finnhub news found in the last 30–90 days for this ticker.")
        else:
            st.warning("Finnhub news disabled. Enter a valid Finnhub API key in the sidebar.")

# =========================
# Consensus & My Opinion (includes Fib in scoring & colored label)
# =========================
# Momentum inputs (prefer Yahoo)
y_cp, y_pc = y.get("current_price"), y.get("previous_close")
if not y_cp or not y_pc:
    y_cp, y_pc = f.get("current_price"), f.get("previous_close")
change_pct = ((y_cp - y_pc)/y_pc*100) if (y_cp and y_pc) else None
rsi_last = float(ta["RSI"].dropna().iloc[-1]) if (ta is not None and "RSI" in ta and not ta["RSI"].dropna().empty) else None
macd_last = float(ta["MACD"].dropna().iloc[-1]) if (ta is not None and "MACD" in ta and not ta["MACD"].dropna().empty) else None
macd_sig_last = float(ta["MACD_Signal"].dropna().iloc[-1]) if (ta is not None and "MACD_Signal" in ta and not ta["MACD_Signal"].dropna().empty) else None

yahoo_rec_score = score_yahoo_recs(y.get("recommendations"))
finnhub_rec_score = score_finnhub_recs(f.get("recommendations"))
momentum_score = score_momentum(change_pct, rsi_last, macd_last, macd_sig_last)
fib_score = score_fib(fib)

# Normalize & combine (includes Fib)
yahoo_norm = max(min(yahoo_rec_score, 1), -1)            # −1..+1
finnhub_norm = (finnhub_rec_score / 2)                   # scale to −1..+1 approx
consensus_score = 0.28*yahoo_norm + 0.28*finnhub_norm + 0.32*momentum_score + 0.12*fib_score
consensus_label = label_from_score(consensus_score)

# Render Consensus View (bigger font)
box_class = "recommendation-hold"
if consensus_label in ("Strong Buy", "Buy"):
    box_class = "recommendation-buy"
elif consensus_label in ("Sell", "Strong Sell"):
    box_class = "recommendation-sell"

st.markdown('<h2 class="sub-header">Consensus View</h2>', unsafe_allow_html=True)
st.markdown(
    f'<div class="{box_class}"><span style="font-size:1.6rem;">Consensus: <b>{consensus_label}</b></span><br>'
    f'(Score {consensus_score:.2f}; Momentum {momentum_score:.2f}, Yahoo rec {yahoo_norm:.2f}, '
    f'Finnhub rec {finnhub_norm:.2f}, Fib {fib_score:.2f})</div>',
    unsafe_allow_html=True
)

# My Opinion (ChatGPT if available; else rules)
snapshot = {
    "y_cp": y_cp, "y_pc": y_pc,
    "rsi": rsi_last, "macd": macd_last, "macd_sig": macd_sig_last,
    "yahoo_rec_score": yahoo_norm, "finnhub_rec_score": finnuhb_norm if 'finnuhb_norm' in locals() else finnuhb_norm,  # guard typo
    "consensus_label": consensus_label, "fib": fib
}
# Fix variable if typo occurred
if 'finnuhb_norm' in locals():
    snapshot["finnhub_rec_score"] = finnuhb_norm

# Opinion text
st.markdown('<div class="opinion-header">🧠 My Opinion</div>', unsafe_allow_html=True)
opinion_text = get_chatgpt_opinion(symbol, snapshot)

# Color-coded My Opinion label
if consensus_label in ("Strong Buy", "Buy"):
    color = "#2E7D32"  # green
elif consensus_label in ("Sell", "Strong Sell"):
    color = "#C62828"  # red
else:
    color = "#F57C00"  # orange

st.markdown(f'<div style="font-weight:800;font-size:1.1rem;color:{color};">My Opinion: {consensus_label}</div>', unsafe_allow_html=True)
st.write(opinion_text)
