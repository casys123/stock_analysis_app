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
st.markdown("### Yahoo + Finnhub • RSI/MACD • Consensus & My Opinion")

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
    FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY"  # replace locally if you don't want to use secrets

def get_openai_key():
    try:
        return st.secrets["openai"]["api_key"]
    except Exception:
        return None

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
    """Map UI timeframe to yfinance history period."""
    if tf == "1D": return "5d"
    if tf == "1W": return "1mo"
    if tf == "1M": return "3mo"
    if tf == "3M": return "6mo"
    if tf == "6M": return "1y"
    if tf == "YTD": return "ytd"
    if tf == "1Y": return "1y"
    if tf == "5Y": return "5y"
    return "1y"

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

    # Info
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # Recommendations
    try:
        recs_df = t.recommendations
        if recs_df is not None and not recs_df.empty:
            recs_df = recs_df.tail(30).copy()
        else:
            recs_df = pd.DataFrame()
    except Exception:
        recs_df = pd.DataFrame()

    # News (robust)
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_finnhub_data(ticker: str, tf: str):
    """Fetch candles + profile + quote + recommendations + news + metrics from Finnhub."""
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

    # Quote
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
        if not finnhub_recs_df.empty:
            finnhub_recs_df = finnhub_recs_df.head(12).copy()
    except Exception:
        finnhub_recs_df = pd.DataFrame()

    # Company news (30 days)
    try:
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=30)
        news = requests.get(
            f"{FINNHUB_URL}/company-news",
            params={"symbol": ticker, "from": str(from_date), "to": str(to_date), "token": FINNHUB_API_KEY},
            timeout=15
        ).json() or []
        rows = []
        for n in news[:30]:
            dt = n.get("datetime")
            rows.append({
                "provider": n.get("source", ""),
                "title": n.get("headline", ""),
                "link": n.get("url", ""),
                "published": datetime.utcfromtimestamp(dt).strftime("%Y-%m-%d %H:%M:%S") if dt else ""
            })
        finnhub_news_df = pd.DataFrame(rows)
    except Exception:
        finnhub_news_df = pd.DataFrame()

    # 52w high via metrics
    try:
        metrics = requests.get(
            f"{FINNHUB_URL}/stock/metric",
            params={"symbol": ticker, "metric": "price", "token": FINNHUB_API_KEY},
            timeout=15
        ).json() or {}
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
        "industry": profile.get("ipo"),  # Finnhub doesn't expose granular industry here
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

    # MAs
    out["MA20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["MA50"] = out["Close"].rolling(50, min_periods=50).mean()

    # RSI (Wilder)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.ewm(alpha=1/14, adjust=False).mean()
    roll_down = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    out["RSI"] = out["RSI"].clip(0, 100)

    # MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    return out

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
    if "MACD_Hist" in df:
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
        buys = ["buy", "strong buy", "overweight", "outperform"]
        sells = ["sell", "strong sell", "underperform"]
        holds = ["hold", "market perform", "neutral"]
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

def score_momentum(change_pct: float | None, rsi: float | None, macd: float | None, macd_sig: float | None) -> float:
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

def label_from_score(x: float) -> str:
    if x >= 1.5: return "Strong Buy"
    if x >= 0.5: return "Buy"
    if x > -0.5: return "Hold"
    if x > -1.5: return "Sell"
    return "Strong Sell"

# =========================================
# ChatGPT Opinion (optional; falls back to rules)
# =========================================
def get_chatgpt_opinion(symbol: str, snapshot: dict) -> str:
    y_cp = snapshot.get("y_cp"); y_pc = snapshot.get("y_pc")
    change_pct = ((y_cp - y_pc)/y_pc*100) if (y_cp and y_pc) else None
    rsi = snapshot.get("rsi")
    macd = snapshot.get("macd"); macd_sig = snapshot.get("macd_sig")
    yahoo_s = snapshot.get("yahoo_rec_score", 0.0)
    finn_s  = snapshot.get("finnhub_rec_score", 0.0)
    consensus = snapshot.get("consensus_label", "Hold")

    api_key = get_openai_key()
    if not api_key:
        # Rules-based fallback text
        macd_rel = ">" if (macd is not None and macd_sig is not None and macd > macd_sig) else "<="
        return (f"My take (rules-based): **{consensus}**. "
                f"Momentum Δ={change_pct:.2f}% | RSI={rsi:.1f} | MACD {macd_rel} Signal | "
                f"Yahoo recs={yahoo_s:.2f} | Finnhub recs={finn_s:.2f}") if change_pct is not None and rsi is not None else \
               (f"My take (rules-based): **{consensus}**. Yahoo recs={yahoo_s:.2f} | Finnhub recs={finn_s:.2f}")

    # Try OpenAI (optional)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        macd_state = ("bullish" if (macd is not None and macd_sig is not None and macd > macd_sig)
                      else "bearish" if (macd is not None and macd_sig is not None and macd < macd_sig)
                      else "N/A")
        chg_str  = f"{change_pct:.2f}%" if change_pct is not None else "N/A"
        rsi_str  = f"{rsi:.1f}" if rsi is not None else "N/A"

        prompt = f"""
You are an equity analyst. Give a concise Buy/Sell/Hold view for {symbol} in 3 short bullets.
Inputs:
- Price change vs prev close: {chg_str}
- RSI(14): {rsi_str}
- MACD vs Signal: {macd_state}
- Yahoo analyst score (−1..+1): {yahoo_s:.2f}
- Finnhub analyst score (−2..+2 normalized to similar scale): {finn_s:.2f}
- Preliminary consensus: {consensus}

Output:
1) Overall rating: **Buy**, **Hold**, or **Sell**
2) One-line rationale using inputs
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
        return (f"My take (rules-based): **{consensus}**. "
                f"Momentum Δ={change_pct:.2f}% | RSI={rsi:.1f} | MACD {macd_rel} Signal | "
                f"Yahoo recs={yahoo_s:.2f} | Finnhub recs={finn_s:.2f}") if change_pct is not None and rsi is not None else \
               (f"My take (rules-based): **{consensus}**. Yahoo recs={yahoo_s:.2f} | Finnhub recs={finn_s:.2f}")

# =========================================
# MAIN APP FLOW
# =========================================
with st.spinner("Fetching data from Yahoo & Finnhub…"):
    y = fetch_yahoo_data(symbol, time_frame)
    f = fetch_finnhub_data(symbol, time_frame)

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
        show_cols = [c for c in ["period", "strongBuy", "buy", "hold", "sell", "strongSell"] if c in fr.columns]
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
                if link:
                    st.write(f"[Open article]({link})")
    else:
        st.info("No Yahoo news found (or response had missing fields).")

with nc2:
    st.markdown("**Finnhub Company News (30d)**")
    fn = f.get("news")
    if isinstance(fn, pd.DataFrame) and not fn.empty:
        for _, row in fn.head(10).iterrows():
            title = row.get("title", "(no title)") or "(no title)"
            provider = row.get("provider", "")
            published = row.get("published", "")
            with st.expander(f"{title} — {provider} ({published})"):
                link = row.get("link", "")
                if link:
                    st.write(f"[Open article]({link})")
    else:
        st.info("No Finnhub news available (or missing API key).")

# =========================
# Consensus & My Opinion
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

# Normalize & combine
yahoo_norm = max(min(yahoo_rec_score, 1), -1)            # −1..+1
finnhub_norm = (finnhub_rec_score / 2)                   # scale to −1..+1 approx
consensus_score = 0.3*yahoo_norm + 0.3*finnhub_norm + 0.4*momentum_score
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
    f'(Score {consensus_score:.2f}; Momentum {momentum_score:.2f}, Yahoo rec {yahoo_norm:.2f}, Finnhub rec {finnhub_norm:.2f})</div>',
    unsafe_allow_html=True
)

# My Opinion (ChatGPT if available; else rules)
snapshot = {
    "y_cp": y_cp, "y_pc": y_pc,
    "rsi": rsi_last, "macd": macd_last, "macd_sig": macd_sig_last,
    "yahoo_rec_score": yahoo_norm, "finnhub_rec_score": finnhub_norm,
    "consensus_label": consensus_label
}
st.markdown('<div class="opinion-header">🧠 My Opinion</div>', unsafe_allow_html=True)
opinion = get_chatgpt_opinion(symbol, snapshot)
st.write(opinion)
