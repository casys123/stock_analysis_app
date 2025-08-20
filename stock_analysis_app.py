import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import seaborn as sns
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="StockInsight Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        border-bottom: 2px solid #64B5F6;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stock-card {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive {
        color: #2E7D32;
        font-weight: bold;
    }
    .negative {
        color: #C62828;
        font-weight: bold;
    }
    .metric-card {
        background-color: #BBDEFB;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    .recommendation-buy {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    .recommendation-hold {
        background-color: #FFF8E1;
        color: #F57C00;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    .recommendation-sell {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    .news-card {
        background-color: #FFF3E0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">📈 StockInsight Pro</h1>', unsafe_allow_html=True)
st.markdown("### Your All-in-One Stock Analysis Dashboard")

# Sidebar for user input
st.sidebar.header("Stock Selection")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", "AAPL").upper()

analysis_type = st.sidebar.selectbox(
    "Select Analysis Type:",
    ["Comprehensive Analysis", "Technical Analysis", "Fundamental Analysis", "Options Analysis"]
)

time_frame = st.sidebar.selectbox(
    "Select Time Frame:",
    ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "5Y"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 This tool aggregates data from multiple sources to provide comprehensive stock analysis and investment recommendations.")

# Function to fetch stock data from Yahoo Finance
def fetch_yahoo_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        
        # Get current price and previous close
        current_price = info.get('currentPrice', hist['Close'][-1] if not hist.empty else 0)
        previous_close = info.get('previousClose', hist['Close'][-2] if len(hist) > 1 else current_price)
        
        # Get analyst recommendations
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            latest_recommendation = recommendations.iloc[-1] if not recommendations.empty else None
        else:
            latest_recommendation = None
            
        return {
            'history': hist,
            'info': info,
            'current_price': current_price,
            'previous_close': previous_close,
            'recommendations': recommendations,
            'latest_recommendation': latest_recommendation
        }
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return None

# Function to fetch financial news
def fetch_financial_news(ticker):
    """Fetch financial news for a given stock ticker"""
    try:
        # In a real application, you would use a news API here
        # For demonstration, we'll return some sample news
        news_items = [
            {
                "title": f"{ticker} Announces Strong Quarterly Results",
                "source": "Financial Times",
                "date": "2023-06-15",
                "summary": f"{ticker} reported better-than-expected earnings for the last quarter, with revenue growth of 15% year-over-year."
            },
            {
                "title": f"Analysts Upgrade {ticker} to Buy Rating",
                "source": "Bloomberg",
                "date": "2023-06-10",
                "summary": f"Several analysts have upgraded {ticker} from Hold to Buy, citing strong growth potential in emerging markets."
            },
            {
                "title": f"{ticker} Launches New Product Line",
                "source": "Reuters",
                "date": "2023-06-05",
                "summary": f"{ticker} has announced a new product line that is expected to generate $500M in revenue next year."
            },
            {
                "title": f"{ticker} Faces Regulatory Challenges",
                "source": "Wall Street Journal",
                "date": "2023-05-28",
                "summary": f"{ticker} is facing new regulatory challenges in European markets that could impact future growth."
            },
            {
                "title": f"{ticker} Expands Partnership with Major Tech Company",
                "source": "CNBC",
                "date": "2023-05-20",
                "summary": f"{ticker} has expanded its partnership with a major tech company, which could lead to new revenue streams."
            }
        ]
        return news_items
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Function to calculate technical indicators
def calculate_technical_indicators(hist_data):
    """Calculate technical indicators from historical data"""
    try:
        # Calculate moving averages
        hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
        hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = hist_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist_data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp12 = hist_data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist_data['Close'].ewm(span=26, adjust=False).mean()
        hist_data['MACD'] = exp12 - exp26
        hist_data['MACD_Signal'] = hist_data['MACD'].ewm(span=9, adjust=False).mean()
        hist_data['MACD_Histogram'] = hist_data['MACD'] - hist_data['MACD_Signal']
        
        return hist_data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return hist_data

# Function to generate investment recommendation
def generate_recommendation(data, technical_data):
    """Generate investment recommendation based on multiple factors"""
    try:
        if not data or technical_data.empty:
            return "Insufficient data for recommendation"
        
        # Get current price and previous close
        current_price = data['current_price']
        previous_close = data['previous_close']
        
        # Calculate price change
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100
        
        # Get RSI value
        rsi = technical_data['RSI'].iloc[-1] if 'RSI' in technical_data else 50
        
        # Simple recommendation logic (in a real app, this would be more sophisticated)
        score = 0
        
        # Price momentum
        if price_change_pct > 2:
            score += 1
        elif price_change_pct < -2:
            score -= 1
            
        # RSI analysis
        if rsi < 30:
            score += 1  # Oversold - potential buying opportunity
        elif rsi > 70:
            score -= 1  # Overbought - potential selling opportunity
            
        # Generate final recommendation
        if score >= 2:
            return "Strong Buy"
        elif score >= 1:
            return "Buy"
        elif score == 0:
            return "Hold"
        elif score <= -1:
            return "Sell"
        else:
            return "Strong Sell"
            
    except Exception as e:
        st.error(f"Error generating recommendation: {e}")
        return "Hold"

# Function to create price chart
def create_price_chart(hist_data, ticker):
    """Create a price chart with moving averages"""
    try:
        # Set up the plot
        rcParams['figure.figsize'] = 12, 8
        fig, ax = plt.subplots()
        
        # Plot closing price and moving averages
        ax.plot(hist_data.index, hist_data['Close'], label='Close', linewidth=2, color='#1E88E5')
        ax.plot(hist_data.index, hist_data['MA20'], label='20-Day MA', linestyle='--', color='#FF9800')
        ax.plot(hist_data.index, hist_data['MA50'], label='50-Day MA', linestyle='--', color='#D81B60')
        
        # Format the plot
        ax.set_title(f'{ticker} Price Chart', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for better date display
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error creating price chart: {e}")
        return None

# Function to create technical indicators chart
def create_technical_indicators_chart(hist_data, ticker):
    """Create a chart with technical indicators"""
    try:
        # Set up the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot RSI
        ax1.plot(hist_data.index, hist_data['RSI'], label='RSI', linewidth=2, color='#26A69A')
        ax1.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax1.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax1.set_title(f'{ticker} RSI (14 days)', fontsize=14)
        ax1.set_ylabel('RSI')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Plot MACD
        ax2.plot(hist_data.index, hist_data['MACD'], label='MACD', linewidth=2, color='#7B1FA2')
        ax2.plot(hist_data.index, hist_data['MACD_Signal'], label='Signal', linewidth=2, color='#FF9800')
        ax2.bar(hist_data.index, hist_data['MACD_Histogram'], label='Histogram', alpha=0.3, color='#26A69A')
        ax2.set_title(f'{ticker} MACD', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('MACD')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis for better date display
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error creating technical indicators chart: {e}")
        return None

# Main app logic
def main():
    # Display loading spinner while fetching data
    with st.spinner('Fetching data from multiple sources...'):
        # Fetch data from various sources
        yahoo_data = fetch_yahoo_data(ticker_symbol, time_frame)
        news_data = fetch_financial_news(ticker_symbol)
        
        # Calculate technical indicators if we have historical data
        if yahoo_data and not yahoo_data['history'].empty:
            technical_data = calculate_technical_indicators(yahoo_data['history'].copy())
        else:
            technical_data = pd.DataFrame()
            
        # Generate investment recommendation
        recommendation = generate_recommendation(yahoo_data, technical_data)
    
    # Display main stock information
    if yahoo_data:
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.markdown(f"### {ticker_symbol}")
            st.markdown(f"**Sector:** {yahoo_data['info'].get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {yahoo_data['info'].get('industry', 'N/A')}")
            
        with col2:
            current_price = yahoo_data['current_price']
            previous_close = yahoo_data['previous_close']
            price_change = current_price - previous_close
            price_change_pct = (price_change / previous_close) * 100
            
            st.markdown(f"### ${current_price:.2f}")
            if price_change >= 0:
                st.markdown(f"<p class='positive'>+{price_change:.2f} ({price_change_pct:.2f}%)</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='negative'>{price_change:.2f} ({price_change_pct:.2f}%)</p>", unsafe_allow_html=True)
                
        with col3:
            if recommendation == "Strong Buy" or recommendation == "Buy":
                st.markdown(f'<div class="recommendation-buy">Recommendation: {recommendation}</div>', unsafe_allow_html=True)
            elif recommendation == "Hold":
                st.markdown(f'<div class="recommendation-hold">Recommendation: {recommendation}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="recommendation-sell">Recommendation: {recommendation}</div>', unsafe_allow_html=True)
            
            st.markdown(f"**Market Cap:** ${yahoo_data['info'].get('marketCap', 0):,}")
            st.markdown(f"**P/E Ratio:** {yahoo_data['info'].get('trailingPE', 'N/A')}")
    
    # Display charts and analysis
    st.markdown('<h2 class="sub-header">Price Chart & Technical Analysis</h2>', unsafe_allow_html=True)
    
    if not technical_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Create and display price chart
            price_chart = create_price_chart(technical_data, ticker_symbol)
            if price_chart:
                st.pyplot(price_chart)
        
        with col2:
            # Create and display technical indicators chart
            tech_chart = create_technical_indicators_chart(technical_data, ticker_symbol)
            if tech_chart:
                st.pyplot(tech_chart)
    else:
        st.warning("No historical data available for technical analysis.")
    
    # Display fundamental analysis
    st.markdown('<h2 class="sub-header">Fundamental Analysis</h2>', unsafe_allow_html=True)
    
    if yahoo_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**P/E Ratio:** {yahoo_data['info'].get('trailingPE', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**EPS:** {yahoo_data['info'].get('trailingEps', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Profit Margin:** {yahoo_data['info'].get('profitMargins', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**ROE:** {yahoo_data['info'].get('returnOnEquity', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Dividend Yield:** {yahoo_data['info'].get('dividendYield', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**52 Week High:** {yahoo_data['info'].get('fiftyTwoWeekHigh', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No fundamental data available for this stock.")
    
    # Display recent news
    st.markdown('<h2 class="sub-header">Recent News</h2>', unsafe_allow_html=True)
    
    if news_data:
        for news_item in news_data:
            with st.expander(f"{news_item['title']} - {news_item['source']} ({news_item['date']})"):
                st.markdown(f"<div class='news-card'>{news_item['summary']}</div>", unsafe_allow_html=True)
    else:
        st.info("No recent news available for this stock.")
    
    # Display investment thesis
    st.markdown('<h2 class="sub-header">Investment Thesis</h2>', unsafe_allow_html=True)
    
    if yahoo_data and not technical_data.empty:
        # Generate investment thesis based on available data
        current_price = yahoo_data['current_price']
        previous_close = yahoo_data['previous_close']
        price_change_pct = ((current_price - previous_close) / previous_close) * 100
        
        rsi = technical_data['RSI'].iloc[-1] if 'RSI' in technical_data else 50
        
        st.markdown("### Summary Analysis")
        
        if price_change_pct > 5:
            st.markdown(f"- The stock has shown <span class='positive'>strong positive momentum</span> recently, gaining {price_change_pct:.2f}%", unsafe_allow_html=True)
        elif price_change_pct < -5:
            st.markdown(f"- The stock has experienced <span class='negative'>significant downward pressure</span>, declining {abs(price_change_pct):.2f}%", unsafe_allow_html=True)
        else:
            st.markdown(f"- The stock has shown relatively <span class='neutral'>stable performance</span> recently, with a change of {price_change_pct:.2f}%", unsafe_allow_html=True)
        
        if rsi < 30:
            st.markdown(f"- The RSI of {rsi:.2f} indicates the stock is <span class='positive'>oversold</span>, which could present a buying opportunity", unsafe_allow_html=True)
        elif rsi > 70:
            st.markdown(f"- The RSI of {rsi:.2f} suggests the stock is <span class='negative'>overbought</span>, which might indicate a potential pullback", unsafe_allow_html=True)
        else:
            st.markdown(f"- The RSI of {rsi:.2f} is in neutral territory, suggesting balanced buying and selling pressure", unsafe_allow_html=True)
        
        st.markdown("### Potential Risks")
        st.markdown("- Market volatility could impact short-term performance")
        st.markdown("- Changes in industry regulations could affect future growth")
        st.markdown("- Economic conditions may influence consumer demand")
        
        st.markdown("### Potential Opportunities")
        st.markdown("- Strong fundamentals support long-term growth prospects")
        st.markdown("- Innovation in products/services could drive market share gains")
        st.markdown("- Expansion into new markets may provide additional revenue streams")

if __name__ == "__main__":
    main()
