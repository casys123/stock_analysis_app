import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import requests
import base64
import json

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
    .github-section {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        border-left: 5px solid #24292e;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">📈 StockInsight Pro</h1>', unsafe_allow_html=True)
st.markdown("### Your All-in-One Stock Analysis Dashboard with GitHub Integration")

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

# GitHub integration section
st.sidebar.markdown("---")
st.sidebar.header("GitHub Integration")
github_repo = st.sidebar.text_input("GitHub Repository (username/repo):", "your-username/stock-analysis")
github_token = st.sidebar.text_input("GitHub Token (optional):", type="password")
save_to_github = st.sidebar.checkbox("Save analysis to GitHub")

st.sidebar.markdown("---")
st.sidebar.info("💡 This tool aggregates data from multiple sources to provide comprehensive stock analysis and investment recommendations.")

# Function to fetch stock data from Yahoo Finance
@st.cache_data(ttl=3600)  # Cache for 1 hour
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

# Function to fetch data from GitHub
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_github_data(repo, ticker):
    """Fetch fundamental data from GitHub repository"""
    try:
        # URL to raw fundamental data (this is a placeholder - you would need to set up your own data)
        url = f"https://raw.githubusercontent.com/{repo}/main/data/{ticker}_fundamentals.json"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            # Return sample data for demonstration
            return {
                "pe_ratio": np.random.uniform(10, 30),
                "eps": np.random.uniform(1, 5),
                "profit_margin": np.random.uniform(0.1, 0.3),
                "roe": np.random.uniform(0.1, 0.25),
                "dividend_yield": np.random.uniform(0.01, 0.04),
                "debt_to_equity": np.random.uniform(0.5, 1.5),
                "source": "Sample Data (add your GitHub repo for real data)"
            }
    except Exception as e:
        st.error(f"Error fetching data from GitHub: {e}")
        return None

# Function to save analysis to GitHub
def save_analysis_to_github(repo, token, ticker, analysis_data):
    """Save analysis results to GitHub"""
    try:
        if not token:
            st.warning("GitHub token is required to save analysis")
            return False
            
        # Prepare the data
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{ticker}_analysis_{date_str}.json"
        
        # GitHub API URL
        url = f"https://api.github.com/repos/{repo}/contents/{filename}"
        
        # Encode content
        content = json.dumps(analysis_data, indent=2)
        content_bytes = content.encode("ascii")
        content_base64 = base64.b64encode(content_bytes).decode("ascii")
        
        # Prepare headers
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Check if file exists
        response = requests.get(url, headers=headers)
        sha = None
        if response.status_code == 200:
            sha = response.json().get("sha")
        
        # Prepare data for PUT request
        data = {
            "message": f"Add analysis for {ticker} on {date_str}",
            "content": content_base64,
        }
        
        if sha:
            data["sha"] = sha
            
        # Make the request
        response = requests.put(url, headers=headers, json=data)
        
        if response.status_code in [200, 201]:
            st.success(f"Analysis successfully saved to GitHub: {filename}")
            return True
        else:
            st.error(f"Failed to save to GitHub: {response.json().get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        st.error(f"Error saving to GitHub: {e}")
        return False

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
def generate_recommendation(data, technical_data, github_data):
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
        
        # Simple recommendation logic
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
            
        # Fundamental analysis from GitHub data
        if github_data:
            pe_ratio = github_data.get('pe_ratio', 0)
            if pe_ratio < 15:
                score += 1  # Potentially undervalued
            elif pe_ratio > 25:
                score -= 1  # Potentially overvalued
                
            roe = github_data.get('roe', 0)
            if roe > 0.15:
                score += 1  # Good profitability
                
            debt_to_equity = github_data.get('debt_to_equity', 1)
            if debt_to_equity < 1:
                score += 1  # Healthy debt levels
        
        # Generate final recommendation
        if score >= 3:
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

# Main app logic
def main():
    # Display loading spinner while fetching data
    with st.spinner('Fetching data from multiple sources...'):
        # Fetch data from various sources
        yahoo_data = fetch_yahoo_data(ticker_symbol, time_frame)
        github_data = fetch_github_data(github_repo, ticker_symbol)
        
        # Calculate technical indicators if we have historical data
        if yahoo_data and not yahoo_data['history'].empty:
            technical_data = calculate_technical_indicators(yahoo_data['history'].copy())
        else:
            technical_data = pd.DataFrame()
            
        # Generate investment recommendation
        recommendation = generate_recommendation(yahoo_data, technical_data, github_data)
    
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
        # Create price chart with moving averages
        fig = go.Figure()
        
        # Add closing price
        fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['Close'],
                                mode='lines', name='Close', line=dict(color='#1E88E5')))
        
        # Add moving averages
        fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['MA20'],
                                mode='lines', name='20-Day MA', line=dict(color='#FF9800')))
        
        fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['MA50'],
                                mode='lines', name='50-Day MA', line=dict(color='#D81B60')))
        
        fig.update_layout(
            title=f'{ticker_symbol} Price Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create technical indicators charts
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=technical_data.index, y=technical_data['RSI'],
                                        mode='lines', name='RSI', line=dict(color='#26A69A')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(title='RSI (14 days)', height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)
            
        with col2:
            # MACD chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=technical_data.index, y=technical_data['MACD'],
                                         mode='lines', name='MACD', line=dict(color='#7B1FA2')))
            fig_macd.add_trace(go.Scatter(x=technical_data.index, y=technical_data['MACD_Signal'],
                                         mode='lines', name='Signal', line=dict(color='#FF9800')))
            fig_macd.update_layout(title='MACD', height=300)
            st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.warning("No historical data available for technical analysis.")
    
    # Display fundamental analysis from GitHub
    st.markdown('<h2 class="sub-header">Fundamental Analysis</h2>', unsafe_allow_html=True)
    
    if github_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**P/E Ratio:** {github_data.get('pe_ratio', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**EPS:** {github_data.get('eps', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Profit Margin:** {github_data.get('profit_margin', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**ROE:** {github_data.get('roe', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Dividend Yield:** {github_data.get('dividend_yield', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Debt to Equity:** {github_data.get('debt_to_equity', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.caption(f"Source: {github_data.get('source', 'GitHub')}")
    else:
        st.info("No fundamental data available. Set up a GitHub repository with fundamental data.")
    
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
        
        if github_data:
            pe_ratio = github_data.get('pe_ratio', 0)
            if pe_ratio < 15:
                st.markdown(f"- The P/E ratio of {pe_ratio:.2f} suggests the stock may be <span class='positive'>undervalued</span> compared to industry averages", unsafe_allow_html=True)
            elif pe_ratio > 25:
                st.markdown(f"- The P/E ratio of {pe_ratio:.2f} suggests the stock may be <span class='negative'>overvalued</span> compared to industry averages", unsafe_allow_html=True)
            else:
                st.markdown(f"- The P/E ratio of {pe_ratio:.2f} is in line with industry averages", unsafe_allow_html=True)
                
            roe = github_data.get('roe', 0)
            if roe > 0.15:
                st.markdown(f"- The Return on Equity (ROE) of {roe:.2%} indicates <span class='positive'>strong profitability</span>", unsafe_allow_html=True)
        
        st.markdown("### Potential Risks")
        st.markdown("- Market volatility could impact short-term performance")
        st.markdown("- Changes in industry regulations could affect future growth")
        st.markdown("- Economic conditions may influence consumer demand")
        
        st.markdown("### Potential Opportunities")
        st.markdown("- Strong fundamentals support long-term growth prospects")
        st.markdown("- Innovation in products/services could drive market share gains")
        st.markdown("- Expansion into new markets may provide additional revenue streams")
    
    # GitHub integration section
    st.markdown("---")
    st.markdown('<div class="github-section">', unsafe_allow_html=True)
    st.markdown("### 💾 GitHub Integration")
    
    if save_to_github:
        # Prepare analysis data for saving
        analysis_data = {
            "ticker": ticker_symbol,
            "date": datetime.now().isoformat(),
            "price": yahoo_data['current_price'] if yahoo_data else 0,
            "recommendation": recommendation,
            "technical_indicators": {
                "rsi": technical_data['RSI'].iloc[-1] if not technical_data.empty and 'RSI' in technical_data else "N/A",
                "macd": technical_data['MACD'].iloc[-1] if not technical_data.empty and 'MACD' in technical_data else "N/A"
            },
            "fundamentals": github_data if github_data else {}
        }
        
        if st.button("Save Analysis to GitHub"):
            success = save_analysis_to_github(github_repo, github_token, ticker_symbol, analysis_data)
            if success:
                st.success("Analysis saved successfully!")
    
    st.markdown("""
    #### How to set up GitHub integration:
    1. Create a GitHub repository for your stock analysis
    2. Generate a personal access token with repo permissions
    3. Add your repository name (username/repo) and token in the sidebar
    4. Check "Save analysis to GitHub" and click the save button
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
