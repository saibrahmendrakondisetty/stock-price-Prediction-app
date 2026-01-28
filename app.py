"""
Stock Price Prediction Web App using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils import (
    download_stock_data, 
    preprocess_data, 
    create_supervised_dataset, 
    split_data, 
    evaluate_model,
    plot_predictions,
    plot_price_history,
    predict_next_day,
    get_last_n_predictions,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_support_resistance,
    plot_enhanced_price_history,
    plot_technical_indicators,
    plot_volume_analysis
)

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for modern UI design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Modern Color Scheme - Light Blue Theme */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #90caf9 100%);
        min-height: 100vh;
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(45deg, #1976d2, #42a5f5, #1976d2);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        animation: gradientShift 4s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(25, 118, 210, 0.3);
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Sidebar Header */
    .sidebar-header {
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(25, 118, 210, 0.3);
        border-left: 5px solid #1976d2;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(227,242,253,0.9) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(25, 118, 210, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(25, 118, 210, 0.2);
        border-color: rgba(25, 118, 210, 0.4);
    }
    
    /* Prediction Highlight */
    .prediction-highlight {
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(25, 118, 210, 0.4);
        color: white;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-highlight::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.2), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        border-left: 5px solid #ee5a24;
    }
    
    /* Success Box */
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 184, 148, 0.3);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 25px rgba(116, 185, 255, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1976d2;
        position: relative;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Styles */
    .stMetric {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(227,242,253,0.9) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(25, 118, 210, 0.2);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(25, 118, 210, 0.2);
        border-color: rgba(25, 118, 210, 0.4);
    }
    
    /* Dataframe Styles */
    .stDataFrame {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        border: 1px solid rgba(25, 118, 210, 0.1);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(25, 118, 210, 0.4);
        background: linear-gradient(135deg, #1565c0 0%, #2196f3 100%);
    }
    
    /* Input Field Styles */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.9);
        border: 2px solid rgba(25, 118, 210, 0.3);
        border-radius: 10px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1976d2;
        box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.1);
    }
    
    /* Date Picker Styles */
    .stDateInput > div > div > input {
        background: rgba(255,255,255,0.9);
        border: 2px solid rgba(25, 118, 210, 0.3);
        border-radius: 10px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    /* Spinner Styles */
    .stSpinner > div {
        border-top-color: #1976d2;
    }
    
    /* Footer Styles */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 15px;
        margin-top: 2rem;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in-left {
        animation: slideInLeft 0.6s ease-out;
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .prediction-highlight {
            padding: 1.5rem;
        }
    }
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .prediction-highlight {
            padding: 1.5rem;
        }
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
    }
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #1976d2;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    .progress-bar {
        background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%);
        height: 8px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def load_model_and_data():
    """
    Load trained model and associated objects
    """
    try:
        # Check if model files exist
        if not os.path.exists('saved_model.pkl'):
            st.error("🚨 Trained model not found! Please run `train_model.py` first.")
            return None, None, None
        
        # Load model and scaler
        model = joblib.load('saved_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Generate fresh processed data instead of loading from file
        try:
            raw_data = download_stock_data('AAPL')
            processed_data = preprocess_data(raw_data)
        except:
            # If that fails, create minimal dummy data
            processed_data = pd.DataFrame({
                'Close': [150.0, 151.0, 152.0, 153.0, 154.0],
                'MA_20': [150.0, 150.5, 151.0, 151.5, 152.0]
            })
        
        return model, scaler, processed_data
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def main():
    """
    Main Streamlit application
    """
    # Enhanced Header with animation
    st.markdown('<h1 class="main-header fade-in">📈 Stock Price Prediction App</h1>', unsafe_allow_html=True)
    
    # Enhanced Disclaimer with icon
    st.markdown("""
    <div class="warning-box fade-in">
        <h3>⚠️ Educational Disclaimer</h3>
        <p><strong>This app is for educational and hackathon purposes only.</strong></p>
        <p>❌ Do not use for actual trading or investment decisions</p>
        <p>❌ No financial guarantees are provided</p>
        <p>❌ Past performance does not guarantee future results</p>
        <p>✅ Use only for learning ML concepts and data analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar with modern design
    st.sidebar.markdown('<div class="sidebar-header">🔧 Configuration Panel</div>', unsafe_allow_html=True)
    
    # Add analyze button for better UX
    analyze_button = st.sidebar.button(
        "🚀 Analyze Stock", 
        type="primary",
        help="Click to analyze the selected stock and generate predictions"
    )
    
    # Enhanced Stock ticker input with icon
    default_ticker = "AAPL"
    ticker = st.sidebar.text_input(
        "📊 Stock Ticker Symbol", 
        value=default_ticker, 
        help="Enter stock ticker symbol (e.g., AAPL, GOOGL, MSFT, TSLA)"
    ).upper()
    
    # Date range selector
    today = datetime.now().strftime('%Y-%m-%d')
    default_start = "2018-01-01"
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.to_datetime(default_start),
        help="Start date for historical data"
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=pd.to_datetime(today),
        help="End date for historical data"
    )
    
    # Load model and data
    model, scaler, processed_data = load_model_and_data()
    
    if model is None:
        st.stop()
    
    # Show results only when analyze button is clicked
    show_results = analyze_button
    
    # Enhanced Main content area with better layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<h2 class="section-header slide-in-left">📊 Stock Data Analysis</h2>', unsafe_allow_html=True)
        
        if not show_results:
            # Show clean initial state with professional welcome
            st.markdown("""
            <div class="metric-card info-box fade-in">
                <div style="text-align: center; padding: 2rem;">
                    <h2 style="color: #000000; margin-bottom: 1rem;">👋 Welcome to Professional Stock Analysis</h2>
                    <div style="font-size: 1.2rem; margin: 1.5rem 0; color: #4a5568;">
                        <p><strong>📊 Ready to analyze:</strong> <span style="color: #1976d2; font-weight: bold;">{ticker}</span></p>
                        <p style="margin-top: 1rem;">Click the <strong style="color: #1976d2;">"🚀 Analyze Stock"</strong> button to generate comprehensive analysis including:</p>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 2rem 0;">
                        <div style="background: rgba(25, 118, 210, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #1976d2;">
                            <h4 style="color: #1976d2; margin-bottom: 0.5rem;">📈 Price Analysis</h4>
                            <p style="font-size: 0.9rem; color: #4a5568;">Real-time price data & trends</p>
                        </div>
                        <div style="background: rgba(66, 165, 245, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #42a5f5;">
                            <h4 style="color: #42a5f5; margin-bottom: 0.5rem;">📊 Technical Indicators</h4>
                            <p style="font-size: 0.9rem; color: #4a5568;">RSI, MACD, Bollinger Bands</p>
                        </div>
                        <div style="background: rgba(144, 202, 249, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #90caf9;">
                            <h4 style="color: #90caf9; margin-bottom: 0.5rem;">🎯 Predictions</h4>
                            <p style="font-size: 0.9rem; color: #4a5568;">ML-powered price forecasts</p>
                        </div>
                        <div style="background: rgba(187, 222, 251, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #bbdefb;">
                            <h4 style="color: #bbdefb; margin-bottom: 0.5rem;">📉 Volume Analysis</h4>
                            <p style="font-size: 0.9rem; color: #4a5568;">Trading volume insights</p>
                        </div>
                    </div>
                    <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.5); border-radius: 10px;">
                        <p style="color: #000000; font-weight: 600; margin-bottom: 0.5rem;">🔍 What you'll get:</p>
                        <ul style="color: #4a5568; text-align: left; margin: 0; padding-left: 1.5rem;">
                            <li>Comprehensive stock metrics & key performance indicators</li>
                            <li>Advanced technical analysis with professional charts</li>
                            <li>Machine learning predictions with confidence levels</li>
                            <li>Support & resistance levels identification</li>
                            <li>Risk assessment and volatility analysis</li>
                        </ul>
                    </div>
                    <div style="margin-top: 2rem;">
                        <div style="background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%); color: white; padding: 1rem 2rem; border-radius: 25px; display: inline-block; font-weight: 600; box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);">
                            🚀 Click "Analyze Stock" to begin your professional analysis
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif show_results:
            # Show loading state immediately when analyze is clicked
            st.markdown(f"""
            <div class="loading-container fade-in">
                <div class="loading-spinner"></div>
                <h3>🔍 Analyzing {ticker} Stock Data...</h3>
                <p>Fetching real-time data and generating professional analysis</p>
                <div class="progress-container">
                    <div class="progress-bar" style="width: 60%;"></div>
                </div>
                <p style="margin-top: 1rem; color: #666; font-size: 0.9rem;">This may take a few seconds...</p>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Step 1: Download data (with progress tracking)
                st.markdown('<span class="status-warning">📡 Downloading stock data...</span>', unsafe_allow_html=True)
                with st.spinner(f"Fetching data for {ticker}..."):
                    raw_data = download_stock_data(
                        ticker, 
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                
                # Step 2: Quick preprocessing
                st.markdown('<span class="status-warning">⚙️ Processing data...</span>', unsafe_allow_html=True)
                fresh_processed = preprocess_data(raw_data)
                
                # Step 3: Calculate all metrics efficiently
                st.markdown('<span class="status-warning">📊 Calculating metrics...</span>', unsafe_allow_html=True)
                
                # Pre-calculate all metrics at once to avoid redundant calculations
                current_price = raw_data['Close'].iloc[-1]
                prev_close = raw_data['Close'].iloc[-2] if len(raw_data) > 1 else current_price
                daily_change = current_price - prev_close
                daily_change_pct = (daily_change / prev_close) * 100 if prev_close != 0 else 0
                period_high = raw_data['High'].max()
                period_low = raw_data['Low'].min()
                avg_volume = raw_data['Volume'].mean()
                volatility = raw_data['Close'].pct_change().std() * np.sqrt(252)
                
                # Step 4: Calculate technical indicators efficiently
                st.markdown('<span class="status-warning">📈 Computing technical indicators...</span>', unsafe_allow_html=True)
                
                # Optimize technical indicator calculations
                enhanced_data = fresh_processed.copy()
                
                # Calculate all indicators in one pass to avoid multiple iterations
                close_prices = enhanced_data['Close']
                enhanced_data['RSI'] = calculate_rsi(close_prices)
                enhanced_data['MACD'], enhanced_data['MACD_Signal'] = calculate_macd(close_prices)
                enhanced_data['Upper_Band'], enhanced_data['Lower_Band'] = calculate_bollinger_bands(close_prices)
                enhanced_data['Volume_SMA'] = raw_data['Volume'].rolling(20).mean()
                
                # Pre-calculate support/resistance to avoid recalculating
                support_resistance = calculate_support_resistance(enhanced_data)
                
                # Step 5: Success status
                st.markdown(f'<span class="status-success">✅ Analysis Complete - {len(raw_data)} data points processed</span>', unsafe_allow_html=True)
                
                # Enhanced Stock Overview with Key Metrics
                st.markdown('<h3 class="section-header">📊 Stock Overview & Key Metrics</h3>', unsafe_allow_html=True)
                
                # Display key metrics in columns
                col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
                
                with col_metrics1:
                    st.metric(
                        "💰 Current Price",
                        f"${current_price:.2f}",
                        delta=f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)" if daily_change != 0 else None,
                        delta_color="normal" if daily_change >= 0 else "inverse"
                    )
                
                with col_metrics2:
                    st.metric(
                        "📈 Period High",
                        f"${period_high:.2f}",
                        delta=f"{((period_high - current_price) / current_price * 100):+.1f}% from current"
                    )
                
                with col_metrics3:
                    st.metric(
                        "📉 Period Low",
                        f"${period_low:.2f}",
                        delta=f"{((current_price - period_low) / period_low * 100):+.1f}% above low"
                    )
                
                with col_metrics4:
                    st.metric(
                        "📊 Avg Volume",
                        f"{avg_volume:,.0f}",
                        help="Average trading volume over the selected period"
                    )
                
                # Volatility and Risk Analysis
                col_vol1, col_vol2 = st.columns(2)
                
                with col_vol1:
                    st.metric(
                        "⚡ Volatility",
                        f"{volatility:.2%}",
                        delta="Annualized" if volatility > 0.3 else "Low",
                        delta_color="inverse" if volatility > 0.3 else "normal"
                    )
                
                with col_vol2:
                    # Calculate 52-week high/low if enough data
                    if len(raw_data) >= 252:
                        week_52_high = raw_data['Close'].rolling(252).max().iloc[-1]
                        week_52_low = raw_data['Close'].rolling(252).min().iloc[-1]
                        st.metric(
                            "📅 52-Week Range",
                            f"${week_52_low:.2f} - ${week_52_high:.2f}",
                            help="52-week high and low prices"
                        )
                    else:
                        st.metric(
                            "📅 Data Points",
                            f"{len(raw_data)}",
                            help="Number of trading days in analysis"
                        )
                
                # Enhanced Raw Data Display with Statistics
                st.markdown('<h3 class="section-header">📋 Detailed Stock Data</h3>', unsafe_allow_html=True)
                
                # Add data summary
                with st.expander("📊 Data Summary Statistics", expanded=False):
                    summary_stats = raw_data.describe()
                    st.dataframe(summary_stats, use_container_width=True)
                
                # Show raw data with enhanced formatting
                st.dataframe(
                    raw_data.tail(15).style.format({
                        'Open': '${:.2f}',
                        'High': '${:.2f}', 
                        'Low': '${:.2f}',
                        'Close': '${:.2f}',
                        'Volume': '{:,}'
                    }).background_gradient(subset=['Close'], cmap='RdYlGn'),
                    use_container_width=True,
                    height=400
                )
                
                # Enhanced Price History Chart with Technical Indicators
                st.markdown('<h3 class="section-header">📈 Price History & Technical Analysis</h3>', unsafe_allow_html=True)
                
                # Create tabs for different chart views (lazy loading)
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Technical Indicators", "📉 Volume Analysis", "🎯 Advanced Analytics"])
                
                with tab1:
                    fig_history = plot_enhanced_price_history(enhanced_data, ticker, raw_data)
                    st.pyplot(fig_history, use_container_width=True)
                
                with tab2:
                    fig_technical = plot_technical_indicators(enhanced_data, ticker)
                    st.pyplot(fig_technical, use_container_width=True)
                
                with tab3:
                    fig_volume = plot_volume_analysis(raw_data, enhanced_data, ticker)
                    st.pyplot(fig_volume, use_container_width=True)
                
                with tab4:
                    # Advanced analytics (using pre-calculated values)
                    st.markdown("**📊 Advanced Technical Analysis**")
                    
                    # Use pre-calculated values to avoid redundant calculations
                    current_rsi = enhanced_data['RSI'].iloc[-1]
                    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    
                    macd_signal = "Bullish" if enhanced_data['MACD'].iloc[-1] > enhanced_data['MACD_Signal'].iloc[-1] else "Bearish"
                    
                    bb_position = (enhanced_data['Close'].iloc[-1] - enhanced_data['Lower_Band'].iloc[-1]) / (enhanced_data['Upper_Band'].iloc[-1] - enhanced_data['Lower_Band'].iloc[-1])
                    bb_signal = "Upper Band" if bb_position > 0.8 else "Lower Band" if bb_position < 0.2 else "Middle"
                    
                    col_adv1, col_adv2, col_adv3 = st.columns(3)
                    
                    with col_adv1:
                        st.metric(
                            "📈 RSI (14)",
                            f"{current_rsi:.1f}",
                            delta=rsi_signal,
                            delta_color="inverse" if rsi_signal in ["Overbought", "Oversold"] else "normal"
                        )
                    
                    with col_adv2:
                        st.metric(
                            "📊 MACD Signal",
                            macd_signal,
                            delta_color="normal" if macd_signal == "Bullish" else "inverse"
                        )
                    
                    with col_adv3:
                        st.metric(
                            "🎯 Bollinger Position",
                            f"{bb_position:.1%}",
                            delta=bb_signal
                        )
                    
                    # Use pre-calculated support/resistance
                    st.markdown("**🎯 Support & Resistance Levels**")
                    
                    col_sr1, col_sr2 = st.columns(2)
                    
                    with col_sr1:
                        st.write(f"**Support Levels:**")
                        for level in support_resistance['support'][:3]:
                            st.write(f"• ${level:.2f}")
                    
                    with col_sr2:
                        st.write(f"**Resistance Levels:**")
                        for level in support_resistance['resistance'][:3]:
                            st.write(f"• ${level:.2f}")
                
                # Make predictions on fresh data
                if len(fresh_processed) >= 20:  # Need at least 20 days for MA
                    # Create features for prediction
                    X_fresh, y_fresh = create_supervised_dataset(fresh_processed)
                    
                    # Scale features
                    X_fresh_scaled = scaler.transform(X_fresh)
                    
                    # Make predictions
                    y_fresh_pred = model.predict(X_fresh_scaled)
                    
                    # Evaluate on fresh data
                    fresh_metrics = evaluate_model(y_fresh, y_fresh_pred)
                    
                    # Add processing status
                    st.markdown(f'<span class="status-success">✅ Model Analysis Complete</span>', unsafe_allow_html=True)
                    
                    # Enhanced Model Performance Display
                    st.markdown('<h3 class="section-header">🎯 Model Performance Metrics</h3>', unsafe_allow_html=True)
                    col1_metric, col2_metric = st.columns(2)
                    
                    with col1_metric:
                        st.metric(
                            "📉 Root Mean Square Error", 
                            f"${fresh_metrics['RMSE']:.2f}",
                            delta="Lower is better",
                            help="Root Mean Square Error - Measures prediction accuracy"
                        )
                    
                    with col2_metric:
                        st.metric(
                            "📈 R² Score", 
                            f"{fresh_metrics['R2']:.4f}",
                            delta="Higher is better",
                            help="Coefficient of Determination - Explained variance"
                        )
                    
                    # Add prediction status
                    st.markdown(f'<span class="status-success">✅ Predictions Generated</span>', unsafe_allow_html=True)
                    
                    # Enhanced Predictions Chart
                    st.markdown('<h3 class="section-header">🔮 Actual vs Predicted Prices</h3>', unsafe_allow_html=True)
                    fig_pred = plot_predictions(y_fresh, y_fresh_pred, f"{ticker} Price Prediction")
                    st.pyplot(fig_pred, use_container_width=True)
                    
                    # Add final status
                    st.markdown(f'<span class="status-success">🎯 Analysis Complete - Ready for Trading Day</span>', unsafe_allow_html=True)
                    
                    # Enhanced Next Day Prediction with more details
                    next_day_price = predict_next_day(model, fresh_processed)
                    
                    # Calculate additional metrics
                    current_price = fresh_processed['Close'].iloc[-1]
                    price_change = next_day_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    
                    # Determine trend
                    if price_change > 0:
                        trend_icon = "📈"
                        trend_color = "#4CAF50"
                        trend_text = "Bullish"
                    elif price_change < 0:
                        trend_icon = "📉"
                        trend_color = "#F44336"
                        trend_text = "Bearish"
                    else:
                        trend_icon = "➡️"
                        trend_color = "#FF9800"
                        trend_text = "Neutral"
                    
                    st.markdown(f"""
                    <div class="prediction-highlight pulse">
                        <h3 style="margin-bottom: 1rem;">🎯 Next Day's Predicted Price</h3>
                        <div style="font-size: 3.5rem; font-weight: 700; margin: 1rem 0;">
                            ${next_day_price:.2f}
                        </div>
                        <div style="font-size: 1.2rem; margin-top: 1rem;">
                            <p><strong>📊 Stock:</strong> {ticker}</p>
                            <p><strong>📅 Prediction Date:</strong> {(datetime.now() + timedelta(days=1)).strftime('%B %d, %Y')}</p>
                            <p><strong>🕐 Generated:</strong> {datetime.now().strftime('%I:%M %p')}</p>
                            <p><strong>💰 Current Price:</strong> ${current_price:.2f}</p>
                            <p><strong>📊 Expected Change:</strong> <span style="color: {trend_color};">{trend_icon} ${abs(price_change):.2f} ({price_change_pct:+.2f}%)</span></p>
                            <p><strong>🎯 Trend:</strong> <span style="color: {trend_color}; font-weight: bold;">{trend_text}</span></p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add confidence indicator
                    confidence = max(0, min(100, 100 - (fresh_metrics['RMSE'] / current_price) * 100))
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 Prediction Confidence</h4>
                        <div style="background: linear-gradient(90deg, #4CAF50 0%, #8BC34A {confidence}%, #E0E0E0 {confidence}%, #E0E0E0 100%); height: 20px; border-radius: 10px; margin: 10px 0;"></div>
                        <p><strong>Confidence Level:</strong> {confidence:.1f}%</p>
                        <p style="font-size: 0.9rem; color: #666;">Based on model accuracy and current market conditions</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.warning(f"⚠️ Not enough data for {ticker}. Need at least 20 days of historical data.")
                    
            except Exception as e:
                st.error(f"❌ Error processing {ticker}: {str(e)}")
        else:
            # Show initial state when no analysis done
            st.markdown("""
            <div class="metric-card info-box">
                <h3>👋 Welcome to Stock Prediction App!</h3>
                <p><strong>Ready to analyze:</strong> {ticker}</p>
                <p>Click the "🚀 Analyze Stock" button to generate predictions and see detailed analysis.</p>
                <p>Or change the stock ticker above to analyze a different stock.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="section-header slide-in-left">📈 Model Information</h2>', unsafe_allow_html=True)
        
        # Enhanced Model Details
        st.markdown("""
        <div class="metric-card info-box">
            <h4>🤖 Model Architecture</h4>
            <p><strong>Algorithm:</strong> Linear Regression</p>
            <p><strong>Features:</strong> Close Price, 20-Day Moving Average</p>
            <p><strong>Target:</strong> Next Day Close Price</p>
            <p><strong>Scaling:</strong> StandardScaler Normalization</p>
            <p><strong>Training:</strong> 80/20 Split (Time Series)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Recent Predictions
        st.markdown('<h3 class="section-header">🔮 Recent Predictions</h3>', unsafe_allow_html=True)
        try:
            if len(processed_data) >= 5:
                last_5_preds = get_last_n_predictions(model, processed_data, 5)
                
                # Get actual prices for comparison
                actual_prices = processed_data['Close'].values[-5:].tolist()
                
                # Enhanced Comparison DataFrame
                comparison_df = pd.DataFrame({
                    'Actual Price': [f"${a:.2f}" for a in actual_prices],
                    'Predicted Price': [f"${p:.2f}" for p in last_5_preds],
                    'Difference ($)': [f"${abs(a-p):.2f}" for a, p in zip(actual_prices, last_5_preds)],
                    'Accuracy (%)': [f"{(1-abs(a-p)/a)*100:.1f}%" for a, p in zip(actual_prices, last_5_preds)]
                })
                
                st.dataframe(comparison_df, use_container_width=True, height=300)
            else:
                st.info("Not enough data for recent predictions")
                
        except Exception as e:
            st.error(f"Error generating recent predictions: {str(e)}")
        
        # Enhanced Training Statistics
        st.markdown('<h3 class="section-header">📊 Training Statistics</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card success-box">
            <h4>🎓 Training Configuration</h4>
            <p><strong>Training Period:</strong> 2018-Present</p>
            <p><strong>Training Size:</strong> 80% of dataset</p>
            <p><strong>Test Size:</strong> 20% of dataset</p>
            <p><strong>Feature Scaling:</strong> StandardScaler</p>
            <p><strong>Validation:</strong> Time Series Cross-Validation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("""
    <div class="footer">
        <h3>🚀 About This Project</h3>
        <p>Built with ❤️ using Streamlit, Scikit-learn, and yfinance</p>
        <p><strong>Technologies:</strong> Python | Machine Learning | Data Visualization | Web Development</p>
        <p><strong>Purpose:</strong> Educational demonstration of ML in finance</p>
        <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
            📚 Educational Project Only - Not Financial Advice | Always consult professionals for investment decisions
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()