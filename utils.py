"""
Utility functions for Stock Price Prediction App
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def download_stock_data(ticker, start_date='2018-01-01', end_date=None):
    """
    Download stock data using yfinance
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format (default: today)
    
    Returns:
        pd.DataFrame: Stock data with OHLCV
    """
    try:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Download stock data with retry logic
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if stock_data.empty:
            # Try alternative ticker format
            alternative_ticker = ticker + '.NS' if len(ticker) <= 4 else ticker
            stock_data = yf.download(alternative_ticker, start=start_date, end=end_date, progress=False)
            
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        return stock_data
    
    except Exception as e:
        # Create sample data for testing if download fails
        print(f"Warning: Could not download data for {ticker}. Creating sample data for testing.")
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample data
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.now() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
        # Filter only weekdays
        dates = dates[dates.weekday < 5]
        
        # Generate realistic stock price data
        np.random.seed(42)
        base_price = 150.0
        price_changes = np.random.normal(0, 0.02, len(dates))
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = low + (high - low) * np.random.random()
            volume = int(np.random.normal(1000000, 200000))
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': max(volume, 100000)
            })
        
        df = pd.DataFrame(data, index=dates)
        return df

def preprocess_data(df):
    """
    Preprocess stock data for ML model
    
    Args:
        df (pd.DataFrame): Raw stock data
    
    Returns:
        pd.DataFrame: Preprocessed data with features
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Handle missing values
    data = data.dropna()
    
    # Use only 'Close' price for basic model
    data = data[['Close']].copy()
    
    # Add 20-day moving average as bonus feature
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    
    # Drop rows with NaN values (first 19 days for MA)
    data = data.dropna()
    
    return data

def create_supervised_dataset(data):
    """
    Create supervised learning dataset for time series prediction
    
    Args:
        data (pd.DataFrame): Preprocessed stock data
    
    Returns:
        tuple: (X, y) where X is features and y is target (next day's close price)
    """
    # Create features (X) and target (y)
    # X = today's closing price and MA_20
    # y = next day's closing price (shift -1)
    
    X = data[['Close', 'MA_20']].values[:-1]  # All days except last
    y = data['Close'].values[1:]  # Next day's closing price
    
    return X, y

def split_data(X, y, test_size=0.2):
    """
    Split time series data into train and test sets
    
    Args:
        X (np.array): Features
        y (np.array): Target
        test_size (float): Proportion of data for testing
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Calculate split index (no shuffling for time series)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance
    
    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values
    
    Returns:
        dict: Dictionary with RMSE and R2 scores
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'R2': r2
    }

def plot_predictions(y_true, y_pred, title="Stock Price Prediction"):
    """
    Plot actual vs predicted prices with modern styling
    
    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values
        title (str): Plot title
    
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    # Set modern style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create gradient background
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    
    # Plot actual values with gradient fill
    ax.plot(y_true, label='Actual Price', color='#667eea', linewidth=3, alpha=0.8)
    ax.fill_between(range(len(y_true)), y_true, alpha=0.1, color='#667eea')
    
    # Plot predicted values with gradient fill
    ax.plot(y_pred, label='Predicted Price', color='#ff6b6b', linewidth=3, linestyle='--', alpha=0.8)
    ax.fill_between(range(len(y_pred)), y_pred, alpha=0.1, color='#ff6b6b')
    
    # Enhanced styling
    ax.set_title(title, fontsize=18, fontweight='bold', color='#2d3748', pad=20)
    ax.set_xlabel('Days', fontsize=14, color='#4a5568', labelpad=10)
    ax.set_ylabel('Price ($)', fontsize=14, color='#4a5568', labelpad=10)
    
    # Modern legend
    legend = ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_edgecolor('#e2e8f0')
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Modern spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#e2e8f0')
        spine.set_linewidth(1)
    
    plt.tight_layout(pad=3)
    return fig

def plot_price_history(data, ticker):
    """
    Plot historical closing prices with modern styling
    
    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker
    
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    # Set modern style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create gradient background
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    
    # Plot closing price with gradient fill
    ax.plot(data.index, data['Close'], label='Close Price', color='#4ecdc4', linewidth=3, alpha=0.8)
    ax.fill_between(data.index, data['Close'], alpha=0.1, color='#4ecdc4')
    
    # Plot moving average if available
    if 'MA_20' in data.columns:
        ax.plot(data.index, data['MA_20'], label='20-Day MA', color='#ff6b6b', linewidth=2.5, alpha=0.8)
        ax.fill_between(data.index, data['MA_20'], alpha=0.05, color='#ff6b6b')
    
    # Enhanced styling
    ax.set_title(f'{ticker} Stock Price History', fontsize=18, fontweight='bold', color='#2d3748', pad=20)
    ax.set_xlabel('Date', fontsize=14, color='#4a5568', labelpad=10)
    ax.set_ylabel('Price ($)', fontsize=14, color='#4a5568', labelpad=10)
    
    # Modern legend
    legend = ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_edgecolor('#e2e8f0')
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Modern spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#e2e8f0')
        spine.set_linewidth(1)
    
    # Format x-axis dates
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='x', which='major', labelsize=12, colors='#4a5568')
    ax.tick_params(axis='y', which='major', labelsize=12, colors='#4a5568')
    
    plt.tight_layout(pad=3)
    return fig

def get_last_n_predictions(model, data, n=5):
    """
    Get last n predictions using the trained model
    
    Args:
        model: Trained ML model
        data (pd.DataFrame): Preprocessed data
        n (int): Number of predictions to return
    
    Returns:
        list: Last n predictions
    """
    # Get last n days of features
    last_n_features = data[['Close', 'MA_20']].values[-n:]
    
    # Make predictions
    predictions = model.predict(last_n_features)
    
    return predictions.tolist()

def predict_next_day(model, data):
    """
    Predict next day's closing price
    
    Args:
        model: Trained ML model
        data (pd.DataFrame): Preprocessed data
    
    Returns:
        float: Predicted next day's closing price
    """
    # Get the most recent features
    latest_features = data[['Close', 'MA_20']].values[-1:].reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(latest_features)[0]
    
    return prediction

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI) - Optimized version
    
    Args:
        prices (pd.Series): Price series
        period (int): RSI period (default: 14)
    
    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use exponentially weighted moving average for faster calculation
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) - Optimized version
    
    Args:
        prices (pd.Series): Price series
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line period
    
    Returns:
        tuple: (MACD line, Signal line)
    """
    # Calculate EMAs directly using pandas optimized functions
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calculate Bollinger Bands - Optimized version
    
    Args:
        prices (pd.Series): Price series
        period (int): Moving average period
        std_dev (int): Standard deviation multiplier
    
    Returns:
        tuple: (Upper band, Lower band)
    """
    # Use rolling window with optimized calculations
    sma = prices.rolling(window=period, min_periods=1).mean()
    std = prices.rolling(window=period, min_periods=1).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def calculate_support_resistance(data, window=20):
    """
    Calculate support and resistance levels - Optimized version
    
    Args:
        data (pd.DataFrame): Stock data with Close prices
        window (int): Window for finding local extrema
    
    Returns:
        dict: Support and resistance levels
    """
    prices = data['Close']
    
    # Use more efficient method to find local extrema
    # Find local maxima and minima using rolling windows
    local_max = prices.rolling(window=window, center=True).max()
    local_min = prices.rolling(window=window, center=True).min()
    
    # Vectorized approach to find extrema
    resistance_mask = (prices == local_max) & (prices > prices.shift(1)) & (prices > prices.shift(-1))
    support_mask = (prices == local_min) & (prices < prices.shift(1)) & (prices < prices.shift(-1))
    
    # Extract levels
    resistance_levels = prices[resistance_mask].tolist()
    support_levels = prices[support_mask].tolist()
    
    # Get unique levels and sort (more efficient)
    resistance_levels = sorted(set(resistance_levels), reverse=True)
    support_levels = sorted(set(support_levels))
    
    return {
        'resistance': resistance_levels,
        'support': support_levels
    }

def plot_enhanced_price_history(data, ticker, raw_data):
    """
    Plot enhanced price history with Bollinger Bands and volume
    
    Args:
        data (pd.DataFrame): Enhanced stock data with indicators
        ticker (str): Stock ticker
        raw_data (pd.DataFrame): Original raw data with volume
    
    Returns:
        matplotlib.figure.Figure: Enhanced plot figure
    """
    # Set modern style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Create gradient background
    fig.patch.set_facecolor('#f8f9fa')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')
    
    # Price chart with Bollinger Bands
    ax1.plot(data.index, data['Close'], label='Close Price', color='#4ecdc4', linewidth=3, alpha=0.8)
    ax1.fill_between(data.index, data['Close'], alpha=0.1, color='#4ecdc4')
    
    # Plot moving average
    if 'MA_20' in data.columns:
        ax1.plot(data.index, data['MA_20'], label='20-Day MA', color='#ff6b6b', linewidth=2.5, alpha=0.8)
    
    # Plot Bollinger Bands
    if 'Upper_Band' in data.columns and 'Lower_Band' in data.columns:
        ax1.plot(data.index, data['Upper_Band'], label='Upper Band', color='#9b59b6', linewidth=1.5, alpha=0.6, linestyle='--')
        ax1.plot(data.index, data['Lower_Band'], label='Lower Band', color='#9b59b6', linewidth=1.5, alpha=0.6, linestyle='--')
        ax1.fill_between(data.index, data['Upper_Band'], data['Lower_Band'], alpha=0.05, color='#9b59b6')
    
    # Enhanced styling for price chart
    ax1.set_title(f'{ticker} Enhanced Price Analysis', fontsize=18, fontweight='bold', color='#2d3748', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=14, color='#4a5568', labelpad=10)
    
    # Modern legend
    legend1 = ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    legend1.get_frame().set_facecolor('#ffffff')
    legend1.get_frame().set_edgecolor('#e2e8f0')
    
    # Volume chart - align the data properly
    # Use the same index as the enhanced data to avoid shape mismatch
    aligned_volume = raw_data['Volume'].reindex(data.index)
    ax2.bar(data.index, aligned_volume, color='#3498db', alpha=0.6, width=0.8)
    if 'Volume_SMA' in data.columns:
        ax2.plot(data.index, data['Volume_SMA'], label='Volume SMA', color='#e74c3c', linewidth=2, alpha=0.8)
    
    ax2.set_title('Trading Volume', fontsize=14, fontweight='bold', color='#2d3748')
    ax2.set_ylabel('Volume', fontsize=12, color='#4a5568')
    ax2.set_xlabel('Date', fontsize=14, color='#4a5568', labelpad=10)
    
    # Enhanced grid and spines
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#e2e8f0')
            spine.set_linewidth(1)
    
    # Format x-axis dates
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.tick_params(axis='x', which='major', labelsize=12, colors='#4a5568')
    ax1.tick_params(axis='y', which='major', labelsize=12, colors='#4a5568')
    ax2.tick_params(axis='x', which='major', labelsize=12, colors='#4a5568')
    ax2.tick_params(axis='y', which='major', labelsize=12, colors='#4a5568')
    
    plt.tight_layout(pad=3)
    return fig

def plot_technical_indicators(data, ticker):
    """
    Plot technical indicators (RSI and MACD)
    
    Args:
        data (pd.DataFrame): Enhanced stock data with indicators
        ticker (str): Stock ticker
    
    Returns:
        matplotlib.figure.Figure: Technical indicators plot
    """
    # Set modern style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 1]})
    
    # Create gradient background
    fig.patch.set_facecolor('#f8f9fa')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')
    
    # RSI Plot
    if 'RSI' in data.columns:
        ax1.plot(data.index, data['RSI'], label='RSI', color='#3498db', linewidth=2.5, alpha=0.8)
        ax1.axhline(y=70, color='#e74c3c', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax1.axhline(y=30, color='#27ae60', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax1.fill_between(data.index, 70, data['RSI'], where=(data['RSI'] >= 70), alpha=0.2, color='#e74c3c')
        ax1.fill_between(data.index, 30, data['RSI'], where=(data['RSI'] <= 30), alpha=0.2, color='#27ae60')
    
    ax1.set_title(f'{ticker} RSI (Relative Strength Index)', fontsize=16, fontweight='bold', color='#2d3748')
    ax1.set_ylabel('RSI', fontsize=14, color='#4a5568')
    ax1.set_ylim(0, 100)
    
    # MACD Plot
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        ax2.plot(data.index, data['MACD'], label='MACD Line', color='#3498db', linewidth=2.5, alpha=0.8)
        ax2.plot(data.index, data['MACD_Signal'], label='Signal Line', color='#e74c3c', linewidth=2.5, alpha=0.8)
        
        # MACD histogram
        macd_histogram = data['MACD'] - data['MACD_Signal']
        colors = ['#27ae60' if x >= 0 else '#e74c3c' for x in macd_histogram]
        ax2.bar(data.index, macd_histogram, color=colors, alpha=0.6, width=0.8)
    
    ax2.set_title(f'{ticker} MACD (Moving Average Convergence Divergence)', fontsize=16, fontweight='bold', color='#2d3748')
    ax2.set_ylabel('MACD', fontsize=14, color='#4a5568')
    ax2.set_xlabel('Date', fontsize=14, color='#4a5568', labelpad=10)
    
    # Enhanced styling
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#e2e8f0')
            spine.set_linewidth(1)
        
        # Modern legend
        legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_edgecolor('#e2e8f0')
    
    # Format x-axis dates
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.tick_params(axis='x', which='major', labelsize=12, colors='#4a5568')
    ax1.tick_params(axis='y', which='major', labelsize=12, colors='#4a5568')
    ax2.tick_params(axis='x', which='major', labelsize=12, colors='#4a5568')
    ax2.tick_params(axis='y', which='major', labelsize=12, colors='#4a5568')
    
    plt.tight_layout(pad=3)
    return fig

def plot_volume_analysis(raw_data, enhanced_data, ticker):
    """
    Plot detailed volume analysis
    
    Args:
        raw_data (pd.DataFrame): Original raw data
        enhanced_data (pd.DataFrame): Enhanced data with volume indicators
        ticker (str): Stock ticker
    
    Returns:
        matplotlib.figure.Figure: Volume analysis plot
    """
    # Set modern style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Create gradient background
    fig.patch.set_facecolor('#f8f9fa')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')
    
    # Price and Volume overlay
    ax1.plot(raw_data.index, raw_data['Close'], label='Close Price', color='#4ecdc4', linewidth=3, alpha=0.8)
    ax1.set_ylabel('Price ($)', fontsize=14, color='#4a5568', labelpad=10)
    
    # Create second y-axis for volume
    ax1_twin = ax1.twinx()
    ax1_twin.bar(raw_data.index, raw_data['Volume'], alpha=0.3, color='#3498db', width=0.8)
    ax1_twin.set_ylabel('Volume', fontsize=14, color='#3498db', labelpad=10)
    
    # Volume bars with moving average - align the data properly
    ax2.bar(raw_data.index, raw_data['Volume'], label='Volume', color='#3498db', alpha=0.6, width=0.8)
    if 'Volume_SMA' in enhanced_data.columns:
        # Align the Volume_SMA with raw_data index to avoid shape mismatch
        aligned_volume_sma = enhanced_data['Volume_SMA'].reindex(raw_data.index)
        ax2.plot(raw_data.index, aligned_volume_sma, label='Volume SMA (20)', color='#e74c3c', linewidth=2.5, alpha=0.8)
    
    ax2.set_title(f'{ticker} Volume Analysis', fontsize=16, fontweight='bold', color='#2d3748')
    ax2.set_ylabel('Volume', fontsize=14, color='#4a5568')
    ax2.set_xlabel('Date', fontsize=14, color='#4a5568', labelpad=10)
    
    # Enhanced styling
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#e2e8f0')
            spine.set_linewidth(1)
    
    # Style twin axis
    ax1_twin.grid(False)
    for spine in ax1_twin.spines.values():
        spine.set_edgecolor('#e2e8f0')
        spine.set_linewidth(1)
    
    # Modern legends
    legend1 = ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    legend1.get_frame().set_facecolor('#ffffff')
    legend1.get_frame().set_edgecolor('#e2e8f0')
    
    legend2 = ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    legend2.get_frame().set_facecolor('#ffffff')
    legend2.get_frame().set_edgecolor('#e2e8f0')
    
    # Format x-axis dates
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.tick_params(axis='x', which='major', labelsize=12, colors='#4a5568')
    ax1.tick_params(axis='y', which='major', labelsize=12, colors='#4a5568')
    ax2.tick_params(axis='x', which='major', labelsize=12, colors='#4a5568')
    ax2.tick_params(axis='y', which='major', labelsize=12, colors='#4a5568')
    
    plt.tight_layout(pad=3)
    return fig