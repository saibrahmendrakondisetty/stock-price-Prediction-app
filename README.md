# 📈 Stock Price Prediction Web App using Machine Learning

A comprehensive machine learning project that predicts the next-day closing price of stocks using historical data. Built with Python, Scikit-learn, and Streamlit.

## 🎯 Project Goal

Predict the next-day closing price of a selected stock using historical data with a simple yet effective Linear Regression model.

## 🛠️ Tech Stack

- **Python 3** - Core programming language
- **yfinance** - Stock data collection from Yahoo Finance
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **scikit-learn** - Machine Learning (Linear Regression)
- **streamlit** - Interactive web application
- **joblib** - Model serialization

## 📁 Project Structure

```
stock_prediction_project/
│
├── app.py                 # Main Streamlit web application
├── train_model.py         # Model training script
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── saved_model.pkl        # Trained model (generated after training)
├── scaler.pkl            # Feature scaler (generated after training)
├── processed_data.pkl    # Processed data (generated after training)
└── README.md             # This file
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone/Download the Project

```bash
# If using git
git clone <repository-url>
cd stock_prediction_project

# Or download and extract the project folder
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

```bash
python train_model.py
```

This will:
- Download historical stock data for AAPL (default)
- Preprocess the data and create features
- Train a Linear Regression model
- Save the trained model and related files
- Display training metrics

### Step 4: Run the Web Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🤖 Machine Learning Workflow

### 1. Data Collection
- Uses `yfinance` to download historical stock data
- Default stock: AAPL (Apple Inc.)
- Date range: 2018-01-01 to current date
- Fetches Open, High, Low, Close, Volume data

### 2. Data Preprocessing
- **Feature Selection**: Uses 'Close' price as primary feature
- **Missing Values**: Handles and removes missing data
- **Feature Engineering**: Adds 20-day Moving Average as bonus feature
- **Supervised Learning**: Creates dataset where:
  - **X** = [Today's Close Price, 20-Day MA]
  - **y** = Next Day's Close Price (shifted by -1)

### 3. Model Training
- **Algorithm**: Linear Regression
- **Feature Scaling**: StandardScaler for normalization
- **Data Split**: 80% training, 20% testing (no shuffling for time series)
- **Model Persistence**: Saves trained model using joblib

### 4. Model Evaluation
- **RMSE** (Root Mean Square Error): Measures prediction accuracy
- **R² Score**: Coefficient of determination (explained variance)

### 5. Prediction & Visualization
- Real-time stock data fetching
- Next-day price prediction
- Interactive plots using matplotlib
- Performance metrics display

## 🌐 Web Application Features

### Main Features
- **Interactive Sidebar**: Input stock ticker and select date range
- **Real-time Data**: Downloads fresh stock data on demand
- **Data Visualization**: 
  - Historical price charts with moving averages
  - Actual vs Predicted price comparisons
- **Model Performance**: Displays RMSE and R² scores
- **Next-Day Prediction**: Highlights predicted next-day closing price
- **Recent Predictions**: Shows last 5 predictions with actual comparisons

### Bonus Features
- **20-Day Moving Average**: Technical indicator for trend analysis
- **Professional UI**: Clean, responsive design with custom CSS
- **Error Handling**: Graceful handling of invalid tickers and data issues
- **Educational Disclaimer**: Clear warnings about educational use only

## 📊 Model Performance

Typical performance metrics (trained on AAPL data):
- **Training R²**: ~0.95-0.98
- **Test R²**: ~0.90-0.95
- **RMSE**: Varies by stock price range

*Note: Performance varies by stock volatility and market conditions.*

## 🎮 How to Use the Web App

1. **Launch the App**: Run `streamlit run app.py`
2. **Enter Stock Ticker**: Use the sidebar to input any valid stock symbol (e.g., GOOGL, MSFT, TSLA)
3. **Select Date Range**: Choose custom start and end dates for analysis
4. **View Results**: 
   - Raw stock data table
   - Historical price charts
   - Model performance metrics
   - Actual vs Predicted price plots
   - Next-day predicted price (highlighted)
5. **Compare Predictions**: Check the "Recent Predictions" section for accuracy

## ⚠️ Important Disclaimer

**This application is for educational and hackathon purposes only.**

- ❌ **Do NOT use for actual trading or investment decisions**
- ❌ **No financial guarantees are provided**
- ❌ **Past performance does not guarantee future results**
- ✅ **Use only for learning ML concepts and data analysis**

## 🔧 Customization & Extensions

### Adding New Features
- **Technical Indicators**: Add RSI, MACD, Bollinger Bands
- **Multiple Models**: Implement LSTM, Random Forest, XGBoost
- **Sentiment Analysis**: Incorporate news sentiment data
- **Portfolio Management**: Track multiple stocks simultaneously

### Model Improvements
- **Feature Engineering**: Add more technical indicators
- **Hyperparameter Tuning**: Optimize model parameters
- **Ensemble Methods**: Combine multiple models
- **Deep Learning**: Implement LSTM/GRU networks

### Data Sources
- **Alternative APIs**: Alpha Vantage, Quandl, Polygon.io
- **Fundamental Data**: Add financial ratios and metrics
- **News Data**: Incorporate sentiment analysis
- **Economic Indicators**: Add macroeconomic features

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found Error**
   ```bash
   # Solution: Train the model first
   python train_model.py
   ```

2. **Invalid Stock Ticker**
   - Verify ticker symbol is valid
   - Check if the stock is actively traded
   - Try with major stocks (AAPL, GOOGL, MSFT)

3. **Data Download Issues**
   - Check internet connection
   - Verify yfinance is working: `yfinance.download('AAPL')`
   - Try different date ranges

4. **Streamlit Not Found**
   ```bash
   # Solution: Install streamlit
   pip install streamlit
   ```

5. **Port Already in Use**
   ```bash
   # Solution: Use different port
   streamlit run app.py --server.port 8502
   ```

## 📚 Learning Resources

- **Machine Learning**: [Scikit-learn Documentation](https://scikit-learn.org/)
- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io/)
- **Financial Data**: [yfinance Documentation](https://pypi.org/project/yfinance/)
- **Time Series Analysis**: Check online courses on financial modeling

## 🤝 Contributing

Feel free to:
- Report issues and bugs
- Suggest improvements
- Add new features
- Improve documentation
- Share your enhancements

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing stock data via yfinance
- **Scikit-learn** team for excellent ML library
- **Streamlit** team for the amazing web app framework
- Open-source community for various tools and libraries

---

**🚀 Happy Learning and Happy Coding!**

*Remember: This is an educational project. Always consult financial professionals for investment advice.*
