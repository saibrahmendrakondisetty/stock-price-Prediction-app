"""
Train Machine Learning Model for Stock Price Prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from utils import (
    download_stock_data, 
    preprocess_data, 
    create_supervised_dataset, 
    split_data, 
    evaluate_model
)

def train_stock_prediction_model(ticker='AAPL', save_model=True):
    """
    Train a Linear Regression model for stock price prediction
    
    Args:
        ticker (str): Stock ticker symbol (default: AAPL)
        save_model (bool): Whether to save the trained model
    
    Returns:
        dict: Training results including model and metrics
    """
    print(f"Starting model training for {ticker}...")
    
    try:
        # Step 1: Download stock data
        print("Downloading stock data...")
        raw_data = download_stock_data(ticker)
        print(f"Downloaded {len(raw_data)} days of data")
        
        # Step 2: Preprocess data
        print("Preprocessing data...")
        processed_data = preprocess_data(raw_data)
        print(f"Processed data shape: {processed_data.shape}")
        
        # Step 3: Create supervised dataset
        print("Creating supervised learning dataset...")
        X, y = create_supervised_dataset(processed_data)
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Step 4: Split data
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Step 5: Feature scaling
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Step 6: Train model
        print("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        print("Model training completed!")
        
        # Step 7: Make predictions
        print("Making predictions...")
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Step 8: Evaluate model
        print("Evaluating model performance...")
        train_metrics = evaluate_model(y_train, y_train_pred)
        test_metrics = evaluate_model(y_test, y_test_pred)
        
        print(f"\nModel Performance:")
        print(f"   Training RMSE: {train_metrics['RMSE']:.4f}")
        print(f"   Training R²: {train_metrics['R2']:.4f}")
        print(f"   Test RMSE: {test_metrics['RMSE']:.4f}")
        print(f"   Test R²: {test_metrics['R2']:.4f}")
        
        # Step 9: Save model and scaler
        if save_model:
            print("Saving model and scaler...")
            try:
                joblib.dump(model, 'saved_model.pkl')
                print("Model saved as 'saved_model.pkl'")
            except:
                print("Warning: Could not save model file")
            
            try:
                joblib.dump(scaler, 'scaler.pkl')
                print("Scaler saved as 'scaler.pkl'")
            except:
                print("Warning: Could not save scaler file")
            
            # Save only the last 100 rows of processed data to save space
            try:
                joblib.dump(processed_data.tail(100), 'processed_data.pkl')
                print("Processed data saved as 'processed_data.pkl'")
            except:
                print("Warning: Could not save processed data file")
        
        # Return results
        results = {
            'model': model,
            'scaler': scaler,
            'processed_data': processed_data,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': {
                'y_train': y_train,
                'y_train_pred': y_train_pred,
                'y_test': y_test,
                'y_test_pred': y_test_pred
            }
        }
        
        print(f"\nModel training completed successfully for {ticker}!")
        return results
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def load_trained_model():
    """
    Load the trained model and associated objects
    
    Returns:
        dict: Dictionary containing model, scaler, and processed data
    """
    try:
        # Check if files exist
        required_files = ['saved_model.pkl', 'scaler.pkl', 'processed_data.pkl']
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file {file} not found. Please train the model first.")
        
        # Load objects
        model = joblib.load('saved_model.pkl')
        scaler = joblib.load('scaler.pkl')
        processed_data = joblib.load('processed_data.pkl')
        
        print("Successfully loaded trained model and associated objects")
        
        return {
            'model': model,
            'scaler': scaler,
            'processed_data': processed_data
        }
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def main():
    """
    Main function to train the model
    """
    print("=" * 60)
    print("STOCK PRICE PREDICTION MODEL TRAINER")
    print("=" * 60)
    
    # Train model for default stock (AAPL)
    results = train_stock_prediction_model(ticker='AAPL', save_model=True)
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model trained and saved successfully!")
    print(f"Test R2 Score: {results['test_metrics']['R2']:.4f}")
    print(f"Test RMSE: ${results['test_metrics']['RMSE']:.2f}")
    print(f"Files saved: saved_model.pkl, scaler.pkl, processed_data.pkl")
    print("\nYou can now run the Streamlit app:")
    print("   streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
