#!/usr/bin/env python3
"""
Test script for Sappo Preprocessing GUI
"""

import os
import sys
import pandas as pd
import numpy as np
from preprocessing import DataPreprocessor

def create_sample_data():
    """Create sample CSV files for testing"""
    os.makedirs("test_data", exist_ok=True)
    
    # Create sample data for different timeframes
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1H')
    
    timeframes = ['hour1', 'hour4', 'day', 'week']
    coins = ['BTC', 'ETH']
    
    for coin in coins:
        for tf in timeframes:
            # Generate sample OHLCV data
            n_points = len(dates)
            base_price = 50000 if coin == 'BTC' else 3000
            
            data = {
                'timestamp': dates,
                'open': base_price + np.random.randn(n_points) * 1000,
                'high': base_price + np.random.randn(n_points) * 1000 + 500,
                'low': base_price + np.random.randn(n_points) * 1000 - 500,
                'close': base_price + np.random.randn(n_points) * 1000,
                'volume': np.random.randint(1000, 10000, n_points)
            }
            
            # Ensure high >= low and close between open/high/low
            df = pd.DataFrame(data)
            df['high'] = np.maximum(df[['open', 'close']].max(axis=1), df['high'])
            df['low'] = np.minimum(df[['open', 'close']].min(axis=1), df['low'])
            
            filename = f"test_data/upbit_{coin}_{tf}.csv"
            df.to_csv(filename, index=False)
            print(f"Created {filename}")

def test_preprocessing():
    """Test the preprocessing pipeline"""
    print("Testing preprocessing pipeline...")
    
    # Load test data
    data_files = {}
    for filename in os.listdir("test_data"):
        if filename.endswith('.csv'):
            filepath = os.path.join("test_data", filename)
            data_files[filename] = pd.read_csv(filepath)
    
    print(f"Loaded {len(data_files)} test files")
    
    # Test preprocessing
    preprocessor = DataPreprocessor(window_size=12)  # Smaller window for testing
    
    try:
        def progress_callback(progress, status=""):
            print(f"Progress: {progress:.1f}% - {status}")
        
        result = preprocessor.preprocess_pipeline(data_files, progress_callback)
        print(f"Success! Output shape: {result.shape}")
        
        # Save test result
        np.save("test_output.npy", result)
        print("Test output saved to test_output.npy")
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    
    return True

def main():
    print("Sappo Preprocessing Test")
    print("=" * 30)
    
    # Create sample data
    print("Creating sample data...")
    create_sample_data()
    
    # Test preprocessing
    if test_preprocessing():
        print("\n✅ All tests passed!")
        print("\nTo run the GUI application, execute:")
        print("python main.py")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()