import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import os

class DataPreprocessor:
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.scaler = StandardScaler()
        
    def unify_timeframes(self, data_files: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Unify all timeframes to 1-hour base by resampling and forward-filling.
        Expects files with format: upbit_{coin}_{timeframe}.csv
        """
        unified_data = {}
        
        # Group files by coin
        coins = {}
        for filename, df in data_files.items():
            # Parse filename: upbit_COIN_timeframe.csv
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 3:
                coin = parts[1]
                timeframe = parts[2]
                
                if coin not in coins:
                    coins[coin] = {}
                coins[coin][timeframe] = df
        
        # Process each coin
        for coin, timeframes in coins.items():
            coin_data = {}
            
            for tf, df in timeframes.items():
                # Ensure datetime index
                df_copy = df.copy()
                if 'timestamp' in df_copy.columns:
                    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
                    df_copy.set_index('timestamp', inplace=True)
                elif 'date' in df_copy.columns:
                    df_copy['date'] = pd.to_datetime(df_copy['date'])
                    df_copy.set_index('date', inplace=True)
                else:
                    raise ValueError(f"No timestamp or date column found in {filename}")
                
                df = df_copy
                
                # Map timeframe names to pandas frequency strings
                freq_map = {
                    'hour1': '1h',
                    'hour4': '4h', 
                    'day': '1D',
                    'week': '1W'
                }
                
                if tf in freq_map:
                    # Resample to 1-hour and forward fill
                    hourly_data = df.resample('1h').ffill()
                    
                    # Add timeframe suffix to column names
                    hourly_data.columns = [f"{col}_{tf}" for col in hourly_data.columns]
                    coin_data[tf] = hourly_data
            
            # Combine all timeframes for this coin
            if coin_data:
                combined = pd.concat(coin_data.values(), axis=1)
                combined = combined.dropna()  # Remove rows with missing data
                unified_data[coin] = combined
        
        # Combine all coins
        if unified_data:
            final_data = pd.concat(unified_data.values(), axis=1, keys=unified_data.keys())
            return final_data
        else:
            raise ValueError("No valid data found for unification")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert absolute prices to percentage changes and create additional features.
        """
        feature_data = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            if isinstance(col, tuple):
                # Multi-level columns (coin, feature_timeframe)
                coin, feature_tf = col
                col_name = f"{coin}_{feature_tf}"
            else:
                col_name = str(col)
            
            # Calculate percentage change
            if 'close' in col_name.lower() or 'open' in col_name.lower() or 'high' in col_name.lower() or 'low' in col_name.lower():
                pct_change = data[col].pct_change()
                feature_data[f"{col_name}_pct"] = pct_change
            
            # Volume features (normalize by rolling mean)
            elif 'volume' in col_name.lower():
                volume_norm = data[col] / data[col].rolling(window=24).mean()
                feature_data[f"{col_name}_norm"] = volume_norm
            
            # Keep original data as well for some features
            else:
                feature_data[col_name] = data[col]
        
        # Remove infinite and NaN values
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        feature_data = feature_data.dropna()
        
        return feature_data
    
    def create_sliding_windows(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform time-series data into 3D array of (samples, window_size, features).
        """
        if len(data) < self.window_size:
            raise ValueError(f"Data length ({len(data)}) is less than window size ({self.window_size})")
        
        n_samples = len(data) - self.window_size + 1
        n_features = len(data.columns)
        
        # Create 3D array
        windows = np.zeros((n_samples, self.window_size, n_features))
        
        for i in range(n_samples):
            windows[i] = data.iloc[i:i + self.window_size].values
        
        return windows
    
    def normalize_windows(self, windowed_data: np.ndarray, progress_callback=None) -> np.ndarray:
        """
        Apply Z-score normalization to each window independently.
        """
        n_samples, window_size, n_features = windowed_data.shape
        normalized_data = np.zeros_like(windowed_data)
        
        for i in range(n_samples):
            # Flatten the window for scaling
            window = windowed_data[i].reshape(-1, 1)
            
            # Fit and transform
            scaler = StandardScaler()
            normalized_window = scaler.fit_transform(window)
            
            # Reshape back to original window shape
            normalized_data[i] = normalized_window.reshape(window_size, n_features)
            
            # Update progress if callback provided
            if progress_callback and i % 100 == 0:
                progress = (i / n_samples) * 100
                progress_callback(progress)
        
        return normalized_data
    
    def get_timeframe_mapping(self, filename: str) -> str:
        """
        Extract timeframe from filename.
        """
        if 'hour1' in filename:
            return '1h'
        elif 'hour4' in filename:
            return '4h'
        elif 'day' in filename:
            return '1d'
        elif 'week' in filename:
            return '1w'
        else:
            return 'unknown'
    
    def validate_data_files(self, data_files: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that loaded data files have the expected format.
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for filename, df in data_files.items():
            # Check if required columns exist (case insensitive)
            df_columns_lower = [col.lower() for col in df.columns]
            missing_columns = [col for col in required_columns if col not in df_columns_lower]
            
            if missing_columns:
                print(f"Warning: {filename} missing columns: {missing_columns}")
                return False
        
        return True
    
    def preprocess_pipeline(self, data_files: Dict[str, pd.DataFrame], progress_callback=None) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        """
        try:
            # Step 1: Validate data
            if not self.validate_data_files(data_files):
                raise ValueError("Data validation failed")
            
            if progress_callback:
                progress_callback(10, "Data validated")
            
            # Step 2: Unify timeframes
            unified_data = self.unify_timeframes(data_files)
            if progress_callback:
                progress_callback(30, "Timeframes unified")
            
            # Step 3: Create features
            feature_data = self.create_features(unified_data)
            if progress_callback:
                progress_callback(50, "Features created")
            
            # Step 4: Create sliding windows
            windowed_data = self.create_sliding_windows(feature_data)
            if progress_callback:
                progress_callback(70, "Windows created")
            
            # Step 5: Normalize windows
            def norm_progress(norm_pct):
                total_progress = 70 + (norm_pct * 0.25)  # 70% to 95%
                if progress_callback:
                    progress_callback(total_progress, f"Normalizing windows... {norm_pct:.1f}%")
            
            normalized_data = self.normalize_windows(windowed_data, norm_progress)
            if progress_callback:
                progress_callback(100, "Preprocessing complete")
            
            return normalized_data
            
        except Exception as e:
            raise RuntimeError(f"Preprocessing pipeline failed: {str(e)}")