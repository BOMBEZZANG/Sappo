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
        Unify all coins and timeframes to 1-hour base, creating a single unified dataset.
        Expects files with format: upbit_{coin}_{timeframe}.csv
        """
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
        
        print(f"Found {len(coins)} coins: {list(coins.keys())}")
        
        # Process each coin and collect all unified data
        all_coin_data = []
        coin_names = []
        
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
                    
                    # Add coin and timeframe prefix to column names
                    hourly_data.columns = [f"{coin}_{col}_{tf}" for col in hourly_data.columns]
                    coin_data[tf] = hourly_data
            
            # Combine all timeframes for this coin
            if coin_data:
                combined_coin = pd.concat(coin_data.values(), axis=1)
                all_coin_data.append(combined_coin)
                coin_names.append(coin)
                print(f"Processed {coin}: {combined_coin.shape[1]} features")
        
        # Combine all coins into single dataset
        if all_coin_data:
            # Find common time range across all coins
            start_date = max([df.index.min() for df in all_coin_data])
            end_date = min([df.index.max() for df in all_coin_data])
            
            print(f"Common time range: {start_date} to {end_date}")
            
            # Align all coins to common time range
            aligned_data = []
            for i, df in enumerate(all_coin_data):
                aligned = df.loc[start_date:end_date]
                aligned_data.append(aligned)
                print(f"{coin_names[i]} aligned: {aligned.shape}")
            
            # Concatenate all coins horizontally
            final_data = pd.concat(aligned_data, axis=1)
            final_data = final_data.dropna()  # Remove any remaining NaN rows
            
            print(f"Final unified dataset: {final_data.shape}")
            print(f"Features per coin: ~{final_data.shape[1] // len(coin_names)}")
            print(f"Total features: {final_data.shape[1]}")
            
            return final_data
        else:
            raise ValueError("No valid data found for unification")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert absolute prices to percentage changes and create additional features.
        Enhanced for multi-coin unified dataset.
        """
        feature_data = pd.DataFrame(index=data.index)
        
        print(f"Creating features from {len(data.columns)} raw columns...")
        
        for col in data.columns:
            col_name = str(col)
            
            # Price features: convert to percentage changes
            if any(price_type in col_name.lower() for price_type in ['close', 'open', 'high', 'low']):
                pct_change = data[col].pct_change()
                feature_data[f"{col_name}_pct"] = pct_change
                
                # Add volatility (rolling std of returns)
                if 'close' in col_name.lower():
                    volatility = pct_change.rolling(window=24).std()
                    feature_data[f"{col_name}_volatility"] = volatility
            
            # Volume features: normalize and add momentum
            elif 'volume' in col_name.lower():
                # Volume normalized by 24h rolling average
                volume_norm = data[col] / data[col].rolling(window=24).mean()
                feature_data[f"{col_name}_norm"] = volume_norm
                
                # Volume momentum (current vs previous)
                volume_momentum = data[col].pct_change()
                feature_data[f"{col_name}_momentum"] = volume_momentum
            
            # Value features (if present)
            elif 'value' in col_name.lower():
                value_norm = data[col] / data[col].rolling(window=24).mean()
                feature_data[f"{col_name}_norm"] = value_norm
        
        print(f"Generated {len(feature_data.columns)} features")
        
        # Add cross-coin correlation features for major pairs
        feature_data = self._add_correlation_features(feature_data, data)
        
        # Remove infinite and NaN values
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        initial_rows = len(feature_data)
        feature_data = feature_data.dropna()
        final_rows = len(feature_data)
        
        print(f"Removed {initial_rows - final_rows} rows with NaN values")
        print(f"Final feature dataset: {feature_data.shape}")
        
        return feature_data
    
    def _add_correlation_features(self, feature_data: pd.DataFrame, raw_data: pd.DataFrame):
        """
        Add correlation features between major coin pairs.
        """
        # Find close prices for correlation
        close_cols = [col for col in raw_data.columns if 'close_hour1' in str(col)]
        
        if len(close_cols) >= 2:
            print(f"Adding correlation features for {len(close_cols)} coins...")
            
            # Collect all correlation features in a dictionary first
            correlation_features = {}
            
            # Calculate rolling correlations between coins
            for i, col1 in enumerate(close_cols[:5]):  # Limit to first 5 coins for performance
                for col2 in close_cols[i+1:6]:  # Avoid duplicate pairs
                    if col1 != col2:
                        # 24h rolling correlation of returns
                        returns1 = raw_data[col1].pct_change()
                        returns2 = raw_data[col2].pct_change()
                        correlation = returns1.rolling(window=24).corr(returns2)
                        
                        coin1 = str(col1).split('_')[0]
                        coin2 = str(col2).split('_')[0]
                        correlation_features[f"corr_{coin1}_{coin2}_24h"] = correlation
            
            # Add all correlation features at once using pd.concat
            if correlation_features:
                corr_df = pd.DataFrame(correlation_features, index=feature_data.index)
                # Use pd.concat instead of individual assignments to avoid fragmentation
                feature_data = pd.concat([feature_data, corr_df], axis=1)
                print(f"Added {len(correlation_features)} correlation features")
        
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