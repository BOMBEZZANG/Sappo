import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from datetime import datetime

class RawDataPreprocessor:
    """
    Pure Raw Data Preprocessor for AI Self-Learning
    
    This approach provides minimal preprocessing, allowing the AI to learn
    patterns and relationships directly from raw OHLCV data with only
    time information added.
    """
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        
    def pure_raw_approach(self, data_files: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        진정한 Raw Data 접근법
        AI가 스스로 발견해야 할 것들:
        - 가격 변화율의 중요성
        - 변동성 패턴
        - 코인 간 상관관계
        - 시간적 주기성
        """
        # Group files by coin and timeframe
        coins = {}
        for filename, df in data_files.items():
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 3:
                coin = parts[1]
                timeframe = parts[2]
                
                if coin not in coins:
                    coins[coin] = {}
                coins[coin][timeframe] = df
        
        print(f"Found {len(coins)} coins: {list(coins.keys())}")
        
        # Create feature list
        features = []
        timeframes = ['hour1', 'hour4', 'day', 'week']
        
        # Process each coin and collect all unified data
        all_coin_data = []
        
        for coin in coins.keys():
            for timeframe in timeframes:
                if timeframe in coins[coin]:
                    # 원시 OHLCV 그대로 추가
                    features.extend([
                        f"{coin}_open_{timeframe}",
                        f"{coin}_high_{timeframe}", 
                        f"{coin}_low_{timeframe}",
                        f"{coin}_close_{timeframe}",
                        f"{coin}_volume_{timeframe}"
                    ])
        
        # 시간 정보만 추가 (절대 필요)
        features.extend([
            "hour_of_day",    # 0-23
            "day_of_week",    # 0-6  
            "day_of_month"    # 1-31
        ])
        
        # Unify all coins and timeframes to 1-hour base
        unified_data = self._unify_raw_timeframes(data_files)
        
        # Add time features
        unified_data = self._add_time_features(unified_data)
        
        print(f"Raw features created: {unified_data.shape[1]} total features")
        print("AI will discover patterns in:")
        print("- Price movements and volatility")
        print("- Cross-coin correlations")
        print("- Time-based patterns")
        print("- Volume-price relationships")
        
        return unified_data
    
    def _unify_raw_timeframes(self, data_files: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Unify all coins and timeframes to 1-hour base with RAW OHLCV values
        """
        # Group files by coin
        coins = {}
        for filename, df in data_files.items():
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 3:
                coin = parts[1]
                timeframe = parts[2]
                
                if coin not in coins:
                    coins[coin] = {}
                coins[coin][timeframe] = df
        
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
                    # Resample to 1-hour and forward fill (keeping RAW values)
                    hourly_data = df.resample('1h').ffill()
                    
                    # Add coin and timeframe prefix to column names
                    hourly_data.columns = [f"{coin}_{col}_{tf}" for col in hourly_data.columns]
                    coin_data[tf] = hourly_data
            
            # Combine all timeframes for this coin
            if coin_data:
                combined_coin = pd.concat(coin_data.values(), axis=1)
                all_coin_data.append(combined_coin)
                coin_names.append(coin)
                print(f"Processed {coin}: {combined_coin.shape[1]} raw features")
        
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
            
            print(f"Final raw unified dataset: {final_data.shape}")
            print(f"Raw features per coin: ~{final_data.shape[1] // len(coin_names)}")
            print(f"Total raw features: {final_data.shape[1]}")
            
            return final_data
        else:
            raise ValueError("No valid data found for unification")
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add essential time features that AI needs to learn temporal patterns
        """
        # Create copy to avoid modifying original
        result = data.copy()
        
        # Extract time components from index
        result['hour_of_day'] = data.index.hour  # 0-23
        result['day_of_week'] = data.index.dayofweek  # 0-6 (Monday=0)
        result['day_of_month'] = data.index.day  # 1-31
        
        print("Added time features: hour_of_day, day_of_week, day_of_month")
        
        return result
    
    def create_raw_sliding_windows(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform raw time-series data into 3D array of (samples, window_size, features)
        No normalization - pure raw values for AI to learn from
        """
        if len(data) < self.window_size:
            raise ValueError(f"Data length ({len(data)}) is less than window size ({self.window_size})")
        
        n_samples = len(data) - self.window_size + 1
        n_features = len(data.columns)
        
        # Create 3D array with raw values
        windows = np.zeros((n_samples, self.window_size, n_features))
        
        for i in range(n_samples):
            windows[i] = data.iloc[i:i + self.window_size].values
        
        print(f"Created {n_samples} raw windows, each with {self.window_size} timesteps and {n_features} features")
        
        return windows
    
    def validate_raw_data_files(self, data_files: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that loaded data files have the expected format for raw processing
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
    
    def raw_preprocess_pipeline(self, data_files: Dict[str, pd.DataFrame], progress_callback=None) -> np.ndarray:
        """
        Complete RAW preprocessing pipeline - minimal processing for AI self-learning
        """
        try:
            # Step 1: Validate data
            if not self.validate_raw_data_files(data_files):
                raise ValueError("Raw data validation failed")
            
            if progress_callback:
                progress_callback(10, "Raw data validated")
            
            # Step 2: Create pure raw unified dataset
            raw_unified_data = self.pure_raw_approach(data_files)
            if progress_callback:
                progress_callback(60, "Raw data unified with time features")
            
            # Step 3: Create sliding windows (no normalization!)
            raw_windowed_data = self.create_raw_sliding_windows(raw_unified_data)
            if progress_callback:
                progress_callback(90, "Raw windows created")
            
            if progress_callback:
                progress_callback(100, "Raw preprocessing complete - AI ready for self-learning")
            
            print("\n" + "="*60)
            print("RAW PREPROCESSING COMPLETE")
            print("="*60)
            print(f"✓ AI will learn from {raw_windowed_data.shape[0]} raw samples")
            print(f"✓ Each sample: {raw_windowed_data.shape[1]} timesteps × {raw_windowed_data.shape[2]} features")
            print(f"✓ No human-engineered features - pure market data + time")
            print(f"✓ AI must discover: price patterns, volatility, correlations, cycles")
            print("="*60)
            
            return raw_windowed_data
            
        except Exception as e:
            raise RuntimeError(f"Raw preprocessing pipeline failed: {str(e)}")
    
    def get_feature_info(self, data_files: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Get information about features that will be created in raw approach
        """
        coins = set()
        timeframes = set()
        
        for filename in data_files.keys():
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 3:
                coins.add(parts[1])
                timeframes.add(parts[2])
        
        raw_features = []
        for coin in sorted(coins):
            for tf in sorted(timeframes):
                raw_features.extend([
                    f"{coin}_open_{tf}",
                    f"{coin}_high_{tf}", 
                    f"{coin}_low_{tf}",
                    f"{coin}_close_{tf}",
                    f"{coin}_volume_{tf}"
                ])
        
        time_features = ["hour_of_day", "day_of_week", "day_of_month"]
        
        return {
            "raw_ohlcv_features": raw_features,
            "time_features": time_features,
            "total_features": len(raw_features) + len(time_features),
            "coins": list(sorted(coins)),
            "timeframes": list(sorted(timeframes))
        }