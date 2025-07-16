import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning
    
    Supports both standard and raw preprocessing modes:
    - Standard: Engineered features with percentage changes and indicators
    - Raw: Pure OHLCV data + time features for AI self-learning
    
    State: (window_size, n_features) tensor with lookback window
    - market features from preprocessing (varies by mode)
    - 2 portfolio features (position, unrealized PnL)
    
    Actions: 
    - 0: Hold
    - 1: Buy 
    - 2: Sell
    """
    
    def __init__(self, 
                 data_array: np.ndarray,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001,
                 reward_weights: Dict[str, float] = None,
                 price_column_name: str = None):
        """
        Initialize Trading Environment
        
        Args:
            data_array: Preprocessed data (samples, window_size, features)
            initial_balance: Starting portfolio balance
            transaction_cost: Cost per trade (as fraction)
            reward_weights: Weights for reward components
            price_column_name: Name/pattern to identify price column (auto-detect if None)
        """
        super().__init__()
        
        # Data setup
        self.data = data_array
        self.n_samples, self.window_size, self.n_features = data_array.shape
        self.price_column_name = price_column_name
        
        # Auto-detect price column for different preprocessing modes
        self.price_column_idx = self._detect_price_column()
        
        # Portfolio setup
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Reward weights (default values)
        if reward_weights is None:
            reward_weights = {
                'profit': 1.0,
                'sharpe': 0.5, 
                'cost': 1.0,
                'mdd': 0.5
            }
        self.reward_weights = reward_weights
        
        # Gym spaces
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.n_features + 2),  # +2 for portfolio features
            dtype=np.float32
        )
        
        # Environment state
        self.last_action = 0
        self.reset()
    
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Portfolio state
        self.balance = self.initial_balance
        self.position = 0.0  # -1: Short, 0: Neutral, 1: Long
        self.holdings_value = 0.0
        self.entry_price = 0.0
        
        # Episode tracking
        self.current_step = self.window_size - 1  # Start after window
        self.done = False
        
        # Performance metrics
        self.portfolio_values = [self.initial_balance]
        self.trade_count = 0
        self.total_cost = 0.0
        self.max_portfolio_value = self.initial_balance
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one trading step"""
        
        # Get current price (using first coin's close price as reference)
        current_price = self._get_current_price()
        
        # Store position before action for trade tracking
        position_before = self.position
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.n_samples - 1
        truncated = False
        
        # Update portfolio value
        total_value = self._calculate_portfolio_value(current_price)
        self.portfolio_values.append(total_value)
        self.max_portfolio_value = max(self.max_portfolio_value, total_value)
        
        # Get next observation
        observation = self._get_observation()
        info = self._get_info()
        
        # Add trade tracking information
        info['position_before'] = position_before
        info['trade_executed'] = position_before != self.position
        info['trade_cost'] = self.total_cost - info.get('previous_total_cost', 0)
        info['previous_total_cost'] = self.total_cost
        
        # Calculate unrealized PnL
        if self.position != 0 and self.entry_price > 0:
            info['unrealized_pnl'] = (current_price - self.entry_price) / self.entry_price * self.position
        else:
            info['unrealized_pnl'] = 0.0
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
            """Execute trading action and return reward"""
            
            # Store previous portfolio value for reward calculation
            prev_portfolio_value = self._calculate_portfolio_value(current_price)

            # Store the current action to be used in the reward calculation
            self.last_action = action
            
            # Execute the trading action (Buy, Sell, or Hold)
            if action == 1 and self.position <= 0:  # Buy
                self._execute_buy(current_price)
            elif action == 2 and self.position >= 0:  # Sell
                self._execute_sell(current_price)
            # if action is 0 (Hold), no execution is needed
            
            # Calculate new portfolio value after the action
            new_portfolio_value = self._calculate_portfolio_value(current_price)
            
            # Calculate reward based on the outcome of the action
            reward = self._calculate_reward(prev_portfolio_value, new_portfolio_value)
            
            return reward
    
    def _execute_buy(self, price: float):
        """Execute buy order"""
        if self.balance > 0:
            # Calculate position size (use available balance)
            position_size = self.balance / price
            cost = position_size * price * self.transaction_cost
            
            # Update portfolio
            self.holdings_value = position_size * price
            self.balance -= (self.holdings_value + cost)
            self.position = 1.0
            self.entry_price = price
            
            # Track costs and trades
            self.total_cost += cost
            self.trade_count += 1
    
    def _execute_sell(self, price: float):
        """Execute sell order"""
        if self.holdings_value > 0:
            # Calculate proceeds from sale
            position_size = self.holdings_value / self.entry_price
            proceeds = position_size * price
            cost = proceeds * self.transaction_cost
            
            # Update portfolio
            self.balance += (proceeds - cost)
            self.holdings_value = 0.0
            self.position = 0.0
            self.entry_price = 0.0
            
            # Track costs and trades
            self.total_cost += cost
            self.trade_count += 1
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        if self.holdings_value > 0 and self.entry_price > 0:
            # Update holdings value based on current price
            position_size = self.holdings_value / self.entry_price
            current_holdings_value = position_size * current_price
            return self.balance + current_holdings_value
        else:
            return self.balance
    
    def _calculate_reward(self, prev_value: float, new_value: float) -> float:
        """
        Calculate multi-component reward:
        R_t = w1*R_profit + w2*R_sharpe - w3*R_cost - w4*R_mdd + w5*R_trade
        """
        
        # ... (profit_return, sharpe_component, cost_penalty, current_drawdown 계산 로직은 기존과 동일)
        # R_profit: 포트폴리오 수익률
        if prev_value > 0:
            profit_return = (new_value - prev_value) / prev_value
        else:
            profit_return = 0.0
        
        # R_sharpe: 간소화된 샤프 지수 요소
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_component = np.mean(returns) / np.std(returns)
            else:
                sharpe_component = 0.0
        else:
            sharpe_component = 0.0
        
        # R_cost: 거래 비용 패널티
        cost_penalty = self.total_cost / self.initial_balance
        
        # R_mdd: 최대 낙폭 패널티
        if self.max_portfolio_value > 0:
            current_drawdown = (self.max_portfolio_value - new_value) / self.max_portfolio_value
        else:
            current_drawdown = 0.0

        # ⭐ 2. 거래 장려 보너스 계산
        trade_incentive_reward = 0.0
        if self.last_action == 1 or self.last_action == 2:  # 행동이 매수 또는 매도일 경우
            # self.reward_weights 딕셔너리에서 'trade_incentive' 값을 가져옵니다.
            # 만약 값이 없으면 0을 기본값으로 사용합니다.
            trade_incentive_reward = self.reward_weights.get('trade_incentive', 0.0)

        # ⭐ 3. 최종 보상 조합 업데이트
        reward = (
            self.reward_weights.get('profit', 1.0) * profit_return +
            self.reward_weights.get('sharpe', 0.0) * sharpe_component -
            self.reward_weights.get('cost', 1.0) * cost_penalty -
            self.reward_weights.get('mdd', 0.0) * current_drawdown +
            trade_incentive_reward  # 거래 장려 보상을 최종 보상에 더합니다.
        )
        
        return reward
    
    def _detect_price_column(self) -> int:
        """
        Auto-detect price column index based on preprocessing mode
        
        For raw mode: looks for time features to determine where price data ends
        For standard mode: uses the last feature as reference price
        """
        # For raw mode, time features are at the end (hour_of_day, day_of_week, day_of_month)
        # So we want the close price just before time features
        # For standard mode, the last feature should be raw_close_price
        
        # Simple heuristic: if last 3 features seem like time data (small integer values)
        # then we're in raw mode, otherwise standard mode
        sample_data = self.data[0, -1, :]  # First sample, last timestep
        
        # Check if last 3 features look like time data
        last_three = sample_data[-3:]
        if (all(0 <= val <= 31 for val in last_three) and  # reasonable time ranges
            last_three[0] <= 23 and  # hour_of_day
            last_three[1] <= 6):     # day_of_week
            # Raw mode: find first close column (should be index 3, 8, 13, etc. pattern)
            # For simplicity, use the 4th column (first close) as reference
            return 3  # First close column in raw OHLCV data
        else:
            # Standard mode: use last column (raw_close_price)
            return -1
    
    def _get_current_price(self) -> float:
        """Get current reference price using detected price column"""
        step_data = self.data[self.current_step, -1, :]  # Last time step of current window
        return float(step_data[self.price_column_idx])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        if self.current_step >= self.data.shape[0]:
            # Return zeros if we're past the data
            return np.zeros((self.window_size, self.n_features + 2), dtype=np.float32)
        
        # Get market data window
        market_data = self.data[self.current_step]  # Shape: (window_size, n_features)
        
        # Calculate unrealized PnL
        current_price = self._get_current_price()
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.position
        
        # Create portfolio features
        portfolio_features = np.full((self.window_size, 2), 
                                   [self.position, unrealized_pnl], 
                                   dtype=np.float32)
        
        # Combine market and portfolio data
        observation = np.concatenate([market_data, portfolio_features], axis=1)
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        current_price = self._get_current_price()
        portfolio_value = self._calculate_portfolio_value(current_price)
        
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.position
        
        return {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'holdings_value': self.holdings_value,
            'trade_count': self.trade_count,
            'total_cost': self.total_cost,
            'current_step': self.current_step,
            'current_price': current_price,
            'unrealized_pnl': unrealized_pnl,
            'entry_price': self.entry_price
        }
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Calculate final portfolio statistics"""
        if len(self.portfolio_values) < 2:
            return {}
        
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic statistics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Sharpe ratio (assuming 252 trading days per year)
        if np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values[-1],
            'trade_count': self.trade_count,
            'total_cost': self.total_cost
        }