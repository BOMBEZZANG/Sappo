import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning
    
    State: (24, 529) tensor with 24-hour lookback window
    - 527 market features from preprocessing
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
                 reward_weights: Dict[str, float] = None):
        """
        Initialize Trading Environment
        
        Args:
            data_array: Preprocessed data (samples, window_size, features)
            initial_balance: Starting portfolio balance
            transaction_cost: Cost per trade (as fraction)
            reward_weights: Weights for reward components
        """
        super().__init__()
        
        # Data setup
        self.data = data_array
        self.n_samples, self.window_size, self.n_features = data_array.shape
        
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
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action and return reward"""
        
        # Store previous portfolio value for reward calculation
        prev_portfolio_value = self._calculate_portfolio_value(current_price)
        
        # Execute action
        if action == 1 and self.position <= 0:  # Buy
            self._execute_buy(current_price)
        elif action == 2 and self.position >= 0:  # Sell
            self._execute_sell(current_price)
        # action == 0 (Hold) requires no execution
        
        # Calculate new portfolio value
        new_portfolio_value = self._calculate_portfolio_value(current_price)
        
        # Calculate reward
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
        R_t = w1*R_profit + w2*R_sharpe - w3*R_cost - w4*R_mdd
        """
        
        # R_profit: Portfolio return
        if prev_value > 0:
            profit_return = (new_value - prev_value) / prev_value
        else:
            profit_return = 0.0
        
        # R_sharpe: Simplified Sharpe ratio component
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_component = np.mean(returns) / np.std(returns)
            else:
                sharpe_component = 0.0
        else:
            sharpe_component = 0.0
        
        # R_cost: Transaction cost penalty
        cost_penalty = self.total_cost / self.initial_balance
        
        # R_mdd: Maximum drawdown penalty
        if self.max_portfolio_value > 0:
            current_drawdown = (self.max_portfolio_value - new_value) / self.max_portfolio_value
        else:
            current_drawdown = 0.0
        
        # Combine rewards
        reward = (
            self.reward_weights['profit'] * profit_return +
            self.reward_weights['sharpe'] * sharpe_component -
            self.reward_weights['cost'] * cost_penalty -
            self.reward_weights['mdd'] * current_drawdown
        )
        
        return reward
    
    def _get_current_price(self) -> float:
        """Get current reference price (first coin's close price)"""
        # Find first close price column
        step_data = self.data[self.current_step, -1, :]  # Last time step of current window
        
        # Use first available close price as reference
        # This is a simplified approach - in practice you might want to use a specific coin
        close_price_idx = -1 # <- 수정된 코드 (마지막 인덱스)
        return float(step_data[close_price_idx])
    
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
        
        return {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'holdings_value': self.holdings_value,
            'trade_count': self.trade_count,
            'total_cost': self.total_cost,
            'current_step': self.current_step,
            'current_price': current_price
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