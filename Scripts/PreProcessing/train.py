import numpy as np
import os
from datetime import datetime
from typing import Dict, Tuple, Callable
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import json

from TradingEnv import TradingEnv
from agent import create_trading_agent

class ValidationCallback(BaseCallback):
    """
    Custom callback for validation during training
    """
    
    def __init__(self, 
                 validation_env,
                 eval_freq: int = 10000,
                 n_eval_episodes: int = 5,
                 best_model_save_path: str = None,
                 log_callback: Callable = None,
                 progress_callback: Callable = None,
                 verbose: int = 1):
        """
        Initialize validation callback
        
        Args:
            validation_env: Environment for validation
            eval_freq: Frequency of evaluation (in training steps)
            n_eval_episodes: Number of episodes for each evaluation
            best_model_save_path: Path to save the best model
            log_callback: Callback function for logging messages
            progress_callback: Callback function for structured progress updates
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.validation_env = validation_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        
        # Tracking variables
        self.best_mean_reward = -np.inf
        self.best_sharpe_ratio = -np.inf
        self.evaluations = []
        self.detailed_training_history = []
        
    def _init_callback(self) -> None:
        """Initialize callback"""
        if self.best_model_save_path is not None:
            os.makedirs(os.path.dirname(self.best_model_save_path), exist_ok=True)
    
    def _on_step(self) -> bool:
        """Called at each training step"""
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_agent()
        return True
    
    def _evaluate_agent(self):
        """Evaluate agent on validation set"""
        episode_rewards = []
        portfolio_stats = []
        trade_summaries = []
        
        for episode in range(self.n_eval_episodes):
            obs, _ = self.validation_env.reset()
            episode_reward = 0
            done = False
            
            # Track trade data only
            trades = []
            current_trade = None
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.validation_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                # Track trades only
                if 'trade_executed' in info and info['trade_executed']:
                    current_price = float(info.get('current_price', 0))
                    position_before = float(info.get('position_before', 0))
                    position_after = float(info.get('position', 0))
                    
                    if action == 1 and position_before <= 0:  # Buy (entry)
                        current_trade = {
                            'trade_type': 'buy',
                            'entry_price': current_price,
                            'entry_time': len(trades),  # Use trade count as time reference
                            'cost': float(info.get('trade_cost', 0)),
                            'portfolio_value_at_entry': float(info.get('portfolio_value', 0))
                        }
                    elif action == 2 and position_before >= 0 and current_trade:  # Sell (exit)
                        current_trade.update({
                            'exit_price': current_price,
                            'exit_time': len(trades),  # Use trade count as time reference
                            'exit_cost': float(info.get('trade_cost', 0)),
                            'portfolio_value_at_exit': float(info.get('portfolio_value', 0))
                        })
                        
                        # Calculate profit/loss
                        profit_pct = ((current_trade['exit_price'] - current_trade['entry_price']) / current_trade['entry_price']) * 100
                        current_trade['profit_pct'] = profit_pct
                        current_trade['profit_absolute'] = current_trade['portfolio_value_at_exit'] - current_trade['portfolio_value_at_entry']
                        current_trade['total_cost'] = current_trade['cost'] + current_trade['exit_cost']
                        current_trade['holding_period'] = current_trade['exit_time'] - current_trade['entry_time']
                        
                        trades.append(current_trade)
                        current_trade = None
            
            episode_rewards.append(episode_reward)
            
            # Get portfolio statistics for this episode
            stats = self.validation_env.get_portfolio_stats()
            portfolio_stats.append(stats)
            
            # Create episode summary focused on trades
            episode_summary = {
                'episode': episode,
                'total_reward': float(episode_reward),
                'trades': trades,
                'total_trades': len(trades),
                'profitable_trades': len([t for t in trades if t.get('profit_pct', 0) > 0]),
                'avg_profit_per_trade': np.mean([t.get('profit_pct', 0) for t in trades]) if trades else 0,
                'total_trading_cost': sum([t.get('total_cost', 0) for t in trades]),
                'final_stats': stats
            }
            
            trade_summaries.append(episode_summary)
        
        # Calculate validation metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        # Calculate average portfolio metrics
        avg_stats = self._average_portfolio_stats(portfolio_stats)
        sharpe_ratio = avg_stats.get('sharpe_ratio', 0)
        total_return = avg_stats.get('total_return', 0)
        max_drawdown = avg_stats.get('max_drawdown', 0)
        
        # Log results
        log_msg = (f"Validation Step {self.n_calls}: "
                  f"Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}, "
                  f"Sharpe: {sharpe_ratio:.4f}, "
                  f"Return: {total_return:.4f}, "
                  f"MDD: {max_drawdown:.4f}")
        
        if self.log_callback:
            self.log_callback(log_msg)
        else:
            print(log_msg)
        
        # Save best model based on Sharpe ratio
        if sharpe_ratio > self.best_sharpe_ratio:
            self.best_sharpe_ratio = sharpe_ratio
            self.best_mean_reward = mean_reward
            
            if self.best_model_save_path:
                self.model.save(self.best_model_save_path)
                if self.log_callback:
                    self.log_callback(f"New best model saved! Sharpe: {sharpe_ratio:.4f}")
                else:
                    print(f"New best model saved! Sharpe ratio: {sharpe_ratio:.4f}")
        
        # Calculate aggregated trade statistics
        all_trades = []
        for episode in trade_summaries:
            all_trades.extend(episode['trades'])
        
        trade_stats = {
            'total_trades': len(all_trades),
            'profitable_trades': len([t for t in all_trades if t.get('profit_pct', 0) > 0]),
            'win_rate': len([t for t in all_trades if t.get('profit_pct', 0) > 0]) / len(all_trades) * 100 if all_trades else 0,
            'avg_profit_per_trade': np.mean([t.get('profit_pct', 0) for t in all_trades]) if all_trades else 0,
            'best_trade': max(all_trades, key=lambda x: x.get('profit_pct', 0)) if all_trades else None,
            'worst_trade': min(all_trades, key=lambda x: x.get('profit_pct', 0)) if all_trades else None,
            'total_trading_cost': sum([t.get('total_cost', 0) for t in all_trades]),
            'avg_holding_period': np.mean([t.get('holding_period', 0) for t in all_trades]) if all_trades else 0
        }
        
        # Store evaluation results
        evaluation_result = {
            'timestep': self.n_calls,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'portfolio_stats': avg_stats,
            'trade_summaries': trade_summaries,
            'aggregated_trade_stats': trade_stats
        }
        self.evaluations.append(evaluation_result)
        
        # Store in detailed training history
        self.detailed_training_history.append({
            'training_step': self.n_calls,
            'validation_result': evaluation_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Send structured progress update to GUI
        if self.progress_callback:
            self.progress_callback(evaluation_result)
    
    def _average_portfolio_stats(self, stats_list):
        """Average portfolio statistics across episodes"""
        if not stats_list:
            return {}
        
        # Filter out empty stats
        valid_stats = [stats for stats in stats_list if stats]
        if not valid_stats:
            return {}
        
        averaged = {}
        for key in valid_stats[0].keys():
            values = [stats[key] for stats in valid_stats if key in stats]
            if values:
                averaged[key] = np.mean(values)
        
        return averaged

class TrainingPipeline:
    """
    Complete training pipeline for the trading agent
    """
    
    def __init__(self, 
                 data_path: str,
                 hyperparameters: Dict = None,
                 reward_weights: Dict = None,
                 log_callback: Callable = None,
                 progress_callback: Callable = None):
        """
        Initialize training pipeline
        
        Args:
            data_path: Path to preprocessed data (.npy file)
            hyperparameters: Training hyperparameters
            reward_weights: Reward function weights
            log_callback: Callback for logging messages
            progress_callback: Callback for structured progress updates
        """
        self.data_path = data_path
        self.hyperparameters = hyperparameters or {}
        self.reward_weights = reward_weights or {
            'profit': 1.0, 'sharpe': 0.5, 'cost': 1.0, 'mdd': 0.5
        }
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        
        # Training components
        self.train_env = None
        self.val_env = None
        self.agent = None
        self.callback = None
        
        # Results
        self.training_results = {}
        
    def setup_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split data chronologically
        
        Returns:
            train_data, val_data, test_data
        """
        if self.log_callback:
            self.log_callback(f"Loading data from {self.data_path}")
        
        # Load preprocessed data
        data = np.load(self.data_path)
        if self.log_callback:
            self.log_callback(f"Loaded data shape: {data.shape}")
        
        # Chronological split: 70% train, 15% val, 15% test
        n_samples = data.shape[0]
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        if self.log_callback:
            self.log_callback(f"Data split - Train: {train_data.shape[0]}, "
                            f"Val: {val_data.shape[0]}, Test: {test_data.shape[0]}")
        
        return train_data, val_data, test_data
    
    def setup_environments(self, train_data: np.ndarray, val_data: np.ndarray):
        """Setup training and validation environments"""
        if self.log_callback:
            self.log_callback("Setting up training and validation environments...")
        
        # Create environments
        self.train_env = TradingEnv(
            data_array=train_data,
            reward_weights=self.reward_weights
        )
        
        self.val_env = TradingEnv(
            data_array=val_data,
            reward_weights=self.reward_weights
        )
        
        # Wrap for monitoring
        self.train_env = Monitor(self.train_env)
        
        if self.log_callback:
            self.log_callback("Environments created successfully")
    
    def setup_agent(self, resume_from_model=None):
        """Setup trading agent"""
        if resume_from_model and os.path.exists(resume_from_model):
            if self.log_callback:
                self.log_callback(f"Loading agent from saved model: {resume_from_model}")
            
            # Load existing model
            from stable_baselines3 import PPO
            self.agent = PPO.load(resume_from_model)
            
            # Set the environment for the loaded model
            self.agent.set_env(self.train_env)
            
            if self.log_callback:
                self.log_callback(f"Agent loaded successfully from: {os.path.basename(resume_from_model)}")
                self.log_callback(f"Model will continue training from previous state")
        else:
            if self.log_callback:
                self.log_callback("Initializing new trading agent...")
            
            self.agent = create_trading_agent(
                env=self.train_env,
                hyperparameters=self.hyperparameters
            )
            
            # Get network summary
            summary = self.agent.get_network_summary()
            if self.log_callback:
                self.log_callback(f"Agent created - Total parameters: {summary['total_parameters']:,}")
                self.log_callback(f"Device: {summary['device']}")
    
    def setup_callback(self, best_model_path: str):
        """Setup validation callback"""
        self.callback = ValidationCallback(
            validation_env=self.val_env,
            eval_freq=10000,  # Evaluate every 10k steps
            n_eval_episodes=5,
            best_model_save_path=best_model_path,
            log_callback=self.log_callback,
            progress_callback=self.progress_callback
        )
    
    def train(self, 
              total_timesteps: int = 100000,
              model_save_dir: str = "models",
              resume_from_model: str = None) -> Dict:
        """
        Execute complete training pipeline
        
        Args:
            total_timesteps: Total training timesteps
            model_save_dir: Directory to save models
            resume_from_model: Path to model file to resume from (optional)
            
        Returns:
            Dictionary with training results
        """
        try:
            # Create model directory
            os.makedirs(model_save_dir, exist_ok=True)
            
            # Setup data
            train_data, val_data, test_data = self.setup_data()
            
            # Setup environments
            self.setup_environments(train_data, val_data)
            
            # Setup agent (new or resumed)
            self.setup_agent(resume_from_model)
            
            # Setup callback
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if resume_from_model:
                # For resume, use a continuation suffix
                best_model_path = os.path.join(model_save_dir, f"resumed_best_model_{timestamp}.zip")
            else:
                best_model_path = os.path.join(model_save_dir, f"best_model_{timestamp}.zip")
            self.setup_callback(best_model_path)
            
            # Start training
            if self.log_callback:
                if resume_from_model:
                    self.log_callback(f"Resuming training for {total_timesteps:,} additional timesteps...")
                else:
                    self.log_callback(f"Starting training for {total_timesteps:,} timesteps...")
            
            # For resumed training, we don't want to reset the timestep counter
            reset_num_timesteps = not bool(resume_from_model)
            
            self.agent.learn(
                total_timesteps=total_timesteps,
                callback=self.callback,
                reset_num_timesteps=reset_num_timesteps
            )
            
            # Save final model
            if resume_from_model:
                final_model_path = os.path.join(model_save_dir, f"resumed_final_model_{timestamp}.zip")
            else:
                final_model_path = os.path.join(model_save_dir, f"final_model_{timestamp}.zip")
            self.agent.save(final_model_path)
            
            # Compile results
            self.training_results = {
                'success': True,
                'best_model_path': best_model_path,
                'final_model_path': final_model_path,
                'best_sharpe_ratio': self.callback.best_sharpe_ratio,
                'best_mean_reward': self.callback.best_mean_reward,
                'evaluations': self.callback.evaluations,
                'detailed_training_history': self.callback.detailed_training_history,
                'training_summary': {
                    'total_timesteps': total_timesteps,
                    'evaluation_frequency': self.callback.eval_freq,
                    'n_evaluation_episodes': self.callback.n_eval_episodes,
                    'total_evaluations': len(self.callback.evaluations),
                    'resumed_from_model': resume_from_model,
                    'is_resumed_training': bool(resume_from_model)
                },
                'data_split': {
                    'train_samples': train_data.shape[0],
                    'val_samples': val_data.shape[0],
                    'test_samples': test_data.shape[0]
                },
                'hyperparameters': self.hyperparameters,
                'reward_weights': self.reward_weights,
                'timestamp': timestamp
            }
            
            if self.log_callback:
                self.log_callback("Training completed successfully!")
                self.log_callback(f"Best model saved to: {best_model_path}")
                self.log_callback(f"Best Sharpe ratio: {self.callback.best_sharpe_ratio:.4f}")
            
            return self.training_results
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            if self.log_callback:
                self.log_callback(error_msg)
            
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
    
    def save_training_log(self, log_path: str):
        """Save training results to JSON file"""
        if self.training_results:
            with open(log_path, 'w') as f:
                json.dump(self.training_results, f, indent=2, default=str)

def train_sappo_agent(data_path: str,
                     hyperparameters: Dict = None,
                     reward_weights: Dict = None,
                     total_timesteps: int = 100000,
                     model_save_dir: str = "models",
                     log_callback: Callable = None,
                     progress_callback: Callable = None,
                     resume_from_model: str = None) -> Dict:
    """
    Main function to train Sappo trading agent
    
    Args:
        data_path: Path to preprocessed data
        hyperparameters: Training hyperparameters
        reward_weights: Reward function weights
        total_timesteps: Total training timesteps
        model_save_dir: Directory to save models
        log_callback: Callback for logging
        progress_callback: Callback for structured progress updates
        resume_from_model: Path to model file to resume from (optional)
        
    Returns:
        Training results dictionary
    """
    pipeline = TrainingPipeline(
        data_path=data_path,
        hyperparameters=hyperparameters,
        reward_weights=reward_weights,
        log_callback=log_callback,
        progress_callback=progress_callback
    )
    
    results = pipeline.train(
        total_timesteps=total_timesteps,
        model_save_dir=model_save_dir,
        resume_from_model=resume_from_model
    )
    
    # Save training log
    if results.get('success'):
        timestamp = results['timestamp']
        log_path = os.path.join(model_save_dir, f"training_log_{timestamp}.json")
        pipeline.save_training_log(log_path)
    
    return results

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_path> [total_timesteps]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    total_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    
    def print_log(message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    results = train_sappo_agent(
        data_path=data_path,
        total_timesteps=total_timesteps,
        log_callback=print_log
    )
    
    if results['success']:
        print(f"\nTraining completed successfully!")
        print(f"Best model: {results['best_model_path']}")
        print(f"Best Sharpe ratio: {results['best_sharpe_ratio']:.4f}")
    else:
        print(f"\nTraining failed: {results['error']}")