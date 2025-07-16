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
                 verbose: int = 1):
        """
        Initialize validation callback
        
        Args:
            validation_env: Environment for validation
            eval_freq: Frequency of evaluation (in training steps)
            n_eval_episodes: Number of episodes for each evaluation
            best_model_save_path: Path to save the best model
            log_callback: Callback function for logging messages
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.validation_env = validation_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.log_callback = log_callback
        
        # Tracking variables
        self.best_mean_reward = -np.inf
        self.best_sharpe_ratio = -np.inf
        self.evaluations = []
        
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
        
        for episode in range(self.n_eval_episodes):
            obs, _ = self.validation_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.validation_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            
            # Get portfolio statistics for this episode
            stats = self.validation_env.get_portfolio_stats()
            portfolio_stats.append(stats)
        
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
        
        # Store evaluation results
        evaluation_result = {
            'timestep': self.n_calls,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'portfolio_stats': avg_stats
        }
        self.evaluations.append(evaluation_result)
    
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
                 log_callback: Callable = None):
        """
        Initialize training pipeline
        
        Args:
            data_path: Path to preprocessed data (.npy file)
            hyperparameters: Training hyperparameters
            reward_weights: Reward function weights
            log_callback: Callback for logging messages
        """
        self.data_path = data_path
        self.hyperparameters = hyperparameters or {}
        self.reward_weights = reward_weights or {
            'profit': 1.0, 'sharpe': 0.5, 'cost': 1.0, 'mdd': 0.5
        }
        self.log_callback = log_callback
        
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
    
    def setup_agent(self):
        """Setup trading agent"""
        if self.log_callback:
            self.log_callback("Initializing trading agent...")
        
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
            log_callback=self.log_callback
        )
    
    def train(self, 
              total_timesteps: int = 100000,
              model_save_dir: str = "models") -> Dict:
        """
        Execute complete training pipeline
        
        Args:
            total_timesteps: Total training timesteps
            model_save_dir: Directory to save models
            
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
            
            # Setup agent
            self.setup_agent()
            
            # Setup callback
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(model_save_dir, f"best_model_{timestamp}.zip")
            self.setup_callback(best_model_path)
            
            # Start training
            if self.log_callback:
                self.log_callback(f"Starting training for {total_timesteps:,} timesteps...")
            
            self.agent.train(
                total_timesteps=total_timesteps,
                callback=self.callback
            )
            
            # Save final model
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
                     log_callback: Callable = None) -> Dict:
    """
    Main function to train Sappo trading agent
    
    Args:
        data_path: Path to preprocessed data
        hyperparameters: Training hyperparameters
        reward_weights: Reward function weights
        total_timesteps: Total training timesteps
        model_save_dir: Directory to save models
        log_callback: Callback for logging
        
    Returns:
        Training results dictionary
    """
    pipeline = TrainingPipeline(
        data_path=data_path,
        hyperparameters=hyperparameters,
        reward_weights=reward_weights,
        log_callback=log_callback
    )
    
    results = pipeline.train(
        total_timesteps=total_timesteps,
        model_save_dir=model_save_dir
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