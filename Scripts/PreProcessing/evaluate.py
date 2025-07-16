import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Callable, Optional
import json
import os

from TradingEnv import TradingEnv
from agent import TradingAgent
from stable_baselines3 import PPO

class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for trading agents
    """
    
    def __init__(self, log_callback: Callable = None):
        """
        Initialize evaluator
        
        Args:
            log_callback: Callback function for logging messages
        """
        self.log_callback = log_callback
        self.evaluation_results = {}
        
    def log(self, message: str):
        """Log message"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def evaluate_agent(self,
                      model_path: str,
                      test_data: np.ndarray,
                      reward_weights: Dict = None,
                      n_episodes: int = 1) -> Dict:
        """
        Evaluate trained agent on test data
        
        Args:
            model_path: Path to trained model
            test_data: Test dataset
            reward_weights: Reward function weights
            n_episodes: Number of evaluation episodes
            
        Returns:
            Comprehensive evaluation results
        """
        self.log(f"Starting evaluation with {n_episodes} episodes...")
        
        # Setup environment
        if reward_weights is None:
            reward_weights = {'profit': 1.0, 'sharpe': 0.5, 'cost': 1.0, 'mdd': 0.5}
        
        test_env = TradingEnv(data_array=test_data, reward_weights=reward_weights)
        
        # Load model
        self.log(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=test_env)
        
        # Run evaluation episodes
        all_results = []
        for episode in range(n_episodes):
            self.log(f"Running evaluation episode {episode + 1}/{n_episodes}")
            episode_results = self._run_single_episode(model, test_env, episode)
            all_results.append(episode_results)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)
        
        # Calculate additional metrics
        enhanced_results = self._calculate_enhanced_metrics(aggregated_results)
        
        # Create benchmark comparison
        benchmark_results = self._create_benchmark(test_data, test_env)
        enhanced_results['benchmark'] = benchmark_results
        
        self.evaluation_results = enhanced_results
        self.log("Evaluation completed successfully!")
        
        return enhanced_results
    
    def _run_single_episode(self, model, env, episode_num: int) -> Dict:
        """Run a single evaluation episode"""
        obs, _ = env.reset()
        
        # Track episode data
        portfolio_values = []
        actions = []
        rewards = []
        positions = []
        timestamps = []
        prices = []
        
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record data
            portfolio_values.append(info['portfolio_value'])
            actions.append(action)
            rewards.append(reward)
            positions.append(info['position'])
            prices.append(info['current_price'])
            timestamps.append(step_count)
            
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
        
        # Get final portfolio statistics
        portfolio_stats = env.get_portfolio_stats()
        
        return {
            'episode': episode_num,
            'episode_reward': episode_reward,
            'step_count': step_count,
            'portfolio_values': portfolio_values,
            'actions': actions,
            'rewards': rewards,
            'positions': positions,
            'prices': prices,
            'timestamps': timestamps,
            'portfolio_stats': portfolio_stats
        }
    
    def _aggregate_results(self, episode_results: List[Dict]) -> Dict:
        """Aggregate results across episodes"""
        if not episode_results:
            return {}
        
        # For single episode evaluation, just return the first episode
        # For multiple episodes, we can average metrics
        main_episode = episode_results[0]
        
        # Calculate average statistics across episodes
        avg_stats = {}
        for key in main_episode['portfolio_stats'].keys():
            values = [ep['portfolio_stats'][key] for ep in episode_results if key in ep['portfolio_stats']]
            if values:
                avg_stats[key] = np.mean(values)
        
        return {
            'n_episodes': len(episode_results),
            'avg_episode_reward': np.mean([ep['episode_reward'] for ep in episode_results]),
            'avg_step_count': np.mean([ep['step_count'] for ep in episode_results]),
            'portfolio_stats': avg_stats,
            'detailed_results': main_episode,  # Use first episode for detailed analysis
            'all_episodes': episode_results
        }
    
    def _calculate_enhanced_metrics(self, results: Dict) -> Dict:
        """Calculate additional performance metrics"""
        main_episode = results['detailed_results']
        portfolio_values = np.array(main_episode['portfolio_values'])
        
        if len(portfolio_values) < 2:
            return results
        
        # Time-based analysis
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        # Enhanced metrics
        enhanced_metrics = {
            # Return metrics
            'total_return_pct': ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) * 100,
            'annualized_return': self._calculate_annualized_return(portfolio_values),
            'volatility': np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0,
            
            # Risk metrics
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(portfolio_values),
            'max_drawdown_pct': self._calculate_max_drawdown(portfolio_values) * 100,
            
            # Trading metrics
            'win_rate': self._calculate_win_rate(returns),
            'profit_factor': self._calculate_profit_factor(returns),
            'avg_trade_return': np.mean(returns) * 100 if len(returns) > 0 else 0,
            
            # Portfolio progression
            'portfolio_progression': portfolio_values.tolist(),
            'returns_series': returns.tolist(),
            'cumulative_returns': ((portfolio_values / portfolio_values[0] - 1) * 100).tolist()
        }
        
        results['enhanced_metrics'] = enhanced_metrics
        return results
    
    def _calculate_annualized_return(self, portfolio_values: np.ndarray) -> float:
        """Calculate annualized return"""
        if len(portfolio_values) < 2:
            return 0.0
        
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        n_periods = len(portfolio_values)
        periods_per_year = 252 * 24  # Assuming hourly data
        
        if n_periods >= periods_per_year:
            years = n_periods / periods_per_year
            annualized = (1 + total_return) ** (1/years) - 1
        else:
            # Extrapolate for shorter periods
            annualized = total_return * (periods_per_year / n_periods)
        
        return annualized * 100
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return / std_return) * np.sqrt(252 * 24)  # Annualized
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        return (mean_return / downside_std) * np.sqrt(252 * 24)
    
    def _calculate_calmar_ratio(self, portfolio_values: np.ndarray) -> float:
        """Calculate Calmar ratio (return/max drawdown)"""
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        max_dd = self._calculate_max_drawdown(portfolio_values)
        
        if max_dd == 0:
            return float('inf') if total_return > 0 else 0.0
        
        return total_return / max_dd
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if len(returns) == 0:
            return 0.0
        
        positive_returns = returns > 0
        return (np.sum(positive_returns) / len(returns)) * 100
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = np.abs(np.sum(returns[returns < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss
    
    def _create_benchmark(self, test_data: np.ndarray, test_env: TradingEnv) -> Dict:
        """Create buy-and-hold benchmark"""
        self.log("Creating buy-and-hold benchmark...")
        
        # Simple buy-and-hold strategy
        obs, _ = test_env.reset()
        
        # Buy at the beginning and hold
        buy_price = test_env._get_current_price()
        
        # Get final price
        while True:
            # Always hold (action = 0)
            obs, reward, terminated, truncated, info = test_env.step(0)
            if terminated or truncated:
                break
        
        sell_price = test_env._get_current_price()
        benchmark_return = ((sell_price - buy_price) / buy_price) * 100
        
        return {
            'strategy': 'Buy and Hold',
            'total_return_pct': benchmark_return,
            'buy_price': buy_price,
            'sell_price': sell_price
        }
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            return "No evaluation results available"
        
        results = self.evaluation_results
        enhanced = results.get('enhanced_metrics', {})
        portfolio_stats = results.get('portfolio_stats', {})
        benchmark = results.get('benchmark', {})
        
        report = f"""
=== SAPPO Trading Bot Evaluation Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 PERFORMANCE SUMMARY
Total Return: {enhanced.get('total_return_pct', 0):.2f}%
Annualized Return: {enhanced.get('annualized_return', 0):.2f}%
Volatility: {enhanced.get('volatility', 0):.2f}%
Sharpe Ratio: {enhanced.get('sharpe_ratio', 0):.3f}
Max Drawdown: {enhanced.get('max_drawdown_pct', 0):.2f}%

🎯 TRADING METRICS
Win Rate: {enhanced.get('win_rate', 0):.1f}%
Profit Factor: {enhanced.get('profit_factor', 0):.2f}
Average Trade Return: {enhanced.get('avg_trade_return', 0):.3f}%
Total Trades: {portfolio_stats.get('trade_count', 0)}

📈 RISK METRICS
Sortino Ratio: {enhanced.get('sortino_ratio', 0):.3f}
Calmar Ratio: {enhanced.get('calmar_ratio', 0):.3f}
Final Portfolio Value: ${portfolio_stats.get('final_value', 0):,.2f}
Total Transaction Costs: ${portfolio_stats.get('total_cost', 0):.2f}

🏆 BENCHMARK COMPARISON
Agent Return: {enhanced.get('total_return_pct', 0):.2f}%
Buy & Hold Return: {benchmark.get('total_return_pct', 0):.2f}%
Outperformance: {enhanced.get('total_return_pct', 0) - benchmark.get('total_return_pct', 0):.2f}%

📋 EVALUATION DETAILS
Episodes: {results.get('n_episodes', 0)}
Avg Episode Reward: {results.get('avg_episode_reward', 0):.3f}
Avg Steps per Episode: {results.get('avg_step_count', 0)}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.log(f"Report saved to {save_path}")
        
        return report
    
    def plot_performance(self, save_path: str = None, show_plot: bool = False) -> plt.Figure:
        """Create performance visualization"""
        if not self.evaluation_results:
            return None
        
        enhanced = self.evaluation_results.get('enhanced_metrics', {})
        portfolio_values = enhanced.get('portfolio_progression', [])
        benchmark = self.evaluation_results.get('benchmark', {})
        
        if not portfolio_values:
            return None
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sappo Trading Bot Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Portfolio Value Over Time
        time_steps = range(len(portfolio_values))
        ax1.plot(time_steps, portfolio_values, 'b-', linewidth=2, label='Agent Portfolio')
        
        # Add benchmark line (simple calculation)
        if benchmark.get('total_return_pct'):
            initial_value = portfolio_values[0]
            final_benchmark = initial_value * (1 + benchmark['total_return_pct'] / 100)
            benchmark_line = np.linspace(initial_value, final_benchmark, len(portfolio_values))
            ax1.plot(time_steps, benchmark_line, 'r--', linewidth=2, label='Buy & Hold')
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Returns
        cumulative_returns = enhanced.get('cumulative_returns', [])
        if cumulative_returns:
            ax2.plot(time_steps, cumulative_returns, 'g-', linewidth=2)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.set_title('Cumulative Returns (%)')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Return (%)')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        if portfolio_values:
            portfolio_array = np.array(portfolio_values)
            peak = np.maximum.accumulate(portfolio_array)
            drawdown = (peak - portfolio_array) / peak * 100
            
            ax3.fill_between(time_steps, 0, -drawdown, alpha=0.3, color='red')
            ax3.plot(time_steps, -drawdown, 'r-', linewidth=1)
            ax3.set_title('Drawdown (%)')
            ax3.set_xlabel('Time Steps')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Key Metrics Bar Chart
        metrics_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        metrics_values = [
            enhanced.get('total_return_pct', 0),
            enhanced.get('sharpe_ratio', 0) * 10,  # Scale for visibility
            enhanced.get('max_drawdown_pct', 0),
            enhanced.get('win_rate', 0)
        ]
        
        colors = ['green' if v >= 0 else 'red' for v in metrics_values]
        bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax4.set_title('Key Performance Metrics')
        ax4.set_ylabel('Value')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.log(f"Performance plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def save_results(self, save_dir: str, prefix: str = "evaluation"):
        """Save all evaluation results"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = os.path.join(save_dir, f"{prefix}_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        # Save text report
        report_path = os.path.join(save_dir, f"{prefix}_report_{timestamp}.txt")
        report = self.generate_report(report_path)
        
        # Save performance plot
        plot_path = os.path.join(save_dir, f"{prefix}_performance_{timestamp}.png")
        self.plot_performance(plot_path)
        
        self.log(f"All evaluation results saved to {save_dir}")
        
        return {
            'json_path': json_path,
            'report_path': report_path,
            'plot_path': plot_path
        }

def evaluate_sappo_model(model_path: str,
                        test_data: np.ndarray,
                        reward_weights: Dict = None,
                        save_dir: str = "evaluation_results",
                        log_callback: Callable = None) -> Dict:
    """
    Main function to evaluate Sappo trading model
    
    Args:
        model_path: Path to trained model
        test_data: Test dataset
        reward_weights: Reward function weights
        save_dir: Directory to save results
        log_callback: Callback for logging
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = PerformanceEvaluator(log_callback=log_callback)
    
    results = evaluator.evaluate_agent(
        model_path=model_path,
        test_data=test_data,
        reward_weights=reward_weights
    )
    
    # Save all results
    if save_dir:
        saved_files = evaluator.save_results(save_dir)
        results['saved_files'] = saved_files
    
    return results

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <model_path> <test_data_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    
    # Load test data
    test_data = np.load(test_data_path)
    
    def print_log(message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    results = evaluate_sappo_model(
        model_path=model_path,
        test_data=test_data,
        log_callback=print_log
    )
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETED")
    print("="*50)
    
    enhanced = results.get('enhanced_metrics', {})
    print(f"Total Return: {enhanced.get('total_return_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {enhanced.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {enhanced.get('max_drawdown_pct', 0):.2f}%")