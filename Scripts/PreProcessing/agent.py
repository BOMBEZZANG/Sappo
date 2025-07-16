import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Type, Union

class GRUFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using GRU layers for time-series data
    
    Processes (batch_size, 24, 529) -> (batch_size, features_dim)
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        """
        Initialize GRU Feature Extractor
        
        Args:
            observation_space: Environment observation space
            features_dim: Number of output features
        """
        super().__init__(observation_space, features_dim)
        
        # Input dimensions
        self.input_dim = observation_space.shape[-1]  # 529 features
        self.sequence_length = observation_space.shape[0]  # 24 time steps
        
        # GRU layers for temporal feature extraction
        self.gru1 = nn.GRU(
            input_size=self.input_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )
        
        self.gru2 = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )
        
        # Dense layers for feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRU feature extractor
        
        Args:
            observations: Tensor of shape (batch_size, 24, 529)
            
        Returns:
            features: Tensor of shape (batch_size, features_dim)
        """
        batch_size = observations.shape[0]
        
        # Reshape if needed
        if len(observations.shape) == 3:
            # Input: (batch_size, sequence_length, features)
            x = observations
        else:
            # Flatten and reshape
            x = observations.view(batch_size, self.sequence_length, self.input_dim)
        
        # First GRU layer
        gru1_out, _ = self.gru1(x)  # (batch_size, seq_len, 128)
        
        # Second GRU layer
        gru2_out, hidden = self.gru2(gru1_out)  # (batch_size, seq_len, 64)
        
        # Use the last hidden state
        last_hidden = hidden[-1]  # (batch_size, 64)
        
        # Process through dense layers
        features = self.feature_processor(last_hidden)  # (batch_size, features_dim)
        
        return features

class GRUActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy with GRU feature extractor
    """
    
    def __init__(self, 
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule,
                 net_arch: List[Union[int, Dict[str, List[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 features_extractor_class: Type[BaseFeaturesExtractor] = GRUFeatureExtractor,
                 features_extractor_kwargs: Dict = None,
                 *args, **kwargs):
        
        # Set default network architecture if not provided
        if net_arch is None:
            net_arch = [128, 64]
        
        # Set default feature extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"features_dim": 128}
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args, **kwargs
        )

class TradingAgent:
    """
    Wrapper class for the trading agent with utilities
    """
    
    def __init__(self, 
                 env,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 ent_coef: float = 0.01, # <-- 파라미터 추가

                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 clip_range: float = 0.2):
        """
        Initialize Trading Agent
        
        Args:
            env: Trading environment
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            n_steps: Number of steps to run for each environment per update
            batch_size: Batch size for optimization
            n_epochs: Number of epochs when optimizing the surrogate loss
            clip_range: Clipping parameter for PPO
        """
        self.env = env
        
        # Create PPO agent with custom GRU policy
        self.model = PPO(
            GRUActorCriticPolicy,
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            ent_coef=ent_coef, # <-- PPO 모델에 전달
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            device="auto"  # Use GPU if available
        )
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'portfolio_values': [],
            'validation_scores': []
        }
    
    def train(self, total_timesteps: int, callback=None, reset_num_timesteps=True):
        """
        Train the agent
        
        Args:
            total_timesteps: Total number of training steps
            callback: Optional callback for monitoring
            reset_num_timesteps: Whether to reset timestep counter (False for resuming)
        """
        print(f"Starting training for {total_timesteps} timesteps...")
        print(f"Using device: {self.model.device}")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="sappo_trading",
            reset_num_timesteps=reset_num_timesteps
        )
        
        print("Training completed!")
    
    def learn(self, total_timesteps: int, callback=None, reset_num_timesteps=True):
        """
        Direct access to PPO learn method for resuming training
        
        Args:
            total_timesteps: Total number of training steps
            callback: Optional callback for monitoring
            reset_num_timesteps: Whether to reset timestep counter (False for resuming)
        """
        return self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="sappo_trading",
            reset_num_timesteps=reset_num_timesteps
        )
    
    def predict(self, observation, deterministic=True):
        """
        Predict action for given observation
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            action, _states
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load a trained model"""
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    @staticmethod
    def load_model(path: str, env):
        """Load a trained model (static method)"""
        return PPO.load(path, env=env)
    
    def evaluate(self, env, n_episodes: int = 10, deterministic: bool = True):
        """
        Evaluate the agent
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dict with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            portfolio_values.append(info.get('portfolio_value', 0))
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_portfolio_value': np.mean(portfolio_values),
            'episode_rewards': episode_rewards,
            'portfolio_values': portfolio_values
        }
    
    def get_network_summary(self):
        """Get summary of the network architecture"""
        return {
            'policy_class': 'GRUActorCriticPolicy',
            'feature_extractor': 'GRUFeatureExtractor',
            'observation_space': self.env.observation_space,
            'action_space': self.env.action_space,
            'device': str(self.model.device),
            'total_parameters': sum(p.numel() for p in self.model.policy.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
        }

def create_trading_agent(env, hyperparameters: Dict = None):
    """
    Factory function to create a trading agent with default or custom hyperparameters
    
    Args:
        env: Trading environment
        hyperparameters: Dictionary of hyperparameters
        
    Returns:
        TradingAgent instance
    """
    if hyperparameters is None:
        hyperparameters = {}
    
    # Default hyperparameters
    default_params = {
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'ent_coef': 0.01, # 기본값 설정

        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'clip_range': 0.2
    }
    
    # Update with provided hyperparameters
    default_params.update(hyperparameters)
    
    return TradingAgent(env, **default_params)