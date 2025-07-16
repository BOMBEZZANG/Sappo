#!/usr/bin/env python3
"""
Test script for resume training functionality
"""

import numpy as np
import os
import json
from datetime import datetime
from train import train_sappo_agent

def test_resume_training():
    """Test the resume training functionality"""
    
    print("Testing Resume Training Functionality")
    print("=" * 50)
    
    # Create sample data for testing
    print("Creating sample data...")
    n_samples = 1000
    window_size = 50
    n_features = 20
    
    # Generate synthetic trading data
    np.random.seed(42)
    sample_data = np.random.randn(n_samples, window_size, n_features)
    
    # Add some trend to make it more realistic
    for i in range(n_samples):
        trend = np.sin(i * 0.01) * 0.1
        sample_data[i, :, 3] = 100 + trend + np.cumsum(np.random.randn(window_size) * 0.01)  # Close price
    
    # Save sample data
    data_path = "/tmp/test_resume_data.npy"
    np.save(data_path, sample_data)
    print(f"Sample data saved to {data_path}")
    
    # Test parameters
    hyperparameters = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'ent_coef': 0.01
    }
    
    reward_weights = {
        'profit': 1.0,
        'sharpe': 0.5,
        'cost': 0.1,
        'mdd': 0.1
    }
    
    def log_callback(message):
        print(f"[LOG] {message}")
    
    # Step 1: Initial training
    print("\\n" + "="*50)
    print("STEP 1: Initial Training")
    print("="*50)
    
    initial_results = train_sappo_agent(\n        data_path=data_path,\n        hyperparameters=hyperparameters,\n        reward_weights=reward_weights,\n        total_timesteps=5000,  # Short training for testing\n        model_save_dir="/tmp/test_models",\n        log_callback=log_callback\n    )\n    \n    if initial_results['success']:\n        print("✓ Initial training completed successfully!")\n        print(f"Best model saved: {initial_results['best_model_path']}")\n        print(f"Final model saved: {initial_results['final_model_path']}")\n        \n        # Step 2: Resume training\n        print("\\n" + "="*50)\n        print("STEP 2: Resume Training")\n        print("="*50)\n        \n        resume_results = train_sappo_agent(\n            data_path=data_path,\n            hyperparameters=hyperparameters,\n            reward_weights=reward_weights,\n            total_timesteps=3000,  # Additional training steps\n            model_save_dir="/tmp/test_models",\n            log_callback=log_callback,\n            resume_from_model=initial_results['best_model_path']\n        )\n        \n        if resume_results['success']:\n            print("✓ Resume training completed successfully!")\n            print(f"Resumed best model: {resume_results['best_model_path']}")\n            print(f"Resumed final model: {resume_results['final_model_path']}")\n            \n            # Compare results\n            print("\\n" + "="*50)\n            print("RESULTS COMPARISON")\n            print("="*50)\n            \n            print("Initial Training:")\n            print(f"  - Best Sharpe Ratio: {initial_results['best_sharpe_ratio']:.4f}")\n            print(f"  - Best Mean Reward: {initial_results['best_mean_reward']:.4f}")\n            print(f"  - Evaluations: {len(initial_results['evaluations'])}")\n            print(f"  - Is Resumed: {initial_results['training_summary'].get('is_resumed_training', False)}")\n            \n            print("\\nResumed Training:")\n            print(f"  - Best Sharpe Ratio: {resume_results['best_sharpe_ratio']:.4f}")\n            print(f"  - Best Mean Reward: {resume_results['best_mean_reward']:.4f}")\n            print(f"  - Evaluations: {len(resume_results['evaluations'])}")\n            print(f"  - Is Resumed: {resume_results['training_summary'].get('is_resumed_training', False)}")\n            print(f"  - Resumed From: {resume_results['training_summary'].get('resumed_from_model', 'N/A')}")\n            \n            # Test model loading\n            print("\\n" + "="*50)\n            print("STEP 3: Model Loading Test")\n            print("="*50)\n            \n            try:\n                from stable_baselines3 import PPO\n                from TradingEnv import TradingEnv\n                \n                # Create test environment\n                test_data = sample_data[:100]  # Small subset for testing\n                test_env = TradingEnv(data_array=test_data)\n                \n                # Load the resumed model\n                loaded_model = PPO.load(resume_results['best_model_path'])\n                loaded_model.set_env(test_env)\n                \n                # Test prediction\n                obs, _ = test_env.reset()\n                action, _ = loaded_model.predict(obs)\n                \n                print(f"✓ Model loaded successfully!")\n                print(f"✓ Model can make predictions: action = {action}")\n                \n            except Exception as e:\n                print(f"✗ Model loading test failed: {str(e)}")\n            \n            print("\\n" + "="*50)\n            print("RESUME TRAINING TEST COMPLETED SUCCESSFULLY!")\n            print("="*50)\n            \n        else:\n            print(f"✗ Resume training failed: {resume_results['error']}")\n    \n    else:\n        print(f"✗ Initial training failed: {initial_results['error']}")\n    \n    # Cleanup\n    try:\n        os.remove(data_path)\n        print(f"\\nCleaned up test data: {data_path}")\n    except:\n        pass\n\nif __name__ == "__main__":\n    test_resume_training()