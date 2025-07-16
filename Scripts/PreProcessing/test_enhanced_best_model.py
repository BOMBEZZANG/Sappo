#!/usr/bin/env python3
"""
Test script for enhanced best model saving functionality
"""

import numpy as np
import os
import json
from datetime import datetime
from train import ValidationCallback
from TradingEnv import TradingEnv

def test_enhanced_best_model_logic():
    """Test the enhanced best model saving logic"""
    
    print("Testing Enhanced Best Model Saving Logic")
    print("=" * 50)
    
    # Create sample data for testing
    print("Creating test scenario...")
    n_samples = 200
    window_size = 50
    n_features = 20
    
    # Generate synthetic trading data
    np.random.seed(42)
    sample_data = np.random.randn(n_samples, window_size, n_features)
    
    # Create test environment
    test_env = TradingEnv(data_array=sample_data)
    
    # Mock model for testing
    class MockModel:
        def predict(self, obs, deterministic=True):
            return np.random.choice([0, 1, 2]), None
        
        def save(self, path):
            print(f"  [MOCK] Saving model to: {os.path.basename(path)}")
            # Create a dummy file to simulate saving
            with open(path, 'w') as f:
                f.write("dummy model")
    
    # Create validation callback
    callback = ValidationCallback(
        validation_env=test_env,
        eval_freq=1000,
        n_eval_episodes=3,
        best_model_save_path="/tmp/test_best_model.zip",
        log_callback=print
    )
    
    # Mock the model
    callback.model = MockModel()
    callback.n_calls = 0
    
    # Test scenarios with different metric combinations
    test_scenarios = [
        # Scenario 1: First model (should be saved)
        {
            "name": "First Model",
            "sharpe": 0.5,
            "mean_reward": 0.1,
            "final_value": 10500,
            "timestep": 10000,
            "expected_save": True
        },
        # Scenario 2: Better Sharpe (should be saved)
        {
            "name": "Better Sharpe",
            "sharpe": 0.7,
            "mean_reward": 0.08,
            "final_value": 10300,
            "expected_save": True
        },
        # Scenario 3: Same Sharpe, better mean reward (should be saved)
        {
            "name": "Same Sharpe, Better Mean Reward",
            "sharpe": 0.7,
            "mean_reward": 0.12,
            "final_value": 10200,
            "expected_save": True
        },
        # Scenario 4: Same Sharpe and mean reward, better final value (should be saved)
        {
            "name": "Same Sharpe/Reward, Better Final Value",
            "sharpe": 0.7,
            "mean_reward": 0.12,
            "final_value": 10800,
            "expected_save": True
        },
        # Scenario 5: Worse on all metrics (should not be saved)
        {
            "name": "Worse on All Metrics",
            "sharpe": 0.6,
            "mean_reward": 0.10,
            "final_value": 10600,
            "expected_save": False
        },
        # Scenario 6: Same Sharpe, worse mean reward (should not be saved)
        {
            "name": "Same Sharpe, Worse Mean Reward",
            "sharpe": 0.7,
            "mean_reward": 0.11,
            "final_value": 11000,
            "expected_save": False
        },
        # Scenario 7: Better Sharpe (should be saved)
        {
            "name": "Much Better Sharpe",
            "sharpe": 0.9,
            "mean_reward": 0.05,
            "final_value": 9500,
            "expected_save": True
        }
    ]
    
    print("\\nTesting multi-level comparison logic...")
    print("-" * 50)
    
    for i, scenario in enumerate(test_scenarios):
        callback.n_calls = (i + 1) * 10000
        
        print(f"\\nTest {i+1}: {scenario['name']}")
        print(f"  Metrics: Sharpe={scenario['sharpe']:.3f}, Mean Reward={scenario['mean_reward']:.3f}, Final Value=${scenario['final_value']:,}")
        
        # Test the comparison logic
        is_new_best = callback._is_new_best_model(
            scenario['sharpe'],
            scenario['mean_reward'],
            scenario['final_value']
        )
        
        print(f"  Current Best: Sharpe={callback.best_sharpe_ratio:.3f}, Mean Reward={callback.best_mean_reward:.3f}, Final Value=${callback.best_final_value:,}")
        print(f"  Is New Best: {is_new_best} (Expected: {scenario['expected_save']})\")\n        \n        if is_new_best == scenario['expected_save']:\n            print(f\"  ✓ PASSED\")\n            \n            # Update best values if it's truly better\n            if is_new_best:\n                callback.best_sharpe_ratio = scenario['sharpe']\n                callback.best_mean_reward = scenario['mean_reward']\n                callback.best_final_value = scenario['final_value']\n                callback.best_model_timestep = callback.n_calls\n                \n                # Test filename generation\n                filename = callback._generate_best_model_filename()\n                print(f\"  Generated filename: {os.path.basename(filename)}\")\n                \n                # Verify filename format\n                expected_pattern = f\"best_model_step_{callback.n_calls}\"\n                if expected_pattern in filename:\n                    print(f\"  ✓ Filename format correct\")\n                else:\n                    print(f\"  ✗ Filename format incorrect\")\n        else:\n            print(f\"  ✗ FAILED - Expected {scenario['expected_save']}, got {is_new_best}\")\n    \n    print(\"\\n\" + \"=\" * 50)\n    print(\"TESTING FILENAME GENERATION\")\n    print(\"=\" * 50)\n    \n    # Test filename generation for different scenarios\n    callback.best_model_timestep = 45000\n    \n    # Test normal training\n    callback.best_model_save_path = \"/tmp/best_model_20250716_123456.zip\"\n    normal_filename = callback._generate_best_model_filename()\n    print(f\"Normal training filename: {os.path.basename(normal_filename)}\")\n    \n    # Test resumed training\n    callback.best_model_save_path = \"/tmp/resumed_best_model_20250716_123456.zip\"\n    resumed_filename = callback._generate_best_model_filename()\n    print(f\"Resumed training filename: {os.path.basename(resumed_filename)}\")\n    \n    # Verify format\n    if \"step_45000\" in normal_filename and \"step_45000\" in resumed_filename:\n        print(\"✓ Timestep correctly included in both filenames\")\n    else:\n        print(\"✗ Timestep not correctly included\")\n        \n    if \"resumed_best_model_step\" in resumed_filename:\n        print(\"✓ Resumed prefix correctly maintained\")\n    else:\n        print(\"✗ Resumed prefix not maintained\")\n    \n    print(\"\\n\" + \"=\" * 50)\n    print(\"ENHANCED BEST MODEL TESTING COMPLETED\")\n    print(\"=\" * 50)\n    \n    # Cleanup\n    for path in [\"/tmp/test_best_model.zip\", normal_filename, resumed_filename]:\n        try:\n            if os.path.exists(path):\n                os.remove(path)\n        except:\n            pass\n\nif __name__ == \"__main__\":\n    test_enhanced_best_model_logic()