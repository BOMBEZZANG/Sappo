#!/usr/bin/env python3
"""
Test script to demonstrate the evaluation progress display
"""

import tkinter as tk
from main_integrated import SappoIntegratedGUI
import threading
import time

def simulate_training_progress(app):
    """Simulate training progress with fake evaluation data"""
    print("Starting simulated training progress...")
    
    # Simulate evaluation results every few seconds
    evaluation_data_list = [
        {
            "timestep": 10000,
            "mean_reward": -0.002,
            "std_reward": 0.015,
            "sharpe_ratio": -0.134,
            "total_return": -0.05,
            "max_drawdown": 0.08,
            "portfolio_stats": {
                "final_value": 9500.0,
                "trade_count": 15,
                "total_cost": 12.5
            }
        },
        {
            "timestep": 20000,
            "mean_reward": 0.001,
            "std_reward": 0.012,
            "sharpe_ratio": 0.083,
            "total_return": 0.02,
            "max_drawdown": 0.06,
            "portfolio_stats": {
                "final_value": 10200.0,
                "trade_count": 22,
                "total_cost": 18.7
            }
        },
        {
            "timestep": 30000,
            "mean_reward": 0.005,
            "std_reward": 0.018,
            "sharpe_ratio": 0.278,
            "total_return": 0.08,
            "max_drawdown": 0.04,
            "portfolio_stats": {
                "final_value": 10800.0,
                "trade_count": 31,
                "total_cost": 25.2
            }
        },
        {
            "timestep": 40000,
            "mean_reward": 0.008,
            "std_reward": 0.020,
            "sharpe_ratio": 0.400,
            "total_return": 0.12,
            "max_drawdown": 0.035,
            "portfolio_stats": {
                "final_value": 11200.0,
                "trade_count": 38,
                "total_cost": 31.8
            }
        },
        {
            "timestep": 50000,
            "mean_reward": 0.012,
            "std_reward": 0.022,
            "sharpe_ratio": 0.545,
            "total_return": 0.15,
            "max_drawdown": 0.03,
            "portfolio_stats": {
                "final_value": 11500.0,
                "trade_count": 45,
                "total_cost": 38.5
            }
        }
    ]
    
    for i, eval_data in enumerate(evaluation_data_list):
        time.sleep(3)  # Wait 3 seconds between updates
        print(f"Sending evaluation {i+1}/5: Step {eval_data['timestep']}")
        
        # Simulate training log message
        app.log_training_message(f"Validation Step {eval_data['timestep']}: "
                                f"Mean Reward: {eval_data['mean_reward']:.4f}, "
                                f"Sharpe: {eval_data['sharpe_ratio']:.4f}")
        
        # Send structured progress update
        app.update_evaluation_progress(eval_data)
    
    print("Simulation completed!")
    app.log_training_message("🎉 Training simulation completed!")

def main():
    """Main function to run the test"""
    root = tk.Tk()
    app = SappoIntegratedGUI(root)
    
    # Switch to the Training & Evaluation tab
    app.notebook.select(1)
    
    # Switch to the Training Progress sub-tab
    app.results_notebook.select(2)
    
    print("Sappo GUI loaded. Starting simulation in 5 seconds...")
    app.log_training_message("Demo mode: Will simulate training progress in 5 seconds...")
    
    # Start simulation after 5 seconds
    def start_simulation():
        time.sleep(5)
        simulate_training_progress(app)
    
    simulation_thread = threading.Thread(target=start_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    # Run the GUI
    root.mainloop()

if __name__ == "__main__":
    main()