#!/usr/bin/env python3
"""
Sappo Trading Bot - Integrated ML Pipeline Launcher
"""

import sys
import os

def check_basic_dependencies():
    """Check if basic dependencies are installed"""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append('scikit-learn')
    
    try:
        import tkinter
    except ImportError:
        missing_deps.append('tkinter')
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append('matplotlib')
    
    return missing_deps

def check_rl_dependencies():
    """Check if RL dependencies are installed"""
    missing_rl = []
    
    try:
        import gymnasium
    except ImportError:
        missing_rl.append('gymnasium')
    
    try:
        import stable_baselines3
    except ImportError:
        missing_rl.append('stable-baselines3')
    
    try:
        import torch
    except ImportError:
        missing_rl.append('torch')
    
    try:
        import tensorboard
    except ImportError:
        missing_rl.append('tensorboard')
    
    return missing_rl

def main():
    print("Sappo Trading Bot - Integrated ML Pipeline")
    print("=" * 55)
    
    # Check basic dependencies
    missing_basic = check_basic_dependencies()
    if missing_basic:
        print("❌ Missing basic dependencies:")
        for dep in missing_basic:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies:")
        print("pip install pandas numpy scikit-learn matplotlib")
        print("\nNote: tkinter should be included with Python")
        sys.exit(1)
    
    print("✅ Basic dependencies found")
    
    # Check RL dependencies
    missing_rl = check_rl_dependencies()
    if missing_rl:
        print("⚠️  Missing RL dependencies (training will be disabled):")
        for dep in missing_rl:
            print(f"   - {dep}")
        print("\nTo enable RL training, install:")
        print("pip install gymnasium stable-baselines3 torch tensorboard")
        print("\nContinuing with preprocessing-only mode...")
    else:
        print("✅ RL dependencies found")
    
    print("🚀 Starting Sappo Integrated GUI...")
    print()
    
    # Import and run the integrated application
    try:
        # Add current directory to Python path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from main_integrated import main as run_gui
        run_gui()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all files are in the Scripts/PreProcessing directory")
        print("2. Check that all required files exist:")
        print("   - main_integrated.py")
        print("   - preprocessing.py")
        print("   - TradingEnv.py (for RL)")
        print("   - agent.py (for RL)")
        print("   - train.py (for RL)")
        print("   - evaluate.py (for RL)")
        print("3. Verify Python version (3.7+ recommended)")
        sys.exit(1)

if __name__ == "__main__":
    main()