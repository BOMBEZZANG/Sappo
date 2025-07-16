#!/usr/bin/env python3
"""
Sappo Trading Bot - Data Preprocessing Tool Launcher
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
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
    
    return missing_deps

def main():
    print("Sappo Trading Bot - Data Preprocessing Tool")
    print("=" * 50)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies:")
        print("pip install pandas numpy scikit-learn")
        print("\nNote: tkinter should be included with Python")
        sys.exit(1)
    
    print("✅ All dependencies found")
    print("🚀 Starting Sappo Preprocessing GUI...")
    print()
    
    # Import and run the main application
    try:
        # Add current directory to Python path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from main import main as run_gui
        run_gui()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\nTroubleshooting:")
        print("1. Run from Scripts/PreProcessing directory: cd Scripts/PreProcessing && python main.py")
        print("2. Or run from project root: python Scripts/PreProcessing/main.py")
        print("3. Check that main.py and preprocessing.py exist")
        print("4. Verify Python version (3.7+ recommended)")
        sys.exit(1)

if __name__ == "__main__":
    main()