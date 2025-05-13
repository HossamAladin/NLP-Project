#!/usr/bin/env python
"""
Arabic Autocorrect GUI Launcher
This script launches the Arabic Autocorrect GUI application.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import torch
        import transformers
        import pandas
        import tkinter
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")

def main():
    """Main function to run the GUI"""
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check dependencies
    check_dependencies()
    
    # Launch the GUI
    try:
        from gui import ArabicAutocorrectApp
        import tkinter as tk
        
        root = tk.Tk()
        app = ArabicAutocorrectApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 