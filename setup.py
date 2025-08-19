#!/usr/bin/env python3
"""
Setup script for YouTube Transcript Summarizer
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("Setting up YouTube Transcript Summarizer...")
    print("=" * 50)
    
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Consider creating one: python -m venv venv")
        print("   Then activate it and run this script again")
        print()
    
    print("Installing required packages...")
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
        print("Failed to install packages. Please check your internet connection and try again.")
        return False
    
    # Install spaCy model
    print("\nInstalling spaCy English model...")
    if not run_command(f"{sys.executable} -m spacy download en_core_web_sm"):
        print("Failed to install spaCy model. You can still use NLTK summarization.")
    
    print("\n" + "=" * 50)
    print("Setup complete! You can now run:")
    print("streamlit run app.py")
    
    return True

if __name__ == "__main__":
    main()
