#!/bin/bash
echo "🚀 Starting EVE Terminal - Installing dependencies..."

# Ensure pip is up to date
python -m pip install --upgrade pip

# Install requirements with error handling
echo "📦 Installing Python packages..."
pip install -r requirements.txt || echo "Warning: Some packages failed to install"

# Install essential packages individually
pip install flask requests psycopg2-binary gunicorn python-dotenv waitress

# Try to install replicate with newer version
pip install "replicate>=0.30.0" || echo "Warning: replicate not installed, some features may be disabled"

# Start the web application
echo "🌟 Starting EVE web interface..."
python web_run.py
