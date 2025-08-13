#!/bin/bash
# Entrypoint script for EVE Terminal
# This script is executed when your Replit deployment starts

echo "===== EVE Terminal Entrypoint ====="
echo "Starting at: $(date)"
echo "Current directory: $(pwd)"
echo "Listing files: $(ls -la)"

# Make sure all dependencies are installed
echo "Ensuring dependencies are installed..."
pip install --no-cache-dir -r requirements.txt

# Set environment variables
export PORT=8888
export PYTHONUNBUFFERED=1

echo "Python executable: $(which python)"
echo "Gunicorn executable: $(which gunicorn)"

# Start the application with explicit path to Python
echo "Starting Gunicorn server..."
gunicorn --bind 0.0.0.0:8888 --timeout 60 --workers 1 --log-level debug app:app
