#!/bin/bash
# Simple direct entrypoint for Replit deployments
# This bypasses Gunicorn and directly runs app.py with Python

echo "===== EVE Terminal Direct Entrypoint ====="
echo "Starting at: $(date)"
echo "Current directory: $(pwd)"
echo "Listing files: $(ls -la)"

# Make sure all dependencies are installed
echo "Ensuring dependencies are installed..."
pip install --no-cache-dir -r requirements.txt

# Export the port for Flask to use
export PORT=8888
export PYTHONUNBUFFERED=1

# Run app.py directly with Python
echo "Starting Flask app directly..."
exec python app.py
