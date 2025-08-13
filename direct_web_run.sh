#!/bin/bash
# Direct web_run.py launcher with deployment fixes

echo "========================================="
echo "  DIRECT WEB_RUN.PY LAUNCHER (ENHANCED)"
echo "========================================="
echo "$(date)"
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la

# Ensure dependencies are installed
echo "Installing dependencies..."
pip install flask requests psycopg2-binary python-dotenv waitress

# Debug information
echo "Python path:"
which python3
echo "Python version:"
python3 --version

# Start the server
echo "Starting EVE Terminal web interface..."
python3 web_run.py

# CRUCIAL: Run web_run.py DIRECTLY with explicit python3
echo "========================================"
echo "LAUNCHING WEB_RUN.PY DIRECTLY"
echo "========================================"

exec python3 -u web_run.py
