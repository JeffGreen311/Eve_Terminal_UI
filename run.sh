#!/bin/bash
# Run script for EVE Terminal web interface

# Debug information
echo "==============================================================="
echo "EVE Terminal - Deployment Script"
echo "==============================================================="
echo "Current directory: $(pwd)"
echo "PATH: $PATH"
echo "System info: $(uname -a)"
echo "User: $(whoami)"

# Python setup
echo "Setting up Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Python not found! Attempting to install Python..."
    apt-get update && apt-get install -y python3 python3-pip || echo "Failed to install Python"
    PYTHON=python3
fi

# Add Python to PATH if not already there
export PATH=$PATH:$(dirname $(which $PYTHON 2>/dev/null) || echo "/usr/bin")

echo "$PYTHON version:"
$PYTHON --version || echo "Failed to get Python version"
echo "$PYTHON executable: $(which $PYTHON 2>/dev/null || echo 'Not found')"

# Install only essential dependencies (lightweight deployment)
echo "Installing essential dependencies..."
$PYTHON -m pip install --no-cache-dir flask requests gunicorn || echo "Warning: Failed to install all dependencies"

# Try to install psycopg2-binary with fallbacks
echo "Installing database dependencies..."
$PYTHON -m pip install --no-cache-dir psycopg2-binary || \
$PYTHON -m pip install --no-cache-dir psycopg2 || \
echo "Warning: Failed to install PostgreSQL adapter. Will use SQLite fallback."

# Print installed packages
echo "Installed packages:"
$PYTHON -m pip list

# Add current directory to Python path
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Prefer full cosmic interface (web_run.py). Fallback to app.py only if it fails.
echo "Starting EVE cosmic interface (web_run.py)..."
if [ -n "$PORT" ]; then
    echo "Using PORT from environment: $PORT"
else
    export PORT=8080
    echo "Setting default PORT: $PORT"
fi

# Try gunicorn first (best for production)
echo "Starting with gunicorn (web_run:app)..."
if command -v gunicorn &> /dev/null; then
    # Try serving the Flask 'app' from web_run module
    gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 2 --preload web_run:app && exit 0 || echo "Gunicorn web_run failed"
fi

echo "Falling back to direct Python execution (web_run.py)..."
$PYTHON web_run.py && exit 0 || echo "Direct execution of web_run.py failed"

echo "Falling back to legacy app.py launcher..."
if command -v gunicorn &> /dev/null; then
    gunicorn --bind 0.0.0.0:$PORT --timeout 60 --workers 1 --threads 2 --preload app:app || echo "Gunicorn app.py failed"
fi

echo "Using direct Python for app.py as last standard fallback..."
$PYTHON app.py || echo "Direct Python execution of app.py failed"

# If all else fails, use waitress
echo "Trying waitress as last resort..."
$PYTHON -m pip install --no-cache-dir waitress || echo "Failed to install waitress"
exec $PYTHON -c "from waitress import serve; from app import app; serve(app, host='0.0.0.0', port=$PORT)" || echo "Waitress failed to start"

# Last resort - direct Flask execution
echo "All server methods failed, using Flask development server..."
exec $PYTHON -c "from app import app; app.run(host='0.0.0.0', port=$PORT, debug=False)"
