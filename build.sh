#!/bin/bash
# Build script for EVE Terminal deployment

echo "===== Starting EVE Terminal Build Process ====="
echo "Build started at: $(date)"

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Use Replit package manager as fallback
echo "Installing dependencies using UPM (Replit package manager)..."
if command -v upm &> /dev/null; then
    upm add requests psycopg2-binary flask gunicorn python-dotenv
else
    echo "UPM not available, skipping..."
fi

# Verify installations
echo "Verifying critical package installations..."
python -c "
import sys
try:
    import requests
    import flask
    import psycopg2
    print('All critical packages verified!')
except ImportError as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"

# Create necessary directories if they don't exist
echo "Creating necessary directories..."
mkdir -p static
mkdir -p templates

echo "Build completed at: $(date)"
echo "===== EVE Terminal Build Process Completed ====="

# Exit with success
exit 0
