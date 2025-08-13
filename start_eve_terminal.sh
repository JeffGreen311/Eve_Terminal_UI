#!/bin/bash
# Eve Terminal startup script for Replit deployment

echo "==============================================="
echo "   EVE Terminal - Replit Deployment Script"
echo "==============================================="
echo "Starting setup at $(date)"

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install requests flask psycopg2-binary gunicorn

# Verify critical packages
echo "Verifying critical packages..."
python -c "
try:
    import requests, flask, psycopg2
    print('All critical packages verified!')
except ImportError as e:
    print(f'ERROR: {e}')
    print('Installing missing package...')
    import subprocess, sys
    package = str(e).split(\"'\")[1]
    if package == 'psycopg2':
        package = 'psycopg2-binary'
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    print(f'Installed {package}')
"

# Start the web application
echo "Starting EVE Terminal web interface..."
echo "==============================================="

# Use the new web_run_new.py file
python web_run_new.py
