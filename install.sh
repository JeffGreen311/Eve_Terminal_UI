#!/bin/bash
# Install script for EVE Terminal dependencies

echo "=== Installing EVE Terminal dependencies ==="

# Install using pip
echo "Installing via pip..."
pip install -r requirements.txt

# Try UPM as a fallback (Replit package manager)
echo "Installing via UPM (Replit package manager)..."
upm add requests psycopg2-binary flask gunicorn python-dotenv

echo "=== Verifying installations ==="

# Verify installations
python -c "
try:
    import requests
    print('✓ requests installed')
except ImportError:
    print('✗ requests FAILED')

try:
    import flask
    print('✓ flask installed')
except ImportError:
    print('✗ flask FAILED')

try:
    import psycopg2
    print('✓ psycopg2 installed')
except ImportError:
    print('✗ psycopg2 FAILED')

try:
    import gunicorn
    print('✓ gunicorn installed')
except ImportError:
    print('✗ gunicorn FAILED')
"

echo "=== Installation complete ==="
