#!/bin/bash
# Simple deployment script for Replit

echo "=== EVE Terminal Simple Deployment ==="
echo "Starting deployment at $(date)"

# Install dependencies
echo "Installing essential dependencies..."
pip install flask gunicorn

# Verify Flask is installed
echo "Verifying Flask installation..."
python -c "import flask; print(f'Flask {flask.__version__} is installed')"

echo "=== Deployment setup complete ==="
echo "You can now use this in your .replit file:"
echo "run = \"python minimal_app.py\""
echo "And for deployment:"
echo "[deployment]"
echo "run = [\"sh\", \"-c\", \"pip install flask gunicorn && gunicorn --bind 0.0.0.0:8888 minimal_app:app\"]"
echo "===============================
