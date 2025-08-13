#!/bin/bash
# Simplified deployment script for Replit

echo "🚀 EVE Terminal - Simplified Deployment"

# Try to install packages with fallback
pip install flask requests gunicorn python-dotenv || {
    echo "Warning: Basic package installation failed"
}

# Try to install additional packages (non-critical)
pip install psycopg2-binary waitress Pillow || {
    echo "Warning: Optional packages not installed"
}

# Try newer replicate version
pip install "replicate>=0.30.0" || {
    echo "Warning: Replicate not available - image generation disabled"
}

# Start with Python directly (more reliable than bash in some environments)
echo "🌟 Starting EVE Terminal..."
exec python web_run.py
