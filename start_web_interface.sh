#!/bin/bash
# Entrypoint script for EVE Web Interface

# Print Python version and path
python --version
which python
echo "Python executable: $(which python)"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Ensure the web interface has all required packages
pip install flask requests psycopg2-binary gunicorn

# Print installed packages
echo "Installed packages:"
pip list

# Start the application directly for development
echo "Starting EVE Web Interface..."
python web_run.py
