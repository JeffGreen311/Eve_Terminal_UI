#!/usr/bin/env python
"""
Import Wrapper for EVE Terminal
This wrapper ensures all dependencies are properly loaded
"""

import sys
import os
import subprocess
import importlib.util
import time

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

print("EVE Terminal Import Wrapper")
print("===========================")
print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Required packages
REQUIRED_PACKAGES = [
    "flask",
    "requests",
    "psycopg2-binary",
    "gunicorn",
    "python-dotenv",
    "spacy",
    "nltk",
    "numpy",
    "pandas"
]

def check_and_install_package(package_name):
    """Check if a package is installed and install it if not"""
    package_to_import = package_name.split("==")[0]  # Remove version if present
    
    if package_to_import == "psycopg2-binary":
        package_to_import = "psycopg2"
    
    try:
        spec = importlib.util.find_spec(package_to_import)
        if spec is None:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Successfully installed {package_name}")
        else:
            print(f"✓ {package_to_import} is already installed")
        return True
    except Exception as e:
        print(f"Error with {package_name}: {str(e)}")
        return False

# Check and install required packages
print("\nChecking required packages...")
for package in REQUIRED_PACKAGES:
    check_and_install_package(package)

# Now import flask and create a basic app in case the main import fails
try:
    import flask
    fallback_app = flask.Flask(__name__)
    
    @fallback_app.route('/')
    def fallback_index():
        return "EVE Terminal is starting up. The main application couldn't be loaded."
except Exception as e:
    print(f"Error creating fallback app: {str(e)}")
    fallback_app = None

# Try to import the main app
print("\nImporting main application...")
try:
    from main import app
    print("✓ Successfully imported main application")
except Exception as e:
    print(f"Error importing main application: {str(e)}")
    
    if fallback_app:
        print("Using fallback application")
        app = fallback_app
    else:
        print("Fatal error: Cannot create a fallback application")
        sys.exit(1)

# If running as script, start the server
if __name__ == "__main__":
    print("\nStarting application server...")
    try:
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8888)))
    except Exception as e:
        print(f"Error starting application server: {str(e)}")
        sys.exit(1)
