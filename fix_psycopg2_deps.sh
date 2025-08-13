#!/bin/bash
# Enhanced fix script for psycopg2 dependency issues on Replit

echo "===================================================="
echo "  ENHANCED PSYCOPG2 DEPENDENCY FIX FOR REPLIT"
echo "===================================================="

# Create directories needed for psycopg2 installation
echo "Creating required directories..."
mkdir -p ~/.local/lib
mkdir -p ~/.local/include

# Try different approaches to install system dependencies
echo "Installing system dependencies (attempting multiple methods)..."

# 1. Try apt-get first
if command -v apt-get &> /dev/null; then
    echo "Using apt-get..."
    apt-get update || echo "apt-get update failed, continuing..."
    apt-get install -y --no-install-recommends \
        zlib1g zlib1g-dev \
        libpq-dev \
        gcc \
        python3-dev || echo "apt-get install failed, trying alternatives..."
fi

# 2. Try installing with nix-env for Replit environment
if command -v nix-env &> /dev/null; then
    echo "Using nix-env (Replit environment)..."
    nix-env -i zlib postgresql || echo "nix-env install failed, continuing..."
fi

echo "Attempting to install psycopg2-binary with special options..."

# Try multiple installation methods
echo "Method 1: Standard pip install with no cache..."
python -m pip install --no-cache-dir psycopg2-binary

# If that fails, try with --no-binary
if [ $? -ne 0 ]; then
    echo "Method 2: Trying with --no-binary flag..."
    python -m pip install --no-cache-dir --no-binary :all: psycopg2-binary
fi

# If that fails, try regular psycopg2
if [ $? -ne 0 ]; then
    echo "Method 3: Trying regular psycopg2..."
    python -m pip install --no-cache-dir psycopg2
fi

# Verify installation
echo "Checking installation..."
python -c "import psycopg2; print('✅ psycopg2 installation successful!')" || echo "❌ psycopg2 installation failed"

echo "===================================================="
echo "Dependency fix script completed. If psycopg2 is still not working,"
echo "your application will fall back to SQLite database."
echo "===================================================="
