#!/usr/bin/env python
"""
Enhanced psycopg2 dependency fix script for EVE Terminal
Works across all platforms with special handling for Replit environments
"""

import os
import sys
import subprocess
import platform
import importlib.util

def run_command(command, shell=True):
    """Run a command and return its output"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def fix_replit_environment():
    """Fix the psycopg2 dependency issue on Replit"""
    print("=== Fixing psycopg2 dependencies on Replit ===")
    
    # Create directories needed for psycopg2 installation
    print("Creating required directories...")
    os.makedirs(os.path.expanduser("~/.local/lib"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/.local/include"), exist_ok=True)
    
    # Check if we have apt-get (Linux/Replit)
    if run_command("which apt-get", shell=True):
        print("Installing system dependencies with apt-get...")
        run_command("apt-get update")
        run_command("apt-get install -y --no-install-recommends zlib1g zlib1g-dev libpq-dev gcc python3-dev")
    else:
        print("apt-get not found - trying nix-env...")
        run_command("which nix-env && nix-env -i zlib postgresql")
    
    # Update the .replit file if it exists
    if os.path.exists(".replit"):
        print("Updating .replit file...")
        with open(".replit", "r") as f:
            content = f.read()
        
        if "[nix]" in content:
            # Make sure zlib is in the packages
            if "packages =" in content and "zlib" not in content:
                content = content.replace(
                    "packages =", 
                    'packages = ["zlib", "zlib.dev", "postgresql", '
                )
                # Fix the end of the packages line if needed
                content = content.replace('[", ', '["')
                content = content.replace('"]', ', "]')
                
                with open(".replit", "w") as f:
                    f.write(content)
                print("Added zlib and postgresql to .replit packages")
    
    # Install psycopg2 using multiple methods
    print("Trying multiple psycopg2 installation methods...")
    
    # Method 1: Standard install with no cache
    print("\nMethod 1: Standard install with no cache")
    success1 = run_command(f"{sys.executable} -m pip install --no-cache-dir psycopg2-binary")
    
    # If Method 1 fails, try Method 2
    if not success1:
        print("\nMethod 2: Source install with no binary")
        success2 = run_command(f"{sys.executable} -m pip install --no-cache-dir --no-binary :all: psycopg2-binary")
        
        # If Method 2 fails, try Method 3
        if not success2:
            print("\nMethod 3: Regular psycopg2")
            run_command(f"{sys.executable} -m pip install --no-cache-dir psycopg2")

def fix_windows_environment():
    """Fix the psycopg2 dependency issue on Windows"""
    print("=== Fixing psycopg2 dependencies on Windows ===")
    
    # Install psycopg2-binary
    print("Installing psycopg2-binary...")
    run_command(f"{sys.executable} -m pip uninstall -y psycopg2 psycopg2-binary")
    run_command(f"{sys.executable} -m pip install --upgrade psycopg2-binary==2.9.9")

def test_psycopg2():
    """Test if psycopg2 imports correctly"""
    print("\n=== Testing psycopg2 import ===")
    
    # Try direct import first
    try:
        import psycopg2
        print(f"Direct import successful: psycopg2 version {psycopg2.__version__}")
        print("✅ SUCCESS: psycopg2 is working correctly!")
        return True
    except ImportError:
        print("Direct import failed, trying as a subprocess...")
    except Exception as e:
        print(f"Error during direct import: {str(e)}")
    
    # Try as a subprocess if direct import fails
    test_code = """
import psycopg2
print("psycopg2 version:", psycopg2.__version__)
print("psycopg2 successfully imported!")
"""
    
    # Write test to a file
    with open("test_psycopg2_temp.py", "w") as f:
        f.write(test_code)
    
    # Run the test
    success = run_command(f"{sys.executable} test_psycopg2_temp.py")
    
    if success:
        print("\n✅ SUCCESS: psycopg2 is working correctly!")
        return True
    else:
        print("\n❌ ERROR: psycopg2 still has issues.")
        
        if platform.system() == "Windows":
            print("Windows might need additional dependencies installed.")
            print("Try installing PostgreSQL client tools locally.")
        else:
            print("Replit environment might need additional configuration.")
            print("The application will fall back to SQLite if needed.")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("EVE Terminal - Enhanced psycopg2 Fix Script")
    print("=" * 60)
    
    system = platform.system()
    print(f"Detected system: {system}")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check for environment variables
    print("\nChecking database environment variables:")
    env_vars = ["DATABASE_URL", "PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD", "PGPORT"]
    for var in env_vars:
        if var in os.environ:
            # Mask passwords
            if var == "PGPASSWORD" or var == "DATABASE_URL":
                print(f"  {var}: [REDACTED]")
            else:
                print(f"  {var}: {os.environ[var]}")
        else:
            print(f"  {var}: Not set")
    
    if system == "Windows":
        fix_windows_environment()
    else:
        fix_replit_environment()
    
    psycopg2_working = test_psycopg2()
    
    print("\n" + "=" * 60)
    if psycopg2_working:
        print("✅ psycopg2 installation successful!")
        print("The EVE Terminal application should now be able to use PostgreSQL.")
    else:
        print("⚠️ psycopg2 installation had issues.")
        print("The EVE Terminal application will fall back to SQLite.")
        print("This is normal for some deployment environments.")
    print("=" * 60)

if __name__ == "__main__":
    main()
