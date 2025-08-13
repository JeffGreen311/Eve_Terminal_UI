#!/usr/bin/env python3
"""
EVE Terminal - Clean Entrypoint for Replit Deployment
Preserves FLUX Dev-1 Image Generation with REPLICATE_API_TOKEN
"""

import os
import sys

# Environment validation
def check_environment():
    """Check critical environment variables and provide startup info"""
    print("🌟 EVE Terminal - Clean Deployment Entrypoint")
    print("=" * 50)
    
    # Check REPLICATE_API_TOKEN for FLUX image generation
    replicate_token = os.environ.get('REPLICATE_API_TOKEN')
    if replicate_token:
        print(f"✅ REPLICATE_API_TOKEN: {'*' * 20}{replicate_token[-8:]}")
        print("✅ FLUX Dev-1 Image Generation: ENABLED")
    else:
        print("⚠️  REPLICATE_API_TOKEN: Not set")
        print("⚠️  FLUX Dev-1 Image Generation: DISABLED")
    
    # Check available packages
    available_packages = []
    try:
        import flask
        available_packages.append(f"flask=={flask.__version__}")
    except ImportError:
        print("❌ flask not available")
    
    try:
        import requests
        available_packages.append(f"requests=={requests.__version__}")
    except ImportError:
        print("❌ requests not available")
    
    try:
        import replicate
        available_packages.append(f"replicate=={replicate.__version__}")
        print("✅ Replicate package: Available for FLUX generation")
    except ImportError:
        print("⚠️  replicate package not available - FLUX generation may fail")
    
    if available_packages:
        print(f"📦 Available packages: {', '.join(available_packages)}")
    
    print("=" * 50)

if __name__ == "__main__":
    check_environment()
    
    # Import and run the main web application
    try:
        print("🚀 Starting EVE Terminal with FLUX capabilities...")
        from web_run import app
        
        # Get port from environment (Replit compatibility)
        port = int(os.environ.get('PORT', 8080))
        
        # Run the application
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except ImportError as e:
        print(f"❌ Failed to import web_run: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start EVE Terminal: {e}")
        sys.exit(1)
