#!/usr/bin/env python3
"""
Emergency deployment script - tries multiple approaches for maximum success
"""
import os
import sys
import subprocess

def try_run_script(script_name):
    """Try to run a Python script and return success status"""
    try:
        if os.path.exists(script_name):
            print(f"🔄 Attempting to run {script_name}...")
            subprocess.run([sys.executable, script_name], check=True)
            return True
    except Exception as e:
        print(f"❌ Failed to run {script_name}: {e}")
    return False

def main():
    print("🚨 EVE EMERGENCY DEPLOYMENT SYSTEM 🚨")
    print("Trying multiple deployment approaches...")
    
    # Set basic environment variables
    os.environ.setdefault('PORT', '8080')
    os.environ.setdefault('HOST', '0.0.0.0')
    
    # Try approaches in order of preference
    approaches = [
        "start_eve.py",     # Our enhanced startup script
        "web_run.py",       # Main application
        "minimal_app.py",   # Lightweight fallback
        "main.py"           # Original fallback
    ]
    
    for approach in approaches:
        print(f"\n{'='*50}")
        print(f"TRYING: {approach}")
        print(f"{'='*50}")
        
        if try_run_script(approach):
            print(f"✅ SUCCESS: {approach} is running!")
            break
        else:
            print(f"⚠️  FAILED: {approach} did not work, trying next...")
    else:
        # If nothing works, try to at least start a basic HTTP server
        print(f"\n{'='*50}")
        print("EMERGENCY MODE: Starting basic Python HTTP server")
        print(f"{'='*50}")
        
        try:
            from http.server import HTTPServer, SimpleHTTPRequestHandler
            
            class EmergencyHandler(SimpleHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status":"emergency","message":"EVE Terminal in emergency mode"}')
            
            port = int(os.environ.get('PORT', 8080))
            server = HTTPServer(('0.0.0.0', port), EmergencyHandler)
            print(f"🆘 Emergency server running on port {port}")
            server.serve_forever()
            
        except Exception as e:
            print(f"💥 COMPLETE FAILURE: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
