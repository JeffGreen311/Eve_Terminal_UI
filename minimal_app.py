#!/usr/bin/env python
"""
Ultra Minimal EVE Terminal App for Replit Recovery Mode
"""

import os
import sys
import datetime

# Create a basic Flask app with minimal dependencies
try:
    from flask import Flask, jsonify
    print("✓ Flask successfully imported")
    
    app = Flask(__name__)
    
    @app.route('/')
    def health_check():
        """Simple health check endpoint for deployment verification"""
        return jsonify({
            'status': 'ok',
            'service': 'eve-terminal',
            'mode': 'minimal',
            'timestamp': datetime.datetime.now().isoformat()
        }), 200
    
    @app.route('/health')
    def health():
        """Simple health check endpoint for deployment"""
        return jsonify({
            'status': 'healthy',
            'service': 'eve-terminal',
            'mode': 'minimal'
        }), 200
        
    @app.route('/status')
    def status():
        """Status endpoint for monitoring"""
        return jsonify({
            "status": "online", 
            "mode": "minimal-recovery", 
            "time": str(datetime.datetime.now()),
            "python_version": sys.version,
            "env_vars": {
                "PORT": os.environ.get("PORT", "not set"),
                "HOST": os.environ.get("HOST", "not set"),
                "REPLICATE_API_TOKEN": "present" if os.environ.get("REPLICATE_API_TOKEN") else "missing"
            }
        })
        
    @app.route('/ui')
    def ui():
        """Simple UI that doesn't require heavy dependencies"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>EVE Terminal - Minimal Mode</title>
            <style>
                body { 
                    background-color: #000; 
                    color: #0f0; 
                    font-family: monospace;
                    padding: 20px;
                }
                h1 { color: #0f0; }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    border: 1px solid #0f0;
                }
                .status { color: #ff0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>EVE Terminal - Minimal Mode</h1>
                <p>The EVE Terminal system is running in minimal recovery mode.</p>
                <p>This lightweight version is designed for deployment troubleshooting.</p>
                <p class="status">Status: <strong>Online</strong></p>
                <p>Environment: Replit Nix-based deployment</p>
                <p>REPLICATE_API_TOKEN: """ + ("✅ Present" if os.environ.get("REPLICATE_API_TOKEN") else "❌ Missing") + """</p>
            </div>
        </body>
        </html>
        """
        
except ImportError:
    # If Flask isn't available, create a minimal HTTP server with Python standard library
    print("⚠ Flask not available - creating minimal HTTP server")
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json
    
    class MinimalHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            """Handle GET requests with minimal responses"""
            if self.path == '/' or self.path == '/health':
                # Return simple JSON for health checks
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    'status': 'ok',
                    'service': 'eve-terminal',
                    'mode': 'emergency-minimal',
                    'timestamp': datetime.datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                # For any other path, return simple HTML
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>EVE Terminal - Emergency Mode</title>
                    <style>
                        body { background-color: #000; color: #0f0; font-family: monospace; padding: 20px; }
                        h1 { color: #0f0; }
                        .container { max-width: 800px; margin: 0 auto; padding: 20px; border: 1px solid #0f0; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>EVE Terminal - Emergency Mode</h1>
                        <p>Running with Python's built-in HTTP server (no Flask).</p>
                        <p>This is an ultra-minimal version to recover from deployment issues.</p>
                        <p>Dependencies are being resolved...</p>
                    </div>
                </body>
                </html>
                """)
    
    def run_minimal_server(port=8080):
        """Run a minimal HTTP server for emergency recovery"""
        server_address = ('', port)
        httpd = HTTPServer(server_address, MinimalHandler)
        print(f"Starting minimal HTTP server on port {port}")
        httpd.serve_forever()
    
    # Run the minimal server if this file is executed directly
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8080))
        run_minimal_server(port)
    
    # Exit the script - the code below will only run if Flask is available
    sys.exit(0)

# This section only runs if Flask is available
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting minimal Flask app on {host}:{port}")
    app.run(host=host, port=port, debug=False)
