#!/usr/bin/env python3
"""
EVE Health Check
A simple health check app for deployment verification
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def health_check():
    """Simple health check endpoint for deployment verification"""
    return jsonify({
        'status': 'healthy',
        'service': 'eve-terminal',
        'message': 'Health check server is running'
    }), 200

@app.route('/ready')
def readiness():
    """Readiness probe endpoint"""
    return jsonify({
        'status': 'ready',
        'service': 'eve-terminal'
    }), 200

@app.route('/live')
def liveness():
    """Liveness probe endpoint"""
    return jsonify({
        'status': 'alive',
        'service': 'eve-terminal'
    }), 200

if __name__ == '__main__':
    # Use waitress for production server
    try:
        from waitress import serve
        print("Starting health check server with Waitress...")
        serve(app, host='0.0.0.0', port=8080)
    except ImportError:
        print("Waitress not found, falling back to Flask server")
        app.run(host='0.0.0.0', port=8080, debug=False)
