
import os
from flask import Flask, jsonify

app = Flask(__name__)

# Health check endpoint for Railway
@app.route('/health')
def health_check():
    return jsonify(status="ok"), 200

# Lazy-load heavy modules after app starts
def init_heavy_components():
    global torch, replicate
    try:
        import torch
        import replicate
    except ImportError as e:
        print(f"[WARN] Heavy component not available: {e}")

@app.before_first_request
def startup_tasks():
    import threading
    threading.Thread(target=init_heavy_components).start()

# Import the rest of Eve's UI routes and handlers
from eve_ui_routes import *  # Assuming your original file imports routes

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
