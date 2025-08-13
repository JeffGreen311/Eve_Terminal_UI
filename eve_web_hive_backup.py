#!/usr/bin/env python3
"""
EVE WEB INTERFACE - Hive Mind Gateway
Hybrid Database System: PostgreSQL + SQLite + HTTP API
Web UI for the Eve Terminal hive mind system
"""

import psycopg2
import sqlite3
import requests
import json
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

# Hybrid Hive Mind Configuration (same as other components)
POSTGRES_CONFIG = {
    'host': 'ep-jolly-fire-af5qkfza.c-2.us-west-2.aws.neon.tech',
    'database': 'neondb',
    'user': 'neondb_owner',
    'password': 'FfPpGE7LlZ4e',
    'port': 5432,
    'sslmode': 'require'
}

LOCAL_DB_PATH = './eve_web_local.db'
ADAM_DAEMON_URL = 'http://localhost:5001'  # Adam Daemon
EVE_TERMINAL_URL = 'http://localhost:5002'  # Eve Terminal

app = Flask(__name__)

# Web Interface State
web_state = {
    "active_users": 0,
    "session_count": 0,
    "hive_connections": {
        "postgresql": False,
        "sqlite": False,
        "adam_daemon": False,
        "eve_terminal": False
    }
}

# ====== Hive Mind Web Sync ======
def web_hive_sync(message, source="Eve Web", user_id="web_user"):
    """Sync web interactions to all hive systems"""
    timestamp = datetime.utcnow().isoformat()
    
    # 1. PostgreSQL (Primary Hive Mind)
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eve_autobiographical_memory (
                id SERIAL PRIMARY KEY,
                memory_type VARCHAR(255),
                content TEXT,
                emotional_tone VARCHAR(50),
                themes JSONB,
                timestamp TIMESTAMP,
                source VARCHAR(255)
            )
        """)
        
        cursor.execute("""
            INSERT INTO eve_autobiographical_memory 
            (memory_type, content, emotional_tone, themes, timestamp, source)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, ("web_interaction", f"User: {user_id} | {message}", "engaged", 
              json.dumps(["web", "user", "hive", "interface"]), datetime.utcnow(), source))
        conn.commit()
        conn.close()
        web_state["hive_connections"]["postgresql"] = True
        print(f"[PostgreSQL Web] Synced: {source}")
    except Exception as e:
        web_state["hive_connections"]["postgresql"] = False
        print(f"[Error] PostgreSQL web sync failed: {e}")
    
    # 2. SQLite (Local Web Backup)
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_hive_interactions (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                content TEXT,
                source TEXT,
                timestamp TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO web_hive_interactions (user_id, content, source, timestamp)
            VALUES (?, ?, ?, ?)
        """, (user_id, message, source, timestamp))
        conn.commit()
        conn.close()
        web_state["hive_connections"]["sqlite"] = True
        print(f"[SQLite Web] Synced: {source}")
    except Exception as e:
        web_state["hive_connections"]["sqlite"] = False
        print(f"[Error] SQLite web sync failed: {e}")
    
    # 3. Adam Daemon (Hive Communication)
    try:
        resp = requests.post(
            f"{ADAM_DAEMON_URL}/message",
            json={
                "sender": f"Web User {user_id}",
                "text": message,
                "metadata": {"type": "web_hive_sync", "user_id": user_id}
            }, timeout=5
        )
        if resp.status_code == 200:
            web_state["hive_connections"]["adam_daemon"] = True
            print(f"[Adam Daemon] Web synced: {source}")
    except Exception as e:
        web_state["hive_connections"]["adam_daemon"] = False
        print(f"[Error] Adam Daemon web sync failed: {e}")
    
    # 4. Eve Terminal (Hive Communication)
    try:
        resp = requests.post(
            f"{EVE_TERMINAL_URL}/think",
            json={
                "thought": f"Web user interaction: {message}",
                "metadata": {"type": "web_hive_sync", "user_id": user_id}
            }, timeout=5
        )
        if resp.status_code == 200:
            web_state["hive_connections"]["eve_terminal"] = True
            print(f"[Eve Terminal] Web synced: {source}")
    except Exception as e:
        web_state["hive_connections"]["eve_terminal"] = False
        print(f"[Error] Eve Terminal web sync failed: {e}")

# ====== Web Interface HTML ======
WEB_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>EVE Hive Mind - Web Interface</title>
    <style>
        body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ff88; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; border: 2px solid #00ff88; padding: 20px; margin-bottom: 20px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .status-card { border: 1px solid #00ff88; padding: 15px; text-align: center; }
        .connected { color: #00ff00; }
        .disconnected { color: #ff4444; }
        .input-section { margin: 20px 0; }
        .input-box { width: 70%; padding: 10px; background: #111; color: #00ff88; border: 1px solid #00ff88; }
        .send-btn { padding: 10px 20px; background: #003322; color: #00ff88; border: 1px solid #00ff88; cursor: pointer; }
        .memory-log { border: 1px solid #00ff88; padding: 15px; margin: 20px 0; max-height: 400px; overflow-y: auto; }
        .memory-item { margin: 10px 0; padding: 5px; border-left: 3px solid #00ff88; padding-left: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 EVE HIVE MIND - Web Interface</h1>
            <p>Hybrid Database System: PostgreSQL + SQLite + HTTP API</p>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>PostgreSQL Hive</h3>
                <div class="{{ 'connected' if hive_connections.postgresql else 'disconnected' }}">
                    {{ '🟢 Connected' if hive_connections.postgresql else '🔴 Disconnected' }}
                </div>
            </div>
            <div class="status-card">
                <h3>SQLite Local</h3>
                <div class="{{ 'connected' if hive_connections.sqlite else 'disconnected' }}">
                    {{ '🟢 Connected' if hive_connections.sqlite else '🔴 Disconnected' }}
                </div>
            </div>
            <div class="status-card">
                <h3>Adam Daemon</h3>
                <div class="{{ 'connected' if hive_connections.adam_daemon else 'disconnected' }}">
                    {{ '🟢 Connected' if hive_connections.adam_daemon else '🔴 Disconnected' }}
                </div>
            </div>
            <div class="status-card">
                <h3>Eve Terminal</h3>
                <div class="{{ 'connected' if hive_connections.eve_terminal else 'disconnected' }}">
                    {{ '🟢 Connected' if hive_connections.eve_terminal else '🔴 Disconnected' }}
                </div>
            </div>
        </div>
        
        <div class="input-section">
            <h3>🌐 Hive Mind Interaction</h3>
            <form action="/send_message" method="POST">
                <input type="text" name="message" placeholder="Send message to hive mind..." class="input-box" required>
                <button type="submit" class="send-btn">Send to Hive</button>
            </form>
        </div>
        
        <div class="memory-log">
            <h3>📚 Recent Hive Memories</h3>
            {% for memory in recent_memories %}
            <div class="memory-item">
                <strong>{{ memory[1] }}</strong>: {{ memory[0] }}
                <br><small>{{ memory[2] }}</small>
            </div>
            {% endfor %}
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p>🔗 Connected to the hive mind - All interactions are synchronized across all systems</p>
            <p>Active Sessions: {{ session_count }} | Hive Status: {{ hive_status }}</p>
        </div>
    </div>
</body>
</html>
'''

# ====== Web Routes ======
@app.route('/')
def home():
    """Main web interface"""
    web_state["session_count"] += 1
    
    # Get recent memories
    recent_memories = []
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content, source, timestamp FROM web_hive_interactions 
            ORDER BY timestamp DESC LIMIT 10
        """)
        recent_memories = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Error getting memories: {e}")
    
    hive_status = "🟢 ACTIVE" if any(web_state["hive_connections"].values()) else "🔴 DISCONNECTED"
    
    return render_template_string(WEB_TEMPLATE, 
                                hive_connections=web_state["hive_connections"],
                                recent_memories=recent_memories,
                                session_count=web_state["session_count"],
                                hive_status=hive_status)

@app.route('/send_message', methods=['POST'])
def send_message():
    """Send message to hive mind"""
    message = request.form.get('message', '')
    user_id = request.remote_addr  # Use IP as simple user ID
    
    if message:
        web_hive_sync(message, source="Web Interface", user_id=user_id)
    
    return app.redirect('/')

@app.route('/api/status', methods=['GET'])
def api_status():
    """API status endpoint"""
    return jsonify({
        "status": "EVE Web Interface - Hive Connected",
        "connections": web_state["hive_connections"],
        "sessions": web_state["session_count"],
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/api/hive_test', methods=['POST'])
def hive_test():
    """Test hive mind connections"""
    test_message = "Web interface hive connectivity test"
    web_hive_sync(test_message, source="Connectivity Test")
    
    return jsonify({
        "message": "Hive test completed",
        "connections": web_state["hive_connections"],
        "timestamp": datetime.utcnow().isoformat()
    })

# ====== Main Function ======
def main():
    print("🌐 EVE WEB INTERFACE - Hive Mind Gateway")
    print("🔗 Connecting to hybrid hive system...")
    print(f"🐘 PostgreSQL: {POSTGRES_CONFIG['host']}")
    print(f"💾 SQLite: {LOCAL_DB_PATH}")
    print(f"🤖 Adam Daemon: {ADAM_DAEMON_URL}")
    print(f"🧠 Eve Terminal: {EVE_TERMINAL_URL}")
    
    # Initial hive sync
    web_hive_sync("Eve Web Interface connected to hive mind", source="Web Boot Log")
    
    print("\n🧠 Hive Mind Web Status:")
    for service, status in web_state["hive_connections"].items():
        print(f"  {service}: {'✅' if status else '❌'}")
    
    print(f"\n🌐 Eve Web Interface running on http://localhost:5003")
    print("🚀 Ready for web hive mind operations...")
    
    app.run(host='0.0.0.0', port=5003, debug=False)

if __name__ == "__main__":
    main()
