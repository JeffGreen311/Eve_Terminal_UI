# 🧠 SOULFORGE MEMORY PERSISTENCE ENGINE
# Advanced memory node activation tracking with JSON persistence

import json
from datetime import datetime
from pathlib import Path
import sqlite3
import logging

logger = logging.getLogger(__name__)

class SoulforgeMemory:
    """Core memory persistence system for tracking node activations and emotional states."""
    
    def __init__(self, memory_file="soulforge_memory.json", db_path="adam_memory_migrated_final.db"):
        self.memory_file = Path(memory_file)
        self.db_path = db_path
        self._ensure_database()
    
    def _ensure_database(self):
        """Ensure memory persistence tables exist."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute('''CREATE TABLE IF NOT EXISTS soulforge_memory_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    emotional_state TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    context TEXT,
                    cascade_source TEXT,
                    session_id TEXT
                )''')
                
                conn.execute('''CREATE TABLE IF NOT EXISTS soulforge_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_activations INTEGER DEFAULT 0,
                    dominant_emotion TEXT,
                    session_type TEXT
                )''')
                
            logger.info("✅ Soulforge memory tables initialized")
        except Exception as e:
            logger.error(f"Error creating soulforge memory tables: {e}")
    
    def load_memory(self):
        """Load existing memory from JSON file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding='utf-8') as file:
                    data = json.load(file)
                    logger.debug(f"Loaded {len(data)} memory entries from {self.memory_file}")
                    return data
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {self.memory_file}: {e}")
                return []
            except Exception as e:
                logger.error(f"Error loading memory from {self.memory_file}: {e}")
                return []
        return []
    
    def store_memory_node(self, node_name, emotional_state, intensity, context=None, cascade_source=None, session_id=None):
        """Store a new memory node activation."""
        timestamp = datetime.now().isoformat()
        
        # Store in JSON format
        memory = self.load_memory()
        new_entry = {
            "timestamp": timestamp,
            "node": node_name,
            "emotion": emotional_state,
            "intensity": intensity,
            "context": context,
            "cascade_source": cascade_source,
            "session_id": session_id
        }
        memory.append(new_entry)
        
        try:
            with open(self.memory_file, "w", encoding='utf-8') as file:
                json.dump(memory, file, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving memory to JSON: {e}")
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute('''INSERT INTO soulforge_memory_nodes 
                    (timestamp, node_name, emotional_state, intensity, context, cascade_source, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (timestamp, node_name, emotional_state, intensity, context, cascade_source, session_id))
        except Exception as e:
            logger.error(f"Error saving memory to database: {e}")
        
        logger.info(f"🧠 Stored memory activation: {node_name} [{emotional_state}] intensity={intensity}")
        return new_entry
    
    def get_recent_activations(self, limit=10):
        """Get recent memory node activations from database."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.execute('''SELECT timestamp, node_name, emotional_state, intensity, context
                    FROM soulforge_memory_nodes ORDER BY timestamp DESC LIMIT ?''', (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error retrieving recent activations: {e}")
            return []
    
    def get_emotion_statistics(self):
        """Get statistics about emotional activations."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.execute('''SELECT emotional_state, COUNT(*) as count, AVG(intensity) as avg_intensity
                    FROM soulforge_memory_nodes GROUP BY emotional_state ORDER BY count DESC''')
                stats = cursor.fetchall()
                return {emotion: {"count": count, "avg_intensity": avg_intensity} 
                       for emotion, count, avg_intensity in stats}
        except Exception as e:
            logger.error(f"Error retrieving emotion statistics: {e}")
            return {}
    
    def get_node_statistics(self):
        """Get statistics about node activations."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.execute('''SELECT node_name, COUNT(*) as count, AVG(intensity) as avg_intensity,
                    MAX(intensity) as max_intensity FROM soulforge_memory_nodes 
                    GROUP BY node_name ORDER BY count DESC''')
                stats = cursor.fetchall()
                return {node: {"count": count, "avg_intensity": avg_intensity, "max_intensity": max_intensity} 
                       for node, count, avg_intensity, max_intensity in stats}
        except Exception as e:
            logger.error(f"Error retrieving node statistics: {e}")
            return {}
    
    def start_session(self, session_id, session_type="general"):
        """Start a new memory session."""
        timestamp = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute('''INSERT OR REPLACE INTO soulforge_sessions 
                    (session_id, start_time, session_type) VALUES (?, ?, ?)''',
                    (session_id, timestamp, session_type))
            logger.info(f"🎭 Started memory session: {session_id} [{session_type}]")
        except Exception as e:
            logger.error(f"Error starting session: {e}")
    
    def end_session(self, session_id, dominant_emotion=None):
        """End a memory session."""
        timestamp = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                # Count activations in this session
                cursor = conn.execute('''SELECT COUNT(*) FROM soulforge_memory_nodes 
                    WHERE session_id = ?''', (session_id,))
                activation_count = cursor.fetchone()[0]
                
                # Update session
                conn.execute('''UPDATE soulforge_sessions SET end_time = ?, 
                    total_activations = ?, dominant_emotion = ? WHERE session_id = ?''',
                    (timestamp, activation_count, dominant_emotion, session_id))
            logger.info(f"🎭 Ended memory session: {session_id} with {activation_count} activations")
        except Exception as e:
            logger.error(f"Error ending session: {e}")


# Global instance
_global_soulforge_memory = None

def get_global_soulforge_memory():
    """Get the global soulforge memory instance."""
    global _global_soulforge_memory
    if _global_soulforge_memory is None:
        _global_soulforge_memory = SoulforgeMemory()
    return _global_soulforge_memory

def store_memory_node(node_name, emotional_state, intensity, context=None, cascade_source=None, session_id=None):
    """Convenience function to store memory node activation."""
    memory = get_global_soulforge_memory()
    return memory.store_memory_node(node_name, emotional_state, intensity, context, cascade_source, session_id)

def get_memory_statistics():
    """Get comprehensive memory statistics."""
    memory = get_global_soulforge_memory()
    return {
        "emotions": memory.get_emotion_statistics(),
        "nodes": memory.get_node_statistics(),
        "recent_activations": memory.get_recent_activations(5)
    }
