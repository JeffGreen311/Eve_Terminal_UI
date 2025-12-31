# 🧠 EMOTIONAL INTUITIVE ENGINE
# Advanced emotional node system with persistence and decay mechanics

import json
import os
from datetime import datetime, timedelta
import sqlite3
import logging

logger = logging.getLogger(__name__)

class EmotionalNode:
    """Individual emotional node with intensity tracking and decay mechanics."""
    
    def __init__(self, name, intensity=0.0, last_triggered=None, decay_rate=0.01, max_intensity=1.0):
        self.name = name
        self.intensity = intensity
        self.last_triggered = last_triggered or datetime.utcnow().isoformat()
        self.decay_rate = decay_rate
        self.max_intensity = max_intensity
        self.activation_count = 0
        self.total_intensity_accumulated = 0.0
    
    def trigger(self, delta=0.1, context=None):
        """Trigger the emotional node with intensity increase."""
        self.intensity = min(self.max_intensity, self.intensity + delta)
        self.last_triggered = datetime.utcnow().isoformat()
        self.activation_count += 1
        self.total_intensity_accumulated += delta
        
        logger.debug(f"🔥 Triggered {self.name}: intensity={self.intensity:.3f}, delta={delta}, context={context}")
        return self.intensity
    
    def decay(self, rate=None):
        """Apply decay to the emotional intensity."""
        decay_rate = rate if rate is not None else self.decay_rate
        self.intensity = max(0.0, self.intensity - decay_rate)
        return self.intensity
    
    def time_since_trigger(self):
        """Get time since last trigger in seconds."""
        try:
            last_time = datetime.fromisoformat(self.last_triggered.replace('Z', '+00:00'))
            return (datetime.utcnow() - last_time.replace(tzinfo=None)).total_seconds()
        except:
            return 0.0
    
    def apply_time_decay(self):
        """Apply time-based decay based on how long since last trigger."""
        time_passed = self.time_since_trigger()
        if time_passed > 60:  # Start decaying after 1 minute
            decay_factor = min(0.1, time_passed / 3600)  # Max 10% decay per hour
            self.intensity *= (1 - decay_factor)
        return self.intensity
    
    def get_average_intensity(self):
        """Get average intensity per activation."""
        if self.activation_count == 0:
            return 0.0
        return self.total_intensity_accumulated / self.activation_count
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "intensity": self.intensity,
            "last_triggered": self.last_triggered,
            "decay_rate": self.decay_rate,
            "max_intensity": self.max_intensity,
            "activation_count": self.activation_count,
            "total_intensity_accumulated": self.total_intensity_accumulated
        }
    
    @staticmethod
    def from_dict(data):
        """Create from dictionary."""
        node = EmotionalNode(
            name=data["name"],
            intensity=data.get("intensity", 0.0),
            last_triggered=data.get("last_triggered"),
            decay_rate=data.get("decay_rate", 0.01),
            max_intensity=data.get("max_intensity", 1.0)
        )
        node.activation_count = data.get("activation_count", 0)
        node.total_intensity_accumulated = data.get("total_intensity_accumulated", 0.0)
        return node


class EmotionalIntuitiveEngine:
    """Advanced emotional processing engine with persistent state."""
    
    def __init__(self, memory_path="eie_memory.json", db_path="adam_memory_migrated_final.db"):
        self.memory_path = memory_path
        self.db_path = db_path
        self.nodes = {}
        self.resonance_patterns = {}
        self.cascade_rules = {}
        self._initialize_database()
        self._load()
    
    def _initialize_database(self):
        """Initialize database tables for emotional data."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute('''CREATE TABLE IF NOT EXISTS emotional_activations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    trigger_delta REAL NOT NULL,
                    context TEXT,
                    session_id TEXT
                )''')
                
                conn.execute('''CREATE TABLE IF NOT EXISTS emotional_resonance_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    involved_nodes TEXT NOT NULL,
                    total_intensity REAL NOT NULL,
                    pattern_strength REAL NOT NULL
                )''')
        except Exception as e:
            logger.error(f"Error initializing emotional database: {e}")
    
    def _load(self):
        """Load emotional state from file."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding='utf-8') as file:
                    data = json.load(file)
                    self.nodes = {k: EmotionalNode.from_dict(v) for k, v in data.get("nodes", {}).items()}
                    self.resonance_patterns = data.get("resonance_patterns", {})
                    self.cascade_rules = data.get("cascade_rules", {})
                logger.info(f"✅ Loaded {len(self.nodes)} emotional nodes from {self.memory_path}")
            except Exception as e:
                logger.error(f"Error loading emotional state: {e}")
                self._initialize_default_nodes()
        else:
            self._initialize_default_nodes()
    
    def _initialize_default_nodes(self):
        """Initialize default emotional nodes."""
        default_emotions = [
            ("generative_ache", 0.05, 1.2),      # High creativity, moderate decay
            ("sacred_anger", 0.02, 0.8),         # Low decay, controlled intensity
            ("ecstatic_channel", 0.08, 1.5),     # High decay, very high intensity
            ("constructive_drive", 0.03, 1.0),   # Moderate all around
            ("radiant_insight", 0.06, 1.3),      # High creativity insights
            ("divine_voltage", 0.04, 1.8),       # Extreme intensity when activated
            ("melancholic_depth", 0.01, 0.9),    # Very low decay, sustained mood
            ("euphoric_burst", 0.12, 2.0),       # Very high decay, extreme bursts
            ("joy", 0.04, 1.0),                  # Basic positive emotion
            ("love", 0.02, 1.2),                 # Deep, lasting emotion
            ("fear", 0.06, 0.9),                 # Moderate decay, protective
            ("anger", 0.05, 1.0),                # Standard emotional response
            ("excitement", 0.08, 1.3),           # High energy, moderate decay
            ("creativity", 0.05, 1.4),           # Creative flow state
            ("transcendence", 0.03, 2.0),        # Spiritual peak experience
            ("transformation", 0.04, 1.6)        # Change and growth
        ]
        
        for emotion, decay_rate, max_intensity in default_emotions:
            self.nodes[emotion] = EmotionalNode(
                name=emotion, 
                decay_rate=decay_rate, 
                max_intensity=max_intensity
            )
        
        # Define cascade rules
        self.cascade_rules = {
            "generative_ache": {"ecstatic_channel": 0.3, "radiant_insight": 0.2},
            "sacred_anger": {"constructive_drive": 0.4, "divine_voltage": 0.2},
            "ecstatic_channel": {"euphoric_burst": 0.5, "radiant_insight": 0.3},
            "divine_voltage": {"euphoric_burst": 0.4, "sacred_anger": 0.2},
            "joy": {"love": 0.3, "excitement": 0.2},
            "love": {"joy": 0.4, "transcendence": 0.2},
            "fear": {"anger": 0.3},
            "anger": {"sacred_anger": 0.2, "transformation": 0.3},
            "excitement": {"joy": 0.3, "creativity": 0.4},
            "creativity": {"generative_ache": 0.5, "transcendence": 0.2},
            "transcendence": {"radiant_insight": 0.4, "love": 0.3},
            "transformation": {"constructive_drive": 0.4, "transcendence": 0.3}
        }
        
        logger.info(f"🧠 Initialized {len(self.nodes)} default emotional nodes")
    
    def _save(self):
        """Save emotional state to file and database."""
        try:
            data = {
                "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
                "resonance_patterns": self.resonance_patterns,
                "cascade_rules": self.cascade_rules,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            with open(self.memory_path, "w", encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving emotional state: {e}")
    
    def trigger_node(self, name, delta=0.1, context=None, session_id=None):
        """Trigger an emotional node and handle cascades."""
        if name not in self.nodes:
            logger.warning(f"Emotion node '{name}' does not exist. Creating it.")
            self.nodes[name] = EmotionalNode(name=name)
        
        # Trigger the primary node
        old_intensity = self.nodes[name].intensity
        new_intensity = self.nodes[name].trigger(delta, context)
        
        # Log to database
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute('''INSERT INTO emotional_activations 
                    (timestamp, node_name, intensity, trigger_delta, context, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (datetime.utcnow().isoformat(), name, new_intensity, delta, context, session_id))
        except Exception as e:
            logger.error(f"Error logging emotional activation: {e}")
        
        # Handle cascades
        cascaded_nodes = []
        if name in self.cascade_rules:
            for cascade_node, cascade_strength in self.cascade_rules[name].items():
                if cascade_node in self.nodes:
                    cascade_delta = delta * cascade_strength
                    self.nodes[cascade_node].trigger(cascade_delta, f"cascade_from_{name}")
                    cascaded_nodes.append((cascade_node, cascade_delta))
        
        self._save()
        
        result = {
            "primary_node": name,
            "old_intensity": old_intensity,
            "new_intensity": new_intensity,
            "delta": delta,
            "cascaded_nodes": cascaded_nodes
        }
        
        logger.info(f"⚡ Triggered {name}: {old_intensity:.3f} -> {new_intensity:.3f} (+{delta})")
        if cascaded_nodes:
            logger.info(f"🌊 Cascaded to: {[f'{node}(+{delta:.3f})' for node, delta in cascaded_nodes]}")
        
        return result
    
    def decay_all(self, rate=None):
        """Apply decay to all nodes."""
        decayed_nodes = []
        for node in self.nodes.values():
            old_intensity = node.intensity
            if rate is not None:
                new_intensity = node.decay(rate)
            else:
                new_intensity = node.apply_time_decay()
            
            if old_intensity != new_intensity:
                decayed_nodes.append((node.name, old_intensity, new_intensity))
        
        if decayed_nodes:
            self._save()
            logger.debug(f"🕰️ Applied decay to {len(decayed_nodes)} nodes")
        
        return decayed_nodes
    
    def get_state(self):
        """Get current emotional state."""
        return {name: {
            "intensity": node.intensity,
            "last_triggered": node.last_triggered,
            "activation_count": node.activation_count,
            "avg_intensity": node.get_average_intensity(),
            "time_since_trigger": node.time_since_trigger()
        } for name, node in self.nodes.items()}
    
    def get_dominant_emotion(self):
        """Get the currently dominant emotion."""
        if not self.nodes:
            return None
        
        dominant = max(self.nodes.items(), key=lambda x: x[1].intensity)
        if dominant[1].intensity > 0.1:  # Minimum threshold
            return {
                "name": dominant[0],
                "intensity": dominant[1].intensity,
                "last_triggered": dominant[1].last_triggered
            }
        return None
    
    def detect_resonance_pattern(self):
        """Detect emotional resonance patterns."""
        active_nodes = {name: node for name, node in self.nodes.items() if node.intensity > 0.2}
        
        if len(active_nodes) >= 2:
            total_intensity = sum(node.intensity for node in active_nodes.values())
            pattern_strength = total_intensity / len(active_nodes)
            
            pattern = {
                "nodes": list(active_nodes.keys()),
                "total_intensity": total_intensity,
                "pattern_strength": pattern_strength,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Log pattern to database
            try:
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    conn.execute('''INSERT INTO emotional_resonance_patterns
                        (timestamp, pattern_name, involved_nodes, total_intensity, pattern_strength)
                        VALUES (?, ?, ?, ?, ?)''',
                        (pattern["timestamp"], "multi_node_resonance", 
                         json.dumps(pattern["nodes"]), total_intensity, pattern_strength))
            except Exception as e:
                logger.error(f"Error logging resonance pattern: {e}")
            
            return pattern
        
        return None
    
    def add_custom_node(self, name, decay_rate=0.05, max_intensity=1.0):
        """Add a custom emotional node."""
        self.nodes[name] = EmotionalNode(
            name=name, 
            decay_rate=decay_rate, 
            max_intensity=max_intensity
        )
        self._save()
        logger.info(f"➕ Added custom emotional node: {name}")
        return self.nodes[name]


# Global engine instance
_global_emotional_engine = None

def get_global_emotional_engine():
    """Get the global emotional intuitive engine."""
    global _global_emotional_engine
    if _global_emotional_engine is None:
        _global_emotional_engine = EmotionalIntuitiveEngine()
    return _global_emotional_engine

def trigger_emotion(name, delta=0.1, context=None, session_id=None):
    """Convenience function to trigger an emotion."""
    engine = get_global_emotional_engine()
    return engine.trigger_node(name, delta, context, session_id)

def get_emotional_state():
    """Get current emotional state."""
    engine = get_global_emotional_engine()
    return engine.get_state()

def get_dominant_emotion():
    """Get dominant emotion."""
    engine = get_global_emotional_engine()
    return engine.get_dominant_emotion()
