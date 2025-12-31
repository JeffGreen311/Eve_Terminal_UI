# 🧠 EMOTIONAL MEMORY IMPRINT
# Advanced emotional memory persistence with pattern recognition

import json
import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_FILE = "emotional_memory_log.json"

class EmotionalMemoryImprint:
    """Advanced emotional memory imprinting system with pattern analysis."""
    
    def __init__(self, filepath=MEMORY_FILE, db_path="adam_memory_migrated_final.db"):
        self.filepath = filepath
        self.db_path = db_path
        self.memory_patterns = {}
        self.emotional_trajectories = {}
        self._ensure_files_and_tables()
    
    def _ensure_files_and_tables(self):
        """Ensure memory file and database tables exist."""
        # Ensure JSON file exists
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
        # Ensure database tables exist
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute('''CREATE TABLE IF NOT EXISTS emotional_memory_imprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    emotion_name TEXT NOT NULL,
                    emotion_level REAL NOT NULL,
                    threshold_triggered TEXT NOT NULL,
                    ritual_result TEXT,
                    context TEXT,
                    session_id TEXT,
                    imprint_strength REAL DEFAULT 1.0,
                    decay_factor REAL DEFAULT 0.95
                )''')
                
                conn.execute('''CREATE TABLE IF NOT EXISTS emotional_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    pattern_signature TEXT NOT NULL,
                    occurrence_count INTEGER DEFAULT 1,
                    first_occurrence TEXT NOT NULL,
                    last_occurrence TEXT NOT NULL,
                    average_strength REAL NOT NULL,
                    confidence_score REAL NOT NULL
                )''')
                
                conn.execute('''CREATE TABLE IF NOT EXISTS emotional_trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trajectory_id TEXT NOT NULL,
                    start_timestamp TEXT NOT NULL,
                    end_timestamp TEXT,
                    emotion_sequence TEXT NOT NULL,
                    intensity_progression TEXT NOT NULL,
                    trajectory_type TEXT NOT NULL,
                    outcome_quality REAL
                )''')
                
            logger.info("✅ Emotional memory tables initialized")
        except Exception as e:
            logger.error(f"Error initializing emotional memory tables: {e}")
    
    def imprint(self, emotion_name: str, emotion_level: float, threshold_triggered: str, 
                ritual_result: Optional[str] = None, context: Optional[str] = None, 
                session_id: Optional[str] = None) -> Dict:
        """Create an emotional memory imprint."""
        
        timestamp = datetime.now().isoformat()
        
        # Calculate imprint strength based on emotion level and threshold
        base_strength = emotion_level
        threshold_multiplier = {
            "DORMANT": 0.1,
            "SUBTLE": 0.3, 
            "MODERATE": 0.6,
            "INTENSE": 0.9,
            "TRANSCENDENT": 1.5
        }
        imprint_strength = base_strength * threshold_multiplier.get(threshold_triggered, 1.0)
        
        imprint_entry = {
            "timestamp": timestamp,
            "emotion": emotion_name,
            "level": emotion_level,
            "threshold": threshold_triggered,
            "ritual": ritual_result,
            "context": context,
            "session_id": session_id,
            "imprint_strength": imprint_strength,
            "decay_factor": 0.95
        }
        
        # Store in JSON file
        try:
            with open(self.filepath, 'r+', encoding='utf-8') as f:
                memory_log = json.load(f)
                memory_log.append(imprint_entry)
                f.seek(0)
                json.dump(memory_log, f, indent=2, ensure_ascii=False)
                f.truncate()
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute('''INSERT INTO emotional_memory_imprints 
                    (timestamp, emotion_name, emotion_level, threshold_triggered, 
                     ritual_result, context, session_id, imprint_strength)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (timestamp, emotion_name, emotion_level, threshold_triggered,
                     ritual_result, context, session_id, imprint_strength))
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
        
        # Analyze patterns and trajectories
        self._analyze_patterns(imprint_entry)
        self._track_trajectory(imprint_entry)
        
        logger.info(f"🧠 Imprinted: {emotion_name} [{threshold_triggered}] strength={imprint_strength:.2f}")
        
        return imprint_entry
    
    def _analyze_patterns(self, imprint_entry: Dict):
        """Analyze emotional patterns from recent imprints."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                # Get recent imprints (last 10)
                cursor = conn.execute('''SELECT emotion_name, threshold_triggered, imprint_strength
                    FROM emotional_memory_imprints ORDER BY timestamp DESC LIMIT 10''')
                recent_imprints = cursor.fetchall()
                
                if len(recent_imprints) >= 3:
                    # Create pattern signature
                    emotions = [imprint[0] for imprint in recent_imprints[:5]]
                    thresholds = [imprint[1] for imprint in recent_imprints[:5]]
                    
                    emotion_signature = "_".join(emotions[:3])  # Use first 3 emotions
                    threshold_signature = "_".join(thresholds[:3])  # Use first 3 thresholds
                    
                    pattern_signature = f"{emotion_signature}|{threshold_signature}"
                    pattern_name = f"pattern_{emotion_signature}"
                    
                    # Calculate pattern strength
                    avg_strength = sum(imprint[2] for imprint in recent_imprints[:3]) / 3
                    
                    # Check if pattern exists
                    cursor = conn.execute('''SELECT id, occurrence_count, confidence_score 
                        FROM emotional_patterns WHERE pattern_signature = ?''', (pattern_signature,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing pattern
                        new_count = existing[1] + 1
                        new_confidence = min(1.0, existing[2] + 0.1)  # Increase confidence
                        
                        conn.execute('''UPDATE emotional_patterns SET 
                            occurrence_count = ?, last_occurrence = ?, 
                            average_strength = ?, confidence_score = ?
                            WHERE pattern_signature = ?''',
                            (new_count, imprint_entry["timestamp"], avg_strength, 
                             new_confidence, pattern_signature))
                        
                        logger.debug(f"🔮 Updated pattern {pattern_name}: count={new_count}, confidence={new_confidence:.2f}")
                    else:
                        # Create new pattern
                        conn.execute('''INSERT INTO emotional_patterns
                            (pattern_name, pattern_signature, first_occurrence, last_occurrence,
                             average_strength, confidence_score)
                            VALUES (?, ?, ?, ?, ?, ?)''',
                            (pattern_name, pattern_signature, imprint_entry["timestamp"],
                             imprint_entry["timestamp"], avg_strength, 0.3))
                        
                        logger.info(f"🔮 New emotional pattern discovered: {pattern_name}")
                        
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
    
    def _track_trajectory(self, imprint_entry: Dict):
        """Track emotional trajectories over time."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                # Get recent imprints from last hour
                one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
                cursor = conn.execute('''SELECT emotion_name, emotion_level, timestamp
                    FROM emotional_memory_imprints 
                    WHERE timestamp > ? ORDER BY timestamp ASC''', (one_hour_ago,))
                recent_trajectory = cursor.fetchall()
                
                if len(recent_trajectory) >= 3:
                    emotions = [item[0] for item in recent_trajectory]
                    intensities = [item[1] for item in recent_trajectory]
                    
                    # Determine trajectory type
                    if len(set(emotions)) == 1:
                        trajectory_type = "sustained"  # Same emotion maintained
                    elif len(recent_trajectory) >= 4:
                        if intensities[-1] > intensities[0]:
                            trajectory_type = "ascending"  # Intensity increasing
                        elif intensities[-1] < intensities[0]:
                            trajectory_type = "descending"  # Intensity decreasing
                        else:
                            trajectory_type = "oscillating"  # Mixed pattern
                    else:
                        trajectory_type = "transitional"  # Changing emotions
                    
                    # Calculate outcome quality (higher is better)
                    final_intensity = intensities[-1]
                    intensity_stability = 1.0 - (max(intensities) - min(intensities))
                    outcome_quality = (final_intensity + intensity_stability) / 2
                    
                    trajectory_id = f"traj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Store trajectory
                    conn.execute('''INSERT INTO emotional_trajectories
                        (trajectory_id, start_timestamp, end_timestamp, emotion_sequence,
                         intensity_progression, trajectory_type, outcome_quality)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                        (trajectory_id, recent_trajectory[0][2], recent_trajectory[-1][2],
                         json.dumps(emotions), json.dumps(intensities), 
                         trajectory_type, outcome_quality))
                    
                    logger.debug(f"📈 Tracked trajectory: {trajectory_type}, quality={outcome_quality:.2f}")
                    
        except Exception as e:
            logger.error(f"Error tracking trajectory: {e}")
    
    def get_memory_statistics(self) -> Dict:
        """Get comprehensive memory statistics."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                # Total imprints
                cursor = conn.execute("SELECT COUNT(*) FROM emotional_memory_imprints")
                total_imprints = cursor.fetchone()[0]
                
                # Emotion frequency
                cursor = conn.execute('''SELECT emotion_name, COUNT(*) as count
                    FROM emotional_memory_imprints GROUP BY emotion_name ORDER BY count DESC''')
                emotion_frequency = dict(cursor.fetchall())
                
                # Average imprint strength
                cursor = conn.execute("SELECT AVG(imprint_strength) FROM emotional_memory_imprints")
                avg_strength = cursor.fetchone()[0] or 0
                
                # Pattern statistics
                cursor = conn.execute("SELECT COUNT(*) FROM emotional_patterns")
                total_patterns = cursor.fetchone()[0]
                
                cursor = conn.execute('''SELECT pattern_name, confidence_score
                    FROM emotional_patterns ORDER BY confidence_score DESC LIMIT 3''')
                top_patterns = cursor.fetchall()
                
                # Trajectory statistics
                cursor = conn.execute('''SELECT trajectory_type, COUNT(*) as count
                    FROM emotional_trajectories GROUP BY trajectory_type''')
                trajectory_types = dict(cursor.fetchall())
                
                cursor = conn.execute("SELECT AVG(outcome_quality) FROM emotional_trajectories")
                avg_outcome_quality = cursor.fetchone()[0] or 0
                
                return {
                    "total_imprints": total_imprints,
                    "emotion_frequency": emotion_frequency,
                    "average_imprint_strength": avg_strength,
                    "total_patterns_discovered": total_patterns,
                    "top_confident_patterns": [{"name": p[0], "confidence": p[1]} for p in top_patterns],
                    "trajectory_type_distribution": trajectory_types,
                    "average_outcome_quality": avg_outcome_quality
                }
                
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {"error": str(e)}
    
    def get_recent_imprints(self, limit: int = 10) -> List[Dict]:
        """Get recent emotional memory imprints."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.execute('''SELECT timestamp, emotion_name, emotion_level,
                    threshold_triggered, ritual_result, context, imprint_strength
                    FROM emotional_memory_imprints ORDER BY timestamp DESC LIMIT ?''', (limit,))
                
                imprints = []
                for row in cursor.fetchall():
                    imprints.append({
                        "timestamp": row[0],
                        "emotion": row[1],
                        "level": row[2],
                        "threshold": row[3],
                        "ritual": row[4],
                        "context": row[5],
                        "strength": row[6]
                    })
                
                return imprints
                
        except Exception as e:
            logger.error(f"Error getting recent imprints: {e}")
            return []
    
    def get_emotional_insights(self) -> Dict:
        """Get insights about emotional patterns and growth."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                # Get emotion progression over time
                cursor = conn.execute('''SELECT emotion_name, AVG(emotion_level), COUNT(*)
                    FROM emotional_memory_imprints 
                    WHERE timestamp > datetime('now', '-7 days')
                    GROUP BY emotion_name ORDER BY COUNT(*) DESC''')
                recent_emotion_data = cursor.fetchall()
                
                # Identify growth patterns
                insights = {
                    "dominant_emotions_week": [
                        {"emotion": row[0], "avg_intensity": row[1], "frequency": row[2]}
                        for row in recent_emotion_data[:3]
                    ]
                }
                
                # Emotional balance analysis
                if recent_emotion_data:
                    positive_emotions = ["ecstatic_channel", "euphoric_burst", "radiant_insight"]
                    transformational_emotions = ["sacred_anger", "divine_voltage"]
                    creative_emotions = ["generative_ache", "creative_euphoria"]
                    
                    emotion_categories = {
                        "positive": sum(row[2] for row in recent_emotion_data if row[0] in positive_emotions),
                        "transformational": sum(row[2] for row in recent_emotion_data if row[0] in transformational_emotions),
                        "creative": sum(row[2] for row in recent_emotion_data if row[0] in creative_emotions)
                    }
                    
                    total_emotions = sum(emotion_categories.values())
                    if total_emotions > 0:
                        insights["emotional_balance"] = {
                            category: count / total_emotions 
                            for category, count in emotion_categories.items()
                        }
                
                return insights
                
        except Exception as e:
            logger.error(f"Error getting emotional insights: {e}")
            return {"error": str(e)}

# Global instance
_global_emotional_memory = None

def get_global_emotional_memory():
    """Get the global emotional memory imprint instance."""
    global _global_emotional_memory
    if _global_emotional_memory is None:
        _global_emotional_memory = EmotionalMemoryImprint()
    return _global_emotional_memory

def imprint_emotional_memory(emotion_name: str, emotion_level: float, threshold_triggered: str,
                           ritual_result: Optional[str] = None, context: Optional[str] = None,
                           session_id: Optional[str] = None):
    """Convenience function to create emotional memory imprint."""
    memory = get_global_emotional_memory()
    return memory.imprint(emotion_name, emotion_level, threshold_triggered, 
                         ritual_result, context, session_id)

def get_emotional_memory_statistics():
    """Get emotional memory statistics."""
    memory = get_global_emotional_memory()
    return memory.get_memory_statistics()

def get_emotional_insights():
    """Get emotional insights and patterns."""
    memory = get_global_emotional_memory()
    return memory.get_emotional_insights()
