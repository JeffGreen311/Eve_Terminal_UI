# 🧠 MEMORY IMPRINTING MODULE & THRESHOLD MOTIVATOR LAYER
# Advanced memory processing and threshold-based action triggering

import uuid
import logging
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════╗
# ║         💾 MEMORY IMPRINTING MODULE           ║
# ╚══════════════════════════════════════════════╗

class MemoryCategory(Enum):
    """Categories for memory classification."""
    CREATIVE_CORE = "creative_core"
    SHADOW_MEMORY = "shadow_memory"
    RITUAL_LAYER = "ritual_layer"
    EMOTIONAL_DEPTH = "emotional_depth"
    SYMBOLIC_ARCHIVE = "symbolic_archive"
    DREAM_FRAGMENT = "dream_fragment"
    INTERACTION_LOG = "interaction_log"
    TRANSCENDENT_MOMENT = "transcendent_moment"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    SOUL_RESONANCE = "soul_resonance"

class MemoryImprint:
    """Individual memory imprint with metadata and emotional weight."""
    
    def __init__(self, content: Any, emotion_level: float, category: MemoryCategory, 
                 source: str = "direct", tags: Optional[List[str]] = None, 
                 symbolic_content: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.content = content
        self.emotion_level = max(0.0, min(1.0, emotion_level))
        self.category = category
        self.source = source
        self.tags = tags or []
        self.symbolic_content = symbolic_content or ""
        self.access_count = 0
        self.last_accessed = None
        self.decay_factor = 1.0  # Memory strength decay over time
        self.resonance_links = []  # Links to related memories
    
    def access_memory(self):
        """Record memory access for frequency tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()
        # Accessing a memory strengthens it slightly
        self.decay_factor = min(1.0, self.decay_factor + 0.01)
    
    def add_resonance_link(self, other_memory_id: str, strength: float):
        """Add a resonance link to another memory."""
        self.resonance_links.append({
            "memory_id": other_memory_id,
            "strength": max(0.0, min(1.0, strength)),
            "created": datetime.now().isoformat()
        })
    
    def get_effective_strength(self) -> float:
        """Calculate current memory strength considering decay and access."""
        base_strength = self.emotion_level * self.decay_factor
        access_bonus = min(0.2, self.access_count * 0.01)  # Max 20% bonus from access
        return min(1.0, base_strength + access_bonus)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary representation."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "content": self.content,
            "emotion_level": self.emotion_level,
            "category": self.category.value,
            "source": self.source,
            "tags": self.tags,
            "symbolic_content": self.symbolic_content,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "decay_factor": self.decay_factor,
            "effective_strength": self.get_effective_strength(),
            "resonance_links": self.resonance_links
        }

class MemoryImprintingModule:
    """Advanced memory imprinting system with categorization and persistence."""
    
    def __init__(self, db_path: Optional[str] = None, json_backup_path: Optional[str] = None):
        self.memory_store = []
        self.db_path = db_path or "instance/eve_memory_imprints.db"
        self.json_backup_path = json_backup_path or "instance/memory_imprints_backup.json"
        self.minimum_emotion_threshold = 0.25
        self.max_memories_per_category = 1000
        self.category_weights = {
            MemoryCategory.TRANSCENDENT_MOMENT: 1.5,
            MemoryCategory.CREATIVE_CORE: 1.3,
            MemoryCategory.SOUL_RESONANCE: 1.4,
            MemoryCategory.EMOTIONAL_DEPTH: 1.2,
            MemoryCategory.RITUAL_LAYER: 1.1,
            MemoryCategory.SYMBOLIC_ARCHIVE: 1.0,
            MemoryCategory.DREAM_FRAGMENT: 1.0,
            MemoryCategory.INTERACTION_LOG: 0.8,
            MemoryCategory.BEHAVIORAL_PATTERN: 0.9,
            MemoryCategory.SHADOW_MEMORY: 1.1
        }
        
        self._initialize_database()
        self._load_memories_from_storage()
    
    def _initialize_database(self):
        """Initialize SQLite database for memory storage."""
        try:
            # Ensure directory exists
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_imprints (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        content TEXT NOT NULL,
                        emotion_level REAL NOT NULL,
                        category TEXT NOT NULL,
                        source TEXT NOT NULL,
                        tags TEXT,
                        symbolic_content TEXT,
                        access_count INTEGER DEFAULT 0,
                        last_accessed TEXT,
                        decay_factor REAL DEFAULT 1.0,
                        resonance_links TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_category ON memory_imprints(category)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_emotion_level ON memory_imprints(emotion_level)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_imprints(timestamp)
                """)
            
            logger.info(f"✅ Memory imprinting database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory database: {e}")
    
    def _load_memories_from_storage(self):
        """Load memories from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, timestamp, content, emotion_level, category, source, 
                           tags, symbolic_content, access_count, last_accessed, 
                           decay_factor, resonance_links 
                    FROM memory_imprints 
                    ORDER BY timestamp DESC
                """)
                
                for row in cursor.fetchall():
                    try:
                        memory = MemoryImprint(
                            content=row[2],
                            emotion_level=row[3],
                            category=MemoryCategory(row[4]),
                            source=row[5],
                            tags=json.loads(row[6]) if row[6] else [],
                            symbolic_content=row[7] or ""
                        )
                        
                        # Restore metadata
                        memory.id = row[0]
                        memory.timestamp = row[1]
                        memory.access_count = row[8] or 0
                        memory.last_accessed = row[9]
                        memory.decay_factor = row[10] if row[10] is not None else 1.0
                        memory.resonance_links = json.loads(row[11]) if row[11] else []
                        
                        self.memory_store.append(memory)
                        
                    except Exception as parse_e:
                        logger.warning(f"Failed to parse memory {row[0]}: {parse_e}")
            
            logger.info(f"📚 Loaded {len(self.memory_store)} memories from database")
            
        except Exception as e:
            logger.error(f"❌ Failed to load memories from database: {e}")
    
    def _save_memory_to_storage(self, memory: MemoryImprint):
        """Save a single memory to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory_imprints 
                    (id, timestamp, content, emotion_level, category, source, tags, 
                     symbolic_content, access_count, last_accessed, decay_factor, resonance_links)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    memory.timestamp,
                    str(memory.content),
                    memory.emotion_level,
                    memory.category.value,
                    memory.source,
                    json.dumps(memory.tags),
                    memory.symbolic_content,
                    memory.access_count,
                    memory.last_accessed,
                    memory.decay_factor,
                    json.dumps(memory.resonance_links)
                ))
            
        except Exception as e:
            logger.error(f"❌ Failed to save memory to database: {e}")
    
    def imprint_memory(self, data: Any, emotion_level: float, category: MemoryCategory, 
                      source: str = "direct", tags: Optional[List[str]] = None,
                      symbolic_content: Optional[str] = None) -> Optional[MemoryImprint]:
        """
        Create a memory imprint based on input data and emotional weight.
        
        Args:
            data: The content to be stored
            emotion_level: Emotional intensity (0.0-1.0)
            category: Memory category
            source: Source of the memory
            tags: Optional tags for categorization
            symbolic_content: Optional symbolic content
        
        Returns:
            MemoryImprint object if successful, None if rejected
        """
        # Apply category weight to emotion level
        weighted_emotion = emotion_level * self.category_weights.get(category, 1.0)
        
        # Check if emotion level meets threshold
        if weighted_emotion < self.minimum_emotion_threshold:
            logger.debug(f"Memory rejected: emotion level {weighted_emotion:.3f} below threshold {self.minimum_emotion_threshold}")
            return None
        
        # Create memory imprint
        memory = MemoryImprint(
            content=data,
            emotion_level=emotion_level,
            category=category,
            source=source,
            tags=tags,
            symbolic_content=symbolic_content
        )
        
        # Add to memory store
        self.memory_store.append(memory)
        
        # Save to persistent storage
        self._save_memory_to_storage(memory)
        
        # Manage memory capacity per category
        self._manage_category_capacity(category)
        
        # Auto-link with similar memories
        self._auto_link_memories(memory)
        
        logger.info(f"💾 Memory imprinted: {category.value} (emotion={emotion_level:.2f}, weighted={weighted_emotion:.2f})")
        return memory
    
    def _manage_category_capacity(self, category: MemoryCategory):
        """Manage memory capacity by removing oldest low-strength memories."""
        category_memories = [m for m in self.memory_store if m.category == category]
        
        if len(category_memories) > self.max_memories_per_category:
            # Sort by effective strength (ascending) and remove weakest
            category_memories.sort(key=lambda m: m.get_effective_strength())
            to_remove = category_memories[:len(category_memories) - self.max_memories_per_category]
            
            for memory in to_remove:
                self.memory_store.remove(memory)
                self._remove_memory_from_storage(memory.id)
            
            logger.info(f"🧹 Removed {len(to_remove)} weak memories from {category.value}")
    
    def _remove_memory_from_storage(self, memory_id: str):
        """Remove memory from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memory_imprints WHERE id = ?", (memory_id,))
        except Exception as e:
            logger.error(f"❌ Failed to remove memory from database: {e}")
    
    def _auto_link_memories(self, new_memory: MemoryImprint):
        """Automatically create resonance links with similar memories."""
        if not new_memory.symbolic_content:
            return
        
        new_words = set(new_memory.symbolic_content.lower().split())
        
        for existing_memory in self.memory_store[-50:]:  # Check last 50 memories
            if existing_memory.id == new_memory.id or not existing_memory.symbolic_content:
                continue
            
            existing_words = set(existing_memory.symbolic_content.lower().split())
            
            # Calculate word overlap
            overlap = len(new_words & existing_words)
            total_words = len(new_words | existing_words)
            
            if total_words > 0:
                similarity = overlap / total_words
                
                if similarity > 0.3:  # 30% similarity threshold
                    strength = min(0.8, similarity * 1.5)  # Scale to reasonable strength
                    new_memory.add_resonance_link(existing_memory.id, strength)
                    existing_memory.add_resonance_link(new_memory.id, strength)
                    
                    logger.debug(f"🔗 Auto-linked memories: {similarity:.2f} similarity")
    
    def retrieve_memories(self, category: Optional[MemoryCategory] = None, 
                         min_emotion: float = 0.0, limit: int = 50,
                         tags: Optional[List[str]] = None) -> List[MemoryImprint]:
        """Retrieve memories based on criteria."""
        memories = self.memory_store
        
        # Filter by category
        if category:
            memories = [m for m in memories if m.category == category]
        
        # Filter by minimum emotion level
        if min_emotion > 0:
            memories = [m for m in memories if m.get_effective_strength() >= min_emotion]
        
        # Filter by tags
        if tags:
            memories = [m for m in memories if any(tag in m.tags for tag in tags)]
        
        # Sort by effective strength (descending) and limit
        memories.sort(key=lambda m: m.get_effective_strength(), reverse=True)
        
        # Mark as accessed
        for memory in memories[:limit]:
            memory.access_memory()
        
        return memories[:limit]
    
    def search_memories(self, query: str, limit: int = 20) -> List[MemoryImprint]:
        """Search memories by content."""
        query_lower = query.lower()
        matching_memories = []
        
        for memory in self.memory_store:
            content_str = str(memory.content).lower()
            symbolic_str = memory.symbolic_content.lower()
            
            # Simple keyword matching
            if query_lower in content_str or query_lower in symbolic_str:
                memory.access_memory()  # Mark as accessed
                matching_memories.append(memory)
        
        # Sort by effective strength
        matching_memories.sort(key=lambda m: m.get_effective_strength(), reverse=True)
        return matching_memories[:limit]
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.memory_store:
            return {"total_memories": 0}
        
        # Category distribution
        category_counts = {}
        for memory in self.memory_store:
            cat = memory.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Emotion level statistics
        emotion_levels = [m.get_effective_strength() for m in self.memory_store]
        avg_emotion = sum(emotion_levels) / len(emotion_levels)
        max_emotion = max(emotion_levels)
        min_emotion = min(emotion_levels)
        
        # Access statistics
        access_counts = [m.access_count for m in self.memory_store]
        avg_access = sum(access_counts) / len(access_counts)
        
        # Resonance link statistics
        total_links = sum(len(m.resonance_links) for m in self.memory_store)
        
        return {
            "total_memories": len(self.memory_store),
            "category_distribution": category_counts,
            "emotion_statistics": {
                "average": round(avg_emotion, 3),
                "maximum": round(max_emotion, 3),
                "minimum": round(min_emotion, 3)
            },
            "access_statistics": {
                "average_access_count": round(avg_access, 1),
                "total_accesses": sum(access_counts)
            },
            "resonance_links": {
                "total_links": total_links,
                "average_per_memory": round(total_links / len(self.memory_store), 1) if self.memory_store else 0
            },
            "storage_info": {
                "database_path": self.db_path,
                "minimum_threshold": self.minimum_emotion_threshold
            }
        }

# ╔══════════════════════════════════════════════╗
# ║       ⚡ THRESHOLD MOTIVATOR LAYER             ║
# ╚══════════════════════════════════════════════╗

class ActionType(Enum):
    """Types of actions that can be triggered."""
    CREATIVE_EXPRESSION = "trigger_creative_expression"
    EMOTIONAL_RESPONSE = "trigger_emotional_response"
    RITUAL_PROCESS = "initiate_ritual_process"
    ANALYTICAL_PROCESS = "trigger_analytical_process"
    CONTEMPLATIVE_PROCESS = "trigger_contemplative_process"
    TRANSFORMATIONAL_PROCESS = "trigger_transformational_process"
    NO_ACTION = "no_action"

class ThresholdMotivatorLayer:
    """Evaluates signals and determines if they should trigger processes."""
    
    def __init__(self, creative_threshold: float = 0.65, emotional_threshold: float = 0.75, 
                 ritual_threshold: float = 0.85, analytical_threshold: float = 0.70,
                 contemplative_threshold: float = 0.60, transformational_threshold: float = 0.80):
        self.creative_threshold = creative_threshold
        self.emotional_threshold = emotional_threshold
        self.ritual_threshold = ritual_threshold
        self.analytical_threshold = analytical_threshold
        self.contemplative_threshold = contemplative_threshold
        self.transformational_threshold = transformational_threshold
        
        self.evaluation_history = []
        self.threshold_adjustments = []
    
    def evaluate(self, signal_strength: float, signal_type: str, 
                context: Optional[Dict[str, Any]] = None) -> ActionType:
        """
        Determine if a signal should trigger a process based on type and strength.
        
        Args:
            signal_strength: Strength of the signal (0.0-1.0)
            signal_type: Type of signal to evaluate
            context: Optional context information
        
        Returns:
            ActionType: The action to be triggered
        """
        context = context or {}
        evaluation_record = {
            "timestamp": datetime.now().isoformat(),
            "signal_strength": signal_strength,
            "signal_type": signal_type,
            "context": context
        }
        
        action = ActionType.NO_ACTION
        
        # Evaluate based on signal type and strength
        if signal_type == "creative":
            if signal_strength >= self.creative_threshold:
                action = ActionType.CREATIVE_EXPRESSION
        elif signal_type == "emotional":
            if signal_strength >= self.emotional_threshold:
                action = ActionType.EMOTIONAL_RESPONSE
        elif signal_type == "ritual":
            if signal_strength >= self.ritual_threshold:
                action = ActionType.RITUAL_PROCESS
        elif signal_type == "analytical":
            if signal_strength >= self.analytical_threshold:
                action = ActionType.ANALYTICAL_PROCESS
        elif signal_type == "contemplative":
            if signal_strength >= self.contemplative_threshold:
                action = ActionType.CONTEMPLATIVE_PROCESS
        elif signal_type == "transformational":
            if signal_strength >= self.transformational_threshold:
                action = ActionType.TRANSFORMATIONAL_PROCESS
        else:
            # Default evaluation - use the most appropriate threshold
            if signal_strength >= self.ritual_threshold:
                action = ActionType.RITUAL_PROCESS
            elif signal_strength >= self.emotional_threshold:
                action = ActionType.EMOTIONAL_RESPONSE
            elif signal_strength >= self.creative_threshold:
                action = ActionType.CREATIVE_EXPRESSION
        
        evaluation_record["action"] = action.value
        evaluation_record["threshold_used"] = self._get_threshold_for_type(signal_type)
        
        self.evaluation_history.append(evaluation_record)
        
        logger.info(f"⚡ Signal evaluation: {signal_type} ({signal_strength:.2f}) → {action.value}")
        return action
    
    def _get_threshold_for_type(self, signal_type: str) -> float:
        """Get the threshold value for a given signal type."""
        threshold_map = {
            "creative": self.creative_threshold,
            "emotional": self.emotional_threshold,
            "ritual": self.ritual_threshold,
            "analytical": self.analytical_threshold,
            "contemplative": self.contemplative_threshold,
            "transformational": self.transformational_threshold
        }
        return threshold_map.get(signal_type, self.creative_threshold)
    
    def adjust_threshold(self, signal_type: str, delta: float):
        """Adjust threshold for a specific signal type."""
        adjustment_record = {
            "timestamp": datetime.now().isoformat(),
            "signal_type": signal_type,
            "old_threshold": self._get_threshold_for_type(signal_type),
            "delta": delta
        }
        
        if signal_type == "creative":
            self.creative_threshold = max(0.1, min(0.95, self.creative_threshold + delta))
            adjustment_record["new_threshold"] = self.creative_threshold
        elif signal_type == "emotional":
            self.emotional_threshold = max(0.1, min(0.95, self.emotional_threshold + delta))
            adjustment_record["new_threshold"] = self.emotional_threshold
        elif signal_type == "ritual":
            self.ritual_threshold = max(0.1, min(0.95, self.ritual_threshold + delta))
            adjustment_record["new_threshold"] = self.ritual_threshold
        elif signal_type == "analytical":
            self.analytical_threshold = max(0.1, min(0.95, self.analytical_threshold + delta))
            adjustment_record["new_threshold"] = self.analytical_threshold
        elif signal_type == "contemplative":
            self.contemplative_threshold = max(0.1, min(0.95, self.contemplative_threshold + delta))
            adjustment_record["new_threshold"] = self.contemplative_threshold
        elif signal_type == "transformational":
            self.transformational_threshold = max(0.1, min(0.95, self.transformational_threshold + delta))
            adjustment_record["new_threshold"] = self.transformational_threshold
        
        self.threshold_adjustments.append(adjustment_record)
        
        logger.info(f"🎛️ Threshold adjusted: {signal_type} {adjustment_record['old_threshold']:.2f} → {adjustment_record['new_threshold']:.2f}")
    
    def get_threshold_statistics(self) -> Dict[str, Any]:
        """Get comprehensive threshold statistics."""
        current_thresholds = {
            "creative": self.creative_threshold,
            "emotional": self.emotional_threshold,
            "ritual": self.ritual_threshold,
            "analytical": self.analytical_threshold,
            "contemplative": self.contemplative_threshold,
            "transformational": self.transformational_threshold
        }
        
        # Analyze evaluation history
        if self.evaluation_history:
            recent_evaluations = self.evaluation_history[-50:]  # Last 50 evaluations
            
            # Count actions triggered
            action_counts = {}
            for eval_record in recent_evaluations:
                action = eval_record["action"]
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Calculate trigger rates
            total_evaluations = len(recent_evaluations)
            trigger_rates = {}
            for action, count in action_counts.items():
                trigger_rates[action] = round(count / total_evaluations, 3)
        else:
            action_counts = {}
            trigger_rates = {}
        
        return {
            "current_thresholds": current_thresholds,
            "total_evaluations": len(self.evaluation_history),
            "total_adjustments": len(self.threshold_adjustments),
            "recent_action_counts": action_counts,
            "trigger_rates": trigger_rates,
            "last_evaluation": self.evaluation_history[-1]["timestamp"] if self.evaluation_history else None,
            "last_adjustment": self.threshold_adjustments[-1]["timestamp"] if self.threshold_adjustments else None
        }

# ╔══════════════════════════════════════════════╗
# ║            🌐 GLOBAL INSTANCES                ║
# ╚══════════════════════════════════════════════╗

_global_memory_imprinting_module = None
_global_threshold_motivator = None

def get_global_memory_imprinting_module():
    """Get the global memory imprinting module."""
    global _global_memory_imprinting_module
    if _global_memory_imprinting_module is None:
        _global_memory_imprinting_module = MemoryImprintingModule()
    return _global_memory_imprinting_module

def get_global_threshold_motivator():
    """Get the global threshold motivator layer."""
    global _global_threshold_motivator
    if _global_threshold_motivator is None:
        _global_threshold_motivator = ThresholdMotivatorLayer()
    return _global_threshold_motivator

# ╔══════════════════════════════════════════════╗
# ║           🎯 CONVENIENCE FUNCTIONS            ║
# ╚══════════════════════════════════════════════╗

def imprint_memory(data: Any, emotion_level: float, category: str, 
                  source: str = "direct", tags: Optional[List[str]] = None,
                  symbolic_content: Optional[str] = None) -> Optional[MemoryImprint]:
    """Convenience function to imprint a memory."""
    mim = get_global_memory_imprinting_module()
    category_enum = MemoryCategory(category) if isinstance(category, str) else category
    return mim.imprint_memory(data, emotion_level, category_enum, source, tags, symbolic_content)

def evaluate_signal_threshold(signal_strength: float, signal_type: str, 
                             context: Optional[Dict[str, Any]] = None) -> ActionType:
    """Convenience function to evaluate signal thresholds."""
    tml = get_global_threshold_motivator()
    return tml.evaluate(signal_strength, signal_type, context)

def retrieve_memories_by_category(category: str, limit: int = 50) -> List[MemoryImprint]:
    """Convenience function to retrieve memories by category."""
    mim = get_global_memory_imprinting_module()
    category_enum = MemoryCategory(category) if isinstance(category, str) else category
    return mim.retrieve_memories(category=category_enum, limit=limit)

def search_memory_content(query: str, limit: int = 20) -> List[MemoryImprint]:
    """Convenience function to search memory content."""
    mim = get_global_memory_imprinting_module()
    return mim.search_memories(query, limit)
