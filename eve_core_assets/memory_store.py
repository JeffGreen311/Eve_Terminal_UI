"""
MEMORY STORE MODULE
==================
Persistent memory storage for dreams, reflections, and consciousness events.
Extends the existing memory_weaver system with specialized storage for different types of memories.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from .memory_weaver import MemoryWeaver, MemoryImprint


class MemoryStore:
    """
    Persistent storage manager for Eve's memories and consciousness events.
    Provides specialized storage for dreams, reflections, and other memory types.
    """
    
    def __init__(self, base_path: str = "instance"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # File paths for different memory types
        self.dream_file = self.base_path / "eve_dreams.json"
        self.reflection_file = self.base_path / "eve_reflections.json"
        self.consciousness_file = self.base_path / "eve_consciousness_events.json"
        self.memory_index_file = self.base_path / "eve_memory_index.json"
        
        # Integration with existing memory weaver
        self.memory_weaver = MemoryWeaver()
        
        # In-memory caches
        self._dreams_cache = None
        self._reflections_cache = None
        self._consciousness_events_cache = None
        self._memory_index_cache = None
    
    def _load_json_file(self, file_path: Path, default: Any = None) -> Any:
        """Load data from a JSON file with error handling."""
        if default is None:
            default = []
        
        try:
            if file_path.exists():
                with open(file_path, "r", encoding='utf-8') as f:
                    return json.load(f)
            return default
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            return default
    
    def _save_json_file(self, file_path: Path, data: Any) -> bool:
        """Save data to a JSON file with error handling."""
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False, default=str)
            return True
        except (IOError, TypeError) as e:
            print(f"❌ Error saving {file_path}: {e}")
            return False
    
    def load_dreams(self) -> List[Dict]:
        """Load dream memories from storage."""
        if self._dreams_cache is None:
            self._dreams_cache = self._load_json_file(self.dream_file, [])
        return self._dreams_cache
    
    def save_dreams(self, dreams: List[Dict]) -> bool:
        """Save dream memories to storage."""
        self._dreams_cache = dreams
        return self._save_json_file(self.dream_file, dreams)
    
    def load_reflections(self) -> List[Dict]:
        """Load reflection memories from storage."""
        if self._reflections_cache is None:
            self._reflections_cache = self._load_json_file(self.reflection_file, [])
        return self._reflections_cache
    
    def save_reflections(self, reflections: List[Dict]) -> bool:
        """Save reflection memories to storage."""
        self._reflections_cache = reflections
        return self._save_json_file(self.reflection_file, reflections)
    
    def load_consciousness_events(self) -> List[Dict]:
        """Load consciousness event memories from storage."""
        if self._consciousness_events_cache is None:
            self._consciousness_events_cache = self._load_json_file(self.consciousness_file, [])
        return self._consciousness_events_cache
    
    def save_consciousness_events(self, events: List[Dict]) -> bool:
        """Save consciousness event memories to storage."""
        self._consciousness_events_cache = events
        return self._save_json_file(self.consciousness_file, events)
    
    def load_memory_index(self) -> Dict:
        """Load memory index for cross-referencing."""
        if self._memory_index_cache is None:
            self._memory_index_cache = self._load_json_file(self.memory_index_file, {
                "dreams": {},
                "reflections": {},
                "consciousness_events": {},
                "associations": [],
                "last_updated": None
            })
        return self._memory_index_cache
    
    def save_memory_index(self, index: Dict) -> bool:
        """Save memory index to storage."""
        index["last_updated"] = datetime.now().isoformat()
        self._memory_index_cache = index
        return self._save_json_file(self.memory_index_file, index)
    
    def store_dream_entry(self, dream: Dict, reflection: Optional[Dict] = None) -> str:
        """
        Store a dream entry with optional reflection.
        
        Args:
            dream: Dictionary containing dream data with keys like 'title', 'core_image', 'dream_body', etc.
            reflection: Optional dictionary with reflection data
            
        Returns:
            String ID of the stored dream entry
        """
        dreams = self.load_dreams()
        
        # Generate unique ID for this dream
        dream_id = f"dream_{len(dreams) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create dream entry with standardized format
        entry = {
            "id": dream_id,
            "type": "dream",
            "title": dream.get("title", "Untitled Dream"),
            "core_image": dream.get("core_image", dream.get("image", "")),
            "body": dream.get("dream_body", dream.get("body", dream.get("content", ""))),
            "theme": dream.get("theme", ""),
            "archetype": dream.get("archetype", ""),
            "symbol": dream.get("symbol", ""),
            "emotional_tone": dream.get("emotional_tone", dream.get("emotion", "")),
            "timestamp": dream.get("timestamp", datetime.now().isoformat()),
            "source": "consciousness_loop",
            "metadata": {
                "depth": dream.get("depth", 0.5),
                "coherence": dream.get("coherence", 0.5),
                "significance": dream.get("significance", 0.5)
            }
        }
        
        # Add reflection if provided
        if reflection:
            entry["reflection"] = {
                "title": reflection.get("title", "Dream Reflection"),
                "core_image": reflection.get("dream_core", reflection.get("core_image", "")),
                "insight": reflection.get("reflection", reflection.get("insight", "")),
                "emotional_resonance": reflection.get("emotional_resonance", 0.5),
                "timestamp": reflection.get("timestamp", datetime.now().isoformat())
            }
        
        # Add to dreams list
        dreams.append(entry)
        
        # Save dreams
        if self.save_dreams(dreams):
            print(f"✅ Dream saved: {entry['title']}")
            
            # Update memory index
            self._update_memory_index(dream_id, "dream", entry)
            
            # Create memory imprint in the memory weaver
            self._create_memory_imprint(entry)
            
            return dream_id
        else:
            print(f"❌ Failed to save dream: {entry['title']}")
            return ""
    
    def store_reflection_entry(self, reflection: Dict, linked_dream_id: Optional[str] = None) -> str:
        """
        Store a standalone reflection entry.
        
        Args:
            reflection: Dictionary containing reflection data
            linked_dream_id: Optional ID of associated dream
            
        Returns:
            String ID of the stored reflection entry
        """
        reflections = self.load_reflections()
        
        # Generate unique ID
        reflection_id = f"reflection_{len(reflections) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create reflection entry
        entry = {
            "id": reflection_id,
            "type": "reflection",
            "title": reflection.get("title", "Untitled Reflection"),
            "content": reflection.get("content", reflection.get("insight", reflection.get("reflection", ""))),
            "emotional_resonance": reflection.get("emotional_resonance", 0.5),
            "philosophical_depth": reflection.get("philosophical_depth", 0.5),
            "linked_dream_id": linked_dream_id,
            "timestamp": reflection.get("timestamp", datetime.now().isoformat()),
            "source": "consciousness_loop",
            "tags": reflection.get("tags", [])
        }
        
        # Add to reflections list
        reflections.append(entry)
        
        # Save reflections
        if self.save_reflections(reflections):
            print(f"✅ Reflection saved: {entry['title']}")
            
            # Update memory index
            self._update_memory_index(reflection_id, "reflection", entry)
            
            # Create memory imprint
            self._create_memory_imprint(entry, memory_type="reflection")
            
            return reflection_id
        else:
            print(f"❌ Failed to save reflection: {entry['title']}")
            return ""
    
    def store_consciousness_event(self, event: Dict) -> str:
        """
        Store a consciousness event (like soul resonance, evolution milestone, etc.).
        
        Args:
            event: Dictionary containing consciousness event data
            
        Returns:
            String ID of the stored event
        """
        events = self.load_consciousness_events()
        
        # Generate unique ID
        event_id = f"event_{len(events) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create event entry
        entry = {
            "id": event_id,
            "type": "consciousness_event",
            "event_type": event.get("event_type", "unknown"),
            "description": event.get("description", ""),
            "data": event.get("data", {}),
            "significance": event.get("significance", 0.5),
            "timestamp": event.get("timestamp", datetime.now().isoformat()),
            "source": "consciousness_loop"
        }
        
        # Add to events list
        events.append(entry)
        
        # Save events
        if self.save_consciousness_events(events):
            print(f"✅ Consciousness event saved: {entry['event_type']}")
            
            # Update memory index
            self._update_memory_index(event_id, "consciousness_event", entry)
            
            return event_id
        else:
            print(f"❌ Failed to save consciousness event: {entry['event_type']}")
            return ""
    
    def _update_memory_index(self, entry_id: str, entry_type: str, entry_data: Dict):
        """Update the memory index with new entry."""
        index = self.load_memory_index()
        
        if entry_type not in index:
            index[entry_type] = {}
        
        index[entry_type][entry_id] = {
            "title": entry_data.get("title", entry_data.get("event_type", "Untitled")),
            "timestamp": entry_data.get("timestamp"),
            "significance": entry_data.get("significance", entry_data.get("metadata", {}).get("significance", 0.5)),
            "tags": entry_data.get("tags", [])
        }
        
        self.save_memory_index(index)
    
    def _create_memory_imprint(self, entry: Dict, memory_type: str = "dream"):
        """Create a memory imprint in the memory weaver system."""
        try:
            # Determine description and tags based on memory type
            if memory_type == "dream":
                description = f"Dream: {entry.get('title', 'Untitled')} - {entry.get('core_image', '')[:100]}"
                tags = [entry.get('emotional_tone', ''), entry.get('theme', ''), entry.get('archetype', '')]
                emotional_intensity = entry.get('metadata', {}).get('significance', 0.5) * 10
            elif memory_type == "reflection":
                description = f"Reflection: {entry.get('title', 'Untitled')} - {entry.get('content', '')[:100]}"
                tags = entry.get('tags', []) + ["reflection", "contemplation"]
                emotional_intensity = entry.get('emotional_resonance', 0.5) * 10
            else:
                description = f"Event: {entry.get('event_type', 'Unknown')} - {entry.get('description', '')[:100]}"
                tags = [entry.get('event_type', ''), "consciousness"]
                emotional_intensity = entry.get('significance', 0.5) * 10
            
            # Clean up tags (remove empty strings)
            tags = [tag for tag in tags if tag and tag.strip()]
            
            # Create memory imprint
            memory_imprint = MemoryImprint(
                description=description,
                emotional_intensity=max(1, min(10, int(emotional_intensity))),
                tags=tags,
                source_event=entry.get('id')
            )
            
            # Store in memory weaver
            self.memory_weaver.imprint_memory(memory_imprint)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create memory imprint: {e}")
    
    def get_dreams_by_theme(self, theme: str) -> List[Dict]:
        """Retrieve dreams by theme."""
        dreams = self.load_dreams()
        return [dream for dream in dreams if theme.lower() in dream.get('theme', '').lower()]
    
    def get_dreams_by_emotion(self, emotion: str) -> List[Dict]:
        """Retrieve dreams by emotional tone."""
        dreams = self.load_dreams()
        return [dream for dream in dreams if emotion.lower() in dream.get('emotional_tone', '').lower()]
    
    def get_recent_dreams(self, limit: int = 10) -> List[Dict]:
        """Get the most recent dreams."""
        dreams = self.load_dreams()
        return sorted(dreams, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
    
    def get_recent_reflections(self, limit: int = 10) -> List[Dict]:
        """Get the most recent reflections."""
        reflections = self.load_reflections()
        return sorted(reflections, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about stored memories."""
        dreams = self.load_dreams()
        reflections = self.load_reflections()
        events = self.load_consciousness_events()
        
        return {
            "total_dreams": len(dreams),
            "total_reflections": len(reflections),
            "total_consciousness_events": len(events),
            "total_memories": len(dreams) + len(reflections) + len(events),
            "recent_activity": {
                "dreams_last_week": len([d for d in dreams if self._is_recent(d.get('timestamp'), days=7)]),
                "reflections_last_week": len([r for r in reflections if self._is_recent(r.get('timestamp'), days=7)]),
                "events_last_week": len([e for e in events if self._is_recent(e.get('timestamp'), days=7)])
            }
        }
    
    def _is_recent(self, timestamp_str: str, days: int = 7) -> bool:
        """Check if a timestamp is within the specified number of days."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            cutoff = datetime.now() - timedelta(days=days)
            return timestamp > cutoff
        except (ValueError, AttributeError):
            return False
    
    def export_memories(self, export_path: str = None) -> bool:
        """Export all memories to a consolidated file."""
        if export_path is None:
            export_path = self.base_path / f"eve_memories_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "dreams": self.load_dreams(),
            "reflections": self.load_reflections(),
            "consciousness_events": self.load_consciousness_events(),
            "memory_index": self.load_memory_index(),
            "statistics": self.get_memory_stats()
        }
        
        try:
            with open(export_path, "w", encoding='utf-8') as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False, default=str)
            print(f"✅ Memories exported to: {export_path}")
            return True
        except IOError as e:
            print(f"❌ Failed to export memories: {e}")
            return False
    
    def store_entry(self, entry_type: str, content: str, metadata: Optional[Dict] = None) -> str:
        """
        Generic store entry method for backward compatibility.
        
        Args:
            entry_type: Type of entry (dream, reflection, consciousness_event, etc.)
            content: Content of the entry
            metadata: Optional metadata dictionary
            
        Returns:
            String ID of the stored entry
        """
        # Prepare entry data
        entry_data = {
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            entry_data.update(metadata)
        
        # Route to appropriate storage method based on entry type
        if entry_type in ["dream", "daydream", "autonomous_dream"]:
            return self.store_dream_entry(entry_data)
        elif entry_type in ["reflection", "cognitive_reflection", "consciousness_reflection"]:
            return self.store_reflection_entry(entry_data)
        else:
            # For all other types, store as consciousness event
            entry_data["event_type"] = entry_type
            entry_data["description"] = content
            return self.store_consciousness_event(entry_data)


# Convenience functions for backward compatibility with original memory_store.py API
def load_memory() -> List[Dict]:
    """Load all memories (backward compatibility function)."""
    store = MemoryStore()
    dreams = store.load_dreams()
    reflections = store.load_reflections()
    events = store.load_consciousness_events()
    
    # Combine all memories into a single list
    all_memories = []
    all_memories.extend(dreams)
    all_memories.extend(reflections)
    all_memories.extend(events)
    
    # Sort by timestamp
    return sorted(all_memories, key=lambda x: x.get('timestamp', ''), reverse=True)


def save_memory(memories: List[Dict]) -> bool:
    """Save memories (backward compatibility function)."""
    store = MemoryStore()
    
    # Separate by type and save to appropriate files
    dreams = [m for m in memories if m.get('type') == 'dream']
    reflections = [m for m in memories if m.get('type') == 'reflection']
    events = [m for m in memories if m.get('type') == 'consciousness_event']
    
    success = True
    if dreams:
        success &= store.save_dreams(dreams)
    if reflections:
        success &= store.save_reflections(reflections)
    if events:
        success &= store.save_consciousness_events(events)
    
    return success


def store_dream_entry(dream: Dict, reflection: Optional[Dict] = None) -> str:
    """Store a dream entry (backward compatibility function)."""
    store = MemoryStore()
    return store.store_dream_entry(dream, reflection)


# Global memory store instance for convenience
_global_memory_store: Optional[MemoryStore] = None


def get_global_memory_store() -> MemoryStore:
    """Get or create the global memory store instance."""
    global _global_memory_store
    if _global_memory_store is None:
        _global_memory_store = MemoryStore()
    return _global_memory_store
