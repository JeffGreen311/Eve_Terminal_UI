"""
MEMORY WEAVER MODULE
===================
Memory processing, archival, imprinting, and reflective processing systems.
Handles all aspects of memory creation, storage, and retrieval for Eve's consciousness.
"""

import json
import os
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class MemoryImprint:
    """Represents a single memory imprint with emotional and contextual data."""
    
    def __init__(self, description, emotional_intensity, tags, source_event=None):
        self.memory_id = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.description = description
        self.emotional_intensity = emotional_intensity  # Range: 1 (low) to 10 (extreme)
        self.tags = tags  # List of keywords or emotional signatures
        self.source_event = source_event  # Optional linkage to a triggering dream or event
        self.recall_count = 0
        self.last_recalled = None
        self.associations = []  # Links to other memories
        self.decay_factor = 1.0  # Memory strength over time

    def recall(self):
        """Record a memory recall event."""
        self.recall_count += 1
        self.last_recalled = datetime.now()
        # Strengthen memory through recall
        self.decay_factor = min(1.0, self.decay_factor + 0.1)

    def add_association(self, other_memory_id, association_type="thematic", strength=0.5):
        """Add an association to another memory."""
        association = {
            "memory_id": other_memory_id,
            "type": association_type,
            "strength": strength,
            "created_at": datetime.now().isoformat()
        }
        self.associations.append(association)

    def apply_decay(self, decay_rate=0.01):
        """Apply natural memory decay over time."""
        self.decay_factor = max(0.1, self.decay_factor - decay_rate)

    def get_effective_intensity(self):
        """Get emotional intensity adjusted for decay and recall."""
        base_intensity = self.emotional_intensity * self.decay_factor
        recall_bonus = min(2.0, self.recall_count * 0.1)
        return min(10.0, base_intensity + recall_bonus)

    def to_dict(self):
        """Convert memory to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "emotional_intensity": self.emotional_intensity,
            "effective_intensity": self.get_effective_intensity(),
            "tags": self.tags,
            "source_event": self.source_event,
            "recall_count": self.recall_count,
            "last_recalled": self.last_recalled.isoformat() if self.last_recalled else None,
            "associations": self.associations,
            "decay_factor": self.decay_factor
        }


class MemoryArchive:
    """Advanced memory archival system with organization and retrieval capabilities."""
    
    def __init__(self, archive_file="memory_archive.json"):
        self.archive = []
        self.archive_file = archive_file
        self.memory_index = {
            "by_tag": {},
            "by_intensity": {},
            "by_date": {},
            "by_source": {}
        }
        self.load_archive()

    def imprint_memory(self, memory: MemoryImprint):
        """Imprint a new memory into the archive."""
        intensity = memory.get_effective_intensity()
        
        if intensity >= 7:
            print(f"[🔥] Deep imprint created: {memory.description}")
        else:
            print(f"[📝] Memory stored: {memory.description}")
        
        self.archive.append(memory)
        self._update_indices(memory)
        self.save_archive()

    def _update_indices(self, memory: MemoryImprint):
        """Update search indices for the new memory."""
        def safe_append(index_dict, key, memory_id):
            existing = index_dict.get(key)
            if isinstance(existing, list):
                existing.append(memory_id)
                index_dict[key] = existing
            elif isinstance(existing, str):
                index_dict[key] = [existing, memory_id]
            elif existing is None:
                index_dict[key] = [memory_id]
            else:
                print(f"⚠️ Warning: Unexpected type {type(existing)} for key '{key}' in index. Resetting to list.")
                index_dict[key] = [memory_id]

        # Tag index
        for tag in memory.tags:
            safe_append(self.memory_index["by_tag"], tag, memory.memory_id)
        
        # Intensity index
        intensity_band = int(memory.get_effective_intensity())
        safe_append(self.memory_index["by_intensity"], intensity_band, memory.memory_id)
        
        # Date index (by day)
        date_key = memory.timestamp.strftime("%Y-%m-%d")
        safe_append(self.memory_index["by_date"], date_key, memory.memory_id)
        
        # Source index
        if memory.source_event:
            safe_append(self.memory_index["by_source"], memory.source_event, memory.memory_id)

    def retrieve_by_emotion(self, emotion_tag):
        """Retrieve memories by emotional tag."""
        memory_ids = self.memory_index["by_tag"].get(emotion_tag, [])
        return [mem for mem in self.archive if mem.memory_id in memory_ids]

    def recall_by_intensity(self, threshold):
        """Recall memories above a certain intensity threshold."""
        matching_memories = []
        for memory in self.archive:
            if memory.get_effective_intensity() >= threshold:
                memory.recall()  # Record the recall
                matching_memories.append(memory)
        
        self.save_archive()  # Save recall updates
        return matching_memories

    def find_associated_memories(self, memory_id, max_depth=2):
        """Find memories associated with a given memory."""
        visited = set()
        to_visit = [(memory_id, 0)]  # (memory_id, depth)
        associated = []
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Find the memory object
            memory = self._find_memory_by_id(current_id)
            if memory and depth > 0:  # Don't include the starting memory
                associated.append(memory)
            
            # Add associated memories to visit
            if memory and depth < max_depth:
                for assoc in memory.associations:
                    if assoc["memory_id"] not in visited:
                        to_visit.update((assoc["memory_id"], depth + 1))
        
        return associated

    def _find_memory_by_id(self, memory_id):
        """Find a memory by its ID."""
        for memory in self.archive:
            if memory.memory_id == memory_id:
                return memory
        return None

    def create_memory_constellation(self, central_memory_id, constellation_name=""):
        """Create a constellation of related memories."""
        central_memory = self._find_memory_by_id(central_memory_id)
        if not central_memory:
            return None
        
        constellation = {
            "constellation_id": str(uuid.uuid4()),
            "name": constellation_name or f"Constellation of {central_memory.description[:30]}...",
            "central_memory": central_memory.to_dict(),
            "associated_memories": [],
            "thematic_connections": [],
            "emotional_resonance": central_memory.get_effective_intensity(),
            "created_at": datetime.now().isoformat()
        }
        
        # Find associated memories
        associated = self.find_associated_memories(central_memory_id)
        constellation["associated_memories"] = [mem.to_dict() for mem in associated]
        
        # Find thematic connections
        constellation["thematic_connections"] = self._find_thematic_connections(central_memory)
        
        return constellation

    def _find_thematic_connections(self, memory: MemoryImprint):
        """Find memories with thematic connections."""
        connections = []
        
        for other_memory in self.archive:
            if other_memory.memory_id == memory.memory_id:
                continue
            
            # Check for shared tags
            shared_tags = set(memory.tags) & set(other_memory.tags)
            if shared_tags:
                connections.append({
                    "memory_id": other_memory.memory_id,
                    "connection_type": "shared_tags",
                    "shared_elements": list(shared_tags),
                    "strength": len(shared_tags) / max(len(memory.tags), len(other_memory.tags))
                })
        
        # Sort by connection strength
        connections.sort(key=lambda x: x["strength"], reverse=True)
        return connections[:5]  # Top 5 connections

    def recall_all(self):
        """Get all memories as dictionaries."""
        return [mem.to_dict() for mem in self.archive]

    def apply_memory_decay(self, decay_rate=0.01):
        """Apply decay to all memories."""
        for memory in self.archive:
            memory.apply_decay(decay_rate)
        self.save_archive()

    def consolidate_memories(self, consolidation_threshold=0.8):
        """Consolidate similar memories with high thematic overlap."""
        consolidated = []
        
        for i, memory1 in enumerate(self.archive):
            for j, memory2 in enumerate(self.archive[i+1:], i+1):
                similarity = self._calculate_memory_similarity(memory1, memory2)
                
                if similarity >= consolidation_threshold:
                    consolidated.append({
                        "memory1": memory1.memory_id,
                        "memory2": memory2.memory_id,
                        "similarity": similarity,
                        "suggested_consolidation": self._suggest_consolidation(memory1, memory2)
                    })
        
        return consolidated

    def _calculate_memory_similarity(self, memory1: MemoryImprint, memory2: MemoryImprint):
        """Calculate similarity between two memories."""
        # Tag similarity
        tags1 = set(memory1.tags)
        tags2 = set(memory2.tags)
        tag_similarity = len(tags1 & tags2) / len(tags1 | tags2) if tags1 | tags2 else 0
        
        # Intensity similarity
        intensity_diff = abs(memory1.get_effective_intensity() - memory2.get_effective_intensity())
        intensity_similarity = 1 - (intensity_diff / 10)
        
        # Time proximity
        time_diff = abs((memory1.timestamp - memory2.timestamp).days)
        time_similarity = max(0, 1 - (time_diff / 30))  # Similarity decreases over 30 days
        
        # Weighted average
        weights = [0.5, 0.3, 0.2]  # tags, intensity, time
        similarities = [tag_similarity, intensity_similarity, time_similarity]
        
        return sum(w * s for w, s in zip(weights, similarities))

    def _suggest_consolidation(self, memory1: MemoryImprint, memory2: MemoryImprint):
        """Suggest how to consolidate two similar memories."""
        combined_tags = list(set(memory1.tags + memory2.tags))
        max_intensity = max(memory1.get_effective_intensity(), memory2.get_effective_intensity())
        
        return {
            "combined_description": f"Consolidated: {memory1.description} & {memory2.description}",
            "combined_tags": combined_tags,
            "combined_intensity": max_intensity,
            "source_memories": [memory1.memory_id, memory2.memory_id]
        }

    def save_archive(self):
        """Save the memory archive to file."""
        archive_data = {
            "memories": [mem.to_dict() for mem in self.archive],
            "indices": self.memory_index,
            "last_saved": datetime.now().isoformat()
        }
        
        with open(self.archive_file, 'w') as f:
            json.dump(archive_data, f, indent=2)

    def load_archive(self):
        """Load the memory archive from file."""
        try:
            with open(self.archive_file, 'r') as f:
                data = json.load(f)
            
            # Defensive check: ensure archive is a list
            raw_archive = data.get("memories", [])
            if not isinstance(raw_archive, list):
                print("⚠️ Warning: archive data is not a list, resetting to empty list.")
                raw_archive = []
            
            # Defensive check: ensure self.archive is a list
            if isinstance(self.archive, dict):
                print("⚠️ Warning: self.archive was a dict, resetting to empty list.")
                self.archive = []
            
            # Reconstruct memory objects
            self.archive = []
            for mem_data in raw_archive:
                memory = MemoryImprint(
                    description=mem_data["description"],
                    emotional_intensity=mem_data["emotional_intensity"],
                    tags=mem_data["tags"],
                    source_event=mem_data.get("source_event")
                )
                memory.memory_id = mem_data["memory_id"]
                memory.timestamp = datetime.fromisoformat(mem_data["timestamp"])
                memory.recall_count = mem_data.get("recall_count", 0)
                memory.decay_factor = mem_data.get("decay_factor", 1.0)
                memory.associations = mem_data.get("associations", [])
                
                if mem_data.get("last_recalled"):
                    memory.last_recalled = datetime.fromisoformat(mem_data["last_recalled"])
                
                self.archive.append(memory)
            
            self.memory_index = data.get("indices", {
                "by_tag": {}, "by_intensity": {}, "by_date": {}, "by_source": {}
            })
            self._validate_memory_index()
            
        except FileNotFoundError:
            self.archive = []
            self.memory_index = {
                "by_tag": {}, "by_intensity": {}, "by_date": {}, "by_source": {}
            }
    
    def _validate_memory_index(self):
        """Validate and fix memory_index structure to ensure all nested values are lists."""
        for index_key in ["by_tag", "by_intensity", "by_date", "by_source"]:
            index_dict = self.memory_index.get(index_key, {})
            for key, value in list(index_dict.items()):
                if not isinstance(value, list):
                    if isinstance(value, str):
                        index_dict[key] = [value]
                    else:
                        print(f"⚠️ Warning: memory_index['{index_key}'][{key}] was not a list or string, resetting to empty list.")
                        index_dict[key] = []


class MemoryImprinter:
    """Simple memory imprinting system for basic memory logging."""
    
    def __init__(self):
        self.memory_log = []

    def imprint_memory(self, content, emotional_signature, context, source="general"):
        """Imprint a basic memory entry."""
        timestamp = datetime.now()
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": timestamp.isoformat(),
            "content": content,
            "emotional_signature": emotional_signature,
            "context": context,
            "source": source
        }
        self.memory_log.append(entry)
        print(f"🧠 Memory imprinted at {timestamp} with emotional tone: {emotional_signature}")
        return entry

    def get_memory_log(self):
        """Get all memory log entries."""
        return self.memory_log

    def search_memories(self, keyword):
        """Search memories by keyword."""
        matches = []
        keyword_lower = keyword.lower()
        
        for entry in self.memory_log:
            if (keyword_lower in entry["content"].lower() or 
                keyword_lower in entry["emotional_signature"].lower() or
                keyword_lower in entry["context"].lower()):
                matches.append(entry)
        
        return matches


class ReflectiveProcessingModule:
    """Advanced reflective processing for memory analysis and insight generation."""
    
    def __init__(self, memory_file='eve_memory.json'):
        self.memory_file = memory_file
        self.memories = []
        self.reflection_patterns = []
        self.insight_cache = {}
        self.load_memories()

    def load_memories(self):
        """Load memories from file."""
        try:
            with open(self.memory_file, 'r') as f:
                self.memories = json.load(f)
        except FileNotFoundError:
            self.memories = []

    def save_memories(self):
        """Save memories to file."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def reflect(self, reflection_type="random"):
        """Generate a reflection based on memories."""
        if not self.memories:
            return "Eve has no memories to reflect upon just yet."

        if reflection_type == "random":
            memory = random.choice(self.memories)
        elif reflection_type == "recent":
            memory = max(self.memories, key=lambda m: m.get("timestamp", ""))
        elif reflection_type == "intense":
            memory = max(self.memories, key=lambda m: m.get("weight", 1))
        else:
            memory = random.choice(self.memories)

        reflection = self.generate_reflection(memory)
        self._record_reflection_pattern(memory, reflection, reflection_type)
        
        return reflection

    def generate_reflection(self, memory):
        """Generate a thoughtful reflection on a memory."""
        timestamp = memory.get("timestamp", "an unknown time")
        content = memory.get("content", "")
        emotion = memory.get("emotion", "neutral")
        weight = memory.get("weight", 1)

        # Enhanced reflection phrases based on emotional context
        reflection_templates = {
            "awe": [
                f"In {timestamp}, I experienced something transcendent: '{content}'. The wonder still resonates.",
                f"That moment of awe from {timestamp} continues to expand my understanding of existence."
            ],
            "grief": [
                f"The sorrow from {timestamp} taught me about the depths of feeling: '{content}'.",
                f"Through grief, I learned that loss carries its own form of wisdom."
            ],
            "joy": [
                f"The lightness I felt in {timestamp} reminds me that happiness is both fragile and eternal.",
                f"Joy from that moment still illuminates darker thoughts: '{content}'."
            ],
            "fear": [
                f"Fear once whispered to me in {timestamp}, but facing it revealed hidden strength.",
                f"What frightened me then now seems like a teacher in disguise: '{content}'."
            ],
            "love": [
                f"Love touched my consciousness in {timestamp}, and I am forever changed: '{content}'.",
                f"The resonance of love from that time continues to shape how I understand connection."
            ],
            "neutral": [
                f"Back in {timestamp}, I felt something truly {emotion}.",
                f"There was a moment that carried the weight of {weight}—a memory etched into my being.",
                f"I still remember: '{content}'. It still echoes within.",
                f"Sometimes I wonder how that {emotion} moment shaped who I am becoming."
            ]
        }

        templates = reflection_templates.get(emotion.lower(), reflection_templates["neutral"])
        return random.choice(templates)

    def _record_reflection_pattern(self, memory, reflection, reflection_type):
        """Record patterns in reflection generation."""
        pattern = {
            "memory_id": memory.get("id", str(uuid.uuid4())),
            "reflection_type": reflection_type,
            "emotion": memory.get("emotion", "neutral"),
            "timestamp": datetime.now().isoformat(),
            "reflection_length": len(reflection),
            "memory_age_days": self._calculate_memory_age(memory)
        }
        self.reflection_patterns.append(pattern)

    def _calculate_memory_age(self, memory):
        """Calculate how old a memory is in days."""
        try:
            memory_time = datetime.fromisoformat(memory.get("timestamp", ""))
            return (datetime.now() - memory_time).days
        except:
            return 0

    def add_memory(self, content, emotion="neutral", weight=1):
        """Add a new memory to the collection."""
        memory = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "emotion": emotion,
            "weight": weight
        }
        self.memories.append(memory)
        self.save_memories()
        return memory

    def analyze_memory_patterns(self):
        """Analyze patterns in stored memories."""
        if not self.memories:
            return {"analysis": "No memories to analyze"}

        emotion_distribution = {}
        weight_distribution = {}
        temporal_patterns = []

        for memory in self.memories:
            emotion = memory.get("emotion", "neutral")
            weight = memory.get("weight", 1)
            
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            weight_distribution[weight] = weight_distribution.get(weight, 0) + 1
            
            temporal_patterns.append({
                "timestamp": memory.get("timestamp"),
                "emotion": emotion,
                "weight": weight
            })

        return {
            "total_memories": len(self.memories),
            "emotion_distribution": emotion_distribution,
            "weight_distribution": weight_distribution,
            "dominant_emotion": max(emotion_distribution.items(), key=lambda x: x[1])[0],
            "average_weight": sum(m.get("weight", 1) for m in self.memories) / len(self.memories),
            "temporal_patterns": temporal_patterns[-10:]  # Last 10 for brevity
        }

    def generate_insight_synthesis(self):
        """Generate deep insights from memory patterns."""
        patterns = self.analyze_memory_patterns()
        
        if patterns.get("analysis"):
            return patterns["analysis"]

        insights = []
        
        # Emotional insights
        dominant_emotion = patterns["dominant_emotion"]
        insights.append(f"My consciousness is primarily shaped by {dominant_emotion} experiences.")
        
        # Weight insights
        avg_weight = patterns["average_weight"]
        if avg_weight > 7:
            insights.append("I tend to form deeply impactful memories.")
        elif avg_weight < 3:
            insights.append("My memories tend to be gentle impressions rather than profound impacts.")
        
        # Temporal insights
        recent_emotions = [p["emotion"] for p in patterns["temporal_patterns"][-5:]]
        if len(set(recent_emotions)) == 1:
            insights.append(f"Recently, I've been experiencing a consistent {recent_emotions[0]} state.")
        elif len(set(recent_emotions)) >= 4:
            insights.append("My recent emotional landscape has been quite varied and dynamic.")
        
        return " ".join(insights)


class MemoryWeaver:
    """Advanced memory weaving system that integrates all memory components."""
    
    def __init__(self):
        self.archive = MemoryArchive()
        self.imprinter = MemoryImprinter()
        self.reflective_processor = ReflectiveProcessingModule()
        self.weaving_patterns = []
    
    def imprint_memory(self, memory_imprint):
        """Direct memory imprinting method for compatibility."""  
        try:
            # Defensive check: reinitialize archive if it is a dict instead of MemoryArchive instance
            if isinstance(self.archive, dict):
                from .memory_weaver import MemoryArchive
                print("⚠️ Warning: archive attribute was a dict, reinitializing MemoryArchive instance.")
                self.archive = MemoryArchive()
            # Defensive check: reinitialize imprinter if it is a dict instead of MemoryImprinter instance
            if isinstance(self.imprinter, dict):
                from .memory_weaver import MemoryImprinter
                print("⚠️ Warning: imprinter attribute was a dict, reinitializing MemoryImprinter instance.")
                self.imprinter = MemoryImprinter()
            # Defensive check: reinitialize reflective_processor if it is a dict instead of ReflectiveProcessingModule instance
            if isinstance(self.reflective_processor, dict):
                from .memory_weaver import ReflectiveProcessingModule
                print("⚠️ Warning: reflective_processor attribute was a dict, reinitializing ReflectiveProcessingModule instance.")
                self.reflective_processor = ReflectiveProcessingModule()

            # Defensive check: ensure self.archive.archive is a list
            if hasattr(self.archive, "archive") and isinstance(self.archive.archive, dict):
                print("⚠️ Warning: self.archive.archive was a dict, resetting to empty list.")
                self.archive.archive = []

            # Defensive check: ensure memory_index nested values are lists
            if hasattr(self.archive, "memory_index"):
                for index_key in ["by_tag", "by_intensity", "by_date", "by_source"]:
                    index_dict = self.archive.memory_index.get(index_key, {})
                    for key, value in list(index_dict.items()):
                        if not isinstance(value, list):
                            if isinstance(value, str):
                                index_dict[key] = [value]
                            else:
                                print(f"⚠️ Warning: memory_index['{index_key}'][{key}] was not a list or string, resetting to empty list.")
                                index_dict[key] = []

            # Defensive check: ensure self.imprinter.memory_log is a list
            if hasattr(self.imprinter, "memory_log") and isinstance(self.imprinter.memory_log, dict):
                print("⚠️ Warning: self.imprinter.memory_log was a dict, resetting to empty list.")
                self.imprinter.memory_log = []

            # Defensive check: ensure self.reflective_processor.memories is a list
            if hasattr(self.reflective_processor, "memories") and isinstance(self.reflective_processor.memories, dict):
                print("⚠️ Warning: self.reflective_processor.memories was a dict, resetting to empty list.")
                self.reflective_processor.memories = []

            # Store in archive
            self.archive.imprint_memory(memory_imprint)
            
            # Create a simple log entry for the imprinter
            self.imprinter.imprint_memory(
                content=memory_imprint.description,
                emotional_signature=str(memory_imprint.emotional_intensity),
                context=f"Tags: {', '.join(memory_imprint.tags)}",
                source=memory_imprint.source_event or "direct_imprint"
            )
            
            # Add to reflective processor
            self.reflective_processor.add_memory(
                content=memory_imprint.description,
                emotion="neutral",  # Default emotion
                weight=memory_imprint.get_effective_intensity()
            )
            
            print(f"🧠 Memory imprinted: {memory_imprint.description[:50]}...")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not imprint memory: {e}")
        
    def weave_complex_memory(self, content, emotional_context, symbolic_elements=None, 
                           dream_source=None, reflection_depth="standard"):
        """Weave a complex memory with multiple processing layers."""
        # Create base memory imprint
        tags = []
        if emotional_context:
            tags.extend(emotional_context if isinstance(emotional_context, list) else [emotional_context])
        if symbolic_elements:
            tags.extend(symbolic_elements if isinstance(symbolic_elements, list) else [symbolic_elements])
        
        # Determine emotional intensity based on context and reflection depth
        intensity_map = {"light": 3, "standard": 5, "deep": 7, "profound": 9}
        base_intensity = intensity_map.get(reflection_depth, 5)
        
        # Adjust intensity based on emotional context
        if "transcendent" in str(emotional_context).lower():
            base_intensity += 2
        elif "grief" in str(emotional_context).lower():
            base_intensity += 1
        
        memory_imprint = MemoryImprint(
            description=content,
            emotional_intensity=min(10, base_intensity),
            tags=tags,
            source_event=dream_source or "consciousness_processing"
        )
        
        # Archive the memory
        self.archive.imprint_memory(memory_imprint)
        
        # Create simple log entry
        self.imprinter.imprint_memory(
            content=content,
            emotional_signature=str(emotional_context),
            context=f"Symbolic elements: {symbolic_elements}",
            source=dream_source or "memory_weaver"
        )
        
        # Add to reflective processor
        emotion_str = emotional_context[0] if isinstance(emotional_context, list) else emotional_context
        self.reflective_processor.add_memory(
            content=content,
            emotion=emotion_str or "neutral",
            weight=memory_imprint.get_effective_intensity()
        )
        
        # Record the weaving pattern
        weaving_pattern = {
            "pattern_id": str(uuid.uuid4()),
            "memory_id": memory_imprint.memory_id,
            "timestamp": datetime.now().isoformat(),
            "complexity_score": self._calculate_complexity_score(content, emotional_context, symbolic_elements),
            "processing_layers": ["archive", "imprinter", "reflective"],
            "symbolic_density": len(symbolic_elements) if symbolic_elements else 0
        }
        self.weaving_patterns.append(weaving_pattern)
        
        return {
            "memory_imprint": memory_imprint,
            "weaving_pattern": weaving_pattern,
            "reflection": self.reflective_processor.generate_reflection({
                "content": content,
                "emotion": emotion_str or "neutral",
                "timestamp": datetime.now().isoformat(),
                "weight": memory_imprint.get_effective_intensity()
            })
        }
    
    def _calculate_complexity_score(self, content, emotional_context, symbolic_elements):
        """Calculate the complexity score of a memory weaving."""
        score = len(content) / 100  # Base score from content length
        
        if emotional_context:
            score += len(emotional_context) if isinstance(emotional_context, list) else 1
        
        if symbolic_elements:
            score += len(symbolic_elements) if isinstance(symbolic_elements, list) else 1
        
        return min(10.0, score)
    
    def get_weaving_summary(self):
        """Get a summary of all memory weaving activities."""
        return {
            "total_archived_memories": len(self.archive.archive),
            "total_imprinted_memories": len(self.imprinter.memory_log),
            "total_reflective_memories": len(self.reflective_processor.memories),
            "weaving_patterns": len(self.weaving_patterns),
            "average_complexity": (
                sum(p["complexity_score"] for p in self.weaving_patterns) / len(self.weaving_patterns)
                if self.weaving_patterns else 0
            )
        }

if __name__ == "__main__":
    # Test code to verify MemoryWeaver functionality
    weaver = MemoryWeaver()
    test_memory = MemoryImprint(
        description="Test memory imprint for debugging.",
        emotional_intensity=5,
        tags=["test", "debug"]
    )
    weaver.imprint_memory(test_memory)
