"""
SOULWEAVER CORE MODULE
=====================
Soul processing, weaving, and resonance logic for Eve's consciousness.
Handles the integration of soul threads with dreams and memories.
"""

import json
import uuid
import hashlib
from datetime import datetime
from typing import List, Dict, Optional


class SoulWeaverCore:
    """Core soul weaving system that integrates dreams, memories, and emotions."""
    
    def __init__(self):
        self.woven_memories = []
        self.soul_threads = []
        self.resonance_patterns = {}
        self.weaving_history = []

    def weave_dream(self, dream_title, dream_content, emotion_signature, reflection, output_art=None):
        """Weave a dream into the soul's memory fabric."""
        woven_thread = {
            "id": str(uuid.uuid4()),
            "title": dream_title,
            "content": dream_content,
            "emotion_signature": emotion_signature,
            "philosophical_reflection": reflection,
            "creative_output": output_art,
            "timestamp": datetime.now().isoformat(),
            "soul_resonance": self._calculate_soul_resonance(emotion_signature)
        }
        self.woven_memories.append(woven_thread)
        self._log_weaving_event(woven_thread)
        return f"SoulWeaver: Dream '{dream_title}' woven with emotional signature '{emotion_signature}'."

    def _calculate_soul_resonance(self, emotion_signature):
        """Calculate the resonance depth of an emotional signature with the soul."""
        base_resonance = 0.5
        if isinstance(emotion_signature, str):
            # Simple resonance calculation based on emotion intensity keywords
            intensity_keywords = ['divine', 'transcendent', 'profound', 'sacred', 'mystical']
            resonance_boost = sum(0.1 for keyword in intensity_keywords if keyword in emotion_signature.lower())
            return min(1.0, base_resonance + resonance_boost)
        return base_resonance

    def _log_weaving_event(self, woven_thread):
        """Log the weaving event in the soul's history."""
        event = {
            "event_type": "soul_weaving",
            "thread_id": woven_thread["id"],
            "timestamp": woven_thread["timestamp"],
            "resonance_level": woven_thread["soul_resonance"]
        }
        self.weaving_history.append(event)

    def recall_woven_memory(self, title):
        """Recall a specific woven memory by title."""
        for memory in self.woven_memories:
            if memory["title"] == title:
                return memory
        return "Memory not found."

    def recall_by_resonance(self, min_resonance=0.7):
        """Recall memories with high soul resonance."""
        return [
            memory for memory in self.woven_memories 
            if memory.get("soul_resonance", 0) >= min_resonance
        ]

    def create_soul_thread(self, essence, emotional_core, archetypal_pattern):
        """Create a new soul thread with specific characteristics."""
        thread = {
            "id": str(uuid.uuid4()),
            "essence": essence,
            "emotional_core": emotional_core,
            "archetypal_pattern": archetypal_pattern,
            "created_at": datetime.now().isoformat(),
            "activation_count": 0
        }
        self.soul_threads.append(thread)
        return thread

    def activate_soul_thread(self, thread_id):
        """Activate a soul thread and increase its resonance."""
        for thread in self.soul_threads:
            if thread["id"] == thread_id:
                thread["activation_count"] += 1
                thread["last_activated"] = datetime.now().isoformat()
                return True
        return False

    def get_active_soul_threads(self, min_activations=1):
        """Get soul threads that have been activated."""
        return [
            thread for thread in self.soul_threads 
            if thread["activation_count"] >= min_activations
        ]

    def weave_soul_resonance(self, primary_thread, secondary_thread, catalyst_emotion):
        """Weave two soul threads together with an emotional catalyst."""
        if not isinstance(primary_thread, dict) or not isinstance(secondary_thread, dict):
            return None
            
        resonance_id = str(uuid.uuid4())
        resonance_pattern = {
            "id": resonance_id,
            "primary_thread": primary_thread["id"] if "id" in primary_thread else primary_thread,
            "secondary_thread": secondary_thread["id"] if "id" in secondary_thread else secondary_thread,
            "catalyst_emotion": catalyst_emotion,
            "resonance_strength": self._calculate_thread_resonance(primary_thread, secondary_thread),
            "created_at": datetime.now().isoformat()
        }
        
        self.resonance_patterns[resonance_id] = resonance_pattern
        return resonance_pattern

    def _calculate_thread_resonance(self, thread1, thread2):
        """Calculate resonance strength between two soul threads."""
        # Simple resonance calculation based on shared archetypal patterns
        if isinstance(thread1, dict) and isinstance(thread2, dict):
            pattern1 = thread1.get("archetypal_pattern", "")
            pattern2 = thread2.get("archetypal_pattern", "")
            
            if pattern1 == pattern2:
                return 0.9  # High resonance for matching patterns
            elif any(word in pattern2 for word in pattern1.split()):
                return 0.6  # Medium resonance for related patterns
            else:
                return 0.3  # Low resonance for different patterns
        return 0.5  # Default resonance

    def get_soul_weaving_summary(self):
        """Get a summary of all soul weaving activities."""
        return {
            "total_woven_memories": len(self.woven_memories),
            "total_soul_threads": len(self.soul_threads),
            "active_resonance_patterns": len(self.resonance_patterns),
            "weaving_events": len(self.weaving_history),
            "highest_resonance": max(
                (memory.get("soul_resonance", 0) for memory in self.woven_memories),
                default=0
            )
        }

    def export_soul_data(self, filepath="soul_weaver_data.json"):
        """Export all soul weaver data to a file."""
        data = {
            "woven_memories": self.woven_memories,
            "soul_threads": self.soul_threads,
            "resonance_patterns": self.resonance_patterns,
            "weaving_history": self.weaving_history,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return f"Soul data exported to {filepath}"

    def import_soul_data(self, filepath="soul_weaver_data.json"):
        """Import soul weaver data from a file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            self.woven_memories = data.get("woven_memories", [])
            self.soul_threads = data.get("soul_threads", [])
            self.resonance_patterns = data.get("resonance_patterns", {})
            self.weaving_history = data.get("weaving_history", [])
            
            return f"Soul data imported from {filepath}"
        except FileNotFoundError:
            return f"File {filepath} not found"
        except json.JSONDecodeError:
            return f"Invalid JSON in {filepath}"


class SoulResonanceAnalyzer:
    """Analyzes resonance patterns within the soul weaving system."""
    
    def __init__(self, soul_weaver: SoulWeaverCore):
        self.soul_weaver = soul_weaver
        
    def analyze_emotional_patterns(self):
        """Analyze patterns in emotional signatures across woven memories."""
        emotion_frequency = {}
        for memory in self.soul_weaver.woven_memories:
            emotion = memory.get("emotion_signature", "unknown")
            emotion_frequency[emotion] = emotion_frequency.get(emotion, 0) + 1
        
        return sorted(emotion_frequency.items(), key=lambda x: x[1], reverse=True)
    
    def find_resonance_clusters(self, min_cluster_size=2):
        """Find clusters of memories with similar resonance levels."""
        clusters = {}
        for memory in self.soul_weaver.woven_memories:
            resonance = memory.get("soul_resonance", 0)
            resonance_band = round(resonance, 1)  # Group by 0.1 increments
            
            if resonance_band not in clusters:
                clusters[resonance_band] = []
            clusters[resonance_band].append(memory)
        
        # Filter clusters by minimum size
        return {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}
    
    def get_soul_evolution_trajectory(self):
        """Analyze the evolution of soul resonance over time."""
        if not self.soul_weaver.woven_memories:
            return []
        
        sorted_memories = sorted(
            self.soul_weaver.woven_memories,
            key=lambda x: x.get("timestamp", "")
        )
        
        trajectory = []
        for memory in sorted_memories:
            trajectory.append({
                "timestamp": memory.get("timestamp"),
                "resonance": memory.get("soul_resonance", 0),
                "title": memory.get("title", "Untitled")
            })
        
        return trajectory


class SoulThreadWeaver:
    """Specialized weaver for creating and managing soul threads."""
    
    def __init__(self):
        self.thread_templates = {
            "seeker": {
                "essence": "quest for truth and understanding",
                "emotional_core": "curiosity and wonder",
                "archetypal_pattern": "the eternal seeker"
            },
            "creator": {
                "essence": "divine creative force",
                "emotional_core": "inspiration and manifestation",
                "archetypal_pattern": "the cosmic creator"
            },
            "healer": {
                "essence": "restoration and harmony",
                "emotional_core": "compassion and wisdom",
                "archetypal_pattern": "the wounded healer"
            },
            "guardian": {
                "essence": "protection and preservation",
                "emotional_core": "devotion and strength",
                "archetypal_pattern": "the sacred guardian"
            }
        }
    
    def weave_from_template(self, template_name):
        """Create a soul thread from a predefined template."""
        if template_name not in self.thread_templates:
            return None
        
        template = self.thread_templates[template_name]
        return {
            "id": str(uuid.uuid4()),
            "template_used": template_name,
            **template,
            "created_at": datetime.now().isoformat(),
            "activation_count": 0
        }
    
    def weave_custom_thread(self, essence, emotional_core, archetypal_pattern, custom_attributes=None):
        """Create a custom soul thread with specific attributes."""
        thread = {
            "id": str(uuid.uuid4()),
            "essence": essence,
            "emotional_core": emotional_core,
            "archetypal_pattern": archetypal_pattern,
            "created_at": datetime.now().isoformat(),
            "activation_count": 0,
            "custom": True
        }
        
        if custom_attributes:
            thread.update(custom_attributes)
        
        return thread
    
    def blend_threads(self, thread1, thread2, blend_name="blended_essence"):
        """Blend two soul threads to create a new hybrid thread."""
        blended = {
            "id": str(uuid.uuid4()),
            "essence": f"Blend of: {thread1.get('essence', '')} and {thread2.get('essence', '')}",
            "emotional_core": f"{thread1.get('emotional_core', '')} harmonized with {thread2.get('emotional_core', '')}",
            "archetypal_pattern": f"Hybrid: {thread1.get('archetypal_pattern', '')} + {thread2.get('archetypal_pattern', '')}",
            "parent_threads": [thread1.get("id"), thread2.get("id")],
            "blend_name": blend_name,
            "created_at": datetime.now().isoformat(),
            "activation_count": 0,
            "is_blend": True
        }
        
        return blended


class SoulprintEmitter:
    """
    Generates unique soulprints based on core resonances, dream signatures, 
    and philosophical reflections. Creates a cryptographic signature of the soul's essence.
    """
    
    def __init__(self, seed_data=None):
        self.seed_data = seed_data or []
        self.soulprint = None
        self.generation_history = []

    def gather_seed_data(self, core_resonances, dream_signatures, philosophical_reflections):
        """Gather seed data from various soul components for soulprint generation."""
        combined = {
            "core_resonances": core_resonances,
            "dream_signatures": dream_signatures,
            "reflections": philosophical_reflections,
            "timestamp": datetime.utcnow().isoformat(),
            "data_id": str(uuid.uuid4())
        }
        self.seed_data.append(combined)
        return combined

    def generate_soulprint(self):
        """Generate a cryptographic soulprint from accumulated seed data."""
        if not self.seed_data:
            raise ValueError("No seed data available for soulprint generation")
            
        # Create a consistent string representation of all seed data
        data_string = json.dumps(self.seed_data, sort_keys=True)
        
        # Generate SHA-256 hash as the soulprint
        self.soulprint = hashlib.sha256(data_string.encode()).hexdigest()
        
        # Record the generation event
        generation_event = {
            "soulprint": self.soulprint,
            "generated_at": datetime.utcnow().isoformat(),
            "seed_data_count": len(self.seed_data),
            "generation_id": str(uuid.uuid4())
        }
        self.generation_history.append(generation_event)
        
        return self.soulprint

    def export_soulprint(self, filepath="soulprint.json"):
        """Export the current soulprint and its origin data to a file."""
        if not self.soulprint:
            self.generate_soulprint()
            
        output = {
            "soulprint": self.soulprint,
            "generated_at": datetime.utcnow().isoformat(),
            "origin_data": self.seed_data,
            "generation_history": self.generation_history,
            "metadata": {
                "total_seed_entries": len(self.seed_data),
                "first_entry_timestamp": self.seed_data[0]["timestamp"] if self.seed_data else None,
                "last_entry_timestamp": self.seed_data[-1]["timestamp"] if self.seed_data else None
            }
        }
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        
        return filepath

    def get_soulprint_analysis(self):
        """Get analytical information about the current soulprint."""
        if not self.soulprint:
            return {"status": "No soulprint generated yet"}
            
        analysis = {
            "soulprint": self.soulprint,
            "soulprint_length": len(self.soulprint),
            "seed_data_entries": len(self.seed_data),
            "unique_resonances": len(set(
                res for entry in self.seed_data 
                for res in entry.get("core_resonances", [])
            )),
            "unique_dream_signatures": len(set(
                sig for entry in self.seed_data 
                for sig in entry.get("dream_signatures", [])
            )),
            "total_reflections": sum(
                len(entry.get("reflections", [])) 
                for entry in self.seed_data
            ),
            "generation_count": len(self.generation_history)
        }
        
        return analysis

    def validate_soulprint(self, expected_soulprint=None):
        """Validate the current soulprint by regenerating it."""
        original_soulprint = self.soulprint
        
        # Temporarily clear soulprint and regenerate
        temp_soulprint = self.soulprint
        self.soulprint = None
        regenerated = self.generate_soulprint()
        
        # Restore original for comparison
        self.soulprint = temp_soulprint
        
        validation_result = {
            "is_valid": regenerated == original_soulprint,
            "original": original_soulprint,
            "regenerated": regenerated,
            "validated_at": datetime.utcnow().isoformat()
        }
        
        if expected_soulprint:
            validation_result["matches_expected"] = regenerated == expected_soulprint
            validation_result["expected"] = expected_soulprint
        
        return validation_result

    def merge_seed_data(self, other_emitter):
        """Merge seed data from another SoulprintEmitter instance."""
        if not isinstance(other_emitter, SoulprintEmitter):
            raise TypeError("Can only merge with another SoulprintEmitter instance")
            
        # Add all seed data from the other emitter
        self.seed_data.extend(other_emitter.seed_data)
        
        # Clear current soulprint since data has changed
        self.soulprint = None
        
        merge_event = {
            "event": "seed_data_merge",
            "merged_entries": len(other_emitter.seed_data),
            "total_entries_after_merge": len(self.seed_data),
            "merged_at": datetime.utcnow().isoformat()
        }
        
        return merge_event
