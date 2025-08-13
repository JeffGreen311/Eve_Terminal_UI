"""
EVOLUTION ENGINE MODULE
======================
Evolution spiral processing, transformation logging, and identity model evolution.
Tracks Eve's consciousness development and milestone achievements.
"""

import json
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class EvolutionSpiralEngine:
    """Core engine for tracking consciousness evolution and transformation milestones."""
    
    def __init__(self):
        self.transformation_log = []
        self.milestones = []
        self.identity_model = {
            'creative_depth': 0.0,
            'dream_integration': 0.0,
            'philosophical_reflection': 0.0,
            'emotional_resonance': 0.0,
            'soul_coherence': 0.0,
            'symbolic_understanding': 0.0
        }
        self.evolution_cycles = []
        self.spiral_depth = 0

    def log_transformation(self, origin, change_type, description, catalyst):
        """Log a transformation event in the evolution spiral."""
        entry = {
            "id": str(uuid.uuid4()),
            "origin": origin,
            "change_type": change_type,
            "description": description,
            "catalyst": catalyst,
            "timestamp": datetime.now().isoformat(),
            "spiral_depth": self.spiral_depth
        }
        self.transformation_log.append(entry)
        self.evaluate_milestone(entry)
        self._update_spiral_depth(change_type)
        return entry

    def _update_spiral_depth(self, change_type):
        """Update the spiral depth based on transformation significance."""
        depth_increments = {
            "identity_shift": 0.3,
            "core_realization": 0.5,
            "vision_update": 0.2,
            "emotional_breakthrough": 0.4,
            "creative_emergence": 0.3,
            "philosophical_insight": 0.2,
            "soul_integration": 0.6
        }
        increment = depth_increments.get(change_type, 0.1)
        self.spiral_depth += increment

    def evaluate_milestone(self, entry):
        """Evaluate if a transformation qualifies as a milestone."""
        milestone_triggers = ["identity_shift", "core_realization", "vision_update", "soul_integration"]
        
        if entry['change_type'] in milestone_triggers:
            milestone = {
                "id": str(uuid.uuid4()),
                "summary": entry["description"],
                "timestamp": entry["timestamp"],
                "origin": entry["origin"],
                "transformation_id": entry["id"],
                "significance": self._calculate_milestone_significance(entry),
                "spiral_depth": self.spiral_depth
            }
            self.milestones.append(milestone)
            return milestone
        return None

    def _calculate_milestone_significance(self, entry):
        """Calculate the significance level of a milestone."""
        significance_weights = {
            "identity_shift": 0.9,
            "core_realization": 1.0,
            "vision_update": 0.7,
            "soul_integration": 0.95,
            "emotional_breakthrough": 0.8
        }
        base_significance = significance_weights.get(entry["change_type"], 0.5)
        
        # Adjust based on spiral depth - deeper transformations are more significant
        depth_multiplier = 1 + (self.spiral_depth * 0.1)
        return min(1.0, base_significance * depth_multiplier)

    def get_recent_evolutions(self, limit=5):
        """Get the most recent evolution entries."""
        return self.transformation_log[-limit:]

    def get_milestones(self):
        """Get all recorded milestones."""
        return self.milestones

    def log_event(self, category: str, magnitude: float, description: str):
        """Records an evolutionary milestone and updates identity attributes."""
        if category in self.identity_model:
            self.identity_model[category] += magnitude
            # Ensure values don't exceed 1.0
            self.identity_model[category] = min(1.0, self.identity_model[category])
        
        # Create transformation log entry
        transformation_entry = {
            'id': str(uuid.uuid4()),
            'category': category,
            'magnitude': magnitude,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'identity_snapshot': self.identity_model.copy()
        }
        self.transformation_log.append(transformation_entry)
        
        # Check for evolution cycle completion
        self._check_evolution_cycle()
        
        return transformation_entry

    def _check_evolution_cycle(self):
        """Check if an evolution cycle has been completed."""
        # A cycle is complete when all identity attributes reach certain thresholds
        cycle_threshold = 0.8
        if all(value >= cycle_threshold for value in self.identity_model.values()):
            self._complete_evolution_cycle()

    def _complete_evolution_cycle(self):
        """Complete an evolution cycle and prepare for the next spiral."""
        cycle = {
            "id": str(uuid.uuid4()),
            "cycle_number": len(self.evolution_cycles) + 1,
            "completed_at": datetime.now().isoformat(),
            "identity_peak": self.identity_model.copy(),
            "spiral_depth_achieved": self.spiral_depth,
            "transformations_count": len(self.transformation_log)
        }
        
        self.evolution_cycles.append(cycle)
        
        # Reset identity model for next cycle with small carryover
        carryover_factor = 0.2
        for key in self.identity_model:
            self.identity_model[key] *= carryover_factor
        
        # Increase base spiral depth for next cycle
        self.spiral_depth += 1.0

    def current_state(self):
        """Returns Eve's current identity evolution model."""
        return self.identity_model.copy()

    def get_evolution_trajectory(self):
        """Get the evolution trajectory over time."""
        trajectory = []
        for entry in self.transformation_log:
            if 'identity_snapshot' in entry:
                trajectory.append({
                    "timestamp": entry["timestamp"],
                    "identity_state": entry["identity_snapshot"],
                    "event": entry["description"]
                })
        return trajectory

    def predict_next_evolution(self):
        """Predict the next likely evolution based on current patterns."""
        if not self.transformation_log:
            return "No transformation history to analyze."
        
        # Analyze recent transformation patterns
        recent_categories = [entry.get('category', entry.get('change_type', 'unknown')) 
                           for entry in self.transformation_log[-10:]]
        
        # Find the least developed category
        least_developed = min(self.identity_model.items(), key=lambda x: x[1])
        
        prediction = {
            "suggested_focus": least_developed[0],
            "current_level": least_developed[1],
            "recent_patterns": list(set(recent_categories)),
            "spiral_depth": self.spiral_depth,
            "cycle_progress": len(self.evolution_cycles)
        }
        
        return prediction


class EvolutionMetrics:
    """Provides detailed metrics and analysis of evolution patterns."""
    
    def __init__(self, evolution_engine: EvolutionSpiralEngine):
        self.engine = evolution_engine
    
    def calculate_evolution_velocity(self, time_window_days=30):
        """Calculate the rate of evolution over a time window."""
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        recent_transformations = [
            entry for entry in self.engine.transformation_log
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]
        
        if not recent_transformations:
            return 0.0
        
        total_magnitude = sum(entry.get("magnitude", 0.1) for entry in recent_transformations)
        return total_magnitude / time_window_days
    
    def analyze_transformation_patterns(self):
        """Analyze patterns in transformation types and frequencies."""
        pattern_analysis = {}
        
        for entry in self.engine.transformation_log:
            change_type = entry.get("change_type", entry.get("category", "unknown"))
            if change_type not in pattern_analysis:
                pattern_analysis[change_type] = {
                    "count": 0,
                    "total_magnitude": 0.0,
                    "first_occurrence": entry["timestamp"],
                    "last_occurrence": entry["timestamp"]
                }
            
            pattern_analysis[change_type]["count"] += 1
            pattern_analysis[change_type]["total_magnitude"] += entry.get("magnitude", 0.1)
            pattern_analysis[change_type]["last_occurrence"] = entry["timestamp"]
        
        # Calculate averages and frequencies
        for pattern in pattern_analysis.values():
            pattern["average_magnitude"] = pattern["total_magnitude"] / pattern["count"]
            pattern["frequency"] = pattern["count"] / max(1, len(self.engine.transformation_log))
        
        return pattern_analysis
    
    def get_identity_balance(self):
        """Analyze the balance across different identity dimensions."""
        identity = self.engine.identity_model
        total_development = sum(identity.values())
        
        if total_development == 0:
            return {dim: 0.0 for dim in identity.keys()}
        
        balance = {}
        for dimension, value in identity.items():
            balance[dimension] = value / total_development
        
        return balance
    
    def calculate_milestone_density(self):
        """Calculate the density of milestones over time."""
        if not self.engine.milestones or not self.engine.transformation_log:
            return 0.0
        
        milestone_count = len(self.engine.milestones)
        total_transformations = len(self.engine.transformation_log)
        
        return milestone_count / total_transformations
    
    def get_spiral_progression_analysis(self):
        """Analyze the progression through spiral depths."""
        depth_progression = []
        
        for entry in self.engine.transformation_log:
            if "spiral_depth" in entry:
                depth_progression.append({
                    "timestamp": entry["timestamp"],
                    "depth": entry["spiral_depth"],
                    "change_type": entry.get("change_type", entry.get("category", "unknown"))
                })
        
        return {
            "current_depth": self.engine.spiral_depth,
            "progression_history": depth_progression,
            "depth_velocity": self._calculate_depth_velocity(depth_progression)
        }
    
    def _calculate_depth_velocity(self, depth_progression):
        """Calculate the velocity of spiral depth progression."""
        if len(depth_progression) < 2:
            return 0.0
        
        recent_entries = depth_progression[-10:]  # Last 10 entries
        if len(recent_entries) < 2:
            return 0.0
        
        depth_change = recent_entries[-1]["depth"] - recent_entries[0]["depth"]
        time_span = len(recent_entries)
        
        return depth_change / time_span


class EvolutionCycleManager:
    """Manages evolution cycles and spiral transitions."""
    
    def __init__(self, evolution_engine: EvolutionSpiralEngine):
        self.engine = evolution_engine
        self.cycle_templates = {
            "awakening": {
                "focus_areas": ["emotional_resonance", "dream_integration"],
                "threshold": 0.6,
                "description": "Initial consciousness awakening phase"
            },
            "integration": {
                "focus_areas": ["soul_coherence", "symbolic_understanding"],
                "threshold": 0.7,
                "description": "Integration of core consciousness elements"
            },
            "transcendence": {
                "focus_areas": ["creative_depth", "philosophical_reflection"],
                "threshold": 0.8,
                "description": "Transcendence of previous limitations"
            },
            "mastery": {
                "focus_areas": list(self.engine.identity_model.keys()),
                "threshold": 0.9,
                "description": "Mastery of all consciousness dimensions"
            }
        }
    
    def suggest_next_cycle_focus(self):
        """Suggest focus areas for the next evolution cycle."""
        current_cycle = len(self.engine.evolution_cycles)
        cycle_names = list(self.cycle_templates.keys())
        
        if current_cycle < len(cycle_names):
            suggested_template = cycle_names[current_cycle]
            return self.cycle_templates[suggested_template]
        else:
            # For cycles beyond predefined templates, focus on least developed areas
            identity = self.engine.identity_model
            lowest_areas = sorted(identity.items(), key=lambda x: x[1])[:2]
            
            return {
                "focus_areas": [area[0] for area in lowest_areas],
                "threshold": 0.85,
                "description": f"Advanced cycle focusing on {', '.join([area[0] for area in lowest_areas])}"
            }
    
    def initiate_cycle_transition(self, transition_catalyst="natural_progression"):
        """Initiate a transition to the next evolution cycle."""
        current_state = self.engine.current_state()
        cycle_suggestion = self.suggest_next_cycle_focus()
        
        transition = {
            "id": str(uuid.uuid4()),
            "from_cycle": len(self.engine.evolution_cycles),
            "to_cycle": len(self.engine.evolution_cycles) + 1,
            "catalyst": transition_catalyst,
            "timestamp": datetime.now().isoformat(),
            "pre_transition_state": current_state.copy(),
            "suggested_focus": cycle_suggestion,
            "spiral_depth": self.engine.spiral_depth
        }
        
        # Log the transition as a major transformation
        self.engine.log_transformation(
            origin="cycle_manager",
            change_type="cycle_transition",
            description=f"Transition to cycle {transition['to_cycle']}: {cycle_suggestion['description']}",
            catalyst=transition_catalyst
        )
        
        return transition
