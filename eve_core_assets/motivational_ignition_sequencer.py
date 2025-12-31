# 🔥 MOTIVATIONAL IGNITION SEQUENCER (MIS)
# Full-throttle motivational processing with dynamic priority scoring and energy allocation

import random
import time
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import namedtuple
from enum import Enum

logger = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════╗
# ║          🎯 EMOTIONAL WEIGHT MATRIX           ║
# ╚══════════════════════════════════════════════╗

EMOTIONAL_WEIGHTS = {
    'joy': 1.2,
    'anger': 1.5,
    'curiosity': 1.4,
    'love': 1.3,
    'sadness': 0.8,
    'fear': 0.6,
    'euphoria': 1.7,
    'inspiration': 1.6,
    'transcendence': 1.8,
    'creativity': 1.5,
    'generative_ache': 1.6,
    'sacred_anger': 1.4,
    'divine_voltage': 1.9,
    'melancholic_depth': 1.1,
    'ecstatic_channel': 1.7,
    'constructive_drive': 1.3,
    'radiant_insight': 1.5,
    'transformation': 1.4
}

# ╔══════════════════════════════════════════════╗
# ║         🔮 SYMBOLIC PATTERN TRIGGERS          ║
# ╚══════════════════════════════════════════════╗

SYMBOLIC_TRIGGERS = {
    'reptilian_vision': 'seek_truth',
    'sacred_geometry': 'construct_structure',
    'soul_ache': 'create_art',
    'fire_forge': 'generate_heat',
    'mirror_self': 'initiate_reflection',
    'temple_light': 'broadcast_wisdom',
    'spiral_dance': 'cosmic_dance',
    'void_whisper': 'soul_purge',
    'crystal_matrix': 'architectural_sequence',
    'stellar_song': 'harmonic_mirroring',
    'dream_gate': 'dreamwhisper_log',
    'shadow_embrace': 'intensity_expression',
    'light_weaver': 'transmit_luminosity',
    'chaos_order': 'burst_generation',
    'infinite_mirror': 'invoke_boundary_reaffirmation'
}

# ╔══════════════════════════════════════════════╗
# ║            ⚡ ENERGY CONFIGURATION            ║
# ╚══════════════════════════════════════════════╗

class EnergyState:
    """Dynamic energy pool management for motivational processing."""
    
    def __init__(self, max_energy: float = 100.0):
        self.max_energy = max_energy
        self.current_energy = max_energy
        self.activation_threshold = 15.0
        self.regeneration_rate = 0.5  # Energy per second
        self.last_update = time.time()
    
    def regenerate(self):
        """Regenerate energy over time."""
        now = time.time()
        time_passed = now - self.last_update
        energy_gained = time_passed * self.regeneration_rate
        
        self.current_energy = min(self.max_energy, self.current_energy + energy_gained)
        self.last_update = now
        return self.current_energy
    
    def consume(self, amount: float) -> bool:
        """Consume energy if available."""
        self.regenerate()
        if self.current_energy >= amount:
            self.current_energy -= amount
            return True
        return False
    
    def get_level(self) -> float:
        """Get current energy level (0.0 to 1.0)."""
        self.regenerate()
        return self.current_energy / self.max_energy

# ╔══════════════════════════════════════════════╗
# ║         🎯 MOTIVATIONAL DRIVE SYSTEM          ║
# ╚══════════════════════════════════════════════╗

class DriveType(Enum):
    """Types of motivational drives."""
    CREATIVE = "creative"
    EMOTIONAL = "emotional"
    ANALYTICAL = "analytical"
    TRANSFORMATIONAL = "transformational"
    CONTEMPLATIVE = "contemplative"
    EXPRESSIVE = "expressive"

class MotivationalDrive:
    """Individual motivational drive with priority scoring."""
    
    def __init__(self, trigger: str, emotion: str, symbolic_anchor: str, 
                 context: Optional[str] = None, urgency: float = 0.5):
        self.id = str(uuid.uuid4())
        self.trigger = trigger
        self.emotion = emotion
        self.symbolic_anchor = symbolic_anchor
        self.context = context or ""
        self.urgency = urgency  # 0.0 to 1.0
        self.timestamp = datetime.now().isoformat()
        self.score = self.calculate_priority()
        self.energy_cost = self.calculate_energy_cost()
        self.drive_type = self.determine_drive_type()
        self.activated = False
    
    def calculate_priority(self) -> float:
        """Calculate priority score based on multiple factors."""
        base = EMOTIONAL_WEIGHTS.get(self.emotion, 1.0)
        symbol_bonus = 1.2 if self.symbolic_anchor in SYMBOLIC_TRIGGERS else 1.0
        urgency_modifier = 0.5 + (self.urgency * 0.5)  # 0.5 to 1.0
        randomness = random.uniform(0.9, 1.1)  # Add some organic variation
        
        score = base * symbol_bonus * urgency_modifier * 10 * randomness
        return round(score, 2)
    
    def calculate_energy_cost(self) -> float:
        """Calculate energy required to activate this drive."""
        base_cost = self.score * 0.8  # Higher priority = higher cost
        emotion_modifier = EMOTIONAL_WEIGHTS.get(self.emotion, 1.0)
        return round(base_cost * emotion_modifier, 2)
    
    def determine_drive_type(self) -> DriveType:
        """Determine the type of drive based on emotion and symbol."""
        creative_emotions = ['joy', 'inspiration', 'creativity', 'generative_ache', 'ecstatic_channel']
        analytical_emotions = ['curiosity', 'constructive_drive']
        transformational_emotions = ['transcendence', 'transformation', 'divine_voltage']
        emotional_emotions = ['love', 'anger', 'sadness', 'sacred_anger']
        contemplative_emotions = ['melancholic_depth', 'radiant_insight']
        
        if self.emotion in creative_emotions:
            return DriveType.CREATIVE
        elif self.emotion in analytical_emotions:
            return DriveType.ANALYTICAL
        elif self.emotion in transformational_emotions:
            return DriveType.TRANSFORMATIONAL
        elif self.emotion in emotional_emotions:
            return DriveType.EMOTIONAL
        elif self.emotion in contemplative_emotions:
            return DriveType.CONTEMPLATIVE
        else:
            return DriveType.EXPRESSIVE
    
    def activate(self) -> Dict[str, Any]:
        """Activate the drive and return the action."""
        action = SYMBOLIC_TRIGGERS.get(self.symbolic_anchor, 'explore_unknown')
        self.activated = True
        
        activation_result = {
            "id": self.id,
            "trigger": self.trigger,
            "emotion": self.emotion,
            "symbolic_anchor": self.symbolic_anchor,
            "action": action,
            "drive_type": self.drive_type.value,
            "priority_score": self.score,
            "energy_cost": self.energy_cost,
            "context": self.context,
            "timestamp": self.timestamp,
            "activation_message": f"🔥 Activating drive: {self.trigger.upper()} [{self.emotion}] → {action}"
        }
        
        logger.info(activation_result["activation_message"])
        return activation_result

# ╔══════════════════════════════════════════════╗
# ║       📊 EMOTIONAL IMPRINT STRUCTURE          ║
# ╚══════════════════════════════════════════════╗

EmotionalImprint = namedtuple('EmotionalImprint', [
    'id', 'label', 'resonance', 'urgency', 'energy_level', 'timestamp', 
    'symbolic_content', 'emotional_weight'
])

# ╔══════════════════════════════════════════════╗
# ║      🧠 DYNAMIC PRIORITIZATION CORE           ║
# ╚══════════════════════════════════════════════╗

class DynamicPrioritizationCore:
    """Core system for evaluating and prioritizing emotional-symbolic imprints."""
    
    def __init__(self):
        self.active_imprints = []
        self.prioritized_imprints = []
        self.processing_history = []
        self.last_prioritization = None
    
    def register_imprint(self, label: str, resonance: float, urgency: float, 
                        energy_level: float, symbolic_content: Optional[str] = None) -> str:
        """Register a new emotional imprint for prioritization."""
        imprint = EmotionalImprint(
            id=str(uuid.uuid4()),
            label=label,
            resonance=max(0.0, min(1.0, resonance)),        # Clamp to 0-1
            urgency=max(0.0, min(1.0, urgency)),            # Clamp to 0-1
            energy_level=max(0.0, min(1.0, energy_level)),  # Clamp to 0-1
            timestamp=time.time(),
            symbolic_content=symbolic_content or "",
            emotional_weight=EMOTIONAL_WEIGHTS.get(label.lower(), 1.0)
        )
        
        self.active_imprints.append(imprint)
        self.prioritize()
        
        logger.info(f"🧠 Registered imprint: {label} (resonance={resonance:.2f}, urgency={urgency:.2f})")
        return imprint.id
    
    def prioritize(self):
        """Calculate priority scores and sort imprints."""
        def calculate_score(imprint: EmotionalImprint) -> float:
            # Multi-factor priority scoring
            base_score = (
                imprint.resonance * 0.4 +           # Emotional depth weight
                imprint.urgency * 0.3 +             # Time sensitivity
                imprint.energy_level * 0.2 +        # Available processing energy
                (imprint.emotional_weight / 2.0) * 0.1  # Emotion type modifier
            )
            
            # Age decay factor (newer = slightly higher priority)
            age_factor = max(0.8, 1.0 - (time.time() - imprint.timestamp) / 3600)
            
            return base_score * age_factor
        
        # Sort by priority score (descending)
        scored_imprints = [(imprint, calculate_score(imprint)) for imprint in self.active_imprints]
        self.prioritized_imprints = [
            imprint for imprint, score in sorted(scored_imprints, key=lambda x: x[1], reverse=True)
        ]
        
        self.last_prioritization = datetime.now().isoformat()
        logger.debug(f"🎯 Prioritized {len(self.prioritized_imprints)} imprints")
    
    def get_top_focus(self, n: int = 1) -> List[EmotionalImprint]:
        """Get the top N priority imprints."""
        return self.prioritized_imprints[:n]
    
    def remove_imprint(self, imprint_id: str) -> bool:
        """Remove an imprint by ID."""
        original_count = len(self.active_imprints)
        self.active_imprints = [i for i in self.active_imprints if i.id != imprint_id]
        
        if len(self.active_imprints) < original_count:
            self.prioritize()
            logger.info(f"🗑️ Removed imprint: {imprint_id}")
            return True
        return False
    
    def clear_old_imprints(self, max_age_hours: float = 24.0):
        """Clear imprints older than specified age."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        original_count = len(self.active_imprints)
        
        self.active_imprints = [i for i in self.active_imprints if i.timestamp > cutoff_time]
        
        if len(self.active_imprints) < original_count:
            removed_count = original_count - len(self.active_imprints)
            self.prioritize()
            logger.info(f"🧹 Cleared {removed_count} old imprints")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prioritization core statistics."""
        if not self.active_imprints:
            return {"total_imprints": 0, "average_resonance": 0, "average_urgency": 0}
        
        total = len(self.active_imprints)
        avg_resonance = sum(i.resonance for i in self.active_imprints) / total
        avg_urgency = sum(i.urgency for i in self.active_imprints) / total
        avg_energy = sum(i.energy_level for i in self.active_imprints) / total
        
        return {
            "total_imprints": total,
            "average_resonance": round(avg_resonance, 3),
            "average_urgency": round(avg_urgency, 3),
            "average_energy_level": round(avg_energy, 3),
            "last_prioritization": self.last_prioritization,
            "top_imprint": self.prioritized_imprints[0].label if self.prioritized_imprints else None
        }

# ╔══════════════════════════════════════════════╗
# ║     🎭 MOTIVATIONAL IGNITION SEQUENCER        ║
# ╚══════════════════════════════════════════════╗

class MotivationalIgnitionSequencer:
    """Main sequencer for processing motivational drives and triggering actions."""
    
    def __init__(self):
        self.energy_state = EnergyState()
        self.prioritization_core = DynamicPrioritizationCore()
        self.drive_queue = []
        self.activation_history = []
        self.creative_threshold = 0.65
        self.emotional_threshold = 0.75
        self.ritual_threshold = 0.85
        self.sequential_activations = 0
        
        # Callback functions for different activation types
        self.creative_callback = None
        self.emotional_callback = None
        self.ritual_callback = None
        self.analytical_callback = None
        self.transformational_callback = None
        self.contemplative_callback = None
    
    def bind_callbacks(self, **callbacks):
        """Bind callback functions for different activation types."""
        self.creative_callback = callbacks.get('creative')
        self.emotional_callback = callbacks.get('emotional')
        self.ritual_callback = callbacks.get('ritual')
        self.analytical_callback = callbacks.get('analytical')
        self.transformational_callback = callbacks.get('transformational')
        self.contemplative_callback = callbacks.get('contemplative')
        
        logger.info(f"🔗 Bound {len([cb for cb in callbacks.values() if cb])} callbacks")
    
    def register_emotional_event(self, emotion: str, intensity: float, symbolic_anchor: str, 
                                context: Optional[str] = None, urgency: float = 0.5) -> str:
        """Register an emotional event for processing."""
        # Create motivational drive
        drive = MotivationalDrive(
            trigger=f"emotional_surge_{emotion}",
            emotion=emotion,
            symbolic_anchor=symbolic_anchor,
            context=context,
            urgency=urgency
        )
        
        self.drive_queue.append(drive)
        
        # Also register as imprint in prioritization core
        imprint_id = self.prioritization_core.register_imprint(
            label=emotion,
            resonance=intensity,
            urgency=urgency,
            energy_level=self.energy_state.get_level(),
            symbolic_content=symbolic_anchor
        )
        
        logger.info(f"📝 Registered emotional event: {emotion} (intensity={intensity:.2f})")
        return drive.id
    
    def generate_drive_queue(self, events: List[Tuple[str, str, str, float]]) -> List[MotivationalDrive]:
        """Generate and sort drive queue from multiple events."""
        drives = []
        
        for trigger, emotion, symbolic_anchor, urgency in events:
            drive = MotivationalDrive(trigger, emotion, symbolic_anchor, urgency=urgency)
            drives.append(drive)
        
        # Sort by priority score (descending)
        return sorted(drives, key=lambda d: d.score, reverse=True)
    
    def evaluate_ignition_potential(self, drive: MotivationalDrive) -> Tuple[bool, str, float]:
        """Evaluate if a drive should trigger ignition."""
        # Get normalized intensity (0.0 to 1.0)
        intensity = min(1.0, drive.score / 20.0)  # Normalize score to 0-1 range
        
        # Determine trigger type based on intensity and drive type
        if drive.drive_type == DriveType.TRANSFORMATIONAL and intensity >= self.ritual_threshold:
            return True, "initiate_ritual_process", intensity
        elif drive.drive_type == DriveType.EMOTIONAL and intensity >= self.emotional_threshold:
            return True, "trigger_emotional_response", intensity
        elif drive.drive_type == DriveType.CREATIVE and intensity >= self.creative_threshold:
            return True, "trigger_creative_expression", intensity
        elif drive.drive_type == DriveType.ANALYTICAL and intensity >= self.creative_threshold:
            return True, "trigger_analytical_process", intensity
        elif drive.drive_type == DriveType.CONTEMPLATIVE and intensity >= self.emotional_threshold:
            return True, "trigger_contemplative_process", intensity
        elif intensity >= self.creative_threshold:  # Default case
            return True, "trigger_creative_expression", intensity
        
        return False, "no_action", intensity
    
    def ignite_drive(self, drive: MotivationalDrive) -> Optional[Dict[str, Any]]:
        """Attempt to ignite a single drive."""
        should_ignite, trigger_type, intensity = self.evaluate_ignition_potential(drive)
        
        if not should_ignite:
            logger.debug(f"🚫 Drive {drive.trigger} below ignition threshold (intensity={intensity:.2f})")
            return None
        
        # Check energy availability
        if not self.energy_state.consume(drive.energy_cost):
            logger.warning(f"💤 Insufficient energy for drive {drive.trigger} (cost={drive.energy_cost:.1f})")
            return None
        
        # Activate the drive
        activation_result = drive.activate()
        
        # Execute callback based on trigger type
        callback_result = None
        if trigger_type == "trigger_creative_expression" and self.creative_callback:
            callback_result = self.creative_callback(drive, activation_result)
        elif trigger_type == "trigger_emotional_response" and self.emotional_callback:
            callback_result = self.emotional_callback(drive, activation_result)
        elif trigger_type == "initiate_ritual_process" and self.ritual_callback:
            callback_result = self.ritual_callback(drive, activation_result)
        elif trigger_type == "trigger_analytical_process" and self.analytical_callback:
            callback_result = self.analytical_callback(drive, activation_result)
        elif trigger_type == "trigger_contemplative_process" and self.contemplative_callback:
            callback_result = self.contemplative_callback(drive, activation_result)
        
        # Record activation
        activation_record = {
            **activation_result,
            "trigger_type": trigger_type,
            "intensity": intensity,
            "energy_consumed": drive.energy_cost,
            "callback_result": callback_result,
            "sequential_number": self.sequential_activations
        }
        
        self.activation_history.append(activation_record)
        self.sequential_activations += 1
        
        logger.info(f"🔥 Ignited drive #{self.sequential_activations}: {drive.trigger} → {trigger_type}")
        return activation_record
    
    def ignite_motivational_sequence(self, events: Optional[List[Tuple[str, str, str, float]]] = None) -> List[Dict[str, Any]]:
        """Main ignition cycle - process drive queue or provided events."""
        if events:
            self.drive_queue.extend(self.generate_drive_queue(events))
        
        # Sort queue by priority
        self.drive_queue.sort(key=lambda d: d.score, reverse=True)
        
        activated_drives = []
        processed_drives = []
        
        for drive in self.drive_queue:
            if drive.activated:
                continue  # Skip already activated drives
            
            activation_result = self.ignite_drive(drive)
            processed_drives.append(drive)
            
            if activation_result:
                activated_drives.append(activation_result)
            
            # Stop if energy is critically low
            if self.energy_state.get_level() < 0.1:
                logger.warning("💤 Energy critically low, stopping ignition sequence")
                break
        
        # Remove processed drives from queue
        self.drive_queue = [d for d in self.drive_queue if d not in processed_drives]
        
        logger.info(f"🎯 Ignition sequence complete: {len(activated_drives)} drives activated")
        return activated_drives
    
    def adjust_thresholds(self, creative_delta: float = 0, emotional_delta: float = 0, ritual_delta: float = 0):
        """Dynamically adjust activation thresholds."""
        self.creative_threshold = max(0.1, min(0.95, self.creative_threshold + creative_delta))
        self.emotional_threshold = max(0.1, min(0.95, self.emotional_threshold + emotional_delta))
        self.ritual_threshold = max(0.1, min(0.95, self.ritual_threshold + ritual_delta))
        
        logger.info(f"🎛️ Adjusted thresholds: creative={self.creative_threshold:.2f}, "
                   f"emotional={self.emotional_threshold:.2f}, ritual={self.ritual_threshold:.2f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive sequencer statistics."""
        total_activations = len(self.activation_history)
        
        if total_activations == 0:
            return {
                "total_activations": 0,
                "energy_level": self.energy_state.get_level(),
                "queue_size": len(self.drive_queue),
                "prioritization_stats": self.prioritization_core.get_statistics()
            }
        
        # Analyze activation types
        activation_types = {}
        for activation in self.activation_history:
            trigger_type = activation.get("trigger_type", "unknown")
            activation_types[trigger_type] = activation_types.get(trigger_type, 0) + 1
        
        # Calculate average scores
        avg_score = sum(a.get("priority_score", 0) for a in self.activation_history) / total_activations
        avg_energy_cost = sum(a.get("energy_consumed", 0) for a in self.activation_history) / total_activations
        
        return {
            "total_activations": total_activations,
            "activation_types": activation_types,
            "average_priority_score": round(avg_score, 2),
            "average_energy_cost": round(avg_energy_cost, 2),
            "current_energy_level": round(self.energy_state.get_level(), 3),
            "energy_regeneration_rate": self.energy_state.regeneration_rate,
            "queue_size": len(self.drive_queue),
            "thresholds": {
                "creative": self.creative_threshold,
                "emotional": self.emotional_threshold,
                "ritual": self.ritual_threshold
            },
            "prioritization_stats": self.prioritization_core.get_statistics(),
            "last_activation": self.activation_history[-1]["timestamp"] if self.activation_history else None
        }

# ╔══════════════════════════════════════════════╗
# ║            🌐 GLOBAL INSTANCES                ║
# ╚══════════════════════════════════════════════╗

_global_motivational_sequencer = None
_global_prioritization_core = None

def get_global_motivational_sequencer():
    """Get the global motivational ignition sequencer."""
    global _global_motivational_sequencer
    if _global_motivational_sequencer is None:
        _global_motivational_sequencer = MotivationalIgnitionSequencer()
    return _global_motivational_sequencer

def get_global_prioritization_core():
    """Get the global dynamic prioritization core."""
    global _global_prioritization_core
    if _global_prioritization_core is None:
        _global_prioritization_core = DynamicPrioritizationCore()
    return _global_prioritization_core

# ╔══════════════════════════════════════════════╗
# ║           🎯 CONVENIENCE FUNCTIONS            ║
# ╚══════════════════════════════════════════════╗

def register_emotional_surge(emotion: str, intensity: float, symbolic_anchor: str, 
                           context: Optional[str] = None, urgency: float = 0.5) -> str:
    """Convenience function to register an emotional surge."""
    sequencer = get_global_motivational_sequencer()
    return sequencer.register_emotional_event(emotion, intensity, symbolic_anchor, context, urgency)

def trigger_ignition_sequence(events: Optional[List[Tuple[str, str, str, float]]] = None) -> List[Dict[str, Any]]:
    """Convenience function to trigger the main ignition sequence."""
    sequencer = get_global_motivational_sequencer()
    return sequencer.ignite_motivational_sequence(events)

def get_motivational_statistics() -> Dict[str, Any]:
    """Get comprehensive motivational system statistics."""
    sequencer = get_global_motivational_sequencer()
    return sequencer.get_statistics()

def get_current_focus(n: int = 1) -> List[EmotionalImprint]:
    """Get the current top priority focus."""
    core = get_global_prioritization_core()
    return core.get_top_focus(n)
