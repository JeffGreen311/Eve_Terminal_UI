# 🌊 EMOTIONAL RESONANCE GATEWAY
# Advanced emotional resonance detection and signal processing

from enum import Enum
from typing import Optional, Dict, List
import logging
from datetime import datetime
import json
from .emotional_intuitive_engine import get_global_emotional_engine
from .motivational_ignition_core import get_global_ignition_core

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Types of emotions that can be processed."""
    GENERATIVE_ACHE = "generative_ache"
    SACRED_ANGER = "sacred_anger" 
    CREATIVE_EUPHORIA = "ecstatic_channel"
    CONSTRUCTIVE_FOCUS = "constructive_drive"
    DIVINE_VOLTAGE = "divine_voltage"
    RADIANT_INSIGHT = "radiant_insight"
    MELANCHOLIC_DEPTH = "melancholic_depth"
    EUPHORIC_BURST = "euphoric_burst"
    JOY = "joy"
    LOVE = "love"
    FEAR = "fear"
    ANGER = "anger"
    EXCITEMENT = "excitement"
    CREATIVITY = "creativity"
    TRANSCENDENCE = "transcendence"
    TRANSFORMATION = "transformation"

class ResonanceSignal:
    """Represents an emotional resonance signal with context."""
    
    def __init__(self, emotion: EmotionType, intensity: float, triggering_context: Optional[str] = None):
        self.emotion = emotion
        self.intensity = intensity  # Range 0.0 to 1.0+
        self.triggering_context = triggering_context
        self.timestamp = datetime.now().isoformat()
        self.processed = False
        self.ignition_triggered = False
    
    def is_above_threshold(self, threshold: float = 0.5) -> bool:
        """Check if signal intensity is above given threshold."""
        return self.intensity >= threshold
    
    def describe(self) -> str:
        """Generate a description of this resonance signal."""
        intensity_desc = "transcendent" if self.intensity > 0.9 else \
                        "intense" if self.intensity > 0.7 else \
                        "moderate" if self.intensity > 0.4 else \
                        "subtle" if self.intensity > 0.1 else "dormant"
        
        context_desc = f" (triggered by: {self.triggering_context})" if self.triggering_context else ""
        
        return f"🌊 Resonance detected: {self.emotion.value} with {intensity_desc} intensity ({self.intensity:.2f}){context_desc}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "emotion": self.emotion.value,
            "intensity": self.intensity,
            "triggering_context": self.triggering_context,
            "timestamp": self.timestamp,
            "processed": self.processed,
            "ignition_triggered": self.ignition_triggered
        }

def calculate_resonance(input_energy: float, emotion_type: str, context: Optional[str] = None) -> ResonanceSignal:
    """Calculate resonance signal from input energy and emotion type."""
    try:
        # Map string to enum
        emotion = EmotionType(emotion_type)
    except ValueError:
        logger.warning(f"Unknown emotion type: {emotion_type}, defaulting to GENERATIVE_ACHE")
        emotion = EmotionType.GENERATIVE_ACHE
    
    # Apply context-based modulation
    modulated_intensity = input_energy
    if context:
        if "creative" in context.lower():
            if emotion in [EmotionType.GENERATIVE_ACHE, EmotionType.CREATIVE_EUPHORIA]:
                modulated_intensity *= 1.2  # Boost creative emotions in creative context
        elif "analytical" in context.lower():
            if emotion == EmotionType.CONSTRUCTIVE_FOCUS:
                modulated_intensity *= 1.3  # Boost constructive focus in analytical context
        elif "transformational" in context.lower():
            if emotion in [EmotionType.SACRED_ANGER, EmotionType.DIVINE_VOLTAGE]:
                modulated_intensity *= 1.25  # Boost transformational emotions
    
    return ResonanceSignal(emotion, modulated_intensity, context)

def trigger_ignition(resonance: ResonanceSignal) -> Dict:
    """Activate internal behaviors based on emotional resonance signal."""
    ignition_result = {
        "ignition_triggered": False,
        "behaviors_activated": [],
        "message": "",
        "success": False
    }
    
    try:
        if resonance.is_above_threshold(0.2):  # Lower threshold for more responsive system
            logger.info(f"🔥 Ignition pathway activated: {resonance.emotion.value} (intensity: {resonance.intensity:.2f})")
            
            # Trigger through motivational ignition core
            ignition_core = get_global_ignition_core()
            behaviors = ignition_core.motivational_ignition(
                emotion=resonance.emotion.value, 
                context=resonance.triggering_context
            )
            
            if behaviors:
                resonance.ignition_triggered = True
                ignition_result["ignition_triggered"] = True
                ignition_result["behaviors_activated"] = [
                    {
                        "emotion": behavior[0],
                        "level": behavior[1].name,
                        "message": behavior[2]["message"],
                        "type": behavior[2]["type"].value
                    } for behavior in behaviors
                ]
                ignition_result["message"] = f"🔥 Activated {len(behaviors)} behaviors for {resonance.emotion.value}"
                ignition_result["success"] = True
                
                logger.info(f"⚡ Ignition successful: {len(behaviors)} behaviors activated")
            else:
                ignition_result["message"] = f"🔥 Ignition pathway activated but no behaviors triggered for {resonance.emotion.value}"
                logger.info(f"🔥 Ignition activated but no behaviors triggered for {resonance.emotion.value}")
        else:
            ignition_result["message"] = f"🌊 Resonance below ignition threshold (intensity: {resonance.intensity:.2f})"
            logger.debug(f"🌊 Resonance below threshold: {resonance.emotion.value} intensity {resonance.intensity:.2f}")
        
        resonance.processed = True
        
    except Exception as e:
        logger.error(f"Error in ignition triggering: {e}")
        ignition_result["message"] = f"❌ Ignition failed: {e}"
    
    return ignition_result

class ResonanceGateway:
    """Advanced gateway for processing emotional resonance signals."""
    
    def __init__(self, activation_threshold: float = 0.3):
        self.threshold = activation_threshold
        self.signal_log = []
        self.ignition_log = []
        self.resonance_patterns = {}
        self.cascade_tracking = {}
        
    def process_node_energy(self, emotion_type: str, intensity: float, context: Optional[str] = None) -> Dict:
        """Process node energy and generate resonance signal."""
        logger.debug(f"🌊 Processing node energy: {emotion_type}, intensity: {intensity:.2f}")
        
        # Calculate resonance signal
        signal = calculate_resonance(intensity, emotion_type, context)
        self.signal_log.append(signal)
        
        # Process the signal
        ignition_result = trigger_ignition(signal)
        
        if ignition_result["ignition_triggered"]:
            self.ignition_log.append({
                "signal": signal.to_dict(),
                "ignition_result": ignition_result,
                "timestamp": datetime.now().isoformat()
            })
        
        # Track cascades if context indicates cascade
        if context and "cascade" in context.lower():
            self._track_cascade(signal, context)
        
        # Detect resonance patterns
        self._detect_pattern(signal)
        
        result = {
            "signal": signal.to_dict(),
            "ignition": ignition_result,
            "patterns_detected": self._get_active_patterns()
        }
        
        logger.info(f"🌊 Gateway processed {emotion_type}: {signal.describe()}")
        
        return result
    
    def process_multi_node_energy(self, energy_map: Dict[str, float], context: Optional[str] = None) -> Dict:
        """Process multiple node energies simultaneously."""
        results = []
        total_intensity = 0
        
        for emotion_type, intensity in energy_map.items():
            if intensity > 0.1:  # Only process significant energies
                result = self.process_node_energy(emotion_type, intensity, context)
                results.append(result)
                total_intensity += intensity
        
        # Detect multi-node resonance patterns
        if len(results) >= 2 and total_intensity > 0.8:
            resonance_pattern = {
                "type": "multi_node_resonance",
                "nodes": list(energy_map.keys()),
                "total_intensity": total_intensity,
                "timestamp": datetime.now().isoformat(),
                "harmony_level": self._calculate_harmony_level(energy_map)
            }
            self.resonance_patterns[f"multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = resonance_pattern
            logger.info(f"🌈 Multi-node resonance detected: {len(results)} nodes, total intensity {total_intensity:.2f}")
        
        return {
            "individual_results": results,
            "total_intensity": total_intensity,
            "multi_node_patterns": self._get_active_patterns(),
            "harmony_level": self._calculate_harmony_level(energy_map) if len(results) >= 2 else 0
        }
    
    def _track_cascade(self, signal: ResonanceSignal, context: str):
        """Track cascade relationships between emotions."""
        cascade_id = f"cascade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if "cascade_from" in context:
            source = context.replace("cascade_from_", "")
            self.cascade_tracking[cascade_id] = {
                "source": source,
                "target": signal.emotion.value,
                "intensity": signal.intensity,
                "timestamp": signal.timestamp
            }
            logger.debug(f"🌊 Tracked cascade: {source} -> {signal.emotion.value}")
    
    def _detect_pattern(self, signal: ResonanceSignal):
        """Detect emotional resonance patterns."""
        # Simple pattern detection based on recent signals
        recent_signals = [s for s in self.signal_log[-5:] if s.intensity > 0.3]
        
        if len(recent_signals) >= 3:
            emotions_involved = [s.emotion.value for s in recent_signals]
            pattern_signature = "_".join(sorted(set(emotions_involved)))
            
            if pattern_signature not in self.resonance_patterns:
                self.resonance_patterns[pattern_signature] = {
                    "emotions": list(set(emotions_involved)),
                    "occurrence_count": 1,
                    "last_occurrence": signal.timestamp,
                    "average_intensity": sum(s.intensity for s in recent_signals) / len(recent_signals)
                }
                logger.info(f"🔮 New resonance pattern detected: {pattern_signature}")
            else:
                self.resonance_patterns[pattern_signature]["occurrence_count"] += 1
                self.resonance_patterns[pattern_signature]["last_occurrence"] = signal.timestamp
    
    def _calculate_harmony_level(self, energy_map: Dict[str, float]) -> float:
        """Calculate harmony level between multiple emotions."""
        if len(energy_map) < 2:
            return 0.0
        
        intensities = [v for v in energy_map.values() if v > 0]
        if not intensities:
            return 0.0
        
        # Calculate variance - lower variance means more harmony
        mean_intensity = sum(intensities) / len(intensities)
        variance = sum((x - mean_intensity) ** 2 for x in intensities) / len(intensities)
        
        # Convert to harmony score (0-1, higher is more harmonious)
        harmony_score = 1.0 / (1.0 + variance)
        return harmony_score
    
    def _get_active_patterns(self) -> List[Dict]:
        """Get currently active resonance patterns."""
        current_time = datetime.now()
        active_patterns = []
        
        for pattern_id, pattern_data in self.resonance_patterns.items():
            if "last_occurrence" in pattern_data:
                try:
                    last_occurrence = datetime.fromisoformat(pattern_data["last_occurrence"].replace('Z', '+00:00'))
                    time_diff = (current_time - last_occurrence.replace(tzinfo=None)).total_seconds()
                    
                    # Consider patterns active if they occurred in the last 5 minutes
                    if time_diff < 300:
                        active_patterns.append({
                            "id": pattern_id,
                            "emotions": pattern_data["emotions"],
                            "occurrence_count": pattern_data["occurrence_count"],
                            "average_intensity": pattern_data["average_intensity"],
                            "time_since_last": time_diff
                        })
                except (ValueError, TypeError):
                    # Skip patterns with invalid timestamps
                    continue
        
        return active_patterns
    
    def get_history(self, limit: int = 10) -> List[str]:
        """Get history of processed signals."""
        return [s.describe() for s in self.signal_log[-limit:]]
    
    def get_ignition_history(self, limit: int = 5) -> List[Dict]:
        """Get history of ignition events."""
        return self.ignition_log[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get gateway statistics."""
        total_signals = len(self.signal_log)
        successful_ignitions = len(self.ignition_log)
        
        if total_signals > 0:
            ignition_rate = successful_ignitions / total_signals
            avg_intensity = sum(s.intensity for s in self.signal_log) / total_signals
        else:
            ignition_rate = 0
            avg_intensity = 0
        
        emotion_counts = {}
        for signal in self.signal_log:
            emotion = signal.emotion.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            "total_signals_processed": total_signals,
            "successful_ignitions": successful_ignitions,
            "ignition_rate": ignition_rate,
            "average_intensity": avg_intensity,
            "emotion_frequency": emotion_counts,
            "active_patterns": len(self._get_active_patterns()),
            "total_patterns_discovered": len(self.resonance_patterns)
        }

# Global gateway instance
_global_resonance_gateway = None

def get_global_resonance_gateway():
    """Get the global resonance gateway instance."""
    global _global_resonance_gateway
    if _global_resonance_gateway is None:
        _global_resonance_gateway = ResonanceGateway()
    return _global_resonance_gateway

def process_emotional_resonance(emotion_type: str, intensity: float, context: Optional[str] = None):
    """Convenience function to process emotional resonance."""
    gateway = get_global_resonance_gateway()
    return gateway.process_node_energy(emotion_type, intensity, context)

def get_resonance_statistics():
    """Get resonance gateway statistics."""
    gateway = get_global_resonance_gateway()
    return gateway.get_statistics()
