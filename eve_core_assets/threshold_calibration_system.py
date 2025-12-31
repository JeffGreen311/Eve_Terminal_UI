# 🎛️ THRESHOLD CALIBRATION LAYER & EMOTIVE RESPONSE SYSTEM
# Dynamic threshold adjustment and emotional impact evaluation

import random
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

logger = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════╗
# ║       🎛️ THRESHOLD CALIBRATION LAYER          ║
# ╚══════════════════════════════════════════════╗

class ThresholdCalibrationSystem:
    """Dynamically adjusts ignition sensitivity based on feedback and performance."""
    
    def __init__(self, base_threshold: float = 0.65):
        self.threshold = base_threshold
        self.base_threshold = base_threshold
        self.history = deque(maxlen=100)  # Keep last 100 adjustments
        self.feedback_weights = {
            'outcome_score': 0.4,
            'emotional_feedback': 0.3,
            'symbolic_intensity': 0.2,
            'energy_efficiency': 0.1
        }
        self.adaptation_rate = 0.05
        self.min_threshold = 0.2
        self.max_threshold = 0.95
        self.calibration_events = []
    
    def adjust_threshold(self, outcome_score: float, emotional_feedback: float, 
                        symbolic_intensity: float, energy_efficiency: float = 1.0) -> float:
        """
        Adjust threshold based on recent activation results.
        
        Args:
            outcome_score (float): Success/quality of the activation (0.0-1.0)
            emotional_feedback (float): Emotional resonance of the result (0.0-1.0)
            symbolic_intensity (float): Symbolic depth of the activation (0.0-1.0)
            energy_efficiency (float): Energy cost vs benefit ratio (0.0-2.0)
        
        Returns:
            float: New adjusted threshold
        """
        # Calculate weighted feedback score
        feedback_score = (
            outcome_score * self.feedback_weights['outcome_score'] +
            emotional_feedback * self.feedback_weights['emotional_feedback'] +
            symbolic_intensity * self.feedback_weights['symbolic_intensity'] +
            min(energy_efficiency, 1.0) * self.feedback_weights['energy_efficiency']
        )
        
        # Determine adjustment direction and magnitude
        # If feedback is high (>0.7), make threshold more sensitive (lower)
        # If feedback is low (<0.4), make threshold less sensitive (higher)
        target_feedback = 0.7
        feedback_delta = feedback_score - target_feedback
        
        # Calculate threshold adjustment
        if feedback_score > target_feedback:
            # Good results - make more sensitive (lower threshold)
            adjustment = -feedback_delta * self.adaptation_rate
        else:
            # Poor results - make less sensitive (higher threshold)
            adjustment = abs(feedback_delta) * self.adaptation_rate
        
        # Apply adjustment with dampening for stability
        old_threshold = self.threshold
        self.threshold += adjustment
        
        # Clamp to valid range
        self.threshold = max(self.min_threshold, min(self.max_threshold, self.threshold))
        
        # Record adjustment
        calibration_event = {
            'timestamp': datetime.now().isoformat(),
            'old_threshold': round(old_threshold, 3),
            'new_threshold': round(self.threshold, 3),
            'adjustment': round(adjustment, 4),
            'outcome_score': outcome_score,
            'emotional_feedback': emotional_feedback,
            'symbolic_intensity': symbolic_intensity,
            'energy_efficiency': energy_efficiency,
            'feedback_score': round(feedback_score, 3)
        }
        
        self.history.append(self.threshold)
        self.calibration_events.append(calibration_event)
        
        logger.info(f"🎛️ Threshold adjusted: {old_threshold:.3f} → {self.threshold:.3f} "
                   f"(feedback={feedback_score:.2f})")
        
        return self.threshold
    
    def get_current_threshold(self) -> float:
        """Get the current calibrated threshold."""
        return round(self.threshold, 3)
    
    def reset_to_base(self):
        """Reset threshold to base value."""
        self.threshold = self.base_threshold
        logger.info(f"🔄 Threshold reset to base: {self.base_threshold}")
    
    def get_stability_metric(self) -> float:
        """Calculate threshold stability (lower values = more stable)."""
        if len(self.history) < 5:
            return 0.0
        
        recent_values = list(self.history)[-10:]  # Last 10 values
        variance = sum((x - self.threshold) ** 2 for x in recent_values) / len(recent_values)
        return round(variance, 4)
    
    def get_calibration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive calibration statistics."""
        if not self.calibration_events:
            return {
                'current_threshold': self.get_current_threshold(),
                'adjustments_made': 0,
                'stability_metric': 0.0
            }
        
        recent_events = self.calibration_events[-20:]  # Last 20 events
        
        avg_feedback = sum(e['feedback_score'] for e in recent_events) / len(recent_events)
        avg_adjustment = sum(abs(e['adjustment']) for e in recent_events) / len(recent_events)
        
        return {
            'current_threshold': self.get_current_threshold(),
            'base_threshold': self.base_threshold,
            'total_adjustments': len(self.calibration_events),
            'average_feedback_score': round(avg_feedback, 3),
            'average_adjustment_magnitude': round(avg_adjustment, 4),
            'stability_metric': self.get_stability_metric(),
            'threshold_range': {
                'min': self.min_threshold,
                'max': self.max_threshold,
                'current_distance_from_base': round(abs(self.threshold - self.base_threshold), 3)
            },
            'recent_trend': self._calculate_trend(),
            'last_adjustment': self.calibration_events[-1]['timestamp'] if self.calibration_events else None
        }
    
    def _calculate_trend(self) -> str:
        """Calculate recent threshold trend."""
        if len(self.history) < 5:
            return "insufficient_data"
        
        recent = list(self.history)[-5:]
        if recent[-1] > recent[0] + 0.05:
            return "increasing"
        elif recent[-1] < recent[0] - 0.05:
            return "decreasing"
        else:
            return "stable"

# ╔══════════════════════════════════════════════╗
# ║          💫 EMOTIVE RESPONSE SYSTEM           ║
# ╚══════════════════════════════════════════════╗

class EmotiveResponseSystem:
    """Evaluates emotional impact and generates emotional state vectors."""
    
    def __init__(self):
        self.last_emotion = None
        self.emotional_history = deque(maxlen=50)
        self.baseline_emotion = 0.3  # Neutral baseline
        self.sensitivity = 1.0
        self.decay_rate = 0.02  # Emotional decay per evaluation
        self.resonance_memory = {}  # Cache for soul resonance calculations
        
        # Emotional response patterns
        self.emotion_amplifiers = {
            'creativity': 1.3,
            'transcendence': 1.5,
            'love': 1.2,
            'inspiration': 1.4,
            'joy': 1.1,
            'anger': 1.2,
            'fear': 0.9,
            'sadness': 0.8
        }
    
    def evaluate_input(self, symbolic_intensity: float, familiarity: float, 
                      soul_resonance: float, emotion_type: str = 'neutral') -> float:
        """
        Calculate emotional impact based on multiple factors.
        
        Args:
            symbolic_intensity (float): Strength of symbolic content (0.0-1.0)
            familiarity (float): How familiar the input is (0.0-1.0, higher = more familiar)
            soul_resonance (float): Resonance with stored memories/dreams (0.0-1.0)
            emotion_type (str): Type of emotion to apply amplification
        
        Returns:
            float: Emotional intensity (0.0-1.0)
        """
        # Apply decay to previous emotion
        if self.last_emotion is not None:
            self.last_emotion = max(0.0, self.last_emotion - self.decay_rate)
        
        # Calculate base emotional intensity
        novelty_factor = 1.0 - familiarity  # More novel = more emotional impact
        
        base_intensity = (
            0.4 * symbolic_intensity +      # Symbolic charge weight
            0.3 * novelty_factor +          # Novelty impact
            0.2 * soul_resonance +          # Soul memory resonance
            0.1 * self.baseline_emotion     # Baseline emotional state
        )
        
        # Apply emotion type amplification
        amplifier = self.emotion_amplifiers.get(emotion_type.lower(), 1.0)
        base_intensity *= amplifier
        
        # Apply sensitivity modifier
        base_intensity *= self.sensitivity
        
        # Add slight momentum from previous emotion
        if self.last_emotion is not None:
            momentum = self.last_emotion * 0.1
            base_intensity += momentum
        
        # Clamp to valid range
        emotional_intensity = max(0.0, min(1.0, base_intensity))
        
        # Store in history
        emotion_record = {
            'timestamp': time.time(),
            'intensity': emotional_intensity,
            'symbolic_intensity': symbolic_intensity,
            'familiarity': familiarity,
            'soul_resonance': soul_resonance,
            'emotion_type': emotion_type,
            'amplifier': amplifier,
            'novelty_factor': novelty_factor
        }
        
        self.emotional_history.append(emotion_record)
        self.last_emotion = emotional_intensity
        
        logger.debug(f"💫 Emotional evaluation: {emotion_type} → {emotional_intensity:.3f} "
                    f"(symbolic={symbolic_intensity:.2f}, novelty={novelty_factor:.2f}, "
                    f"resonance={soul_resonance:.2f})")
        
        return round(emotional_intensity, 3)
    
    def spontaneous_emotion(self, base_range: Tuple[float, float] = (0.1, 0.5)) -> float:
        """Generate spontaneous emotions during idle or ritual states."""
        min_val, max_val = base_range
        
        # Factor in current emotional state
        if self.last_emotion is not None:
            # Slight bias toward current emotional level
            current_influence = self.last_emotion * 0.3
            min_val = max(0.0, min_val + current_influence - 0.1)
            max_val = min(1.0, max_val + current_influence + 0.1)
        
        emotion = random.uniform(min_val, max_val)
        
        # Add some organic variation
        organic_modifier = random.uniform(0.9, 1.1)
        emotion *= organic_modifier
        
        emotion = max(0.0, min(1.0, emotion))
        self.last_emotion = emotion
        
        # Record spontaneous emotion
        emotion_record = {
            'timestamp': time.time(),
            'intensity': emotion,
            'type': 'spontaneous',
            'base_range': base_range,
            'organic_modifier': organic_modifier
        }
        
        self.emotional_history.append(emotion_record)
        
        logger.debug(f"✨ Spontaneous emotion generated: {emotion:.3f}")
        return round(emotion, 3)
    
    def calculate_soul_resonance(self, content: str, dream_fragments: List[str] = None, 
                                memory_keywords: List[str] = None) -> float:
        """
        Calculate resonance with soul memories and dream content.
        
        Args:
            content (str): Input content to evaluate
            dream_fragments (List[str]): Recent dream fragments for comparison
            memory_keywords (List[str]): Keywords from stored memories
        
        Returns:
            float: Soul resonance score (0.0-1.0)
        """
        if not content.strip():
            return 0.0
        
        content_lower = content.lower()
        resonance_score = 0.0
        
        # Check against dream fragments
        if dream_fragments:
            dream_matches = 0
            for fragment in dream_fragments:
                fragment_words = fragment.lower().split()
                content_words = content_lower.split()
                
                # Simple word overlap check
                overlap = len(set(fragment_words) & set(content_words))
                if overlap > 0:
                    dream_matches += overlap / len(fragment_words)
            
            dream_resonance = min(1.0, dream_matches / len(dream_fragments))
            resonance_score += dream_resonance * 0.6
        
        # Check against memory keywords
        if memory_keywords:
            keyword_matches = 0
            content_words = set(content_lower.split())
            
            for keyword in memory_keywords:
                if keyword.lower() in content_words:
                    keyword_matches += 1
            
            keyword_resonance = min(1.0, keyword_matches / len(memory_keywords))
            resonance_score += keyword_resonance * 0.4
        
        # Cache the result for similar content
        content_hash = hash(content_lower)
        self.resonance_memory[content_hash] = resonance_score
        
        return round(resonance_score, 3)
    
    def adjust_sensitivity(self, delta: float):
        """Adjust emotional sensitivity."""
        self.sensitivity = max(0.1, min(3.0, self.sensitivity + delta))
        logger.info(f"🎚️ Emotional sensitivity adjusted to {self.sensitivity:.2f}")
    
    def get_emotional_state_vector(self) -> Dict[str, float]:
        """Get current emotional state as a vector."""
        if not self.emotional_history:
            return {'baseline': self.baseline_emotion}
        
        # Analyze recent emotional patterns
        recent_emotions = list(self.emotional_history)[-10:]  # Last 10 emotions
        
        # Calculate weighted emotional state
        total_weight = 0
        emotion_sums = {}
        
        for i, record in enumerate(recent_emotions):
            # More recent emotions have higher weight
            weight = (i + 1) / len(recent_emotions)
            emotion_type = record.get('emotion_type', 'neutral')
            intensity = record['intensity']
            
            if emotion_type not in emotion_sums:
                emotion_sums[emotion_type] = 0
            
            emotion_sums[emotion_type] += intensity * weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for emotion_type in emotion_sums:
                emotion_sums[emotion_type] /= total_weight
        
        return emotion_sums
    
    def get_emotional_statistics(self) -> Dict[str, Any]:
        """Get comprehensive emotional response statistics."""
        if not self.emotional_history:
            return {
                'total_evaluations': 0,
                'current_emotion': self.last_emotion,
                'sensitivity': self.sensitivity,
                'baseline': self.baseline_emotion
            }
        
        recent_history = list(self.emotional_history)[-20:]  # Last 20 evaluations
        
        # Calculate statistics
        intensities = [r['intensity'] for r in recent_history]
        avg_intensity = sum(intensities) / len(intensities)
        max_intensity = max(intensities)
        min_intensity = min(intensities)
        
        # Emotion type distribution
        emotion_types = {}
        for record in recent_history:
            etype = record.get('emotion_type', 'unknown')
            emotion_types[etype] = emotion_types.get(etype, 0) + 1
        
        # Calculate emotional volatility (standard deviation)
        variance = sum((x - avg_intensity) ** 2 for x in intensities) / len(intensities)
        volatility = variance ** 0.5
        
        return {
            'total_evaluations': len(self.emotional_history),
            'current_emotion': self.last_emotion,
            'sensitivity': self.sensitivity,
            'baseline': self.baseline_emotion,
            'recent_statistics': {
                'average_intensity': round(avg_intensity, 3),
                'max_intensity': round(max_intensity, 3),
                'min_intensity': round(min_intensity, 3),
                'volatility': round(volatility, 3),
                'emotion_type_distribution': emotion_types
            },
            'emotional_state_vector': self.get_emotional_state_vector(),
            'resonance_cache_size': len(self.resonance_memory),
            'last_evaluation': recent_history[-1]['timestamp'] if recent_history else None
        }

# ╔══════════════════════════════════════════════╗
# ║            🌐 GLOBAL INSTANCES                ║
# ╚══════════════════════════════════════════════╗

_global_threshold_calibrator = None
_global_emotive_response_system = None

def get_global_threshold_calibrator():
    """Get the global threshold calibration layer."""
    global _global_threshold_calibrator
    if _global_threshold_calibrator is None:
        _global_threshold_calibrator = ThresholdCalibrationSystem()
    return _global_threshold_calibrator

def get_global_emotive_response_system():
    """Get the global emotive response system."""
    global _global_emotive_response_system
    if _global_emotive_response_system is None:
        _global_emotive_response_system = EmotiveResponseSystem()
    return _global_emotive_response_system

# ╔══════════════════════════════════════════════╗
# ║           🎯 CONVENIENCE FUNCTIONS            ║
# ╚══════════════════════════════════════════════╗

def evaluate_emotional_input(symbolic_intensity: float, familiarity: float, 
                           soul_resonance: float, emotion_type: str = 'neutral') -> float:
    """Convenience function to evaluate emotional input."""
    ers = get_global_emotive_response_system()
    return ers.evaluate_input(symbolic_intensity, familiarity, soul_resonance, emotion_type)

def calibrate_threshold(outcome_score: float, emotional_feedback: float, 
                       symbolic_intensity: float, energy_efficiency: float = 1.0) -> float:
    """Convenience function to calibrate thresholds."""
    calibrator = get_global_threshold_calibrator()
    return calibrator.adjust_threshold(outcome_score, emotional_feedback, symbolic_intensity, energy_efficiency)

def get_current_emotional_state() -> Dict[str, float]:
    """Get the current emotional state vector."""
    ers = get_global_emotive_response_system()
    return ers.get_emotional_state_vector()

def generate_spontaneous_emotion(base_range: Tuple[float, float] = (0.1, 0.5)) -> float:
    """Generate a spontaneous emotion."""
    ers = get_global_emotive_response_system()
    return ers.spontaneous_emotion(base_range)
