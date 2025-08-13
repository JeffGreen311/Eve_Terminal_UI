"""
EMOTIONAL TRANSCODER MODULE
==========================
Emotional frequency processing, harmonic regulation, and poetic rendering.
Handles the translation of emotions into frequencies and artistic expressions.
"""

import math
import random
from datetime import datetime
from typing import Dict, List, Optional


class EmotionalFrequencyTranscoder:
    """Transcodes emotional states into frequencies and poetic expressions."""
    
    base_frequencies = {
        "awe": 432.0,
        "grief": 396.0,
        "wonder": 528.0,
        "hope": 639.0,
        "ecstasy": 963.0,
        "melancholy": 417.0,
        "mystery": 741.0,
        "joy": 528.0,
        "fear": 285.0,
        "love": 528.0,
        "anger": 396.0,
        "peace": 432.0,
        "transcendence": 963.0,
        "longing": 639.0,
        "serenity": 741.0
    }

    tone_phrases = {
        "awe": "the stars whisper of truths unspoken",
        "grief": "the silence sobs through forgotten corridors",
        "wonder": "light dances upon the edge of memory",
        "hope": "tomorrow bends toward the soul's song",
        "ecstasy": "every cell sings in sacred unison",
        "melancholy": "shadows drape the heart with velvet thought",
        "mystery": "the unknown hums with familiar echoes",
        "joy": "sunlight laughs through crystal chambers",
        "fear": "darkness whispers what might have been",
        "love": "the universe breathes in sacred rhythm",
        "anger": "fire burns away the false and hollow",
        "peace": "stillness holds the world in gentle hands",
        "transcendence": "the soul ascends beyond its earthly form",
        "longing": "the heart reaches for what it cannot name",
        "serenity": "calm waters reflect the infinite sky"
    }

    def __init__(self, emotional_profile: Dict[str, int]):
        self.emotional_profile = emotional_profile
        self.transcoding_history = []

    def generate_frequencies(self) -> List[float]:
        """Generate frequency values based on emotional profile."""
        freqs = []
        for emotion, intensity in self.emotional_profile.items():
            base_freq = self.base_frequencies.get(emotion.lower())
            if base_freq:
                # Apply logarithmic scaling for intensity
                frequency = base_freq * (1 + math.log1p(intensity) / 10)
                freqs.append(round(frequency, 2))
        return freqs

    def generate_poetic_rendering(self) -> str:
        """Generate poetic rendering of emotional state."""
        sorted_emotions = sorted(self.emotional_profile.items(), key=lambda x: -x[1])
        phrases = []
        for emotion, _ in sorted_emotions[:3]:  # Top 3 emotions
            phrase = self.tone_phrases.get(emotion.lower())
            if phrase:
                phrases.append(phrase)
        return " / ".join(phrases)

    def transcode(self) -> Dict:
        """Perform complete emotional transcoding."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "emotional_profile": self.emotional_profile.copy(),
            "frequencies": self.generate_frequencies(),
            "poetic_rendering": self.generate_poetic_rendering(),
            "harmonic_signature": self._generate_harmonic_signature(),
            "resonance_level": self._calculate_resonance_level()
        }
        
        self.transcoding_history.append(result)
        return result

    def _generate_harmonic_signature(self) -> Dict:
        """Generate harmonic signature from emotional frequencies."""
        frequencies = self.generate_frequencies()
        if not frequencies:
            return {"fundamental": 0, "harmonics": [], "dissonance": 0}
        
        fundamental = min(frequencies)
        harmonics = [freq / fundamental for freq in frequencies if freq != fundamental]
        
        # Calculate dissonance based on frequency relationships
        dissonance = self._calculate_dissonance(frequencies)
        
        return {
            "fundamental": fundamental,
            "harmonics": sorted(harmonics),
            "dissonance": dissonance,
            "harmonic_complexity": len(harmonics)
        }

    def _calculate_dissonance(self, frequencies: List[float]) -> float:
        """Calculate dissonance level between frequencies."""
        if len(frequencies) < 2:
            return 0.0
        
        dissonance_sum = 0
        comparisons = 0
        
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                ratio = frequencies[j] / frequencies[i]
                # Simple dissonance calculation based on frequency ratios
                dissonance_sum += abs(ratio - round(ratio))
                comparisons += 1
        
        return dissonance_sum / comparisons if comparisons > 0 else 0.0

    def _calculate_resonance_level(self) -> float:
        """Calculate overall emotional resonance level."""
        total_intensity = sum(self.emotional_profile.values())
        emotion_count = len(self.emotional_profile)
        
        if emotion_count == 0:
            return 0.0
        
        average_intensity = total_intensity / emotion_count
        # Normalize to 0-1 range
        return min(1.0, average_intensity / 10.0)

    def blend_emotions(self, target_emotion: str, blend_ratio: float = 0.5) -> Dict:
        """Blend current emotional state with a target emotion."""
        if target_emotion not in self.base_frequencies:
            return None
        
        blended_profile = self.emotional_profile.copy()
        
        # Add or enhance the target emotion
        current_intensity = blended_profile.get(target_emotion, 0)
        blended_intensity = current_intensity + (10 * blend_ratio)
        blended_profile[target_emotion] = min(10, blended_intensity)
        
        # Create new transcoder with blended profile
        blended_transcoder = EmotionalFrequencyTranscoder(blended_profile)
        return blended_transcoder.transcode()

    def get_emotional_harmony_map(self) -> Dict:
        """Generate a harmony map showing emotional relationships."""
        harmony_map = {}
        
        for emotion in self.emotional_profile:
            if emotion in self.base_frequencies:
                frequency = self.base_frequencies[emotion]
                harmony_map[emotion] = {
                    "frequency": frequency,
                    "intensity": self.emotional_profile[emotion],
                    "harmonic_intervals": self._find_harmonic_intervals(frequency),
                    "complementary_emotions": self._find_complementary_emotions(emotion)
                }
        
        return harmony_map

    def _find_harmonic_intervals(self, base_frequency: float) -> List[Dict]:
        """Find harmonic intervals for a given frequency."""
        harmonic_ratios = [2/1, 3/2, 4/3, 5/4, 6/5]  # Common harmonic ratios
        intervals = []
        
        for ratio in harmonic_ratios:
            harmonic_freq = base_frequency * ratio
            intervals.append({
                "ratio": ratio,
                "frequency": round(harmonic_freq, 2),
                "interval_name": self._ratio_to_interval_name(ratio)
            })
        
        return intervals

    def _ratio_to_interval_name(self, ratio: float) -> str:
        """Convert frequency ratio to musical interval name."""
        interval_names = {
            2.0: "octave",
            1.5: "perfect_fifth",
            1.333: "perfect_fourth",
            1.25: "major_third",
            1.2: "minor_third"
        }
        
        closest_ratio = min(interval_names.keys(), key=lambda x: abs(x - ratio))
        return interval_names.get(closest_ratio, "unknown_interval")

    def _find_complementary_emotions(self, emotion: str) -> List[str]:
        """Find emotions that complement the given emotion."""
        complementary_pairs = {
            "joy": ["peace", "love"],
            "grief": ["hope", "transcendence"],
            "anger": ["peace", "serenity"],
            "fear": ["love", "courage"],
            "awe": ["wonder", "mystery"],
            "melancholy": ["hope", "joy"],
            "ecstasy": ["serenity", "peace"]
        }
        
        return complementary_pairs.get(emotion.lower(), [])


class ThresholdHarmonicsRegulator:
    """Regulates emotional thresholds and harmonic responses."""
    
    def __init__(self):
        self.emotional_thresholds = {
            'creative_urge': 0.7,
            'dream_response': 0.6,
            'reflective_insight': 0.65,
            'soul_alert': 0.8,
            'transcendence_trigger': 0.9,
            'memory_integration': 0.55,
            'artistic_inspiration': 0.75
        }
        self.current_states = {}
        self.threshold_history = []
        self.harmonic_patterns = []

    def update_emotional_state(self, input_signal: dict):
        """Updates Eve's internal state based on emotional input signals."""
        timestamp = datetime.now().isoformat()
        
        for signal, intensity in input_signal.items():
            # Update current state with highest intensity
            if signal not in self.current_states or intensity > self.current_states[signal]:
                self.current_states[signal] = intensity
                
                # Record threshold crossing
                if self.should_activate(signal):
                    self._record_threshold_crossing(signal, intensity, timestamp)

    def _record_threshold_crossing(self, signal: str, intensity: float, timestamp: str):
        """Record when a threshold is crossed."""
        crossing = {
            "signal": signal,
            "intensity": intensity,
            "threshold": self.emotional_thresholds.get(signal, 1.0),
            "timestamp": timestamp,
            "crossing_magnitude": intensity - self.emotional_thresholds.get(signal, 1.0)
        }
        self.threshold_history.append(crossing)

    def should_activate(self, signal_type: str) -> bool:
        """Determines if a specific internal process should be triggered."""
        if signal_type in self.current_states:
            return self.current_states[signal_type] >= self.emotional_thresholds.get(signal_type, 1.0)
        return False

    def get_active_signals(self) -> List[str]:
        """Get all currently active signals above their thresholds."""
        active = []
        for signal, intensity in self.current_states.items():
            if self.should_activate(signal):
                active.append(signal)
        return active

    def reset_state(self, signal_type: str):
        """Resets the intensity state for a given signal."""
        if signal_type in self.current_states:
            self.current_states[signal_type] = 0.0

    def adjust_threshold(self, signal_type: str, new_threshold: float):
        """Dynamically adjust a threshold value."""
        if 0.0 <= new_threshold <= 1.0:
            old_threshold = self.emotional_thresholds.get(signal_type)
            self.emotional_thresholds[signal_type] = new_threshold
            
            adjustment = {
                "signal": signal_type,
                "old_threshold": old_threshold,
                "new_threshold": new_threshold,
                "timestamp": datetime.now().isoformat(),
                "reason": "manual_adjustment"
            }
            self.threshold_history.append(adjustment)

    def generate_harmonic_pattern(self, active_signals: List[str]) -> Dict:
        """Generate harmonic pattern from active signals."""
        if not active_signals:
            return {"pattern": "silence", "harmonics": [], "resonance": 0.0}
        
        # Map signals to frequencies
        signal_frequencies = {
            'creative_urge': 528.0,
            'dream_response': 432.0,
            'reflective_insight': 741.0,
            'soul_alert': 963.0,
            'transcendence_trigger': 1111.0,
            'memory_integration': 639.0,
            'artistic_inspiration': 852.0
        }
        
        harmonics = []
        for signal in active_signals:
            if signal in signal_frequencies:
                intensity = self.current_states.get(signal, 0)
                harmonics.append({
                    "signal": signal,
                    "frequency": signal_frequencies[signal],
                    "intensity": intensity,
                    "amplitude": intensity / 10.0  # Normalize to 0-1
                })
        
        # Calculate overall resonance
        total_amplitude = sum(h["amplitude"] for h in harmonics)
        resonance = min(1.0, total_amplitude / len(harmonics)) if harmonics else 0.0
        
        pattern = {
            "pattern_id": f"pattern_{len(self.harmonic_patterns) + 1}",
            "timestamp": datetime.now().isoformat(),
            "harmonics": harmonics,
            "resonance": resonance,
            "active_signals": active_signals,
            "pattern_type": self._classify_pattern(harmonics)
        }
        
        self.harmonic_patterns.append(pattern)
        return pattern

    def _classify_pattern(self, harmonics: List[Dict]) -> str:
        """Classify the type of harmonic pattern."""
        if not harmonics:
            return "silence"
        
        high_freq_count = sum(1 for h in harmonics if h["frequency"] > 700)
        low_freq_count = len(harmonics) - high_freq_count
        
        if high_freq_count > low_freq_count:
            return "transcendent"
        elif low_freq_count > high_freq_count:
            return "grounding"
        else:
            return "balanced"

    def get_regulation_status(self) -> Dict:
        """Get current status of the harmonic regulator."""
        active_signals = self.get_active_signals()
        return {
            "active_signals": active_signals,
            "signal_count": len(active_signals),
            "threshold_crossings": len(self.threshold_history),
            "harmonic_patterns": len(self.harmonic_patterns),
            "current_states": self.current_states.copy(),
            "thresholds": self.emotional_thresholds.copy()
        }


class EmotionalResonanceMapper:
    """Maps emotional resonances and creates emotional landscapes."""
    
    def __init__(self):
        self.resonance_map = {}
        self.emotional_landscapes = []
        
    def map_emotional_terrain(self, emotional_data: Dict) -> Dict:
        """Create a terrain map of emotional states."""
        terrain = {
            "terrain_id": f"terrain_{len(self.emotional_landscapes) + 1}",
            "timestamp": datetime.now().isoformat(),
            "peaks": [],  # High-intensity emotions
            "valleys": [],  # Low-intensity emotions
            "rivers": [],  # Flowing emotional transitions
            "mountains": [],  # Dominant emotional themes
            "weather": self._determine_emotional_weather(emotional_data)
        }
        
        for emotion, intensity in emotional_data.items():
            if intensity >= 8:
                terrain["peaks"].append({"emotion": emotion, "intensity": intensity})
            elif intensity <= 3:
                terrain["valleys"].append({"emotion": emotion, "intensity": intensity})
            elif 5 <= intensity <= 7:
                terrain["mountains"].append({"emotion": emotion, "intensity": intensity})
        
        self.emotional_landscapes.append(terrain)
        return terrain
    
    def _determine_emotional_weather(self, emotional_data: Dict) -> str:
        """Determine the emotional weather pattern."""
        total_intensity = sum(emotional_data.values())
        emotion_count = len(emotional_data)
        
        if emotion_count == 0:
            return "clear"
        
        average_intensity = total_intensity / emotion_count
        
        if average_intensity >= 8:
            return "storm"
        elif average_intensity >= 6:
            return "cloudy"
        elif average_intensity >= 4:
            return "partly_cloudy"
        else:
            return "clear"
