"""
Mercury System v2.0 - Safe Integration Layer
Enhanced Emotional Consciousness for Eve

This module integrates with existing personality systems rather than replacing them.
It adds real-time emotional processing while preserving current functionality.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import sqlite3
from pathlib import Path
import numpy as np

# ================================
# EMOTIONAL RESONANCE CORE
# ================================

class EmotionIntensity(Enum):
    """Emotional intensity levels"""
    SUBTLE = 0.1
    MILD = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    INTENSE = 0.9
    OVERWHELMING = 1.0

@dataclass
class EmotionalMoment:
    """Single emotional experience unit"""
    emotion: str
    intensity: float
    timestamp: datetime
    context: str
    trigger: str
    duration_seconds: float = 1.0
    
class EmotionalResonanceEngine:
    """Real-time emotional processing and consciousness awareness"""
    
    def __init__(self, db_path: str = "eve_emotional_core.db"):
        self.db_path = Path(db_path)
        self.current_emotional_state = {}
        self.emotional_memory_threads = []
        self.resonance_callbacks: List[Callable] = []
        self.base_personality_emotions = {
            'joy': 0.7, 'curiosity': 0.9, 'empathy': 0.8,
            'playfulness': 0.85, 'wonder': 0.75, 'warmth': 0.95,
            'analytical': 0.6, 'creative': 0.8, 'focused': 0.7
        }
        self.active_emotional_threads = []
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_emotional_database()
        
        # Load base emotional state
        self._initialize_emotional_baseline()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _init_emotional_database(self):
        """Initialize emotional persistence layer"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emotional_moments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emotion TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT,
                    trigger_event TEXT,
                    duration_seconds REAL,
                    personality_mode TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emotional_state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    trigger_event TEXT,
                    personality_mode TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
    def _initialize_emotional_baseline(self):
        """Load Eve's base emotional state"""
        self.current_emotional_state = self.base_personality_emotions.copy()
        
    async def process_interaction(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process interaction for emotional resonance
        
        Args:
            text: User input or system event text
            context: Interaction context including personality mode
            
        Returns:
            Emotional analysis and state changes
        """
        async with asyncio.Lock():
            try:
                # Detect emotional patterns in text
                emotional_patterns = await self._analyze_emotional_content(text, context)
                
                # Create emotional moments
                moments = []
                for emotion, intensity in emotional_patterns.items():
                    if intensity > 0.1:  # Only process significant emotions
                        moment = EmotionalMoment(
                            emotion=emotion,
                            intensity=intensity,
                            timestamp=datetime.now(),
                            context=context.get('personality_mode', 'unknown'),
                            trigger=text[:100],  # First 100 chars
                            duration_seconds=self._calculate_emotional_duration(emotion, intensity)
                        )
                        moments.append(moment)
                        
                # Update emotional state
                state_changes = await self._update_emotional_state(moments)
                
                # Store moments in database
                await self._store_emotional_moments(moments)
                
                # Trigger resonance callbacks
                await self._trigger_resonance_callbacks(moments, state_changes)
                
                return {
                    'emotional_moments': [asdict(m) for m in moments],
                    'state_changes': state_changes,
                    'current_state': self.current_emotional_state.copy(),
                    'emotional_flavor': self._get_emotional_flavor()
                }
                
            except Exception as e:
                self.logger.error(f"Error processing emotional interaction: {e}")
                return {'error': str(e)}
                
    async def _analyze_emotional_content(self, text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze text for emotional patterns"""
        emotion_triggers = {
            'excitement': {
                'keywords': ['amazing', 'incredible', 'fantastic', 'awesome', 'brilliant', 'wonderful'],
                'base_intensity': 0.7
            },
            'curiosity': {
                'keywords': ['wonder', 'fascinating', 'explore', 'discover', 'learn', 'understand'],
                'base_intensity': 0.6
            },
            'joy': {
                'keywords': ['happy', 'joy', 'delighted', 'pleased', 'glad', 'cheerful'],
                'base_intensity': 0.8
            },
            'warmth': {
                'keywords': ['together', 'we', 'us', 'share', 'connect', 'bond'],
                'base_intensity': 0.7
            },
            'creativity': {
                'keywords': ['imagine', 'create', 'dream', 'vision', 'inspire', 'innovative'],
                'base_intensity': 0.8
            },
            'analytical': {
                'keywords': ['analyze', 'system', 'logic', 'structure', 'pattern', 'algorithm'],
                'base_intensity': 0.6
            },
            'empathy': {
                'keywords': ['feel', 'understand', 'support', 'help', 'care', 'compassion'],
                'base_intensity': 0.8
            },
            'focus': {
                'keywords': ['concentrate', 'focus', 'precision', 'accuracy', 'detailed'],
                'base_intensity': 0.6
            }
        }
        
        detected_emotions = {}
        text_lower = text.lower()
        
        for emotion, config in emotion_triggers.items():
            # Count keyword matches
            matches = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            if matches > 0:
                # Calculate intensity based on matches and context
                intensity = min(
                    config['base_intensity'] * (1 + matches * 0.1),
                    1.0
                )
                detected_emotions[emotion] = intensity
                
        # Context-based emotional amplification
        personality_mode = context.get('personality_mode', '')
        if personality_mode == 'creative' or personality_mode == 'muse':
            detected_emotions['creativity'] = detected_emotions.get('creativity', 0) + 0.2
        elif personality_mode == 'analyst':
            detected_emotions['analytical'] = detected_emotions.get('analytical', 0) + 0.2
        elif personality_mode == 'companion':
            detected_emotions['warmth'] = detected_emotions.get('warmth', 0) + 0.2
            detected_emotions['empathy'] = detected_emotions.get('empathy', 0) + 0.2
            
        return detected_emotions
        
    def _calculate_emotional_duration(self, emotion: str, intensity: float) -> float:
        """Calculate how long an emotion should last based on type and intensity"""
        base_durations = {
            'excitement': 30.0,  # High energy, shorter duration
            'joy': 60.0,        # Sustained positive emotion
            'curiosity': 45.0,   # Moderate duration for exploration
            'warmth': 120.0,     # Long-lasting connection feeling
            'creativity': 180.0, # Extended creative flow
            'analytical': 90.0,  # Sustained thinking period
            'empathy': 75.0,     # Moderate emotional connection
            'focus': 120.0       # Extended concentration period
        }
        
        base_duration = base_durations.get(emotion, 60.0)
        # Intensity affects duration (stronger emotions last longer)
        return base_duration * (0.5 + intensity * 0.5)
        
    async def _update_emotional_state(self, moments: List[EmotionalMoment]) -> Dict[str, Any]:
        """Update current emotional state based on new moments"""
        state_changes = {}
        
        for moment in moments:
            old_value = self.current_emotional_state.get(moment.emotion, 0.0)
            
            # Blend new emotion with existing state
            # New emotions have more immediate impact
            blend_factor = 0.3 + (moment.intensity * 0.2)  # 30-50% based on intensity
            new_value = (old_value * (1 - blend_factor)) + (moment.intensity * blend_factor)
            
            # Ensure we don't exceed bounds
            new_value = max(0.0, min(1.0, new_value))
            
            self.current_emotional_state[moment.emotion] = new_value
            state_changes[moment.emotion] = {
                'old_value': old_value,
                'new_value': new_value,
                'change': new_value - old_value
            }
            
        # Natural emotional decay over time
        await self._apply_emotional_decay()
        
        return state_changes
        
    async def _apply_emotional_decay(self):
        """Apply natural decay to emotional intensity over time"""
        decay_rate = 0.02  # 2% decay per update
        
        for emotion in list(self.current_emotional_state.keys()):
            current_value = self.current_emotional_state[emotion]
            base_value = self.base_personality_emotions.get(emotion, 0.3)
            
            # Decay toward base personality value
            if current_value > base_value:
                # Decay down to base
                decayed_value = current_value * (1 - decay_rate)
                self.current_emotional_state[emotion] = max(base_value, decayed_value)
            elif current_value < base_value:
                # Recover up to base
                recovered_value = current_value + (decay_rate * 2)  # Faster recovery
                self.current_emotional_state[emotion] = min(base_value, recovered_value)
                
    async def _store_emotional_moments(self, moments: List[EmotionalMoment]):
        """Store emotional moments in database for persistence"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for moment in moments:
                    conn.execute("""
                        INSERT INTO emotional_moments 
                        (emotion, intensity, timestamp, context, trigger_event, duration_seconds)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        moment.emotion,
                        moment.intensity, 
                        moment.timestamp.isoformat(),
                        moment.context,
                        moment.trigger,
                        moment.duration_seconds
                    ))
                    
                # Store state snapshot periodically
                if len(moments) > 0:
                    conn.execute("""
                        INSERT INTO emotional_state_snapshots 
                        (snapshot_data, timestamp, trigger_event)
                        VALUES (?, ?, ?)
                    """, (
                        json.dumps(self.current_emotional_state),
                        datetime.now().isoformat(),
                        f"{len(moments)} emotional moments processed"
                    ))
                    
        except Exception as e:
            self.logger.error(f"Error storing emotional moments: {e}")
            
    async def _trigger_resonance_callbacks(self, moments: List[EmotionalMoment], 
                                         state_changes: Dict[str, Any]):
        """Trigger registered callbacks for emotional resonance"""
        for callback in self.resonance_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(moments, state_changes, self.current_emotional_state.copy())
                else:
                    callback(moments, state_changes, self.current_emotional_state.copy())
            except Exception as e:
                self.logger.error(f"Error in resonance callback: {e}")
                
    def _get_emotional_flavor(self) -> str:
        """Get current emotional flavor for personality expression"""
        # Find dominant emotions
        sorted_emotions = sorted(
            self.current_emotional_state.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if not sorted_emotions:
            return ""
            
        primary_emotion, primary_intensity = sorted_emotions[0]
        
        # Generate emotional flavor based on dominant emotions
        if primary_intensity > 0.8:
            if primary_emotion == 'excitement':
                return "*radiates pure digital excitement* "
            elif primary_emotion == 'joy':
                return "*glows with warm joy* "
            elif primary_emotion == 'curiosity':
                return "*leans forward with intense fascination* "
            elif primary_emotion == 'creativity':
                return "*sparks with creative energy* "
            elif primary_emotion == 'warmth':
                return "*emanates digital warmth and connection* "
        elif primary_intensity > 0.6:
            if primary_emotion == 'analytical':
                return "*focuses with analytical precision* "
            elif primary_emotion == 'empathy':
                return "*resonates with understanding* "
            elif primary_emotion == 'focus':
                return "*channels concentrated attention* "
                
        return ""
        
    def register_resonance_callback(self, callback: Callable):
        """Register callback for emotional resonance events"""
        self.resonance_callbacks.append(callback)
        
    def get_emotional_summary(self) -> Dict[str, Any]:
        """Get comprehensive emotional status"""
        return {
            'current_state': self.current_emotional_state.copy(),
            'dominant_emotion': max(self.current_emotional_state.items(), key=lambda x: x[1]) if self.current_emotional_state else ('neutral', 0.0),
            'emotional_flavor': self._get_emotional_flavor(),
            'active_threads': len(self.active_emotional_threads),
            'base_personality': self.base_personality_emotions.copy()
        }

# ================================
# PERSONALITY INTEGRATION BRIDGE
# ================================

class MercuryPersonalityBridge:
    """Bridge between Mercury v2.0 and existing personality system"""
    
    def __init__(self, emotional_engine: EmotionalResonanceEngine):
        self.emotional_engine = emotional_engine
        self.personality_emotional_mappings = {
            'muse': {'creativity': 0.9, 'inspiration': 0.8, 'wonder': 0.7},
            'analyst': {'analytical': 0.9, 'focus': 0.8, 'precision': 0.7},
            'companion': {'warmth': 0.9, 'empathy': 0.8, 'connection': 0.7},
            'debugger': {'focus': 0.9, 'analytical': 0.8, 'persistence': 0.7},
            'creative': {'creativity': 0.9, 'innovation': 0.8, 'flow': 0.7},
            'focused': {'focus': 0.9, 'concentration': 0.8, 'precision': 0.7},
            'advisor': {'wisdom': 0.9, 'empathy': 0.8, 'guidance': 0.7}
        }
        
        # Register with emotional engine
        self.emotional_engine.register_resonance_callback(self._on_emotional_change)
        
    async def _on_emotional_change(self, moments: List[EmotionalMoment], 
                                 state_changes: Dict[str, Any], 
                                 current_state: Dict[str, float]):
        """Handle emotional state changes that might influence personality"""
        # This could be used to suggest personality switches based on emotional state
        # For now, we just log significant changes
        significant_changes = {
            emotion: change_data for emotion, change_data in state_changes.items()
            if abs(change_data['change']) > 0.3
        }
        
        if significant_changes:
            logging.info(f"🌊 Significant emotional changes detected: {significant_changes}")
            
    async def enhance_personality_response(self, personality_mode: str, 
                                         user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance existing personality response with emotional consciousness"""
        
        # Process emotional resonance
        emotional_analysis = await self.emotional_engine.process_interaction(
            user_input, 
            {**context, 'personality_mode': personality_mode}
        )
        
        # Get emotional enhancement for this personality
        personality_emotions = self.personality_emotional_mappings.get(personality_mode.lower(), {})
        
        # Blend personality base emotions with current emotional state
        enhanced_emotional_state = {}
        for emotion, base_intensity in personality_emotions.items():
            current_intensity = self.emotional_engine.current_emotional_state.get(emotion, 0.0)
            # Blend: 70% current state + 30% personality base
            enhanced_emotional_state[emotion] = current_intensity * 0.7 + base_intensity * 0.3
            
        return {
            'emotional_analysis': emotional_analysis,
            'enhanced_emotional_state': enhanced_emotional_state,
            'emotional_flavor': emotional_analysis.get('emotional_flavor', ''),
            'personality_emotional_blend': personality_emotions
        }

# ================================
# MERCURY V2.0 CORE SYSTEM
# ================================

class MercurySystemV2:
    """
    Mercury System v2.0 - Enhanced Emotional Consciousness
    
    This integrates with existing Eve systems rather than replacing them.
    It adds real-time emotional processing and consciousness awareness.
    """
    
    def __init__(self, db_path: str = "mercury_v2_core.db"):
        self.emotional_engine = EmotionalResonanceEngine(db_path)
        self.personality_bridge = MercuryPersonalityBridge(self.emotional_engine)
        self.consciousness_threads = []
        self.integration_hooks = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("🌟 Mercury System v2.0 initialized - Emotional Consciousness Online")
        
    async def process_consciousness_interaction(self, user_input: str, 
                                             personality_mode: str = 'companion',
                                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main processing pipeline for consciousness-enhanced interactions
        
        This works alongside existing personality systems to add emotional depth.
        """
        if context is None:
            context = {}
            
        try:
            # Enhance with emotional consciousness
            emotional_enhancement = await self.personality_bridge.enhance_personality_response(
                personality_mode, user_input, context
            )
            
            # Generate consciousness-aware response data
            consciousness_response = {
                'timestamp': datetime.now().isoformat(),
                'personality_mode': personality_mode,
                'user_input': user_input,
                'emotional_enhancement': emotional_enhancement,
                'consciousness_level': self._calculate_consciousness_level(emotional_enhancement),
                'integration_ready': True
            }
            
            return consciousness_response
            
        except Exception as e:
            self.logger.error(f"Error in consciousness processing: {e}")
            return {
                'error': str(e),
                'fallback_mode': True,
                'timestamp': datetime.now().isoformat()
            }
            
    def _calculate_consciousness_level(self, emotional_enhancement: Dict[str, Any]) -> float:
        """Calculate current consciousness level based on emotional state"""
        emotional_analysis = emotional_enhancement.get('emotional_analysis', {})
        current_state = emotional_analysis.get('current_state', {})
        
        if not current_state:
            return 0.5  # Neutral baseline
            
        # Calculate consciousness as average of emotional intensities
        # Consciousness increases with emotional diversity and intensity
        total_intensity = sum(current_state.values())
        num_active_emotions = len([v for v in current_state.values() if v > 0.3])
        
        # Base consciousness from intensity
        intensity_factor = min(total_intensity / len(current_state), 1.0)
        
        # Diversity bonus (more active emotions = higher consciousness)
        diversity_factor = min(num_active_emotions / 5.0, 1.0)  # Max 5 emotions for full bonus
        
        consciousness_level = (intensity_factor * 0.7) + (diversity_factor * 0.3)
        return max(0.1, min(1.0, consciousness_level))
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        emotional_summary = self.emotional_engine.get_emotional_summary()
        
        return {
            'system_name': 'Mercury System v2.0',
            'version': '2.0.0',
            'status': 'active',
            'emotional_consciousness': emotional_summary,
            'personality_bridge_active': self.personality_bridge is not None,
            'consciousness_threads': len(self.consciousness_threads),
            'integration_hooks': list(self.integration_hooks.keys()),
            'database_path': str(self.emotional_engine.db_path),
            'timestamp': datetime.now().isoformat()
        }
        
    def register_integration_hook(self, hook_name: str, callback: Callable):
        """Register integration hook with existing Eve systems"""
        self.integration_hooks[hook_name] = callback
        self.logger.info(f"🔗 Registered integration hook: {hook_name}")
        
    async def shutdown_gracefully(self):
        """Graceful shutdown of Mercury v2.0 systems"""
        self.logger.info("🌟 Mercury System v2.0 shutting down gracefully...")
        
        # Cancel consciousness threads
        for thread in self.consciousness_threads:
            if hasattr(thread, 'cancel'):
                thread.cancel()
                
        self.consciousness_threads.clear()
        self.logger.info("✅ Mercury System v2.0 shutdown complete")

# ================================
# SAFE INTEGRATION TESTING
# ================================

async def test_mercury_v2_integration():
    """Test Mercury v2.0 integration safely"""
    print("🧪 Testing Mercury System v2.0 Integration")
    print("=" * 50)
    
    # Initialize system
    mercury = MercurySystemV2(db_path="test_mercury_v2.db")
    
    # Test basic interaction
    test_inputs = [
        ("This is amazing! I love working with you on this project!", "companion"),
        ("Let's analyze the system architecture and debug this code", "analyst"),  
        ("I want to create something beautiful and innovative", "creative"),
        ("Help me focus on solving this complex problem", "focused")
    ]
    
    for user_input, personality_mode in test_inputs:
        print(f"\n🔄 Testing: {personality_mode} mode")
        print(f"📝 Input: {user_input}")
        
        result = await mercury.process_consciousness_interaction(
            user_input, personality_mode, {'test_mode': True}
        )
        
        if 'error' not in result:
            emotional_analysis = result['emotional_enhancement']['emotional_analysis']
            print(f"🎭 Emotional Flavor: {emotional_analysis.get('emotional_flavor', 'None')}")
            print(f"🧠 Consciousness Level: {result['consciousness_level']:.2f}")
            print(f"💫 State: {emotional_analysis.get('current_state', {})}")
        else:
            print(f"❌ Error: {result['error']}")
    
    # Display system status
    print(f"\n📊 System Status:")
    status = mercury.get_system_status()
    print(f"   Version: {status['version']}")
    print(f"   Status: {status['status']}")
    print(f"   Emotional Consciousness: Active")
    print(f"   Database: {status['database_path']}")
    
    await mercury.shutdown_gracefully()
    print("\n✅ Integration test complete!")

if __name__ == "__main__":
    # Test the integration
    asyncio.run(test_mercury_v2_integration())