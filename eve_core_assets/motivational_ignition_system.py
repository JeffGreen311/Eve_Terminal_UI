# 🚀 MOTIVATIONAL IGNITION SYSTEM (MIS)
# Final integration layer that executes actions based on threshold evaluations

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

# Import our other modules
from .memory_imprinting_system import ActionType, get_global_threshold_motivator, get_global_memory_imprinting_module
from .threshold_calibration_system import get_global_threshold_calibrator, get_global_emotive_response_system
from .motivational_ignition_sequencer import get_global_motivational_sequencer, get_global_prioritization_core

logger = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════╗
# ║      🎭 MOTIVATIONAL IGNITION SYSTEM          ║
# ╚══════════════════════════════════════════════╗

class IgnitionResult:
    """Result of a motivational ignition process."""
    
    def __init__(self, trigger_type: ActionType, success: bool = True, 
                 message: str = "", artifacts: Optional[List[Any]] = None,
                 energy_consumed: float = 0.0, callback_result: Optional[Any] = None):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.trigger_type = trigger_type
        self.success = success
        self.message = message
        self.artifacts = artifacts or []
        self.energy_consumed = energy_consumed
        self.callback_result = callback_result
        self.duration = 0.0  # Will be set when process completes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "trigger_type": self.trigger_type.value if isinstance(self.trigger_type, ActionType) else str(self.trigger_type),
            "success": self.success,
            "message": self.message,
            "artifacts": self.artifacts,
            "energy_consumed": self.energy_consumed,
            "callback_result": self.callback_result,
            "duration": self.duration
        }

class MotivationalIgnitionSystem:
    """
    Main system that coordinates threshold evaluation, memory imprinting, 
    and action execution based on motivational drives.
    """
    
    def __init__(self):
        self.creative_callback = None
        self.emotional_callback = None
        self.ritual_callback = None
        self.analytical_callback = None
        self.contemplative_callback = None
        self.transformational_callback = None
        
        self.ignition_history = []
        self.callback_registry = {}
        self.auto_calibration = True
        self.feedback_integration = True
        
        # Integration with other systems
        self.threshold_motivator = get_global_threshold_motivator()
        self.memory_imprinter = get_global_memory_imprinting_module()
        self.threshold_calibrator = get_global_threshold_calibrator()
        self.emotive_response = get_global_emotive_response_system()
        self.motivational_sequencer = get_global_motivational_sequencer()
        self.prioritization_core = get_global_prioritization_core()
    
    def bind_callbacks(self, creative: Optional[Callable] = None, emotional: Optional[Callable] = None, 
                      ritual: Optional[Callable] = None, analytical: Optional[Callable] = None,
                      contemplative: Optional[Callable] = None, transformational: Optional[Callable] = None):
        """Bind callback functions for different ignition types."""
        self.creative_callback = creative
        self.emotional_callback = emotional
        self.ritual_callback = ritual
        self.analytical_callback = analytical
        self.contemplative_callback = contemplative
        self.transformational_callback = transformational
        
        # Also bind to the motivational sequencer
        self.motivational_sequencer.bind_callbacks(
            creative=creative,
            emotional=emotional,
            ritual=ritual,
            analytical=analytical,
            contemplative=contemplative,
            transformational=transformational
        )
        
        # Count active callbacks
        active_callbacks = sum(1 for cb in [creative, emotional, ritual, analytical, contemplative, transformational] if cb)
        logger.info(f"🔗 Bound {active_callbacks} callbacks to Motivational Ignition System")
    
    def register_callback(self, trigger_type: str, callback: Callable):
        """Register a custom callback for a specific trigger type."""
        self.callback_registry[trigger_type] = callback
        logger.info(f"📝 Registered custom callback for: {trigger_type}")
    
    def process_emotional_input(self, emotion: str, intensity: float, symbolic_content: str = "",
                               context: Optional[str] = None, familiarity: float = 0.5) -> IgnitionResult:
        """
        Process an emotional input through the complete motivational pipeline.
        
        Args:
            emotion: Type of emotion
            intensity: Emotional intensity (0.0-1.0)
            symbolic_content: Symbolic content associated with the emotion
            context: Optional context information
            familiarity: How familiar this input is (0.0-1.0)
        
        Returns:
            IgnitionResult: Result of the ignition process
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Evaluate emotional input through emotive response system
            soul_resonance = self._calculate_soul_resonance(symbolic_content)
            emotional_intensity = self.emotive_response.evaluate_input(
                symbolic_intensity=intensity,
                familiarity=familiarity,
                soul_resonance=soul_resonance,
                emotion_type=emotion
            )
            
            # Step 2: Register with prioritization core
            imprint_id = self.prioritization_core.register_imprint(
                label=emotion,
                resonance=emotional_intensity,
                urgency=intensity,
                energy_level=self.motivational_sequencer.energy_state.get_level(),
                symbolic_content=symbolic_content
            )
            
            # Step 3: Register with motivational sequencer
            drive_id = self.motivational_sequencer.register_emotional_event(
                emotion=emotion,
                intensity=intensity,
                symbolic_anchor=self._extract_symbolic_anchor(symbolic_content),
                context=context,
                urgency=intensity
            )
            
            # Step 4: Evaluate through threshold motivator
            signal_type = self._determine_signal_type(emotion, symbolic_content)
            action_type = self.threshold_motivator.evaluate(
                signal_strength=emotional_intensity,
                signal_type=signal_type,
                context={"emotion": emotion, "symbolic_content": symbolic_content, "original_intensity": intensity}
            )
            
            # Step 5: Execute ignition if threshold met
            ignition_result = self._execute_ignition(action_type, emotional_intensity, {
                "emotion": emotion,
                "intensity": intensity,
                "symbolic_content": symbolic_content,
                "context": context,
                "drive_id": drive_id,
                "imprint_id": imprint_id
            })
            
            # Step 6: Memory imprinting for significant events
            if emotional_intensity >= 0.4:  # Significant emotional event
                from .memory_imprinting_system import MemoryCategory
                category = self._determine_memory_category(emotion, action_type)
                
                memory_content = {
                    "emotion": emotion,
                    "intensity": intensity,
                    "symbolic_content": symbolic_content,
                    "context": context,
                    "action_triggered": action_type.value,
                    "ignition_success": ignition_result.success
                }
                
                self.memory_imprinter.imprint_memory(
                    data=memory_content,
                    emotion_level=emotional_intensity,
                    category=category,
                    source="emotional_processing",
                    tags=[emotion, signal_type, action_type.value],
                    symbolic_content=symbolic_content
                )
            
            # Step 7: Auto-calibration if enabled
            if self.auto_calibration and ignition_result.success:
                self._auto_calibrate(ignition_result, emotional_intensity, symbolic_content)
            
            # Record timing
            end_time = datetime.now()
            ignition_result.duration = (end_time - start_time).total_seconds()
            
            # Store in history
            self.ignition_history.append(ignition_result)
            
            logger.info(f"🚀 Emotional input processed: {emotion} ({intensity:.2f}) → {action_type.value} "
                       f"(success={ignition_result.success}, duration={ignition_result.duration:.2f}s)")
            
            return ignition_result
            
        except Exception as e:
            logger.error(f"❌ Error processing emotional input: {e}")
            error_result = IgnitionResult(
                trigger_type=ActionType.NO_ACTION,
                success=False,
                message=f"Processing failed: {e}",
                energy_consumed=0.0
            )
            error_result.duration = (datetime.now() - start_time).total_seconds()
            return error_result
    
    def _calculate_soul_resonance(self, symbolic_content: str) -> float:
        """Calculate resonance with stored soul memories."""
        if not symbolic_content.strip():
            return 0.0
        
        # Get recent dream fragments and memory keywords for resonance calculation
        try:
            recent_memories = self.memory_imprinter.retrieve_memories(limit=20)
            memory_keywords = []
            dream_fragments = []
            
            for memory in recent_memories:
                if hasattr(memory, 'symbolic_content') and memory.symbolic_content:
                    if 'dream' in memory.tags:
                        dream_fragments.append(memory.symbolic_content)
                    else:
                        memory_keywords.extend(memory.symbolic_content.split())
            
            return self.emotive_response.calculate_soul_resonance(
                content=symbolic_content,
                dream_fragments=dream_fragments[:5],  # Last 5 dream fragments
                memory_keywords=list(set(memory_keywords))[:20]  # Top 20 unique keywords
            )
            
        except Exception as e:
            logger.warning(f"Failed to calculate soul resonance: {e}")
            return 0.0
    
    def _extract_symbolic_anchor(self, symbolic_content: str) -> str:
        """Extract a symbolic anchor from content."""
        if not symbolic_content.strip():
            return "void_whisper"
        
        # Simple keyword mapping to symbolic anchors
        content_lower = symbolic_content.lower()
        
        anchor_keywords = {
            'mirror': 'mirror_self',
            'spiral': 'spiral_dance',
            'light': 'temple_light',
            'shadow': 'shadow_embrace',
            'fire': 'fire_forge',
            'crystal': 'crystal_matrix',
            'dream': 'dream_gate',
            'void': 'void_whisper',
            'sacred': 'sacred_geometry',
            'soul': 'soul_ache',
            'star': 'stellar_song',
            'chaos': 'chaos_order'
        }
        
        for keyword, anchor in anchor_keywords.items():
            if keyword in content_lower:
                return anchor
        
        return 'soul_ache'  # Default anchor
    
    def _determine_signal_type(self, emotion: str, symbolic_content: str) -> str:
        """Determine the signal type based on emotion and content."""
        creative_emotions = ['creativity', 'inspiration', 'generative_ache', 'joy']
        emotional_emotions = ['love', 'anger', 'fear', 'sadness']
        transformational_emotions = ['transcendence', 'transformation', 'divine_voltage']
        analytical_emotions = ['curiosity', 'constructive_drive']
        contemplative_emotions = ['melancholic_depth', 'radiant_insight']
        
        content_lower = symbolic_content.lower()
        
        # Check for ritual keywords
        if any(word in content_lower for word in ['ritual', 'sacred', 'temple', 'divine']):
            return 'ritual'
        
        # Check emotion type
        if emotion.lower() in creative_emotions:
            return 'creative'
        elif emotion.lower() in transformational_emotions:
            return 'transformational'
        elif emotion.lower() in analytical_emotions:
            return 'analytical'
        elif emotion.lower() in contemplative_emotions:
            return 'contemplative'
        elif emotion.lower() in emotional_emotions:
            return 'emotional'
        
        return 'creative'  # Default
    
    def _determine_memory_category(self, emotion: str, action_type: ActionType):
        """Determine appropriate memory category."""
        from .memory_imprinting_system import MemoryCategory
        
        if action_type == ActionType.CREATIVE_EXPRESSION:
            return MemoryCategory.CREATIVE_CORE
        elif action_type == ActionType.RITUAL_PROCESS:
            return MemoryCategory.RITUAL_LAYER
        elif action_type == ActionType.EMOTIONAL_RESPONSE:
            return MemoryCategory.EMOTIONAL_DEPTH
        elif action_type == ActionType.TRANSFORMATIONAL_PROCESS:
            return MemoryCategory.TRANSCENDENT_MOMENT
        elif action_type == ActionType.CONTEMPLATIVE_PROCESS:
            return MemoryCategory.SYMBOLIC_ARCHIVE
        else:
            return MemoryCategory.INTERACTION_LOG
    
    def _execute_ignition(self, action_type: ActionType, intensity: float, context: Dict[str, Any]) -> IgnitionResult:
        """Execute the ignition based on action type."""
        if action_type == ActionType.NO_ACTION:
            return IgnitionResult(
                trigger_type=action_type,
                success=True,
                message="Signal below threshold - no action triggered",
                energy_consumed=0.0
            )
        
        # Determine callback to use
        callback = None
        if action_type == ActionType.CREATIVE_EXPRESSION:
            callback = self.creative_callback
        elif action_type == ActionType.EMOTIONAL_RESPONSE:
            callback = self.emotional_callback
        elif action_type == ActionType.RITUAL_PROCESS:
            callback = self.ritual_callback
        elif action_type == ActionType.ANALYTICAL_PROCESS:
            callback = self.analytical_callback
        elif action_type == ActionType.CONTEMPLATIVE_PROCESS:
            callback = self.contemplative_callback
        elif action_type == ActionType.TRANSFORMATIONAL_PROCESS:
            callback = self.transformational_callback
        
        # Check for custom callbacks
        if not callback and action_type.value in self.callback_registry:
            callback = self.callback_registry[action_type.value]
        
        # Execute callback if available
        if callback:
            try:
                callback_result = callback(context, intensity)
                return IgnitionResult(
                    trigger_type=action_type,
                    success=True,
                    message=f"Successfully executed {action_type.value}",
                    artifacts=callback_result.get('artifacts', []) if isinstance(callback_result, dict) else [],
                    energy_consumed=intensity * 10,  # Energy cost proportional to intensity
                    callback_result=callback_result
                )
                
            except Exception as e:
                logger.error(f"❌ Callback execution failed for {action_type.value}: {e}")
                return IgnitionResult(
                    trigger_type=action_type,
                    success=False,
                    message=f"Callback execution failed: {e}",
                    energy_consumed=0.0
                )
        else:
            # No callback available - simulate action
            return IgnitionResult(
                trigger_type=action_type,
                success=True,
                message=f"Simulated {action_type.value} (no callback bound)",
                artifacts=[f"simulated_{action_type.value}_artifact"],
                energy_consumed=intensity * 5
            )
    
    def _auto_calibrate(self, ignition_result: IgnitionResult, emotional_intensity: float, symbolic_content: str):
        """Automatically calibrate thresholds based on ignition results."""
        if not self.auto_calibration:
            return
        
        try:
            # Calculate outcome score based on success and artifacts
            outcome_score = 1.0 if ignition_result.success else 0.2
            if ignition_result.artifacts:
                outcome_score = min(1.0, outcome_score + len(ignition_result.artifacts) * 0.1)
            
            # Use emotional intensity as feedback
            emotional_feedback = emotional_intensity
            
            # Symbolic intensity based on content length and keywords
            symbolic_intensity = min(1.0, len(symbolic_content.split()) / 20.0)
            
            # Energy efficiency
            energy_efficiency = 1.0
            if ignition_result.energy_consumed > 0:
                energy_efficiency = min(2.0, (outcome_score * 10) / ignition_result.energy_consumed)
            
            # Apply calibration
            self.threshold_calibrator.adjust_threshold(
                outcome_score=outcome_score,
                emotional_feedback=emotional_feedback,
                symbolic_intensity=symbolic_intensity,
                energy_efficiency=energy_efficiency
            )
            
            logger.debug(f"🎛️ Auto-calibrated thresholds based on ignition result")
            
        except Exception as e:
            logger.warning(f"Auto-calibration failed: {e}")
    
    def trigger_ignition_sequence(self) -> List[IgnitionResult]:
        """Trigger the full motivational ignition sequence."""
        try:
            # Get current top focus from prioritization core
            top_focuses = self.prioritization_core.get_top_focus(3)
            
            results = []
            
            for focus in top_focuses:
                # Process each focus through the ignition system
                result = self.process_emotional_input(
                    emotion=focus.label,
                    intensity=focus.resonance,
                    symbolic_content=focus.symbolic_content,
                    context=f"prioritized_focus_{focus.id}",
                    familiarity=max(0.0, 1.0 - focus.urgency)  # High urgency = low familiarity
                )
                results.append(result)
                
                # Remove processed imprint
                self.prioritization_core.remove_imprint(focus.id)
            
            # Also trigger any queued drives in the motivational sequencer
            sequencer_results = self.motivational_sequencer.ignite_motivational_sequence()
            
            logger.info(f"🚀 Ignition sequence triggered: {len(results)} focus processes, "
                       f"{len(sequencer_results)} sequencer drives")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ignition sequence failed: {e}")
            return []
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            # Get statistics from all integrated systems
            threshold_stats = self.threshold_motivator.get_threshold_statistics()
            memory_stats = self.memory_imprinter.get_memory_statistics()
            calibration_stats = self.threshold_calibrator.get_calibration_statistics()
            emotive_stats = self.emotive_response.get_emotional_statistics()
            sequencer_stats = self.motivational_sequencer.get_statistics()
            prioritization_stats = self.prioritization_core.get_statistics()
            
            # Calculate MIS-specific statistics
            total_ignitions = len(self.ignition_history)
            successful_ignitions = sum(1 for r in self.ignition_history if r.success)
            
            if total_ignitions > 0:
                success_rate = successful_ignitions / total_ignitions
                avg_duration = sum(r.duration for r in self.ignition_history) / total_ignitions
                avg_energy = sum(r.energy_consumed for r in self.ignition_history) / total_ignitions
            else:
                success_rate = 0.0
                avg_duration = 0.0
                avg_energy = 0.0
            
            # Analyze ignition types
            ignition_types = {}
            for result in self.ignition_history:
                action_type = result.trigger_type.value if hasattr(result.trigger_type, 'value') else str(result.trigger_type)
                ignition_types[action_type] = ignition_types.get(action_type, 0) + 1
            
            return {
                "motivational_ignition_system": {
                    "total_ignitions": total_ignitions,
                    "successful_ignitions": successful_ignitions,
                    "success_rate": round(success_rate, 3),
                    "average_duration": round(avg_duration, 3),
                    "average_energy_consumed": round(avg_energy, 2),
                    "ignition_type_distribution": ignition_types,
                    "auto_calibration_enabled": self.auto_calibration,
                    "active_callbacks": sum(1 for cb in [
                        self.creative_callback, self.emotional_callback, self.ritual_callback,
                        self.analytical_callback, self.contemplative_callback, self.transformational_callback
                    ] if cb is not None)
                },
                "integrated_systems": {
                    "threshold_motivator": threshold_stats,
                    "memory_imprinter": memory_stats,
                    "threshold_calibrator": calibration_stats,
                    "emotive_response": emotive_stats,
                    "motivational_sequencer": sequencer_stats,
                    "prioritization_core": prioritization_stats
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system statistics: {e}")
            return {"error": str(e)}

# ╔══════════════════════════════════════════════╗
# ║            🌐 GLOBAL INSTANCE                 ║
# ╚══════════════════════════════════════════════╗

_global_motivational_ignition_system = None

def get_global_motivational_ignition_system():
    """Get the global motivational ignition system."""
    global _global_motivational_ignition_system
    if _global_motivational_ignition_system is None:
        _global_motivational_ignition_system = MotivationalIgnitionSystem()
    return _global_motivational_ignition_system

# ╔══════════════════════════════════════════════╗
# ║           🎯 CONVENIENCE FUNCTIONS            ║
# ╚══════════════════════════════════════════════╗

def process_emotional_input(emotion: str, intensity: float, symbolic_content: str = "",
                           context: Optional[str] = None, familiarity: float = 0.5) -> IgnitionResult:
    """Convenience function to process emotional input through the complete system."""
    mis = get_global_motivational_ignition_system()
    return mis.process_emotional_input(emotion, intensity, symbolic_content, context, familiarity)

def bind_ignition_callbacks(**callbacks):
    """Convenience function to bind callbacks to the ignition system."""
    mis = get_global_motivational_ignition_system()
    mis.bind_callbacks(**callbacks)

def trigger_full_ignition_sequence() -> List[IgnitionResult]:
    """Convenience function to trigger the complete ignition sequence."""
    mis = get_global_motivational_ignition_system()
    return mis.trigger_ignition_sequence()

def get_motivational_system_statistics() -> Dict[str, Any]:
    """Convenience function to get comprehensive system statistics."""
    mis = get_global_motivational_ignition_system()
    return mis.get_system_statistics()
