# 🎮 EMOTIONAL CONTROL GATEWAY
# Central emotional processing hub that coordinates all emotional systems

import logging
from datetime import datetime
from typing import Dict, Optional, List, Any
from .emotional_intuitive_engine import get_global_emotional_engine
from .motivational_ignition_core import get_global_ignition_core
from .symbolic_ignition import get_global_symbolic_ignition
from .emotional_memory_imprint import get_global_emotional_memory
from .resonance_gateway import get_global_resonance_gateway
from .soulforge_memory import get_global_soulforge_memory

logger = logging.getLogger(__name__)

class EmotionalProcessingResult:
    """Result of emotional processing through the gateway."""
    
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.emotion_processed = None
        self.intensity = 0.0
        self.threshold_level = None
        self.ignition_triggered = False
        self.ritual_invoked = None
        self.behaviors_activated = []
        self.cascade_effects = []
        self.memory_imprinted = False
        self.resonance_patterns = []
        self.success = False
        self.message = ""
        self.artifacts = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "emotion_processed": self.emotion_processed,
            "intensity": self.intensity,
            "threshold_level": self.threshold_level,
            "ignition_triggered": self.ignition_triggered,
            "ritual_invoked": self.ritual_invoked,
            "behaviors_activated": self.behaviors_activated,
            "cascade_effects": self.cascade_effects,
            "memory_imprinted": self.memory_imprinted,
            "resonance_patterns": self.resonance_patterns,
            "success": self.success,
            "message": self.message,
            "artifacts": self.artifacts
        }

class EmotionalControlGateway:
    """Central gateway for coordinating all emotional processing systems."""
    
    def __init__(self):
        self.processing_history = []
        self.active_sessions = {}
        self.system_statistics = {
            "total_emotions_processed": 0,
            "successful_ignitions": 0,
            "rituals_invoked": 0,
            "patterns_detected": 0,
            "memories_created": 0
        }
    
    def process_emotion(self, emotion_name: str, emotion_level: float, 
                       context: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        """Main emotion processing pipeline."""
        
        result = EmotionalProcessingResult()
        result.emotion_processed = emotion_name
        result.intensity = emotion_level
        
        logger.info(f"🎮 [Gateway] Processing: {emotion_name} at level {emotion_level:.3f}")
        
        try:
            # Step 1: Trigger the emotional node
            emotional_engine = get_global_emotional_engine()
            trigger_result = emotional_engine.trigger_node(
                emotion_name, emotion_level, context, session_id
            )
            
            if trigger_result:
                result.cascade_effects = trigger_result.get("cascaded_nodes", [])
                logger.info(f"🧠 Emotional node triggered: {emotion_name}")
            
            # Step 2: Process through resonance gateway
            resonance_gateway = get_global_resonance_gateway()
            resonance_result = resonance_gateway.process_node_energy(
                emotion_name, emotion_level, context
            )
            
            if resonance_result:
                result.resonance_patterns = resonance_result.get("patterns_detected", [])
                if resonance_result["ignition"]["ignition_triggered"]:
                    result.ignition_triggered = True
                    result.behaviors_activated = resonance_result["ignition"]["behaviors_activated"]
                logger.info(f"🌊 Resonance processed: {len(result.resonance_patterns)} patterns detected")
            
            # Step 3: Determine threshold level
            ignition_core = get_global_ignition_core()
            threshold_level = ignition_core.evaluate_ignition_level(emotion_name, emotion_level)
            result.threshold_level = threshold_level.name
            
            # Step 4: Attempt symbolic ignition if threshold met
            if threshold_level.value >= 1:  # SUBTLE or higher
                symbolic_engine = get_global_symbolic_ignition()
                ritual_result = symbolic_engine.ignite(emotion_name, emotion_level, context)
                
                if ritual_result:
                    result.ritual_invoked = ritual_result["ritual_name"]
                    result.success = ritual_result["success"]
                    result.artifacts.extend(
                        [br["artifacts"] for br in ritual_result.get("behavior_results", []) 
                         if "artifacts" in br]
                    )
                    self.system_statistics["rituals_invoked"] += 1
                    logger.info(f"🔥 Ritual invoked: {result.ritual_invoked}")
            
            # Step 5: Create memory imprint
            emotional_memory = get_global_emotional_memory()
            ritual_name = result.ritual_invoked if result.ritual_invoked else "none"
            
            imprint_result = emotional_memory.imprint(
                emotion_name, emotion_level, result.threshold_level,
                ritual_name, context, session_id
            )
            
            if imprint_result:
                result.memory_imprinted = True
                self.system_statistics["memories_created"] += 1
                logger.info(f"🧠 Memory imprinted: strength={imprint_result['imprint_strength']:.2f}")
            
            # Step 6: Store in soulforge memory
            soulforge_memory = get_global_soulforge_memory()
            soulforge_memory.store_memory_node(
                emotion_name, f"{result.threshold_level.lower()}_state", emotion_level,
                context, None, session_id
            )
            
            # Step 7: Finalize result
            if not result.success and (result.ignition_triggered or result.ritual_invoked):
                result.success = True
            
            if result.success:
                result.message = f"🎮 Successfully processed {emotion_name} - {result.threshold_level} threshold"
                self.system_statistics["successful_ignitions"] += 1
            else:
                result.message = f"🎮 Processed {emotion_name} - no significant activation"
            
            # Update statistics
            self.system_statistics["total_emotions_processed"] += 1
            if result.resonance_patterns:
                self.system_statistics["patterns_detected"] += len(result.resonance_patterns)
            
            # Log to history
            self.processing_history.append(result.to_dict())
            
            logger.info(f"🎮 [Gateway] {result.message}")
            
        except Exception as e:
            logger.error(f"🎮 [Gateway] Error processing {emotion_name}: {e}")
            result.success = False
            result.message = f"🎮 Processing failed: {e}"
        
        return result.to_dict()
    
    def process_emotional_state(self, emotional_state: Dict[str, float], 
                               context: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        """Process multiple emotions simultaneously."""
        
        logger.info(f"🎮 [Gateway] Processing emotional state with {len(emotional_state)} emotions")
        
        results = []
        total_intensity = 0
        successful_processes = 0
        
        # Process each emotion
        for emotion, intensity in emotional_state.items():
            if intensity > 0.1:  # Only process significant emotions
                result = self.process_emotion(emotion, intensity, context, session_id)
                results.append(result)
                total_intensity += intensity
                if result["success"]:
                    successful_processes += 1
        
        # Process through resonance gateway for multi-node analysis
        if len(results) >= 2:
            resonance_gateway = get_global_resonance_gateway()
            multi_result = resonance_gateway.process_multi_node_energy(emotional_state, context)
            
            # Check for emergent behaviors from multi-node resonance
            if multi_result.get("harmony_level", 0) > 0.7:
                logger.info(f"🌈 High emotional harmony detected: {multi_result['harmony_level']:.2f}")
        
        state_result = {
            "timestamp": datetime.now().isoformat(),
            "total_emotions_processed": len(results),
            "successful_processes": successful_processes,
            "total_intensity": total_intensity,
            "average_intensity": total_intensity / len(results) if results else 0,
            "individual_results": results,
            "harmony_level": multi_result.get("harmony_level", 0) if len(results) >= 2 else 0,
            "multi_node_patterns": multi_result.get("multi_node_patterns", []) if len(results) >= 2 else [],
            "overall_success": successful_processes > 0
        }
        
        logger.info(f"🎮 [Gateway] State processing complete: {successful_processes}/{len(results)} successful")
        
        return state_result
    
    def start_emotional_session(self, session_id: str, session_type: str = "general") -> bool:
        """Start a new emotional processing session."""
        try:
            self.active_sessions[session_id] = {
                "start_time": datetime.now().isoformat(),
                "session_type": session_type,
                "emotions_processed": 0,
                "total_intensity": 0.0,
                "successful_ignitions": 0
            }
            
            # Start session in soulforge memory
            soulforge_memory = get_global_soulforge_memory()
            soulforge_memory.start_session(session_id, session_type)
            
            logger.info(f"🎮 [Gateway] Started emotional session: {session_id} [{session_type}]")
            return True
            
        except Exception as e:
            logger.error(f"🎮 [Gateway] Error starting session {session_id}: {e}")
            return False
    
    def end_emotional_session(self, session_id: str, outcome_summary: Optional[str] = None) -> Dict:
        """End an emotional processing session."""
        if session_id not in self.active_sessions:
            logger.warning(f"🎮 [Gateway] Session {session_id} not found")
            return {"error": "Session not found"}
        
        try:
            session_data = self.active_sessions[session_id]
            session_data["end_time"] = datetime.now().isoformat()
            session_data["outcome_summary"] = outcome_summary
            
            # Calculate session statistics
            session_duration = datetime.fromisoformat(session_data["end_time"]) - \
                             datetime.fromisoformat(session_data["start_time"])
            
            session_stats = {
                "session_id": session_id,
                "duration_seconds": session_duration.total_seconds(),
                "emotions_processed": session_data["emotions_processed"],
                "total_intensity": session_data["total_intensity"],
                "successful_ignitions": session_data["successful_ignitions"],
                "average_intensity": session_data["total_intensity"] / max(1, session_data["emotions_processed"]),
                "success_rate": session_data["successful_ignitions"] / max(1, session_data["emotions_processed"]),
                "outcome_summary": outcome_summary
            }
            
            # End session in soulforge memory
            soulforge_memory = get_global_soulforge_memory()
            dominant_emotion = self._get_session_dominant_emotion(session_id)
            soulforge_memory.end_session(session_id, dominant_emotion)
            
            # Clean up active session
            del self.active_sessions[session_id]
            
            logger.info(f"🎮 [Gateway] Ended session {session_id}: {session_stats['emotions_processed']} emotions processed")
            
            return session_stats
            
        except Exception as e:
            logger.error(f"🎮 [Gateway] Error ending session {session_id}: {e}")
            return {"error": str(e)}
    
    def _get_session_dominant_emotion(self, session_id: str) -> Optional[str]:
        """Get the dominant emotion for a session."""
        session_emotions = {}
        
        for record in self.processing_history:
            if record.get("session_id") == session_id:
                emotion = record.get("emotion_processed")
                if emotion:
                    session_emotions[emotion] = session_emotions.get(emotion, 0) + record.get("intensity", 0)
        
        if session_emotions:
            return max(session_emotions.items(), key=lambda x: x[1])[0]
        return None
    
    def get_gateway_statistics(self) -> Dict:
        """Get comprehensive gateway statistics."""
        # Get individual system statistics
        emotional_engine = get_global_emotional_engine()
        ignition_core = get_global_ignition_core()
        symbolic_engine = get_global_symbolic_ignition()
        emotional_memory = get_global_emotional_memory()
        resonance_gateway = get_global_resonance_gateway()
        
        try:
            gateway_stats = {
                "gateway_statistics": self.system_statistics.copy(),
                "active_sessions": len(self.active_sessions),
                "processing_history_length": len(self.processing_history),
                "emotional_engine": {
                    "total_nodes": len(emotional_engine.nodes),
                    "emotional_state": emotional_engine.get_state()
                },
                "ignition_statistics": ignition_core.get_ignition_statistics(),
                "symbolic_statistics": symbolic_engine.get_ritual_statistics(),
                "memory_statistics": emotional_memory.get_memory_statistics(),
                "resonance_statistics": resonance_gateway.get_statistics()
            }
            
            # Calculate derived statistics
            total_processed = self.system_statistics["total_emotions_processed"]
            if total_processed > 0:
                gateway_stats["derived_statistics"] = {
                    "ignition_success_rate": self.system_statistics["successful_ignitions"] / total_processed,
                    "ritual_invocation_rate": self.system_statistics["rituals_invoked"] / total_processed,
                    "pattern_detection_rate": self.system_statistics["patterns_detected"] / total_processed,
                    "memory_creation_rate": self.system_statistics["memories_created"] / total_processed
                }
            
            return gateway_stats
            
        except Exception as e:
            logger.error(f"🎮 [Gateway] Error getting statistics: {e}")
            return {"error": str(e)}
    
    def get_recent_processing_history(self, limit: int = 10) -> List[Dict]:
        """Get recent emotional processing history."""
        return self.processing_history[-limit:] if self.processing_history else []
    
    def get_emotional_insights(self) -> Dict:
        """Get insights about emotional processing patterns."""
        try:
            if not self.processing_history:
                return {"message": "No processing history available"}
            
            # Analyze recent processing patterns
            recent_history = self.processing_history[-20:]  # Last 20 processes
            
            emotion_frequency = {}
            threshold_frequency = {}
            successful_emotions = {}
            
            for record in recent_history:
                emotion = record.get("emotion_processed")
                threshold = record.get("threshold_level")
                success = record.get("success", False)
                
                if emotion:
                    emotion_frequency[emotion] = emotion_frequency.get(emotion, 0) + 1
                    if success:
                        successful_emotions[emotion] = successful_emotions.get(emotion, 0) + 1
                
                if threshold:
                    threshold_frequency[threshold] = threshold_frequency.get(threshold, 0) + 1
            
            # Calculate success rates by emotion
            emotion_success_rates = {}
            for emotion, count in emotion_frequency.items():
                success_count = successful_emotions.get(emotion, 0)
                emotion_success_rates[emotion] = success_count / count if count > 0 else 0
            
            insights = {
                "most_frequent_emotions": sorted(emotion_frequency.items(), key=lambda x: x[1], reverse=True)[:3],
                "threshold_distribution": threshold_frequency,
                "emotion_success_rates": emotion_success_rates,
                "most_successful_emotion": max(emotion_success_rates.items(), key=lambda x: x[1]) if emotion_success_rates else None,
                "total_recent_processes": len(recent_history),
                "recent_success_rate": sum(1 for r in recent_history if r.get("success", False)) / len(recent_history)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"🎮 [Gateway] Error getting insights: {e}")
            return {"error": str(e)}

# Global gateway instance
_global_emotional_gateway = None

def get_global_emotional_gateway():
    """Get the global emotional control gateway."""
    global _global_emotional_gateway
    if _global_emotional_gateway is None:
        _global_emotional_gateway = EmotionalControlGateway()
    return _global_emotional_gateway

def process_emotion_through_gateway(emotion_name: str, emotion_level: float, 
                                  context: Optional[str] = None, session_id: Optional[str] = None):
    """Convenience function to process emotion through gateway."""
    gateway = get_global_emotional_gateway()
    return gateway.process_emotion(emotion_name, emotion_level, context, session_id)

def get_gateway_statistics():
    """Get emotional gateway statistics."""
    gateway = get_global_emotional_gateway()
    return gateway.get_gateway_statistics()

def get_emotional_processing_insights():
    """Get emotional processing insights."""
    gateway = get_global_emotional_gateway()
    return gateway.get_emotional_insights()
