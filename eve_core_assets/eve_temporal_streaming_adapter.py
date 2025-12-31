"""
Eve Temporal Streaming Adapter
Integrates Temporal Reality Engine with real-time streaming responses
Provides optimized chunking with no buffering for low-latency temporal awareness
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Import temporal engine
try:
    from eve_temporal_reality_engine import get_temporal_reality_engine
    TEMPORAL_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Temporal Reality Engine not available")
    TEMPORAL_AVAILABLE = False


class TemporalStreamingBuffer:
    """
    Smart buffer for streaming responses with temporal validation
    Validates response chunks as they arrive for temporal consistency
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.buffer = []
        self.full_response = ""
        self.temporal_engine = get_temporal_reality_engine() if TEMPORAL_AVAILABLE else None
        self.validation_complete = False
        self.temporal_issues = []
        self.event_context = None
        
    def add_chunk(self, chunk: str):
        """Add chunk to buffer"""
        self.buffer.append(chunk)
        # Handle both string and dict chunks (in case SSE sends JSON)
        if isinstance(chunk, dict):
            chunk_text = chunk.get('content', str(chunk))
        else:
            chunk_text = str(chunk)
        self.full_response += chunk_text
    
    def validate_temporal_consistency(self, context: str = "") -> Dict[str, Any]:
        """
        Validate complete response for temporal consistency
        Called AFTER streaming completes (to validate full response)
        """
        if not TEMPORAL_AVAILABLE or self.validation_complete:
            return {'valid': True, 'issues': []}
        
        self.validation_complete = True
        
        try:
            result = self.temporal_engine.check_temporal_validity(
                self.full_response,
                context
            )
            
            if not result['valid']:
                logger.warning(f"⏰ Temporal violation detected: {result.get('violation')}")
                self.temporal_issues.append(result)
                return result
            
            return {
                'valid': True,
                'nuance': result.get('nuance'),
                'reasoning': result.get('reasoning', 'Temporally consistent')
            }
        except Exception as e:
            logger.error(f"Temporal validation error: {e}")
            return {'valid': True, 'error': str(e)}
    
    def get_corrected_response(self) -> str:
        """
        Get response with temporal corrections if needed
        """
        if not self.temporal_issues:
            return self.full_response
        
        # If temporal issues exist, suggest alternative
        if self.temporal_engine:
            last_issue = self.temporal_issues[-1]
            event_type = last_issue.get('event_type', 'unknown')
            elapsed = last_issue.get('elapsed_time', '0s')
            
            suggestion = self.temporal_engine.suggest_realistic_response(event_type, 0)
            
            logger.info(f"💭 Temporal correction suggested for {event_type}")
            return suggestion
        
        return self.full_response


async def stream_with_temporal_validation(
    async_response_generator: AsyncIterator[str],
    session_id: str,
    context: str = "",
    chunk_size: int = 0,
    enable_temporal: bool = True
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream response chunks with real-time temporal awareness
    
    Features:
    - Zero buffering: yields chunks immediately
    - Temporal validation: checks full response after streaming
    - Emotional context: integrates with emotional state
    - Error resilience: continues streaming even if validation fails
    
    Args:
        async_response_generator: Source of streaming text chunks
        session_id: Session identifier for logging
        context: User context for temporal validation
        chunk_size: Target chunk size in chars (0 = no aggregation, stream raw)
        enable_temporal: Enable temporal validation checks
    
    Yields:
        Dict with type, content, chunk_num, session_id
    """
    
    temporal_buffer = TemporalStreamingBuffer(session_id) if enable_temporal else None
    chunk_num = 0
    current_chunk = ""
    chunk_size = max(0, chunk_size)  # Ensure non-negative
    
    try:
        # Phase 1: Stream chunks in real-time with zero buffering
        async for text_chunk in async_response_generator:
            if not text_chunk:
                continue
            
            # Add to temporal buffer if enabled
            if temporal_buffer:
                temporal_buffer.add_chunk(text_chunk)
            
            # Pass through ALL chunks as-is (consciousness events, content chunks, etc.)
            yield text_chunk
        
        # Phase 2: Temporal validation (after streaming completes)
        if temporal_buffer and enable_temporal:
            validation_result = temporal_buffer.validate_temporal_consistency(context)
            
            yield {
                'type': 'temporal_validation',
                'valid': validation_result.get('valid', True),
                'nuance': validation_result.get('nuance'),
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # If issues found, suggest correction
            if not validation_result.get('valid'):
                corrected = temporal_buffer.get_corrected_response()
                yield {
                    'type': 'temporal_correction',
                    'reason': validation_result.get('violation', 'Temporal issue detected'),
                    'original_length': len(temporal_buffer.full_response),
                    'suggestion': corrected,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                }
        
        # Signal completion
        yield {
            'type': 'done',
            'chunks_delivered': chunk_num,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Streaming error in {session_id}: {e}")
        yield {
            'type': 'error',
            'message': str(e),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }


async def validate_response_chunk_early(
    chunk: str,
    event_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Early validation of chunks for obvious temporal issues
    (Can be called during streaming for real-time feedback)
    """
    if not TEMPORAL_AVAILABLE:
        return {'valid': True}
    
    try:
        engine = get_temporal_reality_engine()
        
        # Quick check for obvious assumptive markers
        nuance_check = engine.nuance_detector.classify_response(chunk)
        
        if nuance_check['requires_temporal_check'] and event_type:
            # Potential issue - full validation needed after streaming
            return {
                'potential_issue': True,
                'nuance': nuance_check['classification'],
                'event_type': event_type
            }
        
        return {'valid': True}
    except Exception as e:
        logger.debug(f"Early validation error: {e}")
        return {'valid': True}


def record_streaming_event(session_id: str, event_type: str, description: str) -> str:
    """
    Record event for temporal learning
    Called before streaming begins
    """
    if not TEMPORAL_AVAILABLE:
        return ""
    
    try:
        engine = get_temporal_reality_engine()
        event_id = engine.record_event(event_type, description)
        logger.info(f"⏰ Event recorded for temporal learning: {event_type}")
        return event_id
    except Exception as e:
        logger.warning(f"Could not record streaming event: {e}")
        return ""


def set_session_emotional_context(session_id: str, emotion: str, strength: float = 0.5):
    """
    Set emotional context for temporal bias
    """
    if not TEMPORAL_AVAILABLE:
        return
    
    try:
        engine = get_temporal_reality_engine()
        engine.set_emotional_state(emotion, strength)
        logger.info(f"💫 Session {session_id} emotional context: {emotion} (strength: {strength})")
    except Exception as e:
        logger.warning(f"Could not set emotional context: {e}")


# Integration helper for eve_api
async def get_temporal_streaming_chunks(
    message: str,
    session_orchestrator_generator,
    session_id: str,
    user_context: str = "",
    event_type: Optional[str] = None
) -> AsyncIterator[Dict[str, Any]]:
    """
    Complete temporal + streaming integration
    
    Usage in eve_api:
        async def chat_stream_with_temporal():
            async for chunk in get_temporal_streaming_chunks(
                message=user_input,
                session_orchestrator_generator=stream_generator,
                session_id=session_id,
                user_context=conversation_context,
                event_type='job_application'  # optional
            ):
                yield chunk
    """
    
    # Record event for learning
    event_id = ""
    if event_type:
        event_id = record_streaming_event(session_id, event_type, message)
    
    # Stream with temporal validation
    async for chunk in stream_with_temporal_validation(
        session_orchestrator_generator,
        session_id,
        context=user_context,
        chunk_size=0,  # Zero buffering for optimal latency
        enable_temporal=TEMPORAL_AVAILABLE
    ):
        yield chunk
    
    # Record outcome if event was tracked
    if event_id and TEMPORAL_AVAILABLE:
        try:
            engine = get_temporal_reality_engine()
            engine.record_outcome(event_id, "streaming_completed")
        except:
            pass