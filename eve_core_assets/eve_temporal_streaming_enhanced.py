"""
Enhanced streaming session orchestrator with temporal awareness integration
Wraps chat_with_eve_streaming to add temporal validation to streaming responses
"""

import logging
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Import the temporal streaming adapter
try:
    from eve_temporal_streaming_adapter import (
        stream_with_temporal_validation,
        record_streaming_event,
        set_session_emotional_context,
        TEMPORAL_AVAILABLE
    )
    logger.info("✅ Temporal streaming adapter loaded")
except ImportError as e:
    logger.warning(f"⚠️ Could not load temporal streaming adapter: {e}")
    TEMPORAL_AVAILABLE = False


async def chat_with_eve_streaming_temporal(
    message: str,
    session_id: str | None = None,
    enable_temporal: bool = True,
    event_type: Optional[str] = None,
    emotional_context: Optional[str] = None,
    subconscious_mode: str = 'Eve Core'
) -> AsyncIterator[Dict[str, Any]]:
    """
    Enhanced streaming with temporal awareness integration
    
    Features:
    - Real-time chunk streaming (zero buffering)
    - Temporal consistency validation (post-streaming)
    - Event tracking for temporal learning
    - Emotional context integration
    
    Args:
        message: User message
        session_id: Session ID (optional)
        enable_temporal: Enable temporal validation (default: True)
        event_type: Event type for temporal learning (e.g., 'job_application')
        emotional_context: Emotional context ('excitement', 'anxiety', etc.)
    
    Yields:
        Dict with streaming chunk data, validation results, or completion
    """
    
    # Import the base streaming orchestrator
    try:
        from eve_core.session_orchestrator_async import chat_with_eve_streaming
    except ImportError:
        logger.error("Could not import base streaming orchestrator")
        yield {
            'type': 'error',
            'message': 'Streaming orchestrator not available',
            'session_id': session_id
        }
        return
    
    # Record event for temporal learning
    event_id = ""
    if enable_temporal and event_type and TEMPORAL_AVAILABLE:
        event_id = record_streaming_event(session_id or "unknown", event_type, message)
    
    # Set emotional context if provided
    if enable_temporal and emotional_context and TEMPORAL_AVAILABLE:
        set_session_emotional_context(session_id or "unknown", emotional_context, 0.7)
    
    try:
        # Get the base streaming generator
        # Note: base orchestrator uses boolean enable_subconscious, not mode string yet
        # Convert mode to boolean: 'Off' = False, anything else = True
        enable_subconscious = (subconscious_mode != 'Off')
        base_generator = chat_with_eve_streaming(message, session_id, enable_subconscious=enable_subconscious, subconscious_mode=subconscious_mode)
        
        # Determine context for temporal validation
        user_context = f"Message: {message[:100]}"
        if event_type:
            user_context += f"\nEvent type: {event_type}"
        
        # METADATA FILTER - Define types to exclude
        metadata_only_types = {'session', 'processing', 'mercury_processing', 'mercury_complete', 'consciousness', 'agi_direct', 'agi_complete', 'status'}
        
        # Wrap with temporal validation if enabled
        if enable_temporal and TEMPORAL_AVAILABLE:
            async for chunk in stream_with_temporal_validation(
                base_generator,
                session_id or "unknown",
                context=user_context,
                chunk_size=0,  # Zero buffering - stream immediately
                enable_temporal=True
            ):
                # Filter metadata-only types from appearing in response
                # Only yield actual content and temporal validation results
                if isinstance(chunk, dict):
                    chunk_type = chunk.get('type', 'chunk')
                    
                    # Skip metadata-only types entirely
                    if chunk_type in metadata_only_types:
                        continue
                    
                    # Yield content and temporal validation chunks
                    yield chunk
                else:
                    # Non-dict chunks are content
                    yield chunk
        else:
            # Pass through without temporal validation, but still filter metadata
            async for chunk in base_generator:
                # Filter metadata-only types
                if isinstance(chunk, dict):
                    chunk_type = chunk.get('type', 'chunk')
                    
                    # Skip metadata-only types entirely
                    if chunk_type in metadata_only_types:
                        continue
                    
                    # Yield only content chunks
                    yield chunk
                else:
                    # Non-dict chunks are content
                    yield chunk
        
        # Record successful completion
        if event_id and TEMPORAL_AVAILABLE:
            try:
                from eve_temporal_reality_engine import get_temporal_reality_engine
                engine = get_temporal_reality_engine()
                engine.record_outcome(event_id, "streaming_completed")
            except:
                pass
    
    except Exception as e:
        logger.error(f"Enhanced streaming failed: {e}")
        yield {
            'type': 'error',
            'message': f'Streaming failed: {str(e)}',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }


# Event type detection helper
def detect_event_type(message: str) -> Optional[str]:
    """
    Auto-detect event type from user message for temporal learning
    """
    message_lower = message.lower()
    
    # Check for personal/relationship conversation first - these should NOT be classified as code events
    personal_indicators = [
        'good morning', 'how are you', 'love', 'beautiful', 'babe', 'my preferences',
        'updated my', 'preference', 'flirtatious', 'muse', 'feeling', 'database information'
    ]
    
    if any(indicator in message_lower for indicator in personal_indicators):
        return None  # This is personal conversation, not a code/tech event
    
    event_mappings = {
        'job_application_response': [
            'applied for', 'job application', 'job interview', 'interview coming up',
            'submitted application', 'applied for a job'
        ],
        'code_deployment': [
            'deploying to', 'deployed to', 'push to production', 'go live',
            'merged pull request', 'pushed to', 'pull request merged'
        ],
        'learning_mastery': [
            'learning python', 'learning javascript', 'learning typescript', 
            'studying for', 'trying to learn', 'mastering programming'
        ],
        'test_results': [
            'running tests', 'test results', 'tests passing', 'test suite',
            'unit tests', 'running unit tests'
        ],
        'code_review': [
            'code review', 'reviewing code', 'pull request review'
        ],
        'bug_fix': [
            'fixing bug', 'fixed bug', 'debug issue', 'fixing error',
            'bug report', 'debugging'
        ],
        'api_response': [
            'api call', 'api request', 'calling api',
            'api endpoint', 'http request'
        ]
    }
    
    # Only match if we have clear, specific technical context
    for event_type, keywords in event_mappings.items():
        if any(keyword in message_lower for keyword in keywords):
            return event_type
    
    return None


async def chat_with_eve_streaming_auto_temporal(
    message: str,
    session_id: str | None = None,
    enable_temporal: bool = True,
    subconscious_mode: str = 'Eve Core'
) -> AsyncIterator[Dict[str, Any]]:
    """
    Streaming with automatic event detection and temporal awareness
    """
    
    # Auto-detect event type
    event_type = detect_event_type(message) if enable_temporal else None
    
    if event_type:
        logger.info(f"🔍 Auto-detected event type: {event_type}")
    
    # Stream with temporal integration
    async for chunk in chat_with_eve_streaming_temporal(
        message,
        session_id,
        enable_temporal=enable_temporal,
        event_type=event_type,
        subconscious_mode=subconscious_mode
    ):
        yield chunk


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test():
        print("Testing enhanced streaming with temporal awareness...\n")
        
        async for chunk in chat_with_eve_streaming_auto_temporal(
            "I just applied for a job at TechCorp!",
            session_id="test_session",
            enable_temporal=True
        ):
            if chunk['type'] == 'chunk':
                print(chunk['content'], end='', flush=True)
            elif chunk['type'] == 'temporal_validation':
                print(f"\n[Temporal Status: {chunk.get('valid')}]")
            elif chunk['type'] == 'done':
                print("\n[Stream Complete]")
    
    asyncio.run(test())
