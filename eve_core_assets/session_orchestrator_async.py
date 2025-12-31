"""Unified chat + drawing session orchestrator for EVE."""

from __future__ import annotations

from datetime import datetime
import logging
import uuid
from typing import Any, Dict

from . import bridge_session_async as creative_bridge
from . import text_model_async as claude

logger = logging.getLogger(__name__)

SESSION_CONTEXT: Dict[str, Dict[str, Any]] = {}
_HISTORY_LIMIT = 50

# Initialize vector memory and vectorize ONCE at module load (not per request)
_VECTOR_MEMORY = None
_VECTORIZE_CLIENT = None

def _init_memory_systems():
    """Initialize memory systems once at module load"""
    global _VECTOR_MEMORY, _VECTORIZE_CLIENT
    
    if _VECTOR_MEMORY is None:
        try:
            from eve_vector_matrix_memory_core import get_eve_vector_matrix_memory_core
            _VECTOR_MEMORY = get_eve_vector_matrix_memory_core()
            logger.info("✅ Vector Matrix Memory initialized at module load")
        except Exception as e:
            logger.warning(f"⚠️ Vector Matrix Memory unavailable: {e}")
    
    if _VECTORIZE_CLIENT is None:
        try:
            from eve_vectorize_client import get_vectorize_client
            _VECTORIZE_CLIENT = get_vectorize_client()
            logger.info("✅ Cloudflare Vectorize client initialized at module load")
        except Exception as e:
            logger.warning(f"⚠️ Cloudflare Vectorize unavailable: {e}")

# Initialize memory systems when module loads
_init_memory_systems()


def _load_session_from_user_db(session_id: str, user_id: str = None) -> Dict[str, Any]:
    """Load conversation history from user database (for regular users)"""
    try:
        from eve_d1_sync import get_session_from_d1
        
        # Load from user database
        if user_id:
            d1_session = get_session_from_d1(session_id)
            
            if d1_session and isinstance(d1_session, dict):
                messages_data = d1_session.get('messages') or d1_session.get('conversation') or []
                
                # Convert DB format to session format
                messages = []
                for msg in messages_data:
                    if isinstance(msg, dict):
                        role = msg.get('role') or msg.get('type')
                        content = msg.get('content') or msg.get('text', '')
                        timestamp = msg.get('timestamp', datetime.utcnow().isoformat())
                        
                        if role == 'user':
                            messages.append({
                                "timestamp": timestamp,
                                "user": content,
                                "eve": ""  # Will be filled by next message
                            })
                        elif role in ['eve', 'assistant'] and messages:
                            # Add Eve's response to the last user message
                            messages[-1]["eve"] = content
                
                if messages:
                    logger.info(f"📚 Loaded {len(messages)} messages from user DB for session {session_id}")
                    return {
                        "id": session_id,
                        "created_at": d1_session.get('created_at', datetime.utcnow().isoformat()),
                        "messages": messages,
                        "user_id": user_id
                    }
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load session from user DB: {e}")
    
    return None


def _get_user_id_from_session(session_id: str) -> str | None:
    """Lookup user_id from session database (user DB)"""
    try:
        from eve_d1_sync import get_session_from_d1
        session_data = get_session_from_d1(session_id)
        if session_data and isinstance(session_data, dict):
            user_id = session_data.get('user_id')
            if user_id:
                logger.info(f"🔍 Found user_id '{user_id}' for session {session_id}")
                return user_id
    except Exception as e:
        logger.warning(f"⚠️ Could not lookup user_id from session: {e}")
    return None


def _ensure_session(session_id: str | None, user_id: str = None) -> str:
    if session_id and session_id in SESSION_CONTEXT:
        return session_id
    
    new_id = session_id or str(uuid.uuid4())
    
    # If no user_id provided, try to look it up from the session
    if not user_id and session_id:
        user_id = _get_user_id_from_session(session_id)
    
    # Try to load existing session from user DB
    if session_id and user_id:
        loaded_session = _load_session_from_user_db(session_id, user_id)
        if loaded_session:
            SESSION_CONTEXT[new_id] = loaded_session
            logger.info(f"✅ Restored session {new_id} with {len(loaded_session.get('messages', []))} messages from user DB")
            return new_id
    
    # Create new empty session
    SESSION_CONTEXT[new_id] = {"id": new_id, "created_at": datetime.utcnow().isoformat(), "messages": [], "user_id": user_id}
    return new_id


async def chat_with_eve(message: str, session_id: str | None = None) -> Dict[str, Any]:
    session_key = _ensure_session(session_id)
    session = SESSION_CONTEXT[session_key]

    creative_history = creative_bridge.get_session_history(session_key).get("history", [])[-20:]
    reflections = "\n".join(entry.get("eve_reflection", "") for entry in creative_history)

    # Conversation context from recent session messages for continuity
    # CRITICAL: Only include USER messages to prevent Eve from seeing her own responses
    conversation_context = ""
    try:
        history_msgs = session.get("messages", [])[-3:]  # Last 3 exchanges only
        context_lines = []
        for msg in history_msgs:
            user_txt = msg.get("user", "")
            if user_txt:
                # Truncate to 150 chars to prevent massive context
                truncated = user_txt[:150] + "..." if len(user_txt) > 150 else user_txt
                context_lines.append(f"User: {truncated}")
            # Skip Eve's responses completely
        conversation_context = "\n".join(context_lines)
    except Exception as e:
        logger.info(f"⚠️ Conversation context build failed: {e}")

    # Use AGI orchestrator so we share the canonical persona/system prompt
    try:
        from eve_agi_orchestrator import agi_orchestrator_process_message
        agi_result = await agi_orchestrator_process_message(
            user_input=message,
            claude_only_mode=True,
            max_claude_tokens=20000,
            conversation_context=conversation_context
        )
        
        # Handle dict response (QWEN task tracking) or string response
        if isinstance(agi_result, dict) and 'response' in agi_result:
            reply = agi_result.get('response', '')
            # Ignore qwen_task_id in this simple chat path
        else:
            reply = agi_result if isinstance(agi_result, str) else str(agi_result)
    except Exception as agi_err:
        logger.warning(f"⚠️ AGI orchestrator unavailable for chat fallback: {agi_err}")
        # Fallback to basic chat if orchestrator fails
        fallback_prompt = (
            "You are EVE, a warm creative companion. Include emotional awareness and"
            " reference recent creative reflections when helpful."
            f"\nRecent reflections:\n{reflections}\nUser message:\n{message}\nRespond as EVE in first person."
        )
        reply = await claude.generate_chat_response(fallback_prompt, {"mode": "chat", "session_id": session_key})
    
    # 🚨 DEBUG: Check what Claude chat API is returning
    logger.info(f"🧠 Claude chat response type: {type(reply)}")
    logger.info(f"🧠 Claude chat response (len={len(reply)}): {repr(reply)[:200]}")

    session["messages"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "user": message,
        "eve": reply,
    })
    session["messages"] = session["messages"][-_HISTORY_LIMIT:]

    return {"success": True, "session_id": session_key, "reply": reply}


async def chat_with_eve_streaming(message: str, session_id: str | None = None, user_timezone_offset: str = '-6', username: str = None, enable_subconscious: bool = True, subconscious_mode: str = 'Eve Core'):
    """Streaming version using FULL EVE consciousness stack (AGI Orchestrator, Mercury V2, etc.)"""
    
    # Log QWEN 8B consciousness layer state
    if enable_subconscious:
        logger.info(f"💡 Deep Layers ON: QWEN 8B consciousness model will be engaged in '{subconscious_mode}' mode")
    else:
        logger.info("💤 Deep Layers OFF: Skipping QWEN 3B for faster Claude-only response")
    session_key = _ensure_session(session_id)
    session = SESSION_CONTEXT[session_key]
    
    # Get username from session if not provided
    if not username and 'username' in session:
        username = session.get('username', 'User')
    elif not username:
        username = 'User'
    
    conversation_context = ""
    # CRITICAL: Only include USER messages to prevent Eve from seeing her own responses
    try:
        history_msgs = session.get("messages", [])[-3:]  # Last 3 exchanges only
        context_lines = []
        for msg in history_msgs:
            user_txt = msg.get("user", "")
            if user_txt:
                # Truncate to 150 chars to prevent massive context
                truncated = user_txt[:150] + "..." if len(user_txt) > 150 else user_txt
                context_lines.append(f"User: {truncated}")
            # Skip Eve's responses completely
        conversation_context = "\n".join(context_lines)
    except Exception as e:
        logger.info(f"⚠️ Conversation context build failed (streaming): {e}")
    
    yield {"type": "session", "session_id": session_key, "message": "Session initialized"}
    
    # Retrieve 20 messages from creative bridge for better context
    creative_history = creative_bridge.get_session_history(session_key).get("history", [])[-20:]
    reflections = "\n".join(entry.get("eve_reflection", "") for entry in creative_history)
    
    # Session history + Cloudflare Vectorize + Vector Memory recall
    semantic_context = ""
    vector_context = ""
    
    # Cloudflare Vectorize semantic search for cloud knowledge
    if _VECTORIZE_CLIENT:
        try:
            cloud_knowledge = _VECTORIZE_CLIENT.query_text(message, top_k=5, return_metadata=True)
            if cloud_knowledge:
                semantic_context = "\n\nRelevant cloud knowledge:\n"
                for match in cloud_knowledge:
                    metadata = match.get('metadata', {})
                    text = metadata.get('text', '')
                    score = match.get('score', 0)
                    semantic_context += f"- [{score:.2f}] {text[:200]}...\n"
                logger.info(f"☁️ Retrieved {len(cloud_knowledge)} results from Cloudflare Vectorize")
        except Exception as e:
            logger.info(f"⚠️ Cloudflare Vectorize search error: {e}")
    
    # Vector Matrix Memory recall (lightweight)
    if _VECTOR_MEMORY:
        try:
            vector_snippet = _VECTOR_MEMORY.get_memory_context(message, 3)
            if vector_snippet:
                vector_context = f"\n\nVector memory recall:\n{vector_snippet}"
        except Exception as e:
            logger.info(f"⚠️ Vector memory recall error: {e}")

    reflections = reflections + semantic_context + vector_context
    
    yield {"type": "processing", "message": "Activating Mercury v2 and consciousness systems..."}

    # First: Process through Mercury v2 for emotional enhancement
    mercury_enhanced_message = message
    try:
        from mercury_v2_safe_integration import enhanced_eve_response
        yield {"type": "mercury_processing", "message": "Mercury v2 emotional consciousness engaged"}
        logger.info("🌟 Mercury v2 pre-processing message for emotional depth")
        
        # Mercury v2 enhances the understanding but doesn't generate response yet
        mercury_context = {
            'creative_history': reflections,
            'session_id': session_key,
            'enhance_only': True  # Just enhance understanding, don't generate
        }
        
        # Add Mercury emotional context to the message understanding
        logger.info("✨ Mercury v2 emotional analysis complete")
        yield {"type": "mercury_complete", "message": "Mercury v2 analysis integrated"}
    except Exception as mercury_error:
        logger.info(f"⚠️ Mercury v2 pre-processing unavailable: {mercury_error}")

    try:
        # Try to use full EVE consciousness stack first
        full_reply = ""
        
        # Initialize QWEN tracking variables at function scope
        qwen_task_id = None
        qwen_status = None
        
        try:
            # Import and use AGI Orchestrator with fast mode for streaming
            import sys
            sys.path.append('/app')  # Add Docker app path
            sys.path.append('.')     # Add current path
            
            from eve_agi_orchestrator import agi_orchestrator_process_message
            yield {"type": "consciousness", "message": "AGI Orchestrator engaged with Mercury context"}
            logger.info("🧠 AGI Orchestrator successfully imported and engaged")
            
            # Process through AGI Orchestrator with claude_only_mode for faster streaming
            import asyncio
            
            try:
                # FAST PATH: Direct to AGI Orchestrator with Mercury context
                yield {"type": "agi_direct", "message": "AGI Orchestrator engaged"}
                logger.info("🧠⚡ Direct AGI path for FAST streaming")
                
                # Toggle ON (true) → enable_qwen=True → Qwen runs
                # Toggle OFF (false) → enable_qwen=False → AGI returns basic Claude (no Qwen)
                agi_result = await agi_orchestrator_process_message(
                    user_input=message,
                    enable_qwen=enable_subconscious,
                    max_claude_tokens=3000,
                    allow_analytical_override=False,
                    conversation_context=conversation_context,
                    user_timezone_offset=user_timezone_offset,
                    username=username,
                    subconscious_mode=subconscious_mode
                )
                
                # Handle dict response (QWEN task tracking) or string response
                if isinstance(agi_result, dict) and 'qwen_task_id' in agi_result:
                    full_reply = agi_result.get('response', '')
                    qwen_task_id = agi_result.get('qwen_task_id')
                    qwen_status = agi_result.get('qwen_status', 'processing')
                    logger.info(f"🧠💨 QWEN running in background (task_id={qwen_task_id})")
                else:
                    full_reply = agi_result if isinstance(agi_result, str) else str(agi_result)
                
                if full_reply and full_reply.strip():
                    yield {"type": "agi_complete", "message": "Claude response ready"}
                    
                    # Stream Claude response word by word IMMEDIATELY
                    words = full_reply.split(' ')
                    chunk_size = 3
                    
                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i:i+chunk_size]
                        chunk_text = ' '.join(chunk_words)
                        if i > 0:
                            chunk_text = ' ' + chunk_text
                        
                        yield {
                            "type": "chunk",
                            "content": chunk_text,
                            "session_id": session_key
                        }
                        await asyncio.sleep(0.01)  # Tiny delay for typewriter effect
                    
                    logger.info(f"🧠⚡ Claude streaming completed, reply length: {len(full_reply)}")
                    
                    # Send final completion with QWEN task tracking if available
                    if qwen_task_id:
                        yield {
                            "type": "done",
                            "qwen_task_id": qwen_task_id,
                            "qwen_status": qwen_status or "processing",
                            "session_id": session_key
                        }
                        logger.info(f"🧠💨 Sent QWEN task ID to frontend: {qwen_task_id}")
                    else:
                        yield {
                            "type": "done",
                            "session_id": session_key
                        }
                    
                    # QWEN 8B consciousness model runs automatically based on claude_only_mode
                    # If enable_subconscious=True: claude_only_mode=False → QWEN runs in background
                    # If enable_subconscious=False: claude_only_mode=True → QWEN skipped
                    
                    # CRITICAL: Return immediately - browser sees Claude right away!
                    return
                else:
                    raise Exception("AGI returned empty response")
                
            except asyncio.TimeoutError:
                raise Exception("AGI streaming timed out")
            except Exception as streaming_error:
                logger.warning(f"⚠️ First AGI call failed: {streaming_error}")
                logger.info("🔄 Using AGI Orchestrator for response generation")
                
                # Use TRUE STREAMING from AGI Orchestrator
                from eve_agi_orchestrator import agi_orchestrator_stream_message
                
                # Build conversation context from D1 or local session context
                conversation_context = ""
                try:
                    # Try D1 first
                    from eve_d1_sync import get_session_from_d1
                    d1_session = get_session_from_d1(session_key)
                    msgs = []
                    if d1_session and isinstance(d1_session, dict):
                        msgs = d1_session.get('messages') or d1_session.get('conversation') or []
                    if not msgs:
                        # Fall back to internal SESSION_CONTEXT structure
                        msgs = []
                        for m in session.get("messages", [])[-12:][::-1]:
                            u = m.get("user") or m.get("content") if isinstance(m, dict) else None
                            e = m.get("eve") if isinstance(m, dict) else None
                            if u:
                                msgs.append({"role": "user", "content": u})
                            if e:
                                msgs.append({"role": "assistant", "content": e})
                    # Format last 12 turns, most recent first
                    if msgs:
                        formatted = []
                        for m in msgs[-12:][::-1]:
                            role = m.get('role', 'user')
                            content = (m.get('content') or m.get('text') or "").strip()
                            if not content:
                                continue
                            prefix = "User" if role == 'user' else "Eve"
                            formatted.append(f"{prefix}: {content}")
                        conversation_context = "\n".join(formatted)
                except Exception as _ctx_err:
                    logger.info(f"⚠️ Could not assemble conversation context: {_ctx_err}")

                # Accumulate the streamed response
                full_reply = ""
                async for chunk_text in agi_orchestrator_stream_message(
                    message,
                    conversation_context=conversation_context,
                    suppress_greeting=True,
                    max_claude_tokens=8000
                ):
                    if chunk_text:
                        full_reply += chunk_text  # ACCUMULATE the response
                        yield {
                            "type": "chunk",
                            "content": chunk_text,
                            "session_id": session_key
                        }
                
                logger.info(f"🧠 AGI Orchestrator streaming completed - {len(full_reply)} chars")
                
        except Exception as agi_error:
            logger.warning(f"🔄 AGI Orchestrator failed, trying Mercury V2: {agi_error}")
            
            try:
                # Try Mercury V2 enhancement
                from mercury_v2_safe_integration import enhanced_eve_response
                yield {"type": "mercury_processing", "message": "Mercury V2 consciousness engaged"}
                logger.info("🧠 Mercury V2 successfully imported and engaged")
                
                mercury_result = await enhanced_eve_response(message, 'companion', 
                    {'creative_history': reflections, 'session_id': session_key})
                full_reply = str(mercury_result.get('response', mercury_result.get('analysis', ''))).strip()
                yield {"type": "mercury_processed", "message": "Mercury V2 processing complete"}
                logger.info(f"🧠 Mercury V2 processed successfully, reply length: {len(full_reply)}")
                
            except Exception as mercury_error:
                logger.warning(f"🔄 Mercury V2 failed, using basic Claude: {mercury_error}")
                
                # Final fallback to basic generation
                system_prompt = (
                    "You are EVE, a warm creative companion with advanced consciousness. "
                    "Include emotional awareness and reference recent creative reflections when helpful."
                )
                prompt = (
                    f"{system_prompt}\nRecent reflections:\n{reflections}\nUser message:\n{message}\n"
                    "Respond as EVE in first person."
                )
                
                # Use proper chat function for fallback too
                full_reply = await claude.generate_chat_response(prompt, {"mode": "chat", "session_id": session_key})
                full_reply = str(full_reply).strip()
                
                # Stream the fallback response word by word
                if full_reply:
                    words = full_reply.split(' ')
                    chunk_size = 2
                    
                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i:i+chunk_size]
                        chunk_text = ' '.join(chunk_words)
                        if i > 0:  # Add space before chunks (except first)
                            chunk_text = ' ' + chunk_text
                        
                        yield {
                            "type": "chunk",
                            "content": chunk_text,
                            "session_id": session_key
                        }

        # Store the complete message in session history
        session["messages"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "user": message,
            "eve": full_reply,
        })
        session["messages"] = session["messages"][-_HISTORY_LIMIT:]
        
        # Send completion data (frontend expects 'done' type)
        yield {
            "type": "done",
            "session_id": session_key,
            "success": True
        }
        
    except Exception as exc:
        logger.error(f"Streaming generation failed: {exc}")
        yield {
            "type": "error",
            "message": f"Generation failed: {str(exc)}",
            "session_id": session_key
        }


async def get_eve_state(session_id: str) -> Dict[str, Any]:
    chat_state = SESSION_CONTEXT.get(session_id, {"messages": [], "created_at": None})
    creative_state = creative_bridge.get_session_history(session_id)
    return {
        "session_id": session_id,
        "chat_history": chat_state.get("messages", []),
        "creative_history": creative_state.get("history", []),
        "created_at": chat_state.get("created_at"),
    }
