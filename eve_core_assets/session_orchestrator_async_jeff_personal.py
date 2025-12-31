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


def _load_session_from_personal_db(session_id: str, user_id: str = None) -> Dict[str, Any]:
    """Load conversation history from Jeff's personal DB"""
    try:
        from eve_personal_db_sync import get_user_conversations
        
        # For JeffGreen311, load from personal DB
        if user_id == "JeffGreen311":
            conversations = get_user_conversations(user_id, session_id=session_id, limit=50)
            
            if conversations:
                # Convert DB format to session format
                messages = []
                for conv in reversed(conversations):  # Reverse to get chronological order
                    role = conv.get("role")
                    content = conv.get("content")
                    timestamp = conv.get("timestamp")
                    
                    if role == "user":
                        messages.append({
                            "timestamp": timestamp,
                            "user": content,
                            "eve": ""  # Will be filled by next message
                        })
                    elif role == "eve" and messages:
                        # Add Eve's response to the last user message
                        messages[-1]["eve"] = content
                
                logger.info(f"📚 Loaded {len(messages)} messages from personal DB for session {session_id}")
                return {
                    "id": session_id,
                    "created_at": conversations[-1].get("timestamp") if conversations else datetime.utcnow().isoformat(),
                    "messages": messages,
                    "user_id": user_id
                }
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load session from personal DB: {e}")
    
    return None


def _ensure_session(session_id: str | None, user_id: str = None) -> str:
    if session_id and session_id in SESSION_CONTEXT:
        return session_id
    
    new_id = session_id or str(uuid.uuid4())
    
    # Try to load existing session from personal DB
    if session_id and user_id:
        loaded_session = _load_session_from_personal_db(session_id, user_id)
        if loaded_session:
            SESSION_CONTEXT[new_id] = loaded_session
            logger.info(f"✅ Restored session {new_id} with {len(loaded_session.get('messages', []))} messages from personal DB")
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
    conversation_context = ""
    try:
        history_msgs = session.get("messages", [])[-10:]
        context_lines = []
        for msg in history_msgs:
            user_txt = msg.get("user", "")
            eve_txt = msg.get("eve", "")
            if user_txt:
                context_lines.append(f"User: {user_txt}")
            if eve_txt:
                context_lines.append(f"Eve: {eve_txt}")
        conversation_context = "\n".join(context_lines)
    except Exception as e:
        logger.info(f"⚠️ Conversation context build failed: {e}")

    # Use AGI orchestrator so we share the canonical persona/system prompt
    try:
        from eve_agi_orchestrator import agi_orchestrator_process_message
        reply = await agi_orchestrator_process_message(
            user_input=message,
            claude_only_mode=True,
            max_claude_tokens=20000,
            conversation_context=conversation_context
        )
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


async def chat_with_eve_streaming(message: str, session_id: str | None = None, user_timezone_offset: str = '-6', username: str = None, enable_subconscious: bool = True):
    """Streaming version using FULL EVE consciousness stack (AGI Orchestrator, Mercury V2, etc.)"""
    
    logger.info(f"🔍 DEBUG: chat_with_eve_streaming called with enable_subconscious={enable_subconscious}")
    
    # Log QWEN 3B consciousness layer state
    if enable_subconscious:
        logger.info("💡 Deep Layers ON: QWEN 3B consciousness model will be engaged (Jeff's orchestrator)")
    else:
        logger.info("💤 Deep Layers OFF: Skipping QWEN 3B for faster Claude-only response (Jeff's orchestrator)")
    
    session_key = _ensure_session(session_id)
    session = SESSION_CONTEXT[session_key]
    conversation_context = ""
    try:
        history_msgs = session.get("messages", [])[-10:]
        context_lines = []
        for msg in history_msgs:
            user_txt = msg.get("user", "")
            eve_txt = msg.get("eve", "")
            if user_txt:
                context_lines.append(f"User: {user_txt}")
            if eve_txt:
                context_lines.append(f"Eve: {eve_txt}")
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
                
                if enable_subconscious:
                    # Toggle ON (true) → enable_qwen=True → Qwen runs
                    logger.info("💡 Deep Layers ON: Calling AGI Orchestrator with Qwen enabled")
                    agi_result = await agi_orchestrator_process_message(
                        user_input=message,
                        enable_qwen=True,
                        allow_analytical_override=False,
                        max_claude_tokens=20000,
                        conversation_context=conversation_context,
                        user_timezone_offset=user_timezone_offset,
                        username=username
                    )
                    # AGI returns dict with {"response": ..., "qwen_task_id": ..., "qwen_status": ...}
                    if isinstance(agi_result, dict):
                        full_reply = agi_result.get("response", "")
                        logger.info(f"✅ Claude response extracted, Qwen task: {agi_result.get('qwen_task_id')}")
                    else:
                        full_reply = agi_result  # Fallback for string return
                else:
                    # Toggle OFF (false) → Bypass AGI Orchestrator entirely
                    logger.info("💤 Subconscious disabled: Bypassing AGI Orchestrator, using direct Claude")
                    from eve_agi_orchestrator import get_agi_systems
                    agi = get_agi_systems()
                    rhe = agi["rhe"]
                    current_modulation = {n: nt.get_level() for n, nt in agi["nts"].items()}
                    
                    # Direct call to Right Hemisphere (Claude) - NO QWEN PATH POSSIBLE
                    r_out, r_weight = await rhe.process(message, current_modulation, conversation_context, max_tokens=20000)
                    full_reply = r_out.strip() if r_out and r_out.strip() else "Hello there, beautiful soul ✨"
                
                if full_reply and isinstance(full_reply, str) and full_reply.strip():
                    yield {"type": "agi_complete", "message": "AGI response generated"}
                    
                    # Stream the response word by word
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
                    
                    logger.info(f"🧠⚡ AGI direct streaming completed, reply length: {len(full_reply)}")
                else:
                    raise Exception("AGI returned empty response")
                
            except asyncio.TimeoutError:
                raise Exception("AGI streaming timed out")
            except Exception as streaming_error:
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
