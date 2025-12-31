"""Session aware bridge between Florence analysis and EVE's creative voice."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import uuid
from typing import Any, Dict, List

from . import bridge_async
from . import model_hub_async as florence
from . import text_model_async as claude

logger = logging.getLogger(__name__)

SESSION_MEMORY: Dict[str, Dict[str, Any]] = {}
_HISTORY_LIMIT = 10


def _ensure_session(session_id: str | None) -> str:
    if session_id and session_id in SESSION_MEMORY:
        return session_id
    new_id = session_id or str(uuid.uuid4())
    SESSION_MEMORY[new_id] = {"id": new_id, "history": [], "created_at": datetime.utcnow().isoformat()}
    return new_id


async def process_drawing(svg: str, prompt: str = "", session_id: str | None = None) -> Dict[str, Any]:
    """Process a drawing while tracking conversational state."""
    session_key = _ensure_session(session_id)
    data = svg.encode("utf-8")
    florence_view = await florence.analyze_image_or_svg(data, fmt="svg")
    structure = await florence.extract_structure(data, fmt="svg")

    history = SESSION_MEMORY[session_key]["history"][-3:]
    history_text = "\n".join(h.get("eve_reflection", "") for h in history)

    prompt_block = (
        "Interpret this piece using your voice as EVE, building on prior reflections"
        " when available."
    )

    creative_context = {
        "session_id": session_key,
        "florence_analysis": florence_view,
        "structure": structure,
        "recent_reflections": history_text,
    }

    creative = await claude.generate_enhancement(prompt_block, creative_context)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "florence_analysis": florence_view,
        "eve_reflection": creative.get("analysis", ""),
        "enhanced_svg": creative.get("svg", ""),
        "user_prompt": prompt,
    }
    SESSION_MEMORY[session_key]["history"].append(record)
    SESSION_MEMORY[session_key]["history"] = SESSION_MEMORY[session_key]["history"][-_HISTORY_LIMIT:]

    return {
        "success": True,
        "session_id": session_key,
        "florence_analysis": florence_view,
        "eve_reflection": record["eve_reflection"],
        "enhanced_svg": record["enhanced_svg"],
        "memory_length": len(SESSION_MEMORY[session_key]["history"]),
    }


async def process_image(image_file, prompt: str = "", session_id: str | None = None) -> Dict[str, Any]:
    """Process an uploaded image file while tracking conversational state."""
    session_key = _ensure_session(session_id)
    
    # Read image file data
    image_data = image_file.read()
    image_file.seek(0)  # Reset file pointer
    
    # Analyze with Florence-2
    florence_view = await florence.analyze_image_or_svg(image_data, fmt="image")
    structure = await florence.extract_structure(image_data, fmt="image")

    history = SESSION_MEMORY[session_key]["history"][-3:]
    history_text = "\n".join(h.get("eve_reflection", "") for h in history)

    prompt_block = (
        "Analyze this uploaded image using your voice as EVE, building on prior reflections"
        " when available."
    )

    creative_context = {
        "session_id": session_key,
        "florence_analysis": florence_view,
        "structure": structure,
        "recent_reflections": history_text,
    }

    creative = await claude.generate_enhancement(prompt_block, creative_context)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "florence_analysis": florence_view,
        "eve_reflection": creative.get("analysis", ""),
        "enhanced_svg": creative.get("svg", ""),
        "user_prompt": prompt,
    }
    SESSION_MEMORY[session_key]["history"].append(record)
    SESSION_MEMORY[session_key]["history"] = SESSION_MEMORY[session_key]["history"][-_HISTORY_LIMIT:]

    return {
        "success": True,
        "session_id": session_key,
        "florence_analysis": florence_view,
        "eve_reflection": record["eve_reflection"],
        "enhanced_svg": record["enhanced_svg"],
        "memory_length": len(SESSION_MEMORY[session_key]["history"]),
    }


def get_session_history(session_id: str) -> Dict[str, Any]:
    return SESSION_MEMORY.get(session_id, {"id": session_id, "history": []})
