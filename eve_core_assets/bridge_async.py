"""Lightweight Florence + Claude bridge without long running session state."""

from __future__ import annotations

import logging
from typing import Any, Dict

from . import model_hub_async as florence
from . import text_model_async as claude

logger = logging.getLogger(__name__)


async def process_drawing(svg: str, prompt: str = "") -> Dict[str, Any]:
    """Return analysis and creative response for a single drawing request."""
    data = svg.encode("utf-8")
    florence_view = await florence.analyze_image_or_svg(data, fmt="svg")
    structure = await florence.extract_structure(data, fmt="svg")

    creative_prompt = (
        "Blend this analysis with the user guidance and respond with refined SVG"
        " markup and a short reflection."
    )
    prompt_block = f"Florence analysis: {florence_view}\nUser prompt: {prompt}\n"

    creative = await claude.generate_enhancement(prompt_block, structure)

    return {
        "success": True,
        "analysis": florence_view,
        "enhanced_svg": creative.get("svg", ""),
        "eve_reflection": creative.get("analysis", ""),
    }
