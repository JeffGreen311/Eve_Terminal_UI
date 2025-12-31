"""Async Florence 2 integration helpers for Draw with EVE.

This module keeps the Florence bridge self-contained and non-blocking so the
rest of the application can await the analysis without stalling the event loop.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from asyncio.subprocess import PIPE
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).resolve().parent.parent
_FLORENCE_NODE_SCRIPT = Path(
    os.getenv("FLORENCE_NODE_SCRIPT", _ROOT_DIR / "florence_node.js")
)
_FLORENCE_NODE_TIMEOUT = float(os.getenv("FLORENCE_NODE_TIMEOUT", "75"))
_USE_NODE_BRIDGE = _FLORENCE_NODE_SCRIPT.exists() and os.getenv(
    "FLORENCE_NODE_DISABLED", "false"
).lower() not in {"1", "true", "yes"}


FLORENCE_API_URL = os.getenv("FLORENCE_API_URL", "http://localhost:3001")
FLORENCE_API_KEY = os.getenv("FLORENCE_API_KEY")  # Not needed for local server
FLORENCE_MODEL = os.getenv("FLORENCE_MODEL", "Florence-2")

_DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=30.0)


def _build_headers() -> Dict[str, str]:
    # Local Florence server doesn't need API key authentication
    return {
        "Content-Type": "application/json",
    }


def _encode_image_payload(data: bytes, fmt: str) -> str:
    if fmt.lower() == "svg":
        prefix = "data:image/svg+xml;base64,"
    else:
        prefix = "data:image/png;base64,"
    encoded = base64.b64encode(data).decode("ascii")
    return prefix + encoded


async def _post_to_florence_local(image_data: str, task: str = "Detailed Caption") -> Dict[str, Any]:
    """Send image to local Florence server for analysis."""
    headers = _build_headers()
    
    # Prepare payload for local Florence server
    payload = {
        "image_data": image_data,
        "task": task
    }
    
    endpoint = f"{FLORENCE_API_URL}/analyze-base64"
    
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        try:
            # Check if local server is running
            try:
                health_response = await client.get(f"{FLORENCE_API_URL}/health", timeout=5.0)
                if health_response.status_code != 200:
                    logger.warning("Local Florence server not responding on port 3001")
                    return {"output_text": "Florence server not available"}
            except Exception as health_error:
                logger.warning(f"Cannot reach Florence server health endpoint: {health_error}")
                return {"output_text": "Florence server not reachable"}
                
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if result.get("success"):
                # Extract text from Florence response
                analysis = result.get("analysis", {})
                if isinstance(analysis, dict) and "text" in analysis:
                    text_data = analysis["text"]
                    if isinstance(text_data, dict):
                        # Get the detailed caption
                        caption = text_data.get("<DETAILED_CAPTION>", "")
                        return {"output_text": caption}
                return {"output_text": str(analysis)}
            else:
                logger.error("Florence analysis failed: %s", result.get("error", "Unknown error"))
                return {"output_text": "Analysis failed"}
                
        except httpx.HTTPError as exc:
            logger.error("Florence request failed: %s", exc)
            return {"output_text": "Connection failed"}
        except Exception as exc:
            logger.error("Unexpected Florence error: %s", exc)
            return {"output_text": "Analysis error"}

    return {"output_text": ""}


def _prepare_visual_payload(data: bytes, fmt: str) -> Tuple[bytes, str]:
    if fmt.lower() != "svg":
        return data, fmt
    try:
        import cairosvg  # type: ignore
    except ImportError:
        logger.debug("CairoSVG not installed; sending SVG bytes directly to Florence")
        return data, fmt
    try:
        raster = cairosvg.svg2png(bytestring=data)
        return raster, "png"
    except Exception as exc:
        logger.error("SVG rasterization failed; using original SVG bytes: %s", exc)
        return data, fmt


async def analyze_image_or_svg(data: bytes, fmt: str = "svg") -> str:
    """Return Florence's descriptive take on the supplied visual content."""
    payload_data, payload_fmt = _prepare_visual_payload(data, fmt)
    
    # Try local Florence server first
    try:
        image_data_uri = _encode_image_payload(payload_data, payload_fmt)
        result = await _post_to_florence_local(
            image_data_uri, 
            "Detailed Caption"  # Use valid Florence-2 task
        )
        caption = result.get("output_text", "")
        if caption and caption not in ["Connection failed", "Analysis failed", "Analysis error"]:
            return caption
    except Exception as exc:
        logger.warning("Local Florence server failed, trying Node bridge: %s", exc)
    
    # Fallback to Node bridge if local server fails
    if _USE_NODE_BRIDGE:
        node_payload = await _invoke_florence_node(payload_data, payload_fmt, "Detailed Caption")
        caption = _extract_caption_from_node(node_payload)
        if caption:
            return caption
    
    return "Image analysis temporarily unavailable"


async def analyze_image_florence(image_data: str, task: str = "Detailed Caption") -> str:
    """Analyze PNG image using Florence-2 via local HTTP server."""
    try:
        result = await _post_to_florence_local(image_data, task)
        return result.get("output_text", "Unable to analyze image")
    except Exception as e:
        logger.error(f"Florence PNG analysis failed: {e}")
        return "Analysis unavailable"


async def enhance_image_flux_konext(image_data: str, prompt: str) -> Dict[str, Any]:
    """Enhance image using FLUX Konext PRO via Replicate API.
    
    Args:
        image_data: Base64 data URI of the image
        prompt: Enhancement prompt
        
    Returns:
        Dict with enhanced image URL or error message
    """
    try:
        # TODO: Implement actual FLUX Konext PRO integration
        # This is a placeholder that returns the original image
        logger.warning("FLUX Konext PRO not implemented yet, returning placeholder")
        return {
            "success": True,
            "image_url": image_data,  # Return original for now
            "prompt": prompt,
            "note": "FLUX Konext PRO integration pending"
        }
    except Exception as e:
        logger.error(f"FLUX Konext PRO enhancement failed: {e}")
        return {"success": False, "error": str(e), "image_url": ""}


async def extract_structure(data: bytes, fmt: str = "svg") -> Dict[str, Any]:
    """Ask Florence for structural details that downstream models can reuse."""
    payload_data, payload_fmt = _prepare_visual_payload(data, fmt)
    
    # Try local Florence server first
    try:
        image_data_uri = _encode_image_payload(payload_data, payload_fmt)
        result = await _post_to_florence_local(
            image_data_uri,
            "More Detailed Caption"  # Use valid Florence-2 task for structure
        )
        structure_text = result.get("output_text", "")
        if structure_text and structure_text not in ["Connection failed", "Analysis failed", "Analysis error"]:
            # Try to parse as JSON, fallback to dict with text
            try:
                return json.loads(structure_text)
            except json.JSONDecodeError:
                return {"description": structure_text, "elements": [], "colors": [], "composition": "unknown"}
    except Exception as exc:
        logger.warning("Local Florence server failed for structure, trying Node bridge: %s", exc)
    
    # Fallback to Node bridge
    if _USE_NODE_BRIDGE:
        node_payload = await _invoke_florence_node(payload_data, payload_fmt, "JSON Structure")
        structure = _extract_structure_from_node(node_payload)
        if structure:
            return structure
    payload = {
        "model": FLORENCE_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "List the primary shapes, palette, and focal points in this"
                            " artwork. Keep the response JSON friendly."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": _encode_image_payload(payload_data, payload_fmt),
                    },
                ],
            }
        ],
    }
    result = await _post_to_florence(payload)
    raw_text = result.get("output_text", "")
    if not raw_text:
        return {"structure": ""}

    try:
        parsed: Optional[Any] = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
        return {"structure": parsed}
    except json.JSONDecodeError:
        return {"structure": raw_text}


async def _invoke_florence_node(data: bytes, fmt: str, task: str) -> Optional[Dict[str, Any]]:
    if not _FLORENCE_NODE_SCRIPT.exists():
        return None

    suffix = ".svg" if fmt.lower() == "svg" else ".png"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp_file.write(data)
        tmp_file.flush()
        tmp_file.close()

        process = await asyncio.create_subprocess_exec(
            "node",
            str(_FLORENCE_NODE_SCRIPT),
            tmp_file.name,
            task,
            stdout=PIPE,
            stderr=PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), _FLORENCE_NODE_TIMEOUT
            )
        except asyncio.TimeoutError:
            process.kill()
            logger.error("Florence node script timed out")
            return None

        if stderr:
            logger.debug("Florence node stderr: %s", stderr.decode("utf-8", "ignore"))

        payload = _extract_json_from_output(stdout.decode("utf-8", "ignore"))
        return payload
    except FileNotFoundError:
        logger.error("Node binary not found; cannot execute Florence bridge")
        return None
    except Exception as exc:
        logger.error("Florence node invocation failed: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass


def _extract_json_from_output(output: str) -> Optional[Dict[str, Any]]:
    if not output:
        return None
    matches = re.findall(r"\{[\s\S]*\}", output)
    if not matches:
        return None
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_caption_from_node(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not payload:
        return None
    text_block = payload.get("text") if isinstance(payload, dict) else None
    if isinstance(text_block, dict):
        for value in text_block.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(text_block, str) and text_block.strip():
        return text_block.strip()
    return json.dumps(payload)


def _extract_structure_from_node(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    if isinstance(payload, dict):
        if "structure" in payload and isinstance(payload["structure"], dict):
            return payload
        text_block = payload.get("text")
        if isinstance(text_block, dict):
            combined = {"structure": text_block}
        elif isinstance(text_block, str):
            combined = {"structure": text_block}
        else:
            combined = {"structure": payload}
        return combined
    return {"structure": payload}
