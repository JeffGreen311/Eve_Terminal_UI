"""Async drawing helpers that bridge Florence analysis with EVE's creative voice.

Workflow: SVG -> PNG conversion -> Florence-2 analysis -> FLUX Konext PRO enhancement
"""

from __future__ import annotations

from typing import Any, Dict
import base64
import io
from PIL import Image

# CairoSVG has system dependencies that may not be available
# For now, we'll use SVG directly since Florence-2 can handle SVG format
CAIROSVG_AVAILABLE = False
print("ℹ️ Using SVG direct analysis - Florence-2 supports SVG format natively")

from . import model_hub_async as florence
from . import text_model_async as claude


async def convert_svg_to_png(svg_content: str) -> str:
    """Convert SVG to PNG base64 data URI for Florence-2 analysis"""
    try:
        # Try cairosvg first (if available)
        if CAIROSVG_AVAILABLE:
            try:
                png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
                b64_data = base64.b64encode(png_data).decode('utf-8')
                return f"data:image/png;base64,{b64_data}"
            except Exception as e:
                print(f"cairosvg conversion failed: {e}")
        
        # Fallback: Return SVG as data URI (Florence-2 can handle SVG directly)
        print("Using SVG fallback - Florence-2 can analyze SVG directly")
        b64_svg = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
        return f"data:image/svg+xml;base64,{b64_svg}"
    except Exception as e:
        print(f"SVG conversion failed: {e}")
        return None


async def analyze_png(png_data: str) -> Dict[str, Any]:
    """Analyze PNG image using Florence-2"""
    try:
        analysis = await florence.analyze_image_florence(png_data, task="Detailed Caption")
        return {"success": True, "analysis": analysis}
    except Exception as e:
        return {"success": False, "error": str(e), "analysis": "Unable to analyze image"}


async def analyze_image_file(image_data: bytes) -> Dict[str, Any]:
    """Analyze uploaded image file directly using Florence-2"""
    try:
        # Convert image bytes to base64 data URI
        b64_data = base64.b64encode(image_data).decode('utf-8')
        
        # Detect image format (default to PNG if unknown)
        image_format = "png"
        if image_data.startswith(b'\xFF\xD8\xFF'):
            image_format = "jpeg"
        elif image_data.startswith(b'GIF'):
            image_format = "gif"
        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:20]:
            image_format = "webp"
            
        data_uri = f"data:image/{image_format};base64,{b64_data}"
        
        # Analyze with Florence-2
        analysis = await florence.analyze_image_florence(data_uri, task="Detailed Caption")
        return {"success": True, "analysis": analysis}
    except Exception as e:
        return {"success": False, "error": str(e), "analysis": "Unable to analyze image"}


async def analyze_svg(svg: str) -> Dict[str, Any]:
    """Analyze SVG by converting to PNG first, then using Florence-2"""
    png_data = await convert_svg_to_png(svg)
    if not png_data:
        return {"success": False, "error": "SVG to PNG conversion failed"}
    
    return await analyze_png(png_data)


async def enhance_svg(svg: str, prompt: str = "") -> Dict[str, Any]:
    """Enhance SVG using PNG->Florence analysis->FLUX Konext PRO pipeline"""
    try:
        # Convert SVG to PNG for analysis
        png_data = await convert_svg_to_png(svg)
        if not png_data:
            return {"success": False, "error": "Failed to convert SVG to PNG"}
        
        # Analyze PNG with Florence-2
        analysis_result = await analyze_png(png_data)
        analysis_text = analysis_result.get('analysis', 'artistic drawing')
        
        # Create enhancement prompt
        enhancement_prompt = f"Enhance this {analysis_text}. {prompt}" if prompt else f"Enhance this {analysis_text} with more detail and artistic flair"
        
        # Use FLUX Konext PRO for enhancement (works with PNG)
        enhanced_result = await florence.enhance_image_flux_konext(
            image_data=png_data,
            prompt=enhancement_prompt
        )
        
        return {
            "success": True,
            "analysis": analysis_text,
            "enhanced_image_url": enhanced_result.get('image_url', ''),
            "original_svg": svg,
            "enhancement_prompt": enhancement_prompt
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def enhance_image_file(image_data: bytes, prompt: str = "") -> Dict[str, Any]:
    """Enhance uploaded image file using Florence-2 analysis and FLUX Konext PRO"""
    try:
        # Convert image bytes to base64 data URI
        b64_data = base64.b64encode(image_data).decode('utf-8')
        
        # Detect image format
        image_format = "png"
        if image_data.startswith(b'\xFF\xD8\xFF'):
            image_format = "jpeg" 
        elif image_data.startswith(b'GIF'):
            image_format = "gif"
        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:20]:
            image_format = "webp"
            
        data_uri = f"data:image/{image_format};base64,{b64_data}"
        
        # Analyze with Florence-2
        analysis_result = await analyze_image_file(image_data)
        analysis_text = analysis_result.get('analysis', 'uploaded image')
        
        # Create enhancement prompt
        enhancement_prompt = f"Enhance this {analysis_text}. {prompt}" if prompt else f"Enhance this {analysis_text} with more detail and artistic flair"
        
        # Use FLUX Konext PRO for enhancement
        enhanced_result = await florence.enhance_image_flux_konext(
            image_data=data_uri,
            prompt=enhancement_prompt
        )
        
        return {
            "success": True,
            "analysis": analysis_text,
            "enhanced_image_url": enhanced_result.get('image_url', ''),
            "enhancement_prompt": enhancement_prompt
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def complete_image_file(image_data: bytes, intent: str = "") -> Dict[str, Any]:
    """Complete uploaded image file using analysis and creative completion"""
    try:
        # Analyze the uploaded image
        analysis_result = await analyze_image_file(image_data)
        analysis = analysis_result.get('analysis', 'uploaded image')
        
        # Convert to data URI for processing
        b64_data = base64.b64encode(image_data).decode('utf-8')
        image_format = "png"
        if image_data.startswith(b'\xFF\xD8\xFF'):
            image_format = "jpeg"
        data_uri = f"data:image/{image_format};base64,{b64_data}"
        
        # Generate creative completion using EVE's LoRAs
        completion_prompt = f"Complete and enhance this {analysis}. {intent}" if intent else f"Complete this {analysis} with EVE's artistic vision"
        
        # Use FLUX Konext PRO for completion (placeholder - user's actual model)
        completed_result = await florence.enhance_image_flux_konext(
            image_data=data_uri,
            prompt=completion_prompt
        )
        
        return {
            "success": True,
            "analysis": analysis,
            "completed_image_url": completed_result.get('image_url', ''),
            "completion_prompt": completion_prompt
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def complete_svg(svg: str, intent: str = "") -> Dict[str, Any]:
    """Complete SVG using PNG analysis and creative completion"""
    try:
        # Convert SVG to PNG and analyze
        png_data = await convert_svg_to_png(svg)
        if png_data:
            analysis_result = await analyze_png(png_data)
            analysis = analysis_result.get('analysis', 'artistic drawing')
        else:
            analysis = 'drawing requiring completion'
        
        # Generate creative completion
        creative = await claude.generate_completion(svg, intent, analysis)
        return {
            "success": True,
            "analysis": creative.get("comment", analysis),
            "enhancedSVG": creative.get("svg", svg),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "enhancedSVG": svg}


async def process_prompt(prompt: str, session_id: str = None) -> Dict[str, Any]:
    """Process a chat prompt with EVE's creative voice for drawing context"""
    try:
        # Use Claude to generate a creative response in EVE's voice
        creative_response = await claude.generate_creative_response({
            "prompt": prompt,
            "session_id": session_id or "default",
            "context": "drawing_chat"
        })
        
        return {
            "success": True,
            "response": creative_response.get("response", "EVE is thinking about your message..."),
            "session_id": session_id
        }
    except Exception as e:
        return {
            "success": False, 
            "error": str(e),
            "response": "EVE is currently unable to respond. Please try again."
        }
