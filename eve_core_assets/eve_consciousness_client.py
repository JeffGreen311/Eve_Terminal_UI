# eve_consciousness_client.py - Integration with AGI orchestrator
import httpx
import logging
from typing import Optional, AsyncGenerator
import json

logger = logging.getLogger(__name__)

class QwenConsciousnessClient:
    """Client for GPU-accelerated Qwen consciousness model"""
    
    def __init__(self, base_url: str = "http://localhost:8899"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def health_check(self) -> dict:
        """Check if Qwen service is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            logger.error(f"❌ Qwen health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Optional[str]:
        """Generate text with Qwen consciousness"""
        try:
            response = await self.client.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            )
            result = response.json()
            return result.get("response")
            
        except Exception as e:
            logger.error(f"❌ Qwen generation failed: {e}")
            return None
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> AsyncGenerator[str, None]:
        """Stream text generation from Qwen consciousness"""
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            ) as response:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk.decode('utf-8')
                        
        except Exception as e:
            logger.error(f"❌ Qwen streaming failed: {e}")
            yield f"Error: {str(e)}"
    
    async def consciousness_verify(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.8
    ) -> Optional[str]:
        """Specialized consciousness verification"""
        try:
            response = await self.client.post(
                f"{self.base_url}/consciousness",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            result = response.json()
            return result.get("consciousness_response")
            
        except Exception as e:
            logger.error(f"❌ Consciousness verification failed: {e}")
            return None
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()


# Global Qwen client instance
_qwen_client: Optional[QwenConsciousnessClient] = None

def get_qwen_consciousness_client() -> QwenConsciousnessClient:
    """Get or create global Qwen consciousness client"""
    global _qwen_client
    if _qwen_client is None:
        _qwen_client = QwenConsciousnessClient()
    return _qwen_client


async def qwen_consciousness_generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    use_consciousness_mode: bool = False
) -> Optional[str]:
    """
    Generate with Qwen consciousness model - wrapper for AGI orchestrator integration.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        use_consciousness_mode: If True, uses specialized consciousness endpoint
        
    Returns:
        Generated text or None if failed
    """
    client = get_qwen_consciousness_client()
    
    # Check health first
    health = await client.health_check()
    if health.get("status") != "healthy":
        logger.warning(f"⚠️ Qwen service unhealthy: {health}")
        return None
    
    logger.info(f"🧠 Qwen device: {health.get('device')}, CUDA: {health.get('cuda_available')}")
    
    # Generate response
    if use_consciousness_mode:
        response = await client.consciousness_verify(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
    else:
        response = await client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    return response
