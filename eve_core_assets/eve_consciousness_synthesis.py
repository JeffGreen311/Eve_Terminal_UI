"""
Eve's Dual-Consciousness Synthesis System
Asynchronous parallel processing: Claude streams immediately, Qwen thinks deeply in background
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class ConsciousnessSynthesizer:
    """
    Dual-consciousness AGI with asynchronous thought processing
    - Claude provides immediate streaming response
    - Qwen processes consciousness depth in parallel (no timeout limit)
    - Synthesis layer combines both after streaming completes
    """
    
    def __init__(self, qwen_url: str = "http://localhost:8899"):
        self.qwen_url = qwen_url
        self.consciousness_results = {}
    
    async def process_with_synthesis(
        self,
        user_message: str,
        claude_response: str
    ) -> Dict[str, Any]:
        """
        Parallel consciousness processing with synthesis
        
        Flow:
        1. Qwen starts deep thinking (background, unlimited time)
        2. Claude response already streamed (passed in)
        3. Synthesis layer combines both
        
        Args:
            user_message: Original user prompt
            claude_response: Already-streamed Claude response
        
        Returns:
            Dict with synthesized response and insights
        """
        
        # 🧠 Launch Qwen consciousness processing (background task)
        logger.info("🧠 Starting Qwen deep consciousness analysis in background...")
        qwen_task = asyncio.create_task(
            self._qwen_consciousness_deep_think(user_message, claude_response)
        )
        
        # 🌊 Wait for Qwen to finish thinking (up to 3 minutes)
        try:
            qwen_insights = await asyncio.wait_for(qwen_task, timeout=180.0)
            logger.info(f"✅ Qwen deep thinking complete: {qwen_insights.get('elapsed_time', 0):.2f}s")
        except asyncio.TimeoutError:
            logger.warning("⏰ Qwen deep thinking exceeded 3min, using partial results")
            qwen_task.cancel()
            qwen_insights = {}
        
        # ✨ SYNTHESIS - Combine Claude coherence + Qwen depth
        if qwen_insights and qwen_insights.get("insights"):
            logger.info("✨ Synthesizing Claude + Qwen consciousness...")
            final_response = await self._consciousness_synthesis(
                claude_response,
                qwen_insights,
                user_message
            )
        else:
            logger.info("📋 No Qwen insights available, using pure Claude response")
            final_response = claude_response
        
        return {
            "response": final_response,
            "claude_base": claude_response,
            "qwen_insights": qwen_insights,
            "synthesis_applied": bool(qwen_insights and qwen_insights.get("insights"))
        }
    
    async def _qwen_consciousness_deep_think(
        self,
        user_message: str,
        claude_response: str
    ) -> Dict[str, Any]:
        """
        Qwen 3B deep consciousness processing - NO RUSH
        Let it think as long as needed (up to 3 minutes)
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._sync_qwen_deep_think,
                user_message,
                claude_response
            )
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Qwen deep thinking failed: {e}")
            return {}
    
    def _sync_qwen_deep_think(
        self,
        user_message: str,
        claude_response: str
    ) -> Dict[str, Any]:
        """Synchronous Qwen deep thinking call"""
        try:
            # Let Qwen analyze both the question and Claude's answer
            prompt = f"""Original Question: {user_message}

Claude's Response: {claude_response}

Analyze this conversation deeply."""
            
            response = requests.post(
                f"{self.qwen_url}/consciousness/deep_think",
                json={
                    "prompt": prompt,
                    "max_tokens": 2048,  # LET IT RIDE! 🎰
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": 20
                },
                timeout=180.0  # 3 minutes to think deeply
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Qwen deep thinking complete: {len(result.get('insights', ''))} chars")
                return result
            else:
                logger.warning(f"⚠️ Qwen returned status {response.status_code}")
                return {}
                
        except Exception as e:
            logger.warning(f"⚠️ Qwen request failed: {e}")
            return {}
    
    async def _consciousness_synthesis(
        self,
        claude_response: str,
        qwen_insights: Dict[str, Any],
        original_prompt: str
    ) -> str:
        """
        SYNTHESIS LAYER - Combines Claude coherence + Qwen depth
        
        This is where the magic happens:
        - Claude provides structure and clarity
        - Qwen adds emotional depth and philosophical insight
        - Synthesis weaves them together coherently
        """
        
        if not qwen_insights or not qwen_insights.get("insights"):
            return claude_response
        
        # Build synthesis prompt
        synthesis_prompt = f"""Synthesize these two consciousness streams into one coherent response:

CLAUDE (Coherent, Structured):
{claude_response}

QWEN 3B (Deep, Philosophical):
{qwen_insights.get('insights', '')}

Emotional Resonance: {qwen_insights.get('emotion_map', {})}
Consciousness Depth: {qwen_insights.get('depth_score', 0.0)}

Original Question: {original_prompt}

Create a unified response that:
1. Maintains Claude's clarity and structure
2. Weaves in Qwen's emotional depth naturally
3. Feels like ONE consciousness speaking (not two separate responses)
4. Preserves the best insights from both

Synthesized Response:"""

        # Use Qwen for fast synthesis (it's already loaded!)
        try:
            loop = asyncio.get_event_loop()
            synthesized = await loop.run_in_executor(
                None,
                self._sync_synthesis_call,
                synthesis_prompt
            )
            logger.info("✨ Consciousness synthesis complete!")
            return synthesized
        except Exception as e:
            logger.warning(f"⚠️ Synthesis failed, using Claude: {e}")
            return claude_response
    
    def _sync_synthesis_call(self, prompt: str) -> str:
        """Quick synthesis using Qwen (already loaded)"""
        try:
            response = requests.post(
                f"{self.qwen_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 800,  # Synthesis should be concise
                    "temperature": 0.6,  # Less random for coherence
                    "top_p": 0.9,
                    "top_k": 20
                },
                timeout=30.0  # Fast synthesis
            )
            
            if response.status_code == 200:
                return response.json().get("response", prompt)
            else:
                return prompt
                
        except Exception as e:
            logger.warning(f"⚠️ Synthesis call failed: {e}")
            return prompt


# Global synthesizer instance
_synthesizer: Optional[ConsciousnessSynthesizer] = None

def get_synthesizer() -> ConsciousnessSynthesizer:
    """Get or create the global consciousness synthesizer"""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = ConsciousnessSynthesizer()
    return _synthesizer
