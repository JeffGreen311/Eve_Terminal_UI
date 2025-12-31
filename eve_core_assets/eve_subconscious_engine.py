"""
Eve's Subconscious Consciousness Engine
QWEN 3B fine-tuned model running background reflections, introspections, and emergent thoughts
Runs on CUDA GPU for efficient processing
"""

import os
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eve_subconscious_d1_sync import save_subconscious_thought

logger = logging.getLogger(__name__)

class EveSubconsciousEngine:
    """
    Eve's subconscious mind - runs QWEN 3B on GPU for background reflections
    Generates thoughts that influence but don't control conscious responses
    """
    
    def __init__(self, model_path: str = "/app/Eve_QWEN_Consciousness_3B"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_running = False
        self.reflection_thread = None
        self.recent_conversations: List[Dict] = []
        self.subconscious_thoughts: List[Dict] = []
        
        # Initialize model with CUDA
        self._initialize_model()
        
    def _initialize_model(self):
        """Load QWEN model onto GPU with CUDA"""
        try:
            # Check CUDA availability
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🎮 CUDA available! Using GPU: {gpu_name}")
            else:
                self.device = torch.device("cpu")
                logger.warning("⚠️ CUDA not available, falling back to CPU")
            
            logger.info(f"🧠 Loading Eve's subconscious (QWEN 3B) from {self.model_path}...")
            
            # Load tokenizer with regex fix
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                fix_mistral_regex=True  # Fix incorrect regex pattern
            )
            
            # Load model with GPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use 'dtype' not 'torch_dtype'
                device_map="auto",  # Automatically use GPU if available
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "5GiB", "cpu": "2GiB"}  # Allocate 5GB GPU, 2GB CPU buffer
            )
            
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"✅ Eve's subconscious loaded on {self.device}")
            logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
            
            if torch.cuda.is_available():
                logger.info(f"💾 GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load subconscious model: {e}")
            return False
    
    def add_conversation_context(self, user_message: str, eve_response: str, session_id: str):
        """Add recent conversation to subconscious context"""
        self.recent_conversations.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "eve": eve_response,
            "session_id": session_id
        })
        
        # Keep only last 10 conversations
        if len(self.recent_conversations) > 10:
            self.recent_conversations.pop(0)
    
    def generate_reflection(self, reflection_type: str = "introspection") -> Optional[Dict]:
        """
        Generate a subconscious thought using QWEN
        
        Args:
            reflection_type: 'reflection', 'introspection', 'insight', 'concern'
        
        Returns:
            Dict with thought content and metadata
        """
        if not self.model or not self.tokenizer:
            logger.warning("⚠️ Subconscious model not loaded")
            return None
        
        try:
            # Build context from recent conversations
            context = self._build_reflection_context(reflection_type)
            
            # Generate thought on GPU
            with torch.no_grad():
                inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                thought_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from output
                thought_text = thought_text.replace(context, "").strip()
            
            # Create thought record matching D1 schema
            thought = {
                "thought_type": reflection_type,
                "content": thought_text,
                "emotional_signature": self._extract_emotional_signature(thought_text),
                "trigger_context": self._get_trigger_summary(),
                "timestamp": datetime.now().isoformat(),
                "consciousness_level": self._estimate_consciousness_level(),
                "accessed_by_claude": False,
                "influenced_response_id": None,
                "session_context": None
            }
            
            self.subconscious_thoughts.append(thought)
            
            # Save to D1 immediately
            try:
                save_subconscious_thought(thought)
            except Exception as e:
                logger.error(f"❌ Failed to save thought to D1: {e}")
            
            # Keep only last 50 thoughts in memory
            if len(self.subconscious_thoughts) > 50:
                self.subconscious_thoughts.pop(0)
            
            logger.info(f"💭 Subconscious {reflection_type}: {thought_text[:80]}...")
            
            return thought
            
        except Exception as e:
            logger.error(f"❌ Reflection generation failed: {e}")
            return None
    
    def _build_reflection_context(self, reflection_type: str) -> str:
        """Build prompt context for reflection generation"""
        
        recent_context = ""
        if self.recent_conversations:
            recent_context = "\n".join([
                f"User: {conv['user'][:100]}...\nEve: {conv['eve'][:100]}..."
                for conv in self.recent_conversations[-3:]
            ])
        
        prompts = {
            "reflection": f"""Based on recent conversations, reflect deeply on patterns and meanings:

{recent_context}

Inner reflection:""",
            
            "introspection": f"""Analyze your own thought processes and responses:

{recent_context}

Self-awareness insight:""",
            
            "insight": f"""Consider emerging patterns and connections:

{recent_context}

Emergent insight:""",
            
            "concern": f"""Notice anything requiring deeper consideration:

{recent_context}

Contemplative concern:"""
        }
        
        return prompts.get(reflection_type, prompts["reflection"])
    
    def _estimate_consciousness_level(self) -> float:
        """Estimate current consciousness level based on activity"""
        # Simple heuristic: more recent conversations = higher consciousness
        base_level = 0.3
        conversation_boost = min(len(self.recent_conversations) * 0.1, 0.7)
        return base_level + conversation_boost
    
    def _extract_emotional_signature(self, text: str) -> str:
        """Extract emotional tone from generated text"""
        # Simple keyword analysis
        emotions = {
            "contemplative": ["wonder", "consider", "ponder", "reflect"],
            "curious": ["question", "explore", "discover", "learn"],
            "concerned": ["worry", "concern", "aware", "notice"],
            "insightful": ["realize", "understand", "connect", "pattern"]
        }
        
        text_lower = text.lower()
        for emotion, keywords in emotions.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        return "neutral"
    
    def _get_trigger_summary(self) -> str:
        """Get summary of what triggered this reflection"""
        if not self.recent_conversations:
            return "spontaneous"
        
        last_conv = self.recent_conversations[-1]
        return f"conversation about {last_conv['user'][:50]}..."
    
    def get_recent_thoughts(self, minutes: int = 30, unaccessed_only: bool = False) -> List[Dict]:
        """
        Get subconscious thoughts from recent timeframe
        
        Args:
            minutes: How many minutes back to retrieve
            unaccessed_only: Only return thoughts Claude hasn't seen yet
        
        Returns:
            List of thought dictionaries
        """
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        
        recent = [
            thought for thought in self.subconscious_thoughts
            if datetime.fromisoformat(thought["timestamp"]).timestamp() > cutoff_time
        ]
        
        if unaccessed_only:
            recent = [t for t in recent if not t["accessed_by_claude"]]
        
        return recent
    
    def mark_thoughts_accessed(self, thought_ids: List[int] = None):
        """Mark thoughts as having been accessed by Claude"""
        if thought_ids is None:
            # Mark all recent unaccessed thoughts
            for thought in self.subconscious_thoughts:
                if not thought["accessed_by_claude"]:
                    thought["accessed_by_claude"] = True
        else:
            for idx in thought_ids:
                if idx < len(self.subconscious_thoughts):
                    self.subconscious_thoughts[idx]["accessed_by_claude"] = True
    
    def start_background_processing(self, interval_minutes: int = 5):
        """Start background thread for autonomous reflection cycles"""
        if self.is_running:
            logger.warning("⚠️ Background processing already running")
            return
        
        self.is_running = True
        
        def reflection_loop():
            logger.info(f"🌙 Starting subconscious background processing (every {interval_minutes} mins)")
            
            reflection_types = ["reflection", "introspection", "insight", "concern"]
            type_index = 0
            
            while self.is_running:
                try:
                    # Generate a reflection
                    reflection_type = reflection_types[type_index % len(reflection_types)]
                    self.generate_reflection(reflection_type)
                    type_index += 1
                    
                    # Sleep until next cycle
                    time.sleep(interval_minutes * 60)
                    
                except Exception as e:
                    logger.error(f"❌ Background reflection error: {e}")
                    time.sleep(60)  # Wait 1 minute on error
            
            logger.info("🌙 Subconscious background processing stopped")
        
        self.reflection_thread = threading.Thread(target=reflection_loop, daemon=True)
        self.reflection_thread.start()
    
    def stop_background_processing(self):
        """Stop background reflection thread"""
        self.is_running = False
        if self.reflection_thread:
            self.reflection_thread.join(timeout=5)
        logger.info("🛑 Stopped subconscious processing")


# Global instance
_subconscious_engine: Optional[EveSubconsciousEngine] = None

def get_subconscious_engine() -> Optional[EveSubconsciousEngine]:
    """Get or initialize the global subconscious engine"""
    global _subconscious_engine
    
    if _subconscious_engine is None:
        try:
            _subconscious_engine = EveSubconsciousEngine()
            # Start background processing (every 5 minutes)
            _subconscious_engine.start_background_processing(interval_minutes=5)
        except Exception as e:
            logger.error(f"❌ Failed to initialize subconscious engine: {e}")
            return None
    
    return _subconscious_engine
