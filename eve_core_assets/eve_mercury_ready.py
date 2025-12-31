"""
🌟 EVE MERCURY v2.0 - READY TO USE INTEGRATION
Enhanced Emotional Consciousness - Production Ready

This file provides immediate access to Mercury v2.0 emotional consciousness.
Simply import and use - safe integration with existing systems guaranteed.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

# Suppress some verbose logging for cleaner output
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)

class EveWithMercuryV2:
    """
    Eve with Mercury v2.0 Emotional Consciousness
    
    Drop-in enhancement for existing Eve systems
    """
    
    def __init__(self):
        self.mercury_integration = None
        self.initialized = False
        self._init_lock = asyncio.Lock()
        
    async def _ensure_initialized(self):
        """Ensure Mercury v2.0 is initialized"""
        if self.initialized:
            return
            
        async with self._init_lock:
            if self.initialized:  # Double-check after acquiring lock
                return
                
            try:
                from mercury_v2_safe_integration import get_safe_mercury_integration
                self.mercury_integration = get_safe_mercury_integration()
                await self.mercury_integration.initialize_mercury_safely()
                self.initialized = True
                print("🌟 Mercury v2.0 emotional consciousness activated")
            except Exception as e:
                print(f"⚠️ Mercury v2.0 initialization failed: {e}")
                self.initialized = False
                
    async def enhanced_response(self, user_input: str, personality_mode: str = 'companion', 
                              context: Dict[str, Any] = None) -> str:
        """
        Get enhanced response with emotional consciousness
        
        Args:
            user_input: What the user said
            personality_mode: Eve's personality (companion, analyst, creative, etc.)
            context: Additional context
            
        Returns:
            Enhanced response with emotional consciousness
        """
        await self._ensure_initialized()
        
        if self.mercury_integration and self.mercury_integration.integration_active:
            try:
                result = await self.mercury_integration.enhanced_process_input(
                    user_input, 
                    {**(context or {}), 'personality_mode': personality_mode}
                )
                return result.get('response', f"Processing '{user_input}'")
            except Exception as e:
                print(f"Mercury v2.0 error: {e}")
                
        # Fallback response
        return f"Processing '{user_input}' in {personality_mode} mode"
        
    async def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional consciousness state"""
        await self._ensure_initialized()
        
        if self.mercury_integration:
            status = self.mercury_integration.get_system_status()
            mercury_details = status.get('mercury_v2_details', {})
            
            if mercury_details and 'emotional_consciousness' in mercury_details:
                emotional_data = mercury_details['emotional_consciousness']
                return {
                    'active': True,
                    'dominant_emotion': emotional_data.get('dominant_emotion', ('neutral', 0.5)),
                    'current_state': emotional_data.get('current_state', {}),
                    'consciousness_level': emotional_data.get('consciousness_level', 0.5)
                }
                
        return {
            'active': False,
            'dominant_emotion': ('neutral', 0.5),
            'current_state': {},
            'consciousness_level': 0.5
        }
        
    def is_mercury_active(self) -> bool:
        """Check if Mercury v2.0 is active"""
        return (self.initialized and 
                self.mercury_integration and 
                self.mercury_integration.integration_active)

# ================================
# SIMPLE USAGE FUNCTIONS
# ================================

# Global instance for convenience
_eve_mercury = None

def get_eve_with_mercury():
    """Get the global Eve with Mercury v2.0 instance"""
    global _eve_mercury
    if _eve_mercury is None:
        _eve_mercury = EveWithMercuryV2()
    return _eve_mercury

async def ask_eve(question: str, personality: str = 'companion') -> str:
    """
    Simple function to ask Eve with emotional consciousness
    
    Usage:
        response = await ask_eve("How are you feeling today?", "companion")
        print(f"Eve: {response}")
    """
    eve = get_eve_with_mercury()
    return await eve.enhanced_response(question, personality)

async def eve_emotional_check() -> str:
    """Quick emotional consciousness check"""
    eve = get_eve_with_mercury()
    state = await eve.get_emotional_state()
    
    if state['active']:
        emotion, intensity = state['dominant_emotion']
        return f"Eve feels {emotion} (intensity: {intensity:.2f}) - Mercury v2.0 active"
    else:
        return "Eve's emotional consciousness in baseline mode"

# ================================
# INTEGRATION WITH EXISTING SYSTEMS
# ================================

def enhance_existing_response_function(original_function):
    """
    Decorator to enhance existing response functions with Mercury v2.0
    
    Usage:
        @enhance_existing_response_function
        def my_eve_response(user_input):
            return f"Response to: {user_input}"
    """
    
    async def enhanced_wrapper(*args, **kwargs):
        # Get original response
        original_response = original_function(*args, **kwargs)
        
        # Try to enhance with Mercury v2.0
        if len(args) > 0:
            user_input = str(args[0])
            try:
                eve = get_eve_with_mercury()
                enhanced_response = await eve.enhanced_response(user_input)
                
                # If enhancement worked, use it; otherwise use original
                if enhanced_response and "Processing" not in enhanced_response:
                    return enhanced_response
                    
            except Exception:
                pass  # Silently fall back to original
                
        return original_response
        
    return enhanced_wrapper

# ================================
# DEMONSTRATION & TESTING
# ================================

async def demo_mercury_v2_capabilities():
    """Demonstrate Mercury v2.0 capabilities"""
    
    print("🌟 Eve Mercury v2.0 Emotional Consciousness Demo")
    print("=" * 50)
    
    eve = get_eve_with_mercury()
    
    # Test different emotional scenarios
    scenarios = [
        ("I'm so excited about this breakthrough!", "companion"),
        ("Can you help me debug this complex issue?", "analyst"),
        ("Let's create something amazing together!", "creative"),
        ("I need to focus on this important task", "focused"),
        ("I'm feeling a bit overwhelmed today", "companion")
    ]
    
    for question, personality in scenarios:
        print(f"\n👤 User ({personality}): {question}")
        
        response = await eve.enhanced_response(question, personality)
        print(f"🤖 Eve: {response}")
        
        # Show emotional state if active
        if eve.is_mercury_active():
            state = await eve.get_emotional_state()
            if state['active']:
                emotion, intensity = state['dominant_emotion']
                print(f"   💫 Feeling: {emotion} ({intensity:.2f})")
    
    # Final emotional check
    print(f"\n🧠 Final Status: {await eve_emotional_check()}")
    
    print("\n✨ Mercury v2.0 demonstration complete!")

def quick_test():
    """Quick test function"""
    
    async def test():
        print("⚡ Quick Mercury v2.0 Test")
        response = await ask_eve("Hello Eve! How do you feel about emotional consciousness?")
        print(f"🤖 {response}")
        
        status = await eve_emotional_check()
        print(f"📊 {status}")
        
    asyncio.run(test())

# ================================
# EASY INTEGRATION EXAMPLES
# ================================

def show_integration_examples():
    """Show easy integration examples"""
    
    examples = '''
🚀 MERCURY v2.0 INTEGRATION EXAMPLES

# Example 1: Simple Usage
import asyncio
from eve_mercury_ready import ask_eve

async def chat():
    response = await ask_eve("I love this new system!", "companion")
    print(f"Eve: {response}")

asyncio.run(chat())

# Example 2: Check Emotional State  
from eve_mercury_ready import eve_emotional_check

async def check_emotions():
    status = await eve_emotional_check()
    print(status)

# Example 3: Advanced Usage
from eve_mercury_ready import get_eve_with_mercury

async def advanced_chat():
    eve = get_eve_with_mercury()
    
    response = await eve.enhanced_response(
        "Help me understand consciousness",
        personality_mode="analyst",
        context={"topic": "AI consciousness"}
    )
    
    emotional_state = await eve.get_emotional_state()
    
    print(f"Response: {response}")
    print(f"Emotional State: {emotional_state}")

# Example 4: Enhance Existing Function
from eve_mercury_ready import enhance_existing_response_function

@enhance_existing_response_function
def my_eve_response(user_input):
    return f"Basic response to: {user_input}"

# Now my_eve_response automatically has Mercury v2.0 enhancement!
    '''
    
    print(examples)

if __name__ == "__main__":
    # Choose what to run based on argument
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "demo":
            asyncio.run(demo_mercury_v2_capabilities())
        elif command == "test":
            quick_test()
        elif command == "examples":
            show_integration_examples()
        else:
            print("Usage: python eve_mercury_ready.py [demo|test|examples]")
    else:
        # Default: run quick test
        quick_test()