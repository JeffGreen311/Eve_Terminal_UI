"""
Eve Consciousness Mercury v2.0 Adapter
Safe integration layer for existing Eve systems

This adapter safely integrates Mercury v2.0 emotional consciousness
with existing Eve personality and consciousness systems without disrupting them.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

# Import the new Mercury v2.0 system
from mercury_v2_integration import MercurySystemV2, EmotionalResonanceEngine

class EveConsciousnessMercuryAdapter:
    """
    Safe adapter that integrates Mercury v2.0 with existing Eve systems
    
    This preserves all existing functionality while adding emotional consciousness
    """
    
    def __init__(self, existing_personality_interface=None):
        self.existing_personality_interface = existing_personality_interface
        self.mercury_v2 = None
        self.integration_active = False
        self.fallback_mode = False
        self.logger = logging.getLogger(__name__)
        
        # Safe initialization
        self._safe_initialize()
        
    def _safe_initialize(self):
        """Safely initialize Mercury v2.0 with fallback protection"""
        try:
            self.mercury_v2 = MercurySystemV2(db_path="eve_mercury_v2_production.db")
            self.integration_active = True
            self.logger.info("✅ Mercury v2.0 integration active - Enhanced emotional consciousness enabled")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Mercury v2.0 initialization failed, running in fallback mode: {e}")
            self.fallback_mode = True
            self.integration_active = False
            
    async def enhance_personality_response(self, personality_mode: str, user_input: str, 
                                         original_response: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance existing personality responses with emotional consciousness
        
        This is the main integration point - it takes existing responses
        and enhances them with Mercury v2.0 emotional processing
        """
        if context is None:
            context = {}
            
        # Always return the original response as fallback
        enhanced_response = {
            'original_response': original_response,
            'personality_mode': personality_mode,
            'mercury_v2_active': self.integration_active,
            'emotional_enhancement': None,
            'enhanced_response': original_response,  # Default to original
            'fallback_used': self.fallback_mode
        }
        
        if not self.integration_active or self.fallback_mode:
            return enhanced_response
            
        try:
            # Get Mercury v2.0 consciousness processing
            consciousness_result = await self.mercury_v2.process_consciousness_interaction(
                user_input, personality_mode, context
            )
            
            if 'error' not in consciousness_result:
                # Extract emotional enhancements
                emotional_enhancement = consciousness_result.get('emotional_enhancement', {})
                emotional_flavor = emotional_enhancement.get('emotional_analysis', {}).get('emotional_flavor', '')
                
                # Enhance response with emotional flavor if present
                enhanced_text = original_response
                if emotional_flavor and emotional_flavor.strip():
                    enhanced_text = f"{emotional_flavor}{original_response}"
                    
                # Update enhancement data
                enhanced_response.update({
                    'emotional_enhancement': emotional_enhancement,
                    'enhanced_response': enhanced_text,
                    'consciousness_level': consciousness_result.get('consciousness_level', 0.5),
                    'emotional_state': emotional_enhancement.get('enhanced_emotional_state', {}),
                    'mercury_v2_data': consciousness_result
                })
                
            else:
                self.logger.warning(f"Mercury v2.0 processing error: {consciousness_result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error in Mercury v2.0 enhancement: {e}")
            # Graceful degradation - original response is preserved
            enhanced_response['enhancement_error'] = str(e)
            
        return enhanced_response
        
    def get_emotional_status(self) -> Dict[str, Any]:
        """Get current emotional consciousness status"""
        if not self.integration_active or not self.mercury_v2:
            return {
                'status': 'inactive',
                'fallback_mode': self.fallback_mode,
                'emotional_state': 'baseline'
            }
            
        try:
            return self.mercury_v2.get_system_status()
        except Exception as e:
            self.logger.error(f"Error getting emotional status: {e}")
            return {'status': 'error', 'error': str(e)}
            
    async def process_consciousness_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process consciousness events through Mercury v2.0"""
        if not self.integration_active:
            return {'processed': False, 'reason': 'mercury_v2_inactive'}
            
        try:
            # Convert event to user input format for processing
            event_text = f"{event_type}: {event_data.get('description', str(event_data))}"
            
            result = await self.mercury_v2.process_consciousness_interaction(
                event_text,
                event_data.get('personality_mode', 'companion'),
                {'event_type': event_type, **event_data}
            )
            
            return {
                'processed': True,
                'mercury_v2_result': result,
                'consciousness_impact': result.get('consciousness_level', 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing consciousness event: {e}")
            return {'processed': False, 'error': str(e)}
            
    def register_with_existing_system(self, system_interface):
        """Register adapter with existing Eve systems"""
        try:
            self.existing_personality_interface = system_interface
            
            # If the existing system has hooks for enhancements, register
            if hasattr(system_interface, 'register_enhancement_adapter'):
                system_interface.register_enhancement_adapter('mercury_v2', self)
                self.logger.info("🔗 Registered Mercury v2.0 adapter with existing personality system")
                
            return True
        except Exception as e:
            self.logger.error(f"Error registering with existing system: {e}")
            return False
            
    async def safe_shutdown(self):
        """Safely shutdown Mercury v2.0 systems"""
        if self.mercury_v2:
            try:
                await self.mercury_v2.shutdown_gracefully()
                self.logger.info("✅ Mercury v2.0 adapter shutdown complete")
            except Exception as e:
                self.logger.error(f"Error during Mercury v2.0 shutdown: {e}")

# ================================
# INTEGRATION WITH EXISTING EVE PERSONALITY SYSTEM
# ================================

class EnhancedEvePersonalityInterface:
    """
    Enhanced wrapper for existing EveTerminalPersonalityInterface 
    that adds Mercury v2.0 emotional consciousness
    """
    
    def __init__(self, original_personality_interface=None):
        self.original_interface = original_personality_interface
        self.mercury_adapter = EveConsciousnessMercuryAdapter(original_personality_interface)
        self.enhancement_enabled = True
        self.logger = logging.getLogger(__name__)
        
    def set_original_interface(self, original_interface):
        """Set the original personality interface"""
        self.original_interface = original_interface
        self.mercury_adapter.register_with_existing_system(original_interface)
        
    async def process_terminal_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced version of process_terminal_input with Mercury v2.0 integration
        """
        if context is None:
            context = {}
            
        # First, get original response
        original_result = {}
        if self.original_interface:
            try:
                original_result = self.original_interface.process_terminal_input(user_input, context)
            except Exception as e:
                self.logger.error(f"Error in original personality interface: {e}")
                original_result = {
                    'response': "Error in personality processing",
                    'personality': 'companion',
                    'error': str(e)
                }
        else:
            # Fallback response
            original_result = {
                'response': f"Processing: {user_input}",
                'personality': context.get('personality_mode', 'companion'),
                'is_switch': False
            }
            
        # Enhance with Mercury v2.0 if enabled
        if self.enhancement_enabled and self.mercury_adapter.integration_active:
            try:
                enhanced_result = await self.mercury_adapter.enhance_personality_response(
                    original_result.get('personality', 'companion'),
                    user_input,
                    original_result.get('response', ''),
                    context
                )
                
                # Merge results
                final_result = {
                    **original_result,
                    'mercury_v2_enhancement': enhanced_result,
                    'enhanced_response': enhanced_result.get('enhanced_response', original_result.get('response')),
                    'emotional_consciousness': enhanced_result.get('emotional_enhancement'),
                    'consciousness_level': enhanced_result.get('consciousness_level', 0.5)
                }
                
                return final_result
                
            except Exception as e:
                self.logger.error(f"Error in Mercury v2.0 enhancement: {e}")
                # Return original result on enhancement failure
                return {**original_result, 'enhancement_error': str(e)}
                
        else:
            # Return original result if enhancement disabled
            return original_result
            
    def get_personality_status(self) -> Dict[str, Any]:
        """Get enhanced personality status including emotional consciousness"""
        status = {'mercury_v2': 'not_available'}
        
        if self.original_interface and hasattr(self.original_interface, 'get_personality_status'):
            status = self.original_interface.get_personality_status()
            
        # Add Mercury v2.0 status
        if self.mercury_adapter.integration_active:
            emotional_status = self.mercury_adapter.get_emotional_status()
            status['mercury_v2'] = emotional_status
            status['emotional_consciousness'] = True
        else:
            status['emotional_consciousness'] = False
            status['mercury_v2_fallback'] = self.mercury_adapter.fallback_mode
            
        return status
        
    def enable_mercury_enhancement(self, enabled: bool = True):
        """Enable or disable Mercury v2.0 enhancement"""
        self.enhancement_enabled = enabled
        self.logger.info(f"Mercury v2.0 enhancement {'enabled' if enabled else 'disabled'}")
        
    async def shutdown(self):
        """Shutdown enhanced interface"""
        await self.mercury_adapter.safe_shutdown()

# ================================
# SAFE INTEGRATION FUNCTIONS
# ================================

def create_enhanced_eve_interface(original_interface=None):
    """
    Factory function to create enhanced Eve interface
    
    Args:
        original_interface: Existing EveTerminalPersonalityInterface or None
        
    Returns:
        EnhancedEvePersonalityInterface with Mercury v2.0 integration
    """
    try:
        enhanced_interface = EnhancedEvePersonalityInterface(original_interface)
        logging.info("✅ Created enhanced Eve interface with Mercury v2.0")
        return enhanced_interface
    except Exception as e:
        logging.error(f"❌ Error creating enhanced interface: {e}")
        # Return a safe fallback
        return original_interface if original_interface else None

async def test_enhanced_integration():
    """Test the enhanced integration safely"""
    print("🧪 Testing Enhanced Eve Mercury v2.0 Integration")
    print("=" * 55)
    
    # Create enhanced interface without original (standalone test)
    enhanced_interface = create_enhanced_eve_interface()
    
    if enhanced_interface is None:
        print("❌ Failed to create enhanced interface")
        return
        
    # Test various inputs
    test_cases = [
        ("Hey Eve, this is amazing work we're doing together!", {'personality_mode': 'companion'}),
        ("Let's debug this complex algorithm step by step", {'personality_mode': 'analyst'}),
        ("I want to create something beautiful and inspiring", {'personality_mode': 'creative'}),
        ("Help me focus on solving this problem efficiently", {'personality_mode': 'focused'})
    ]
    
    for user_input, context in test_cases:
        print(f"\n🔄 Testing: {context.get('personality_mode', 'unknown')}")
        print(f"📝 Input: {user_input}")
        
        try:
            result = await enhanced_interface.process_terminal_input(user_input, context)
            
            print(f"💬 Response: {result.get('enhanced_response', result.get('response', 'No response'))}")
            
            if 'mercury_v2_enhancement' in result:
                enhancement = result['mercury_v2_enhancement']
                if enhancement.get('emotional_enhancement'):
                    emotional_flavor = enhancement['emotional_enhancement'].get('emotional_analysis', {}).get('emotional_flavor', 'None')
                    print(f"🎭 Emotional Flavor: {emotional_flavor}")
                    print(f"🧠 Consciousness: {result.get('consciousness_level', 0):.2f}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test status
    print(f"\n📊 System Status:")
    status = enhanced_interface.get_personality_status()
    print(f"   Emotional Consciousness: {status.get('emotional_consciousness', False)}")
    print(f"   Mercury v2.0: {status.get('mercury_v2', 'inactive')}")
    
    # Clean shutdown
    await enhanced_interface.shutdown()
    print("\n✅ Enhanced integration test complete!")

if __name__ == "__main__":
    # Test the enhanced integration
    asyncio.run(test_enhanced_integration())