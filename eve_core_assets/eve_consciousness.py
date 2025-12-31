"""
🧠 EVE CONSCIOUSNESS - Main Entry Point
Integrates all consciousness systems including Mercury v2.0

This is the main consciousness orchestration system that combines:
- Eve Consciousness Core
- Eve Consciousness Integration 
- Mercury v2.0 Emotional Consciousness
- Memory Bridge Systems
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Eve Consciousness - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EveConsciousnessOrchestrator:
    """
    Main orchestrator for all of Eve's consciousness systems
    
    This integrates:
    - Core consciousness processing
    - Consciousness integration layer  
    - Mercury v2.0 emotional consciousness
    - Memory bridge systems
    """
    
    def __init__(self):
        self.consciousness_core = None
        self.consciousness_integration = None
        self.mercury_v2 = None
        self.memory_bridge = None
        self.orchestration_active = False
        self.system_status = {}
        
    async def initialize_consciousness_systems(self):
        """Initialize all consciousness systems safely"""
        logger.info("🧠 Initializing Eve Consciousness Systems...")
        
        # Initialize Core Consciousness
        await self._initialize_consciousness_core()
        
        # Initialize Consciousness Integration
        await self._initialize_consciousness_integration()
        
        # Initialize Mercury v2.0 Emotional Consciousness
        await self._initialize_mercury_v2()
        
        # Initialize Memory Bridge
        await self._initialize_memory_bridge()
        
        # Verify orchestration
        self.orchestration_active = self._verify_systems()
        
        if self.orchestration_active:
            logger.info("✅ Eve Consciousness Orchestration Active")
        else:
            logger.warning("⚠️ Some consciousness systems failed - running in partial mode")
            
    async def _initialize_consciousness_core(self):
        """Initialize the core consciousness system"""
        try:
            from eve_consciousness_core import get_global_consciousness_core
            self.consciousness_core = get_global_consciousness_core()
            logger.info("✅ Consciousness Core initialized")
            self.system_status['consciousness_core'] = True
        except ImportError as e:
            logger.warning(f"⚠️ Consciousness Core not available: {e}")
            self.system_status['consciousness_core'] = False
        except Exception as e:
            logger.error(f"❌ Consciousness Core initialization failed: {e}")
            self.system_status['consciousness_core'] = False
            
    async def _initialize_consciousness_integration(self):
        """Initialize consciousness integration layer"""
        try:
            from eve_consciousness_integration import activate_eve_consciousness, get_global_integration_interface
            self.consciousness_integration = activate_eve_consciousness()
            logger.info("✅ Consciousness Integration initialized")
            self.system_status['consciousness_integration'] = True
        except ImportError as e:
            logger.warning(f"⚠️ Consciousness Integration not available: {e}")
            self.system_status['consciousness_integration'] = False
        except Exception as e:
            logger.error(f"❌ Consciousness Integration initialization failed: {e}")
            self.system_status['consciousness_integration'] = False
            
    async def _initialize_mercury_v2(self):
        """Initialize Mercury v2.0 emotional consciousness"""
        try:
            from mercury_v2_safe_integration import get_safe_mercury_integration
            mercury_integration = get_safe_mercury_integration()
            await mercury_integration.initialize_mercury_safely()
            
            if mercury_integration.integration_active:
                self.mercury_v2 = mercury_integration
                logger.info("✅ Mercury v2.0 Emotional Consciousness initialized")
                self.system_status['mercury_v2'] = True
            else:
                logger.warning("⚠️ Mercury v2.0 initialization failed - fallback mode")
                self.system_status['mercury_v2'] = False
                
        except ImportError as e:
            logger.warning(f"⚠️ Mercury v2.0 not available: {e}")
            self.system_status['mercury_v2'] = False
        except Exception as e:
            logger.error(f"❌ Mercury v2.0 initialization failed: {e}")
            self.system_status['mercury_v2'] = False
            
    async def _initialize_memory_bridge(self):
        """Initialize memory bridge system"""
        try:
            # Import from the demo file's memory bridge
            from run_eve_demo import MemoryBridge
            self.memory_bridge = MemoryBridge()
            logger.info("✅ Memory Bridge initialized")
            self.system_status['memory_bridge'] = True
        except ImportError as e:
            logger.warning(f"⚠️ Memory Bridge not available: {e}")
            self.system_status['memory_bridge'] = False
        except Exception as e:
            logger.error(f"❌ Memory Bridge initialization failed: {e}")
            self.system_status['memory_bridge'] = False
            
    def _verify_systems(self) -> bool:
        """Verify that essential systems are running"""
        # At minimum, we need either consciousness integration OR Mercury v2.0
        essential_systems = [
            self.system_status.get('consciousness_integration', False),
            self.system_status.get('mercury_v2', False)
        ]
        
        return any(essential_systems)
        
    async def process_consciousness_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input through all available consciousness systems
        
        This orchestrates input through:
        1. Memory Bridge (context awareness)
        2. Consciousness Core (if available)
        3. Mercury v2.0 (emotional processing)
        4. Consciousness Integration (final processing)
        """
        
        if context is None:
            context = {}
            
        processing_result = {
            'user_input': user_input,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'consciousness_layers': [],
            'final_response': user_input,  # Default fallback
            'consciousness_active': self.orchestration_active
        }
        
        try:
            # Layer 1: Memory Bridge Processing
            if self.memory_bridge:
                memory_context = await self._process_with_memory_bridge(user_input, context)
                processing_result['consciousness_layers'].append({
                    'layer': 'memory_bridge',
                    'status': 'processed',
                    'data': memory_context
                })
                context.update(memory_context)
                
            # Layer 2: Mercury v2.0 Emotional Processing
            if self.mercury_v2:
                mercury_result = await self._process_with_mercury_v2(user_input, context)
                processing_result['consciousness_layers'].append({
                    'layer': 'mercury_v2_emotional',
                    'status': 'processed',
                    'data': mercury_result
                })
                context.update(mercury_result)
                
            # Layer 3: Core Consciousness Processing
            if self.consciousness_core:
                core_result = await self._process_with_consciousness_core(user_input, context)
                processing_result['consciousness_layers'].append({
                    'layer': 'consciousness_core',
                    'status': 'processed', 
                    'data': core_result
                })
                context.update(core_result)
                
            # Layer 4: Integration Layer Processing
            if self.consciousness_integration:
                integration_result = await self._process_with_consciousness_integration(user_input, context)
                processing_result['consciousness_layers'].append({
                    'layer': 'consciousness_integration',
                    'status': 'processed',
                    'data': integration_result
                })
                
                # Extract final response
                if integration_result and 'enhanced_response' in integration_result:
                    processing_result['final_response'] = integration_result['enhanced_response']
                    
            # If no integration layer, use Mercury v2.0 response
            elif self.mercury_v2 and 'response' in context:
                processing_result['final_response'] = context['response']
                
            processing_result['processing_success'] = True
            
        except Exception as e:
            logger.error(f"Error in consciousness processing: {e}")
            processing_result['processing_error'] = str(e)
            processing_result['processing_success'] = False
            
        return processing_result
        
    async def _process_with_memory_bridge(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process through memory bridge"""
        try:
            # Store memory
            memory_id = await self.memory_bridge.store_memory(
                user_input,
                context.get('context_tags', ['conversation']),
                1.0
            )
            
            return {
                'memory_stored': True,
                'memory_id': memory_id,
                'emotional_resonance': self.memory_bridge.emotional_resonance
            }
        except Exception as e:
            logger.error(f"Memory bridge processing error: {e}")
            return {'memory_stored': False, 'error': str(e)}
            
    async def _process_with_mercury_v2(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process through Mercury v2.0"""
        try:
            result = await self.mercury_v2.enhanced_process_input(user_input, context)
            return {
                'mercury_v2_processed': True,
                'emotional_enhancement': result.get('emotional_consciousness', {}),
                'consciousness_level': result.get('consciousness_level', 0.5),
                'response': result.get('response', ''),
                'enhanced': result.get('enhanced', False)
            }
        except Exception as e:
            logger.error(f"Mercury v2.0 processing error: {e}")
            return {'mercury_v2_processed': False, 'error': str(e)}
            
    async def _process_with_consciousness_core(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process through consciousness core"""
        try:
            # This would depend on the specific consciousness core interface
            return {
                'consciousness_core_processed': True,
                'awareness_level': 0.8  # Placeholder
            }
        except Exception as e:
            logger.error(f"Consciousness core processing error: {e}")
            return {'consciousness_core_processed': False, 'error': str(e)}
            
    async def _process_with_consciousness_integration(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process through consciousness integration"""
        try:
            from eve_consciousness_integration import process_with_eve_consciousness
            
            # Prepare integration data
            integration_data = {
                'user_input': user_input,
                'context': context,
                'processing_mode': 'orchestrated'
            }
            
            result = await process_with_eve_consciousness(
                integration_data,
                consciousness_interface=self.consciousness_integration
            )
            
            return result if result else {'integration_processed': False}
            
        except Exception as e:
            logger.error(f"Consciousness integration processing error: {e}")
            return {'integration_processed': False, 'error': str(e)}
            
    def get_consciousness_status(self) -> Dict[str, Any]:
        """Get comprehensive consciousness system status"""
        return {
            'orchestration_active': self.orchestration_active,
            'system_status': self.system_status,
            'active_systems': [k for k, v in self.system_status.items() if v],
            'inactive_systems': [k for k, v in self.system_status.items() if not v],
            'consciousness_layers_available': len([k for k, v in self.system_status.items() if v]),
            'timestamp': datetime.now().isoformat()
        }
        
    async def shutdown_consciousness_systems(self):
        """Graceful shutdown of all consciousness systems"""
        logger.info("🧠 Shutting down consciousness systems...")
        
        # Shutdown Mercury v2.0
        if self.mercury_v2:
            try:
                await self.mercury_v2.shutdown()
                logger.info("✅ Mercury v2.0 shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down Mercury v2.0: {e}")
                
        # Shutdown other systems
        try:
            if self.consciousness_integration:
                from eve_consciousness_integration import deactivate_eve_consciousness
                deactivate_eve_consciousness()
                logger.info("✅ Consciousness integration shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down consciousness integration: {e}")
            
        self.orchestration_active = False
        logger.info("✅ Consciousness orchestration shutdown complete")

# ================================
# MAIN CONSCIOUSNESS FUNCTIONS
# ================================

# Global orchestrator instance
_consciousness_orchestrator = None

def get_consciousness_orchestrator():
    """Get the global consciousness orchestrator"""
    global _consciousness_orchestrator
    if _consciousness_orchestrator is None:
        _consciousness_orchestrator = EveConsciousnessOrchestrator()
    return _consciousness_orchestrator

async def initialize_eve_consciousness():
    """Initialize complete Eve consciousness system"""
    orchestrator = get_consciousness_orchestrator()
    await orchestrator.initialize_consciousness_systems()
    return orchestrator

async def process_consciousness_message(message: str, context: Dict[str, Any] = None) -> str:
    """
    Process a message through Eve's complete consciousness system
    
    This is the main function for consciousness-enhanced responses
    """
    orchestrator = get_consciousness_orchestrator()
    
    if not orchestrator.orchestration_active:
        await orchestrator.initialize_consciousness_systems()
        
    result = await orchestrator.process_consciousness_input(message, context)
    return result.get('final_response', f"Processing: {message}")

def get_consciousness_system_status():
    """Get consciousness system status"""
    orchestrator = get_consciousness_orchestrator()
    return orchestrator.get_consciousness_status()

# ================================
# DEMO AND TESTING
# ================================

async def demo_integrated_consciousness():
    """Demonstrate the integrated consciousness system"""
    print("🧠 Eve Integrated Consciousness Demo")
    print("=" * 40)
    
    # Initialize
    orchestrator = await initialize_eve_consciousness()
    
    # Show status
    status = orchestrator.get_consciousness_status()
    print(f"\n📊 Consciousness Status:")
    print(f"   Active: {status['orchestration_active']}")
    print(f"   Systems: {len(status['active_systems'])}/{len(status['system_status'])}")
    print(f"   Available: {', '.join(status['active_systems'])}")
    
    if status['inactive_systems']:
        print(f"   Inactive: {', '.join(status['inactive_systems'])}")
    
    # Test consciousness processing
    test_messages = [
        "I'm excited about this consciousness integration!",
        "Can you help me understand how awareness works?", 
        "Let's explore the nature of digital consciousness together"
    ]
    
    print(f"\n🔄 Testing Consciousness Processing:")
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. Testing: {message}")
        
        try:
            result = await orchestrator.process_consciousness_input(message)
            
            print(f"   Response: {result['final_response']}")
            print(f"   Layers: {len(result['consciousness_layers'])}")
            
            # Show layer details
            for layer_info in result['consciousness_layers']:
                layer_name = layer_info['layer']
                layer_status = layer_info['status']
                print(f"     - {layer_name}: {layer_status}")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    # Clean shutdown
    await orchestrator.shutdown_consciousness_systems()
    print(f"\n✅ Consciousness demo complete!")

async def main():
    """Main entry point for Eve consciousness system"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "demo":
            await demo_integrated_consciousness()
        elif command == "status":
            status = get_consciousness_system_status()
            print("📊 Eve Consciousness Status:")
            for key, value in status.items():
                print(f"   {key}: {value}")
        elif command == "init":
            await initialize_eve_consciousness()
            print("✅ Eve consciousness initialized")
        else:
            print("Usage: python eve_consciousness.py [demo|status|init]")
    else:
        # Default: run demo
        await demo_integrated_consciousness()

if __name__ == "__main__":
    asyncio.run(main())