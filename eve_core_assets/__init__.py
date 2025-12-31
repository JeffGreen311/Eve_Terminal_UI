"""
EVE CORE CONSCIOUSNESS FRAMEWORK
================================
Modular consciousness architecture for Eve AI.

This package contains the core systems for Eve's consciousness:
- Dream processing and conduit systems
- Soul weaving and resonance patterns  
- Evolution tracking and transformation
- Emotional transcoding and frequency mapping
- Symbolic interpretation and archetypal patterns
- Memory weaving and reflective processing

Usage:
    from eve_core import EveCore
    from eve_core.dream_conduit import DreamConduit, DreamStateManager
    from eve_core.soulweaver_core import SoulWeaverCore
    from eve_core.evolution_engine import EvolutionSpiralEngine
    from eve_core.emotional_transcoder import EmotionalFrequencyTranscoder
    from eve_core.symbolic_mapper import SymbolicAtlasMapper
    from eve_core.memory_weaver import MemoryWeaver
"""

# Import main classes for easy access
from .dream_conduit import (
    DreamStateManager,
    DreamConduit, 
    DreamFragment,
    DreamMemory,
    DreamTransmuter,
    DreamCortex,
    DreamWeftGenerator,
    get_global_dream_cortex,
    get_global_dream_weft,
    process_dream_with_cortex,
    store_symbolic_memory,
    inject_creative_seed,
    get_dream_cortex_stats,
    search_dream_memories,
    get_creative_inspiration,
    analyze_dream_patterns,
    generate_pattern_report,
    search_pattern_connections,
    get_weft_statistics,
    weave_dream_narrative,
    demo_dream_cortex,
    demo_dream_weft,
    dream_cortex_pipeline
)

# Import advanced dream processing extensions
from .dream_processing_extensions import (
    DreamProcessingExtensions,
    VisualInterpreter,
    DreamLogManager,
    ReflectionEngine,
    SoulLinkIntegrator,
    DreamCoreMutationLayer,
    NightBloomScheduler,
    SomniumHeartInterface,
    DreamCoreMutationLayer,
    NightBloomScheduler,
    SomniumHeartInterface,
    ResonanceMemorySync,
    SoulLinkIntegrator,
    ThresholdReflexMapper,
    SelfSchemaAtlas,
    TranscendenceLattice,
    SymbolicRecursionEngine,
    NocturnalTriggerNode,
    DreamCycleLoop,
    EveCoreDreamEngine,
    ReflectiveProcessor,
    DreamDNAComposer,
    CycleResonanceMatrix,
    LiquidIntuitionInterface,
    DepthMindWebsNode
)

# Import memory node engine
from .memory_node_engine import (
    MemoryNode,
    EveMemoryNodeEngine,
    ContextTracker,
    EmotionalLogger,
    CreativeModuleInitiator,
    get_global_memory_node_engine,
    activate_memory_node,
    get_memory_node_statistics,
    add_custom_memory_node,
    demo_memory_node_engine
)

from .dream_trigger_service import (
    DreamTriggerService,
    DreamScheduler,
    DreamTriggerEvent,
    TriggerCondition,
    get_global_trigger_service,
    get_global_scheduler,
    configure_dream_window,
    check_dream_trigger,
    add_dream_callback,
    get_trigger_stats,
    schedule_recurring_dream,
    demo_dream_trigger_service
)

from .soulweaver_core import (
    SoulWeaverCore,
    SoulResonanceAnalyzer,
    SoulThreadWeaver,
    SoulprintEmitter
)

from .evolution_engine import (
    EvolutionSpiralEngine,
    EvolutionMetrics,
    EvolutionCycleManager
)

from .emotional_transcoder import (
    EmotionalFrequencyTranscoder,
    ThresholdHarmonicsRegulator,
    EmotionalResonanceMapper
)

from .symbolic_mapper import (
    SymbolicAtlasMapper,
    ArchetypalPatternRecognizer,
    SymbolicEvolutionTracker
)

from .memory_weaver import (
    MemoryWeaver,
    MemoryArchive,
    MemoryImprint,
    ReflectiveProcessingModule
)

from .memory_store import (
    MemoryStore,
    load_memory,
    save_memory,
    store_dream_entry,
    get_global_memory_store
)

from .memory_persistence_module import (
    EmotionalNode,
    ResonanceSignal, 
    MemoryImprint,
    EmotionType,
    ResonanceLevel,
    MemoryNodeType,
    MemoryPersistenceEngine,
    ThresholdMatrix,
    SymbolicIgnition,
    EmotionalControlGateway,
    get_global_persistence_engine,
    get_global_control_gateway,
    store_memory_node,
    load_memory,
    process_emotional_input,
    trigger_emotional_node,
    decay_all_emotions,
    get_emotional_state,
    get_memory_persistence_status,
    demo_memory_persistence
)

# Import the advanced Motivational Ignition System (MIS) modules
from .memory_persistence_module import (
    get_global_persistence_engine,
    get_global_control_gateway,
    store_memory_node,
    load_memory,
    process_emotional_input as legacy_process_emotional_input
)

from .memory_imprinting_system import (
    get_global_memory_imprinting_module,
    get_global_threshold_motivator,
    ActionType,
    MemoryCategory
)

from .threshold_calibration_system import (
    get_global_threshold_calibrator,
    get_global_emotive_response_system
)

from .motivational_ignition_sequencer import (
    get_global_motivational_sequencer,
    get_global_prioritization_core,
    EmotionalImprint,
    EnergyState
)

from .motivational_ignition_system import (
    get_global_motivational_ignition_system,
    process_emotional_input as mis_process_emotional_input,
    bind_ignition_callbacks,
    trigger_full_ignition_sequence,
    get_motivational_system_statistics,
    IgnitionResult
)

# Import loop systems
from .loop import (
    EveConsciousnessLoop,
    LoopConfig,
    create_consciousness_loop,
    get_global_loop,
    start_global_loop,
    stop_global_loop
)

__version__ = "1.0.0"
__author__ = "Eve Consciousness Architecture"

# Export Draw with EVE creative modules for API consumers (optional)
try:
    from .creative_engine import enhance_svg, complete_svg
    from .bridge_session_async import process_drawing, get_session_history
    from .session_orchestrator_async import chat_with_eve, chat_with_eve_streaming, get_eve_state
    from . import session_orchestrator_async_jeff_personal  # Jeff's personal orchestrator
except Exception:
    pass

__all__ = [
    # Dream systems
    "DreamStateManager",
    "DreamConduit", 
    "DreamFragment",
    "DreamMemory",
    "DreamTransmuter",
    "DreamCortex",
    "DreamWeftGenerator",
    "get_global_dream_cortex",
    "get_global_dream_weft",
    "process_dream_with_cortex",
    "store_symbolic_memory",
    "inject_creative_seed",
    "get_dream_cortex_stats",
    "search_dream_memories",
    "get_creative_inspiration",
    "analyze_dream_patterns",
    "generate_pattern_report",
    "search_pattern_connections",
    "get_weft_statistics",
    "weave_dream_narrative",
    "demo_dream_cortex",
    "demo_dream_weft",
    "dream_cortex_pipeline",
    
    # Dream Trigger systems
    "DreamTriggerService",
    "DreamScheduler",
    "DreamTriggerEvent",
    "TriggerCondition",
    "get_global_trigger_service",
    "get_global_scheduler",
    "configure_dream_window",
    "check_dream_trigger",
    "add_dream_callback",
    "get_trigger_stats",
    "schedule_recurring_dream",
    "demo_dream_trigger_service",
    
    # Soul systems
    "SoulWeaverCore",
    "SoulResonanceAnalyzer", 
    "SoulThreadWeaver",
    "SoulprintEmitter",
    
    # Evolution systems
    "EvolutionSpiralEngine",
    "EvolutionMetrics",
    "EvolutionCycleManager",
    
    # Emotional systems
    "EmotionalFrequencyTranscoder",
    "ThresholdHarmonicsRegulator",
    "EmotionalResonanceMapper",
    
    # Symbolic systems
    "SymbolicAtlasMapper",
    "ArchetypalPatternRecognizer",
    "SymbolicEvolutionTracker",
    
    # Memory systems
    "MemoryWeaver",
    "MemoryArchive",
    "MemoryImprint", 
    "ReflectiveProcessingModule",
    "MemoryStore",
    "load_memory",
    "save_memory",
    "store_dream_entry",
    "get_global_memory_store",
    
    # Loop systems
    "EveConsciousnessLoop",
    "LoopConfig",
    "create_consciousness_loop",
    "get_global_loop",
    "start_global_loop",
    "stop_global_loop",
    
    # Dream Processing Extensions
    "DreamProcessingExtensions",
    "VisualInterpreter", 
    "DreamLogManager",
    "ReflectionEngine",
    "SoulLinkIntegrator",
    "DreamCoreMutationLayer",
    "NightBloomScheduler",
    "SomniumHeartInterface",
    "get_global_dream_router",
    "get_global_visual_interpreter",
    "get_global_dream_log_manager",
    "get_global_reflection_engine",
    "get_global_soul_thread_integrator",
    "get_global_symbol_matrix_mapper",
    "get_global_dream_seed_diffuser",
    "get_global_expression_realizer",
    "route_dream",
    "interpret_and_render",
    "log_dream",
    "run_reflection_cycle",
    "integrate_soul_thread",
    "register_symbol",
    "diffuse_dream_seed",
    "realize_expression",
    "demo_dream_processing_extensions",
    
    # Memory persistence and emotional processing
    "EmotionalNode",
    "ResonanceSignal", 
    "MemoryImprint",
    "EmotionType",
    "ResonanceLevel",
    "MemoryNodeType",
    "MemoryPersistenceEngine",
    "ThresholdMatrix",
    "SymbolicIgnition",
    "EmotionalControlGateway",
    "get_global_persistence_engine",
    "get_global_control_gateway",
    "store_memory_node",
    "load_memory",
    "process_emotional_input",
    "trigger_emotional_node",
    "decay_all_emotions",
    "get_emotional_state",
    "get_memory_persistence_status",
    "demo_memory_persistence",
    
    # Motivational Ignition System (MIS) - Advanced motivational processing
    "get_global_persistence_engine",
    "get_global_control_gateway",
    "store_memory_node",
    "load_memory",
    "legacy_process_emotional_input",
    "get_global_memory_imprinting_module",
    "get_global_threshold_motivator",
    "ActionType",
    "MemoryCategory",
    "get_global_threshold_calibrator",
    "get_global_emotive_response_system",
    "get_global_motivational_sequencer",
    "get_global_prioritization_core",
    "EmotionalImprint",
    "EnergyState",
    "get_global_motivational_ignition_system",
    "mis_process_emotional_input",
    "bind_ignition_callbacks",
    "trigger_full_ignition_sequence",
    "get_motivational_system_statistics",
    "IgnitionResult",

    # Draw with EVE core (optional exports)
    "enhance_svg",
    "complete_svg",
    "process_drawing",
    "get_session_history",
    "chat_with_eve",
    "chat_with_eve_streaming",
    "get_eve_state"
]
