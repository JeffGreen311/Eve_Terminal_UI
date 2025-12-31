# 🧠 MEMORY PERSISTENCE MODULE
# Advanced emotional memory processing and behavioral triggering system

from .soulforge_memory import SoulforgeMemory, get_global_soulforge_memory
from .emotional_intuitive_engine import EmotionalIntuitiveEngine, get_global_emotional_engine
from .motivational_ignition_core import MotivationalIgnitionCore, get_global_ignition_core
from .eve_behavior_core import *
from .resonance_gateway import ResonanceGateway, get_global_resonance_gateway
from .symbolic_ignition import SymbolicIgnition, get_global_symbolic_ignition
from .emotional_memory_imprint import EmotionalMemoryImprint, get_global_emotional_memory
from .emotional_control_gateway import EmotionalControlGateway, get_global_emotional_gateway, process_emotion_through_gateway

__all__ = [
    'SoulforgeMemory', 'get_global_soulforge_memory',
    'EmotionalIntuitiveEngine', 'get_global_emotional_engine',
    'MotivationalIgnitionCore', 'get_global_ignition_core',
    'ResonanceGateway', 'get_global_resonance_gateway',
    'SymbolicIgnition', 'get_global_symbolic_ignition',
    'EmotionalMemoryImprint', 'get_global_emotional_memory',
    'EmotionalControlGateway', 'get_global_emotional_gateway', 'process_emotion_through_gateway'
]

print("✅ Eve Memory Persistence and Emotional Resonance modules loaded")
