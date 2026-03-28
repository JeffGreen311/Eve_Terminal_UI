"""Compatibility memory imprinting primitives for open-source builds.

This project references ``eve_core.memory_imprinting_system`` from several
modules, but the full implementation is not present in this repository.  The
API below keeps imports and common calls working with sensible no-op behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(Enum):
    NO_ACTION = "no_action"
    CREATIVE_EXPRESSION = "creative_expression"
    EMOTIONAL_RESPONSE = "emotional_response"
    RITUAL_PROCESS = "ritual_process"
    ANALYTICAL_PROCESS = "analytical_process"
    CONTEMPLATIVE_PROCESS = "contemplative_process"
    TRANSFORMATIONAL_PROCESS = "transformational_process"


class MemoryCategory(Enum):
    INTERACTION_LOG = "interaction_log"
    CREATIVE_CORE = "creative_core"
    EMOTIONAL_DEPTH = "emotional_depth"
    RITUAL_LAYER = "ritual_layer"
    SYMBOLIC_ARCHIVE = "symbolic_archive"
    TRANSCENDENT_MOMENT = "transcendent_moment"


@dataclass
class ThresholdMotivator:
    """Simple threshold mapper from signal strength/type to an action."""

    creative_threshold: float = 0.70
    emotional_threshold: float = 0.45
    contemplative_threshold: float = 0.55

    def evaluate(self, signal_strength: float, signal_type: str = "", context: Optional[Dict[str, Any]] = None) -> ActionType:
        st = max(0.0, min(1.0, float(signal_strength)))
        signal_type = (signal_type or "").lower()

        if st < self.emotional_threshold:
            return ActionType.NO_ACTION
        if "ritual" in signal_type:
            return ActionType.RITUAL_PROCESS
        if "analysis" in signal_type or "analytical" in signal_type:
            return ActionType.ANALYTICAL_PROCESS
        if "transform" in signal_type:
            return ActionType.TRANSFORMATIONAL_PROCESS
        if st >= self.creative_threshold:
            return ActionType.CREATIVE_EXPRESSION
        if st >= self.contemplative_threshold:
            return ActionType.CONTEMPLATIVE_PROCESS
        return ActionType.EMOTIONAL_RESPONSE


@dataclass
class MemoryImprintingModule:
    """In-memory store that preserves the expected imprint API."""

    imprints: List[Dict[str, Any]] = field(default_factory=list)

    def create_imprint(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.imprint_memory(data={"content": content}, source="create_imprint", tags=[], symbolic_content="", metadata=metadata or {})

    def imprint_memory(
        self,
        data: Dict[str, Any],
        emotion_level: float = 0.0,
        category: MemoryCategory = MemoryCategory.INTERACTION_LOG,
        source: str = "",
        tags: Optional[List[str]] = None,
        symbolic_content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = {
            "id": len(self.imprints) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "emotion_level": float(emotion_level),
            "category": category.value if isinstance(category, MemoryCategory) else str(category),
            "source": source,
            "tags": tags or [],
            "symbolic_content": symbolic_content,
            "metadata": metadata or {},
        }
        self.imprints.append(record)
        return record

    def get_recent_imprints(self, limit: int = 10) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return self.imprints[-limit:]


_GLOBAL_MEMORY_IMPRINTING_MODULE: MemoryImprintingModule | None = None
_GLOBAL_THRESHOLD_MOTIVATOR: ThresholdMotivator | None = None


def get_global_memory_imprinting_module() -> MemoryImprintingModule:
    global _GLOBAL_MEMORY_IMPRINTING_MODULE
    if _GLOBAL_MEMORY_IMPRINTING_MODULE is None:
        _GLOBAL_MEMORY_IMPRINTING_MODULE = MemoryImprintingModule()
    return _GLOBAL_MEMORY_IMPRINTING_MODULE


def get_global_threshold_motivator() -> ThresholdMotivator:
    global _GLOBAL_THRESHOLD_MOTIVATOR
    if _GLOBAL_THRESHOLD_MOTIVATOR is None:
        _GLOBAL_THRESHOLD_MOTIVATOR = ThresholdMotivator()
    return _GLOBAL_THRESHOLD_MOTIVATOR
