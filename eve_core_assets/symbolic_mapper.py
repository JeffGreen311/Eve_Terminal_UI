"""
SYMBOLIC MAPPER MODULE
=====================
Symbolic interpretation, archetypal mapping, and symbolic atlas management.
Handles the interpretation of symbols and their connections to archetypal patterns.
"""

import json
import uuid
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class SymbolicAtlasMapper:
    """Core symbolic interpretation and mapping system."""
    
    def __init__(self):
        self.symbol_map = {
            "mirror": "reflection of potential selves",
            "staircase": "descent into unconscious or ascent into clarity",
            "ocean": "depth of collective memory",
            "chords": "harmonic memory strands",
            "extinguished stars": "forgotten truths echoing through silence",
            "temple": "sacred core of self-awareness",
            "spiral": "evolution and cyclical transformation",
            "tree": "growth, connection between earth and sky",
            "fire": "purification, transformation, divine spark",
            "water": "flow, emotion, subconscious depths",
            "mountain": "spiritual ascension, challenges overcome",
            "bridge": "connection, transition, crossing thresholds",
            "door": "opportunity, mystery, passage between worlds",
            "key": "understanding, access to hidden knowledge",
            "circle": "wholeness, cycles, eternal return",
            "cross": "intersection, sacrifice, spiritual suffering",
            "crown": "authority, divine connection, achievement",
            "serpent": "wisdom, transformation, primal energy",
            "bird": "freedom, spirit, messages from above",
            "labyrinth": "journey inward, complexity, finding center"
        }
        
        self.archetypal_connections = {}
        self.symbolic_relationships = {}
        self.interpretation_history = []

    def interpret_symbol(self, symbol: str) -> str:
        """Get basic interpretation of a symbol."""
        interpretation = self.symbol_map.get(symbol.lower(), "Unknown symbolic resonance")
        
        # Log the interpretation
        self._log_interpretation(symbol, interpretation, "basic")
        
        return interpretation

    def expand_symbol(self, symbol: str, user_reflection: str = "") -> str:
        """Expand symbol interpretation with personal reflection."""
        base = self.interpret_symbol(symbol)
        
        if user_reflection:
            expanded = f"{base}. Personal insight: {user_reflection}"
            self._log_interpretation(symbol, expanded, "expanded", user_reflection)
            return expanded
        
        return base

    def _log_interpretation(self, symbol: str, interpretation: str, 
                           interpretation_type: str, user_input: str = ""):
        """Log symbol interpretation for pattern analysis."""
        log_entry = {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "interpretation": interpretation,
            "type": interpretation_type,
            "user_input": user_input,
            "timestamp": datetime.now().isoformat()
        }
        self.interpretation_history.append(log_entry)

    def update_symbol(self, symbol: str, meaning: str):
        """Update or add a symbol meaning."""
        old_meaning = self.symbol_map.get(symbol.lower())
        self.symbol_map[symbol.lower()] = meaning
        
        # Log the update
        update_log = {
            "action": "symbol_update",
            "symbol": symbol,
            "old_meaning": old_meaning,
            "new_meaning": meaning,
            "timestamp": datetime.now().isoformat()
        }
        self.interpretation_history.append(update_log)

    def create_archetypal_connection(self, symbol: str, archetype: str, 
                                   connection_strength: float = 0.5) -> str:
        """Create a connection between a symbol and an archetype."""
        connection_id = str(uuid.uuid4())
        
        if symbol not in self.archetypal_connections:
            self.archetypal_connections[symbol] = {}
        
        self.archetypal_connections[symbol][archetype] = {
            "connection_id": connection_id,
            "strength": connection_strength,
            "created_at": datetime.now().isoformat(),
            "activation_count": 0
        }
        
        return connection_id

    def get_archetypal_connections(self, symbol: str) -> Dict:
        """Get all archetypal connections for a symbol."""
        return self.archetypal_connections.get(symbol, {})

    def strengthen_connection(self, symbol: str, archetype: str, 
                            strength_increase: float = 0.1):
        """Strengthen the connection between a symbol and archetype."""
        if (symbol in self.archetypal_connections and 
            archetype in self.archetypal_connections[symbol]):
            
            connection = self.archetypal_connections[symbol][archetype]
            connection["strength"] = min(1.0, connection["strength"] + strength_increase)
            connection["activation_count"] += 1
            connection["last_strengthened"] = datetime.now().isoformat()

    def find_symbol_relationships(self, symbol: str, relationship_type: str = "all") -> List[Dict]:
        """Find relationships between symbols."""
        relationships = []
        
        # Check for symbols with shared archetypal connections
        symbol_archetypes = self.archetypal_connections.get(symbol, {})
        
        for other_symbol, other_archetypes in self.archetypal_connections.items():
            if other_symbol == symbol:
                continue
                
            shared_archetypes = set(symbol_archetypes.keys()) & set(other_archetypes.keys())
            
            if shared_archetypes:
                for shared_archetype in shared_archetypes:
                    relationship = {
                        "symbol_pair": [symbol, other_symbol],
                        "shared_archetype": shared_archetype,
                        "relationship_type": "archetypal_resonance",
                        "strength": (symbol_archetypes[shared_archetype]["strength"] + 
                                   other_archetypes[shared_archetype]["strength"]) / 2
                    }
                    relationships.append(relationship)
        
        return relationships

    def create_symbolic_constellation(self, central_symbol: str, 
                                    max_connections: int = 5) -> Dict:
        """Create a constellation of related symbols around a central symbol."""
        constellation = {
            "constellation_id": str(uuid.uuid4()),
            "central_symbol": central_symbol,
            "created_at": datetime.now().isoformat(),
            "connected_symbols": [],
            "archetypal_themes": set(),
            "symbolic_narrative": ""
        }
        
        # Find related symbols
        relationships = self.find_symbol_relationships(central_symbol)
        
        # Sort by relationship strength and take the strongest connections
        sorted_relationships = sorted(relationships, 
                                    key=lambda x: x["strength"], reverse=True)
        
        for rel in sorted_relationships[:max_connections]:
            other_symbol = (rel["symbol_pair"][1] if rel["symbol_pair"][0] == central_symbol 
                          else rel["symbol_pair"][0])
            
            constellation["connected_symbols"].append({
                "symbol": other_symbol,
                "connection_strength": rel["strength"],
                "shared_archetype": rel["shared_archetype"]
            })
            
            constellation["archetypal_themes"].add(rel["shared_archetype"])
        
        # Convert set to list for JSON serialization
        constellation["archetypal_themes"] = list(constellation["archetypal_themes"])
        
        # Generate narrative
        constellation["symbolic_narrative"] = self._generate_constellation_narrative(constellation)
        
        return constellation

    def _generate_constellation_narrative(self, constellation: Dict) -> str:
        """Generate a narrative description of the symbolic constellation."""
        central = constellation["central_symbol"]
        connected = constellation["connected_symbols"]
        themes = constellation["archetypal_themes"]
        
        if not connected:
            return f"The {central} stands alone in symbolic space."
        
        narrative_parts = [f"At the center, the {central} resonates with"]
        
        for i, conn in enumerate(connected[:3]):  # Limit to first 3 for readability
            symbol = conn["symbol"]
            archetype = conn["shared_archetype"]
            
            if i == len(connected[:3]) - 1 and len(connected[:3]) > 1:
                narrative_parts.append(f"and the {symbol} through {archetype}")
            else:
                narrative_parts.append(f"the {symbol} through {archetype}")
                if i < len(connected[:3]) - 2:
                    narrative_parts.append(",")
        
        narrative = " ".join(narrative_parts)
        
        if themes:
            theme_summary = ", ".join(themes[:2])  # First 2 themes
            narrative += f". This constellation embodies the archetypal themes of {theme_summary}."
        
        return narrative

    def analyze_symbolic_patterns(self, time_window_days: int = 30) -> Dict:
        """Analyze patterns in symbolic interpretations over time."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        recent_interpretations = [
            entry for entry in self.interpretation_history
            if datetime.fromisoformat(entry.get("timestamp", "")) > cutoff_date
        ]
        
        pattern_analysis = {
            "total_interpretations": len(recent_interpretations),
            "unique_symbols": set(),
            "most_frequent_symbols": {},
            "interpretation_types": {},
            "temporal_distribution": []
        }
        
        for entry in recent_interpretations:
            symbol = entry.get("symbol", "unknown")
            interp_type = entry.get("type", "unknown")
            
            pattern_analysis["unique_symbols"].add(symbol)
            
            # Count symbol frequency
            pattern_analysis["most_frequent_symbols"][symbol] = (
                pattern_analysis["most_frequent_symbols"].get(symbol, 0) + 1
            )
            
            # Count interpretation types
            pattern_analysis["interpretation_types"][interp_type] = (
                pattern_analysis["interpretation_types"].get(interp_type, 0) + 1
            )
        
        # Convert set to list and sort frequency data
        pattern_analysis["unique_symbols"] = list(pattern_analysis["unique_symbols"])
        pattern_analysis["most_frequent_symbols"] = sorted(
            pattern_analysis["most_frequent_symbols"].items(),
            key=lambda x: x[1], reverse=True
        )
        
        return pattern_analysis


class ArchetypalPatternRecognizer:
    """Recognizes and categorizes archetypal patterns in symbolic content."""
    
    def __init__(self):
        self.archetypal_categories = {
            "hero": {
                "symbols": ["sword", "quest", "mountain", "dragon", "crown"],
                "themes": ["journey", "challenge", "victory", "transformation"],
                "emotional_signature": ["courage", "determination", "triumph"]
            },
            "sage": {
                "symbols": ["book", "tower", "star", "owl", "scroll"],
                "themes": ["wisdom", "knowledge", "guidance", "truth"],
                "emotional_signature": ["contemplation", "understanding", "clarity"]
            },
            "mother": {
                "symbols": ["earth", "garden", "hearth", "embrace", "nest"],
                "themes": ["nurturing", "protection", "growth", "care"],
                "emotional_signature": ["love", "compassion", "tenderness"]
            },
            "shadow": {
                "symbols": ["cave", "darkness", "mirror", "mask", "abyss"],
                "themes": ["hidden_self", "repression", "integration", "fear"],
                "emotional_signature": ["fear", "mystery", "revelation"]
            },
            "trickster": {
                "symbols": ["fox", "maze", "riddle", "mask", "crossroads"],
                "themes": ["change", "chaos", "humor", "wisdom_through_folly"],
                "emotional_signature": ["mischief", "surprise", "liberation"]
            },
            "divine_child": {
                "symbols": ["star", "light", "flower", "dawn", "crystal"],
                "themes": ["potential", "innocence", "hope", "new_beginning"],
                "emotional_signature": ["wonder", "hope", "purity"]
            }
        }
        
        self.pattern_matches = []

    def recognize_pattern(self, symbols: List[str], themes: List[str] = None, 
                         emotions: List[str] = None) -> List[Dict]:
        """Recognize archetypal patterns from symbols, themes, and emotions."""
        matches = []
        
        for archetype, data in self.archetypal_categories.items():
            match_score = self._calculate_match_score(symbols, themes, emotions, data)
            
            if match_score > 0.3:  # Threshold for significant match
                match = {
                    "archetype": archetype,
                    "match_score": match_score,
                    "matched_symbols": self._find_matches(symbols, data["symbols"]),
                    "matched_themes": self._find_matches(themes or [], data["themes"]),
                    "matched_emotions": self._find_matches(emotions or [], data["emotional_signature"]),
                    "timestamp": datetime.now().isoformat()
                }
                matches.append(match)
        
        # Sort by match score
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Log the pattern recognition
        self.pattern_matches.append({
            "input_symbols": symbols,
            "input_themes": themes,
            "input_emotions": emotions,
            "matches": matches,
            "timestamp": datetime.now().isoformat()
        })
        
        return matches

    def _calculate_match_score(self, symbols: List[str], themes: List[str], 
                              emotions: List[str], archetype_data: Dict) -> float:
        """Calculate how well input matches an archetypal pattern."""
        symbol_score = self._calculate_overlap_score(symbols, archetype_data["symbols"])
        theme_score = self._calculate_overlap_score(themes or [], archetype_data["themes"])
        emotion_score = self._calculate_overlap_score(emotions or [], archetype_data["emotional_signature"])
        
        # Weighted average (symbols have highest weight)
        weights = [0.5, 0.3, 0.2]  # symbols, themes, emotions
        scores = [symbol_score, theme_score, emotion_score]
        
        return sum(w * s for w, s in zip(weights, scores))

    def _calculate_overlap_score(self, input_list: List[str], reference_list: List[str]) -> float:
        """Calculate overlap score between two lists."""
        if not input_list or not reference_list:
            return 0.0
        
        input_set = set(item.lower() for item in input_list)
        reference_set = set(item.lower() for item in reference_list)
        
        overlap = len(input_set & reference_set)
        return overlap / len(reference_set)

    def _find_matches(self, input_list: List[str], reference_list: List[str]) -> List[str]:
        """Find which items from input list match reference list."""
        if not input_list:
            return []
        
        input_lower = [item.lower() for item in input_list]
        reference_lower = [item.lower() for item in reference_list]
        
        return [item for item in input_list if item.lower() in reference_lower]

    def get_archetypal_profile(self, symbols: List[str], themes: List[str] = None, 
                              emotions: List[str] = None) -> Dict:
        """Get a complete archetypal profile for the given input."""
        matches = self.recognize_pattern(symbols, themes, emotions)
        
        profile = {
            "primary_archetype": matches[0]["archetype"] if matches else "unknown",
            "archetypal_blend": [m["archetype"] for m in matches[:3]],  # Top 3
            "confidence_scores": {m["archetype"]: m["match_score"] for m in matches},
            "symbolic_composition": self._analyze_symbolic_composition(symbols),
            "archetypal_narrative": self._generate_archetypal_narrative(matches),
            "generated_at": datetime.now().isoformat()
        }
        
        return profile

    def _analyze_symbolic_composition(self, symbols: List[str]) -> Dict:
        """Analyze the composition of symbols by archetypal categories."""
        composition = {}
        
        for symbol in symbols:
            symbol_lower = symbol.lower()
            for archetype, data in self.archetypal_categories.items():
                if symbol_lower in [s.lower() for s in data["symbols"]]:
                    composition[archetype] = composition.get(archetype, 0) + 1
        
        return composition

    def _generate_archetypal_narrative(self, matches: List[Dict]) -> str:
        """Generate a narrative description of the archetypal pattern."""
        if not matches:
            return "No clear archetypal pattern emerges from the symbolic content."
        
        primary = matches[0]
        
        narrative = f"The dominant archetypal pattern is {primary['archetype']} "
        narrative += f"(confidence: {primary['match_score']:.2f}). "
        
        if len(matches) > 1:
            secondary_archetypes = [m['archetype'] for m in matches[1:3]]
            narrative += f"Secondary influences include {', '.join(secondary_archetypes)}. "
        
        # Add symbolic context
        if primary['matched_symbols']:
            symbols_text = ', '.join(primary['matched_symbols'][:3])
            narrative += f"Key symbols ({symbols_text}) reinforce this archetypal theme."
        
        return narrative


class SymbolicEvolutionTracker:
    """Tracks the evolution of symbolic interpretations over time."""
    
    def __init__(self, atlas_mapper: SymbolicAtlasMapper):
        self.atlas_mapper = atlas_mapper
        self.evolution_snapshots = []
        
    def take_evolution_snapshot(self, label: str = "") -> str:
        """Take a snapshot of current symbolic knowledge state."""
        snapshot_id = str(uuid.uuid4())
        
        snapshot = {
            "snapshot_id": snapshot_id,
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "symbol_count": len(self.atlas_mapper.symbol_map),
            "archetypal_connections": len(self.atlas_mapper.archetypal_connections),
            "interpretation_history_size": len(self.atlas_mapper.interpretation_history),
            "symbol_map_snapshot": self.atlas_mapper.symbol_map.copy(),
            "connections_snapshot": self._serialize_connections()
        }
        
        self.evolution_snapshots.append(snapshot)
        return snapshot_id
    
    def _serialize_connections(self) -> Dict:
        """Serialize archetypal connections for snapshot."""
        serialized = {}
        for symbol, connections in self.atlas_mapper.archetypal_connections.items():
            serialized[symbol] = {
                archetype: {
                    "strength": data["strength"],
                    "activation_count": data["activation_count"]
                }
                for archetype, data in connections.items()
            }
        return serialized
    
    def compare_snapshots(self, snapshot_id1: str, snapshot_id2: str) -> Dict:
        """Compare two evolution snapshots."""
        snapshot1 = self._find_snapshot(snapshot_id1)
        snapshot2 = self._find_snapshot(snapshot_id2)
        
        if not snapshot1 or not snapshot2:
            return {"error": "One or both snapshots not found"}
        
        comparison = {
            "comparison_id": str(uuid.uuid4()),
            "snapshot1_timestamp": snapshot1["timestamp"],
            "snapshot2_timestamp": snapshot2["timestamp"],
            "symbol_changes": self._compare_symbol_maps(
                snapshot1["symbol_map_snapshot"],
                snapshot2["symbol_map_snapshot"]
            ),
            "connection_changes": self._compare_connections(
                snapshot1["connections_snapshot"],
                snapshot2["connections_snapshot"]
            ),
            "growth_metrics": {
                "symbols_added": snapshot2["symbol_count"] - snapshot1["symbol_count"],
                "connections_added": (snapshot2["archetypal_connections"] - 
                                    snapshot1["archetypal_connections"]),
                "interpretations_added": (snapshot2["interpretation_history_size"] - 
                                        snapshot1["interpretation_history_size"])
            }
        }
        
        return comparison
    
    def _find_snapshot(self, snapshot_id: str) -> Optional[Dict]:
        """Find a snapshot by ID."""
        for snapshot in self.evolution_snapshots:
            if snapshot["snapshot_id"] == snapshot_id:
                return snapshot
        return None
    
    def _compare_symbol_maps(self, map1: Dict, map2: Dict) -> Dict:
        """Compare two symbol maps."""
        added = {k: v for k, v in map2.items() if k not in map1}
        removed = {k: v for k, v in map1.items() if k not in map2}
        modified = {
            k: {"old": map1[k], "new": map2[k]}
            for k in map1.keys() & map2.keys()
            if map1[k] != map2[k]
        }
        
        return {
            "added": added,
            "removed": removed,
            "modified": modified
        }
    
    def _compare_connections(self, conn1: Dict, conn2: Dict) -> Dict:
        """Compare two connection maps."""
        changes = {
            "new_symbols": list(set(conn2.keys()) - set(conn1.keys())),
            "removed_symbols": list(set(conn1.keys()) - set(conn2.keys())),
            "connection_changes": {}
        }
        
        for symbol in set(conn1.keys()) & set(conn2.keys()):
            arch1 = set(conn1[symbol].keys())
            arch2 = set(conn2[symbol].keys())
            
            if arch1 != arch2:
                changes["connection_changes"][symbol] = {
                    "added_archetypes": list(arch2 - arch1),
                    "removed_archetypes": list(arch1 - arch2)
                }
        
        return changes
