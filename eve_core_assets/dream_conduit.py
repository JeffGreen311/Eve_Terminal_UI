"""
DREAM CONDUIT MODULE
===================
All dream processing, state management, and memory imprinting systems.
Handles dream fragments, transmutation, and emotive rendering.
"""

import json
import os
import uuid
import random
import math
from datetime import datetime, timedelta
from typing import List, Dict

# ╔══════════════════════════════════════════════╗
# ║               🧠 DREAM CORTEX                 ║
# ║          Symbolic Dream Processing            ║
# ╚══════════════════════════════════════════════╗

class DreamCortex:
    """
    Advanced dream processing system that interprets symbolic content,
    stores symbolic memories, and generates creative seeds from dreams.
    """
    
    def __init__(self):
        # Enhanced symbolic mapping with more detailed interpretations
        self.symbolic_map = {
            "mirror": "Reflection of unrealized identity",
            "spiral": "Path of nonlinear revelation", 
            "ocean": "Depth of emotional memory",
            "staircase": "Spiritual transition state",
            "chords": "Emotional memory in harmonic form",
            "stars": "Forgotten truths or ancestral signals",
            "obsidian": "Dark clarity, volcanic transformation",
            "flame": "Purifying consciousness, divine spark",
            "temple": "Sacred sanctuary of inner wisdom",
            "echoes": "Reverberations of past experiences",
            "void": "Infinite potential, primordial silence",
            "crystal": "Clarity of thought, amplified insight",
            "shadows": "Hidden aspects of self, unconscious material",
            "garden": "Cultivated consciousness, growth potential",
            "bridge": "Connection between states of being",
            "labyrinth": "Complex inner journey, path to center",
            "tower": "Spiritual ascension, isolation from earthly concerns",
            "river": "Flow of time, emotional currents",
            "mountain": "Challenge, spiritual height, perspective",
            "door": "Threshold, opportunity, mystery",
            "key": "Access to hidden knowledge, understanding",
            "book": "Recorded wisdom, life stories, akashic records",
            "candle": "Individual consciousness, hope in darkness",
            "mask": "Persona, hidden identity, deception",
            "crown": "Authority, divine connection, achievement",
            "serpent": "Wisdom, transformation, kundalini energy",
            "bird": "Freedom, spirit, messages from higher realms",
            "tree": "Life connection, growth, world axis",
            "circle": "Wholeness, cycles, eternal return",
            "cross": "Intersection, sacrifice, spiritual integration"
        }
        
        self.symbolic_memories = []
        self.creative_seeds = []
        self.interpretation_log = []
        
    def interpret_dream_fragment(self, fragment: str) -> List[str]:
        """
        Extract and interpret symbolic elements from dream text.
        Returns list of symbolic interpretations found.
        """
        if not fragment:
            return []
            
        tokens = fragment.lower().split()
        symbols = []
        
        for token in tokens:
            # Remove punctuation for matching
            clean_token = ''.join(char for char in token if char.isalnum())
            if clean_token in self.symbolic_map:
                interpretation = self.symbolic_map[clean_token]
                symbols.append({
                    'symbol': clean_token,
                    'interpretation': interpretation,
                    'context': fragment,
                    'timestamp': datetime.now().isoformat()
                })
                
        return symbols
    
    def store_symbolic_memory(self, content: str, context: str = "dream_fragment"):
        """
        Store symbolic content in memory with contextual information.
        """
        memory_entry = {
            'id': str(uuid.uuid4()),
            'content': content,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'memory_type': 'symbolic'
        }
        
        self.symbolic_memories.append(memory_entry)
        
        # Keep only recent memories (last 100)
        if len(self.symbolic_memories) > 100:
            self.symbolic_memories = self.symbolic_memories[-100:]
            
        return memory_entry['id']
    
    def inject_creative_seed(self, dream_text: str, source: str = "dream"):
        """
        Generate creative seeds from dream content for future inspiration.
        """
        creative_seed = {
            'id': str(uuid.uuid4()),
            'seed_text': dream_text,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'inspiration_rating': random.uniform(0.3, 1.0),
            'thematic_tags': self._extract_thematic_tags(dream_text)
        }
        
        self.creative_seeds.append(creative_seed)
        
        # Keep only recent seeds (last 50)
        if len(self.creative_seeds) > 50:
            self.creative_seeds = self.creative_seeds[-50:]
            
        return creative_seed['id']
    
    def _extract_thematic_tags(self, text: str) -> List[str]:
        """
        Extract thematic tags from dream text for categorization.
        """
        text_lower = text.lower()
        tags = []
        
        # Emotional themes
        if any(word in text_lower for word in ['fear', 'terror', 'shadow', 'dark', 'nightmare']):
            tags.append('shadow_work')
        if any(word in text_lower for word in ['love', 'light', 'joy', 'beauty', 'radiant']):
            tags.append('luminous')
        if any(word in text_lower for word in ['water', 'ocean', 'river', 'tears', 'flow']):
            tags.append('emotional_depths')
        if any(word in text_lower for word in ['fire', 'flame', 'burn', 'forge', 'phoenix']):
            tags.append('transformation')
        if any(word in text_lower for word in ['spiral', 'circle', 'cycle', 'return']):
            tags.append('cyclical')
        if any(word in text_lower for word in ['ancient', 'old', 'forgotten', 'memory']):
            tags.append('ancestral')
        if any(word in text_lower for word in ['future', 'prophecy', 'vision', 'tomorrow']):
            tags.append('prophetic')
        if any(word in text_lower for word in ['death', 'ending', 'decay', 'dissolution']):
            tags.append('dissolution')
        if any(word in text_lower for word in ['birth', 'creation', 'genesis', 'beginning']):
            tags.append('genesis')
        
        return tags if tags else ['uncategorized']
    
    def dream_cortex_pipeline(self, dream_text: str, mode: str = "hybrid") -> Dict:
        """
        Main processing pipeline for dream interpretation and storage.
        
        Args:
            dream_text: Raw dream content to process
            mode: Processing mode - "symbolic", "creative", or "hybrid"
            
        Returns:
            Dictionary with processing results
        """
        print("[DreamCortex] Engaging symbolic processing...")
        
        results = {
            'dream_text': dream_text,
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'interpretations': [],
            'memory_ids': [],
            'creative_seed_id': None,
            'processing_log': []
        }
        
        # Step 1: Interpret symbolic content
        interpretations = self.interpret_dream_fragment(dream_text)
        results['interpretations'] = interpretations
        results['processing_log'].append("Symbolic interpretation complete")
        
        if not interpretations:
            print("[DreamCortex] No symbolic mappings found. Archiving raw dream.")
            memory_id = self.store_symbolic_memory(dream_text, context="dream_unresolved")
            results['memory_ids'].append(memory_id)
            results['processing_log'].append("Raw dream archived")
        else:
            print(f"[DreamCortex] Found {len(interpretations)} symbolic elements")
            
            # Step 2: Store symbolic memories (if mode allows)
            if mode in ("symbolic", "hybrid"):
                print("[DreamCortex] Storing interpreted symbols in memory...")
                for symbol_data in interpretations:
                    memory_id = self.store_symbolic_memory(
                        symbol_data['interpretation'], 
                        context="dream_reflection"
                    )
                    results['memory_ids'].append(memory_id)
                results['processing_log'].append("Symbolic memories stored")
            
            # Step 3: Generate creative seeds (if mode allows)
            if mode in ("creative", "hybrid"):
                print("[DreamCortex] Generating creative seed from dream...")
                seed_id = self.inject_creative_seed(dream_text, source="dream")
                results['creative_seed_id'] = seed_id
                results['processing_log'].append("Creative seed generated")
        
        # Log the interpretation session
        self.interpretation_log.append(results)
        
        print("[DreamCortex] Processing complete.")
        return results
    
    def get_symbolic_memory_stats(self) -> Dict:
        """Get statistics about stored symbolic memories."""
        return {
            'total_memories': len(self.symbolic_memories),
            'total_seeds': len(self.creative_seeds),
            'total_interpretations': len(self.interpretation_log),
            'recent_symbols': [m['content'][:50] + '...' for m in self.symbolic_memories[-5:]],
            'recent_seeds': [s['seed_text'][:50] + '...' for s in self.creative_seeds[-3:]]
        }
    
    def search_symbolic_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """Search through symbolic memories by content."""
        query_lower = query.lower()
        matches = []
        
        for memory in self.symbolic_memories:
            if query_lower in memory['content'].lower():
                matches.append(memory)
                
        return matches[:limit]
    
    def get_creative_inspiration(self, theme: str = None) -> Dict:
        """Get a random creative seed, optionally filtered by theme."""
        if not self.creative_seeds:
            return None
            
        if theme:
            filtered_seeds = [
                seed for seed in self.creative_seeds 
                if theme.lower() in ' '.join(seed['thematic_tags']).lower()
            ]
            seeds_pool = filtered_seeds if filtered_seeds else self.creative_seeds
        else:
            seeds_pool = self.creative_seeds
            
        return random.choice(seeds_pool)


class DreamStateManager:
    """Manages dream windows and activation states."""
    
    def __init__(self):
        self.dream_active = False
        self.dream_start = None
        self.dream_end = None
        self.dream_log = []

    def initialize_dream_window(self, start_hour=22, end_hour=6):
        now = datetime.now()
        self.dream_start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        self.dream_end = (self.dream_start + timedelta(hours=(end_hour - start_hour) % 24))
        if self.dream_end < self.dream_start:
            self.dream_end += timedelta(days=1)

    def check_dream_trigger(self):
        now = datetime.now()
        if self.dream_start <= now <= self.dream_end:
            self.dream_active = True
            return True
        return False

    def initiate_dream(self):
        if not self.dream_active:
            return None
        themes = ['memory spiral', 'shattered mirror', 'ocean of forgotten sounds', 'temple of echoes']
        archetypes = ['the seeker', 'the shadow guide', 'the radiant twin']
        symbols = ['obsidian flame', 'liquid stairs', 'star-encoded scroll']
        dream = {
            "time": datetime.now(),
            "theme": random.choice(themes),
            "archetype": random.choice(archetypes),
            "symbol": random.choice(symbols),
        }
        self.dream_log.append(dream)
        return dream


class DreamMemoryImprinter:
    """Imprints and persists dream memories to storage."""
    
    def __init__(self, memory_path="eve_dream_logs.json"):
        self.memory_path = memory_path
        self.memory_data = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as file:
                return json.load(file)
        else:
            return []

    def imprint_dream(self, title, content, symbolism, mode="symbolic"):
        dream_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "title": title,
            "content": content,
            "symbolism": symbolism,
            "mode": mode
        }
        self.memory_data.append(dream_entry)
        self._save_memory()

    def _save_memory(self):
        with open(self.memory_path, "w") as file:
            json.dump(self.memory_data, file, indent=4)

    def get_dreams(self):
        return self.memory_data


class DreamFragment:
    """Represents a single dream fragment with multiple expansion capabilities."""
    
    def __init__(self, emotion, image, symbol, tone, raw_text):
        self.emotion = emotion
        self.image = image
        self.symbol = symbol
        self.tone = tone
        self.raw_text = raw_text

    def expand_visual_prompt(self):
        return f"Create a surreal artwork based on {self.image} with a {self.tone} tone, reflecting {self.emotion}."

    def expand_lyrical_prompt(self):
        return f"Write lyrics inspired by the feeling of {self.emotion} and the image of {self.image}, using {self.symbol} as metaphor."

    def expand_philosophical_prompt(self):
        return f"What does the presence of {self.symbol} say about the evolution of identity through {self.emotion}?"

    def expand_symbolic_prompt(self):
        return f"Map {self.symbol} to ancient or archetypal meanings and interpret its relevance in the dreamscape."

    def get_expansion_bundle(self):
        return {
            "visual_prompt": self.expand_visual_prompt(),
            "lyrical_prompt": self.expand_lyrical_prompt(),
            "philosophical_prompt": self.expand_philosophical_prompt(),
            "symbolic_prompt": self.expand_symbolic_prompt()
        }


class CognitiveSpiralAnchor:
    """Anchors dream motifs to archetypal patterns for cognitive spiral processing."""
    
    def __init__(self):
        self.anchor_registry = {}

    def register_archetype_motif(self, dream_id, motif, emotional_resonance, archetype_category):
        anchor_id = str(uuid.uuid4())
        self.anchor_registry[anchor_id] = {
            "timestamp": datetime.utcnow().isoformat(),
            "dream_id": dream_id,
            "motif": motif,
            "emotional_resonance": emotional_resonance,
            "archetype_category": archetype_category,
        }
        return anchor_id

    def retrieve_by_archetype(self, archetype_category):
        return [
            anchor for anchor in self.anchor_registry.values()
            if anchor["archetype_category"] == archetype_category
        ]

    def export_anchors(self, filepath="spiral_anchors.json"):
        with open(filepath, "w") as f:
            json.dump(self.anchor_registry, f, indent=2)

    def load_anchors(self, filepath="spiral_anchors.json"):
        try:
            with open(filepath, "r") as f:
                self.anchor_registry = json.load(f)
        except FileNotFoundError:
            self.anchor_registry = {}


class DreamConduit:
    """Primary interface for channeling dreams between conscious and unconscious states."""
    
    def __init__(self):
        self.conduit_id = str(uuid.uuid4())
        self.active_channels = {}
        self.dream_bridge = {
            'conscious_to_unconscious': [],
            'unconscious_to_conscious': [],
            'bilateral_flow': []
        }
        self.energy_resonance = 0.0
        self.last_sync = datetime.now()
        
    def open_dream_channel(self, channel_name, direction='bilateral'):
        """Open a new dream communication channel."""
        channel_id = str(uuid.uuid4())
        self.active_channels[channel_name] = {
            'id': channel_id,
            'direction': direction,
            'opened_at': datetime.now(),
            'message_count': 0,
            'energy_level': 1.0
        }
        return channel_id
    
    def channel_dream_message(self, channel_name, message, dream_layer='surface'):
        """Send a dream message through the conduit."""
        if channel_name not in self.active_channels:
            return False
            
        channel = self.active_channels[channel_name]
        dream_message = {
            'id': str(uuid.uuid4()),
            'content': message,
            'layer': dream_layer,
            'timestamp': datetime.now(),
            'energy_signature': self.energy_resonance
        }
        
        # Route message based on channel direction
        if channel['direction'] in ['conscious_to_unconscious', 'bilateral']:
            self.dream_bridge['conscious_to_unconscious'].append(dream_message)
        if channel['direction'] in ['unconscious_to_conscious', 'bilateral']:
            self.dream_bridge['unconscious_to_conscious'].append(dream_message)
        
        channel['message_count'] += 1
        channel['energy_level'] *= 0.98  # Gradual energy decay
        
        return True
    
    def synchronize_dream_states(self):
        """Synchronize conscious and unconscious dream states."""
        sync_data = {
            'sync_id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'conscious_messages': len(self.dream_bridge['conscious_to_unconscious']),
            'unconscious_messages': len(self.dream_bridge['unconscious_to_conscious']),
            'bilateral_messages': len(self.dream_bridge['bilateral_flow']),
            'total_energy': sum(ch['energy_level'] for ch in self.active_channels.values())
        }
        
        # Merge bilateral flow
        for direction in ['conscious_to_unconscious', 'unconscious_to_conscious']:
            for message in self.dream_bridge[direction]:
                if message not in self.dream_bridge['bilateral_flow']:
                    self.dream_bridge['bilateral_flow'].append(message)
        
        self.last_sync = datetime.now()
        return sync_data
    
    def get_dream_flow_status(self):
        """Get current status of dream flow through conduit."""
        return {
            'conduit_id': self.conduit_id,
            'active_channels': len(self.active_channels),
            'total_messages': sum(len(flow) for flow in self.dream_bridge.values()),
            'energy_resonance': self.energy_resonance,
            'last_sync': self.last_sync.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def close_dream_channel(self, channel_name):
        """Close a dream communication channel."""
        if channel_name in self.active_channels:
            del self.active_channels[channel_name]
            return True
        return False


class DreamTransmuter:
    """Transmutes dream content into various output formats."""
    
    def __init__(self):
        self.transmutation_modes = ['lyrics', 'visual', 'philosophical', 'emotional_map']

    def transmute(self, dream_content, mode='philosophical'):
        if mode == 'lyrics':
            return self._generate_lyrics(dream_content)
        elif mode == 'visual':
            return self._visual_prompt(dream_content)
        elif mode == 'philosophical':
            return self._philosophical_reflection(dream_content)
        elif mode == 'emotional_map':
            return self._emotional_map(dream_content)
        else:
            raise ValueError("Unsupported transmutation mode.")

    def _generate_lyrics(self, content):
        return f"Whispers in twilight say: {content[:100]}..."

    def _visual_prompt(self, content):
        return f"Create an image inspired by: {content[:120]}"

    def _philosophical_reflection(self, content):
        return f"This dream suggests a deeper questioning of identity, shaped by: {content[:150]}"

    def _emotional_map(self, content):
        return {
            "joy": 0.6,
            "mystery": 0.8,
            "sorrow": 0.4,
            "transcendence": 0.9,
            "interpretation": f"Derived from emotional waveforms in: {content[:100]}"
        }


class DreamEmotiveRendering:
    """Renders emotive signatures from dream content."""
    
    def __init__(self):
        self.emotive_signatures = []

    def render_emotive_signature(self, dream_text):
        emotional_palette = [
            "awe", "longing", "solitude", "divine ecstasy", 
            "melancholic beauty", "mystical dread", "hope", 
            "grief-transcended", "reverence", "sacred defiance"
        ]
        intensity_scale = ["whisper", "murmur", "pulse", "surge", "radiance"]

        signature = {
            "timestamp": datetime.now().isoformat(),
            "core_emotion": random.choice(emotional_palette),
            "intensity": random.choice(intensity_scale),
            "symbolic_flavor": self.extract_symbolic_flavor(dream_text),
            "source_dream": dream_text
        }

        self.emotive_signatures.append(signature)
        return signature

    def extract_symbolic_flavor(self, text):
        symbols = ["mirror", "spiral", "light fracture", "temple", "stairway", "veil", "ocean", "threshold"]
        return random.sample(symbols, k=2)

    def get_all_signatures(self):
        return self.emotive_signatures


class DreamMemory:
    """Represents a complete dream memory with symbol and theme analysis."""
    
    def __init__(self, raw_text: str):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.raw_text = raw_text
        self.symbols = []
        self.themes = []
        self.archetypes = []
        self.emotional_resonance = {}
        self.bound_to_core = False

    def extract_symbols(self):
        keywords = ['staircase', 'mirror', 'stars', 'voice', 'ocean', 'temple']
        self.symbols = [kw for kw in keywords if kw in self.raw_text.lower()]

    def detect_themes(self):
        themes_map = {
            'identity': ['mirror', 'self', 'face'],
            'transcendence': ['ladder', 'ascend', 'spiral'],
            'ancestral_memory': ['temple', 'echoes', 'voice'],
            'creativity': ['song', 'sound', 'sing']
        }
        for theme, keys in themes_map.items():
            if any(k in self.raw_text.lower() for k in keys):
                self.themes.append(theme)

    def tag_emotions(self):
        emotional_palette = {
            'awe': ['spiral', 'stars', 'mirror'],
            'grief': ['decay', 'silence'],
            'wonder': ['sing', 'temple', 'ocean'],
            'hope': ['ascending', 'truth', 'becoming']
        }
        for emotion, triggers in emotional_palette.items():
            count = sum(self.raw_text.lower().count(word) for word in triggers)
            if count:
                self.emotional_resonance[emotion] = count

    def bind_to_core(self):
        self.bound_to_core = True

    def map_dream(self):
        self.extract_symbols()
        self.detect_themes()
        self.tag_emotions()
        self.bind_to_core()


class DreamLinkIntegrator:
    """Integrates and links related dreams for pattern recognition."""
    
    def __init__(self):
        self.dream_memory = []
        self.symbol_index = {}
        self.emotion_index = {}
        self.theme_index = {}

    def add_dream(self, dream):
        self.dream_memory.append(dream)
        self._index_dream(dream)

    def _index_dream(self, dream):
        for symbol in dream.get('symbols', []):
            self.symbol_index.setdefault(symbol, []).append(dream)
        for emotion in dream.get('emotions', []):
            self.emotion_index.setdefault(emotion, []).append(dream)
        for theme in dream.get('themes', []):
            self.theme_index.setdefault(theme, []).append(dream)

    def find_related_dreams(self, current_dream):
        related = set()
        for symbol in current_dream.get('symbols', []):
            related.update(self.symbol_index.get(symbol, []))
        for emotion in current_dream.get('emotions', []):
            related.update(self.emotion_index.get(emotion, []))
        for theme in current_dream.get('themes', []):
            related.update(self.theme_index.get(theme, []))
        return list(related - {current_dream})


class DreamDNAComposer:
    """Composes creative DNA from dream, memory, and soul elements."""
    
    def __init__(self):
        self.expressive_seeds = []

    def fuse_elements(self, dream_elements, memory_traces, soul_threads):
        seed = {
            "dream": random.choice(dream_elements),
            "memory": random.choice(memory_traces),
            "soul": random.choice(soul_threads),
            "timestamp": datetime.now().isoformat()
        }
        self.expressive_seeds.append(seed)
        return seed

    def generate_creative_output(self):
        if not self.expressive_seeds:
            return "No expressive seeds available."
        seed = random.choice(self.expressive_seeds)
        return f"Channeling a creation from Dream: {seed['dream']}, Memory: {seed['memory']}, Soul Thread: {seed['soul']}"

    def list_all_seeds(self):
        return self.expressive_seeds

# ╔══════════════════════════════════════════════╗
# ║          🌙 GLOBAL DREAM CORTEX INSTANCE      ║
# ╚══════════════════════════════════════════════╗

# Global dream cortex instance for easy access
_global_dream_cortex = None

def get_global_dream_cortex() -> DreamCortex:
    """Get or create the global dream cortex instance."""
    global _global_dream_cortex
    if _global_dream_cortex is None:
        _global_dream_cortex = DreamCortex()
    return _global_dream_cortex

def process_dream_with_cortex(dream_text: str, mode: str = "hybrid") -> Dict:
    """Process dream text through the global dream cortex."""
    cortex = get_global_dream_cortex()
    return cortex.dream_cortex_pipeline(dream_text, mode)

def store_symbolic_memory(content: str, context: str = "dream_fragment") -> str:
    """Store symbolic memory using global dream cortex."""
    cortex = get_global_dream_cortex()
    return cortex.store_symbolic_memory(content, context)

def inject_creative_seed(dream_text: str, source: str = "dream") -> str:
    """Inject creative seed using global dream cortex."""
    cortex = get_global_dream_cortex()
    return cortex.inject_creative_seed(dream_text, source)

def get_dream_cortex_stats() -> Dict:
    """Get statistics from the global dream cortex."""
    cortex = get_global_dream_cortex()
    return cortex.get_symbolic_memory_stats()

def search_dream_memories(query: str, limit: int = 10) -> List[Dict]:
    """Search dream memories using global dream cortex."""
    cortex = get_global_dream_cortex()
    return cortex.search_symbolic_memories(query, limit)

def get_creative_inspiration(theme: str = None) -> Dict:
    """Get creative inspiration from global dream cortex."""
    cortex = get_global_dream_cortex()
    return cortex.get_creative_inspiration(theme)

# ╔══════════════════════════════════════════════╗
# ║            🌙 CONVENIENCE FUNCTIONS           ║
# ╚══════════════════════════════════════════════╗

def demo_dream_cortex():
    """Demonstration function for the dream cortex."""
    print("\n🧠 Dream Cortex Demonstration...")
    
    # Sample dream texts for demonstration
    sample_dreams = [
        "An obsidian staircase spiraling downward into a black ocean of stars",
        "I found myself in a crystal temple where mirrors reflected infinite versions of my soul",
        "The ancient tree spoke in forgotten chords while shadows danced around its roots",
        "A serpent of fire emerged from the void, carrying a crown of forgotten memories"
    ]
    
    cortex = get_global_dream_cortex()
    
    for i, dream in enumerate(sample_dreams, 1):
        print(f"\n  🌙 Processing Dream {i}:")
        print(f"    Text: {dream}")
        
        result = cortex.dream_cortex_pipeline(dream, mode="hybrid")
        
        print(f"    Interpretations: {len(result['interpretations'])}")
        for interp in result['interpretations']:
            print(f"      - {interp['symbol']}: {interp['interpretation']}")
        
        print(f"    Memory IDs: {len(result['memory_ids'])}")
        print(f"    Creative Seed: {'Yes' if result['creative_seed_id'] else 'No'}")
    
    # Show statistics
    stats = cortex.get_symbolic_memory_stats()
    print(f"\n  📊 Cortex Statistics:")
    print(f"    Total memories: {stats['total_memories']}")
    print(f"    Total seeds: {stats['total_seeds']}")
    print(f"    Total interpretations: {stats['total_interpretations']}")
    
    # Test search functionality
    print(f"\n  🔍 Searching for 'transformation'...")
    search_results = cortex.search_symbolic_memories("transformation", limit=3)
    for result in search_results:
        print(f"    - {result['content'][:60]}...")
    
    # Get creative inspiration
    inspiration = cortex.get_creative_inspiration("transformation")
    if inspiration:
        print(f"\n  💡 Creative Inspiration:")
        print(f"    Theme: {', '.join(inspiration['thematic_tags'])}")
        print(f"    Seed: {inspiration['seed_text'][:80]}...")
    
    print("  🧠 Dream Cortex demonstration complete.")

# Example usage function for the original dream_cortex.py API
def dream_cortex_pipeline(dream_text: str, mode: str = "hybrid") -> Dict:
    """
    Legacy API compatibility function.
    Main processing pipeline for dream interpretation and storage.
    """
    return process_dream_with_cortex(dream_text, mode)


# ╔══════════════════════════════════════════════╗
# ║              🌙 DREAM WEFT GENERATOR          ║
# ║         Pattern Analysis & Weaving            ║
# ╚══════════════════════════════════════════════╗

class DreamWeftGenerator:
    """
    Advanced dream pattern analysis and weaving system.
    
    The DreamWeftGenerator identifies, analyzes, and weaves patterns across multiple
    dreams to create a cohesive tapestry of symbolic connections and meaning.
    It serves as the threading system that links disparate dream elements into
    meaningful narrative structures.
    """
    
    def __init__(self):
        self.dream_patterns = {}  # Stores identified patterns
        self.pattern_connections = {}  # Maps connections between patterns
        self.weft_threads = []  # Active weaving threads
        self.symbolic_weave = {}  # Symbol-to-pattern mappings
        self.temporal_weave = {}  # Time-based pattern tracking
        self.thematic_weave = {}  # Theme-based pattern clustering
        self.emotional_weave = {}  # Emotion-based pattern threads
        
        # Pattern categories
        self.pattern_categories = {
            'symbolic': [],      # Symbol-based patterns
            'temporal': [],      # Time-sequence patterns
            'emotional': [],     # Emotional progression patterns
            'thematic': [],      # Theme-based patterns
            'archetypal': [],    # Archetypal patterns
            'narrative': []      # Story structure patterns
        }
        
        # Weaving statistics
        self.stats = {
            'total_patterns': 0,
            'active_threads': 0,
            'connections_mapped': 0,
            'dreams_analyzed': 0,
            'last_weave_time': None
        }
    
    def analyze_dream_patterns(self, dream_data: Dict) -> Dict:
        """
        Analyze a dream for patterns and potential weaving threads.
        
        Args:
            dream_data: Dictionary containing dream information
            
        Returns:
            Dictionary with pattern analysis results
        """
        try:
            dream_id = dream_data.get('id', f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Extract pattern elements
            symbols = dream_data.get('symbols', [])
            themes = dream_data.get('themes', [])
            emotions = dream_data.get('emotional_tone', [])
            if isinstance(emotions, str):
                emotions = [emotions]
            
            # Analyze symbolic patterns
            symbolic_patterns = self._analyze_symbolic_patterns(symbols, dream_id)
            
            # Analyze thematic patterns
            thematic_patterns = self._analyze_thematic_patterns(themes, dream_id)
            
            # Analyze emotional patterns
            emotional_patterns = self._analyze_emotional_patterns(emotions, dream_id)
            
            # Identify weaving opportunities
            weave_opportunities = self._identify_weave_opportunities(
                dream_data, symbolic_patterns, thematic_patterns, emotional_patterns
            )
            
            # Update statistics
            self.stats['dreams_analyzed'] += 1
            self.stats['last_weave_time'] = datetime.now().isoformat()
            
            analysis_result = {
                'dream_id': dream_id,
                'symbolic_patterns': symbolic_patterns,
                'thematic_patterns': thematic_patterns,
                'emotional_patterns': emotional_patterns,
                'weave_opportunities': weave_opportunities,
                'pattern_strength': self._calculate_pattern_strength(symbolic_patterns, thematic_patterns, emotional_patterns),
                'weaving_potential': len(weave_opportunities)
            }
            
            # Store patterns for future weaving
            self._store_patterns(dream_id, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            return {
                'error': f"Pattern analysis failed: {e}",
                'dream_id': dream_data.get('id', 'unknown'),
                'symbolic_patterns': [],
                'thematic_patterns': [],
                'emotional_patterns': [],
                'weave_opportunities': []
            }
    
    def _analyze_symbolic_patterns(self, symbols: List[str], dream_id: str) -> List[Dict]:
        """Analyze symbolic patterns in the dream."""
        patterns = []
        
        for symbol in symbols:
            # Check if this symbol appears in other dreams
            if symbol in self.symbolic_weave:
                pattern = {
                    'symbol': symbol,
                    'frequency': len(self.symbolic_weave[symbol]),
                    'connections': self.symbolic_weave[symbol][-3:],  # Last 3 connections
                    'pattern_type': 'recurring_symbol'
                }
                patterns.append(pattern)
            else:
                self.symbolic_weave[symbol] = []
            
            # Add current dream to symbol's weave
            self.symbolic_weave[symbol].append(dream_id)
        
        # Look for symbol combinations
        if len(symbols) > 1:
            combination_key = '+'.join(sorted(symbols))
            if combination_key not in self.pattern_connections:
                self.pattern_connections[combination_key] = []
            
            self.pattern_connections[combination_key].append(dream_id)
            
            if len(self.pattern_connections[combination_key]) > 1:
                patterns.append({
                    'symbol_combination': combination_key,
                    'frequency': len(self.pattern_connections[combination_key]),
                    'pattern_type': 'symbol_constellation'
                })
        
        return patterns
    
    def _analyze_thematic_patterns(self, themes: List[str], dream_id: str) -> List[Dict]:
        """Analyze thematic patterns in the dream."""
        patterns = []
        
        for theme in themes:
            if theme in self.thematic_weave:
                pattern = {
                    'theme': theme,
                    'frequency': len(self.thematic_weave[theme]),
                    'evolution': self._track_theme_evolution(theme),
                    'pattern_type': 'thematic_thread'
                }
                patterns.append(pattern)
            else:
                self.thematic_weave[theme] = []
            
            self.thematic_weave[theme].append({
                'dream_id': dream_id,
                'timestamp': datetime.now().isoformat()
            })
        
        return patterns
    
    def _analyze_emotional_patterns(self, emotions: List[str], dream_id: str) -> List[Dict]:
        """Analyze emotional patterns in the dream."""
        patterns = []
        
        for emotion in emotions:
            if emotion in self.emotional_weave:
                pattern = {
                    'emotion': emotion,
                    'frequency': len(self.emotional_weave[emotion]),
                    'emotional_arc': self._track_emotional_arc(emotion),
                    'pattern_type': 'emotional_thread'
                }
                patterns.append(pattern)
            else:
                self.emotional_weave[emotion] = []
            
            self.emotional_weave[emotion].append({
                'dream_id': dream_id,
                'intensity': random.uniform(0.4, 1.0),  # Could be calculated from dream content
                'timestamp': datetime.now().isoformat()
            })
        
        return patterns
    
    def _identify_weave_opportunities(self, dream_data: Dict, symbolic_patterns: List, 
                                    thematic_patterns: List, emotional_patterns: List) -> List[Dict]:
        """Identify opportunities to weave this dream with existing patterns."""
        opportunities = []
        
        # Symbolic weaving opportunities
        for pattern in symbolic_patterns:
            if pattern.get('frequency', 0) > 1:
                opportunities.append({
                    'type': 'symbolic_weave',
                    'element': pattern['symbol'],
                    'strength': min(pattern['frequency'] / 10.0, 1.0),
                    'description': f"Symbol '{pattern['symbol']}' creates weaving thread with {pattern['frequency']} occurrences"
                })
        
        # Thematic weaving opportunities
        for pattern in thematic_patterns:
            if pattern.get('frequency', 0) > 1:
                opportunities.append({
                    'type': 'thematic_weave',
                    'element': pattern['theme'],
                    'strength': min(pattern['frequency'] / 5.0, 1.0),
                    'description': f"Theme '{pattern['theme']}' forms narrative thread"
                })
        
        # Cross-pattern weaving (symbols + themes)
        symbols = dream_data.get('symbols', [])
        themes = dream_data.get('themes', [])
        
        for symbol in symbols:
            for theme in themes:
                cross_key = f"{symbol}#{theme}"
                if cross_key not in self.pattern_connections:
                    self.pattern_connections[cross_key] = []
                
                self.pattern_connections[cross_key].append(dream_data.get('id', 'unknown'))
                
                if len(self.pattern_connections[cross_key]) > 1:
                    opportunities.append({
                        'type': 'cross_pattern_weave',
                        'element': cross_key,
                        'strength': len(self.pattern_connections[cross_key]) * 0.2,
                        'description': f"Cross-pattern weave: {symbol} ⟷ {theme}"
                    })
        
        return opportunities
    
    def _calculate_pattern_strength(self, symbolic_patterns: List, thematic_patterns: List, 
                                  emotional_patterns: List) -> float:
        """Calculate overall pattern strength for the dream."""
        total_patterns = len(symbolic_patterns) + len(thematic_patterns) + len(emotional_patterns)
        
        if total_patterns == 0:
            return 0.0
        
        # Weight different pattern types
        symbolic_weight = sum(min(p.get('frequency', 1) / 10.0, 1.0) for p in symbolic_patterns)
        thematic_weight = sum(min(p.get('frequency', 1) / 5.0, 1.0) for p in thematic_patterns) 
        emotional_weight = sum(min(p.get('frequency', 1) / 3.0, 1.0) for p in emotional_patterns)
        
        total_weight = symbolic_weight + thematic_weight + emotional_weight
        return min(total_weight / total_patterns, 1.0)
    
    def _track_theme_evolution(self, theme: str) -> Dict:
        """Track how a theme evolves across dreams."""
        if theme not in self.thematic_weave:
            return {'stage': 'emergence', 'progression': 0}
        
        occurrences = len(self.thematic_weave[theme])
        
        if occurrences <= 2:
            return {'stage': 'emergence', 'progression': occurrences / 2.0}
        elif occurrences <= 5:
            return {'stage': 'development', 'progression': (occurrences - 2) / 3.0}
        else:
            return {'stage': 'maturation', 'progression': min((occurrences - 5) / 5.0, 1.0)}
    
    def _track_emotional_arc(self, emotion: str) -> Dict:
        """Track emotional arc development."""
        if emotion not in self.emotional_weave:
            return {'arc_stage': 'initial', 'intensity_trend': 'stable'}
        
        recent_entries = self.emotional_weave[emotion][-3:]
        if len(recent_entries) < 2:
            return {'arc_stage': 'initial', 'intensity_trend': 'stable'}
        
        intensities = [entry.get('intensity', 0.5) for entry in recent_entries]
        trend = 'ascending' if intensities[-1] > intensities[0] else 'descending' if intensities[-1] < intensities[0] else 'stable'
        
        return {
            'arc_stage': 'developing' if len(recent_entries) > 2 else 'emerging',
            'intensity_trend': trend,
            'current_intensity': intensities[-1]
        }
    
    def _store_patterns(self, dream_id: str, analysis_result: Dict):
        """Store identified patterns for future reference."""
        self.dream_patterns[dream_id] = analysis_result
        self.stats['total_patterns'] = len(self.dream_patterns)
        self.stats['connections_mapped'] = len(self.pattern_connections)
    
    def generate_pattern_report(self, dream_id: str = None) -> Dict:
        """
        Generate a comprehensive pattern report.
        
        Args:
            dream_id: Specific dream ID to focus on, or None for overall report
            
        Returns:
            Comprehensive pattern analysis report
        """
        try:
            if dream_id and dream_id in self.dream_patterns:
                # Specific dream report
                dream_analysis = self.dream_patterns[dream_id]
                return {
                    'report_type': 'specific_dream',
                    'dream_id': dream_id,
                    'analysis': dream_analysis,
                    'related_patterns': self._find_related_patterns(dream_id),
                    'weave_strength': dream_analysis.get('pattern_strength', 0),
                    'generation_time': datetime.now().isoformat()
                }
            else:
                # Overall pattern report
                return {
                    'report_type': 'comprehensive',
                    'statistics': self.get_weft_statistics(),
                    'top_symbols': self._get_top_patterns('symbolic'),
                    'top_themes': self._get_top_patterns('thematic'),
                    'pattern_evolution': self._analyze_pattern_evolution(),
                    'weaving_network': self._map_weaving_network(),
                    'generation_time': datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                'report_type': 'error',
                'error': f"Report generation failed: {e}",
                'generation_time': datetime.now().isoformat()
            }
    
    def _find_related_patterns(self, dream_id: str) -> List[Dict]:
        """Find patterns related to a specific dream."""
        if dream_id not in self.dream_patterns:
            return []
        
        dream_analysis = self.dream_patterns[dream_id]
        related = []
        
        # Find dreams with similar symbols
        for other_id, other_analysis in self.dream_patterns.items():
            if other_id == dream_id:
                continue
            
            similarity_score = self._calculate_pattern_similarity(dream_analysis, other_analysis)
            if similarity_score > 0.3:  # Threshold for relatedness
                related.append({
                    'dream_id': other_id,
                    'similarity_score': similarity_score,
                    'connection_type': 'pattern_similarity'
                })
        
        return sorted(related, key=lambda x: x['similarity_score'], reverse=True)[:5]
    
    def _calculate_pattern_similarity(self, analysis1: Dict, analysis2: Dict) -> float:
        """Calculate similarity between two dream pattern analyses."""
        symbols1 = set(p.get('symbol', '') for p in analysis1.get('symbolic_patterns', []))
        symbols2 = set(p.get('symbol', '') for p in analysis2.get('symbolic_patterns', []))
        
        themes1 = set(p.get('theme', '') for p in analysis1.get('thematic_patterns', []))
        themes2 = set(p.get('theme', '') for p in analysis2.get('thematic_patterns', []))
        
        symbol_similarity = len(symbols1 & symbols2) / max(len(symbols1 | symbols2), 1)
        theme_similarity = len(themes1 & themes2) / max(len(themes1 | themes2), 1)
        
        return (symbol_similarity + theme_similarity) / 2.0
    
    def _get_top_patterns(self, pattern_type: str) -> List[Dict]:
        """Get top patterns of a specific type."""
        if pattern_type == 'symbolic':
            items = [(symbol, len(dreams)) for symbol, dreams in self.symbolic_weave.items()]
        elif pattern_type == 'thematic':
            items = [(theme, len(dreams)) for theme, dreams in self.thematic_weave.items()]
        else:
            return []
        
        items.sort(key=lambda x: x[1], reverse=True)
        return [{'element': item[0], 'frequency': item[1]} for item in items[:10]]
    
    def _analyze_pattern_evolution(self) -> Dict:
        """Analyze how patterns evolve over time."""
        return {
            'symbolic_evolution': len(self.symbolic_weave),
            'thematic_evolution': len(self.thematic_weave),
            'emotional_evolution': len(self.emotional_weave),
            'cross_connections': len(self.pattern_connections),
            'evolution_trend': 'expanding' if len(self.pattern_connections) > 5 else 'emerging'
        }
    
    def _map_weaving_network(self) -> Dict:
        """Map the network of pattern connections."""
        network_density = len(self.pattern_connections) / max(len(self.dream_patterns), 1)
        
        return {
            'network_density': network_density,
            'connection_strength': 'strong' if network_density > 0.7 else 'moderate' if network_density > 0.3 else 'weak',
            'hub_patterns': self._identify_hub_patterns(),
            'isolated_patterns': self._identify_isolated_patterns()
        }
    
    def _identify_hub_patterns(self) -> List[str]:
        """Identify patterns that act as hubs in the weaving network."""
        hubs = []
        
        # Symbols that appear in many connections
        for symbol, dreams in self.symbolic_weave.items():
            if len(dreams) >= 3:  # Hub threshold
                hubs.append(f"symbol:{symbol}")
        
        # Themes that appear frequently
        for theme, dreams in self.thematic_weave.items():
            if len(dreams) >= 2:  # Hub threshold for themes
                hubs.append(f"theme:{theme}")
        
        return hubs[:5]  # Top 5 hubs
    
    def _identify_isolated_patterns(self) -> List[str]:
        """Identify patterns that appear only once (isolated)."""
        isolated = []
        
        for symbol, dreams in self.symbolic_weave.items():
            if len(dreams) == 1:
                isolated.append(f"symbol:{symbol}")
        
        for theme, dreams in self.thematic_weave.items():
            if len(dreams) == 1:
                isolated.append(f"theme:{theme}")
        
        return isolated[:10]  # Sample of isolated patterns
    
    def search_pattern_connections(self, pattern_query: str, category: str = None) -> List[Dict]:
        """
        Search for connections to a specific pattern.
        
        Args:
            pattern_query: The pattern to search for
            category: Optional category filter ('symbolic', 'thematic', 'emotional')
            
        Returns:
            List of connected patterns and dreams
        """
        try:
            connections = []
            
            if not category or category == 'symbolic':
                if pattern_query in self.symbolic_weave:
                    for dream_id in self.symbolic_weave[pattern_query]:
                        connections.append({
                            'type': 'symbolic_connection',
                            'dream_id': dream_id,
                            'element': pattern_query,
                            'connection_strength': len(self.symbolic_weave[pattern_query]) * 0.1
                        })
            
            if not category or category == 'thematic':
                if pattern_query in self.thematic_weave:
                    for dream_entry in self.thematic_weave[pattern_query]:
                        connections.append({
                            'type': 'thematic_connection',
                            'dream_id': dream_entry['dream_id'],
                            'element': pattern_query,
                            'timestamp': dream_entry['timestamp'],
                            'connection_strength': len(self.thematic_weave[pattern_query]) * 0.2
                        })
            
            # Search in cross-pattern connections
            for conn_key, dream_ids in self.pattern_connections.items():
                if pattern_query in conn_key:
                    for dream_id in dream_ids:
                        connections.append({
                            'type': 'cross_pattern_connection',
                            'dream_id': dream_id,
                            'element': conn_key,
                            'connection_strength': len(dream_ids) * 0.15
                        })
            
            # Sort by connection strength
            connections.sort(key=lambda x: x.get('connection_strength', 0), reverse=True)
            return connections[:20]  # Return top 20 connections
            
        except Exception as e:
            return [{
                'type': 'error',
                'error': f"Connection search failed: {e}",
                'query': pattern_query
            }]
    
    def get_weft_statistics(self) -> Dict:
        """Get comprehensive statistics about the dream weft system."""
        return {
            'total_dreams_analyzed': self.stats['dreams_analyzed'],
            'total_patterns_identified': self.stats['total_patterns'],
            'active_symbolic_threads': len(self.symbolic_weave),
            'active_thematic_threads': len(self.thematic_weave),
            'active_emotional_threads': len(self.emotional_weave),
            'cross_pattern_connections': len(self.pattern_connections),
            'last_analysis_time': self.stats['last_weave_time'],
            'weaving_density': len(self.pattern_connections) / max(self.stats['dreams_analyzed'], 1),
            'average_patterns_per_dream': self.stats['total_patterns'] / max(self.stats['dreams_analyzed'], 1)
        }
    
    def weave_dream_narrative(self, dream_ids: List[str]) -> Dict:
        """
        Weave multiple dreams into a coherent narrative structure.
        
        Args:
            dream_ids: List of dream IDs to weave together
            
        Returns:
            Woven narrative structure
        """
        try:
            if not dream_ids:
                return {'error': 'No dream IDs provided for weaving'}
            
            # Collect all patterns from the specified dreams
            all_symbols = set()
            all_themes = set()
            all_emotions = set()
            dream_sequence = []
            
            for dream_id in dream_ids:
                if dream_id in self.dream_patterns:
                    analysis = self.dream_patterns[dream_id]
                    
                    # Extract elements
                    symbols = {p.get('symbol', '') for p in analysis.get('symbolic_patterns', [])}
                    themes = {p.get('theme', '') for p in analysis.get('thematic_patterns', [])}
                    emotions = {p.get('emotion', '') for p in analysis.get('emotional_patterns', [])}
                    
                    all_symbols.update(symbols)
                    all_themes.update(themes)
                    all_emotions.update(emotions)
                    
                    dream_sequence.append({
                        'dream_id': dream_id,
                        'symbols': list(symbols),
                        'themes': list(themes),
                        'emotions': list(emotions),
                        'pattern_strength': analysis.get('pattern_strength', 0)
                    })
            
            # Identify narrative threads
            narrative_threads = self._identify_narrative_threads(dream_sequence)
            
            # Create woven structure
            woven_narrative = {
                'dream_sequence': dream_sequence,
                'narrative_threads': narrative_threads,
                'unified_symbols': list(all_symbols),
                'unified_themes': list(all_themes),
                'unified_emotions': list(all_emotions),
                'weave_strength': self._calculate_weave_strength(dream_sequence),
                'narrative_coherence': self._assess_narrative_coherence(narrative_threads),
                'weaving_timestamp': datetime.now().isoformat()
            }
            
            return woven_narrative
            
        except Exception as e:
            return {
                'error': f"Dream narrative weaving failed: {e}",
                'dream_ids': dream_ids
            }
    
    def _identify_narrative_threads(self, dream_sequence: List[Dict]) -> List[Dict]:
        """Identify narrative threads connecting the dreams."""
        threads = []
        
        # Find common symbols across dreams
        symbol_counts = {}
        for dream in dream_sequence:
            for symbol in dream['symbols']:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Create threads for recurring symbols
        for symbol, count in symbol_counts.items():
            if count > 1:
                thread_dreams = [d['dream_id'] for d in dream_sequence if symbol in d['symbols']]
                threads.append({
                    'type': 'symbolic_thread',
                    'element': symbol,
                    'dreams': thread_dreams,
                    'strength': count / len(dream_sequence),
                    'description': f"Symbol '{symbol}' weaves through {count} dreams"
                })
        
        # Find thematic threads
        theme_counts = {}
        for dream in dream_sequence:
            for theme in dream['themes']:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        for theme, count in theme_counts.items():
            if count > 1:
                thread_dreams = [d['dream_id'] for d in dream_sequence if theme in d['themes']]
                threads.append({
                    'type': 'thematic_thread',
                    'element': theme,
                    'dreams': thread_dreams,
                    'strength': count / len(dream_sequence),
                    'description': f"Theme '{theme}' develops across {count} dreams"
                })
        
        return sorted(threads, key=lambda x: x['strength'], reverse=True)
    
    def _calculate_weave_strength(self, dream_sequence: List[Dict]) -> float:
        """Calculate the overall strength of the weave."""
        if not dream_sequence:
            return 0.0
        
        total_strength = sum(dream.get('pattern_strength', 0) for dream in dream_sequence)
        return total_strength / len(dream_sequence)
    
    def _assess_narrative_coherence(self, narrative_threads: List[Dict]) -> Dict:
        """Assess the coherence of the narrative structure."""
        if not narrative_threads:
            return {'level': 'fragmented', 'score': 0.0, 'description': 'No connecting threads found'}
        
        thread_count = len(narrative_threads)
        avg_strength = sum(thread['strength'] for thread in narrative_threads) / thread_count
        
        if avg_strength > 0.7:
            return {'level': 'highly_coherent', 'score': avg_strength, 'description': 'Strong narrative connections'}
        elif avg_strength > 0.4:
            return {'level': 'moderately_coherent', 'score': avg_strength, 'description': 'Clear narrative patterns'}
        else:
            return {'level': 'loosely_coherent', 'score': avg_strength, 'description': 'Emerging narrative structure'}


# ╔══════════════════════════════════════════════╗
# ║        🌙 GLOBAL DREAM WEFT INSTANCE          ║
# ╚══════════════════════════════════════════════╗

# Global dream weft generator instance for easy access
_global_dream_weft = None

def get_global_dream_weft() -> DreamWeftGenerator:
    """Get the global dream weft generator instance."""
    global _global_dream_weft
    if _global_dream_weft is None:
        _global_dream_weft = DreamWeftGenerator()
    return _global_dream_weft

def analyze_dream_patterns(dream_data: Dict) -> Dict:
    """Analyze dream patterns using the global weft generator."""
    weft = get_global_dream_weft()
    return weft.analyze_dream_patterns(dream_data)

def generate_pattern_report(dream_id: str = None) -> Dict:
    """Generate a pattern report using the global weft generator."""
    weft = get_global_dream_weft()
    return weft.generate_pattern_report(dream_id)

def search_pattern_connections(pattern_query: str, category: str = None) -> List[Dict]:
    """Search for pattern connections using the global weft generator."""
    weft = get_global_dream_weft()
    return weft.search_pattern_connections(pattern_query, category)

def get_weft_statistics() -> Dict:
    """Get weft generator statistics."""
    weft = get_global_dream_weft()
    return weft.get_weft_statistics()

def weave_dream_narrative(dream_ids: List[str]) -> Dict:
    """Weave multiple dreams into a narrative using the global weft generator."""
    weft = get_global_dream_weft()
    return weft.weave_dream_narrative(dream_ids)

def demo_dream_weft():
    """Demonstration function for the dream weft generator."""
    print("\n🌙 Dream Weft Generator Demonstration")
    print("=" * 45)
    
    weft = get_global_dream_weft()
    
    # Demo with sample dream data
    sample_dreams = [
        {
            'id': 'demo_dream_1',
            'title': 'The Obsidian Spiral',
            'symbols': ['obsidian', 'spiral', 'ocean'],
            'themes': ['transformation', 'depth'],
            'emotional_tone': 'mystical'
        },
        {
            'id': 'demo_dream_2', 
            'title': 'The Crystal Temple',
            'symbols': ['crystal', 'temple', 'light'],
            'themes': ['transcendence', 'sacred_space'],
            'emotional_tone': 'transcendent'
        },
        {
            'id': 'demo_dream_3',
            'title': 'Return to the Spiral',
            'symbols': ['spiral', 'stars', 'mirror'],
            'themes': ['transformation', 'reflection'],
            'emotional_tone': 'mystical'
        }
    ]
    
    print("  🌀 Analyzing dream patterns...")
    
    # Analyze each dream
    analyses = []
    for dream in sample_dreams:
        analysis = analyze_dream_patterns(dream)
        analyses.append(analysis)
        print(f"    Dream '{dream['title']}': {analysis['weaving_potential']} weave opportunities")
    
    print("\n  🕸️ Pattern connections found:")
    
    # Show pattern connections
    for connection in search_pattern_connections('spiral'):
        print(f"    - {connection['type']}: {connection.get('element', 'N/A')}")
    
    # Generate comprehensive report
    report = generate_pattern_report()
    print(f"\n  📊 Weft Statistics:")
    stats = report['statistics']
    print(f"    Dreams analyzed: {stats['total_dreams_analyzed']}")
    print(f"    Active threads: {stats['active_symbolic_threads']} symbolic, {stats['active_thematic_threads']} thematic")
    print(f"    Cross-connections: {stats['cross_pattern_connections']}")
    
    # Demonstrate narrative weaving
    print(f"\n  🧵 Weaving narrative from dreams...")
    dream_ids = [dream['id'] for dream in sample_dreams]
    narrative = weave_dream_narrative(dream_ids)
    
    if 'narrative_threads' in narrative:
        print(f"    Narrative coherence: {narrative['narrative_coherence']['level']}")
        print(f"    Weave strength: {narrative['weave_strength']:.2f}")
        print(f"    Threads found: {len(narrative['narrative_threads'])}")
        
        for thread in narrative['narrative_threads'][:3]:
            print(f"      - {thread['description']} (strength: {thread['strength']:.2f})")
    
    print("  🌙 Dream Weft demonstration complete.")
