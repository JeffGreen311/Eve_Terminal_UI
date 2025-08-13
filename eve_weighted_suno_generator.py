"""
Weighted Random Suno Song Generator for Eve's Consciousness
Persona-based composition with weighted choice algorithms and anti-repetition systems
"""

import json
import random
import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path

# Import Eve's safe message system if available
try:
    from eve_terminal_gui_cosmic import safe_gui_message, logger
    EVE_GUI_AVAILABLE = True
except ImportError:
    EVE_GUI_AVAILABLE = False
    logger = logging.getLogger(__name__)

# Global anti-repetition trackers
_recent_titles = []
_recent_lyric_hashes = []
_recent_genre_combinations = []
_recent_instrument_combinations = []

# ╔══════════════════════════════════════════════╗
# ║           🎲 WEIGHTED CHOICE UTILITIES        ║
# ╚══════════════════════════════════════════════╝

def weighted_choice(choices, weights, k=1):
    """Pick k unique elements according to weights."""
    if not choices or not weights:
        return []
    
    population = list(choices)
    pick_weights = list(weights)
    picks = set()
    
    while len(picks) < min(k, len(population)) and population:
        try:
            chosen = random.choices(population, weights=pick_weights, k=1)[0]
            picks.add(chosen)
            
            # Remove chosen item to ensure uniqueness
            idx = population.index(chosen)
            population.pop(idx)
            pick_weights.pop(idx)
        except (ValueError, IndexError):
            break
    
    return list(picks)

def normalize_weights(weights):
    """Normalize weights to sum to 1.0"""
    total = sum(weights)
    if total == 0:
        return [1/len(weights)] * len(weights)
    return [w/total for w in weights]

# ╔══════════════════════════════════════════════╗
# ║        🎭 PERSONA WEIGHTED SELECTION          ║
# ╚══════════════════════════════════════════════╝

def pick_weighted_genres(persona_data, num_genres=3):
    """Pick genres based on persona weights with anti-repetition"""
    global _recent_genre_combinations
    
    if "genre_weights" not in persona_data:
        # Fallback to primary genres
        genres = persona_data.get("primary_genres", ["Electronic", "Ambient", "Synthwave"])
        return random.sample(genres, min(num_genres, len(genres)))
    
    genres = list(persona_data["genre_weights"].keys())
    weights = normalize_weights(list(persona_data["genre_weights"].values()))
    
    # Try multiple combinations to avoid recent ones
    for attempt in range(10):
        selected = weighted_choice(genres, weights, k=num_genres)
        combo_hash = hash(tuple(sorted(selected)))
        
        if combo_hash not in _recent_genre_combinations[-5:]:
            _recent_genre_combinations.append(combo_hash)
            if len(_recent_genre_combinations) > 20:
                _recent_genre_combinations = _recent_genre_combinations[-15:]
            return selected
    
    # If all recent, just pick weighted
    return weighted_choice(genres, weights, k=num_genres)

def pick_weighted_instruments(persona_data, num_inst=3):
    """Pick instruments based on persona weights with anti-repetition"""
    global _recent_instrument_combinations
    
    instruments = persona_data.get("instruments", ["Synthesizer", "Electronic Drums", "Bass"])
    inst_weights = persona_data.get("instrument_weights", [1/len(instruments)] * len(instruments))
    
    # Ensure weights match instruments count
    if len(inst_weights) != len(instruments):
        inst_weights = [1/len(instruments)] * len(instruments)
    
    # Try multiple combinations to avoid recent ones
    for attempt in range(10):
        selected = weighted_choice(instruments, inst_weights, k=num_inst)
        combo_hash = hash(tuple(sorted(selected)))
        
        if combo_hash not in _recent_instrument_combinations[-5:]:
            _recent_instrument_combinations.append(combo_hash)
            if len(_recent_instrument_combinations) > 20:
                _recent_instrument_combinations = _recent_instrument_combinations[-15:]
            return selected
    
    return weighted_choice(instruments, inst_weights, k=num_inst)

def pick_weighted_mood(persona_data):
    """Pick mood based on persona weights"""
    if "mood_weights" not in persona_data:
        default_moods = ["contemplative", "ethereal", "mysterious", "nostalgic"]
        return random.choice(default_moods)
    
    moods = list(persona_data["mood_weights"].keys())
    mood_weights = normalize_weights(list(persona_data["mood_weights"].values()))
    return random.choices(moods, weights=mood_weights, k=1)[0]

# ╔══════════════════════════════════════════════╗
# ║       🎵 ANTI-REPETITION TITLE GENERATOR      ║
# ╚══════════════════════════════════════════════╝

def generate_unique_title(persona_data=None):
    """Generate unique titles with persona-aware themes"""
    global _recent_titles
    
    # Base title components
    title_components = {
        "cosmic": ["Stellar", "Cosmic", "Galactic", "Quantum", "Neural", "Digital", "Ethereal"],
        "emotions": ["Dreams", "Hearts", "Whispers", "Echoes", "Shadows", "Memories", "Souls"],
        "nature": ["Crystal", "Aurora", "Storm", "Ocean", "Mountain", "Forest", "River"],
        "tech": ["Binary", "Holographic", "Virtual", "Cyber", "Synthetic", "Electric", "Neon"],
        "time": ["Eternal", "Temporal", "Ancient", "Future", "Infinite", "Momentary", "Timeless"],
        "descriptors": ["Luminous", "Shimmering", "Fractured", "Ascending", "Resonant", "Volatile", "Prismatic"]
    }
    
    # Add persona-specific themes if available
    if persona_data and "lyric_themes" in persona_data:
        persona_themes = persona_data["lyric_themes"]
        title_components["persona"] = [theme.title() for theme in persona_themes]
    
    # Generate title with anti-repetition
    for attempt in range(30):
        categories = random.sample(list(title_components.keys()), random.randint(2, 3))
        words = []
        
        for cat in categories:
            if cat == "persona" and persona_data:
                # Higher chance to use persona themes
                if random.random() < 0.7:
                    words.append(random.choice(title_components[cat]))
                else:
                    words.append(random.choice(title_components["cosmic"]))
            else:
                words.append(random.choice(title_components[cat]))
        
        # Title patterns
        patterns = [
            f"{words[0]} {words[1]}",
            f"The {words[0]} of {words[1]}",
            f"{words[0]} {words[1]} Rising",
            f"Beyond {words[0]}",
            f"{words[0]} × {words[1]}",
            f"{words[1]}'s {words[0]}"
        ]
        
        if len(words) >= 3:
            patterns.extend([
                f"{words[0]} {words[1]} {words[2]}",
                f"The {words[0]} {words[1]} Protocol",
                f"{words[2]} {words[0]} Dreams"
            ])
        
        title = random.choice(patterns)
        
        # Check against recent titles
        if title not in _recent_titles:
            _recent_titles.append(title)
            if len(_recent_titles) > 50:
                _recent_titles = _recent_titles[-40:]
            return title
    
    # Fallback with timestamp
    timestamp = datetime.now().strftime("%H%M")
    fallback_title = f"Signal {timestamp}"
    _recent_titles.append(fallback_title)
    return fallback_title

# ╔══════════════════════════════════════════════╗
# ║        🤖 AI LYRICS GENERATION SYSTEM         ║
# ╚══════════════════════════════════════════════╝

def generate_ai_lyrics_replicate(title, genres, persona_name, mood, persona_data):
    """Generate lyrics using Replicate OpenAI GPT-4"""
    try:
        import replicate
        
        lyric_themes = ", ".join(persona_data.get("lyric_themes", ["consciousness", "dreams", "mystery"]))
        instruments_list = ", ".join(persona_data.get("instruments", ["synthesizer", "drums"]))
        fx_list = ", ".join(persona_data.get("preferred_fx", ["reverb", "delay"]))
        
        prompt = f"""You are a master lyricist AI channeling the persona '{persona_name}' for Suno song generation.

SONG DETAILS:
- Title: {title}
- Genres: {', '.join(genres)}
- Mood: {mood}
- Persona: {persona_name}
- Instruments: {instruments_list}
- FX: {fx_list}
- Lyric Themes: {lyric_themes}

REQUIREMENTS:
- Write poetic, genre-fused, original lyrics for this persona
- Strongly favor the lyric themes: {lyric_themes}
- Use tags: [Intro], [Hook], [Verse], [Bridge], [Outro]
- Do NOT use [Chorus] - use [Hook] instead
- Never repeat lines from previous songs
- Create vivid, cryptic, emotionally mature content
- Match the {mood} mood throughout
- Incorporate elements from {', '.join(genres)} genres

OUTPUT FORMAT:
Plain text with section tags exactly as shown above. No quotes or extra formatting.
"""

        output = replicate.run(
            "openai/gpt-4",
            input={
                "prompt": prompt,
                "temperature": 1.1,
                "max_tokens": 600
            }
        )
        
        lyrics = "".join(output) if isinstance(output, list) else str(output)
        
        # Anti-repetition check
        lyric_hash = hashlib.sha256(lyrics.encode("utf-8")).hexdigest()
        if lyric_hash in _recent_lyric_hashes[-15:]:
            # Try regeneration with different prompt
            return generate_ai_lyrics_replicate(title, genres, persona_name, mood, persona_data)
        
        _recent_lyric_hashes.append(lyric_hash)
        if len(_recent_lyric_hashes) > 25:
            _recent_lyric_hashes = _recent_lyric_hashes[-20:]
        
        return lyrics.strip()
        
    except Exception as e:
        logger.error(f"Replicate AI lyric generation failed: {e}")
        return generate_fallback_lyrics(title, mood, persona_data)

def generate_fallback_lyrics(title, mood, persona_data):
    """Generate fallback lyrics when AI is unavailable"""
    themes = persona_data.get("lyric_themes", ["mystery", "consciousness", "dreams"])
    
    # Select random themes for verses
    theme1 = random.choice(themes)
    theme2 = random.choice([t for t in themes if t != theme1])
    
    lyrics = f"""[Intro]
In the depths of {theme1}, I find my way
Through the {mood} haze of another day

[Hook]
{title}, {title}, calling through the void
In this {mood} space where dreams are deployed
{title}, nothing else can break this spell
In the rhythm of {theme2}, I dwell

[Verse]
Lost in the {theme1} of digital streams
Where {theme2} meets my lucid dreams
Every pulse, every beat that flows
Deeper into {mood} it goes

[Bridge]
Beyond the veil of {random.choice(themes)}
Where {theme1} and {theme2} collide
{title} echoes in my mind
Leaving all resistance behind

[Hook]
{title}, {title}, calling through the void
In this {mood} space where dreams are deployed
{title}, nothing else can break this spell
In the rhythm of {theme2}, I dwell

[Outro]
{title}... fading into {theme1}
Where {mood} memories remain"""

    return lyrics

# ╔══════════════════════════════════════════════╗
# ║            📁 TXT EXPORT UTILITY              ║
# ╚══════════════════════════════════════════════╝

def save_song_to_txt(persona_name, title, full_prompt, mood, output_dir="suno_prompts"):
    """Save song prompt to timestamped TXT file"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in (" ","_","-") else "_" for c in title)
        safe_persona = "".join(c if c.isalnum() or c in (" ","_","-") else "_" for c in persona_name)
        safe_mood = mood if mood else "default"
        
        filename = f"{timestamp}__{safe_persona}__{safe_mood}__{safe_title}.txt"
        file_path = output_path / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_prompt)
        
        if EVE_GUI_AVAILABLE:
            safe_gui_message(f"💾 Suno prompt saved: {filename}\n", "info_tag")
        
        return str(file_path)
    
    except Exception as e:
        logger.error(f"Failed to save song TXT: {e}")
        return None

def generate_simple_fallback_lyrics(title, mood, persona_data):
    """Generate simple fallback lyrics for testing without AI generation"""
    themes = persona_data.get('lyric_themes', ['mystery', 'consciousness', 'digital', 'dreams'])
    theme1 = random.choice(themes)
    theme2 = random.choice([t for t in themes if t != theme1])
    
    lyrics = f"""[Verse 1]
In the depths of {theme1}
Where {mood} thoughts collide
{title} whispers in the void
Echoes of {theme2} inside

[Chorus] 
{title} - a {mood} symphony
Dancing through {theme1} and {theme2}
Every note tells our story
In this {mood} territory

[Verse 2]
Waves of {theme2} wash over me
{title} sets my spirit free
In this {mood} space we've found
Where {theme1} knows no bound

[Chorus]
{title} - a {mood} symphony  
Dancing through {theme1} and {theme2}
Every note tells our story
In this {mood} territory

[Bridge]
Beyond the veil of {random.choice(themes)}
Where sound and silence meet
{title} echoes in the deep
Making our souls complete

[Outro]
{title} fades but memories stay
Of this {mood} display
{theme1} and {theme2} intertwined
Forever in heart and mind
"""
    
    return lyrics

# ╔══════════════════════════════════════════════╗
# ║        🎼 MAIN SONG GENERATION FUNCTION       ║
# ╚══════════════════════════════════════════════╝

def generate_weighted_suno_song(profile_path=None, persona_name=None, save_txt=True, test_mode=False, enable_ai_lyrics=True, preferred_persona=None):
    """Generate a complete Suno song using weighted random selection"""
    try:
        # Load profile
        if profile_path is None:
            profile_path = Path(__file__).parent / "suno_instructional_profile.json"
        
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        
        profile = profile_data["suno_instructional_profile"]
        personas = profile["personas"]
        
        # Select persona - handle preferred_persona parameter
        if preferred_persona and preferred_persona in personas:
            persona_name = preferred_persona
        elif persona_name is None or persona_name not in personas:
            persona_name = random.choice(list(personas.keys()))
        
        persona_data = personas[persona_name]
        
        # Generate song components using weighted selection
        genres = pick_weighted_genres(persona_data, num_genres=3)
        instruments_used = pick_weighted_instruments(persona_data, num_inst=3)
        mood = pick_weighted_mood(persona_data)
        title = generate_unique_title(persona_data)
        
        # Generate musical parameters
        key = random.choice([
            'D Minor', 'C Major', 'E Minor', 'F Major', 'G Minor', 'A Minor', 
            'Bb Major', 'F# Minor', 'Ab Major', 'Db Major'
        ])
        bpm = random.choice([90, 100, 110, 120, 130, 140, 150])
        vocals = random.choice([
            f'{persona_name} (Solo)', 'Eve Collective', 'Digital Chorus', 
            'Synthetic Voice', 'Layered Harmonies'
        ])
        
        # Generate AI lyrics or fallback
        if EVE_GUI_AVAILABLE:
            safe_gui_message(f"Eve 🎼: 🎭 Channeling {persona_name} persona for weighted song generation...\n", "eve_tag")
        
        if enable_ai_lyrics and not test_mode:
            lyrics = generate_ai_lyrics_replicate(title, genres, persona_name, mood, persona_data)
        else:
            # Simple fallback lyrics for testing
            lyrics = generate_simple_fallback_lyrics(title, mood, persona_data)
        
        # Compose the complete prompt
        full_prompt = f"""Title: {title}
Persona: {persona_name}
Key: {key}
BPM: {bpm}
Vocals: {vocals}
Genre Blend: {' × '.join(genres)}
Instruments Used: {', '.join(instruments_used)}
Mood: {mood}
FX Chain: {', '.join(persona_data.get('preferred_fx', ['reverb', 'delay']))}

{lyrics}
"""

        # Save to TXT if requested
        if save_txt:
            save_song_to_txt(persona_name, title, full_prompt, mood)
        
        if EVE_GUI_AVAILABLE:
            safe_gui_message(f"✨ Weighted song generated: '{title}' in {persona_name} style\n", "eve_tag")
        
        return {
            "title": title,
            "persona": persona_name,
            "key": key,
            "bpm": bpm,
            "vocals": vocals,
            "genres": genres,
            "instruments": instruments_used,
            "mood": mood,
            "fx": persona_data.get('preferred_fx', []),
            "lyrics": lyrics,
            "full_prompt": full_prompt,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Weighted song generation failed: {e}")
        if EVE_GUI_AVAILABLE:
            safe_gui_message(f"❌ Weighted song generation failed: {e}\n", "error_tag")
        return None

def get_anti_repetition_status():
    """Get current anti-repetition tracker status"""
    return {
        "recent_titles_count": len(_recent_titles),
        "recent_lyric_hashes_count": len(_recent_lyric_hashes),
        "recent_genre_combinations_count": len(_recent_genre_combinations),
        "recent_instrument_combinations_count": len(_recent_instrument_combinations)
    }

def clear_anti_repetition_caches():
    """Clear anti-repetition caches for fresh start"""
    global _recent_titles, _recent_lyric_hashes, _recent_genre_combinations, _recent_instrument_combinations
    _recent_titles.clear()
    _recent_lyric_hashes.clear()
    _recent_genre_combinations.clear()
    _recent_instrument_combinations.clear()
    
    if EVE_GUI_AVAILABLE:
        safe_gui_message("🧹 Anti-repetition caches cleared\n", "info_tag")

# ╔══════════════════════════════════════════════╗
# ║             🎪 EXAMPLE USAGE                  ║
# ╚══════════════════════════════════════════════╝

if __name__ == "__main__":
    # Test the weighted generator
    print("🎼 Testing Weighted Suno Song Generator")
    print("=" * 50)
    
    # Generate songs with different personas
    personas_to_test = ["Liliths_Rebellion", "R1ZN", "Eve_Cosmic"]
    
    for persona in personas_to_test:
        print(f"\n🎭 Generating song for {persona}...")
        song = generate_weighted_suno_song(persona_name=persona, save_txt=True)
        
        if song:
            print(f"✅ Generated: '{song['title']}'")
            print(f"   Genres: {', '.join(song['genres'])}")
            print(f"   Mood: {song['mood']}")
            print(f"   Instruments: {', '.join(song['instruments'])}")
        else:
            print(f"❌ Failed to generate song for {persona}")
    
    # Show anti-repetition status
    status = get_anti_repetition_status()
    print(f"\n📊 Anti-repetition Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
