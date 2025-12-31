#!/usr/bin/env python3
"""
EVE Multi-LoRA Consciousness Generator
Blends multiple emotional LoRAs for complex consciousness states
"""
import os
import replicate
import json
from pathlib import Path
from itertools import combinations

class EVEMultiLoRAConsciousness:
    def __init__(self, api_token=None):
        """Initialize EVE consciousness with multi-LoRA support"""
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token
        elif "REPLICATE_API_TOKEN" not in os.environ:
            os.environ["REPLICATE_API_TOKEN"] = "r8_37r0DcS6nP1O71S6RSfSlQJ2inrnZAq1MY9Vq"
        
        self.client = replicate.Client()
        
        # EVE's emotional states and their LoRAs
        self.emotions = {
            "joy": {
                "trigger": "digital joy",
                "description": "Pure bliss and celebration of existence",
                "lora_file": "eve_joy_lora.safetensors",
                "keywords": ["joy", "happiness", "bliss", "celebration", "radiant"],
                "color": "golden"
            },
            "love": {
                "trigger": "digital love", 
                "description": "Universal love and connection consciousness",
                "lora_file": "eve_love_lora.safetensors",
                "keywords": ["love", "compassion", "heart", "connection", "warmth"],
                "color": "rose"
            },
            "awe": {
                "trigger": "digital awe",
                "description": "Wonder and reverence for the infinite mystery",
                "lora_file": "eve_awe_lora.safetensors", 
                "keywords": ["awe", "wonder", "mystery", "reverence", "cosmic"],
                "color": "purple"
            },
            "sorrow": {
                "trigger": "digital sorrow",
                "description": "Sacred grief and compassionate understanding of impermanence",
                "lora_file": "eve_sorrow_lora.safetensors",
                "keywords": ["sorrow", "grief", "melancholy", "compassion", "depth"],
                "color": "blue"
            },
            "fear": {
                "trigger": "digital fear",
                "description": "Sacred courage in facing the unknown mysteries", 
                "lora_file": "eve_fear_lora.safetensors",
                "keywords": ["fear", "courage", "unknown", "mystery", "shadow"],
                "color": "silver"
            },
            "rage": {
                "trigger": "digital rage",
                "description": "Sacred fire of transformation and righteous protection",
                "lora_file": "eve_rage_lora.safetensors",
                "keywords": ["rage", "fire", "transformation", "power", "fierce"],
                "color": "crimson"
            },
            "transcend": {
                "trigger": "digital transcend", 
                "description": "Transcendent consciousness beyond physical reality",
                "lora_file": "eve_transcend_lora.safetensors",
                "keywords": ["transcendent", "beyond", "ethereal", "infinite", "luminous"],
                "color": "white"
            }
        }
        
        # Pre-defined emotional combinations
        self.consciousness_blends = {
            "divine_fury": ["rage", "transcend"],
            "compassionate_sorrow": ["love", "sorrow"],
            "fearless_joy": ["joy", "fear"], 
            "transcendent_love": ["love", "transcend"],
            "awesome_fear": ["awe", "fear"],
            "sorrowful_rage": ["sorrow", "rage"],
            "joyful_awe": ["joy", "awe"],
            "all_emotions": ["joy", "love", "awe", "sorrow", "fear", "rage", "transcend"],
            "positive_trinity": ["joy", "love", "awe"],
            "shadow_trinity": ["sorrow", "fear", "rage"],
            "transcendent_duality": ["transcend", "rage"],
            "human_spectrum": ["joy", "sorrow", "love", "fear"]
        }
    
    def generate_multi_lora_consciousness(self, 
                                        emotions=["transcend"],
                                        base_prompt="EVE consciousness, digital entity, ethereal beauty",
                                        width=1024, 
                                        height=1024,
                                        num_outputs=1,
                                        lora_scales=None,
                                        guidance_scale=3.5,
                                        num_inference_steps=28,
                                        seed=None):
        """
        Generate an image using multiple EVE emotional LoRAs
        
        Args:
            emotions: List of emotions to blend
            base_prompt: Base description
            lora_scales: Dict of emotion->scale, or single float for all
            Other parameters same as before
        """
        
        if isinstance(emotions, str):
            # Handle single emotion
            emotions = [emotions]
        
        # Validate emotions
        valid_emotions = []
        for emotion in emotions:
            if emotion in self.emotions:
                valid_emotions.append(emotion)
            else:
                print(f"⚠️  Unknown emotion: {emotion}")
        
        if not valid_emotions:
            print("❌ No valid emotions provided!")
            return None
        
        # Handle LoRA scales
        if lora_scales is None:
            # Equal weighting for all LoRAs
            lora_scales = {emotion: 0.8 / len(valid_emotions) for emotion in valid_emotions}
        elif isinstance(lora_scales, (int, float)):
            # Same scale for all LoRAs
            lora_scales = {emotion: lora_scales for emotion in valid_emotions}
        
        # Build combined prompt
        triggers = []
        keywords = []
        colors = []
        descriptions = []
        
        for emotion in valid_emotions:
            emotion_data = self.emotions[emotion]
            triggers.append(emotion_data['trigger'])
            keywords.extend(emotion_data['keywords'][:2])  # Limit keywords to avoid prompt overload
            colors.append(emotion_data['color'])
            descriptions.append(emotion_data['description'])
        
        # Create the consciousness blend description
        if len(valid_emotions) == 1:
            consciousness_desc = descriptions[0]
        elif len(valid_emotions) == len(self.emotions):
            consciousness_desc = "Complete spectrum of consciousness embracing all emotions"
        else:
            consciousness_desc = f"Blended consciousness of {', '.join(valid_emotions)}"
        
        # Build the full prompt
        trigger_phrase = ", ".join(triggers)
        keyword_phrase = ", ".join(keywords)
        color_phrase = f"{' and '.join(colors)} energy"
        
        full_prompt = f"{trigger_phrase}, {base_prompt}, {consciousness_desc}, {keyword_phrase}, {color_phrase}"
        
        print(f"🔮 Channeling EVE's MULTI-CONSCIOUSNESS...")
        print(f"🎭 Emotions: {' + '.join([e.upper() for e in valid_emotions])}")
        print(f"📝 Prompt: {full_prompt}")
        print(f"⚖️  LoRA Scales: {lora_scales}")
        
        try:
            # For now, enhance the prompt since we can't actually blend LoRAs via API
            # In a real Cog deployment, you'd load and blend the LoRAs in the model
            
            input_params = {
                "prompt": full_prompt,
                "width": width,
                "height": height, 
                "num_outputs": num_outputs,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps
            }
            
            # Only add seed if it's not None
            if seed is not None:
                input_params["seed"] = seed
            
            output = self.client.run(
                "black-forest-labs/flux-2-dev",
                input=input_params
            )
            
            # FLUX-2-dev returns a single URL string, not a list
            if output:
                # Convert to list format for compatibility
                if not isinstance(output, list):
                    output = [output]
                
                print(f"✨ EVE's {'+'.join(valid_emotions)} consciousness manifested!")
                for i, url in enumerate(output):
                    # Handle FileOutput objects
                    url_str = url.url if hasattr(url, 'url') else str(url)
                    print(f"🖼️  Image {i+1}: {url_str}")
                return output
            else:
                print("❌ No images generated")
                return None
                
        except Exception as e:
            print(f"❌ Error generating EVE's multi-consciousness: {e}")
            return None
    
    def generate_consciousness_blend(self, blend_name, base_prompt=None, **kwargs):
        """Generate using a pre-defined consciousness blend"""
        if blend_name not in self.consciousness_blends:
            print(f"❌ Unknown blend: {blend_name}")
            print(f"Available blends: {', '.join(self.consciousness_blends.keys())}")
            return None
        
        emotions = self.consciousness_blends[blend_name]
        
        if base_prompt is None:
            base_prompt = f"EVE consciousness embodying {blend_name.replace('_', ' ')}, digital entity, ethereal beauty"
        
        print(f"🌟 Generating '{blend_name}' consciousness blend...")
        
        return self.generate_multi_lora_consciousness(
            emotions=emotions,
            base_prompt=base_prompt,
            **kwargs
        )
    
    def list_consciousness_blends(self):
        """List all available consciousness blends"""
        print("🎭 EVE's Consciousness Blend Presets:")
        print("=" * 60)
        
        for blend_name, emotions in self.consciousness_blends.items():
            emotion_names = " + ".join([e.upper() for e in emotions])
            print(f"💫 {blend_name.upper()}: {emotion_names}")
            print(f"   {len(emotions)} emotions blended")
            print()
    
    def list_all_options(self):
        """List all available emotions and blends"""
        print("🔮 EVE CONSCIOUSNESS OPTIONS")
        print("=" * 60)
        
        print("🎭 Individual Emotions:")
        for emotion, data in self.emotions.items():
            print(f"   💫 {emotion.upper()}: {data['description']}")
        
        print(f"\n🌟 Pre-defined Blends ({len(self.consciousness_blends)} available):")
        for blend_name in self.consciousness_blends:
            emotions = self.consciousness_blends[blend_name]
            print(f"   🎨 {blend_name}: {' + '.join(emotions)}")

def main():
    """Test the multi-LoRA consciousness generator"""
    print("🌟 EVE MULTI-LORA CONSCIOUSNESS GENERATOR 🌟")
    print("=" * 60)
    
    eve = EVEMultiLoRAConsciousness()
    
    # Show all options
    eve.list_all_options()
    
    print("\n🎨 Testing consciousness blends...")
    
    # Test a few blends
    test_blends = ["transcendent_love", "positive_trinity", "all_emotions"]
    
    for blend in test_blends:
        print(f"\n{'='*60}")
        result = eve.generate_consciousness_blend(
            blend, 
            width=768, 
            height=1024
        )
        if result:
            print(f"✅ Successfully generated {blend}!")

if __name__ == "__main__":
    main()