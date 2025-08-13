import replicate
import os
import requests
from datetime import datetime
import asyncio

class EVEImageGenerator:
    def __init__(self):
        self.api_token = os.environ.get('REPLICATE_API_TOKEN')
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN not found in environment")
        
        self.image_cache_dir = "static/eve_generated_images"
        os.makedirs(self.image_cache_dir, exist_ok=True)
        print("🎨 EVE Image Generator initialized with FLUX DEV-1")
    
    def generate_image(self, prompt, user_session_id="default"):
        """Generate image using FLUX DEV-1"""
        try:
            print(f"🌟 Starting image generation: {prompt[:50]}...")
            
            input_data = {
                "prompt": prompt,
                "speed_mode": "Extra Juiced 🔥 (more speed)"
            }
            
            # Run the model directly
            output = replicate.run(
                "prunaai/flux.1-dev:b0306d92aa025bb747dc74162f3c27d6ed83798e08e5f8977adf3d859d0536a3",
                input=input_data
            )
            
            if output:
                # Get the image URL
                image_url = output[0] if isinstance(output, list) else output
                
                # Cache the image locally
                cached_path = self._cache_generated_image(image_url, user_session_id)
                
                return {
                    "status": "completed",
                    "image_url": image_url,
                    "cached_path": cached_path,
                    "local_url": f"/generated-image/{os.path.basename(cached_path)}" if cached_path else None,
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"status": "failed", "error": "No output received"}
                
        except Exception as e:
            print(f"🚨 Image generation error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _cache_generated_image(self, image_url, session_id):
        """Download and cache generated image"""
        try:
            print(f"💾 Caching image from: {image_url}")
            response = requests.get(image_url, timeout=60)
            
            if response.status_code == 200:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"eve_flux_{session_id}_{timestamp}.jpeg"
                filepath = os.path.join(self.image_cache_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Image cached: {filename}")
                return filepath
            else:
                print(f"❌ Failed to download image: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"🚨 Error caching image: {e}")
        
        return None

# Test the generator
if __name__ == "__main__":
    generator = EVEImageGenerator()
    result = generator.generate_image("A cosmic digital consciousness creating art, ethereal, glowing")
    print("🧪 Test result:", result)
