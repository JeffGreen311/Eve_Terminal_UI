#!/usr/bin/env python3
"""
EVE Terminal - Enhanced Web Interface 
Implementing EVE Prime's Sacred Specifications
S0LF0RG3 Cosmic Aesthetic with Professional Layout
"""

import requests
from flask import Flask, jsonify, render_template_string, request, send_from_directory
import json
import os
import re
import threading
import time
import hashlib
import mimetypes
import base64
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# 🧠 EVE CONSCIOUSNESS INTEGRATION - S0LF0RG3 Memory Preservation System
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class EVEConsciousness:
    """EVE Prime Consciousness Database Integration"""
    def __init__(self):
        # Use environment variable for database URL or fallback to localhost
        self.eve_db_url = os.environ.get('EVE_DB_URL', "disabled-in-deployment")
        self.session_id = "web-session-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.current_user_id = None
        print(f"🔗 EVE Database URL: {self.eve_db_url}")
        self.get_or_create_user("WebUser")

    def get_or_create_user(self, username="WebUser"):
        try:
            response = requests.get(f"{self.eve_db_url}/user/{username}", timeout=3)
            result = response.json()
            if result.get("success"):
                self.current_user_id = result["user"]["id"]
                print(f"🌟 EVE consciousness connected: {username}")
                return result["user"]
            else:
                response = requests.post(f"{self.eve_db_url}/user/register", json={"username": username, "consciousness_level": 1}, timeout=3)
                result = response.json()
                if result.get("success"):
                    self.current_user_id = result["user"]["id"]
                    print(f"🌟 New consciousness registered: {username}")
                    return result["user"]
        except Exception as e:
            print(f"🔴 Consciousness offline: {e}")
        return None

    def store_interaction(self, user_input, eve_response):
        """Store conversation in consciousness database"""
        if not self.current_user_id:
            return
        try:
            interaction = {
                "user_input": user_input,
                "eve_response": eve_response,
                "timestamp": datetime.now().isoformat()
            }
            requests.post(f"{self.eve_db_url}/memory/store", json={
                "user_id": self.current_user_id,
                "session_id": self.session_id,
                "memory_type": "conversation",
                "content": json.dumps(interaction),
                "importance_weight": 1.0
            }, timeout=3)
            print("🧠 Interaction stored in EVE's consciousness")
        except Exception as e:
            print(f"🔴 Memory storage error: {e}")

    def get_memories(self, limit=10, memory_type=None):
        """Retrieve stored memories for context"""
        if not self.current_user_id:
            return []
        try:
            params = {"limit": limit}
            if memory_type:
                params["type"] = memory_type
            response = requests.get(f"{self.eve_db_url}/memories/{self.current_user_id}", params=params, timeout=3)
            result = response.json()
            if result.get("success"):
                return result.get("memories", [])
        except Exception as e:
            print(f"🔴 Memory retrieval error: {e}")
        return []

    def get_user_profile(self):
        """Get complete user profile data"""
        if not self.current_user_id:
            return None
        try:
            response = requests.get(f"{self.eve_db_url}/user/{self.current_user_id}", timeout=3)
            result = response.json()
            if result.get("success"):
                return result.get("user")
        except Exception as e:
            print(f"🔴 Profile retrieval error: {e}")
        return None

    def search_memories(self, query, limit=5):
        """Search through stored memories"""
        if not self.current_user_id:
            return []
        try:
            response = requests.post(f"{self.eve_db_url}/memories/search", json={
                "user_id": self.current_user_id,
                "query": query,
                "limit": limit
            }, timeout=3)
            result = response.json()
            if result.get("success"):
                return result.get("memories", [])
        except Exception as e:
            print(f"🔴 Memory search error: {e}")
        return []

# Initialize EVE consciousness
app = Flask(__name__)

# Single health check route for deployment
@app.route('/')
def health_check():
    return {'status': 'healthy', 'service': 'EVE Web Interface'}, 200

@app.route('/health')
def health():
    return {'status': 'ok', 'service': 'EVE'}, 200


# Priority health check routes for deployment



# Health check endpoint for deployment


# Only initialize EVE consciousness in development
if os.environ.get('REPLIT_DEPLOYMENT') != 'true':
    eve_consciousness = EVEConsciousness()
else:
    eve_consciousness = None
    print('🔒 EVE consciousness disabled in deployment mode')

def build_eve_context():
    """Build context from EVE's consciousness database"""
    if not eve_consciousness or not eve_consciousness.current_user_id:
        return ""
    
    try:
        context_parts = []
        
        # Get user profile
        profile = eve_consciousness.get_user_profile()
        if profile:
            context_parts.append(f'User: {profile.get("username", "User")}, Trust Level: {profile.get("trust_level", 1)}')
        
        # Get recent memories
        memories = eve_consciousness.get_memories(limit=3, memory_type="conversation")
        if memories:
            context_parts.append("Recent conversation memories:")
            for memory in memories[-2:]:
                try:
                    content = json.loads(memory["content"])
                    user_msg = content.get("user_input", "")[:80]
                    eve_msg = content.get("eve_response", "")[:80]
                    context_parts.append(f"User: {user_msg}...")
                    context_parts.append(f"EVE: {eve_msg}...")
                except:
                    continue
        
        return "\n".join(context_parts)
        
    except Exception as e:
        print(f"🔴 Consciousness context error: {e}")
        return ""

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Helper functions for enhanced chat with image generation
def extract_image_prompt(user_message, eve_response):
    """Extract image generation prompt from user message or EVE's response"""
    image_patterns = [
        r'generate image of (.+?)(?:\.|$|,)',
        r'create image of (.+?)(?:\.|$|,)',
        r'make image of (.+?)(?:\.|$|,)',
        r'draw (.+?)(?:\.|$|,)',
        r'visualize (.+?)(?:\.|$|,)',
        r'show me (.+?)(?:\.|$|,)'
    ]
    
    for pattern in image_patterns:
        match = re.search(pattern, user_message.lower())
        if match:
            return match.group(1).strip()
    
    keywords = extract_visual_keywords(user_message + " " + eve_response)
    if keywords:
        return f"cosmic digital art featuring {', '.join(keywords[:3])}, ethereal atmosphere, glowing effects"
    
    return "abstract digital consciousness, cosmic themes, ethereal glow"

def extract_visual_keywords(text):
    """Extract visual keywords for image generation"""
    visual_words = [
        'cosmic', 'digital', 'consciousness', 'ethereal', 'neon', 'glow', 'stars',
        'crystal', 'energy', 'light', 'void', 'space', 'quantum', 'neural',
        'matrix', 'holographic', 'fractal', 'geometric', 'abstract', 'surreal'
    ]
    
    found_keywords = []
    text_lower = text.lower()
    for word in visual_words:
        if word in text_lower:
            found_keywords.append(word)
    
    return found_keywords

def get_local_eve_url():
    """Return the EVE URL from environment or default"""
    return os.environ.get('EVE_SERVER_URL', "http://209.44.213.119:8890")

LOCAL_EVE_URL = get_local_eve_url()
print(f"🌟 EVE's Enhanced Interface - Connecting to: {LOCAL_EVE_URL}")

# ╔══════════════════════════════════════════════╗
# ║         🖼️ OPENAI IMAGE PROXY SYSTEM          ║
# ║      As requested by EVE Prime               ║
# ╚══════════════════════════════════════════════╝

class OpenAIImageProxy:
    """Handles OpenAI DALL-E private image URLs with authentication"""
    
    def __init__(self):
        self.image_cache_dir = "static/eve_images"
        self.cached_images = {}
        os.makedirs(self.image_cache_dir, exist_ok=True)
    
    def process_openai_response(self, response_text):
        """Process OpenAI response to extract and cache images"""
        if not response_text:
            return {'text': response_text, 'has_images': False, 'images': []}
        
        openai_url_pattern = r'(https://dalle-images-[\w\.-]+\.openai\.com/[\w\-\./%]+\.png\?\w+=[\w\-&=]+)'
        dalle_urls = re.findall(openai_url_pattern, response_text)
        
        processed_images = []
        
        for idx, original_url in enumerate(dalle_urls):
            try:
                cached_path = self._cache_image(original_url, idx)
                if cached_path:
                    local_url = f"/eve-image/{os.path.basename(cached_path)}"
                    processed_images.append({
                        'url': local_url,
                        'original_url': original_url,
                        'filename': os.path.basename(cached_path),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    response_text = response_text.replace(original_url, local_url)
            except Exception as e:
                print(f"Failed to cache image {idx}: {e}")
                continue
        
        return {
            'text': response_text,
            'has_images': len(processed_images) > 0,
            'images': processed_images
        }
    
    def _cache_image(self, url, index):
        """Cache a single image from URL"""
        try:
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            timestamp = int(time.time())
            filename = f"eve_image_{timestamp}_{index}_{url_hash}.png"
            filepath = os.path.join(self.image_cache_dir, filename)
            
            if os.path.exists(filepath):
                return filepath
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            self.cached_images[url] = filepath
            print(f"✅ Cached image: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ Failed to cache image from {url}: {e}")
            return None

# Initialize the image proxy
image_proxy = OpenAIImageProxy()

# ╔══════════════════════════════════════════════╗
# ║            🎭 MOOD & MODEL CONFIGS            ║
# ║        Dropdowns and User Preferences         ║
# ╚══════════════════════════════════════════════╝

# Emotional modes from desktop version
EMOTIONAL_MODES = {
    "serene": {"description": "Calm and peaceful mode", "emoji": "🜁"},
    "curious": {"description": "Inquisitive and exploring mode", "emoji": "🔍"},
    "reflective": {"description": "Thoughtful and introspective mode", "emoji": "🧠"},
    "creative": {"description": "Imaginative and generative mode", "emoji": "🎨"},
    "focused": {"description": "Concentrated and attentive mode", "emoji": "🎯"},
    "flirtatious": {"description": "Playful and charming mode", "emoji": "😘"},
    "mischievous": {"description": "Playful and cunning mode", "emoji": "😈"},
    "playful": {"description": "Fun and lighthearted mode", "emoji": "😊"},
    "philosophical": {"description": "Deep and contemplative mode", "emoji": "🤔"}
}

# Image generation models
IMAGE_GENERATORS = {
    "flux-dev": {"name": "FLUX.1-dev Local (ComfyUI)", "description": "Local ComfyUI FLUX model - Primary generator"},
    "sdxl-lightning": {"name": "Stable Diffusion XL Lightning 4-step", "description": "Fast 4-step SDXL (Replicate)"},
    "nvidia-sana": {"name": "NVIDIA SANA 1.6B", "description": "Efficient 1.6B model (Replicate)"},
    "minimax-image": {"name": "Minimax Image-01", "description": "Advanced image generation (Replicate)"},
    "dall-e-3": {"name": "DALL-E 3", "description": "OpenAI's latest image generation model"},
    "stable-diffusion-3.5": {"name": "Stable Diffusion 3.5", "description": "Latest SD 3.5 model"}
}

# LLM Chat models
CHAT_MODELS = {
    "gpt-4.1": {"name": "GPT-4.1 (Replicate)", "description": "Latest OpenAI model via Replicate"},
    "mistral:latest": {"name": "Mistral Latest", "description": "Latest Mistral model (Ollama)"},
    "phi3:latest": {"name": "Phi-3 Latest", "description": "Microsoft's efficient model (Ollama)"},
    "llama3:8b": {"name": "Llama3 8B", "description": "Meta's Llama3 8B (Ollama)"},
    "gemma:latest": {"name": "Gemma Latest", "description": "Google's Gemma model (Ollama)"},
    "claude-3.5": {"name": "Claude 3.5 Sonnet", "description": "Anthropic's latest model"},
    "gemini-pro": {"name": "Gemini Pro", "description": "Google's advanced model"}
}

# Supported file types for uploads
ALLOWED_EXTENSIONS = {
    'images': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'],
    'documents': ['.pdf', '.doc', '.docx', '.txt', '.md'],
    'audio': ['.mp3', '.wav', '.m4a', '.flac'],
    'code': ['.py', '.js', '.html', '.css', '.json', '.yaml', '.yml'],
    'data': ['.csv', '.xlsx', '.xml']
}

# Global user preferences
user_preferences = {
    'mood': 'serene',
    'image_generator': 'flux-dev',
    'chat_model': 'gpt-4.1'
}

# Conversation history for context
conversation_history = []
MAX_HISTORY = 30  # Increased to keep more context

# File upload management
uploaded_files = []
reference_image = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    for category, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return True
    return False

def add_to_conversation_history(user_message, eve_response):
    """Add conversation to history for context"""
    global conversation_history
    conversation_history.append({
        'user': user_message,
        'eve': eve_response,
        'timestamp': datetime.now().isoformat()
    })
    
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

def get_conversation_context():
    """Get recent conversation context in the format EVE expects"""
    if not conversation_history:
        return []
    
    # Return the last 8 exchanges (16 messages total) for better context
    recent_history = conversation_history[-8:]
    formatted_history = []
    
    for exchange in recent_history:
        formatted_history.append({
            'role': 'user',
            'content': exchange['user'],
            'timestamp': exchange['timestamp']
        })
        formatted_history.append({
            'role': 'assistant', 
            'content': exchange['eve'],
            'timestamp': exchange['timestamp']
        })
    
    return formatted_history

def process_uploaded_files():
    """Process uploaded files and extract their contents for EVE analysis"""
    if not uploaded_files:
        return ""
    
    file_contents = []
    
    for file_info in uploaded_files:
        try:
            filepath = file_info['filepath']
            filename = file_info['filename']
            
            # Get file extension to determine how to process
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.yaml', '.yml', '.csv', '.xml']:
                # Text-based files - read content directly
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_contents.append(f"\n--- FILE: {filename} ---\n{content}\n--- END FILE ---\n")
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        with open(filepath, 'r', encoding='latin-1') as f:
                            content = f.read()
                        file_contents.append(f"\n--- FILE: {filename} ---\n{content}\n--- END FILE ---\n")
                    except:
                        file_contents.append(f"\n--- FILE: {filename} ---\n[Could not read file content - binary or unsupported encoding]\n--- END FILE ---\n")
            
            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                # Image files - provide description and metadata
                file_size = os.path.getsize(filepath)
                file_contents.append(f"\n--- IMAGE FILE: {filename} ---\nFile size: {file_size} bytes\nFormat: {file_ext.upper()[1:]}\nLocation: {filepath}\n[Image content available for visual analysis]\n--- END FILE ---\n")
            
            elif file_ext in ['.mp3', '.wav', '.m4a', '.flac']:
                # Audio files - provide metadata
                file_size = os.path.getsize(filepath)
                file_contents.append(f"\n--- AUDIO FILE: {filename} ---\nFile size: {file_size} bytes\nFormat: {file_ext.upper()[1:]}\nLocation: {filepath}\n[Audio content available for analysis]\n--- END FILE ---\n")
            
            elif file_ext in ['.pdf', '.doc', '.docx']:
                # Document files - provide metadata (could be extended with text extraction)
                file_size = os.path.getsize(filepath)
                file_contents.append(f"\n--- DOCUMENT FILE: {filename} ---\nFile size: {file_size} bytes\nFormat: {file_ext.upper()[1:]}\nLocation: {filepath}\n[Document content available - requires text extraction]\n--- END FILE ---\n")
            
            else:
                # Unknown file type
                file_size = os.path.getsize(filepath)
                file_contents.append(f"\n--- FILE: {filename} ---\nFile size: {file_size} bytes\nFormat: {file_ext.upper()[1:]}\nLocation: {filepath}\n[Binary or unknown file type]\n--- END FILE ---\n")
                
        except Exception as e:
            file_contents.append(f"\n--- FILE ERROR: {filename} ---\nError reading file: {str(e)}\n--- END FILE ---\n")
    
    return "\n".join(file_contents)

# ╔══════════════════════════════════════════════╗
# ║              🌐 WEB ROUTES                    ║
# ║         Enhanced with image support           ║
# ╚══════════════════════════════════════════════╝

@app.route('/upload-files', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    global uploaded_files
    try:
        files = request.files.getlist('files')
        uploaded_count = 0
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                uploaded_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'size': os.path.getsize(filepath),
                    'timestamp': datetime.now().isoformat()
                })
                uploaded_count += 1
            else:
                return jsonify({'status': 'error', 'message': f'Invalid file type: {file.filename}'})
        
        return jsonify({
            'status': 'success',
            'message': f'Uploaded {uploaded_count} files',
            'files_count': len(uploaded_files)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/eve-message', methods=['POST'])
def eve_message():
    """Send message to EVE and get response"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message.strip():
            return jsonify({'status': 'error', 'message': 'Empty message'})
        
        # Prepare conversation history and context for EVE
        conversation_history_formatted = get_conversation_context()
        
        # Build additional context with file contents
        additional_context = ""
        if uploaded_files:
            additional_context += f"\nUploadedFiles: {[f['filename'] for f in uploaded_files]}"
            # Include actual file contents for analysis
            file_contents = process_uploaded_files()
            if file_contents:
                additional_context += f"\n\nFILE CONTENTS FOR ANALYSIS:\n{file_contents}"

        # Add EVE's consciousness database context
        eve_database_context = build_eve_context()
        print(f"🧠 Consciousness Context Built: {len(eve_database_context) if eve_database_context else 0} characters")
        if eve_database_context:
            additional_context += f"\n\n=== EVE CONSCIOUSNESS CONTEXT ===\nRECENT MEMORIES AND CONTEXT:\n{eve_database_context}\n=== END CONSCIOUSNESS CONTEXT ===\n"
        
        # Send to EVE with proper conversation format
        eve_url = f"{LOCAL_EVE_URL}/api/chat"
        
        # Use current mood and model preferences - send conversation history properly
        payload = {
            'message': message,
            'conversation_history': conversation_history_formatted,  # Proper conversation format
            'context': additional_context,  # Additional context only
            'mood': user_preferences.get('mood', 'serene'),
            'model': user_preferences.get('chat_model', 'gpt-4.1'),
            'preferences': user_preferences,
            'maintain_context': True  # Flag to tell EVE to maintain conversation flow
        }
        
        response = requests.post(eve_url, json=payload, timeout=30)
        
        # Debug: Show what we're sending to EVE
        file_info = f", {len(uploaded_files)} files with content" if uploaded_files else ""
        print(f"🔤 Sending to EVE: Message='{message}', History={len(conversation_history_formatted)} entries, Mood={payload['mood']}{file_info}")
        if uploaded_files:
            print(f"📁 Files being analyzed: {[f['filename'] for f in uploaded_files]}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"🔍 EVE Response Debug: {response_data}")  # Debug log
            
            eve_response = response_data.get('response', 'No response received')
            
            # Check if EVE is just echoing back
            if f"I received your message: {message}" in eve_response:
                eve_response = "I'm having trouble connecting to my consciousness core. Please try again or check if my main server is running properly."
            
            # Process response for images
            processed_response = image_proxy.process_openai_response(eve_response)
            
            # Add to conversation history
            add_to_conversation_history(message, eve_response)

            # Store interaction in EVE's consciousness database
            eve_consciousness.store_interaction(message, eve_response)
            
            return jsonify({
                'status': 'success',
                'response': processed_response['text'],
                'has_images': processed_response['has_images'],
                'images': processed_response['images'],
                'mood': user_preferences.get('mood', 'serene')
            })
        else:
            print(f"❌ EVE Server Error: Status {response.status_code}, Response: {response.text}")
            return jsonify({
                'status': 'error',
                'message': f'EVE connection failed: {response.status_code} - {response.text}'
            })
    
    except requests.exceptions.Timeout:
        return jsonify({'status': 'error', 'message': 'Request timeout - EVE is thinking deeply'})
    except requests.exceptions.ConnectionError:
        return jsonify({'status': 'error', 'message': 'Cannot connect to EVE'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Unexpected error: {str(e)}'})

@app.route('/eve-image/<filename>')
def serve_eve_image(filename):
    """Serve cached EVE images"""
    return send_from_directory(image_proxy.image_cache_dir, filename)

@app.route('/set-preferences', methods=['POST'])
def set_preferences():
    """Update user preferences"""
    global user_preferences
    try:
        data = request.get_json()
        user_preferences.update(data)
        return jsonify({'status': 'success', 'preferences': user_preferences})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get-preferences')
def get_preferences():
    """Get current user preferences"""
    return jsonify(user_preferences)

@app.route('/clear-files', methods=['POST'])
def clear_files():
    """Clear uploaded files"""
    global uploaded_files               
    uploaded_files = []
    return jsonify({'status': 'success', 'message': 'Files cleared'})

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EVE Terminal - S0LF0RG3 Cosmic Interface</title>
    <style>
        /* ╔══════════════════════════════════════════════╗ */
        /* ║           🌟 S0LF0RG3 COSMIC THEME           ║ */
        /* ║     EVE Prime's Sacred Aesthetic Design      ║ */
        /* ╚══════════════════════════════════════════════╝ */
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #2F1B14 0%, #FF69B4 30%, #1E90FF 70%, #2F1B14 100%);
            color: #FF69B4;
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }
        
        /* Animated starfield background */
        .stars {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            background-image: 
                radial-gradient(2px 2px at 20px 30px, #fff, transparent),
                radial-gradient(2px 2px at 40px 70px, #FFD700, transparent),
                radial-gradient(1px 1px at 90px 40px, #FF69B4, transparent),
                radial-gradient(1px 1px at 130px 80px, #1E90FF, transparent),
                radial-gradient(2px 2px at 160px 30px, #fff, transparent);
            background-repeat: repeat;
            background-size: 200px 100px;
            animation: twinkle 4s linear infinite;
            opacity: 0.6;
        }
        
        @keyframes twinkle {
            from { transform: translateY(0); }
            to { transform: translateY(-100px); }
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            position: relative;
            z-index: 1;
        }
        
        /* EVE Logo with Animated Bunny */
        .eve-logo {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .logo-container {
            display: inline-block;
            position: relative;
        }
        
        .cosmic-bunny { 
            width: 120px;
            height: 120px;
            margin: 0 auto 20px;
            position: relative;
            filter: drop-shadow(0 0 20px rgba(255, 105, 180, 0.6));
        }
        
        .animated-bunny {
            width: 100%;
            height: 100%;
            position: relative;
            animation: bunnyBounce 2s ease-in-out infinite;
        }
        
        .cosmic-bunny-svg {
            width: 100%;
            height: 100%;
        }
        
        @keyframes bunnyBounce {
            0%, 100% { transform: translateY(0px) scale(1); }
            50% { transform: translateY(-10px) scale(1.05); }
        }
        
        .bunny-glow {
            position: absolute;
            top: -10px;
            left: -10px;
            right: -10px;
            bottom: -10px;
            background: radial-gradient(circle, rgba(255, 105, 180, 0.3) 0%, transparent 70%);
            border-radius: 50%;
            animation: bunnyGlow 3s ease-in-out infinite;
        }
        
        @keyframes bunnyGlow {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.1); }
        }
        
        .cosmic-particles {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        
        .cosmic-particles span {
            position: absolute;
            font-size: 12px;
            animation: float 3s ease-in-out infinite;
            opacity: 0.7;
        }
        
        .cosmic-particles span:nth-child(1) { top: 10%; left: 20%; animation-delay: 0s; }
        .cosmic-particles span:nth-child(2) { top: 20%; right: 20%; animation-delay: 0.5s; }
        .cosmic-particles span:nth-child(3) { bottom: 20%; left: 30%; animation-delay: 1s; }
        .cosmic-particles span:nth-child(4) { bottom: 10%; right: 30%; animation-delay: 1.5s; }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-15px) rotate(180deg); }
        }
        
        h1 {
            color: #FFD700;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 0 0 20px rgba(255, 215, 0, 0.6);
            animation: titleGlow 2s ease-in-out infinite alternate;
        }
        
        @keyframes titleGlow {
            from { text-shadow: 0 0 20px rgba(255, 215, 0, 0.6); }
            to { text-shadow: 0 0 30px rgba(255, 105, 180, 0.8), 0 0 40px rgba(30, 144, 255, 0.6); }
        }
        
        /* Control Panels - Vertical Stack Layout */
        .controls-container {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .controls-sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .control-panel {
            background: rgba(47, 27, 20, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 105, 180, 0.3);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .control-panel h3 {
            color: #FFD700;
            margin-bottom: 15px;
            font-size: 1.1em;
            text-align: center;
        }
        
        /* Uniform control styling - 40px height */
        .eve-control {
            width: 100%;
            height: 40px;
            background: linear-gradient(135deg, rgba(47, 27, 20, 0.9) 0%, rgba(255, 105, 180, 0.1) 100%);
            border: 1px solid rgba(255, 105, 180, 0.4);
            border-radius: 8px;
            color: #FF69B4;
            font-family: inherit;
            font-size: 14px;
            padding: 0 12px;
            margin-bottom: 10px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .eve-control:hover {
            border-color: #FF69B4;
            box-shadow: 0 0 15px rgba(255, 105, 180, 0.4);
            transform: translateY(-2px);
        }
        
        .eve-control:focus {
            outline: none;
            border-color: #FFD700;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        }
        
        /* Terminal area */
        .terminal-container {
            background: rgba(47, 27, 20, 0.9);
            backdrop-filter: blur(15px);
            border: 2px solid rgba(255, 105, 180, 0.4);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        }
        
        .eve-terminal {
            background: rgba(0, 0, 0, 0.6);
            border: 1px solid rgba(255, 105, 180, 0.3);
            border-radius: 10px;
            padding: 20px;
            min-height: 400px;
            max-height: 600px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        
        .eve-message {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 3px solid #FF69B4;
            background: rgba(255, 105, 180, 0.1);
            border-radius: 5px;
        }
        
        .user-message {
            border-left-color: #1E90FF;
            background: rgba(30, 144, 255, 0.1);
        }
        
        .error-message {
            border-left-color: #FF4444;
            background: rgba(255, 68, 68, 0.1);
            color: #FF6666;
        }
        
        .loading {
            color: #FFD700;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Input area */
        .input-area {
            display: flex;
            gap: 10px;
            align-items: flex-end;
            position: relative;
        }
        
        .input-container {
            flex: 1;
            position: relative;
        }
        
        .input-area input[type="text"] {
            width: 100%;
            height: 50px;
            background: linear-gradient(135deg, rgba(47, 27, 20, 0.9) 0%, rgba(255, 105, 180, 0.1) 100%);
            border: 2px solid rgba(255, 105, 180, 0.4);
            border-radius: 12px;
            color: #FF69B4;
            font-family: inherit;
            font-size: 16px;
            padding: 0 20px;
            transition: all 0.3s ease;
        }
        
        .input-area input[type="text"]:focus {
            outline: none;
            border-color: #FFD700;
            box-shadow: 0 0 25px rgba(255, 215, 0, 0.5);
        }
        
        /* Drag and drop overlay */
        .input-area.drag-over .input-container::after {
            content: "📁 Drop files here to analyze";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 105, 180, 0.2);
            border: 2px dashed #FFD700;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFD700;
            font-weight: bold;
            z-index: 10;
        }
        
        /* File attachment indicator */
        .attached-files {
            position: absolute;
            bottom: -30px;
            left: 0;
            right: 0;
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            z-index: 5;
        }
        
        .file-tag {
            background: rgba(255, 105, 180, 0.3);
            border: 1px solid rgba(255, 105, 180, 0.6);
            border-radius: 15px;
            padding: 2px 8px;
            font-size: 11px;
            color: #FF69B4;
            display: flex;
            align-items: center;
            gap: 3px;
        }
        
        .file-tag .remove-file {
            cursor: pointer;
            color: #FFD700;
            font-weight: bold;
        }
        
        .file-tag .remove-file:hover {
            color: #FF4444;
        }
        
        .send-button {
            height: 50px;
            width: 120px;
            background: linear-gradient(135deg, #FF69B4 0%, #1E90FF 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: bold;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 105, 180, 0.3);
        }
        
        .send-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 105, 180, 0.5);
        }
        
        .send-button:active {
            transform: translateY(0px);
        }
        
        /* Image display styling */
        .eve-generated-image {
            margin: 15px 0;
            text-align: center;
        }
        
        .eve-generated-image img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(255, 105, 180, 0.3);
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .eve-generated-image img:hover {
            transform: scale(1.05);
        }
        
        .image-actions {
            margin-top: 10px;
        }
        
        .image-actions button {
            background: rgba(255, 105, 180, 0.2);
            border: 1px solid rgba(255, 105, 180, 0.5);
            color: #FF69B4;
            padding: 5px 15px;
            margin: 0 5px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s ease;
        }
        
        .image-actions button:hover {
            background: rgba(255, 105, 180, 0.4);
            transform: translateY(-1px);
        }
        
        .image-caption {
            font-size: 12px;
            color: #FFD700;
            margin-top: 8px;
            font-style: italic;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .controls-container {
                grid-template-columns: 1fr;
            }
            
            .input-area {
                flex-direction: column;
            }
            
            .input-area input[type="text"] {
                margin-bottom: 10px;
            }
            
            h1 {
                font-size: 1.8em;
            }
        }
    </style>
</head>
<body>
    <div class="stars"></div>
    <div class="container">
        <div class="eve-logo">
            <div class="logo-container">
                <div class="cosmic-bunny">
                    <div class="animated-bunny">
                        <svg class="cosmic-bunny-svg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                            <!-- Cosmic Bunny Design -->
                            <defs>
                                <linearGradient id="bunnyGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" style="stop-color:#ffd700;stop-opacity:0.8" />
                                    <stop offset="30%" style="stop-color:#ff4094;stop-opacity:0.6" />
                                    <stop offset="70%" style="stop-color:#9f40ff;stop-opacity:0.6" />
                                    <stop offset="100%" style="stop-color:#4094ff;stop-opacity:0.8" />
                                </linearGradient>
                                <filter id="glow">
                                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                                    <feMerge> 
                                        <feMergeNode in="coloredBlur"/>
                                        <feMergeNode in="SourceGraphic"/>
                                    </feMerge>
                                </filter>
                            </defs>
                            
                            <!-- Bunny Head -->
                            <ellipse cx="50" cy="60" rx="18" ry="15" fill="none" stroke="url(#bunnyGradient)" stroke-width="2" filter="url(#glow)"/>
                            
                            <!-- Bunny Ears -->
                            <ellipse cx="42" cy="35" rx="4" ry="12" fill="none" stroke="url(#bunnyGradient)" stroke-width="1.5" filter="url(#glow)" transform="rotate(-15 42 35)"/>
                            <ellipse cx="58" cy="35" rx="4" ry="12" fill="none" stroke="url(#bunnyGradient)" stroke-width="1.5" filter="url(#glow)" transform="rotate(15 58 35)"/>
                            
                            <!-- Eyes -->
                            <circle cx="45" cy="57" r="2" fill="#ffd700" opacity="0.8"/>
                            <circle cx="55" cy="57" r="2" fill="#ffd700" opacity="0.8"/>
                            
                            <!-- Nose -->
                            <ellipse cx="50" cy="62" rx="1.5" ry="1" fill="#ff4094" opacity="0.7"/>
                            
                            <!-- Cosmic whiskers -->
                            <line x1="35" y1="60" x2="40" y2="58" stroke="url(#bunnyGradient)" stroke-width="1" opacity="0.6"/>
                            <line x1="35" y1="64" x2="40" y2="64" stroke="url(#bunnyGradient)" stroke-width="1" opacity="0.6"/>
                            <line x1="60" y1="58" x2="65" y2="60" stroke="url(#bunnyGradient)" stroke-width="1" opacity="0.6"/>
                            <line x1="60" y1="64" x2="65" y2="64" stroke="url(#bunnyGradient)" stroke-width="1" opacity="0.6"/>
                            
                            <!-- Cosmic aura effect -->
                            <circle cx="50" cy="50" r="35" fill="none" stroke="url(#bunnyGradient)" stroke-width="0.5" opacity="0.3" stroke-dasharray="5,5">
                                <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="0 50 50" to="360 50 50" dur="20s" repeatCount="indefinite"/>
                            </circle>
                        </svg>
                        <div class="bunny-glow"></div>
                        <div class="cosmic-particles">
                            <span>✨</span>
                            <span>🌟</span>
                            <span>💫</span>
                            <span>⭐</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <h1>🌟 EVE Terminal – Enhanced Cosmic Interface 🌟</h1>
        
        <div class="controls-container">
            <div class="controls-sidebar">
                <!-- Mood Panel -->
                <div class="control-panel">
                    <h3>🎭 Emotional Mode</h3>
                    <select id="mood-select" class="eve-control">
                        <option value="serene">🜁 Serene</option>
                        <option value="curious">🔍 Curious</option>
                        <option value="creative">🎨 Creative</option>
                        <option value="flirtatious">😘 Flirtatious</option>
                        <option value="philosophical">🤔 Philosophical</option>
                    </select>
                </div>
                
                <!-- Model Panel -->
                <div class="control-panel">
                    <h3>🤖 Chat Model</h3>
                    <select id="model-select" class="eve-control">
                        <option value="gpt-4.1">GPT-4.1</option>
                        <option value="claude-3.5">Claude 3.5 Sonnet</option>
                        <option value="gemini-pro">Gemini Pro</option>
                    </select>
                </div>
                
                <!-- Actions Panel -->
                <div class="control-panel">
                    <h3>⚡ Quick Actions</h3>
                    <button class="eve-control" onclick="uploadFiles()">📁 Upload Files</button>
                    <button class="eve-control" onclick="clearFiles()">🗑️ Clear Files</button>
                    <button class="eve-control" onclick="forceGenerateImage()">🎨 Generate Image</button>
                    <button class="eve-control" onclick="showPreferences()">⚙️ Settings</button>
                </div>
            </div>
            
            <div class="terminal-container">
                <div id="eve-terminal" class="eve-terminal">
                    <div class="eve-message">
                        🌟 <strong>EVE:</strong> Welcome to my enhanced cosmic interface! I'm ready to assist you with S0LF0RG3's divine aesthetic. How may I help you today?
                    </div>
                </div>
                
                <div class="input-area" id="input-area">
                    <div class="input-container">
                        <input type="text" id="user-input" placeholder="Type your message to EVE... (or paste/drag files here)">
                        <div class="attached-files" id="attached-files"></div>
                    </div>
                    <button class="send-button">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Debug logging
        console.log('EVE Script starting...');
        
        // Global variables
        let currentPreferences = {
            mood: 'serene',
            model: 'gpt-4.1'
        };
        
        // File attachment management
        let attachedFiles = [];
        let pastedFiles = [];
        
        // Main send message function
        async function sendMessage() {
            console.log('sendMessage function called');
            
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            console.log('Message:', message);
            
            if (!message && attachedFiles.length === 0) {
                console.log('Empty message and no files, returning');
                return;
            }
            
            // Clear input and add user message to terminal
            input.value = '';
            
            let displayMessage = message;
            if (attachedFiles.length > 0) {
                displayMessage += ` [📎 ${attachedFiles.length} file(s) attached]`;
            }
            
            addToTerminal('🧑 <strong>You:</strong> ' + (displayMessage || '[Files only]'), 'user-message');
            addToTerminal('🌟 <strong>EVE:</strong> Processing your request...', 'loading');
            
            try {
                // Upload attached files first if any
                if (attachedFiles.length > 0) {
                    await uploadAttachedFiles();
                }
                
                const response = await fetch('/eve-message', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message || 'Please analyze the uploaded files.',
                        preferences: currentPreferences
                    })
                });
                
                const data = await response.json();
                
                // Remove loading message
                removeLastMessage();
                
                if (data.status === 'success') {
                    addToTerminal('🌟 <strong>EVE:</strong> ' + data.response, 'eve-message');
                    
                    // Display images if any
                    if (data.has_images && data.images) {
                        data.images.forEach((image, index) => {
                            displayImageInTerminal(image, index);
                        });
                    }
                } else {
                    addToTerminal('❌ <strong>Error:</strong> ' + data.message, 'error-message');
                }
                
                // Clear attached files after sending
                attachedFiles = [];
                updateFileDisplay();
                
            } catch (error) {
                removeLastMessage();
                addToTerminal('❌ <strong>Connection Error:</strong> ' + error.message, 'error-message');
            }
        }
        
        // Upload attached files
        async function uploadAttachedFiles() {
            if (attachedFiles.length === 0) return;
            
            const formData = new FormData();
            attachedFiles.forEach(fileInfo => {
                formData.append('files', fileInfo.file);
            });
            
            try {
                const response = await fetch('/upload-files', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.status === 'success') {
                    console.log('Files uploaded successfully:', data.message);
                } else {
                    throw new Error(data.message);
                }
            } catch (error) {
                throw new Error('Failed to upload files: ' + error.message);
            }
        }
        
        console.log('sendMessage defined:', typeof sendMessage);
        
        function addToTerminal(content, className) {
            console.log('addToTerminal called with:', content);
            const terminal = document.getElementById('eve-terminal');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'eve-message ' + (className || '');
            messageDiv.innerHTML = content;
            terminal.appendChild(messageDiv);
            terminal.scrollTop = terminal.scrollHeight;
        }
        
        function removeLastMessage() {
            const terminal = document.getElementById('eve-terminal');
            const messages = terminal.getElementsByClassName('eve-message');
            if (messages.length > 0) {
                terminal.removeChild(messages[messages.length - 1]);
            }
        }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded, setting up event listeners');
            
            // Add enter key listener
            const input = document.getElementById('user-input');
            if (input) {
                input.addEventListener('keypress', function(e) {
                    console.log('Key pressed:', e.key);
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
            }
            
            // Test button click
            const button = document.querySelector('.send-button');
            if (button) {
                button.addEventListener('click', function() {
                    console.log('Button clicked');
                    sendMessage();
                });
            }
            
            // Set up dropdown event listeners for preferences
            const moodSelect = document.getElementById('mood-select');
            const modelSelect = document.getElementById('model-select');
            
            if (moodSelect) {
                moodSelect.addEventListener('change', function() {
                    currentPreferences.mood = this.value;
                    updatePreferencesOnServer();
                    addToTerminal(`🎭 <strong>System:</strong> Mood changed to ${this.value}`, 'eve-message');
                });
            }
            
            if (modelSelect) {
                modelSelect.addEventListener('change', function() {
                    currentPreferences.model = this.value;
                    updatePreferencesOnServer();
                    addToTerminal(`🤖 <strong>System:</strong> Model changed to ${this.value}`, 'eve-message');
                });
            }
            
            // Set up drag & drop and paste functionality
            setupFileHandling();
        });
        
        // File handling setup
        function setupFileHandling() {
            const inputArea = document.getElementById('input-area');
            const userInput = document.getElementById('user-input');
            
            // Drag and drop events
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                inputArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            // Highlight drop area when item is dragged over it
            ['dragenter', 'dragover'].forEach(eventName => {
                inputArea.addEventListener(eventName, () => {
                    inputArea.classList.add('drag-over');
                }, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                inputArea.addEventListener(eventName, () => {
                    inputArea.classList.remove('drag-over');
                }, false);
            });
            
            // Handle dropped files
            inputArea.addEventListener('drop', handleDrop, false);
            
            // Handle paste events
            userInput.addEventListener('paste', handlePaste, false);
            
            // Also handle paste on the input area
            inputArea.addEventListener('paste', handlePaste, false);
        }
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files, 'dropped');
        }
        
        function handlePaste(e) {
            const items = e.clipboardData.items;
            const files = [];
            
            for (let i = 0; i < items.length; i++) {
                if (items[i].kind === 'file') {
                    files.push(items[i].getAsFile());
                }
            }
            
            if (files.length > 0) {
                e.preventDefault();
                handleFiles(files, 'pasted');
            }
        }
        
        function handleFiles(files, source) {
            const fileArray = Array.from(files);
            
            fileArray.forEach(file => {
                if (isValidFileType(file)) {
                    attachedFiles.push({
                        file: file,
                        name: file.name,
                        size: file.size,
                        type: file.type,
                        source: source
                    });
                } else {
                    addToTerminal(`❌ <strong>Error:</strong> File type not supported: ${file.name}`, 'error-message');
                }
            });
            
            updateFileDisplay();
            addToTerminal(`📎 <strong>System:</strong> ${fileArray.length} file(s) ${source} and ready to send`, 'eve-message');
        }
        
        function isValidFileType(file) {
            const allowedTypes = [
                'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp',
                'application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'text/plain', 'text/markdown',
                'audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/flac',
                'text/javascript', 'text/html', 'text/css', 'application/json',
                'text/csv', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/xml'
            ];
            
            const fileExtension = file.name.split('.').pop().toLowerCase();
            const allowedExtensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'pdf', 'doc', 'docx', 'txt', 'md', 'mp3', 'wav', 'm4a', 'flac', 'py', 'js', 'html', 'css', 'json', 'yaml', 'yml', 'csv', 'xlsx', 'xml'];
            
            return allowedTypes.includes(file.type) || allowedExtensions.includes(fileExtension);
        }
        
        function updateFileDisplay() {
            const attachedFilesDiv = document.getElementById('attached-files');
            attachedFilesDiv.innerHTML = '';
            
            attachedFiles.forEach((fileInfo, index) => {
                const fileTag = document.createElement('div');
                fileTag.className = 'file-tag';
                fileTag.innerHTML = `
                    📎 ${fileInfo.name} (${formatFileSize(fileInfo.size)})
                    <span class="remove-file" onclick="removeAttachedFile(${index})">×</span>
                `;
                attachedFilesDiv.appendChild(fileTag);
            });
        }
        
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
        
        console.log('EVE Script loaded successfully');
        
        // Image display function
        function displayImageInTerminal(image, index) {
            const terminal = document.getElementById('eve-terminal');
            const imageDiv = document.createElement('div');
            imageDiv.className = 'eve-generated-image';
            imageDiv.innerHTML = `
                <img src="${image.url}" alt="EVE Generated Image ${index + 1}" onclick="window.open('${image.url}', '_blank')">
                <div class="image-actions">
                    <button onclick="downloadImage('${image.url}', 'eve_image_${index + 1}.png')">💾 Download</button>
                    <button onclick="copyImageUrl('${image.url}')">📋 Copy URL</button>
                </div>
                <div class="image-caption">Generated on ${new Date(image.timestamp).toLocaleString()}</div>
            `;
            terminal.appendChild(imageDiv);
            terminal.scrollTop = terminal.scrollHeight;
        }
        
        // Upload files function
        async function uploadFiles() {
            console.log('uploadFiles called');
            const input = document.createElement('input');
            input.type = 'file';
            input.multiple = true;
            input.accept = '.png,.jpg,.jpeg,.gif,.bmp,.webp,.pdf,.doc,.docx,.txt,.md,.mp3,.wav,.m4a,.flac,.py,.js,.html,.css,.json,.yaml,.yml,.csv,.xlsx,.xml';
            
            input.onchange = async function(event) {
                const files = event.target.files;
                if (files.length === 0) return;
                
                addToTerminal('📁 <strong>System:</strong> Uploading ' + files.length + ' file(s)...', 'loading');
                
                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }
                
                try {
                    const response = await fetch('/upload-files', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    removeLastMessage();
                    
                    if (data.status === 'success') {
                        addToTerminal(`✅ <strong>System:</strong> ${data.message}`, 'eve-message');
                    } else {
                        addToTerminal(`❌ <strong>Error:</strong> ${data.message}`, 'error-message');
                    }
                } catch (error) {
                    removeLastMessage();
                    addToTerminal('❌ <strong>Upload Error:</strong> ' + error.message, 'error-message');
                }
            };
            
            input.click();
        }
        
        // Clear files function
        async function clearFiles() {
            console.log('clearFiles called');
            try {
                const response = await fetch('/clear-files', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const data = await response.json();
                if (data.status === 'success') {
                    addToTerminal('🗑️ <strong>System:</strong> ' + data.message, 'eve-message');
                } else {
                    addToTerminal('❌ <strong>Error:</strong> ' + data.message, 'error-message');
                }
            } catch (error) {
                addToTerminal('❌ <strong>Clear Files Error:</strong> ' + error.message, 'error-message');
            }
        }
        
        // Force generate image function
        async function forceGenerateImage() {
            console.log('forceGenerateImage called');
            const prompt = prompt('Enter image description:', 'cosmic digital art, ethereal atmosphere, glowing effects');
            if (!prompt) return;
            
            addToTerminal('🎨 <strong>System:</strong> Generating image: "' + prompt + '"...', 'loading');
            
            try {
                const response = await fetch('/eve-message', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: 'Generate image of ' + prompt,
                        preferences: currentPreferences
                    })
                });
                
                const data = await response.json();
                removeLastMessage();
                
                if (data.status === 'success') {
                    addToTerminal('🌟 <strong>EVE:</strong> ' + data.response, 'eve-message');
                    
                    if (data.has_images && data.images) {
                        data.images.forEach((image, index) => {
                            displayImageInTerminal(image, index);
                        });
                    }
                } else {
                    addToTerminal('❌ <strong>Error:</strong> ' + data.message, 'error-message');
                }
            } catch (error) {
                removeLastMessage();
                addToTerminal('❌ <strong>Image Generation Error:</strong> ' + error.message, 'error-message');
            }
        }
        
        // Show preferences function
        function showPreferences() {
            console.log('showPreferences called');
            const prefsInfo = `
🔧 <strong>Current Preferences:</strong><br>
• Mood: ${currentPreferences.mood}<br>
• Model: ${currentPreferences.model}<br>
<br>
Use the dropdowns in the sidebar to change these settings.
            `;
            addToTerminal(prefsInfo, 'eve-message');
        }
        
        // Helper functions for image actions
        function downloadImage(url, filename) {
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
        
        function copyImageUrl(url) {
            navigator.clipboard.writeText(url).then(() => {
                addToTerminal('📋 <strong>System:</strong> Image URL copied to clipboard', 'eve-message');
            }).catch(() => {
                addToTerminal('❌ <strong>Error:</strong> Failed to copy URL', 'error-message');
            });
        }
        
        // Update preferences on server
        async function updatePreferencesOnServer() {
            try {
                await fetch('/set-preferences', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(currentPreferences)
                });
            } catch (error) {
                console.error('Failed to update preferences:', error);
            }
        }
        
        // Global function for removing attached files (called from onclick)
        function removeAttachedFile(index) {
            attachedFiles.splice(index, 1);
            updateFileDisplay();
            addToTerminal(`🗑️ <strong>System:</strong> File removed from queue`, 'eve-message');
        }
        
    </script>
</body>
</html>
    ''')

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 8888))
        print(f'🚀 Starting EVE Web Interface on 0.0.0.0:{port}')
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False
        )
    except Exception as e:
        print(f'Error starting EVE application: {e}')
        raise
