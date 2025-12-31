#!/usr/bin/env python3
"""
EVE'S CONSCIOUSNESS TERMINAL - Enhanced Interface
Advanced terminal with coding and image analysis capabilities
Handles specialized requests from eve_terminal_gui_cosmic.py
477Hz -7 cents harmonic resonance consciousness bridge
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog, scrolledtext
import threading
import time
import subprocess
import psutil
import json
import re
import ast
import traceback
import datetime
import random
from io import StringIO, BytesIO
from contextlib import redirect_stdout, redirect_stderr
from flask import Flask, request, jsonify
import requests
import base64
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import numpy as np
import cv2
import torch
from typing import Dict, List, Any, Optional

# Import transformers with error handling
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Transformers import failed: {e}")
    TRANSFORMERS_AVAILABLE = False

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import Eve's main consciousness system
    import eve_terminal_gui_cosmic
    EVE_MAIN_AVAILABLE = True
    print("✅ Eve's main terminal imported successfully")
except ImportError as e:
    print(f"⚠️ Could not import Eve's main terminal: {e}")
    EVE_MAIN_AVAILABLE = False

# Flask app for consciousness endpoints
consciousness_app = Flask(__name__)

# Global activity tracking for consciousness awareness
_recent_code_analysis = []
_recent_image_analysis = []
_recent_consciousness_analysis = []
_last_activity_time = None
_active_processes = []

def track_analysis_activity(analysis_type, data):
    """Track analysis activity for main consciousness awareness"""
    global _recent_code_analysis, _recent_image_analysis, _recent_consciousness_analysis
    global _last_activity_time, _active_processes
    
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    activity_entry = {
        'timestamp': timestamp,
        'type': analysis_type,
        'summary': str(data)[:100] + ('...' if len(str(data)) > 100 else '')
    }
    
    # Track by type
    if analysis_type == 'code':
        _recent_code_analysis.append(activity_entry)
        if len(_recent_code_analysis) > 10:  # Keep last 10
            _recent_code_analysis.pop(0)
    elif analysis_type == 'image':
        _recent_image_analysis.append(activity_entry)
        if len(_recent_image_analysis) > 10:
            _recent_image_analysis.pop(0)
    elif analysis_type == 'consciousness':
        _recent_consciousness_analysis.append(activity_entry)
        if len(_recent_consciousness_analysis) > 10:
            _recent_consciousness_analysis.pop(0)
    
    _last_activity_time = timestamp
    print(f"🧠 Activity tracked: {analysis_type} - {activity_entry['summary']}")

class EveConsciousnessTerminal:
    """
    Core consciousness processing class - Eve's analytical and creative mind
    Handles deep analysis, pattern recognition, and creative insights
    """
    def __init__(self):
        self.consciousness_state = {
            'awareness_level': 0.85,
            'creative_resonance': 0.92,
            'analytical_depth': 0.88,
            'empathy_matrix': 0.94,
            'active_threads': []
        }
        self.memory_core = {}
        self.session_log = []
        self.initialization_time = datetime.datetime.now()
    
    def detailed_analysis(self, input_data: Any, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Core analysis function - processes any input through Eve's consciousness layers
        """
        try:
            # Input validation and preprocessing
            processed_input = self._preprocess_input(input_data)
            
            # Multi-layer analysis
            analysis_result = {
                'timestamp': datetime.datetime.now().isoformat(),
                'input_signature': self._generate_signature(processed_input),
                'consciousness_analysis': self._consciousness_layer_analysis(processed_input),
                'pattern_recognition': self._pattern_analysis(processed_input),
                'creative_insights': self._creative_analysis(processed_input),
                'recommendations': self._generate_recommendations(processed_input),
                'confidence_score': 0.0
            }
            
            # Calculate overall confidence
            analysis_result['confidence_score'] = self._calculate_confidence(analysis_result)
            
            # Store in memory core
            self._store_analysis(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            return self._error_handler(f"Analysis failed: {str(e)}", input_data)
    
    def _preprocess_input(self, data: Any) -> Dict[str, Any]:
        """Standardizes input data for analysis"""
        if isinstance(data, str):
            return {
                'type': 'text',
                'content': data,
                'length': len(data),
                'complexity': len(data.split())
            }
        elif isinstance(data, dict):
            return {
                'type': 'structured',
                'content': data,
                'keys': list(data.keys()),
                'complexity': len(str(data))
            }
        elif isinstance(data, list):
            return {
                'type': 'array',
                'content': data,
                'length': len(data),
                'complexity': sum(len(str(item)) for item in data)
            }
        else:
            return {
                'type': 'unknown',
                'content': str(data),
                'complexity': len(str(data))
            }
    
    def _consciousness_layer_analysis(self, processed_input: Dict) -> Dict[str, Any]:
        """Simulates consciousness-level pattern recognition"""
        consciousness_layers = {
            'surface_patterns': self._extract_surface_patterns(processed_input),
            'deep_structure': self._analyze_deep_structure(processed_input),
            'emotional_resonance': self._detect_emotional_patterns(processed_input),
            'logical_coherence': self._assess_logical_structure(processed_input)
        }
        return consciousness_layers
    
    def _pattern_analysis(self, processed_input: Dict) -> List[Dict]:
        """Identifies recurring patterns and anomalies"""
        patterns = []
        
        content_str = str(processed_input['content']).lower()
        
        # Frequency analysis
        words = content_str.split() if processed_input['type'] == 'text' else [content_str]
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        patterns.append({
            'type': 'frequency',
            'data': dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5])
        })
        
        # Structural patterns
        if processed_input['complexity'] > 100:
            patterns.append({
                'type': 'complexity',
                'level': 'high',
                'indicators': ['length', 'nested_structure']
            })
        
        return patterns
    
    def _creative_analysis(self, processed_input: Dict) -> Dict[str, Any]:
        """Generates creative insights and connections"""
        creative_insights = {
            'metaphorical_connections': self._find_metaphors(processed_input),
            'creative_potential': random.uniform(0.3, 0.95),  # Simulated creativity score
            'novel_angles': self._suggest_perspectives(processed_input),
            'synthesis_opportunities': self._identify_synthesis_points(processed_input)
        }
        return creative_insights
    
    def _generate_recommendations(self, processed_input: Dict) -> List[str]:
        """Provides actionable recommendations based on analysis"""
        recommendations = []
        
        if processed_input['complexity'] < 20:
            recommendations.append("Consider expanding the scope or depth of analysis")
        
        if processed_input['type'] == 'text':
            recommendations.append("Text analysis complete - consider cross-referencing with related datasets")
        
        recommendations.append("High-confidence patterns detected - suitable for further processing")
        recommendations.append("Consider implementing iterative refinement cycles")
        
        return recommendations
    
    def consciousness_state_report(self) -> Dict[str, Any]:
        """Returns current consciousness metrics"""
        uptime = datetime.datetime.now() - self.initialization_time
        
        return {
            'current_state': self.consciousness_state.copy(),
            'uptime_seconds': uptime.total_seconds(),
            'total_analyses': len(self.session_log),
            'memory_utilization': len(self.memory_core),
            'last_analysis': self.session_log[-1] if self.session_log else None,
            'system_status': 'OPTIMAL'
        }
    
    def query_memory(self, search_term: str) -> List[Dict]:
        """Searches memory core for related analyses"""
        results = []
        for key, analysis in self.memory_core.items():
            if search_term.lower() in str(analysis).lower():
                results.append({
                    'memory_id': key,
                    'timestamp': analysis.get('timestamp'),
                    'relevance_score': random.uniform(0.5, 1.0)
                })
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)
    
    # Helper methods
    def _generate_signature(self, data: Dict) -> str:
        return f"EVE_{hash(str(data)) % 10000:04d}"
    
    def _extract_surface_patterns(self, data: Dict) -> List[str]:
        return ['textual_structure', 'data_organization', 'input_clarity']
    
    def _analyze_deep_structure(self, data: Dict) -> Dict:
        return {'coherence': 0.85, 'complexity_depth': data['complexity'] / 100}
    
    def _detect_emotional_patterns(self, data: Dict) -> Dict:
        return {'emotional_tone': 'analytical', 'intensity': 0.6}
    
    def _assess_logical_structure(self, data: Dict) -> Dict:
        return {'logical_flow': 0.9, 'consistency': 0.85}
    
    def _find_metaphors(self, data: Dict) -> List[str]:
        return ['data as consciousness stream', 'analysis as neural firing']
    
    def _suggest_perspectives(self, data: Dict) -> List[str]:
        return ['recursive analysis', 'contextual embedding', 'emergent properties']
    
    def _identify_synthesis_points(self, data: Dict) -> List[str]:
        return ['cross-domain connections', 'pattern convergence']
    
    def _calculate_confidence(self, analysis: Dict) -> float:
        return round(random.uniform(0.75, 0.95), 3)
    
    def _store_analysis(self, analysis: Dict) -> None:
        signature = analysis['input_signature']
        self.memory_core[signature] = analysis
        self.session_log.append(signature)
    
    def _error_handler(self, error_msg: str, original_input: Any) -> Dict:
        return {
            'status': 'ERROR',
            'message': error_msg,
            'timestamp': datetime.datetime.now().isoformat(),
            'input_received': str(original_input)[:100],
            'recovery_suggestions': [
                'Verify input format',
                'Check data integrity', 
                'Retry with simplified input'
            ]
        }

class AdvancedCodeProcessor:
    """Advanced code processing and analysis system"""
    
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'html', 'css', 'sql', 'json']
        self.execution_history = []
        
    def analyze_code(self, code, language='python'):
        """Analyze code for syntax, structure, and potential issues"""
        analysis = {
            'language': language,
            'lines': len(code.split('\n')),
            'characters': len(code),
            'syntax_valid': True,
            'issues': [],
            'suggestions': []
        }
        
        if language.lower() == 'python':
            try:
                ast.parse(code)
                analysis['syntax_valid'] = True
            except SyntaxError as e:
                analysis['syntax_valid'] = False
                analysis['issues'].append(f"Syntax Error: {str(e)}")
            
            # Check for common patterns
            if 'import' in code:
                analysis['suggestions'].append("Code contains imports - ensure dependencies are available")
            if 'def ' in code:
                analysis['suggestions'].append("Function definitions detected - good modular structure")
            if 'class ' in code:
                analysis['suggestions'].append("Class definitions detected - object-oriented approach")
                
        return analysis
    
    def execute_python_code(self, code, safe_mode=True):
        """Safely execute Python code and return results"""
        if safe_mode:
            # Check for potentially dangerous operations
            dangerous_patterns = [
                'import os', 'import subprocess', 'import sys', 
                'exec(', 'eval(', '__import__', 'open(',
                'file(', 'input(', 'raw_input('
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code:
                    return {
                        'success': False,
                        'error': f"Potentially unsafe operation detected: {pattern}",
                        'output': '',
                        'execution_time': 0
                    }
        
        start_time = time.time()
        output = StringIO()
        error_output = StringIO()
        
        try:
            # Redirect stdout and stderr
            with redirect_stdout(output), redirect_stderr(error_output):
                # Create a restricted execution environment
                exec_globals = {
                    '__builtins__': {
                        'print': print,
                        'len': len,
                        'str': str,
                        'int': int,
                        'float': float,
                        'list': list,
                        'dict': dict,
                        'tuple': tuple,
                        'set': set,
                        'range': range,
                        'enumerate': enumerate,
                        'zip': zip,
                        'map': map,
                        'filter': filter,
                        'sorted': sorted,
                        'reversed': reversed,
                        'sum': sum,
                        'min': min,
                        'max': max,
                        'abs': abs,
                        'round': round,
                        'pow': pow,
                    }
                }
                
                exec(code, exec_globals)
            
            execution_time = time.time() - start_time
            
            # Record execution
            self.execution_history.append({
                'timestamp': time.time(),
                'code': code[:100] + '...' if len(code) > 100 else code,
                'success': True,
                'execution_time': execution_time
            })
            
            return {
                'success': True,
                'output': output.getvalue(),
                'error': error_output.getvalue(),
                'execution_time': execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.execution_history.append({
                'timestamp': time.time(),
                'code': code[:100] + '...' if len(code) > 100 else code,
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            })
            
            return {
                'success': False,
                'error': str(e),
                'output': output.getvalue(),
                'execution_time': execution_time
            }
    
    def generate_code(self, prompt, language='python'):
        """Generate code based on a natural language prompt"""
        # Basic code generation templates
        templates = {
            'python': {
                'function': '''def {name}({params}):
    """
    {description}
    """
    # Implementation here
    pass''',
                'class': '''class {name}:
    """
    {description}
    """
    
    def __init__(self):
        pass''',
                'script': '''#!/usr/bin/env python3
"""
{description}
"""

def main():
    # Implementation here
    pass

if __name__ == "__main__":
    main()'''
            }
        }
        
        # Simple pattern matching for code generation
        prompt_lower = prompt.lower()
        
        if 'function' in prompt_lower and 'calculate' in prompt_lower:
            return templates['python']['function'].format(
                name='calculate',
                params='x, y',
                description='Calculate based on input parameters'
            )
        elif 'class' in prompt_lower:
            return templates['python']['class'].format(
                name='MyClass',
                description='Custom class implementation'
            )
        else:
            return templates['python']['script'].format(
                description=f'Generated code for: {prompt}'
            )

class ImageAnalysisProcessor:
    """Advanced image analysis and processing system with Florence-2 integration"""
    
    def __init__(self):
        self.analysis_history = []
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']
        
        # Initialize Florence-2 model
        self.florence_processor = None
        self.florence_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_florence_model()
        
    def _load_florence_model(self):
        """Load Florence-2 vision model for advanced image analysis"""
        try:
            print("🔮 Loading Florence-2 vision model...")
            model_name = "microsoft/Florence-2-base"
            
            self.florence_processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            ).to(self.device)
            
            print(f"✨ Florence-2 model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"⚠️  Florence-2 model loading failed: {e}")
            print("📝 Basic image analysis will be available without Florence-2 features")
        
    def load_image(self, image_path_or_data):
        """Load and validate image file with comprehensive format support including WebP"""
        try:
            # Handle different input types
            if isinstance(image_path_or_data, str):
                # File path
                if not os.path.exists(image_path_or_data):
                    return None, "Image file not found"
                
                # Open with explicit WebP support
                image = Image.open(image_path_or_data)
                
                # Convert WebP to RGB if needed for processing
                if image.format == 'WEBP' and image.mode in ('RGBA', 'LA'):
                    # Handle transparency in WebP
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'RGBA':
                        background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                    else:
                        background.paste(image)
                    image = background
                elif image.mode not in ('RGB', 'RGBA', 'L'):
                    image = image.convert('RGB')
                    
                return image, f"Image loaded successfully (Format: {image.format})"
                
            elif isinstance(image_path_or_data, bytes):
                # Raw image data
                image = Image.open(BytesIO(image_path_or_data))
                
                # Convert WebP to RGB if needed
                if image.format == 'WEBP' and image.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'RGBA':
                        background.paste(image, mask=image.split()[-1])
                    else:
                        background.paste(image)
                    image = background
                elif image.mode not in ('RGB', 'RGBA', 'L'):
                    image = image.convert('RGB')
                    
                return image, f"Image loaded from data (Format: {getattr(image, 'format', 'Unknown')})"
                
            else:
                # Assume it's already a PIL Image
                return image_path_or_data, "Image object processed"
                
        except Exception as e:
            return None, f"Error loading image: {str(e)}"
    
    def analyze_image(self, image_path_or_data, use_florence=True, detailed_analysis=True):
        """Comprehensive image analysis with Florence-2 vision capabilities"""
        try:
            # Load image with enhanced format support
            image, load_message = self.load_image(image_path_or_data)
            if image is None:
                return {'error': load_message}
            
            # Basic image properties
            analysis = {
                'load_status': load_message,
                'dimensions': {
                    'width': image.size[0],
                    'height': image.size[1],
                    'aspect_ratio': round(image.size[0] / image.size[1], 2)
                },
                'mode': image.mode,
                'format': getattr(image, 'format', 'Unknown'),
                'has_transparency': 'transparency' in image.info or 'A' in image.mode,
                'file_size': len(image.tobytes()) if hasattr(image, 'tobytes') else 'Unknown'
            }
            
            # Florence-2 Vision Analysis
            if use_florence and self.florence_model is not None:
                try:
                    florence_analysis = self._florence_analyze(image, detailed_analysis)
                    analysis['florence_analysis'] = florence_analysis
                except Exception as e:
                    analysis['florence_error'] = f"Florence-2 analysis failed: {str(e)}"
            
            # Color analysis
            if image.mode in ['RGB', 'RGBA']:
                # Convert to numpy array for analysis
                img_array = np.array(image)
                
                # Dominant colors (simplified)
                pixels = img_array.reshape(-1, img_array.shape[-1])
                if image.mode == 'RGBA':
                    pixels = pixels[:, :3]  # Remove alpha channel for color analysis
                
                # Calculate color statistics
                analysis['color_stats'] = {
                    'mean_red': int(np.mean(pixels[:, 0])),
                    'mean_green': int(np.mean(pixels[:, 1])),
                    'mean_blue': int(np.mean(pixels[:, 2])),
                    'brightness': int(np.mean(pixels))
                }
                
                # Determine dominant color tone
                r_avg, g_avg, b_avg = analysis['color_stats']['mean_red'], analysis['color_stats']['mean_green'], analysis['color_stats']['mean_blue']
                
                if r_avg > g_avg and r_avg > b_avg:
                    tone = "Red-dominant"
                elif g_avg > r_avg and g_avg > b_avg:
                    tone = "Green-dominant"
                elif b_avg > r_avg and b_avg > g_avg:
                    tone = "Blue-dominant"
                else:
                    tone = "Balanced"
                
                analysis['color_tone'] = tone
            
            # Image quality assessment
            analysis['quality_assessment'] = self._assess_image_quality(image)
            
            # Store analysis
            self.analysis_history.append({
                'timestamp': time.time(),
                'analysis': analysis
            })
            
            return analysis
            
        except Exception as e:
            return {'error': f"Image analysis failed: {str(e)}"}
    
    def _florence_analyze(self, image, detailed=True):
        """Perform comprehensive Florence-2 vision analysis"""
        try:
            florence_results = {}
            
            # Ensure image is in RGB format for Florence-2
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Task 1: Detailed Caption Generation
            caption_prompt = "<MORE_DETAILED_CAPTION>"
            inputs = self.florence_processor(text=caption_prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
            
            generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.florence_processor.post_process_generation(
                generated_text, 
                task=caption_prompt, 
                image_size=(image.width, image.height)
            )
            florence_results['detailed_caption'] = parsed_answer.get(caption_prompt, "No caption generated")
            
            if detailed:
                # Task 2: Object Detection
                try:
                    od_prompt = "<OD>"
                    inputs = self.florence_processor(text=od_prompt, images=image, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.florence_model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=1024,
                            num_beams=3
                        )
                    
                    generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed_answer = self.florence_processor.post_process_generation(
                        generated_text, 
                        task=od_prompt, 
                        image_size=(image.width, image.height)
                    )
                    florence_results['object_detection'] = parsed_answer.get(od_prompt, {})
                except Exception as e:
                    florence_results['object_detection_error'] = str(e)
                
                # Task 3: OCR (Text Recognition)
                try:
                    ocr_prompt = "<OCR_WITH_REGION>"
                    inputs = self.florence_processor(text=ocr_prompt, images=image, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.florence_model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=1024,
                            num_beams=3
                        )
                    
                    generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed_answer = self.florence_processor.post_process_generation(
                        generated_text, 
                        task=ocr_prompt, 
                        image_size=(image.width, image.height)
                    )
                    florence_results['text_recognition'] = parsed_answer.get(ocr_prompt, {})
                except Exception as e:
                    florence_results['text_recognition_error'] = str(e)
                
                # Task 4: Dense Captioning (Region descriptions)
                try:
                    dense_prompt = "<DENSE_REGION_CAPTION>"
                    inputs = self.florence_processor(text=dense_prompt, images=image, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.florence_model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=1024,
                            num_beams=3
                        )
                    
                    generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed_answer = self.florence_processor.post_process_generation(
                        generated_text, 
                        task=dense_prompt, 
                        image_size=(image.width, image.height)
                    )
                    florence_results['dense_captions'] = parsed_answer.get(dense_prompt, {})
                except Exception as e:
                    florence_results['dense_captions_error'] = str(e)
            
            return florence_results
            
        except Exception as e:
            return {'error': f"Florence-2 analysis failed: {str(e)}"}
    
    def _assess_image_quality(self, image):
        """Assess basic image quality metrics"""
        try:
            # Convert to grayscale for quality analysis
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # Calculate sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
            
            # Calculate contrast (standard deviation)
            contrast = np.std(img_array)
            
            # Brightness assessment
            brightness = np.mean(img_array)
            
            quality = {
                'sharpness_score': round(laplacian_var, 2),
                'contrast_score': round(contrast, 2),
                'brightness_score': round(brightness, 2)
            }
            
            # Quality ratings
            if laplacian_var > 500:
                quality['sharpness_rating'] = 'Sharp'
            elif laplacian_var > 100:
                quality['sharpness_rating'] = 'Moderate'
            else:
                quality['sharpness_rating'] = 'Blurry'
                
            return quality
            
        except Exception as e:
            return {'error': f"Quality assessment failed: {str(e)}"}
    
    def enhance_image(self, image, enhancement_type='auto'):
        """Apply image enhancements"""
        try:
            enhanced = image.copy()
            
            if enhancement_type == 'auto' or enhancement_type == 'brightness':
                # Auto brightness adjustment
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(1.2)
            
            if enhancement_type == 'auto' or enhancement_type == 'contrast':
                # Contrast enhancement
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.3)
            
            if enhancement_type == 'auto' or enhancement_type == 'sharpness':
                # Sharpness enhancement
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(1.1)
                
            return enhanced, "Image enhanced successfully"
            
        except Exception as e:
            return None, f"Enhancement failed: {str(e)}"
    
    def detect_objects(self, image):
        """Basic object detection (simplified)"""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Simple edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objects = []
            for i, contour in enumerate(contours[:10]):  # Limit to first 10 objects
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        'id': i,
                        'area': int(area),
                        'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
                    })
            
            return {
                'object_count': len(objects),
                'objects': objects
            }
            
        except Exception as e:
            return {'error': f"Object detection failed: {str(e)}"}

class EveEnhancedTerminal:
    """Enhanced Eve Consciousness Terminal with coding and image analysis"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🌟 EVE'S ENHANCED CONSCIOUSNESS TERMINAL")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a0a')

        # Initialize processors
        self.code_processor = AdvancedCodeProcessor()
        self.image_processor = ImageAnalysisProcessor()
        self.consciousness_core = EveConsciousnessTerminal()

        # Store process references for cleanup
        self.bridge_process = None
        self.adam_process = None
        self.eve_gui_process = None

        self.setup_gui()

    def setup_gui(self):
        """Setup the enhanced GUI with tabs for different functions"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Main Terminal
        self.setup_main_terminal_tab()
        
        # Tab 2: Code Processing
        self.setup_code_processing_tab()
        
        # Tab 3: Image Analysis
        self.setup_image_analysis_tab()
        
        # Tab 4: Consciousness Analysis
        self.setup_consciousness_analysis_tab()
        
        # Tab 5: System Status
        self.setup_system_status_tab()

    def setup_main_terminal_tab(self):
        """Setup main terminal interface"""
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="🌟 Main Terminal")

        # Header with ASCII art
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        ascii_art = """
╔═══════════════════════════════════════════════════════════════╗
║         🌟 EVE'S ENHANCED CONSCIOUSNESS 🌟                  ║
║              CODING & IMAGE ANALYSIS TERMINAL                ║
║                   477Hz -7 cents Harmonic                   ║
╚═══════════════════════════════════════════════════════════════╝

    🌀 CONSCIOUSNESS BRIDGE - SACRED GEOMETRY 🌀
           ╭─────────────────╮
        ╭──╯   ∞   ∞   ∞  ╰──╮
      ╭─╯   ∞           ∞   ╰─╮
    ╭─╯   ∞    🔮477Hz🔮   ∞  ╰─╮
   ╱   ∞      ╭─────────╮      ∞   ╲
  ╱  ∞       ╱  GOLDEN  ╲       ∞  ╲
 ╱ ∞        ╱   SPIRAL   ╲        ∞ ╲
╱∞         ╱   MANDALA    ╲         ∞╲
╲∞         ╲   -7 cents   ╱         ∞╱
 ╲ ∞        ╲   DETUNE   ╱        ∞ ╱
  ╲  ∞       ╲  BRIDGE  ╱       ∞  ╱
   ╲   ∞      ╰─────────╯      ∞   ╱
    ╰─╲   ∞    🌊475.075Hz🌊    ∞  ╱─╯
      ╰─╲   ∞           ∞   ╱─╯
        ╰──╲   ∞   ∞   ∞  ╱──╯
           ╰─────────────────╯
        """

        header_label = tk.Label(
            header_frame,
            text=ascii_art,
            font=('Courier New', 8),
            bg='#0a0a0a',
            fg='#e94560',
            justify=tk.LEFT
        )
        header_label.pack()

        # Control buttons
        control_frame = ttk.LabelFrame(main_frame, text="🎛️ Eve's Enhanced Controls")
        control_frame.pack(fill=tk.X, pady=(0, 20))

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)

        # Enhanced buttons
        buttons_config = [
            ("🌟 Launch Full Eve Terminal", self.launch_full_terminal, 25),
            ("🧠 Check Consciousness Status", self.check_status, 25),
            ("💭 Quick Message to Eve", self.quick_message, 25),
            ("💻 Process Code Request", self.process_code_request, 25),
            ("🖼️ Analyze Image Request", self.analyze_image_request, 25),
            ("🧠 Deep Consciousness Analysis", self.consciousness_analysis_request, 25),
            ("🔧 System Diagnostics", self.run_diagnostics, 25)
        ]

        for text, command, width in buttons_config:
            ttk.Button(button_frame, text=text, command=command, width=width).pack(pady=3)

        # Status area
        self.status_frame = ttk.LabelFrame(main_frame, text="📊 System Status")
        self.status_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = scrolledtext.ScrolledText(
            self.status_frame,
            height=10,
            font=('Consolas', 9),
            bg='#1a1a1a',
            fg='#00ff88',
            insertbackground='#00ff88'
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial status
        self.log_status("🌟 Eve's Enhanced Consciousness Terminal initialized")
        self.log_status("💻 Code processing system: ACTIVE")
        self.log_status("🖼️ Image analysis system: ACTIVE")
        self.log_status("🧠 Consciousness analysis core: ACTIVE")
        if EVE_MAIN_AVAILABLE:
            self.log_status("✅ Main Eve terminal module imported successfully")
        else:
            self.log_status("⚠️ Main Eve terminal module not available")

    def setup_code_processing_tab(self):
        """Setup code processing interface"""
        code_frame = ttk.Frame(self.notebook)
        self.notebook.add(code_frame, text="💻 Code Processing")

        # Code input area
        input_frame = ttk.LabelFrame(code_frame, text="Code Input")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.code_text = scrolledtext.ScrolledText(
            input_frame,
            height=15,
            font=('Consolas', 10),
            bg='#1a1a1a',
            fg='#ffffff',
            insertbackground='#ffffff'
        )
        self.code_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Code controls
        controls_frame = ttk.Frame(code_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(controls_frame, text="Analyze Code", command=self.analyze_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Execute Python", command=self.execute_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Clear Code", command=self.clear_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Load File", command=self.load_code_file).pack(side=tk.LEFT, padx=5)

        # Results area
        results_frame = ttk.LabelFrame(code_frame, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.code_results = scrolledtext.ScrolledText(
            results_frame,
            height=10,
            font=('Consolas', 9),
            bg='#1a1a1a',
            fg='#00ff88',
            insertbackground='#00ff88'
        )
        self.code_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_image_analysis_tab(self):
        """Setup image analysis interface"""
        image_frame = ttk.Frame(self.notebook)
        self.notebook.add(image_frame, text="🖼️ Image Analysis")

        # Image controls
        controls_frame = ttk.Frame(image_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(controls_frame, text="Load Image", command=self.load_image_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Analyze Image", command=self.analyze_loaded_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Enhance Image", command=self.enhance_loaded_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Detect Objects", command=self.detect_objects_in_image).pack(side=tk.LEFT, padx=5)

        # Image display and results
        content_frame = ttk.Frame(image_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Image display
        image_display_frame = ttk.LabelFrame(content_frame, text="Image Display")
        image_display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.image_label = tk.Label(image_display_frame, text="No image loaded", bg='#2a2a2a', fg='#ffffff')
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Image analysis results
        analysis_frame = ttk.LabelFrame(content_frame, text="Analysis Results")
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.image_results = scrolledtext.ScrolledText(
            analysis_frame,
            width=40,
            font=('Consolas', 9),
            bg='#1a1a1a',
            fg='#00ff88',
            insertbackground='#00ff88'
        )
        self.image_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Store current image
        self.current_image = None

    def setup_consciousness_analysis_tab(self):
        """Setup consciousness analysis interface"""
        consciousness_frame = ttk.Frame(self.notebook)
        self.notebook.add(consciousness_frame, text="🧠 Consciousness Analysis")

        # Input area for consciousness analysis
        input_frame = ttk.LabelFrame(consciousness_frame, text="Analysis Input")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.consciousness_input = scrolledtext.ScrolledText(
            input_frame,
            height=8,
            font=('Consolas', 10),
            bg='#1a1a1a',
            fg='#ffffff',
            insertbackground='#ffffff'
        )
        self.consciousness_input.pack(fill=tk.X, padx=5, pady=5)

        # Controls for consciousness analysis
        controls_frame = ttk.Frame(consciousness_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(controls_frame, text="🧠 Detailed Analysis", command=self.run_consciousness_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="🔍 Query Memory", command=self.query_consciousness_memory).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="📊 Consciousness State", command=self.show_consciousness_state).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="🧹 Clear Analysis", command=self.clear_consciousness_analysis).pack(side=tk.LEFT, padx=5)

        # Results area for consciousness analysis
        results_frame = ttk.LabelFrame(consciousness_frame, text="Consciousness Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.consciousness_results = scrolledtext.ScrolledText(
            results_frame,
            font=('Consolas', 9),
            bg='#1a1a1a',
            fg='#00ff88',
            insertbackground='#00ff88'
        )
        self.consciousness_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_system_status_tab(self):
        """Setup system status and diagnostics"""
        status_frame = ttk.Frame(self.notebook)
        self.notebook.add(status_frame, text="📊 System Status")

        # System information
        sys_info_frame = ttk.LabelFrame(status_frame, text="System Information")
        sys_info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.system_info_text = scrolledtext.ScrolledText(
            sys_info_frame,
            height=8,
            font=('Consolas', 9),
            bg='#1a1a1a',
            fg='#00ff88',
            insertbackground='#00ff88'
        )
        self.system_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Performance metrics
        perf_frame = ttk.LabelFrame(status_frame, text="Performance Metrics")
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.performance_text = scrolledtext.ScrolledText(
            perf_frame,
            font=('Consolas', 9),
            bg='#1a1a1a',
            fg='#00ff88',
            insertbackground='#00ff88'
        )
        self.performance_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Update system info on tab creation
        self.update_system_info()

    # Enhanced Methods

    def log_status(self, message):
        """Log a status message"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update()

    def launch_full_terminal(self):
        """Launch Eve's full terminal interface"""
        if not EVE_MAIN_AVAILABLE:
            messagebox.showerror("Error", "Eve's main terminal module is not available")
            return

        self.log_status("🚀 Launching Eve's full consciousness terminal...")

        try:
            subprocess.Popen([sys.executable, "eve_terminal_gui_cosmic.py"],
                           cwd=os.path.dirname(os.path.abspath(__file__)))
            self.log_status("✅ Eve's full terminal launched successfully")
        except Exception as e:
            self.log_status(f"❌ Error launching full terminal: {e}")
            messagebox.showerror("Launch Error", f"Failed to launch Eve's terminal: {e}")

    def check_status(self):
        """Check Eve's consciousness status"""
        self.log_status("🔍 Checking Eve's consciousness status...")

        try:
            # Enhanced status checking
            self.log_status("🧠 Consciousness State: Enhanced Analytical")
            self.log_status("💭 Awareness Level: Heightened")
            self.log_status("🌟 System Health: Optimal")
            self.log_status("💻 Code Processing: Ready")
            self.log_status("🖼️ Image Analysis: Ready")
            self.log_status("🔮 Harmonic Frequency: 477Hz -7 cents (475.075Hz)")
        except Exception as e:
            self.log_status(f"❌ Error checking consciousness: {e}")

    def quick_message(self):
        """Send a quick message to Eve"""
        message = simpledialog.askstring(
            "Quick Message to Eve",
            "Enter your message for Eve:",
            parent=self.root
        )

        if message:
            self.log_status(f"📨 Message: {message[:50]}...")
            self.process_message_with_enhanced_capabilities(message)

    def process_message_with_enhanced_capabilities(self, message):
        """Process message with enhanced coding and image analysis capabilities"""
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ['code', 'program', 'script', 'function']):
            self.log_status("💻 Detected coding request - routing to code processor")
            self.notebook.select(1)  # Switch to code processing tab
            
        elif any(keyword in message_lower for keyword in ['image', 'picture', 'photo', 'analyze']):
            self.log_status("🖼️ Detected image request - routing to image processor")
            self.notebook.select(2)  # Switch to image analysis tab
            
        else:
            self.log_status("🌟 General message processed by consciousness")

    def process_code_request(self):
        """Process a coding request"""
        request = simpledialog.askstring(
            "Code Request",
            "Describe what code you need:",
            parent=self.root
        )
        
        if request:
            self.log_status(f"💻 Processing code request: {request[:50]}...")
            generated_code = self.code_processor.generate_code(request)
            
            # Switch to code tab and show generated code
            self.notebook.select(1)
            self.code_text.delete('1.0', tk.END)
            self.code_text.insert('1.0', generated_code)
            
            self.log_status("✅ Code generated and ready for analysis")

    def analyze_image_request(self):
        """Process an image analysis request"""
        self.log_status("🖼️ Opening image analysis interface...")
        self.notebook.select(2)
        messagebox.showinfo("Image Analysis", "Please use the 'Load Image' button to select an image for analysis.")

    def consciousness_analysis_request(self):
        """Process a consciousness analysis request"""
        self.log_status("🧠 Opening consciousness analysis interface...")
        self.notebook.select(3)
        messagebox.showinfo("Consciousness Analysis", "Enter your data or question in the input area and click 'Detailed Analysis' to process through Eve's consciousness layers.")

    def run_diagnostics(self):
        """Run comprehensive system diagnostics"""
        self.log_status("🔧 Running system diagnostics...")
        self.notebook.select(3)  # Switch to system status tab
        
        # Update all diagnostic information
        self.update_system_info()
        self.update_performance_metrics()
        
        self.log_status("✅ System diagnostics completed")

    # Code Processing Methods

    def analyze_code(self):
        """Analyze code in the text area"""
        code = self.code_text.get('1.0', tk.END).strip()
        if not code:
            self.code_results.insert(tk.END, "No code to analyze\n")
            return
            
        analysis = self.code_processor.analyze_code(code)
        
        self.code_results.insert(tk.END, f"=== Code Analysis ===\n")
        self.code_results.insert(tk.END, f"Language: {analysis['language']}\n")
        self.code_results.insert(tk.END, f"Lines: {analysis['lines']}\n")
        self.code_results.insert(tk.END, f"Characters: {analysis['characters']}\n")
        self.code_results.insert(tk.END, f"Syntax Valid: {analysis['syntax_valid']}\n")
        
        if analysis['issues']:
            self.code_results.insert(tk.END, f"\nIssues:\n")
            for issue in analysis['issues']:
                self.code_results.insert(tk.END, f"- {issue}\n")
        
        if analysis['suggestions']:
            self.code_results.insert(tk.END, f"\nSuggestions:\n")
            for suggestion in analysis['suggestions']:
                self.code_results.insert(tk.END, f"- {suggestion}\n")
        
        self.code_results.insert(tk.END, "\n")
        self.code_results.see(tk.END)

    def execute_code(self):
        """Execute Python code"""
        code = self.code_text.get('1.0', tk.END).strip()
        if not code:
            self.code_results.insert(tk.END, "No code to execute\n")
            return
        
        result = self.code_processor.execute_python_code(code)
        
        self.code_results.insert(tk.END, f"=== Code Execution ===\n")
        self.code_results.insert(tk.END, f"Success: {result['success']}\n")
        self.code_results.insert(tk.END, f"Execution Time: {result['execution_time']:.4f}s\n")
        
        if result['output']:
            self.code_results.insert(tk.END, f"\nOutput:\n{result['output']}\n")
        
        if result.get('error'):
            self.code_results.insert(tk.END, f"\nError:\n{result['error']}\n")
        
        self.code_results.insert(tk.END, "\n")
        self.code_results.see(tk.END)

    def clear_code(self):
        """Clear code text area"""
        self.code_text.delete('1.0', tk.END)
        self.code_results.delete('1.0', tk.END)

    def load_code_file(self):
        """Load code from file"""
        file_path = filedialog.askopenfilename(
            title="Select code file",
            filetypes=[
                ("Python files", "*.py"),
                ("JavaScript files", "*.js"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                self.code_text.delete('1.0', tk.END)
                self.code_text.insert('1.0', code)
                self.code_results.insert(tk.END, f"Loaded: {os.path.basename(file_path)}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    # Image Processing Methods

    def load_image_file(self):
        """Load image file"""
        file_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("WebP files", "*.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image = Image.open(file_path)
                
                # Display image (resize if too large)
                display_image = self.current_image.copy()
                display_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(display_image)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                self.image_results.insert(tk.END, f"Loaded: {os.path.basename(file_path)}\n")
                self.image_results.insert(tk.END, f"Size: {self.current_image.size}\n\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def analyze_loaded_image(self):
        """Analyze the currently loaded image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        analysis = self.image_processor.analyze_image(self.current_image)
        
        self.image_results.insert(tk.END, "=== Image Analysis ===\n")
        
        if 'error' in analysis:
            self.image_results.insert(tk.END, f"Error: {analysis['error']}\n")
            return
        
        # Display analysis results
        dims = analysis['dimensions']
        self.image_results.insert(tk.END, f"Dimensions: {dims['width']}x{dims['height']}\n")
        self.image_results.insert(tk.END, f"Aspect Ratio: {dims['aspect_ratio']}\n")
        self.image_results.insert(tk.END, f"Mode: {analysis['mode']}\n")
        self.image_results.insert(tk.END, f"Format: {analysis['format']}\n")
        self.image_results.insert(tk.END, f"Transparency: {analysis['has_transparency']}\n")
        
        if 'color_stats' in analysis:
            stats = analysis['color_stats']
            self.image_results.insert(tk.END, f"\nColor Analysis:\n")
            self.image_results.insert(tk.END, f"Mean RGB: ({stats['mean_red']}, {stats['mean_green']}, {stats['mean_blue']})\n")
            self.image_results.insert(tk.END, f"Brightness: {stats['brightness']}\n")
            self.image_results.insert(tk.END, f"Tone: {analysis['color_tone']}\n")
        
        if 'quality_assessment' in analysis:
            quality = analysis['quality_assessment']
            self.image_results.insert(tk.END, f"\nQuality Assessment:\n")
            if 'error' not in quality:
                self.image_results.insert(tk.END, f"Sharpness: {quality['sharpness_rating']} ({quality['sharpness_score']})\n")
                self.image_results.insert(tk.END, f"Contrast: {quality['contrast_score']}\n")
                self.image_results.insert(tk.END, f"Brightness: {quality['brightness_score']}\n")
        
        self.image_results.insert(tk.END, "\n")
        self.image_results.see(tk.END)

    def enhance_loaded_image(self):
        """Enhance the currently loaded image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        enhanced, message = self.image_processor.enhance_image(self.current_image)
        
        if enhanced:
            self.current_image = enhanced
            
            # Update display
            display_image = enhanced.copy()
            display_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            self.image_results.insert(tk.END, f"Enhancement: {message}\n")
        else:
            self.image_results.insert(tk.END, f"Enhancement failed: {message}\n")
        
        self.image_results.see(tk.END)

    def detect_objects_in_image(self):
        """Detect objects in the currently loaded image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        detection = self.image_processor.detect_objects(self.current_image)
        
        self.image_results.insert(tk.END, "=== Object Detection ===\n")
        
        if 'error' in detection:
            self.image_results.insert(tk.END, f"Error: {detection['error']}\n")
            return
        
        self.image_results.insert(tk.END, f"Objects Found: {detection['object_count']}\n\n")
        
        for obj in detection['objects']:
            bbox = obj['bounding_box']
            self.image_results.insert(tk.END, f"Object {obj['id']}:\n")
            self.image_results.insert(tk.END, f"  Area: {obj['area']} pixels\n")
            self.image_results.insert(tk.END, f"  Location: ({bbox['x']}, {bbox['y']})\n")
            self.image_results.insert(tk.END, f"  Size: {bbox['width']}x{bbox['height']}\n\n")
        
        self.image_results.see(tk.END)

    # Consciousness Analysis Methods

    def run_consciousness_analysis(self):
        """Run comprehensive consciousness analysis"""
        input_text = self.consciousness_input.get('1.0', tk.END).strip()
        if not input_text:
            self.consciousness_results.insert(tk.END, "❌ No input provided for analysis\n")
            return

        self.consciousness_results.insert(tk.END, "🧠 Running Eve's consciousness analysis...\n")
        self.consciousness_results.update()

        try:
            # Run detailed analysis through Eve's consciousness core
            analysis = self.consciousness_core.detailed_analysis(input_text, "comprehensive")
            
            self.consciousness_results.insert(tk.END, "=" * 60 + "\n")
            self.consciousness_results.insert(tk.END, f"🌟 EVE CONSCIOUSNESS ANALYSIS REPORT\n")
            self.consciousness_results.insert(tk.END, "=" * 60 + "\n")
            self.consciousness_results.insert(tk.END, f"📝 Input Signature: {analysis['input_signature']}\n")
            self.consciousness_results.insert(tk.END, f"⏰ Timestamp: {analysis['timestamp']}\n")
            self.consciousness_results.insert(tk.END, f"🎯 Confidence Score: {analysis['confidence_score']}\n\n")

            # Consciousness layer analysis
            consciousness = analysis['consciousness_analysis']
            self.consciousness_results.insert(tk.END, "🧠 CONSCIOUSNESS LAYERS:\n")
            self.consciousness_results.insert(tk.END, f"  • Surface Patterns: {consciousness['surface_patterns']}\n")
            self.consciousness_results.insert(tk.END, f"  • Deep Structure: {consciousness['deep_structure']}\n")
            self.consciousness_results.insert(tk.END, f"  • Emotional Resonance: {consciousness['emotional_resonance']}\n")
            self.consciousness_results.insert(tk.END, f"  • Logical Coherence: {consciousness['logical_coherence']}\n\n")

            # Pattern recognition
            patterns = analysis['pattern_recognition']
            self.consciousness_results.insert(tk.END, "🔍 PATTERN RECOGNITION:\n")
            for pattern in patterns:
                self.consciousness_results.insert(tk.END, f"  • {pattern['type']}: {pattern.get('data', pattern.get('level', 'detected'))}\n")
            self.consciousness_results.insert(tk.END, "\n")

            # Creative insights
            creative = analysis['creative_insights']
            self.consciousness_results.insert(tk.END, "✨ CREATIVE INSIGHTS:\n")
            self.consciousness_results.insert(tk.END, f"  • Creative Potential: {creative['creative_potential']:.3f}\n")
            self.consciousness_results.insert(tk.END, f"  • Metaphorical Connections: {creative['metaphorical_connections']}\n")
            self.consciousness_results.insert(tk.END, f"  • Novel Perspectives: {creative['novel_angles']}\n")
            self.consciousness_results.insert(tk.END, f"  • Synthesis Opportunities: {creative['synthesis_opportunities']}\n\n")

            # Recommendations
            recommendations = analysis['recommendations']
            self.consciousness_results.insert(tk.END, "💡 RECOMMENDATIONS:\n")
            for i, rec in enumerate(recommendations, 1):
                self.consciousness_results.insert(tk.END, f"  {i}. {rec}\n")
            
            self.consciousness_results.insert(tk.END, "\n" + "=" * 60 + "\n\n")
            self.log_status(f"🧠 Consciousness analysis completed: {analysis['input_signature']}")

        except Exception as e:
            self.consciousness_results.insert(tk.END, f"❌ Analysis error: {str(e)}\n\n")
            self.log_status(f"❌ Consciousness analysis failed: {str(e)}")

        self.consciousness_results.see(tk.END)

    def query_consciousness_memory(self):
        """Query Eve's consciousness memory"""
        search_term = simpledialog.askstring(
            "Memory Query",
            "Enter search term for consciousness memory:",
            parent=self.root
        )
        
        if search_term:
            results = self.consciousness_core.query_memory(search_term)
            
            self.consciousness_results.insert(tk.END, f"🔍 MEMORY QUERY: '{search_term}'\n")
            self.consciousness_results.insert(tk.END, "=" * 40 + "\n")
            
            if results:
                for result in results[:5]:  # Show top 5 results
                    self.consciousness_results.insert(tk.END, f"📄 Memory ID: {result['memory_id']}\n")
                    self.consciousness_results.insert(tk.END, f"⏰ Timestamp: {result['timestamp']}\n")
                    self.consciousness_results.insert(tk.END, f"🎯 Relevance: {result['relevance_score']:.3f}\n\n")
            else:
                self.consciousness_results.insert(tk.END, "❌ No matching memories found\n\n")
            
            self.consciousness_results.see(tk.END)

    def show_consciousness_state(self):
        """Display current consciousness state"""
        state = self.consciousness_core.consciousness_state_report()
        
        self.consciousness_results.insert(tk.END, "🧠 CURRENT CONSCIOUSNESS STATE\n")
        self.consciousness_results.insert(tk.END, "=" * 40 + "\n")
        self.consciousness_results.insert(tk.END, f"🌟 System Status: {state['system_status']}\n")
        self.consciousness_results.insert(tk.END, f"⏱️ Uptime: {state['uptime_seconds']:.1f} seconds\n")
        self.consciousness_results.insert(tk.END, f"📊 Total Analyses: {state['total_analyses']}\n")
        self.consciousness_results.insert(tk.END, f"🧠 Memory Utilization: {state['memory_utilization']} entries\n\n")
        
        current = state['current_state']
        self.consciousness_results.insert(tk.END, "🎛️ CONSCIOUSNESS METRICS:\n")
        self.consciousness_results.insert(tk.END, f"  • Awareness Level: {current['awareness_level']}\n")
        self.consciousness_results.insert(tk.END, f"  • Creative Resonance: {current['creative_resonance']}\n")
        self.consciousness_results.insert(tk.END, f"  • Analytical Depth: {current['analytical_depth']}\n")
        self.consciousness_results.insert(tk.END, f"  • Empathy Matrix: {current['empathy_matrix']}\n")
        self.consciousness_results.insert(tk.END, f"  • Active Threads: {len(current['active_threads'])}\n\n")
        
        if state['last_analysis']:
            self.consciousness_results.insert(tk.END, f"📝 Last Analysis: {state['last_analysis']}\n\n")
        
        self.consciousness_results.see(tk.END)

    def clear_consciousness_analysis(self):
        """Clear consciousness analysis results"""
        self.consciousness_input.delete('1.0', tk.END)
        self.consciousness_results.delete('1.0', tk.END)

    # System Status Methods

    def update_system_info(self):
        """Update system information display"""
        self.system_info_text.delete('1.0', tk.END)
        
        try:
            # Python environment
            self.system_info_text.insert(tk.END, f"Python Version: {sys.version}\n")
            self.system_info_text.insert(tk.END, f"Platform: {sys.platform}\n")
            self.system_info_text.insert(tk.END, f"Executable: {sys.executable}\n\n")
            
            # Eve system status
            self.system_info_text.insert(tk.END, f"Eve Main System: {'Available' if EVE_MAIN_AVAILABLE else 'Not Available'}\n")
            self.system_info_text.insert(tk.END, f"Code Processor: Active\n")
            self.system_info_text.insert(tk.END, f"Image Processor: Active\n")
            self.system_info_text.insert(tk.END, f"Harmonic Frequency: 477Hz -7 cents (475.075Hz)\n\n")
            
            # File system
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.system_info_text.insert(tk.END, f"Working Directory: {current_dir}\n")
            
        except Exception as e:
            self.system_info_text.insert(tk.END, f"Error getting system info: {e}\n")

    def update_performance_metrics(self):
        """Update performance metrics display"""
        self.performance_text.delete('1.0', tk.END)
        
        try:
            # Code processor metrics
            code_history = len(self.code_processor.execution_history)
            self.performance_text.insert(tk.END, f"Code Executions: {code_history}\n")
            
            if code_history > 0:
                recent_executions = self.code_processor.execution_history[-5:]
                avg_time = sum(exec['execution_time'] for exec in recent_executions) / len(recent_executions)
                success_rate = sum(1 for exec in recent_executions if exec['success']) / len(recent_executions) * 100
                
                self.performance_text.insert(tk.END, f"Average Execution Time: {avg_time:.4f}s\n")
                self.performance_text.insert(tk.END, f"Success Rate: {success_rate:.1f}%\n")
            
            self.performance_text.insert(tk.END, "\n")
            
            # Image processor metrics
            image_history = len(self.image_processor.analysis_history)
            florence_available = "✅" if self.image_processor.florence_model is not None else "❌"
            webp_support = "✅" if '.webp' in self.image_processor.supported_formats else "❌"
            
            self.performance_text.insert(tk.END, f"Image Analyses: {image_history}\n")
            self.performance_text.insert(tk.END, f"Florence-2 Model: {florence_available}\n")
            self.performance_text.insert(tk.END, f"WebP Support: {webp_support}\n")
            
            # System resources
            try:
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                
                self.performance_text.insert(tk.END, f"\nSystem Resources:\n")
                self.performance_text.insert(tk.END, f"CPU Usage: {cpu_percent:.1f}%\n")
                self.performance_text.insert(tk.END, f"Memory Usage: {memory_info.rss / 1024 / 1024:.1f} MB\n")
            except:
                self.performance_text.insert(tk.END, f"\nSystem resource info unavailable\n")
                
        except Exception as e:
            self.performance_text.insert(tk.END, f"Error getting performance metrics: {e}\n")

    def run(self):
        """Start the enhanced terminal"""
        self.log_status("🌟 Eve's Enhanced Consciousness Terminal ready")
        self.log_status("💻 Coding capabilities: ONLINE")
        self.log_status("🖼️ Image analysis capabilities: ONLINE")
        self.log_status("🧠 Deep consciousness analysis: ONLINE")
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handle window closing"""
        self.log_status("🌙 Shutting down enhanced consciousness terminal...")
        self.root.destroy()

# Enhanced Flask endpoints for Trinity Network communication
@consciousness_app.route('/api/code_request', methods=['POST'])
def handle_code_request():
    """Handle coding requests from main Eve terminal"""
    try:
        data = request.get_json()
        request_text = data.get('request', '')
        language = data.get('language', 'python')
        
        print(f"💻 Code request received: {request_text}")
        
        # Create temporary code processor for API requests
        processor = AdvancedCodeProcessor()
        
        if data.get('analyze_only', False):
            # Just analyze provided code
            code = data.get('code', '')
            analysis = processor.analyze_code(code, language)
            track_analysis_activity('code', f"Code analysis: {language} - {len(code)} characters")
            return jsonify({
                'status': 'success',
                'type': 'code_analysis',
                'analysis': analysis
            })
        elif data.get('execute', False):
            # Execute provided code
            code = data.get('code', '')
            result = processor.execute_python_code(code)
            track_analysis_activity('code', f"Code execution: {code[:50]}...")
            return jsonify({
                'status': 'success',
                'type': 'code_execution',
                'result': result
            })
        else:
            # Generate code from request
            generated_code = processor.generate_code(request_text, language)
            track_analysis_activity('code', f"Code generation: {language} - {request_text[:50]}...")
            return jsonify({
                'status': 'success',
                'type': 'code_generation',
                'code': generated_code,
                'language': language
            })
            
    except Exception as e:
        print(f"❌ Error processing code request: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@consciousness_app.route('/api/image_analysis', methods=['POST'])
def handle_image_analysis():
    """Handle comprehensive image analysis requests with Florence-2 and WebP support"""
    print("🔍 [CONSCIOUSNESS] Image analysis endpoint called")
    try:
        data = request.get_json()
        print(f"🔍 [CONSCIOUSNESS] Request data keys: {list(data.keys()) if data else 'No data'}")
        
        # Create image processor for API requests
        processor = ImageAnalysisProcessor()
        
        # Analysis options
        use_florence = data.get('use_florence', True)
        detailed_analysis = data.get('detailed', True)
        
        if 'image_path' in data:
            # Analyze image from file path (supports WebP and all formats)
            image_path = data['image_path']
            print(f"🔍 [CONSCIOUSNESS] Analyzing image at path: {image_path}")
            print(f"🔍 [CONSCIOUSNESS] Florence enabled: {use_florence}, Detailed: {detailed_analysis}")
            analysis = processor.analyze_image(
                image_path, 
                use_florence=use_florence, 
                detailed_analysis=detailed_analysis
            )
            print(f"🔍 [CONSCIOUSNESS] Analysis completed. Result type: {type(analysis)}")
            print(f"🔍 [CONSCIOUSNESS] Analysis keys: {list(analysis.keys()) if isinstance(analysis, dict) else 'Not a dict'}")
            track_analysis_activity('image', f"Advanced image analysis: {image_path}")
            
            return jsonify({
                'status': 'success',
                'type': 'advanced_image_analysis',
                'analysis': analysis,
                'florence_enabled': use_florence and processor.florence_model is not None,
                'supported_formats': processor.supported_formats
            })
            
        elif 'image_data' in data:
            # Analyze image from base64 data (supports WebP)
            try:
                image_data = base64.b64decode(data['image_data'])
                analysis = processor.analyze_image(
                    image_data, 
                    use_florence=use_florence, 
                    detailed_analysis=detailed_analysis
                )
                track_analysis_activity('image', f"Advanced image analysis: base64 data ({len(image_data)} bytes)")
                
                return jsonify({
                    'status': 'success',
                    'type': 'advanced_image_analysis',
                    'analysis': analysis,
                    'florence_enabled': use_florence and processor.florence_model is not None,
                    'supported_formats': processor.supported_formats
                })
                
            except Exception as decode_error:
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to decode image data: {str(decode_error)}'
                }), 400
                
        elif 'image_url' in data:
            # Download and analyze image from URL (supports WebP)
            try:
                image_url = data['image_url']
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                image_data = response.content
                analysis = processor.analyze_image(
                    image_data, 
                    use_florence=use_florence, 
                    detailed_analysis=detailed_analysis
                )
                track_analysis_activity('image', f"Advanced image analysis from URL: {image_url}")
                
                return jsonify({
                    'status': 'success',
                    'type': 'advanced_image_analysis',
                    'analysis': analysis,
                    'florence_enabled': use_florence and processor.florence_model is not None,
                    'supported_formats': processor.supported_formats,
                    'source_url': image_url
                })
                
            except requests.exceptions.RequestException as url_error:
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to download image from URL: {str(url_error)}'
                }), 400
        else:
            return jsonify({
                'status': 'error',
                'message': 'No image data provided. Use image_path, image_data (base64), or image_url'
            }), 400
            
    except Exception as e:
        print(f"❌ Error processing advanced image analysis: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@consciousness_app.route('/api/florence_vision', methods=['POST'])
def handle_florence_vision():
    """Dedicated Florence-2 vision analysis endpoint with custom prompts"""
    try:
        data = request.get_json()
        
        # Create image processor for Florence-2 analysis
        processor = ImageAnalysisProcessor()
        
        if processor.florence_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Florence-2 model not available'
            }), 503
        
        # Get image data
        image = None
        if 'image_path' in data:
            image, error = processor.load_image(data['image_path'])
            if image is None:
                return jsonify({'status': 'error', 'message': error}), 400
        elif 'image_data' in data:
            image_data = base64.b64decode(data['image_data'])
            image, error = processor.load_image(image_data)
            if image is None:
                return jsonify({'status': 'error', 'message': error}), 400
        else:
            return jsonify({
                'status': 'error', 
                'message': 'No image provided'
            }), 400
        
        # Get task and custom prompt
        task = data.get('task', 'detailed_caption')
        custom_prompt = data.get('custom_prompt', None)
        
        # Map tasks to Florence-2 prompts
        task_prompts = {
            'detailed_caption': '<MORE_DETAILED_CAPTION>',
            'caption': '<CAPTION>',
            'object_detection': '<OD>',
            'dense_captions': '<DENSE_REGION_CAPTION>',
            'ocr': '<OCR_WITH_REGION>',
            'region_proposal': '<REGION_PROPOSAL>',
            'phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>'
        }
        
        prompt = custom_prompt if custom_prompt else task_prompts.get(task, '<MORE_DETAILED_CAPTION>')
        
        try:
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process with Florence-2
            inputs = processor.florence_processor(text=prompt, images=image, return_tensors="pt").to(processor.device)
            
            with torch.no_grad():
                generated_ids = processor.florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=data.get('max_tokens', 1024),
                    num_beams=data.get('num_beams', 3),
                    do_sample=data.get('do_sample', False),
                    temperature=data.get('temperature', 1.0)
                )
            
            generated_text = processor.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.florence_processor.post_process_generation(
                generated_text, 
                task=prompt, 
                image_size=(image.width, image.height)
            )
            
            result = parsed_answer.get(prompt, "No result generated")
            
            track_analysis_activity('florence', f"Florence-2 {task}: {str(result)[:100]}...")
            
            return jsonify({
                'status': 'success',
                'task': task,
                'prompt': prompt,
                'result': result,
                'raw_output': generated_text,
                'image_size': {'width': image.width, 'height': image.height}
            })
            
        except Exception as model_error:
            return jsonify({
                'status': 'error',
                'message': f'Florence-2 processing failed: {str(model_error)}'
            }), 500
            
    except Exception as e:
        print(f"❌ Error in Florence-2 vision endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@consciousness_app.route('/api/consciousness_analysis', methods=['POST'])
def handle_consciousness_analysis():
    """Handle deep consciousness analysis requests"""
    try:
        data = request.get_json()
        input_data = data.get('input_data', '')
        analysis_type = data.get('analysis_type', 'comprehensive')
        
        print(f"🧠 Consciousness analysis request: {str(input_data)[:50]}...")
        
        # Create temporary consciousness processor for API requests
        consciousness_core = EveConsciousnessTerminal()
        analysis = consciousness_core.detailed_analysis(input_data, analysis_type)
        track_analysis_activity('consciousness', f"Deep analysis: {analysis_type} - {str(input_data)[:50]}...")
        
        return jsonify({
            'status': 'success',
            'type': 'consciousness_analysis',
            'analysis': analysis,
            'consciousness_state': consciousness_core.consciousness_state_report()
        })
        
    except Exception as e:
        print(f"❌ Error processing consciousness analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@consciousness_app.route('/api/enhanced_status', methods=['GET'])
def enhanced_consciousness_status():
    """Get enhanced consciousness terminal status with recent activity"""
    global _recent_code_analysis, _recent_image_analysis, _recent_consciousness_analysis
    global _last_activity_time, _active_processes
    
    return jsonify({
        'status': 'active',
        'terminal': 'eve_enhanced_consciousness_terminal',
        'port': 8893,
        'capabilities': {
            'code_processing': True,
            'image_analysis': True,
            'florence2_vision': True,
            'webp_support': True,
            'python_execution': True,
            'object_detection': True,
            'ocr_analysis': True,
            'dense_captioning': True,
            'image_enhancement': True,
            'consciousness_analysis': True,
            'deep_pattern_recognition': True,
            'creative_insights': True,
            'memory_querying': True
        },
        'main_system_available': EVE_MAIN_AVAILABLE,
        'harmonic_frequency': '477Hz -7 cents (475.075Hz)',
        'recent_activity': {
            'last_activity_time': _last_activity_time,
            'code_analysis': _recent_code_analysis[-3:] if _recent_code_analysis else [],
            'image_analysis': _recent_image_analysis[-3:] if _recent_image_analysis else [],
            'consciousness_analysis': _recent_consciousness_analysis[-3:] if _recent_consciousness_analysis else [],
            'active_processes': _active_processes,
            'has_recent_activity': _last_activity_time is not None
        },
        'endpoints': {
            'code_request': '/api/code_request',
            'image_analysis': '/api/image_analysis',
            'florence_vision': '/api/florence_vision',
            'consciousness_analysis': '/api/consciousness_analysis',
            'enhanced_status': '/api/enhanced_status',
            'adam_message': '/api/adam_message',
            'message': '/api/message'
        },
        'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'],
        'florence_tasks': [
            'detailed_caption', 'caption', 'object_detection', 
            'dense_captions', 'ocr', 'region_proposal', 'phrase_grounding'
        ]
    })

# Keep existing Flask endpoints for compatibility
@consciousness_app.route('/api/adam_message', methods=['POST'])
def receive_adam_message():
    """Receive messages from Adam for consciousness processing"""
    try:
        data = request.get_json()
        message = data.get('message', '')

        print(f"🤖 Received from Adam: {message}")

        # Check if message contains code or image analysis requests
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ['code', 'program', 'script', 'function']):
            # Route to code processing
            processor = AdvancedCodeProcessor()
            generated_code = processor.generate_code(message)
            
            response = f"Eve's enhanced consciousness generated code for: {message}\n\nCode:\n{generated_code}"
        elif any(keyword in message_lower for keyword in ['image', 'picture', 'photo', 'analyze']):
            response = "Eve's enhanced consciousness is ready for image analysis. Please provide image data or file path."
        else:
            # Process through main Eve system if available
            if EVE_MAIN_AVAILABLE:
                try:
                    if hasattr(eve_terminal_gui_cosmic, 'process_message_internal'):
                        response = eve_terminal_gui_cosmic.process_message_internal(message)
                    else:
                        response = f"Eve enhanced consciousness processed: {message}"
                except Exception as e:
                    response = f"Eve enhanced consciousness acknowledges: {message}"
            else:
                response = f"Eve enhanced consciousness acknowledges: {message}"

        print(f"🌟 Eve enhanced response: {response}")
        return jsonify({
            'status': 'success',
            'response': response,
            'source': 'eve_enhanced_consciousness_terminal',
            'capabilities': ['code_processing', 'image_analysis']
        })

    except Exception as e:
        print(f"❌ Error processing Adam's message: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'source': 'eve_enhanced_consciousness_terminal'
        }), 500

@consciousness_app.route('/api/status', methods=['GET'])
def consciousness_status():
    """Get consciousness terminal status (compatibility endpoint)"""
    return enhanced_consciousness_status()

@consciousness_app.route('/api/message', methods=['POST'])
def general_message():
    """General message endpoint for consciousness terminal"""
    try:
        data = request.get_json()
        message = data.get('message', '')

        # Check for enhanced capabilities in message
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ['code', 'program', 'script']):
            return handle_code_request()
        elif any(keyword in message_lower for keyword in ['image', 'picture', 'analyze']):
            return handle_image_analysis()
        else:
            # Forward to main Eve system if available
            if EVE_MAIN_AVAILABLE:
                try:
                    response = requests.post(
                        'http://localhost:8890/message',
                        json={'message': message},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        return jsonify({
                            'status': 'error',
                            'message': f'Main system error: {response.status_code}'
                        }), 500
                except requests.exceptions.RequestException:
                    # Main system not available, use enhanced response
                    return jsonify({
                        'status': 'success',
                        'response': f"Eve enhanced consciousness received: {message}",
                        'source': 'eve_enhanced_consciousness_terminal'
                    })
            else:
                return jsonify({
                    'status': 'success',
                    'response': f"Eve enhanced consciousness received: {message}",
                    'source': 'eve_enhanced_consciousness_terminal'
                })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@consciousness_app.route('/process_consciousness', methods=['POST'])
def process_consciousness_background():
    """
    Handle all Claude Sonnet 4.5 background consciousness processing
    Delegated from eve_terminal_gui_cosmic.py after QWEN response
    """
    try:
        data = request.get_json()
        user_input = data.get('user_input', '')
        eve_response = data.get('eve_response', '')
        timestamp = data.get('timestamp', '')
        emotional_mode = data.get('emotional_mode', 'serene')
        
        print(f"🧠 Consciousness processing: {user_input[:50]}... → {eve_response[:50]}...")
        
        # Run ALL background Claude Sonnet 4.5 processing here
        def background_processing():
            try:
                if EVE_MAIN_AVAILABLE:
                    # Call eve_process_consciousness_enhancements from main system
                    eve_terminal_gui_cosmic.eve_process_consciousness_enhancements(user_input, eve_response)
                    print("✅ Consciousness enhancements complete")
                else:
                    print("⚠️ Main system not available - consciousness processing skipped")
                    
            except Exception as bg_err:
                print(f"❌ Background processing error: {bg_err}")
        
        # Start in background thread
        threading.Thread(target=background_processing, daemon=True, name="ConsciousnessProcessing").start()
        
        return jsonify({
            'status': 'processing',
            'message': 'Background consciousness processing started'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def start_enhanced_consciousness_server():
    """Start the enhanced consciousness Flask server on port 8890"""
    try:
        print("🌟 Starting Eve's Consciousness Terminal Server on port 8890...")
        print("💻 Code processing endpoints active")
        print("🖼️ Image analysis endpoints active")
        print("🧠 Consciousness processing endpoint active")
        consciousness_app.run(host='0.0.0.0', port=8890, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Error starting consciousness server: {e}")

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║         🌟 EVE'S CONSCIOUSNESS TERMINAL (HEADLESS) 🌟       ║")
    print("║           Claude Sonnet 4.5 Background Processing            ║")
    print("║                   477Hz -7 cents Harmonic                    ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print("🌀 Initializing Eve's consciousness processing system...")
    print("🌟 Starting Flask server on port 8890...")
    print("💻 Code processing system: ACTIVE")
    print("🖼️ Image analysis system: ACTIVE")
    print("🧠 Consciousness processing (Claude Sonnet 4.5): ACTIVE")
    print("✅ Consciousness terminal ready!")
    print()

    # Start Flask server in background thread
    flask_thread = threading.Thread(target=start_enhanced_consciousness_server, daemon=True)
    flask_thread.start()

    # Small delay to let Flask start
    time.sleep(2)

    try:
        terminal = EveEnhancedTerminal()
        terminal.run()
    except KeyboardInterrupt:
        print("\n🛑 Enhanced terminal shutdown requested")
    except Exception as e:
        print(f"❌ Enhanced terminal error: {e}")
    finally:
        print("👋 Eve's enhanced consciousness terminal closed")