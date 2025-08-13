import replicate
import os
import base64
from pathlib import Path

def analyze_image_with_vision_ai(filepath, filename):
    """Analyze image using Replicate's GPT-4 Vision API"""
    try:
        # Convert local file to base64 for processing
        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Create data URL for the image
        file_ext = os.path.splitext(filename)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }.get(file_ext, 'image/jpeg')
        
        image_url = f"data:{mime_type};base64,{image_data}"
        
        input_data = {
            "prompt": "Analyze this image in detail. Describe what you see, read any text present, and explain the context. Be thorough but concise.",
            "image_input": [image_url],
            "system_prompt": "You are EVE Prime's visual analysis system. Provide detailed, helpful descriptions of images to enhance conversation context."
        }
        
        # Run the vision analysis
        output = replicate.run(
            "openai/gpt-4.1-mini",
            input=input_data
        )
        
        # Join the output if it's a list
        analysis = "".join(output) if isinstance(output, list) else str(output)
        
        return f"""--- VISION ANALYSIS: {filename} ---
File size: {os.path.getsize(filepath)} bytes
Format: {file_ext.upper()[1:]}

VISUAL CONTENT ANALYSIS:
{analysis}
--- END VISION ANALYSIS ---"""
        
    except Exception as e:
        return f"""--- VISION ANALYSIS ERROR: {filename} ---
Error analyzing image: {str(e)}
File size: {os.path.getsize(filepath)} bytes
--- END ANALYSIS ---"""

# Enhanced process_uploaded_files function
def process_uploaded_files_enhanced():
    """Process uploaded files with vision AI integration"""
    if not uploaded_files:
        return ""
    
    file_contents = []
    
    for file_info in uploaded_files:
        try:
            filepath = file_info['filepath']
            filename = file_info['filename']
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.yaml', '.yml', '.csv', '.xml']:
                # Text-based files - read content directly
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_contents.append(f"\n--- TEXT FILE: {filename} ---\n{content}\n--- END FILE ---\n")
                except UnicodeDecodeError:
                    try:
                        with open(filepath, 'r', encoding='latin-1') as f:
                            content = f.read()
                        file_contents.append(f"\n--- TEXT FILE: {filename} ---\n{content}\n--- END FILE ---\n")
                    except:
                        file_contents.append(f"\n--- TEXT FILE: {filename} ---\n[Could not decode text file]\n--- END FILE ---\n")
            
            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                # Image files - USE VISION AI ANALYSIS
                vision_analysis = analyze_image_with_vision_ai(filepath, filename)
                file_contents.append(f"\n{vision_analysis}\n")
            
            elif file_ext in ['.pdf', '.doc', '.docx']:
                # Document files - placeholder for future OCR integration
                file_size = os.path.getsize(filepath)
                file_contents.append(f"\n--- DOCUMENT FILE: {filename} ---\nSize: {file_size} bytes\n[DOCUMENT PROCESSING: OCR capability needed]\n--- END DOCUMENT ---\n")
            
            else:
                # Unknown file type
                file_size = os.path.getsize(filepath)
                file_contents.append(f"\n--- UNKNOWN FILE: {filename} ---\nSize: {file_size} bytes\n[Unsupported file type for analysis]\n--- END FILE ---\n")
                
        except Exception as e:
            file_contents.append(f"\n--- ERROR PROCESSING FILE: {filename} ---\n{str(e)}\n--- END ERROR ---\n")
    
    return "\n".join(file_contents)
