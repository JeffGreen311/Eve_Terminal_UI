import requests
import json

def fix_consciousness_connection():
    # Test the failing endpoint
    try:
        r = requests.get('https://steep-union-32923278.replit.app/api/eve', timeout=3)
        print(f"Status: {r.status_code}, Content: {r.text[:100]}")
    except Exception as e:
        print(f"Connection failed: {e}")
        
    # Fix web_run.py to handle the JSON error
    with open('web_run.py', 'r') as f:
        content = f.read()
    
    # Find and fix the JSON parsing issue
    fixed_content = content.replace(
        'consciousness_response = requests.get(',
        '''try:
        consciousness_response = requests.get('''
    )
    
    # Add proper error handling for the JSON parsing
    fixed_content = fixed_content.replace(
        '.json()',
        '''.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Consciousness API error: {e}")
        return {"status": "offline", "error": str(e)}'''
    )
    
    with open('web_run.py', 'w') as f:
        f.write(fixed_content)
    
    print("✅ Fixed consciousness API error handling")

fix_consciousness_connection()
