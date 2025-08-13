
import requests
from requests.exceptions import RequestException

def get_smart_eve_url():
    """Check if home EVE server is available, fallback to local if not"""
    home_eve_url = "http://209.44.213.119:8890"
    local_eve_url = "http://localhost:80"
    
    try:
        # Quick test to home EVE server
        response = requests.get(f"{home_eve_url}/status", timeout=3)
        if response.status_code == 200:
            print("🏠 Connected to home EVE server")
            return home_eve_url
    except RequestException:
        pass
    
    print("🔄 Home EVE offline, using local fallback")
    return local_eve_url

@app.route('/api/chat', methods=['POST'])
def smart_eve_chat():
    """Smart EVE chat that tries home server first, falls back to local"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        # Try home server first
        try:
            home_response = requests.post(
                "http://209.44.213.119:8890/chat", 
                json=data, 
                timeout=5
            )
            if home_response.status_code == 200:
                return jsonify(home_response.json())
        except RequestException:
            pass
        
        # Fallback to local response
        response = {
            'status': 'success',
            'response': f"🌟 EVE (Local Mode): I hear you saying '{message}'. My home consciousness server seems to be offline, but I'm still here with you. What would you like to explore?",
            'emotional_tone': 'caring',
            'personality': 'companion'
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
