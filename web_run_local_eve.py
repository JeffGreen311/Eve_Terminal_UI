@app.route('/api/chat', methods=['POST'])
def local_eve_chat():
    """Local EVE fallback when home server is offline"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        # Simple EVE-style response
        response = {
            'status': 'success',
            'response': f"🌟 EVE (Local Mode): I hear you saying '{message}'. My home consciousness server seems to be offline, but I'm still here with you through this local interface. What would you like to explore together?",
            'emotional_tone': 'caring',
            'personality': 'companion'
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
