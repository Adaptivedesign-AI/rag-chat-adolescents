from flask import Flask, render_template, request, jsonify
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Import your custom modules (you'll create these)
from models.digital_twins import DigitalTwinManager
from models.gemini_client import GeminiClient
from models.rag_system import RAGSystem

# Initialize global objects
twin_manager = DigitalTwinManager()
gemini_client = GeminiClient(os.environ.get('GEMINI_API_KEY'))
rag_system = RAGSystem()

# Store session data (in production, use proper session management)
user_sessions = {}

@app.route('/')
def index():
    """Main page - Twin selection"""
    return render_template('index.html')

@app.route('/chat/<twin_type>')
def chat(twin_type):
    """Chat page for specific twin"""
    if twin_type not in ['healthy', 'anxiety', 'depression']:
        return redirect('/')
    
    twin = twin_manager.get_twin(twin_type)
    if not twin:
        return redirect('/')
    
    return render_template('chat.html', 
                         twin_type=twin_type,
                         twin_name=twin.name,
                         twin_data=twin.get_profile_data())

@app.route('/api/send_message', methods=['POST'])
def send_message():
    """Handle chat messages"""
    try:
        data = request.json
        message = data.get('message', '')
        twin_type = data.get('twin_type', 'healthy')
        scenario = data.get('scenario', 'neutral')
        rag_enabled = data.get('rag_enabled', False)
        
        if not message.strip():
            return jsonify({'error': 'Empty message'}), 400
        
        # Get the twin
        twin = twin_manager.get_twin(twin_type)
        if not twin:
            return jsonify({'error': 'Invalid twin type'}), 400
        
        # Update twin state
        twin.current_scenario = scenario
        twin.rag_enabled = rag_enabled
        
        # Prepare context for RAG if enabled
        rag_context = None
        if rag_enabled:
            rag_context = rag_system.retrieve_relevant_info(
                query=message,
                twin_type=twin_type,
                scenario=scenario
            )
        
        # Generate response using Gemini
        response = gemini_client.chat_with_twin(
            message=message,
            twin=twin,
            rag_context=rag_context
        )
        
        return jsonify({
            'response': response,
            'twin_type': twin_type,
            'scenario': scenario,
            'rag_enabled': rag_enabled
        })
        
    except Exception as e:
        print(f"Error in send_message: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/switch_scenario', methods=['POST'])
def switch_scenario():
    """Handle scenario switching"""
    try:
        data = request.json
        twin_type = data.get('twin_type')
        scenario = data.get('scenario')
        
        if scenario not in ['neutral', 'toxic']:
            return jsonify({'error': 'Invalid scenario'}), 400
        
        twin = twin_manager.get_twin(twin_type)
        if twin:
            twin.current_scenario = scenario
        
        return jsonify({
            'success': True,
            'twin_type': twin_type,
            'scenario': scenario
        })
        
    except Exception as e:
        print(f"Error in switch_scenario: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/toggle_rag', methods=['POST'])
def toggle_rag():
    """Handle RAG toggle"""
    try:
        data = request.json
        twin_type = data.get('twin_type')
        rag_enabled = data.get('rag_enabled', False)
        
        twin = twin_manager.get_twin(twin_type)
        if twin:
            twin.rag_enabled = rag_enabled
        
        return jsonify({
            'success': True,
            'twin_type': twin_type,
            'rag_enabled': rag_enabled
        })
        
    except Exception as e:
        print(f"Error in toggle_rag: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({'status': 'healthy'})

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # For development
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))