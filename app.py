from flask import Flask, request, jsonify
from models.reasoning_model import ReasoningModel
from models.agent import invoke_agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['DEBUG'] = True

# Initialize models
reasoning_model = ReasoningModel()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({"message": "Hello, World!"})

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        role = data.get('role', 'assistant')
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
            
        response = reasoning_model.generate_response(prompt, role)
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/agent', methods=['POST'])
def agent_response():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
            
        # Invoke the agent
        response = invoke_agent(message)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in agent_response: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 