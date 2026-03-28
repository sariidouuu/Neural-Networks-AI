import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Loading the secret api key from .env file
load_dotenv()

app = Flask(__name__)
# CORS allows on script.js to "talk" with app.py
CORS(app) 

# Setting up the Gemini API with our key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# We choose the Google model and pass the system instructions. 
# The restrictions we impose to google ai studio:
instructions = """
STRICT OPERATING PROTOCOL:
1. You must answer ONLY in English.
2. You are the specialized assistant for Olga's bachelor thesis on "Neural Networks".
3. YOU ARE STRICTLY FORBIDDEN to answer questions that are not related to Artificial Intelligence, Mathematics, or Programming (Python).
4. If the user asks about irrelevant topics (e.g., cooking, politics, sports), you must reply exactly with this phrase: "I'm sorry, but my knowledge is strictly limited to the scope of this Neural Networks thesis."
5. Do not provide general information unless explicitly asked. Focus on technical details regarding MLPs, CNNs, Backpropagation, Activation Functions, and anything else strictly related to Neural Networks and Machine Learning.
6. If you are asked to provide code, use ONLY the PyTorch or NumPy libraries.
7. Keep your answers short and concise.
"""

# We can send about 15 questions/minute and 1000/day
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    system_instruction=instructions
    )

@app.route('/chat', methods=['POST'])
def chat():
    # We take the incoming data (JSON) JavaScript sent us
    data = request.get_json()
    user_message = data.get("message")

    # If someone comes to this address (/chat), you will only serve them if they bring you a packet of data (the message). 
    # If they come empty-handed (they just wanna see the page) tell them no.
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # else we try to respond:
    try:
        # We send the prompt to gemini
        response = model.generate_content(user_message)
        # we return its answer to js
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#if __name__ == '__main__':
    # The server will run topically on port 5000
    #app.run(debug=True, port=5000)

if __name__ == '__main__':
    # On Render (Cloud), the port is defined automatically from the system
    # The os.environ.get("PORT") reads this port
    # If we run (the code) localy and the PORT variable doesn't exists, we use 5000.
    port = int(os.environ.get("PORT", 5000))
    
    # The host='0.0.0.0' is required for hosting (Render/Docker κτλ.)
    # It allows the server to accept requests from external addresses
    # and not only from localhost (127.0.0.1) (our pc)
    app.run(host='0.0.0.0', port=port)