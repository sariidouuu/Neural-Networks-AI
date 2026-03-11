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

# We choose the Google model
# We can send about 15 questions/minute and 1000/day
model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")

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

if __name__ == '__main__':
    # The server will run topically on port 5000
    app.run(debug=True, port=5000)