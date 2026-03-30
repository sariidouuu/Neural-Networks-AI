import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

import torch
import random
import json
from nltk_utils import bag_of_words, tokenize
from model import NeuralNet

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# ────── TRAINDE MODEL ──────

# Loading the secret api key from .env file
load_dotenv()

app = Flask(__name__)
# CORS allows on script.js to "talk" with app.py
CORS(app) 

# Setting up the Gemini API with our key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load the traned model
#FILE = "model.pth"
#data = torch.load(FILE, map_location=torch.device('cpu'))

# Βρίσκει τη διαδρομή του φακέλου στον οποίο βρίσκεται το app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ενώνει το μονοπάτι του φακέλου με το όνομα του αρχείου
FILE = os.path.join(BASE_DIR, "model.pth")
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

# Τώρα στο torch.load χρησιμοποιείς το σωστό FILE
data = torch.load(FILE, map_location=torch.device('cpu'))

# Σιγουρέψου ότι και στο άνοιγμα του intents χρησιμοποιείς το INTENTS_PATH
with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
    intents = json.load(f)


input_size  = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words   = data["all_words"]
tags        = data["tags"]
model_state = data["model_state"]

model1 = NeuralNet(input_size, hidden_size, output_size)
model1.load_state_dict(model_state)
model1.eval()  # Θέτει το μοντέλο σε evaluation mode (απενεργοποιεί το dropout)

# Φόρτωση intents για τις απαντήσεις
#with open('intents.json', 'r', encoding='utf-8') as f:
    #intents_data = json.load(f)

@app.route('/chat1', methods=['POST'])
def chat1():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Επεξεργασία του prompt
    sentence     = tokenize(user_message)
    X            = bag_of_words(sentence, all_words)
    X            = torch.from_numpy(X).unsqueeze(0)  # προσθέτει batch dimension

    # Πρόβλεψη
    output       = model1(X)
    _, predicted = torch.max(output, dim=1)
    tag          = tags[predicted.item()]

    # Έλεγχος confidence με softmax
    probs       = torch.softmax(output, dim=1)
    confidence  = probs[0][predicted.item()].item()

    if confidence > 0.75:
        # Αν είναι αρκετά σίγουρο, επίλεξε τυχαία απάντηση από το intent
        for intent in intents_data['intents']:
            if intent['tag'] == tag:
                reply = random.choice(intent['responses'])
                return jsonify({"reply": reply})
    else:
        return jsonify({"reply": "I'm not sure I understand. Could you rephrase?"})


# ────── API AND STUDIO AI ──────

# We choose the Google model and pass the system instructions. 
# The restrictions we impose to google ai studio:
instructions = """
STRICT OPERATING PROTOCOL:
1. You must answer ONLY in English. You may reply with: "I'm sorry but i'm only allowed to answer in English."
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

@app.route('/chat2', methods=['POST'])
def chat2():
    # We take the incoming data (JSON) JavaScript sent us
    data = request.get_json()

    #user_message = data.get("message")
    history = data.get("history", []);

    # If someone comes to this address (/chat), you will only serve them if they bring you a packet of data (the message). 
    # If they come empty-handed (they just wanna see the page) tell them no.
    if not history:
        return jsonify({"error": "No message provided"}), 400

    # else we try to respond:
    try:
        # We transform the history in a format the google ai studio needs/understands
        gemini_history = []
        for msg in history[:-1]:  # all, exept the last one
            # We tell the ai to read the entire list except the last item (=the last prompt the user sent)
            
            # For each old message, it checks who spoke (user or model).
            # It builts a new dictionary in a format the ai studio understands and puts it in the history.
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_history.append({
                'role': role,
                'parts': [msg['content']]
            })

        # Start a session with the history
        chat_session = model.start_chat(history=gemini_history)
        
        # Send the last message and await a response
        last_message = history[-1]['content']
        response = chat_session.send_message(last_message)
        
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


# ────── MAIN - PORTS ──────
# If u wanna run the project locally you select the first main here.
# and on script.js file you switch lines 173-174 and 197-198.
# The live server VS code has port=5500. 
# The backend (app.py) has port=5000. The forntend (script.js) has to send signals to the backend, so port=5000.
# u write on terminal: cd backend -> py app.py
# Wait until you see:
    # * Debugger is active!
    # * Debugger PIN:
# then u can go live


#if __name__ == '__main__':
    # The server will run topically on port 5000
    #app.run(debug=True, port=5000)


if __name__ == '__main__':
    # On Render (Cloud), the port is defined automatically from the system
    # The os.environ.get("PORT") reads this port
    # If we run (the code) localy and the PORT variable doesn't exists, we use 5000.
    port = int(os.environ.get("PORT", 5000))
    
    # The host='0.0.0.0' is required for hosting (Render/Docker ..)
    # It allows the server to accept requests from external addresses
    # and not only from localhost (127.0.0.1) (our pc)
    app.run(host='0.0.0.0', port=port)