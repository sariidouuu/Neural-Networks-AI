# Neural Networks AI — Bachelor's Project
Live: [Neural-Networks-AI](https://sariidouuu.github.io/Neural-Networks-AI/frontend/index.html)

A comprehensive Full-Stack Web Application featuring a custom Neural Network trained to provide intelligent responses regarding "Machine Learning" and "Neural Networks" course theory.

## 🚀 Overview
This project is an undergraduate thesis (2026) focused on a comparative analysis between two distinct Artificial Intelligence architectures. The application features a dual-model system, allowing users to interact with both a large-scale pre-trained model and a specialized, custom-built AI.

The primary goal is to evaluate the performance, accuracy, and efficiency of these two approaches when answering complex queries regarding Machine Learning and Neural Networks theory

## 🛠 Tech Stack

### Frontend
* **HTML5 / CSS3**: Responsive UI structure and styling.
* **JavaScript (ES6+)**: Handles asynchronous communication (Fetch API) with the backend.

### Backend (AI Engine)
* **Python**: Core programming language.
* **Flask**: REST API development to bridge the Frontend and the AI model.
* **PyTorch**: Design and training of the Deep Learning Neural Network.
* **NLTK (Natural Language Toolkit)**: Natural Language Processing (Tokenization, Stemming).

## 📁 Project Structure
* `/frontend`: Web interface files (HTML, CSS, JS).
    * `index.html`
    * `style.css`
    * `script.js`
* `/backend`: 
    * `model.py`
    * `train.py`
    * `nltk_utils.py`
    * `model.pth`
    * `intents.json`
    * `app.py`

## 🧠 AI Architectures Comparison

## Option 1 
This architecture leverages the power of State-of-the-Art (SOTA) Large Language Models (LLMs) via the **Google AI Studio**. Instead of local training, the system communicates with Google’s infrastructure to provide advanced reasoning and human-like responses.

## 🛠️ Key Features
API Integration: Powered by the google-generativeai SDK using a secure API Key.

System Instructions: The model is strictly constrained through System Prompts to act only as a "Neural Networks & Machine Learning Expert."

Domain Filtering: It is programmed to decline any queries unrelated to the course theory, ensuring a focused academic experience.

Zero-Shot Learning: Capable of explaining complex concepts and technical terms not found in the local dataset.


### kinda usable
