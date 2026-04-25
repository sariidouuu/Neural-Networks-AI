import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Διαθέσιμα μοντέλα για το κλειδί σου:")
for m in genai.list_models():
    # Εμφανίζει το όνομα και τις δυνατότητες του κάθε μοντέλου
    print(f"Name: {m.name} | Capabilities: {m.supported_generation_methods}")