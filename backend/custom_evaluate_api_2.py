import os
import json
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Αρχικοποίηση του Gemini API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: The GEMINI_API_KEY was not found on .env file")
    exit()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# 2. Φόρτωση των Tags
try:
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
    all_tags = [intent['tag'] for intent in intents_data['intents']]
    tags_list_string = ", ".join(all_tags)
except FileNotFoundError:
    print("Error: The file intents.json was not found")
    exit()

# 3. Prompt
instructions = f"""
STRICT OPERATING PROTOCOL:
1. You are the specialized assistant for a bachelor thesis on "Neural Networks".
2. CRITICAL RULE: Provide your answer as plain text. 
3. TASK: Below is a user message. You must answer the message AND classify it using the provided tags.
4. Available tags: {tags_list_string}.
5. Select up to 10 most relevant tags that match the user's prompt.
6. YOU MUST respond ONLY in the following JSON format without any markdown blocks:
{{"reply": "your_response_here", "tags": ["tag1", "tag2"]}}
"""

# Load csv and output file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "custom_evaluation_results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

csv_file = 'test_dataset.csv'
output_path = os.path.join(RESULTS_DIR, 'test_dataset_results_api.csv')

# 5. Checkpointing
if os.path.exists(output_path):
    print("Previous file found! Loading data to continue...")
    df = pd.read_csv(output_path)
else:
    print("Creating a new assessment file...")
    df = pd.read_csv(csv_file)
    # Create empty columns if not already exists 
    df['Predicted_Tags'] = None
    df['Matched_Tags'] = None
    df['Top1_Correct'] = None
    df['Row_Accuracy'] = None

total_questions = len(df)
questions_done = df['Predicted_Tags'].notna().sum()

print(f"--- {questions_done}/{total_questions} questions have already been answered. ---")

# 6. Evaluation Loop (With check every 10 questions)
processed_in_this_run = 0

for index, row in df.iterrows():
    # if we answered the question, go to the next one
    if pd.notna(row['Predicted_Tags']) and str(row['Predicted_Tags']).strip() != "":
        continue

    question = row['Question']
    actual_tags_str = str(row['Actual_Tags'])
    actual_tags = [t.strip() for t in actual_tags_str.split(',')]

    predicted_tags = []
    full_prompt = f"{instructions}\n\nUser Message: {question}"

    # API Call
    try:
        response = model.generate_content(full_prompt)
        response_text = response.text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()

        data = json.loads(response_text)
        predicted_tags = data.get("tags", [])[:10]
    except Exception as e:
        print(f"API Error (probably: Limit) in question {index + 1}: {e}")
        print("We stop here to protect our process.")
        break # If we hit a limit, the loop stops immediately to avoid filling up with errors.

    # Calculate Metrics 
    is_top1 = "NO"
    if len(predicted_tags) > 0 and predicted_tags[0] in actual_tags:
        is_top1 = "YES"

    matched_tags = set(actual_tags).intersection(set(predicted_tags))
    row_acc = (len(matched_tags) / len(actual_tags)) * 100 if len(actual_tags) > 0 else 0

    # Save directly to DataFrame
    df.at[index, 'Predicted_Tags'] = ", ".join(predicted_tags)
    df.at[index, 'Matched_Tags'] = ", ".join(list(matched_tags))
    df.at[index, 'Top1_Correct'] = is_top1
    df.at[index, 'Row_Accuracy'] = f"{row_acc:.0f}%"

    # SAVE AFTER EVERY QUESTION (Absolute security)
    df.to_csv(output_path, index=False)

    print(f"[{index + 1}/{total_questions}] {question}")
    print(f"Πρόβλεψη: {predicted_tags[:5]}")
    print(f"Matched:  {list(matched_tags)}\n")

    time.sleep(4) # Μικρό delay για σταθερότητα
    processed_in_this_run += 1

    # Check for pause every 10 questions
    if processed_in_this_run >= 10:
        print("="*40)
        print("✅ They were processed 10 question with success!")
        print(f"Progress saved to {output_path}")
        print("="*40)
        choice = input("Press ENTER to run another 10, or press 'q' (and Enter) for exit: ")
        
        if choice.lower() == 'q':
            break
        processed_in_this_run = 0

# 7. Final Calculation (Only if ALL 45 have been answered)
completed_count = df['Predicted_Tags'].notna().sum()

if completed_count == total_questions:
    print("\n" + "="*40)
    print("          FINAL RESULTS (GEMINI API)")
    print("="*40)

    top1_correct = (df['Top1_Correct'] == 'YES').sum()
    top1_accuracy = (top1_correct / total_questions) * 100

    total_actual = 0
    total_found = 0

    for idx, r in df.iterrows():
        act = [t.strip() for t in str(r['Actual_Tags']).split(',')]
        if pd.notna(r['Matched_Tags']) and str(r['Matched_Tags']).strip() != "":
            fnd = [t.strip() for t in str(r['Matched_Tags']).split(',')]
        else:
            fnd = []
        total_actual += len(act)
        total_found += len(fnd)

    recall = (total_found / total_actual) * 100

    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-10 Recall:  {recall:.2f}%")
    print("="*40)
else:
    print(f"\nEvaluation interrupted. Completed {completed_count} out of {total_questions} questions.")
    print("When ready again, run the script and it will automatically continue from the next question!")