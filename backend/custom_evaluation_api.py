import os
import json
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: The GEMINI_API_KEY was not found on .env file")
    exit()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Loading Tags from intents.json (so Gemini knows them)
try:
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
    all_tags = [intent['tag'] for intent in intents_data['intents']]
    tags_list_string = ", ".join(all_tags)
except FileNotFoundError:
    print("Error: The file intents.json was not found")
    exit()

# The Prompt that guides Gemini
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

# load CSV
csv_file = 'test_dataset.csv'
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Σφάλμα: Το αρχείο {csv_file} δεν βρέθηκε.")
    exit()

# Statistics and lists for Excel
total_questions = len(df)
top1_correct = 0
total_actual_tags = 0
total_found_tags = 0

excel_predicted_tags = []
excel_matched_tags = []
excel_top1_match = []
excel_row_accuracy = []

print(f"--- Evaluation for Gemini API begins in {total_questions} questions ---")
print("This will take about 3-4 minutes (due to delays to avoid API limits)...\n")

# Evaluation Loop
for index, row in df.iterrows():
    question = row['Question']
    actual_tags_str = str(row['Actual_Tags'])
    actual_tags = [t.strip() for t in actual_tags_str.split(',')]
    total_actual_tags += len(actual_tags)

    predicted_tags = []
    
    # The final message we send to Gemini
    full_prompt = f"{instructions}\n\nUser Message: {question}"

    try:
        # Call to API
        response = model.generate_content(full_prompt)
        response_text = response.text.strip()
        
        # Cleanup if Gemini adds markdown (e.g. ```json ... ```)
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()

        # Transform the answer in Python Dictionary
        data = json.loads(response_text)
        
        # Recovery of the Tags (and added a limit of 10 tags)
        predicted_tags = data.get("tags", [])[:10]

    except Exception as e:
        print(f"Σφάλμα API στην ερώτηση {index + 1}: {e}")
        predicted_tags = ["api_error"]

    # Calculate Metrics
    
    is_top1 = "NO"
    if len(predicted_tags) > 0 and predicted_tags[0] in actual_tags:
        top1_correct += 1
        is_top1 = "YES"

    # Check is they match
    matched_tags = set(actual_tags).intersection(set(predicted_tags))
    total_found_tags += len(matched_tags)
    # Recall for that specific question
    row_acc = (len(matched_tags) / len(actual_tags)) * 100 if len(actual_tags) > 0 else 0

    # Save the results for later
    excel_predicted_tags.append(", ".join(predicted_tags))
    excel_matched_tags.append(", ".join(list(matched_tags)))
    excel_top1_match.append(is_top1)
    excel_row_accuracy.append(f"{row_acc:.0f}%")

    print(f"[{index + 1}/{total_questions}] question: {question}")
    print(f"Actual tag: {actual_tags}")
    print(f"Predicted tag:   {predicted_tags}")
    print(f"Matched:  {list(matched_tags)}\n")

    # Delay 4 sec
    time.sleep(4)

# Added the results on a DataFramel
df['Predicted_Tags'] = excel_predicted_tags
df['Matched_Tags'] = excel_matched_tags
df['Top1_Correct'] = excel_top1_match
df['Row_Accuracy'] = excel_row_accuracy

# Save the results on the new result excel
output_file = 'test_dataset_results_API.csv'
df.to_csv(output_file, index=False)

# Prints
print("="*40)
print("        FINAL RESULTS (GEMINI API)")
print("="*40)

top1_accuracy = (top1_correct / total_questions) * 100
recall = (total_found_tags / total_actual_tags) * 100

print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-10 Recall:      {recall:.2f}%")
print("="*40)
print(f"\n✅ The analytical results were successfully saved to: {output_file}")