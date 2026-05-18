import os
import json
import torch
import pandas as pd
from model import NeuralNet
# remove nltk_utils,include sentence_transformers
from sentence_transformers import SentenceTransformer

# Loading the model and the data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = "model_bert.pth"
data = torch.load(FILE, map_location=device, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tags = data['tags']
model_state = data["model_state"]
# all_words was deleted cause BERT does not have our dictionary

# ΑΛΛΑΓΗ 2: Φόρτωση του BERT
print("Φόρτωση του BERT μοντέλου (all-MiniLM-L6-v2)...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Loading excel
csv_file = 'test_dataset.csv'
try:
    df = pd.read_csv(csv_file) 
    # automatically recognizes that the first row contains the column headings (Headers)
except FileNotFoundError:
    print(f"Error: The file {csv_file} was not found.")
    exit()

# Variables for statistics
total_questions = len(df)
top1_correct = 0 #total number of all tags I have on the excel
total_actual_tags = 0
total_found_tags = 0

THRESHOLD = 0.02
MAX_TAGS = 10

excel_predicted_tags = []
excel_matched_tags = []
excel_top1_match = []
excel_row_accuracy = []

print(f"--- Evaluation (BERT) begins in {total_questions} questions ---\n")

# Evaluation Loop
for index, row in df.iterrows():
    question = row['Question']
    actual_tags_str = str(row['Actual_Tags'])
    actual_tags = [t.strip() for t in actual_tags_str.split(',')]
    total_actual_tags += len(actual_tags)

    # ΑΛΛΑΓΗ 3: Το BERT μετατρέπει την πρόταση απευθείας σε Διάνυσμα 384 αριθμών
    embedding = bert_model.encode(question)
    X = torch.from_numpy(embedding).unsqueeze(0).to(torch.float32).to(device)

    output = model(X)
    probs = torch.softmax(output, dim=1)[0]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    predicted_tags = []
    
    for i in range(len(sorted_probs)):
        if len(predicted_tags) >= MAX_TAGS:
            break
        prob_value = sorted_probs[i].item()
        if prob_value > THRESHOLD:
            tag_name = tags[sorted_indices[i].item()]
            predicted_tags.append(tag_name)

    # Calculate Metrics
    
    # Check Top-1
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

    print(f"Question: {question}")
    print(f"Actual tag: {actual_tags}")
    print(f"Predicted tag:   {predicted_tags[:5]}... (Top 5)")
    print(f"Matched:  {list(matched_tags)}\n")

# Added the results on a DataFrame
df['Predicted_Tags'] = excel_predicted_tags
df['Matched_Tags'] = excel_matched_tags
df['Top1_Correct'] = excel_top1_match
df['Row_Accuracy'] = excel_row_accuracy

# Save the results on the new result excel
# Setup Output Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "custom_evaluation_results")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

output_file = 'test_dataset_results_bert.csv'
output_path = os.path.join(RESULTS_DIR, output_file)
df.to_csv(output_path, index=False) 

# Prints
print("="*40)
print("            FINAL RESULTS (BERT)")
print("="*40)

top1_accuracy = (top1_correct / total_questions) * 100
recall = (total_found_tags / total_actual_tags) * 100

print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-10 Recall:      {recall:.2f}%")
print("="*40)
print(f"\n✅ The analytical results were successfully saved to: {output_path}")