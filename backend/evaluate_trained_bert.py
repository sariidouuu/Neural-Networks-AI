import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

from model import NeuralNet

# Set the device ( GPU -if exists- or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup Output Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results_bert")

# Creates the folder if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# 1. Loading data
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Loading the trained BERT model data
FILE = "model_bert.pth"
data = torch.load(FILE, map_location=device, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tags = data['tags']
model_state = data["model_state"]

# 2. Loading BERT & Model
print("Loading BERTT Model...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Setting the model in evaluation mode (eval)
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Lists to keep the real and predicted values
y_true = []
y_pred = []

# 3. load all intents from the model to see what it predictes
print("Evaluation of the BERT model. Please wait...")
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        # Instead of a bag_of_words, we use the 'encode' from BERT
        X = bert_model.encode(pattern)
        X = torch.from_numpy(X).unsqueeze(0).to(torch.float32).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        y_true.append(tags.index(tag))
        y_pred.append(predicted.item())

# 4. Creating Text Report and extract metrics
print("\n" + "="*80)
print(" METRICS (BERT) ")
print("="*80)

# Prints in terminal and store in txt
report_text = classification_report(y_true, y_pred, target_names=tags)
print(report_text)
report_path = os.path.join(RESULTS_DIR, "metrics_report_bert.txt")
with open(report_path, "w", encoding='utf-8') as f:
    f.write(report_text)

# Extract the metrics in dictionary form, to create Bar Chart
report_dict = classification_report(y_true, y_pred, target_names=tags, output_dict=True)

accuracy = report_dict['accuracy']
macro_precision = report_dict['macro avg']['precision']
macro_recall = report_dict['macro avg']['recall']
macro_f1 = report_dict['macro avg']['f1-score']

# ---------------------------------------------------------
# GRAPH 1: Bar Chart for all metrics
# ---------------------------------------------------------
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, macro_precision, macro_recall, macro_f1]
colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='black')

# settings of Bar Chart
plt.ylim(0, 1.1)
plt.title('Overall Metrics (BERT)', fontsize=14)
plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding the exact number (e.g. 0.95) above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", 
            ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics_barchart_bert.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# CHART 2: Confusion Matrix (Full and Errors-Only)
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)

# --- 2a. FULL Confusion Matrix ---
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tags, yticklabels=tags)
plt.title('Full Confusion Matrix (BERT)', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_full_bert.png'), dpi=300)
plt.close()
print("Saved 'confusion_matrix_full_bert.png'")

# --- 2b. ERRORS-ONLY Confusion Matrix ---
error_indices = set()
for t, p in zip(y_true, y_pred):
    if t != p: 
        error_indices.add(t)
        error_indices.add(p)

error_indices = sorted(list(error_indices))

if len(error_indices) > 0:
    sub_cm = cm[np.ix_(error_indices, error_indices)]
    sub_tags = [tags[i] for i in error_indices]

    plt.figure(figsize=(10, 8)) 
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=sub_tags, yticklabels=sub_tags)

    plt.title('Confusion Matrix (Focus on Errors - BERT)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_errors_bert.png'), dpi=300)
    plt.close()
    print("Saved 'confusion_matrix_errors_bert.png'")
else:
    print("No errors found! 'confusion_matrix_errors_bert.png' was not created. (BERT scored 100% on training data!)")

print(f"\nSUCCESS! All evaluation files were saved in: {RESULTS_DIR}")
print("1. metrics_report_bert.txt (Detailed classification report)")
print("2. metrics_barchart_bert.png (Bar chart of overall metrics)")
print("3. confusion_matrix_full_bert.png (Heatmap of the full confusion matrix)")
print("4. confusion_matrix_errors_bert.png (Heatmap focusing only on misclassifications)")