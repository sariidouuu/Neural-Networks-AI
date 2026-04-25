import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Set the device ( GPU -if exists- or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup Output Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")

# Creates the folder if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# 1. Loading data (intents) and  model
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# 1. Make sure the code reads the correct file the train.py produced
FILE = "model.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# 2. Setting the model in evaluation mode (eval)
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Lists to keep the real and predicted values
y_true = []
y_pred = []

# 3. load all intents from the model to see what it predictes
print("Evaluation of the model. Please wait...")
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        X = bag_of_words(w, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        y_true.append(tags.index(tag))
        y_pred.append(predicted.item())

# 4. Creating Text Report and extract metrics
print("\n" + "="*80)
print(" METRICS ")
print("="*80)

# Prnts in terminal and store in txt
report_text = classification_report(y_true, y_pred, target_names=tags)
print(report_text)
report_path = os.path.join(RESULTS_DIR, "metrics_report.txt")
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
colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336'] # Πράσινο, Μπλε, Κίτρινο, Κόκκινο

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='black')

# settings of Bar Chart
plt.ylim(0, 1.1) # The limit goes up to 1.1 to fit the numbers above the bars.
plt.title('Overall Metrics', fontsize=14)
plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding the exact number (e.g. 0.95) above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", 
            ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics_barchart.png'), dpi=300)
plt.close() # We close the graph so that it does not get confused with the next one

# ---------------------------------------------------------
# CHART 2: Confusion Matrix (Full and Errors-Only)
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)

# --- 2a. FULL Confusion Matrix ---
plt.figure(figsize=(16, 14)) # Large size to accommodate many intents
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=tags, yticklabels=tags)

plt.title('Full Confusion Matrix (All Intents)', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_full.png'), dpi=300)
plt.close()
print("Saved 'confusion_matrix_full.png'")


# --- 2b. ERRORS-ONLY Confusion Matrix ---
# Find which intents had at least one misclassification
error_indices = set()
for t, p in zip(y_true, y_pred):
    if t != p: 
        error_indices.add(t)
        error_indices.add(p)

error_indices = sorted(list(error_indices))

if len(error_indices) > 0:
    # Slice the matrix to keep only rows and columns with errors
    sub_cm = cm[np.ix_(error_indices, error_indices)]
    sub_tags = [tags[i] for i in error_indices]

    plt.figure(figsize=(10, 8)) 
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=sub_tags, yticklabels=sub_tags)

    plt.title('Confusion Matrix (Focus on Errors)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_errors.png'), dpi=300)
    plt.close()
    print("Saved 'confusion_matrix_errors.png'")
else:
    print("No errors found! 'confusion_matrix_errors.png' was not created.")


print("\nSUCCESS! All evaluation files were saved in: {RESULTS_DIR}")
print("1. metrics_report.txt (Detailed classification report)")
print("2. metrics_barchart.png (Bar chart of overall metrics)")
print("3. confusion_matrix_full.png (Heatmap of the full confusion matrix)")
print("4. confusion_matrix_errors.png (Heatmap focusing only on misclassifications)")