import os
import numpy as np
import random
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Opens the intents.json file and readstags and patterns
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# initialization
all_words = []
tags = []
xy = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence 
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = [

    # Punctuation
    '?', '.', '!', ',', ':', ';', '-', '(', ')', '"', "'",

    # Common question words — appear in nearly every pattern
    'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose',

    # Common verbs used in questions
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'do', 'does', 'did', 'have', 'has', 'had',
    'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might',

    # Common articles and prepositions
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for',
    'with', 'about', 'by', 'from', 'into', 'between', 'and', 'or',

    # Common pronouns
    'i', 'you', 'we', 'it', 'this', 'that', 'there', 'they',

    # Common adverbs
    'me', 'my', 'your', 'its', 'their',
    'some', 'any', 'more', 'most', 'so', 'very', 'just', 'also',

    # Common filler verbs
    'tell', 'explain', 'describe', 'give', 'show', 'define', 'mean',
    'get', 'use', 'make', 'know', 'understand', 'learn', 'work'
]



all_words = [stem(w) for w in all_words if w not in ignore_words] # stemming
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"{len(xy)} patterns")
print(f"{len(tags)} tags: {tags}")
print(f"{len(all_words)} unique stemmed words: ", all_words)

# create training data
X_train = [] # questions - prompt
y_train = [] # answer

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words) # Converts every word in 0 and 1 based on the dictionary (all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag) # Converts every tag in 0 and 1
    y_train.append(label)

X_train = np.array(X_train) 
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 600
batch_size = 32
learning_rate = 0.001 # The 'step' with which the optimizer tries to reduce the error
input_size = len(X_train[0])
hidden_size = 32 # the number of neurons on each hidden layer
output_size = len(tags)

print(f"input_size: {input_size}, output_size: {output_size}")

class ChatDataset(Dataset): # A PyTorch class that organizes data (x_train - questions and y_train - answers)

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
# shuffles tha data and separates them into batche
train_loader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

# This line of code is very important for the velocity of the training, as it determines 'where' the neural network's mathematical computation will take place
# CUDA is a technology developed by NVIDIA that allows your computer's GPU to execute/perform mathematical computations much quicke/fasterr than CPU
# Neural Networks love GPUs because they can perform thousands of operations simultaneously (ταυτόχρονα)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Creates a neural network object
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # changes the values of parameters

# Creat an empty list to download the loss
all_losses = [] 

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass - a prediction the model does
        outputs = model(words)

        # calculates the difference between prediction and reality
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad() # sets to 0 the previous errors/gradients
        loss.backward() # calculates how the weights should change (backpropagation)
        optimizer.step() # applies the changes 

    # Αποθηκεύουμε την τιμή του loss στο τέλος κάθε epoch
    all_losses.append(loss.item())

    # prints in every 100 epochs    
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

# ---------------------------------------------------------
# Creating and Saving the Loss Curve
# ---------------------------------------------------------

# Setup Output Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "evaluation_results")
#if the file does not yet exists
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

plt.figure(figsize=(8, 6))
plt.plot(all_losses, label='Training Loss', color='blue', linewidth=2)
plt.title('Neural Network Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss (CrossEntropy)')
plt.legend()
plt.grid(True)
# Insteaf of plt.savefig('loss_curve.png')
plt.savefig(os.path.join(RESULT_DIR, 'loss_curve.png'), dpi=300)
print(f"Loss curve saved in: {RESULT_DIR}")
print("The chart was saved as 'loss_curve.png'")

# ---------------------------------------------------------

# A data package - a dictionary that includes the components for the chatbot to resurrect
data = {
    "model_state": model.state_dict(), # weights
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words, # the dictionary - vocabulary with every word the model recognizes
    "tags": tags # a list with every tag/intent we have on intents.json
}

FILE = "model.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')