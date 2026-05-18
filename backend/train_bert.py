import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from model import NeuralNet

# Load a pre-trained mini BERT model (all-MiniLM-L6-v2)
# First time it will run, it will download about 80MB from the internet
print("Loading BERT model... (May take a few seconds)")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("BERT was loaded succefully!")

# Load Intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

tags = []
xy = []

# Gather tags and patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        xy.append((pattern, tag))

# Remove duplicate tags and make them an alphabetical list
tags = sorted(set(tags))

# EMBEDDING DATA
print("Transform sentences into vectors (Embeddings)...")
X_train = []
y_train = []

for (pattern, tag) in xy:
    # BERT transform the text in 384 numbers (a vector)
    embedding = bert_model.encode(pattern)
    X_train.append(embedding)
    
    # Find the number of the tag ('greeting' is 0,'cnn' is 5)
    label = tags.index(tag)
    y_train.append(label)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Create a PyTorch Dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters (for training)
batch_size = 8
hidden_size = 128
output_size = len(tags)
# input_size is ALWAYS 384 for all-MiniLM-L6-v2
input_size = 384 
learning_rate = 0.001
num_epochs = 200 # BERT learns faster, maybe there will be needed less epochs

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model (same as BoW!)
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
print("Classifier training begins on BERT vectors...")
all_losses = []

# We define the limit that satisfies us
target_loss = 0.01

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    all_losses.append(loss.item())

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # EARLY STOPPING: if the error is less than 0.01 we stop
    if loss.item() < target_loss:
        print(f"\n⚠️ Early Stopping: Training stopped earlier in the epoch {epoch+1}!")
        print(f"The network reached the desired error ({loss.item():.4f}).")
        break # break to break the for loop

print(f'Τελικό Loss: {loss.item():.4f}')


# LOSS CURVE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "evaluation_results_bert")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

plt.figure(figsize=(8, 6))
plt.plot(all_losses, label='Training Loss (BERT-based)', color='red', linewidth=2)
plt.title('Neural Network Training Loss Curve (BERT)')
plt.xlabel('Epochs')
plt.ylabel('Loss (CrossEntropy)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULT_DIR, 'loss_curve_bert.png'), dpi=300)


# Save the model
# Attention: We do NOT save "all_words" cause there is NO dictionary
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "tags": tags
}

FILE = os.path.join(BASE_DIR, "model_bert.pth")
torch.save(data, FILE)

print(f'The training is copleted! The file was saved to: {FILE}')