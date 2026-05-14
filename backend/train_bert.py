import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ΝΕΟ: Εισαγωγή της βιβλιοθήκης για το BERT
from sentence_transformers import SentenceTransformer

# Εισαγωγή του ίδιου Νευρωνικού Δικτύου!
from model import NeuralNet

# 1. Φόρτωση του προ-εκπαιδευμένου BERT (all-MiniLM-L6-v2)
# Την πρώτη φορά που θα τρέξει, θα κατεβάσει περίπου 80MB από το ίντερνετ
print("Φόρτωση του BERT μοντέλου... (Μπορεί να πάρει λίγα δευτερόλεπτα)")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Το BERT φορτώθηκε επιτυχώς!")

# 2. Φόρτωση των Intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

tags = []
xy = []

# Μαζεύουμε τα tags και τα patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        xy.append((pattern, tag))

# Αφαιρούμε τα διπλότυπα tags και τα κάνουμε αλφαβητική λίστα
tags = sorted(set(tags))

# 3. Προετοιμασία Δεδομένων (Embeddings)
print("Μετατροπή των προτάσεων σε διανύσματα (Embeddings)...")
X_train = []
y_train = []

for (pattern, tag) in xy:
    # ΕΔΩ ΓΙΝΕΤΑΙ Η ΜΑΓΕΙΑ: Το BERT μετατρέπει το κείμενο σε 384 αριθμούς (διάνυσμα)
    embedding = bert_model.encode(pattern)
    X_train.append(embedding)
    
    # Βρίσκουμε τον αριθμό του tag (π.χ. το 'greeting' είναι το 0, το 'cnn' είναι το 5)
    label = tags.index(tag)
    y_train.append(label)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# 4. Δημιουργία PyTorch Dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# 5. Hyperparameters (Παράμετροι Εκπαίδευσης)
batch_size = 8
hidden_size = 128
output_size = len(tags)
# Το input_size είναι ΠΑΝΤΑ 384 για το all-MiniLM-L6-v2!
input_size = 384 
learning_rate = 0.001
num_epochs = 200 # Το BERT μαθαίνει γρήγορα, ίσως χρειαστούν λιγότερες εποχές

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Αρχικοποίηση του μοντέλου (ίδιο με το BoW!)
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 6. Εκπαίδευση του Νευρωνικού Δικτύου
print("Ξεκινάει η εκπαίδευση του Classifier πάνω στα διανύσματα του BERT...")
all_losses = []

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

print(f'Τελικό Loss: {loss.item():.4f}')

# 7. Αποθήκευση Γραφήματος Loss
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "evaluation_results")
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

# 8. Αποθήκευση του Μοντέλου
# Προσοχή: Δεν αποθηκεύουμε το "all_words" γιατί πλέον δεν υπάρχει λεξιλόγιο!
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