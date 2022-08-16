import json
import numpy as np

from model import Net
from nltk_utils import bag_of_words, tokenize, stem


# Import pytorch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Open json File
with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # Instance
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    # Label
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
BATCH_SIZE = 8
INPUT_SIZE = len(X_train[0])
HIDDEN_SIZE = 8
OUTPUT_SIZE = len(tags)
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000

# Prepare dataset
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and Optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f" Epoch {epoch+1}/{NUM_EPOCHS}, Loss : {loss.item():.4f}")

print(f" Final Loss, Loss : {loss.item():.4f}")

# Save model output
data = {
    "model_state":model.state_dict(),
    "input_size":INPUT_SIZE,
    "output_size":OUTPUT_SIZE,
    "hidden_size":HIDDEN_SIZE,
    "all_words":all_words,
    "tags":tags
}

FILE = "chat_data.pth"
torch.save(data, FILE)

print(f"Training Completed. \nFile Saved to {FILE}")


