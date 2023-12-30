import random

import torch
from torch.utils.data import DataLoader

from dataset import TextDataset
from models import BigramModel

##### hyperparameters #####
context_length = 8
batch_size = 32
learning_rate = 1e-3
iterations = 10000
device = "cuda" if torch.cuda.is_available() else "cpu"
###########################


# create encode/decode

train_dataset = TextDataset(file_path="./data/train.txt", context_length=context_length)
test_dataset = TextDataset(file_path="./data/test.txt", context_length=context_length)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# init model
model = BigramModel(vocab_size=train_dataset.vocabulary_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# train loop
for i, batch in enumerate(train_loader):
    # sample a batch
    xb, yb = batch
    xb, yb = xb.to(device), yb.to(device)

    # forward
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if i > iterations:
        break

    print(f"{i} / {iterations}")

idx = torch.zeros((1,1), dtype=torch.long)
print(train_dataset.decode(model.generate(idx, max_new_tokens=500)[0].tolist()))