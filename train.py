import torch
from torch.utils.data import DataLoader

from dataset import TextDataset
from models import BigramModel, BigramModelWithSelfAtten
from utils import estimate_loss

##### hyperparameters #####
context_length = 64
batch_size = 64
learning_rate = 3e-4
eval_interval = 10
n_embedding = 128
num_heads = 4
num_layers = 4
dropout = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
###########################

# create data loaders
train_dataset = TextDataset(path_to_data_folder="./data/", set_name='train', context_length=context_length)
test_dataset = TextDataset(path_to_data_folder="./data/", set_name='test', context_length=context_length)
train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_dataset, num_workers=8, shuffle=False, batch_size=batch_size)

# init model
model = BigramModelWithSelfAtten(vocab_size=train_dataset.vocabulary_size, n_embedding=n_embedding,
                                 context_length=context_length, num_heads=num_heads, num_layers=num_layers, dropout=dropout, device=device).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# train loop
for i, batch in enumerate(train_loader):
    # sample a batch
    xb, yb = batch
    xb, yb = xb.to(device), yb.to(device)

    # forward
    _, loss = model(xb, yb)

    # backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # optimize
    optimizer.step()

    if i % eval_interval == 0:
        losses = estimate_loss(model, [train_loader, test_loader], device)
        print(f"{i} / {len(train_loader)}, {losses}")

idx = torch.tensor([64], device=device).view(-1, 1)
print(train_dataset.decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
