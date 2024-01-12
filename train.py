import torch
from torch.utils.data import DataLoader

from dataset import TextDataset
from models import GPTLikeLanguageModel
from utils import EarlyStopping

##### hyperparameters #####
data_path = './data/j_and_silent_bob'
context_length = 256
batch_size = 64
learning_rate = 3e-4
max_iters = 5000
eval_iters = 200
eval_interval = 500
n_embedding = 384
num_heads = 6
num_layers = 6
dropout = 0.2
patience = 5
delta = 0.05
device = "cuda" if torch.cuda.is_available() else "cpu"
###########################

# # create data loaders
# train_dataset = TextDataset(path_to_data_folder="./data/", set_name='train', context_length=context_length)
# test_dataset = TextDataset(path_to_data_folder="./data/", set_name='test', context_length=context_length)
# train_loader = DataLoader(train_dataset, num_workers=8, shuffle=True, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, num_workers=8, shuffle=True, batch_size=batch_size)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in ix])
    y = torch.stack([data[i + 1:i + context_length + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# init model
model = GPTLikeLanguageModel(vocab_size=vocab_size, n_embedding=n_embedding,
                             context_length=context_length, num_heads=num_heads, num_layers=num_layers, dropout=dropout,
                             device=device).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
es = EarlyStopping(tolerance=patience, min_delta=delta)

# train loop
for iter in range(max_iters):
    # sample a batch
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    # forward
    _, loss = model(xb, yb)

    # backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # optimize
    optimizer.step()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"{iter} / {max_iters}, {losses}")

        es(train_loss=losses['train'], validation_loss=losses['val'])
        if es.early_stop:
            print("overfitting detected, stopping training...")
            break

context = torch.zeros((1, 1), dtype=torch.long, device=device).view(-1, 1)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
