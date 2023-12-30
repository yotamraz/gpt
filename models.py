import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (b, t) tensor of type int
        logits = self.token_embedding_table(idx)  # (batch, time, channels)
        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (b, t) array of indices in current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # take only the last time step prediction
            logits = logits[:, -1, :]
            # calculate the probabilities
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)  # (b, t+1)

        return idx