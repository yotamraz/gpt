import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class FeedForward(nn.Module):
    def __init__(self, n_embedding, dropout=0.2, growth_factor=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, growth_factor * n_embedding),
            nn.ReLU(),
            nn.Linear(growth_factor * n_embedding, n_embedding),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class OneHeadSelfAttention(nn.Module):
    """
    one head self-attention
    """

    def __init__(self, head_size, n_embedding, context_length, dropout=0.2):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.shape

        k = self.key(x)  # (b,t,c)
        q = self.query(x)  # (b,t,x)

        # compute the affinities (attention scores)
        wei = q @ k.transpose(-2, -1) * c ** -0.5  # (b, t, c) @ (b, c, t) -> (b, t, t)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))  # (b, t, t)
        wei = F.softmax(wei, dim=-1)  # (b, t, t)
        wei = self.dropout(wei)

        v = self.value(x)  # (b, t, c)

        out = wei @ v  # (b, t, t) @ (b, t, c) -> (b, t, c)

        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embedding, context_length, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList(
            [OneHeadSelfAttention(head_size, n_embedding, context_length) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, num_heads, n_embedding, context_length, dropout):
        super().__init__()
        head_size = n_embedding // num_heads
        self.sa = MultiHeadSelfAttention(num_heads, head_size, n_embedding, context_length, dropout)
        self.ff = FeedForward(n_embedding, dropout)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        """
        introducing residual connections
        :param x:
        :return:
        """
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ff(x)
        return x


class NaiveLanguageModel(nn.Module):
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

class GPTLikeLanguageModel(NaiveLanguageModel):
    def __init__(self, vocab_size, n_embedding, context_length, num_heads, num_layers, dropout, device):
        super().__init__(vocab_size)
        self.context_length = context_length
        # each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(context_length, n_embedding)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(num_heads, n_embedding, context_length, dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(n_embedding)
        self.lm_head = nn.Linear(n_embedding, vocab_size)
        self.device = device

    def forward(self, idx, targets=None):
        # get shape of sequence

        b, t = idx.shape
        # idx and targets are both (b, t) tensor of type int
        tok_emb = self.token_embedding_table(idx)  # (batch, time, channels)
        pos_emb = self.position_embedding_table(torch.arange(t, device=self.device))
        x = tok_emb + pos_emb
        x = self.transformer_blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)  # (batch, time, vocab_size)

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
            idx_cond = idx[:, -self.context_length:]
            # get predictions
            logits, loss = self(idx_cond)
            # take only the last time step prediction
            logits = logits[:, -1, :]
            # calculate the probabilities
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)  # (b, t+1)

        return idx
