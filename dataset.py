import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, file_path, context_length=8):
        self.context_length = context_length
        self.file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()
        self.unique_chars = sorted(list(set(self.data)))
        self.vocabulary_size = len(self.unique_chars)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder_func = {ch: i for i, ch in enumerate(self.unique_chars)}
        self.decoder_func = {i: ch for i, ch in enumerate(self.unique_chars)}
        self.encode = lambda s: [self.encoder_func[c] for c in s]
        self.decode = lambda c: ''.join([self.decoder_func[i] for i in c])
        self.encoded_data = self.encode(self.data)

    def __len__(self):
        return len(self.encoded_data) - self.context_length

    def __getitem__(self, idx):
        x = np.asarray(self.encoded_data[idx:idx + self.context_length])
        y = np.asarray(self.encoded_data[idx + 1:idx + self.context_length + 1])
        return x, y
