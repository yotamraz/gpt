import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, path_to_data_folder, set_name='train', context_length=8):
        self.context_length = context_length
        self.path_to_data_folder = path_to_data_folder
        self.set_name = set_name

        train_path = os.path.join(path_to_data_folder, f'{set_name}.txt')
        test_path = os.path.join(path_to_data_folder, f'{set_name}.txt')

        # we have to make sure the encoder/decoder are aware for the entire dataset
        with open(train_path, 'r', encoding='utf-8') as f:
            self.train_data = f.read()
        with open(test_path, 'r', encoding='utf-8') as f:
            self.test_data = f.read()
        self.combined_data = self.train_data + self.test_data
        self.unique_chars = sorted(list(set(self.combined_data)))
        self.vocabulary_size = len(self.unique_chars)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder_func = {ch: i for i, ch in enumerate(self.unique_chars)}
        self.decoder_func = {i: ch for i, ch in enumerate(self.unique_chars)}
        self.encode = lambda s: [self.encoder_func[c] for c in s]
        self.decode = lambda c: ''.join([self.decoder_func[i] for i in c])

        # encode only the relevant set
        if set_name == 'train':
            self.encoded_data = self.encode(self.train_data)
        else:
            self.encoded_data = self.encode(self.test_data)

    def __len__(self):
        return len(self.encoded_data) - self.context_length

    def __getitem__(self, idx):
        x = np.asarray(self.encoded_data[idx:idx + self.context_length])
        y = np.asarray(self.encoded_data[idx + 1:idx + self.context_length + 1])
        return x, y
