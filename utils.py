from typing import List, Dict

import torch
import torch.nn as nn


@torch.no_grad()
def estimate_loss(model: nn.Module, data_loaders: List[torch.utils.data.DataLoader], device: str) -> Dict[str, float]:
    out = {}
    model.eval()
    for loader in data_loaders:
        losses = torch.zeros(len(loader))
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[i] += loss.item()
        set_name = loader.dataset.file_path.split('/')[-1].split('.')[0]
        out[set_name] = losses.mean().item()
    model.train()
    return out

