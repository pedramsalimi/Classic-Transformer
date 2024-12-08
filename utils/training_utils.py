# utils/training_utils.py
import torch

def generate_subsequent_mask(size: int):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    return mask == 0  # True where we can attend

