import torch

def generate_causal_mask(seq_len):
    return torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).unsqueeze(0).unsqueeze(1).to(bool)

def save_model(model, path):
    pass