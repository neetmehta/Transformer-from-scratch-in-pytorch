import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_causal_mask(seq_len):
    return torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).unsqueeze(0).unsqueeze(1).to(bool)

def save_model(model, path):
    pass

def greedy_decode(src_text, encoder_mask, model, tokenizer, config):
    
    memory = model.encode(src_text, encoder_mask)
    sos_token = tokenizer.encode(['<s>'])[0]
    eos_token = tokenizer.encode(['</s>'])[0]
    pad_token = tokenizer.encode(['<pad>'])[0]
    
    pred = torch.tensor([[tokenizer.encode(tokenizer.bos_token)[0]]])

    