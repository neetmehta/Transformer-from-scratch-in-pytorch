import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_causal_mask(seq_len):
    return torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).unsqueeze(0).unsqueeze(1).to(bool)

def save_model(model, path):
    pass

def greedy_decode(src, encoder_mask, model, tokenizer, config):
    
    model.eval()
    enc_self_attn_mask = encoder_mask.unsqueeze(2) | encoder_mask.unsqueeze(3)
    memory = model.encode(src, enc_self_attn_mask)
    sos_token = tokenizer.encode(['<s>'])[0]
    eos_token = tokenizer.encode(['</s>'])[0]

    pred = torch.tensor([[sos_token]], dtype=torch.long)

    for _ in range(config.seq_len):
        
        masked_self_attn_mask = generate_causal_mask(pred.shape[1])
        cross_attn_mask = torch.ones((1,pred.shape[1]), dtype=bool).unsqueeze(0).unsqueeze(3) | encoder_mask.unsqueeze(2)
        decoder_out = model.decode(pred, memory, masked_self_attn_mask, cross_attn_mask)
        
        logits = model.project(decoder_out)
        
        next_word = torch.argmax((logits[:, -1].softmax(dim=-1)))
        
        pred = torch.cat((pred, torch.tensor([[next_word]], dtype=torch.long)), dim=-1)
        
        if next_word == eos_token:
            break
        
    return tokenizer.decode(pred[0].detach().cpu().tolist())