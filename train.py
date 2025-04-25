import torch
import torch.nn as nn
from transformers import AutoTokenizer
from data import WMTDataset
from config import TransformerConfig
from torch.utils.data import DataLoader
from model import build_transformer
from utils import generate_causal_mask
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = TransformerConfig()

# Initialize tokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
special_token_dict = {"bos_token": "<s>"}
tokenizer.add_special_tokens(special_token_dict)

# Initialize dataset
train_dataset = WMTDataset(data_path=config.train_data_path, src_tokenizer=tokenizer, tgt_tokenizer=tokenizer, seq_len=config.seq_len)
val_dataset = WMTDataset(data_path=config.val_data_path, src_tokenizer=tokenizer, tgt_tokenizer=tokenizer, seq_len=config.seq_len)

# Initialize dataloader

train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=config.shuffle)
val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)

# Initialize model

model = build_transformer(config)

# Initialize loss

criteria = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token), label_smoothing=config.label_smoothing)

# Initialize optimizer

optimizer = torch.optim.Adam(model.parameters(), betas=config.betas, eps=config.optim_eps)

# Move model to device
model.to(device)
model.train()  # Set to training mode

for epoch in range(config.num_epochs):
    print(f"Epoch {epoch + 1}/{config.num_epochs}")
    epoch_loss = 0.0
    num_batches = len(train_dataloader)

    progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        src, tgt, label, src_mask, tgt_mask = batch
        src, tgt, label, src_mask, tgt_mask = src.to(device), tgt.to(device), label.to(device), src_mask.to(device), tgt_mask.to(device)
        enc_self_attn_mask = src_mask.unsqueeze(2) | src_mask.unsqueeze(3)
        
        causal_mask = generate_causal_mask(config.seq_len).to(device)
        
        dec_self_attn_mask = tgt_mask.unsqueeze(2) | tgt_mask.unsqueeze(3) | causal_mask
        
        dec_cross_attn_mask = tgt_mask.unsqueeze(3) | src_mask.unsqueeze(2)

        # Forward pass
        logits = model(src, tgt, enc_self_attn_mask, dec_self_attn_mask, dec_cross_attn_mask)  # (B, T_tgt, vocab_size)

        # Reshape logits and target for loss
        logits = logits.view(-1, logits.size(-1))                # (B*T_tgt, vocab_size)
        label = label.view(-1)                         # (B*T_tgt)

        # Compute loss
        loss = criteria(logits, label)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
