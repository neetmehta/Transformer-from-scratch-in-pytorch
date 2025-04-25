import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import os

from config import TransformerConfig, OverfitConfig
from data import WMTDataset
from model import build_transformer
from utils import generate_causal_mask

def get_dataloaders(config, tokenizer):
    train_dataset = WMTDataset(
        data_path=config.train_data_path,
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        seq_len=config.seq_len
    )
    val_dataset = WMTDataset(
        data_path=config.val_data_path,
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        seq_len=config.seq_len
    )

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=config.shuffle)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)

    return train_loader, val_loader

def train_one_epoch(model, dataloader, optimizer, criterion, config, device):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        src, tgt, label, src_mask, tgt_mask = [x.to(device) for x in batch]

        enc_self_attn_mask = src_mask.unsqueeze(2) | src_mask.unsqueeze(3)
        causal_mask = generate_causal_mask(config.seq_len).to(device)
        dec_self_attn_mask = tgt_mask.unsqueeze(2) | tgt_mask.unsqueeze(3) | causal_mask
        dec_cross_attn_mask = tgt_mask.unsqueeze(3) | src_mask.unsqueeze(2)

        logits = model(src, tgt, enc_self_attn_mask, dec_self_attn_mask, dec_cross_attn_mask)
        logits = logits.view(-1, logits.size(-1))
        label = label.view(-1)

        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / num_batches

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)

def load_checkpoint(model, optimizer, path):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss:.4f}")
        return start_epoch, best_loss
    return 0, float('inf')

def main():
    config = OverfitConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    tokenizer.add_special_tokens({"bos_token": "<s>"})

    train_loader, val_loader = get_dataloaders(config, tokenizer)

    model = build_transformer(config)
    model.to(device)

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), betas=config.betas, eps=config.optim_eps, lr=10**-4)

    if config.resume:
        start_epoch, best_loss = load_checkpoint(model, optimizer, config.checkpoint_path)
    else:
        start_epoch, best_loss = 0, float('inf')

    for epoch in range(start_epoch, config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, config, device)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            print(f"Saving checkpoint... (loss improved from {best_loss:.4f} to {avg_loss:.4f})")
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, config.checkpoint_path)
            best_loss = avg_loss

if __name__ == "__main__":
    main()
