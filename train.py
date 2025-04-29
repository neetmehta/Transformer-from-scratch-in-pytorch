import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

from data import WMTDataset
from model import build_transformer
from utils import (
    generate_causal_mask,
    save_checkpoint,
    load_checkpoint,
    calculate_bleu_score,
    greedy_decode,
    load_config,
)


def get_dataloaders(config, tokenizer):
    train_dataset = WMTDataset(
        data_path=config.train_data_path,
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        seq_len=config.seq_len,
    )
    val_dataset = WMTDataset(
        data_path=config.val_data_path,
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        seq_len=config.seq_len,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=config.shuffle, num_workers=config.train_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=config.val_workers
    )

    return train_loader, val_loader


def train_one_epoch(model, dataloader, optimizer, criterion, tokenizer, config, epoch, device):
    model.train()
    total_loss = 0.0
    steps = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        src, tgt, label, src_mask, tgt_mask = [x.to(device) for x in batch]

        enc_self_attn_mask = src_mask.unsqueeze(2) | src_mask.unsqueeze(3)
        causal_mask = generate_causal_mask(config.seq_len).to(device)
        dec_self_attn_mask = tgt_mask.unsqueeze(2) | tgt_mask.unsqueeze(3) | causal_mask
        dec_cross_attn_mask = tgt_mask.unsqueeze(3) | src_mask.unsqueeze(2)

        logits = model(
            src, tgt, enc_self_attn_mask, dec_self_attn_mask, dec_cross_attn_mask
        )
        logits = logits.view(-1, logits.size(-1))
        
        label = label.view(-1)

        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred_sentence = torch.argmax(logits.softmax(dim=-1), dim=-1)
        if len(torch.where(pred_sentence==2)[0]) > 0:
            stop_index = torch.where(pred_sentence==2)[0][0]
        else:
            stop_index = pred_sentence.shape[0]
            
        if len(torch.where(label==2)[0]) > 0:
            stop_label_index = torch.where(label==2)[0][0]
        else:
            stop_label_index = label.shape[0]
        print("Pred sentence: \n", tokenizer.decode(pred_sentence[:stop_index].tolist()), "\n")
        print("Target sentence: \n", tokenizer.decode(label[:stop_label_index].tolist()), "\n")

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        steps += 1

        if steps % config.save_after_steps == 0:
            print(f"Saving checkpoint... \n")
            save_checkpoint(
                model, optimizer, epoch + 1, total_loss / steps, config.checkpoint_path
            )

    return total_loss / num_batches


def validate(model, dataloader, criterion, config, tokenizer, device):
    model.eval()
    total_bleu_score = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    for batch in progress_bar:
        src, _, label, src_mask, _ = [x.to(device) for x in batch]
        logits, prediction = greedy_decode(
            src, src_mask, model, tokenizer, config, device
        )
        pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        eos_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        # TODO: Remove hardcoding for eos
        raw_label = label[(label != pad_token) & (label != 2)]
        raw_label = tokenizer.decode(raw_label.tolist())
        bleu_score = calculate_bleu_score(raw_label, prediction, "bleu_score")
        total_bleu_score += bleu_score
        progress_bar.set_postfix(loss=bleu_score)

    return total_bleu_score / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    config_module = load_config(args.config)
    config = config_module.Config()  # instantiate your Config class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    tokenizer.add_special_tokens({"bos_token": "<s>"})

    train_loader, val_loader = get_dataloaders(config, tokenizer)

    model = build_transformer(config)
    model.to(device)

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_token_id, label_smoothing=config.label_smoothing
    )
    optimizer = torch.optim.Adam(
        model.parameters(), betas=config.betas, eps=config.optim_eps, lr=10**-4
    )

    print(f"Number of parameters = {sum(i.numel() for i in model.parameters())/1e6}M\n")
    print("Starting training...")

    if config.resume:
        start_epoch, best_bleu_score = load_checkpoint(
            model, optimizer, config.checkpoint_path
        )
    else:
        start_epoch, best_bleu_score = 0, 0.0

    for epoch in range(start_epoch, config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, tokenizer, config, epoch, device
        )
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}\n")
        print(f"Running Validation...")
        avg_bleu_score = validate(
            model, val_loader, criterion, config, tokenizer, device
        )
        print(f"Avg. BLEU score is: {avg_bleu_score}\n")
        if True:
            print(
                f"Saving checkpoint... (BLEU score improved from {best_bleu_score:.4f} to {avg_bleu_score:.4f})\n"
            )
            save_checkpoint(
                model, optimizer, epoch + 1, avg_loss, config.checkpoint_path
            )
            best_bleu_score = avg_bleu_score


if __name__ == "__main__":
    main()
