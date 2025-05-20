import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from tokenizer import TokenizerWrapper, train_tokenizer

from data import get_dataloaders
from model import build_transformer
from utils import (
    generate_causal_mask,
    save_checkpoint,
    load_checkpoint,
    calculate_bleu_score,
    greedy_decode,
    load_config,
    get_transformer_lr_scheduler,
)


def train_one_epoch(
    model, dataloader, optimizer, criterion, scheduler, config, epoch, device
):
    model.train()
    total_loss = 0.0
    step_loss = 0.0
    steps = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        src, tgt, label = [x.to(device) for x in batch]

        # TODO: Left and right shift of tgt
        logits = model(src, tgt)
        logits = logits.view(-1, logits.size(-1))
        label = label.view(-1)

        loss = criterion(logits, label)

        total_loss += loss.item()
        step_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        loss = loss / config.gradient_accumulation_steps

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (steps + 1) % config.gradient_accumulation_steps == 0:

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        steps += 1

        if steps % config.save_after_steps == 0:
            print(f"Saving checkpoint... \n")
            print(f"Avg. Loss: {step_loss/config.save_after_steps}")

            print(f"LR: {scheduler.get_last_lr()[0]}")
            save_checkpoint(
                model, optimizer, epoch + 1, total_loss / steps, config.checkpoint_path
            )
            step_loss = 0.0
    return total_loss / num_batches


def validate(model, dataloader, criterion, config, tokenizer, device):
    model.eval()
    total_bleu_score = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    for batch in progress_bar:
        src, _, label = [x.to(device) for x in batch]
        logits, prediction = greedy_decode(
            src, model, tokenizer, config, device
        )
        pad_token, eos_token = tokenizer.encode(tokenizer.tokenizer.pad_token)[1], tokenizer.encode(tokenizer.tokenizer.pad_token)[2]

        # TODO: Remove hardcoding for eos
        raw_label = label[(label != pad_token) & (label != eos_token)]
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

    if config.train_tokenizer:
        tokenizer = train_tokenizer(config)
    else:
        tokenizer = TokenizerWrapper(config)

    train_loader, val_loader = get_dataloaders(config, tokenizer)

    model = build_transformer(config)
    model.to(device)

    pad_token_id = 0  # TODO: Remove hardcoding
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_token_id, label_smoothing=config.label_smoothing
    )
    optimizer = torch.optim.Adam(
        model.parameters(), betas=config.betas, eps=config.optim_eps, lr=config.lr
    )
    scheduler = get_transformer_lr_scheduler(
        optimizer, d_model=config.d_model, warmup_steps=config.warmup_steps
    )

    print(f"Number of parameters = {sum(i.numel() for i in model.parameters())/1e6}M")
    print(f"Training data size: {len(train_loader.dataset)}")
    print(f"Validation data size: {len(val_loader.dataset)}\n")
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
            model, train_loader, optimizer, criterion, scheduler, config, epoch, device
        )
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}\n")
        print(f"Running Validation...")
        avg_bleu_score = validate(
            model, val_loader, criterion, config, tokenizer, device
        )
        print(f"Avg. BLEU score is: {avg_bleu_score}\n")
        if avg_bleu_score > best_bleu_score:
            print(
                f"Saving checkpoint... (BLEU score improved from {best_bleu_score:.4f} to {avg_bleu_score:.4f})\n"
            )
            save_checkpoint(
                model, optimizer, epoch + 1, avg_loss, config.checkpoint_path
            )
            best_bleu_score = avg_bleu_score


if __name__ == "__main__":
    main()
