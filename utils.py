import torch
import torchmetrics
import os
import importlib.util
import sys

import torchmetrics.text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(model, optimizer, path):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss:.4f}")
        return start_epoch, best_loss
    return 0, float("inf")


def generate_causal_mask(seq_len):
    return (
        torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(1)
        .to(bool)
    )


def greedy_decode(src, src_mask, model, tokenizer, config, device):

    model.eval()
    assert src.shape[0] == 1, "batch size 1 is only supported"
    enc_self_attn_mask = src_mask.unsqueeze(2) | src_mask.unsqueeze(3)
    memory = model.encode(src, enc_self_attn_mask)
    sos_token = tokenizer.encode(["<s>"])[0]
    eos_token = tokenizer.encode(["</s>"])[0]

    pred = torch.tensor([[sos_token]], dtype=torch.long, device=device)

    for _ in range(config.seq_len):

        masked_self_attn_mask = generate_causal_mask(pred.shape[1]).to(device)
        cross_attn_mask = torch.ones(
            (1, pred.shape[1]), dtype=bool, device=device
        ).unsqueeze(0).unsqueeze(3) | src_mask.unsqueeze(2)
        decoder_out = model.decode(pred, memory, masked_self_attn_mask, cross_attn_mask)

        logits = model.project(decoder_out)

        next_word = torch.argmax((logits[:, -1].softmax(dim=-1)))

        pred = torch.cat(
            (pred, torch.tensor([[next_word]], dtype=torch.long, device=device)), dim=-1
        )

        if next_word == eos_token:
            break

    return logits, tokenizer.decode(pred[0, 1:-1].detach().cpu().tolist())


def calculate_bleu_score(label, pred, metric_name):
    if metric_name == "bleu_score":
        metrics = torchmetrics.text.BLEUScore()
        score = metrics([pred], [[label]])

    else:
        raise ("unknown metrics")
    return score.item()
