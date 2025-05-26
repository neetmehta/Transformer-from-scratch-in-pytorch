import torch
import torchmetrics
import os
import importlib.util
import sys

import torchmetrics.text
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transformer_lr_scheduler(
    optimizer: Optimizer, d_model: int = 256, warmup_steps: int = 4000
):
    """Returns a learning rate scheduler with the schedule from the original Transformer paper.
    Args:
        optimizer (Optimizer): The optimizer to apply the schedule to.
        d_model (int): The model dimension.
        warmup_steps (int): Number of warmup steps.
    Returns:
        LambdaLR: Learning rate scheduler.
    """
    def lr_lambda(step: int):
        if step == 0:
            step = 1
        return (d_model**-0.5) * min(step**-0.5, step * (warmup_steps**-1.5))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def load_config(config_path):
    """Dynamically loads a Python config file as a module.
    Args:
        config_path (str): Path to the config .py file.
    Returns:
        module: The loaded config module.
    """
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Saves model and optimizer state to a checkpoint file.
    Args:
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer to save.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        path (str): Path to save the checkpoint.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(model, optimizer=None, path=""):
    """Loads model and optimizer state from a checkpoint file.
    Args:
        model (nn.Module): The model to load state into.
        optimizer (Optimizer, optional): The optimizer to load state into.
        path (str): Path to the checkpoint file.
    Returns:
        tuple: (start_epoch, best_loss)
    """
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss:.4f}")
        return start_epoch, best_loss
    return 0, float("inf")


def generate_causal_mask(seq_len):
    """Generates a causal mask for sequence-to-sequence models.
    Args:
        seq_len (int): Length of the sequence.
    Returns:
        torch.Tensor: Causal mask of shape (1, 1, seq_len, seq_len).
    """
    return (
        torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(1)
        .to(bool)
    )


def greedy_decode(src, model, tokenizer, config, device):
    """Performs greedy decoding for sequence generation.
    Args:
        src (torch.Tensor): Source sequence tensor of shape (1, src_seq_len).
        model (nn.Module): The machine translation model.
        tokenizer: Tokenizer with encode and decode methods.
        config: Configuration object with max_seq_len.
        device: Device to run the model on.
    Returns:
        tuple: (logits, decoded_sentence)
    """
    model.eval()
    assert src.shape[0] == 1, "batch size 1 is only supported"
    src_mask = src == 0
    src_mask = src_mask.unsqueeze(1).unsqueeze(2)
    enc_self_attn_mask = src_mask
    memory = model.encode(src, enc_self_attn_mask)
    sos_token = tokenizer.encode("[BOS]")[1]
    eos_token = tokenizer.encode("[EOS]")[1]

    pred = torch.tensor([[sos_token]], dtype=torch.long, device=device)

    for _ in range(config.max_seq_len):

        masked_self_attn_mask = generate_causal_mask(pred.shape[1]).to(device)

        decoder_out = model.decode(pred, memory, masked_self_attn_mask, src_mask)

        logits = model.project(decoder_out)

        next_word = torch.argmax((logits[:, -1].softmax(dim=-1)))

        pred = torch.cat(
            (pred, torch.tensor([[next_word]], dtype=torch.long, device=device)), dim=-1
        )

        if next_word == eos_token:
            break

    return logits, tokenizer.decode(pred[0, 1:-1].detach().cpu().tolist())


def calculate_bleu_score(label, pred, metric_name):
    """Calculates BLEU score for a predicted sentence.
    Args:
        label (str): Reference sentence.
        pred (str): Predicted sentence.
        metric_name (str): Name of the metric ("bleu_score").
    Returns:
        float: BLEU score.
    """
    if metric_name == "bleu_score":
        metrics = torchmetrics.text.BLEUScore()
        score = metrics([pred], [[label]])

    else:
        raise ("unknown metrics")
    return score.item()