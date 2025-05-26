# Transformer-from-scratch-in-pytorch

This project provides a clean, modular and from scratch PyTorch implementation of the Transformer architecture, as introduced in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). It is designed for sequence-to-sequence tasks such as machine translation and includes all core components: multi-head self-attention, positional encoding, encoder and decoder stacks, masking, and utilities for training and evaluation. The codebase is intended for both educational and research purposes, making it easy to understand, extend, and experiment with the Transformer model.

## Project Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Transformer-from-scratch-in-pytorch.git
cd Transformer-from-scratch-in-pytorch
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Change Config**

```python
class Config:
    def __init__(self):

        self.vocab_size = 60_000
        self.src_vocab_size = 60_000
        self.tgt_vocab_size = 60_000

        # Tokenizer
        self.tokenizer_type = "bpe"
        self.train_tokenizer = False
        self.tokenizer_path = "bpe.json"

        # Data
        self.dataset = "wmt19"
        self.language = "de-en"
        self.train_samples = -1
        self.val_samples = -1
        self.train_batch_size = 128
        self.val_batch_size = 1
        self.workers = 12
        self.pin_memory = True

        # Model
        self.max_seq_len = 100
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dropout = 0.1
        self.weight_tying = True

        # Optimizer and loss
        self.label_smoothing = 0.1
        self.optim_eps = 1e-9
        self.betas = (0.9, 0.98)
        self.lr = 0.1

        # Training
        self.num_epochs = 3
        self.checkpoint_path = "./best_ckpt_base.pth"
        self.resume = True
        self.save_after_steps = 1000
        self.warmup_steps = 4000
        self.gradient_accumulation_steps = 4
        self.overfit = False
```
Notes:

- The bpe.json provided here is Byte-Pair Encoding tokenizer with 60_000 vocabulary size. If you want to train you own tokenizer, make self.train_tokenizer as True.

- This repo only supports training on `wmtYY` datasets. Particularly `wmt14`,`wmt15`,`wmt16`,`wmt17`,`wmt18`,`wmt19`.

4. **Train**
```bash
python3 train.py --config configs/transformer_base.py
```

5. **Infer**
```bash
python3 infer.py --config configs/transformer_base.py
```

## References & Credits

This project is inspired by and based on the following resources:

- Vaswani et al., Attention Is All You Need, https://arxiv.org/pdf/1706.03762

- Brando Koch’s annotated implementation: https://github.com/brandokoch/attention-is-all-you-need-paper

- Minimal PyTorch implementation by hkproj: https://github.com/hkproj/pytorch-transformer

