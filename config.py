import torch

class TransformerConfig:
    def __init__(self):
        self.src_vocab_size = 58102
        self.tgt_vocab_size = 58102
        self.seq_len = 200
        self.d_model = 128
        self.num_heads = 4
        self.d_ff = 512
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.dropout = 0.1
        self.train_data_path = "E:\Transformer-from-scratch-in-pytorch\wmt14_translate_de-en_train.csv"
        self.val_data_path = "E:\Transformer-from-scratch-in-pytorch\wmt14_translate_de-en_validation.csv"
        self.train_batch_size = 16
        self.val_batch_size = 1
        self.shuffle = True
        self.label_smoothing = 0.1
        self.optim_eps = 1e-9
        self.betas = (0.9, 0.98)
        self.num_epochs = 100
        self.checkpoint_path = "./best_ckpt.pth"
        self.resume = False

class OverfitConfig:
    def __init__(self):
        self.src_vocab_size = 58102
        self.tgt_vocab_size = 58102
        self.seq_len = 200
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dropout = 0.1
        self.train_data_path = "E:\Transformer-from-scratch-in-pytorch\overfit.csv"
        self.val_data_path = "E:\Transformer-from-scratch-in-pytorch\overfit.csv"
        self.train_batch_size = 1
        self.val_batch_size = 1
        self.shuffle = False
        self.label_smoothing = 0.1
        self.optim_eps = 1e-9
        self.betas = (0.9, 0.98)
        self.num_epochs = 30
        self.checkpoint_path = "./overfit.pth"
        self.resume = False
