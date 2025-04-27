class Config:
    def __init__(self):
        self.tokenizer = "facebook/wmt19-en-de"
        self.src_vocab_size = 42025
        self.tgt_vocab_size = 42025
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
        self.save_after_steps = 1000
