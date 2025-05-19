class Config:
    def __init__(self):

        self.src_vocab_size = 37000
        self.tgt_vocab_size = 37000

        # Tokenizer
        self.tokenizer_type = "bpe"
        self.train_tokenizer = False
        self.tokenizer_path = "bpe.json"

        # Data
        self.dataset = "wmt14"
        self.language = "de-en"
        self.no_of_samples = 10000
        self.train_batch_size = 28
        self.val_batch_size = 1
        self.train_workers = 0
        self.val_workers = 0

        # Model
        self.max_seq_len = 500
        self.d_model = 256
        self.num_heads = 8
        self.d_ff = 2048
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dropout = 0.0

        # Optimizer and loss
        self.label_smoothing = 0.1
        self.optim_eps = 1e-9
        self.betas = (0.9, 0.98)
        self.lr = 1

        # Training
        self.num_epochs = 2
        self.checkpoint_path = "./best_ckpt_tiny.pth"
        self.resume = True
        self.save_after_steps = 1000
        self.warmup_steps = 4000
        self.gradient_accumulation_steps = 4
