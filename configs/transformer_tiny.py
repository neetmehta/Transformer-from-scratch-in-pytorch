class Config:
    def __init__(self):
        self.tokenizer = "facebook/wmt19-en-de"
        self.src_vocab_size = 42025
        self.tgt_vocab_size = 42025
        self.seq_len = 400
        self.d_model = 256
        self.num_heads = 8
        self.d_ff = 2048
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
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
        self.checkpoint_path = "./best_ckpt_tiny.pth"
        self.resume = False
        self.save_after_steps = 1000
        self.train_workers = 0
        self.val_workers = 0
