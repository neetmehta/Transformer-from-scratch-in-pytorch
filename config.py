# config.py

class TransformerConfig:
    def __init__(self):
        self.src_vocab_size = 58101
        self.tgt_vocab_size = 58101
        self.seq_len = 128
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dropout = 0.1
