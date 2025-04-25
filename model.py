import torch
import torch.nn as nn
import math

def generate_causal_mask(seq_len):
    return torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).unsqueeze(0).unsqueeze(1).to(bool)

class WordEmbeddings(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.word_embd = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens):

        x = self.word_embd(tokens)
        return x

class PositionalEmbedding(nn.Module):

    def __init__(self, seq_len, d_model):
        super().__init__()

        pe = torch.zeros(seq_len, d_model, requires_grad=False)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2)/d_model)
        pos = torch.arange(seq_len).unsqueeze(1)
        pe[:, 0::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        return x + self.pe[:, :x.shape[1], :]

class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.h = num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

        self.d_k = d_model // num_heads


        assert d_model % num_heads == 0, "d_model not divisible by num_heads"

    def forward(self, q, k, v, mask=None):

        B, seq_len, embd_dim = q.shape
        # q -> (B, seq_len, embd_dim)
        # k -> (B, seq_len, embd_dim)
        # v -> (B, seq_len, embd_dim)
        queries = self.w_q(q)
        keys = self.w_k(k)
        values = self.w_v(v)
        # q -> (B, seq_len, embd_dim)
        # k -> (B, seq_len, embd_dim)
        # v -> (B, seq_len, embd_dim)

        queries = queries.view(B, seq_len, self.h, self.d_k).transpose(1,2)
        keys = keys.view(B, seq_len, self.h, self.d_k).transpose(1,2)
        values = values.view(B, seq_len, self.h, self.d_k).transpose(1,2)

        # q -> (B, h, seq_len, dk)
        # k -> (B, h, seq_len, dk)
        # v -> (B, h, seq_len, dk)

        attention_weights = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # boolean mask
            attention_weights = attention_weights.masked_fill(mask, float('-inf'))

        attention_weights = attention_weights.softmax(dim=-1)

        # attention_weights -> (B, h, seq_len, seq_len)

        attention_output = attention_weights @ values

        # attention_weights -> (B, h, seq_len, dk)

        attention_output = attention_output.transpose(1, 2).reshape(B, seq_len, -1)

        attention_output = self.w_o(attention_output)
        return attention_output, attention_weights

class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model, d_ff, p_d=0.1):
        super().__init__()

        # p_d = 0.1 from original paper
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p_d)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):

        x = torch.relu(self.dropout(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, p_d=0.1, layer_norm_eps=1e-5, bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(p_d)
        self.dropout2 = nn.Dropout(p_d)
        self.multi_headed_self_attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, p_d=p_d)

    def forward(self, x, self_attn_mask):

        x = self.norm1(x + self.dropout1(self.multi_headed_self_attention(x, x, x, self_attn_mask)[0]))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x

class DecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, p_d=0.0, layer_norm_eps=1e-5, bias=True):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(p_d)
        self.dropout2 = nn.Dropout(p_d)
        self.dropout3 = nn.Dropout(p_d)
        self.masked_multi_headed_self_attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.multi_headed_cross_attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, p_d=p_d)

    def forward(self, x, memory, masked_self_attn_mask, cross_attn_mask):

        x = self.norm1(x + self.dropout1(self.masked_multi_headed_self_attention(x, x, x, masked_self_attn_mask)[0]))

        x = self.norm2(x + self.dropout2(self.multi_headed_cross_attention(x, memory, memory, cross_attn_mask)[0]))

        x = self.norm3(x + self.dropout3(self.ffn(x)))

        return x
    
class Encoder(nn.Module):
    def __init__(self, encoder_layers):
        super().__init__()
        self.encoder_layers = encoder_layers

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self, decoder_layers):
        super().__init__()
        self.decoder_layers = decoder_layers

    def forward(self, x, memory, masked_self_attn_mask, cross_attn_mask):
        for layer in self.decoder_layers:
            x = layer(x, memory, masked_self_attn_mask, cross_attn_mask)
        return x
        
    
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return self.projection(x)

class Transformer(nn.Module):

    def __init__(self, src_word_embedding, tgt_word_embedding, positional_encoding, encoder, decoder, projection_layer):
        super().__init__()
        self.src_word_embedding = src_word_embedding
        self.tgt_word_embedding = tgt_word_embedding
        self.positional_encoding = positional_encoding
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer
        
    def encode(self, src, self_attn_mask):
        src = self.src_word_embedding(src)
        
        src = self.positional_encoding(src)
        
        memory = self.encoder(src, self_attn_mask)
        
        return memory

    def decode(self, tgt, memory, masked_self_attn_mask, cross_attn_mask):
        tgt = self.tgt_word_embedding(tgt)

        tgt = self.positional_encoding(tgt)

        decoder_out = self.decoder(tgt, memory, masked_self_attn_mask, cross_attn_mask)

        return decoder_out
    
    def project(self, decoder_out):
        
        logits = self.projection_layer(decoder_out)

        return logits

    def forward(self, src, tgt, self_attn_mask, masked_self_attn_mask, cross_attn_mask):
        
        memory = self.encode(src, self_attn_mask)
        
        decoder_out = self.decode(tgt, memory, masked_self_attn_mask, cross_attn_mask)
        
        logits = self.project(decoder_out)
        
        return logits
    
def build_transformer(config):
    
    # Word Embeddings
    src_word_embedding = WordEmbeddings(config.src_vocab_size, config.d_model)
    tgt_word_embedding = WordEmbeddings(config.tgt_vocab_size, config.d_model)

    # Shared Positional Embedding
    positional_encoding = PositionalEmbedding(config.seq_len, config.d_model)

    # Stack of Encoder Layers
    encoder_layers = nn.ModuleList([
        EncoderBlock(config.d_model, config.num_heads, config.d_ff, p_d=config.dropout)
        for _ in range(config.num_encoder_layers)
    ])
    
    encoder = Encoder(encoder_layers)

    # Stack of Decoder Layers
    decoder_layers = nn.ModuleList([
        DecoderBlock(config.d_model, config.num_heads, config.d_ff, p_d=config.dropout)
        for _ in range(config.num_decoder_layers)
    ])

    decoder = Decoder(decoder_layers)

    # Output Projection
    projection_layer = ProjectionLayer(config.d_model, config.tgt_vocab_size)

    # Build Transformer
    model = Transformer(
        src_word_embedding,
        tgt_word_embedding,
        positional_encoding,
        encoder,
        decoder,
        projection_layer
    )

    return model
