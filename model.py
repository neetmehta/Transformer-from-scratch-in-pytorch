import torch
import torch.nn as nn
import math
from utils import generate_causal_mask


class WordEmbeddings(nn.Module):
    """Word Embedding layer for the Transformer model.
    This module creates an embedding layer for the input tokens, mapping them to a higher-dimensional space.
    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model (embedding size).
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.word_embd = nn.Embedding(vocab_size, d_model)  # Embedding layer
        self.d_model = d_model  # Dimension of the model

    def forward(self, tokens):

        x = self.word_embd(tokens)  # tokens -> (B, seq_len)
        return x  # x -> (B, seq_len, d_model)


class PositionalEmbedding(nn.Module):
    """Positional Encoding layer for the Transformer model.
    This module adds positional encodings to the input embeddings to provide information about the position of tokens in the sequence.
    Args:
        seq_len (int): Length of the input sequence.
        d_model (int): Dimension of the model (embedding size).
    """

    def __init__(self, seq_len, d_model):
        super().__init__()

        pe = torch.zeros(seq_len, d_model, requires_grad=False)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        pos = torch.arange(seq_len).unsqueeze(1)
        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

        # add positional encoding to the input tensor x
        return x + self.pe[:, : x.shape[1], :]  # x -> (B, seq_len, d_model)


class MultiheadAttention(nn.Module):
    """Multi-head Attention layer for the Transformer model.
    This module implements multi-head attention mechanism, allowing the model to focus on different parts of the input sequence.
    Args:
        d_model (int): Dimension of the model (embedding size).
        num_heads (int): Number of attention heads.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.h = num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.d_k = d_model // num_heads

        assert d_model % num_heads == 0, "d_model not divisible by num_heads"

    def forward(self, q, k, v, mask=None):

        B, seq_len, _ = q.shape
        # q -> (B, seq_len, embd_dim)
        # k -> (B, seq_len, embd_dim)
        # v -> (B, seq_len, embd_dim)
        queries = self.w_q(q)
        keys = self.w_k(k)
        values = self.w_v(v)
        # q -> (B, seq_len, embd_dim)
        # k -> (B, seq_len, embd_dim)
        # v -> (B, seq_len, embd_dim)

        queries = queries.view(
            queries.shape[0], queries.shape[1], self.h, self.d_k
        ).transpose(1, 2)
        keys = keys.view(keys.shape[0], keys.shape[1], self.h, self.d_k).transpose(1, 2)
        values = values.view(
            values.shape[0], values.shape[1], self.h, self.d_k
        ).transpose(1, 2)

        # q -> (B, h, seq_len, dk)
        # k -> (B, h, seq_len, dk)
        # v -> (B, h, seq_len, dk)

        attention_weights = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # boolean mask
            attention_weights = attention_weights.masked_fill(mask, -1e9)

        attention_weights = attention_weights.softmax(dim=-1)

        # attention_weights -> (B, h, seq_len, seq_len)

        attention_output = attention_weights @ values

        # attention_weights -> (B, h, seq_len, dk)

        attention_output = attention_output.transpose(1, 2).reshape(B, seq_len, -1)

        attention_output = self.w_o(attention_output)
        return attention_output, attention_weights


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network layer for the Transformer model.
    This module implements a feed-forward network with two linear layers and a ReLU activation in between.
    Args:
        d_model (int): Dimension of the model (embedding size).
        d_ff (int): Dimension of the feed-forward network.
        p_d (float): Dropout probability for the dropout layer.
        bias (bool): Whether to use bias in the linear layers.
    """

    def __init__(self, d_model, d_ff, p_d=0.1, bias=True):
        super().__init__()

        # p_d = 0.1 from original paper
        self.linear_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.dropout = nn.Dropout(p_d)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):

        x = torch.relu(self.dropout(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderBlock(nn.Module):
    """Single Encoder Block for the Transformer model.
    Consists of a multi-head self-attention layer followed by a feed-forward network,
    each with residual connections and layer normalization.
    Args:
        d_model (int): Dimension of the model (embedding size).
        num_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward network.
        p_d (float): Dropout probability.
        layer_norm_eps (float): Epsilon for layer normalization.
        bias (bool): Whether to use bias in the linear layers.
    """

    def __init__(
        self, d_model, num_heads, d_ff, p_d=0.1, layer_norm_eps=1e-5, bias=True
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(p_d)
        self.dropout2 = nn.Dropout(p_d)
        self.multi_headed_self_attention = MultiheadAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, p_d=p_d)

    def forward(self, x, self_attn_mask):

        x = self.norm1(
            x
            + self.dropout1(
                self.multi_headed_self_attention(x, x, x, self_attn_mask)[0]
            )
        )
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x


class DecoderBlock(nn.Module):
    """Single Decoder Block for the Transformer model.
    Consists of a masked multi-head self-attention layer, a multi-head cross-attention layer,
    and a feed-forward network, each with residual connections and layer normalization.
    Args:
        d_model (int): Dimension of the model (embedding size).
        num_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward network.
        p_d (float): Dropout probability.
        layer_norm_eps (float): Epsilon for layer normalization.
        bias (bool): Whether to use bias in the linear layers.
    """

    def __init__(
        self, d_model, num_heads, d_ff, p_d=0.0, layer_norm_eps=1e-5, bias=True
    ):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(p_d)
        self.dropout2 = nn.Dropout(p_d)
        self.dropout3 = nn.Dropout(p_d)
        self.masked_multi_headed_self_attention = MultiheadAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.multi_headed_cross_attention = MultiheadAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, p_d=p_d)

    def forward(self, x, memory, masked_self_attn_mask, cross_attn_mask):

        x = self.norm1(
            x
            + self.dropout1(
                self.masked_multi_headed_self_attention(x, x, x, masked_self_attn_mask)[
                    0
                ]
            )
        )

        x = self.norm2(
            x
            + self.dropout2(
                self.multi_headed_cross_attention(x, memory, memory, cross_attn_mask)[0]
            )
        )

        x = self.norm3(x + self.dropout3(self.ffn(x)))

        return x


class Encoder(nn.Module):
    """Encoder stack for the Transformer model.
    Applies a sequence of EncoderBlock layers to the input.
    Args:
        encoder_layers (nn.ModuleList): List of EncoderBlock layers.
    """

    def __init__(self, encoder_layers):
        super().__init__()
        self.encoder_layers = encoder_layers

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    """Decoder stack for the Transformer model.
    Applies a sequence of DecoderBlock layers to the input.
    Args:
        decoder_layers (nn.ModuleList): List of DecoderBlock layers.
    """

    def __init__(self, decoder_layers):
        super().__init__()
        self.decoder_layers = decoder_layers

    def forward(self, x, memory, masked_self_attn_mask, cross_attn_mask):
        for layer in self.decoder_layers:
            x = layer(x, memory, masked_self_attn_mask, cross_attn_mask)
        return x


class ProjectionLayer(nn.Module):
    """Projection layer for the Transformer model.
    Projects the decoder output to the vocabulary size for prediction.
    Args:
        d_model (int): Dimension of the model (embedding size).
        vocab_size (int): Size of the vocabulary.
    """

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.projection(x)


class Transformer(nn.Module):
    """Transformer model consisting of an encoder and a decoder.
    Provides methods for encoding, decoding, and full forward pass.
    Args:
        encoder (Encoder): Encoder stack.
        decoder (Decoder): Decoder stack.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, src, self_attn_mask):

        memory = self.encoder(src, self_attn_mask)

        return memory

    def decode(self, tgt, memory, masked_self_attn_mask, cross_attn_mask):

        decoder_out = self.decoder(tgt, memory, masked_self_attn_mask, cross_attn_mask)

        return decoder_out

    def forward(self, src, tgt, self_attn_mask, masked_self_attn_mask, cross_attn_mask):

        memory = self.encode(src, self_attn_mask)

        decoder_out = self.decode(tgt, memory, masked_self_attn_mask, cross_attn_mask)

        return decoder_out


class MachineTranslationModel(nn.Module):
    """Full Machine Translation model using the Transformer architecture.
    Combines embeddings, positional encoding, transformer, and output projection.
    Supports weight tying between embeddings and projection layer.
    Args:
        src_word_embedding (WordEmbeddings): Source language embedding layer.
        tgt_word_embedding (WordEmbeddings): Target language embedding layer.
        positional_encoding (PositionalEmbedding): Positional encoding layer.
        transformer (Transformer): Transformer model.
        projection_layer (ProjectionLayer): Output projection layer.
        weight_tying (bool): Whether to tie weights between embeddings and projection.
    """

    def __init__(
        self,
        src_word_embedding,
        tgt_word_embedding,
        positional_encoding,
        transformer,
        projection_layer,
        weight_tying=True,
    ):
        super().__init__()
        self.src_word_embedding = src_word_embedding
        self.tgt_word_embedding = tgt_word_embedding
        self.transformer = transformer
        self.positional_encoding = positional_encoding
        self.projection_layer = projection_layer

        self.init_with_xavier()
        # Weight tying
        if weight_tying:
            self.projection_layer.projection.weight = (
                self.tgt_word_embedding.word_embd.weight
            )
            self.src_word_embedding.word_embd.weight = (
                self.tgt_word_embedding.word_embd.weight
            )

    def encode(self, src, self_attn_mask):
        src = self.src_word_embedding(src) * math.sqrt(self.src_word_embedding.d_model)

        src = self.positional_encoding(src)

        memory = self.transformer.encode(src, self_attn_mask)

        return memory

    def decode(self, tgt, memory, masked_self_attn_mask, cross_attn_mask):
        tgt = self.tgt_word_embedding(tgt) * math.sqrt(self.tgt_word_embedding.d_model)

        tgt = self.positional_encoding(tgt)

        decoder_out = self.transformer.decode(
            tgt, memory, masked_self_attn_mask, cross_attn_mask
        )

        return decoder_out

    def project(self, decoder_out):

        logits = self.projection_layer(decoder_out)

        return logits

    def forward(self, src, tgt):

        src_mask = src == 0

        tgt_mask = tgt == 0

        causal_mask = generate_causal_mask(tgt.size(1)).to(tgt_mask.device)

        self_attn_mask = src_mask.unsqueeze(1).unsqueeze(2)

        masked_self_attn_mask = tgt_mask.unsqueeze(1).unsqueeze(2) | causal_mask

        cross_attn_mask = src_mask.unsqueeze(1).unsqueeze(2)

        memory = self.encode(src, self_attn_mask)

        decoder_out = self.decode(tgt, memory, masked_self_attn_mask, cross_attn_mask)

        logits = self.project(decoder_out)

        return logits

    def init_with_xavier(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def build_transformer(config):
    """Builds a complete Transformer-based machine translation model from a configuration object.
    Args:
        config: Configuration object with model hyperparameters.
    Returns:
        MachineTranslationModel: The constructed model.
    """
    # Word Embeddings
    src_word_embedding = WordEmbeddings(config.src_vocab_size, config.d_model)
    tgt_word_embedding = WordEmbeddings(config.tgt_vocab_size, config.d_model)

    # Shared Positional Embedding
    positional_encoding = PositionalEmbedding(config.max_seq_len, config.d_model)

    # Stack of Encoder Layers
    encoder_layers = nn.ModuleList(
        [
            EncoderBlock(
                config.d_model, config.num_heads, config.d_ff, p_d=config.dropout
            )
            for _ in range(config.num_encoder_layers)
        ]
    )

    encoder = Encoder(encoder_layers)

    # Stack of Decoder Layers
    decoder_layers = nn.ModuleList(
        [
            DecoderBlock(
                config.d_model, config.num_heads, config.d_ff, p_d=config.dropout
            )
            for _ in range(config.num_decoder_layers)
        ]
    )

    decoder = Decoder(decoder_layers)

    transformer = Transformer(encoder=encoder, decoder=decoder)

    # Output Projection
    projection_layer = ProjectionLayer(config.d_model, config.tgt_vocab_size)

    # Build Transformer
    model = MachineTranslationModel(
        src_word_embedding,
        tgt_word_embedding,
        positional_encoding,
        transformer,
        projection_layer,
        weight_tying=config.weight_tying,
    )

    return model
