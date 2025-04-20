import torch
import torch.nn as nn
import math

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
    
    def __init__(self, n_embd, num_heads):
        super().__init__()
        
        self.h = num_heads
        self.w_q = nn.Linear(n_embd, n_embd)
        self.w_k = nn.Linear(n_embd, n_embd)
        self.w_v = nn.Linear(n_embd, n_embd)
        
        self.w_o = nn.Linear(n_embd, n_embd)
        
        self.d_k = n_embd // num_heads
        
        
        assert n_embd % num_heads == 0, "d_model not divisible by num_heads"
        
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
    
    def __init__(self, n_embd, num_heads, d_ff, p_d=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(p_d)
        self.multi_headed_self_attention = MultiheadAttention(n_embd=n_embd, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(d_model=n_embd, d_ff=d_ff, p_d=p_d)
        
    def forward(self, x, padding_mask):
        
        x_residual_1 = x
        
        attention_out = self.dropout(self.multi_headed_self_attention(x, x, x), padding_mask)
        
        sublayer_1_out = self.layer_norm(x_residual_1 + attention_out)
        
        x_residual_2 = sublayer_1_out
        
        ffn_out = self.dropout(self.ffn(sublayer_1_out))
        
        sublayer_2_out = self.layer_norm(x_residual_2 + ffn_out)
        
        return sublayer_2_out
    
class DecoderBlock(nn.Module):
    
    def __init__(self, n_embd, num_heads, d_ff, p_d=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.layer_norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(p_d)
        self.masked_multi_headed_self_attention = MultiheadAttention(n_embd=n_embd, num_heads=num_heads)
        self.multi_headed_cross_attention = MultiheadAttention(n_embd=n_embd, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(d_model=n_embd, d_ff=d_ff, p_d=p_d)
        
    def forward(self, x, padding_mask):
        x_residual_1 = x
        
        causal_mask = torch.tril(torch.ones(self.n_embd, self.n_embd))
        mask = torch.logical_or(causal_mask, padding_mask)
        
        