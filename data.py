import torch
from torch.utils.data import Dataset
import pandas as pd

class WMTDataset(Dataset):
    
    def __init__(self, data_path, src_tokenizer, tgt_tokenizer, seq_len):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.src_vocab_size = src_tokenizer.vocab_size
        self.tgt_vocab_size = tgt_tokenizer.vocab_size
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sos_token = self.src_tokenizer.encode(['<s>'])[0]
        eos_token = self.src_tokenizer.encode(['</s>'])[0]
        pad_token = self.src_tokenizer.encode(['<pad>'])[0]
        src_encoding = self.src_tokenizer.encode(self.data.iloc[index]['en'])[:-1] # remove default eos token
        tgt_encoding = self.tgt_tokenizer.encode(self.data.iloc[index]['de'])[:-1] # remove default eos token
        print("len of src sen: ", len(src_encoding))
        print("len of tgt sen: ", len(tgt_encoding))
        assert len(src_encoding) < self.seq_len + 2, "sentence too big"
        assert len(tgt_encoding) < self.seq_len + 2, "sentence too big"
        
        src_padding_len = self.seq_len - (len(src_encoding) + 2)  
        tgt_padding_len = self.seq_len - (len(tgt_encoding) + 2) 
        
        src_encoding = torch.tensor([sos_token] + src_encoding + [eos_token] + [pad_token]*src_padding_len, dtype=torch.uint64)
        tgt_encoding = torch.tensor([sos_token] + tgt_encoding + [eos_token] + [pad_token]*tgt_padding_len, dtype=torch.uint64)
        
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len, dtype=bool), diagonal=1).to(bool)

        encoder_self_attention_mask = (src_encoding == pad_token).unsqueeze(0) | (src_encoding == pad_token).unsqueeze(1)
        decoder_self_attention_mask = (tgt_encoding == pad_token).unsqueeze(0) | (tgt_encoding == pad_token).unsqueeze(1) | causal_mask
        decoder_cross_attention_mask = (tgt_encoding == pad_token).unsqueeze(0) | (src_encoding == pad_token).unsqueeze(1)
        
        return src_encoding, tgt_encoding, encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask