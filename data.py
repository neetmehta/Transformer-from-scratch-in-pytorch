import torch
from torch.utils.data import Dataset
import pandas as pd
from random import randint

class WMTDataset(Dataset):

    def __init__(self, data_path, src_tokenizer, tgt_tokenizer, seq_len):
        super().__init__()
        self.data = pd.read_csv(data_path, lineterminator='\n')
        self.src_vocab_size = src_tokenizer.vocab_size
        self.tgt_vocab_size = tgt_tokenizer.vocab_size
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)
    
    def get_sample(self, index):
        sos_token = self.src_tokenizer.encode(["<s>"])[0]
        eos_token = self.src_tokenizer.encode(["</s>"])[0]
        pad_token = self.src_tokenizer.encode(["<pad>"])[0]
        src_encoding = self.src_tokenizer.encode(self.data.iloc[index]["en"])[
            :-1
        ]  # remove default eos token
        tgt_encoding = self.tgt_tokenizer.encode(self.data.iloc[index]["de"])[
            :-1
        ]  # remove default eos token

        if (len(src_encoding) > self.seq_len) or (len(tgt_encoding) > self.seq_len):
            return

        src_padding_len = self.seq_len - (len(src_encoding) + 2)
        tgt_padding_len = self.seq_len - (len(tgt_encoding) + 1)

        src_encoding = torch.tensor(
            [sos_token] + src_encoding + [eos_token] + [pad_token] * src_padding_len,
            dtype=torch.long,
        )
        label = torch.tensor(
            tgt_encoding + [eos_token] + [pad_token] * tgt_padding_len, dtype=torch.long
        )
        tgt_encoding = torch.tensor(
            [sos_token] + tgt_encoding + [pad_token] * tgt_padding_len, dtype=torch.long
        )

        src_mask = (src_encoding == pad_token).unsqueeze(0)
        tgt_mask = (tgt_encoding == pad_token).unsqueeze(0)

        return src_encoding, tgt_encoding, label, src_mask, tgt_mask


    def __getitem__(self, index):
        out = self.get_sample(index)
        
        if out is None:
            while True:
                index = randint(0, len(self.data))
                out = self.get_sample(index)
                
                if out is not None:
                    return self.get_sample(index)
                
        return out