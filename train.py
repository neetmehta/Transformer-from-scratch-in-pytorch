import torch
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
special_token_dict = {"bos_token": "<s>"}
tokenizer.add_special_tokens(special_token_dict)