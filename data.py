import torch
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from datasets import load_dataset
from tokenizer import *


def map_dataset(data, tokenizer):
    # Do not flatten/rename here if already done in TokenizerWrapper
    def map_example(example):
        src_encoded = tokenizer.batch_encode(example["translation_src"])
        tgt_encoded = tokenizer.batch_encode(example["translation_tgt"])

        example["translation_src_tokens"] = src_encoded["input_ids"]
        example["translation_tgt_tokens"] = tgt_encoded["input_ids"]
        example["src_len"] = [len(ids) for ids in src_encoded["input_ids"]]
        example["tgt_len"] = [len(ids) for ids in tgt_encoded["input_ids"]]
        return example

    return data.map(map_example, batched=True, batch_size=1000)


def pad_collate_fn(batch):
    src_sentences, tgt_sentences = [], []
    for sample in batch:
        src_sentences += [sample[0]]
        tgt_sentences += [sample[1]]

    src_sentences = pad_sequence(src_sentences, batch_first=True, padding_value=0)
    tgt_sentences = pad_sequence(tgt_sentences, batch_first=True, padding_value=0)

    return src_sentences, tgt_sentences


def prepare_data(dataset):
    dataset = dataset.flatten()
    dataset = dataset.rename_column("translation.de", "translation_tgt")
    dataset = dataset.rename_column("translation.en", "translation_src")
    return dataset


class WMTENDE(Dataset):

    def __init__(self, config, tokenizer, split="train"):
        super().__init__()
        data = (
            load_dataset(config.dataset, config.language, split=split)
            .shuffle(seed=42)
            .select(range(config.no_of_samples))
        )
        data = prepare_data(data)
        self.data = (
            map_dataset(data, tokenizer)
            .sort("src_len")
            .filter(
                lambda x: x["src_len"] < config.max_seq_len
                and x["tgt_len"] < config.max_seq_len
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(
            self.data[index]["translation_src_tokens"], dtype=torch.long
        ), torch.tensor(self.data[index]["translation_tgt_tokens"], dtype=torch.long)


class CustomBatchSampler(Sampler):

    def __init__(self, data_len, batch_size):
        self.indices = torch.arange(data_len, dtype=torch.long)
        self.indices = list(torch.split(self.indices, batch_size))
        self.indices = [i.tolist() for i in self.indices]
        random.shuffle(self.indices)
        self.len = data_len

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)
