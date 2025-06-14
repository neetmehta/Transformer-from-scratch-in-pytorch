import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence


def map_dataset(data, tokenizer, workers=0):
    """Maps dataset to a tokenized source and target.

    Args:
        data (_type_): Hugging face data
        tokenizer (_type_): pretrained tokenizer
        workers (int, optional): number of parallel threads to speed up dataset mapping. Defaults to 0.
    """
    def map_example(example):
        src_encoded = tokenizer.batch_encode(example["translation_src"])
        tgt_encoded = tokenizer.batch_encode(example["translation_tgt"])

        example["translation_src_tokens"] = src_encoded["input_ids"]
        example["translation_tgt_tokens"] = tgt_encoded["input_ids"]
        example["src_len"] = [len(ids) for ids in src_encoded["input_ids"]]
        example["tgt_len"] = [len(ids) for ids in tgt_encoded["input_ids"]]
        return example

    return data.map(map_example, batched=True, batch_size=1000, num_proc=workers)


def pad_collate_fn(batch):
    """Collate function to pad sequences in a batch.
    Args:
        batch (list): List of tuples, where each tuple contains (src, tgt, label).
            - src: Source sentence tensor
            - tgt: Target sentence tensor
            - label: Label sentence tensor
    Returns:
        tuple: A tuple containing padded tensors for source, target, and label sentences.
    """
    src_sentences, tgt_sentences, label_sentences = [], [], []
    for sample in batch:
        src_sentences += [sample[0]]
        tgt_sentences += [sample[1]]
        label_sentences += [sample[2]]

    src_sentences = pad_sequence(src_sentences, batch_first=True, padding_value=0)
    tgt_sentences = pad_sequence(tgt_sentences, batch_first=True, padding_value=0)
    label_sentences = pad_sequence(label_sentences, batch_first=True, padding_value=0)

    return src_sentences, tgt_sentences, label_sentences


def prepare_data(dataset):
    """Prepares the dataset by renaming columns and flattening it.
    Args:
        dataset (Dataset): Hugging Face dataset containing translations.
    Returns:
        Dataset: Processed dataset with renamed columns and flattened structure.
    """
    dataset = dataset.flatten()
    dataset = dataset.rename_column("translation.de", "translation_tgt")
    dataset = dataset.rename_column("translation.en", "translation_src")
    return dataset


class WMTENDE(Dataset):
    """Dataset class for WMT English-German translation task.
    This class loads the WMT dataset, tokenizes the source and target translations,
    and prepares the data for training.
    Args:
        config (Config): Configuration object containing dataset settings.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for encoding the translations.
        no_of_samples (int): Number of samples to load from the dataset. If -1, loads all samples.
        split (str): Dataset split to load ("train" or "validation").
    """

    def __init__(self, config, tokenizer, no_of_samples, split="train"):
        """Initialize the WMTENDE dataset.
        Args:
            config (Config): Configuration object containing dataset settings.
            tokenizer (PreTrainedTokenizer): Tokenizer to use for encoding the translations.
            no_of_samples (int): Number of samples to load from the dataset. If -1, loads all samples.
            split (str): Dataset split to load ("train" or "validation").
        """
        super().__init__()
        data = load_dataset(config.dataset, config.language, split=split).shuffle(
            seed=42
        )

        if no_of_samples != -1:
            data = data.select(range(no_of_samples))
        data = prepare_data(data)
        self.data = (
            map_dataset(data, tokenizer, workers=config.workers)
            .sort("src_len")
            .filter(
                lambda x: x["src_len"] < config.max_seq_len
                and x["tgt_len"] < config.max_seq_len,
                num_proc=config.workers,
            )
        )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Get a sample from the dataset by index.
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the source, target, and label tensors.
                - src: Source sentence tensor
                - tgt: Target sentence tensor (without the last token)
                - label: Target sentence tensor (without the first token)
        """
        return (
            torch.tensor(
                self.data[index]["translation_src_tokens"], dtype=torch.long
            ),  # src
            torch.tensor(
                self.data[index]["translation_tgt_tokens"][:-1], dtype=torch.long  # tgt
            ),
            torch.tensor(
                self.data[index]["translation_tgt_tokens"][1:],
                dtype=torch.long,  # label
            ),
        )


class CustomBatchSampler(Sampler):
    """Custom batch sampler to create batches of a specified size.
    Args:
        data_len (int): Total number of samples in the dataset.
        batch_size (int): Size of each batch.
    """

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


def get_dataloaders(config, tokenizer):
    """Get DataLoaders for training and validation datasets.
    Args:
        config (Config): Configuration object containing dataset settings.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for encoding the translations.
    Returns:
        tuple: A tuple containing the training and validation DataLoaders.
    """
    train_dataset = WMTENDE(
        config, tokenizer, no_of_samples=config.train_samples, split="train"
    )
    val_dataset = WMTENDE(
        config, tokenizer, no_of_samples=config.val_samples, split="validation"
    )

    train_sampler = CustomBatchSampler(len(train_dataset), config.train_batch_size)
    val_sampler = CustomBatchSampler(len(val_dataset), config.val_batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=pad_collate_fn,
        num_workers=config.workers,
        pin_memory=config.pin_memory,
    )

    if config.overfit:
        return train_loader, train_loader

    else:
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=pad_collate_fn,
            num_workers=config.workers,
            pin_memory=config.pin_memory,
        )

        return train_loader, val_loader
