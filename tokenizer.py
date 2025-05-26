from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from data import prepare_data
from datasets import load_dataset, concatenate_datasets


def batch_iterator(data, batch_size=100000):
    """ "Yield batches of data for tokenizer training.
    Args:
        data (Dataset): The dataset to iterate over.
        batch_size (int): The size of each batch.
    Yields:
        dict: A dictionary containing the source and target translations for each batch.
    """

    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]["translation_src"]
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]["translation_tgt"]


class TokenizerWrapper:
    """Wrapper for training and using a tokenizer.
    This class handles the training of a tokenizer based on the provided configuration.
    It supports both BPE and word-level tokenizers, and can save the trained tokenizer to a file.
    Args:
        config (Config): Configuration object containing tokenizer settings.
        data (Dataset, optional): Dataset to use for training the tokenizer. Required if `train_tokenizer` is True.
    Attributes:
        cfg (Config): Configuration object.
        tokenizer (PreTrainedTokenizerFast): The trained or loaded tokenizer.
    """

    def __init__(self, config, data=None) -> None:
        """Initialize the TokenizerWrapper with a configuration and optional data.
        Args:
            config (Config): Configuration object containing tokenizer settings.
            data (Dataset, optional): Dataset to use for training the tokenizer. Required if `train_tokenizer` is True.
        """
        self.cfg = config

        if self.cfg.train_tokenizer:
            assert self.cfg.tokenizer_type in [
                "bpe",
                "wordlevel",
            ], "Tokenizer type must be either 'bpe' or 'wordlevel'"
            assert data is not None, "Data must be provided for training"

            batches = batch_iterator(data, batch_size=10000)
            self.tokenizer = self.train_tokenizer(batches)
        else:
            self.tokenizer = self.load_tokenizer(self.cfg.tokenizer_path)

    def train_tokenizer(self, data_iterator):
        """Train a tokenizer based on the configuration and provided data.
        Args:
            data_iterator (iterable): An iterable that yields batches of data for training the tokenizer.
        Returns:
            PreTrainedTokenizerFast: The trained tokenizer.
        """
        if self.cfg.tokenizer_type == "bpe":
            return self.train_bpe_tokenizer(data_iterator)
        else:
            return self.train_word_level_tokenizer(data_iterator)

    def train_bpe_tokenizer(self, data_iterator):
        """Train a BPE tokenizer using the provided data iterator.
        Args:
            data_iterator (iterable): An iterable that yields batches of data for training the tokenizer.
        Returns:
            PreTrainedTokenizerFast: The trained BPE tokenizer.
        """
        
        return self._train_tokenizer_core(
            model=BPE(unk_token="[UNK]"),
            trainer_cls=BpeTrainer,
            data_iterator=data_iterator,
        )

    def train_word_level_tokenizer(self, data_iterator):
        """Train a word-level tokenizer using the provided data iterator.
        Args:
            data_iterator (iterable): An iterable that yields batches of data for training the tokenizer.
        Returns:
            PreTrainedTokenizerFast: The trained word-level tokenizer.
        """
        return self._train_tokenizer_core(
            model=WordLevel(unk_token="[UNK]"),
            trainer_cls=WordLevelTrainer,
            data_iterator=data_iterator,
        )

    def _train_tokenizer_core(self, model, trainer_cls, data_iterator):
        """Core function to train a tokenizer with the specified model and trainer.
        Args:
            model (TokenizerModel): The tokenizer model to use (BPE or WordLevel).
            trainer_cls (TokenizerTrainer): The trainer class for the tokenizer.
            data_iterator (iterable): An iterable that yields batches of data for training the tokenizer.
        Returns:
            PreTrainedTokenizerFast: The trained tokenizer.
        """
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizers.Sequence([NFKC()])
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()

        trainer = trainer_cls(
            special_tokens=special_tokens, vocab_size=self.cfg.vocab_size
        )

        tokenizer.train_from_iterator(data_iterator, trainer=trainer)

        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [EOS] [BOS] $B [EOS]",
            special_tokens=[
                ("[BOS]", tokenizer.token_to_id("[BOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ],
        )

        if self.cfg.tokenizer_path:
            tokenizer.save(self.cfg.tokenizer_path)

        return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]")

    def encode(self, text):
        """Encode a single text string into tokens.
        Args:
            text (str): The text to encode.
        Returns:
            list: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        """Decode a list of token IDs back into a text string.
        Args:
            tokens (list): A list of token IDs to decode.
        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens)

    def batch_decode(self, tokens):
        """Decode a batch of token IDs back into text strings.
        Args:
            tokens (list): A list of lists, where each inner list contains token IDs to decode.
        Returns:
            list: A list of decoded text strings.
        """
        return self.tokenizer.batch_decode(tokens)

    def batch_encode(self, texts):
        """Encode a batch of text strings into token IDs.
        Args:
            texts (list): A list of text strings to encode.
        Returns:
            dict: A dictionary containing the encoded token IDs and attention masks.
        """
        return self.tokenizer.batch_encode_plus(texts)

    def load_tokenizer(self, file_path):
        """Load a tokenizer from a specified file path.
        Args:
            file_path (str): The path to the tokenizer file.
        Returns:
            PreTrainedTokenizerFast: The loaded tokenizer.
        """
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=file_path)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer


def train_tokenizer(config):
    """Train a tokenizer based on the provided configuration.
    Args:
        config (Config): Configuration object containing tokenizer settings.
    Returns:
        TokenizerWrapper: An instance of TokenizerWrapper containing the trained tokenizer.
    """
    dataset = load_dataset(config.dataset, config.language)
    dataset = prepare_data(dataset)
    dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
    return TokenizerWrapper(config, dataset)
