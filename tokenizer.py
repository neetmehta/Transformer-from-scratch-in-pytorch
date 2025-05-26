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
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]["translation_src"]
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]["translation_tgt"]


class TokenizerWrapper:

    def __init__(self, config, data=None) -> None:
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
        if self.cfg.tokenizer_type == "bpe":
            return self.train_bpe_tokenizer(data_iterator)
        else:
            return self.train_word_level_tokenizer(data_iterator)

    def train_bpe_tokenizer(self, data_iterator):
        return self._train_tokenizer_core(
            model=BPE(unk_token="[UNK]"),
            trainer_cls=BpeTrainer,
            data_iterator=data_iterator,
        )

    def train_word_level_tokenizer(self, data_iterator):
        return self._train_tokenizer_core(
            model=WordLevel(unk_token="[UNK]"),
            trainer_cls=WordLevelTrainer,
            data_iterator=data_iterator,
        )

    def _train_tokenizer_core(self, model, trainer_cls, data_iterator):
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
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def batch_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens)

    def batch_encode(self, texts):
        return self.tokenizer.batch_encode_plus(texts)

    def load_tokenizer(self, file_path):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=file_path)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

def train_tokenizer(config):
    dataset = load_dataset(config.dataset, config.language)
    dataset = prepare_data(dataset)
    dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
    return TokenizerWrapper(config, dataset)