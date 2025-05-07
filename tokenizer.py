from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
from tokenizers import normalizers
from tokenizers.processors import TemplateProcessing

def batch_iterator_src(data, batch_size=10000):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]['translation_src']
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]['translation_trg']
        
def prepare_data(dataset):
    dataset = dataset.flatten()
    dataset=dataset.rename_column('translation.de','translation_trg')
    dataset=dataset.rename_column('translation.en','translation_src')
    return dataset
        
def load_tokenizer(file_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=file_path)
    tokenizer.add_special_tokens({'pad_token':'[PAD]'})
    return tokenizer
        
def train_bpe_tokenizer(data, vocab_size, save=None):
    
    # 3. Define special tokens
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    
    # 4. Initialize and configure tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFKC()])
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
    special_tokens=special_tokens,
    vocab_size=vocab_size
)

    # 5. Train the tokenizer
    tokenizer.train_from_iterator(data, trainer=trainer)

    # 6. Set post-processing template to add BOS and EOS
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] [BOS] $B [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if save is not None:
        tokenizer.save(save)
        
    return tokenizer
        
def train_word_level_tokenizer(data, vocab_size, save=None):
    
    # 3. Define special tokens
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    
    # 4. Initialize and configure tokenizer
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFKC()])
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = WordLevelTrainer(
    special_tokens=special_tokens,
    vocab_size=vocab_size
)

    # 5. Train the tokenizer
    tokenizer.train_from_iterator(data, trainer=trainer)

    # 6. Set post-processing template to add BOS and EOS
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] [BOS] $B [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    
    if save is not None:
        tokenizer.save(save)
        
    return tokenizer