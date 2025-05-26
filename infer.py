from model import build_transformer
from tokenizer import TokenizerWrapper
import argparse
from utils import greedy_decode, load_config, load_checkpoint
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    config_module = load_config(args.config)
    config = config_module.Config()  # instantiate your Config class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_transformer(config).to(device)
    
    tokenizer = TokenizerWrapper(config)
    
    start_epoch, loss = load_checkpoint(model, path=config.checkpoint_path)
    
    assert (start_epoch>0) and (loss<float('inf')), "checkpoint not loaded"
    src_text = input("Input your sentence: ")
    
    src = torch.tensor(tokenizer.encode(src_text)).unsqueeze(0).to(device)
    
    _, out = greedy_decode(src, model, tokenizer, config, device)
    
    print(f"German translation: {out}")
    
if __name__ == "__main__":
    main()
