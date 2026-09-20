import numpy as np
from datasets import load_dataset
from config import get_config
from train import get_or_build_transformer

def main():
    config = get_config()
    print(f"Loading dataset opus-100 {config['lang_src']}-{config['lang_tgt']}...")
    ds_raw = load_dataset("Helsinki-NLP/opus-100", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    print("Loading tokenizers...")
    tokenizer_src = get_or_build_transformer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_transformer(config, ds_raw, config["lang_tgt"])
    
    src_lengths = []
    tgt_lengths = []
    
    print("Calculating sequence lengths (this might take a few seconds)...")
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        src_lengths.append(len(src_ids))
        tgt_lengths.append(len(tgt_ids))
        
    print(f"\n--- Source ({config['lang_src']}) Length Distribution ---")
    print(f"Max length: {np.max(src_lengths)}")
    print(f"Mean length: {np.mean(src_lengths):.2f}")
    print(f"90th percentile: {int(np.percentile(src_lengths, 90))}")
    print(f"95th percentile: {int(np.percentile(src_lengths, 95))}")
    print(f"99th percentile: {int(np.percentile(src_lengths, 99))}")
    print(f"99.9th percentile: {int(np.percentile(src_lengths, 99.9))}")

    print(f"\n--- Target ({config['lang_tgt']}) Length Distribution ---")
    print(f"Max length: {np.max(tgt_lengths)}")
    print(f"Mean length: {np.mean(tgt_lengths):.2f}")
    print(f"90th percentile: {int(np.percentile(tgt_lengths, 90))}")
    print(f"95th percentile: {int(np.percentile(tgt_lengths, 95))}")
    print(f"99th percentile: {int(np.percentile(tgt_lengths, 99))}")
    print(f"99.9th percentile: {int(np.percentile(tgt_lengths, 99.9))}")

if __name__ == "__main__":
    main()

