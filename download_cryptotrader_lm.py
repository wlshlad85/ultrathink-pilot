#!/usr/bin/env python3
"""
Download CryptoTrader-LM model from Hugging Face
For benchmarking against ultrathink-pilot PPO agent
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

def download_cryptotrader_lm():
    """Download CryptoTrader-LM model and tokenizer"""

    model_name = "agarkovv/CryptoTrader-LM"
    download_dir = "./models/cryptotrader_lm"

    print(f"Downloading CryptoTrader-LM from Hugging Face...")
    print(f"Model: {model_name}")
    print(f"Destination: {download_dir}\n")

    try:
        # Create directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        # Download the entire repository
        print("Step 1/3: Downloading model files...")
        repo_path = snapshot_download(
            repo_id=model_name,
            local_dir=download_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded to: {repo_path}\n")

        # Load tokenizer to verify
        print("Step 2/3: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(download_dir)
        print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}\n")

        # Load model to verify
        print("Step 3/3: Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(download_dir)
        print(f"✓ Model loaded: {model.config.model_type}")
        print(f"  - Architecture: {model.__class__.__name__}")
        print(f"  - Parameters: {model.num_parameters():,}")
        print(f"  - Labels: {model.config.num_labels}")

        if hasattr(model.config, 'id2label'):
            print(f"  - Label mapping: {model.config.id2label}")

        print(f"\n✅ CryptoTrader-LM successfully downloaded!")
        print(f"\nModel info:")
        print(f"  - Trained on: BTC/ETH data (2022-01-01 to 2024-10-15)")
        print(f"  - Task: Trading decisions (buy, sell, hold)")
        print(f"  - Input: Crypto news + historical price data")
        print(f"  - Challenge: FinNLP @ COLING-2025")

        print(f"\nUsage:")
        print(f"  from transformers import AutoTokenizer, AutoModelForSequenceClassification")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{download_dir}')")
        print(f"  model = AutoModelForSequenceClassification.from_pretrained('{download_dir}')")

        return True

    except Exception as e:
        print(f"❌ Error downloading CryptoTrader-LM: {e}")
        return False

def check_disk_space():
    """Check available disk space"""
    import shutil
    total, used, free = shutil.disk_usage(".")
    print(f"Disk space:")
    print(f"  Total: {total // (2**30)} GB")
    print(f"  Used: {used // (2**30)} GB")
    print(f"  Free: {free // (2**30)} GB")
    print(f"  Estimated model size: ~500 MB\n")

    if free < 1 * (2**30):  # Less than 1 GB free
        print("⚠️  Warning: Low disk space!")
        return False
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("CryptoTrader-LM Model Downloader")
    print("=" * 70)
    print()

    # Check disk space
    if not check_disk_space():
        print("Please free up some disk space before continuing.")
        exit(1)

    # Download model
    success = download_cryptotrader_lm()

    if success:
        print("\n" + "=" * 70)
        print("Next steps:")
        print("=" * 70)
        print("1. Review the model in ./models/cryptotrader_lm/")
        print("2. Create benchmark script to compare with your PPO agent")
        print("3. Test on your Bitcoin data (2022-2024)")
        print("4. Compare performance metrics (Sharpe, returns, drawdown)")
        print()
    else:
        print("\n❌ Download failed. Please check your internet connection and try again.")
        exit(1)
