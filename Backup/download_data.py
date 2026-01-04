#!/usr/bin/env python3
"""
Dataset Downloader Script
=========================
Downloads and merges all training datasets before training.
Run this first to prepare the data.

Usage:
    python download_data.py
"""

import argparse
from pathlib import Path

from dataset import DatasetLoader, DatasetConfig


def main():
    parser = argparse.ArgumentParser(description="Download and merge training datasets")
    parser.add_argument('--cache-dir', type=str, default='./data_cache', help='Cache directory')
    parser.add_argument('--output-dir', type=str, default='./data_merged', help='Output directory')
    parser.add_argument('--force', action='store_true', help='Force re-download even if cached')
    parser.add_argument('--max-speed', type=int, default=None, help='Max download speed in MB/s (default: unlimited)')
    args = parser.parse_args()
    
    print("="*60)
    print("Dataset Downloader")
    print("="*60)
    print()
    print("This will download and merge the following datasets:")
    print("  1. TeichAI/MiMo-V2-Flash-2300x (~2,300 examples)")
    print("  2. TeichAI/glm-4.7-2000x (~2,000 examples)")
    print("  3. TeichAI/minimax-m2.1-1000x (~1,000 examples)")
    print("  4. TeichAI/claude-4.5-opus-high-reasoning-250x (~250 examples)")
    print("  5. open-thoughts/OpenThoughts3-1.2M (1.2M examples)")
    print("  6. a-m-team/AM-DeepSeek-R1-0528-Distilled (Huge, streamed)")
    print()
    print(f"Cache directory: {args.cache_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    

    if args.force:
        merged_path = Path(args.output_dir) / "merged_dataset"
        if merged_path.exists():
            import shutil
            print(f"Removing existing merged dataset: {merged_path}")
            shutil.rmtree(merged_path)
    

    config = DatasetConfig(
        cache_dir=args.cache_dir,
        merged_path=args.output_dir,
        max_download_speed_mb=args.max_speed,
    )
    
    loader = DatasetLoader(config)
    

    print("\nStarting download...")
    datasets = loader.run()
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"Training examples: {len(datasets['train'])}")
    print(f"Validation examples: {len(datasets['validation'])}")
    print(f"\nMerged dataset saved to: {args.output_dir}")
    print("\nYou can now run training with:")
    print("  python train.py --config config.yaml")


if __name__ == "__main__":
    main()
