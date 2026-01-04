"""
Pretraining Dataset Module
==========================
Handles large-scale pretraining datasets like FineWeb-Edu.

This is for PRETRAINING (raw text) not fine-tuning (conversations).
For an 800M model, you need ~16B+ tokens minimum.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Iterator, Dict, Any
from dataclasses import dataclass
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
import numpy as np


@dataclass
class PretrainConfig:
    """Configuration for pretraining dataset."""
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-100BT"
    max_length: int = 1024
    cache_dir: str = "./pretrain_cache"
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    

class StreamingPretrainDataset(IterableDataset):
    """Streaming dataset for pretraining on large text corpora.
    
    Uses HuggingFace streaming to avoid downloading entire dataset.
    Tokenizes on-the-fly and packs sequences for efficiency.
    """
    
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "sample-100BT",
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        max_length: int = 1024,
        cache_dir: str = "./pretrain_cache",
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        

        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vocab_size = len(self.tokenizer)
        print(f"Tokenizer vocab size: {self.vocab_size}")
        
    def _get_stream(self):
        """Get a fresh streaming dataset iterator with retry logic."""
        from datasets import load_dataset
        import time
        import random
        
        print(f"Streaming from: {self.dataset_name} ({self.dataset_config})")
        
        max_retries = 10
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                ds = load_dataset(
                    self.dataset_name,
                    name=self.dataset_config,
                    split="train",
                    streaming=True,
                    cache_dir=str(self.cache_dir),
                )
                return iter(ds)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"Failed after {max_retries} attempts")
                    raise
    
    def _pack_tokens(self, stream: Iterator) -> Iterator[Dict[str, torch.Tensor]]:
        """Pack multiple documents into fixed-length sequences.
        
        This is more efficient than padding each document separately.
        Uses document boundaries with EOS tokens.
        """
        buffer = []
        eos_id = self.tokenizer.eos_token_id
        
        for doc in stream:

            text = doc.get("text", "")
            if not text or len(text.strip()) < 50:
                continue
            

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            

            tokens.append(eos_id)
            

            buffer.extend(tokens)
            

            while len(buffer) >= self.max_length:
                seq = buffer[:self.max_length]
                buffer = buffer[self.max_length:]
                
                input_ids = torch.tensor(seq, dtype=torch.long)

                labels = input_ids.clone()

                label_mask = torch.ones(self.max_length, dtype=torch.float)
                
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "label_mask": label_mask,
                }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over packed sequences."""
        stream = self._get_stream()
        return self._pack_tokens(stream)


class ShuffledPretrainDataset(IterableDataset):
    """Pretraining dataset with shuffling buffer.
    
    Maintains a buffer to provide some randomization even with streaming.
    """
    
    def __init__(
        self,
        base_dataset: StreamingPretrainDataset,
        buffer_size: int = 10000,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.buffer_size = buffer_size
        self.seed = seed
        

        self.tokenizer = base_dataset.tokenizer
        self.vocab_size = base_dataset.vocab_size
        self.max_length = base_dataset.max_length
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate with shuffle buffer."""
        import random
        rng = random.Random(self.seed)
        
        buffer = []
        base_iter = iter(self.base_dataset)
        

        for item in base_iter:
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                break
        

        for item in base_iter:
            idx = rng.randint(0, len(buffer) - 1)
            yield buffer[idx]
            buffer[idx] = item
        

        rng.shuffle(buffer)
        for item in buffer:
            yield item


def create_pretrain_dataloader(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-100BT",
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
    max_length: int = 1024,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle_buffer: int = 10000,
    cache_dir: str = "./pretrain_cache",
) -> tuple:
    """Create a streaming dataloader for pretraining.
    
    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset config (e.g., "sample-100BT" for 100B tokens)
        tokenizer_name: Tokenizer to use
        max_length: Sequence length
        batch_size: Batch size
        num_workers: DataLoader workers
        shuffle_buffer: Size of shuffle buffer
        cache_dir: Cache directory
    
    Returns:
        dataloader: Streaming DataLoader
        tokenizer: The tokenizer (for model vocab size)
    """

    base_ds = StreamingPretrainDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        cache_dir=cache_dir,
    )
    

    dataset = ShuffledPretrainDataset(
        base_dataset=base_ds,
        buffer_size=shuffle_buffer,
    )
    

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return dataloader, base_ds.tokenizer


def estimate_tokens(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-100BT",
    sample_size: int = 1000,
) -> dict:
    """Estimate total tokens in dataset by sampling.
    
    Returns estimate of total tokens and tokens per document.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    print(f"Sampling {sample_size} documents to estimate token count...")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    ds = load_dataset(
        dataset_name,
        name=dataset_config,
        split="train",
        streaming=True,
    )
    
    total_tokens = 0
    doc_count = 0
    
    for doc in ds:
        if doc_count >= sample_size:
            break
        text = doc.get("text", "")
        if text:
            tokens = len(tokenizer.encode(text))
            total_tokens += tokens
            doc_count += 1
    
    avg_tokens_per_doc = total_tokens / doc_count if doc_count > 0 else 0
    

    if "100BT" in dataset_config:
        estimated_total = 100_000_000_000
    elif "10BT" in dataset_config:
        estimated_total = 10_000_000_000
    elif "350BT" in dataset_config:
        estimated_total = 350_000_000_000
    else:
        estimated_total = avg_tokens_per_doc * 1000000
    
    return {
        "sample_tokens": total_tokens,
        "sample_docs": doc_count,
        "avg_tokens_per_doc": avg_tokens_per_doc,
        "estimated_total_tokens": estimated_total,
        "estimated_total_docs": estimated_total / avg_tokens_per_doc if avg_tokens_per_doc > 0 else 0,
    }


if __name__ == "__main__":

    print("Testing StreamingPretrainDataset...")
    

    ds = StreamingPretrainDataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        max_length=512,
    )
    
    print("\nSampling 5 sequences...")
    for i, batch in enumerate(ds):
        if i >= 5:
            break
        print(f"\nSequence {i+1}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  First 20 tokens: {batch['input_ids'][:20].tolist()}")
        decoded = ds.tokenizer.decode(batch['input_ids'][:50])
        print(f"  Decoded start: {decoded[:100]}...")
    
    print("\n✓ Streaming dataset working!")
