"""
Dataset Downloader and Merger
=============================
Downloads and merges the following datasets:
- TeichAI/MiMo-V2-Flash-2300x
- TeichAI/glm-4.7-2000x
- TeichAI/minimax-m2.1-1000x
- TeichAI/claude-4.5-opus-high-reasoning-250x

All datasets share the same conversational format with thinking/reasoning.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    datasets: List[str] = field(default_factory=lambda: [
        "TeichAI/MiMo-V2-Flash-2300x",
        "TeichAI/glm-4.7-2000x",
        "TeichAI/minimax-m2.1-1000x",
        "TeichAI/claude-4.5-opus-high-reasoning-250x",
        "open-thoughts/OpenThoughts3-1.2M",
        "a-m-team/AM-DeepSeek-R1-0528-Distilled",
    ])
    cache_dir: str = "./data_cache"
    merged_path: str = "./data_merged"
    train_split: float = 0.95
    seed: int = 42
    max_length: int = 2048
    max_samples_per_dataset: Optional[int] = None
    num_proc: int = 32
    max_download_speed_mb: Optional[int] = None


class DatasetLoader:
    """Handles downloading, processing, and merging of datasets."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.merged_path = Path(config.merged_path)
        

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.merged_path.mkdir(parents=True, exist_ok=True)
    
    def download_datasets(self) -> List[Dataset]:
        """Download all datasets from HuggingFace.
        
        Uses streaming for large datasets to avoid RAM exhaustion.
        Data is processed in batches and saved incrementally to disk.
        """
        datasets_list = []
        
        for dataset_name in tqdm(self.config.datasets, desc="Downloading datasets"):
            print(f"\nDownloading: {dataset_name}")
            try:

                is_large = any(x in dataset_name for x in ["1.2M", "1M", "OpenThoughts", "DeepSeek", "Distilled"])
                
                if is_large:
                    ds = self._load_large_dataset_streaming(dataset_name)
                else:
                    ds = self._load_standard_dataset(dataset_name)
                
                if ds is None:
                    continue
                    
                print(f"  Loaded {len(ds)} examples")
                print(f"  Columns: {ds.column_names}")
                

                print(f"  Verifying data integrity...")
                sample_indices = [0, min(100, len(ds)-1), len(ds)-1] if len(ds) > 2 else [0]
                for idx in sample_indices:
                    sample = ds[idx]

                    if 'conversations' in sample:
                        convs = sample['conversations']
                        if not convs or (isinstance(convs, list) and len(convs) == 0):
                            print(f"  WARNING: Sample {idx} has empty conversations!")
                        else:
                            print(f"  ✓ Sample {idx} verified: {len(convs)} turns")
                    elif 'messages' in sample:
                        msgs = sample['messages']
                        if not msgs or (isinstance(msgs, list) and len(msgs) == 0):
                            print(f"  WARNING: Sample {idx} has empty messages!")
                        else:
                            print(f"  ✓ Sample {idx} verified: {len(msgs)} messages")
                
                datasets_list.append(ds)
                
            except Exception as e:
                print(f"  Error loading {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return datasets_list
    
    def _load_large_dataset_streaming(self, dataset_name: str) -> Optional[Dataset]:
        """Load large datasets using streaming to control memory usage.
        
        Instead of loading everything into RAM, we:
        1. Stream the dataset from the hub
        2. Write directly to Arrow format on disk using from_generator
        3. Load the memory-mapped dataset
        """
        from datasets import load_dataset, Dataset, load_from_disk
        import gc
        import shutil
        
        print(f"  Large dataset detected - using streaming mode to save RAM...")
        

        safe_name = dataset_name.replace('/', '_')
        cache_dir_name = self.cache_dir / f"{safe_name}_streamed"
        
        if cache_dir_name.exists():
            print(f"  Loading from cache: {cache_dir_name}")
            try:

                ds = load_from_disk(str(cache_dir_name), keep_in_memory=False)
                if self.config.max_samples_per_dataset and len(ds) > self.config.max_samples_per_dataset:
                    print(f"  Limiting cached dataset to {self.config.max_samples_per_dataset} examples")
                    ds = ds.select(range(self.config.max_samples_per_dataset))
                return ds
            except Exception as e:
                print(f"  Error loading cache {e}, re-downloading...")
                shutil.rmtree(str(cache_dir_name), ignore_errors=True)
        

        print(f"  Streaming dataset (memory-efficient)...")
        stream_ds = load_dataset(
            dataset_name,
            cache_dir=str(self.cache_dir),
            split="train",
            streaming=True,
        )
        


        def stream_generator():
            import time
            import sys
            
            count = 0
            is_deepseek = "DeepSeek" in dataset_name
            

            start_time = time.time()
            total_bytes = 0
            

            max_bytes_per_sec = (self.config.max_download_speed_mb * 1024 * 1024) if self.config.max_download_speed_mb else None
            
            if max_bytes_per_sec:
                print(f"  Rate limiting enabled: {self.config.max_download_speed_mb} MB/s")

            for example in stream_ds:
                if self.config.max_samples_per_dataset and count >= self.config.max_samples_per_dataset:
                    break
                

                if max_bytes_per_sec:


                    example_bytes = 0
                    if 'conversations' in example:
                         example_bytes += len(str(example['conversations']))
                    if 'info' in example:
                         example_bytes += len(str(example['info']))
                    
                    total_bytes += example_bytes
                    
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        current_speed = total_bytes / elapsed
                        if current_speed > max_bytes_per_sec:

                            sleep_time = (total_bytes / max_bytes_per_sec) - elapsed
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                

                if is_deepseek:
                    clean_example = {}
                    if 'conversations' in example:
                        clean_convs = []
                        for turn in example['conversations']:
                            clean_turn = {}

                            for k in ['from', 'value', 'role', 'content']:
                                if k in turn:
                                    clean_turn[k] = turn[k]
                            

                            if 'info' in turn and isinstance(turn['info'], dict):
                                safe_info = {}
                                for k in ['think_content', 'answer_content']:
                                    val = turn['info'].get(k)
                                    if val is None:
                                        safe_info[k] = ""
                                    else:
                                        safe_info[k] = str(val)
                                clean_turn['info'] = safe_info
                            
                            clean_convs.append(clean_turn)
                        clean_example['conversations'] = clean_convs
                    
                    yield clean_example
                else:
                    yield example
                
                count += 1
                
        try:
            print(f"  Converting stream directly to Arrow on disk (from_generator)...")
            

            features = stream_ds.features 
            




            ds = Dataset.from_generator(
                stream_generator,
                cache_dir=str(self.cache_dir),
                keep_in_memory=False
            )
            
            print(f"  Saving to reliable cache: {cache_dir_name}")
            ds.save_to_disk(str(cache_dir_name))
            

            del ds
            gc.collect()
            

            print(f"  Reloading from disk...")
            final_ds = load_from_disk(str(cache_dir_name), keep_in_memory=False)
            print(f"  Loaded {len(final_ds):,} examples")
            
            return final_ds
            
        except Exception as e:
            print(f"Error during streaming/processing: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            gc.collect()
    
    def _load_standard_dataset(self, dataset_name: str) -> Optional[Dataset]:
        """Load standard (smaller) datasets normally."""
        dataset = load_dataset(
            dataset_name,
            cache_dir=str(self.cache_dir),
        )
        

        if isinstance(dataset, DatasetDict):
            if 'train' in dataset:
                ds = dataset['train']
            else:

                ds = dataset[list(dataset.keys())[0]]
        else:
            ds = dataset
        
        if self.config.max_samples_per_dataset and len(ds) > self.config.max_samples_per_dataset:
            print(f"  Limiting dataset to {self.config.max_samples_per_dataset} examples")
            ds = ds.select(range(self.config.max_samples_per_dataset))
            
        return ds
    
    def normalize_format(self, dataset: Dataset, source_name: str) -> Dataset:
        """Normalize dataset to a standard conversational format.
        
        Expected output format:
        {
            "conversations": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "...", "thinking": "..."},
            ],
            "source": "dataset_name"
        }
        """
        def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
            conversations = []
            

            if 'conversations' in example:
                convs = example['conversations']
                if isinstance(convs, str):
                    try:
                        convs = json.loads(convs)
                    except:
                        convs = []
                
                for turn in convs:
                    if isinstance(turn, dict):
                        role = turn.get('role', turn.get('from', 'user'))
                        content = turn.get('content', turn.get('value', ''))
                        thinking = turn.get('thinking', turn.get('reasoning', ''))
                        

                        if role in ['human', 'user', 'Human']:
                            role = 'user'
                        elif role in ['assistant', 'gpt', 'Assistant', 'bot']:
                            role = 'assistant'
                        elif role in ['system', 'System']:

                            continue
                        

                        if 'info' in turn and isinstance(turn['info'], dict):
                            info = turn['info']
                            if info.get('think_content'):
                                thinking = info['think_content']
                            if info.get('answer_content'):
                                content = info['answer_content']

                        conv = {'role': role, 'content': content}
                        if thinking:
                            conv['thinking'] = thinking
                        conversations.append(conv)
            
            elif 'messages' in example:
                messages = example['messages']
                if isinstance(messages, str):
                    try:
                        messages = json.loads(messages)
                    except:
                        messages = []
                
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        thinking = msg.get('thinking', msg.get('reasoning', ''))
                        
                        conv = {'role': role, 'content': content}
                        if thinking:
                            conv['thinking'] = thinking
                        conversations.append(conv)
            
            elif 'prompt' in example and 'response' in example:

                conversations = [
                    {'role': 'user', 'content': example['prompt']},
                    {'role': 'assistant', 'content': example['response']},
                ]
                if 'thinking' in example:
                    conversations[-1]['thinking'] = example['thinking']
            
            elif 'instruction' in example and 'output' in example:

                user_content = example['instruction']
                if example.get('input'):
                    user_content = f"{user_content}\n\n{example['input']}"
                
                conversations = [
                    {'role': 'user', 'content': user_content},
                    {'role': 'assistant', 'content': example['output']},
                ]
            
            elif 'question' in example and 'response' in example:


                conversations = [
                    {'role': 'user', 'content': example['question']},
                    {'role': 'assistant', 'content': example['response']},
                ]
            
            return {
                'conversations': conversations,
                'source': source_name,
            }
        

        processed = dataset.map(
            process_example,
            remove_columns=dataset.column_names,
            desc=f"Normalizing {source_name}",
            num_proc=self.config.num_proc,
        )
        

        processed = processed.filter(
            lambda x: len(x['conversations']) >= 2,
            desc="Filtering empty",
        )
        
        return processed
    
    def merge_datasets(self, datasets: List[Dataset]) -> Dataset:
        """Merge all datasets into one."""
        if not datasets:
            raise ValueError("No datasets to merge!")
        
        print(f"\nMerging {len(datasets)} datasets...")
        

        merged = concatenate_datasets(datasets)
        

        merged = merged.shuffle(seed=self.config.seed)
        
        print(f"Total merged examples: {len(merged)}")
        
        return merged
    
    def save_merged(self, dataset: Dataset) -> Path:
        """Save merged dataset to disk."""
        save_path = self.merged_path / "merged_dataset"
        dataset.save_to_disk(str(save_path))
        

        jsonl_path = self.merged_path / "merged_dataset.jsonl"
        with open(jsonl_path, 'w') as f:
            for example in tqdm(dataset, desc="Saving JSONL"):
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {save_path}")
        print(f"JSONL: {jsonl_path}")
        
        return save_path
    
    def create_train_val_split(self, dataset: Dataset) -> DatasetDict:
        """Create train/validation split."""
        split = dataset.train_test_split(
            train_size=self.config.train_split,
            seed=self.config.seed,
        )
        
        return DatasetDict({
            'train': split['train'],
            'validation': split['test'],
        })
    
    def run(self) -> DatasetDict:
        """Run the complete download, normalize, merge pipeline."""

        merged_save = self.merged_path / "merged_dataset"
        if merged_save.exists():
            print(f"Loading existing merged dataset from {merged_save}")
            from datasets import load_from_disk
            merged = load_from_disk(str(merged_save))
            return self.create_train_val_split(merged)
        

        raw_datasets = self.download_datasets()
        

        normalized_datasets = []
        for i, (ds, name) in enumerate(zip(raw_datasets, self.config.datasets)):
            source_name = name.split('/')[-1]
            normalized = self.normalize_format(ds, source_name)
            normalized_datasets.append(normalized)
        

        merged = self.merge_datasets(normalized_datasets)
        

        self.save_merged(merged)
        

        return self.create_train_val_split(merged)


class ConversationTokenizer:
    """Tokenizes conversations with proper label masking.
    
    Key features:
    - Creates input_ids and labels
    - Creates label_mask that masks out USER tokens
    - Model only learns to generate assistant responses (not user prompts!)
    - Handles thinking/reasoning tags properly
    
    Format produced:
    <|im_start|>user
    {user message}<|im_end|>
    <|im_start|>assistant
    <think>{thinking}</think>
    {response}<|im_end|>
    
    Label masking:
    - User turns: label_mask = 0 (don't compute loss)
    - Assistant turns: label_mask = 1 (compute loss)
    - All role tags for user: label_mask = 0
    """
    

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    THINK_START = "<think>"
    THINK_END = "</think>"
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        

        special_tokens = [self.IM_START, self.IM_END, self.THINK_START, self.THINK_END]
        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        if num_added > 0:
            print(f"Added {num_added} special tokens to tokenizer")
        

        self.im_start_id = self.tokenizer.convert_tokens_to_ids(self.IM_START)
        self.im_end_id = self.tokenizer.convert_tokens_to_ids(self.IM_END)
        

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def format_conversation(self, conversations: List[Dict[str, str]]) -> Tuple[str, List[Tuple[int, int]]]:
        """Format conversations into string and track assistant spans.
        
        Returns:
            text: Full formatted conversation text
            assistant_spans: List of (start_char, end_char) for assistant content
        """
        text_parts = []
        assistant_spans = []
        current_pos = 0
        
        for turn in conversations:
            role = turn['role']
            content = turn.get('content', '')
            thinking = turn.get('thinking', '')
            

            turn_start = f"{self.IM_START}{role}\n"
            
            if role == 'assistant':

                assistant_start = current_pos + len(turn_start)
                
                if thinking:
                    turn_content = f"{self.THINK_START}{thinking}{self.THINK_END}\n{content}{self.IM_END}\n"
                else:
                    turn_content = f"{content}{self.IM_END}\n"
                
                assistant_end = current_pos + len(turn_start) + len(turn_content)
                assistant_spans.append((assistant_start, assistant_end))
            else:
                turn_content = f"{content}{self.IM_END}\n"
            
            full_turn = turn_start + turn_content
            text_parts.append(full_turn)
            current_pos += len(full_turn)
        
        return ''.join(text_parts), assistant_spans
    
    def tokenize_with_mask(
        self,
        conversations: List[Dict[str, str]],
    ) -> Dict[str, torch.Tensor]:
        """Tokenize conversations and create label mask.
        
        The label mask ensures:
        1. Loss is only computed on assistant tokens
        2. Model cannot learn to generate "user" role tags
        3. Thinking tokens are included in training
        
        Returns:
            input_ids: Token IDs
            labels: Same as input_ids (for next-token prediction)
            label_mask: 1 for assistant tokens, 0 for user/system tokens
        """
        text, assistant_spans = self.format_conversation(conversations)
        

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True,
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        offsets = encoding['offset_mapping'].squeeze(0)
        

        label_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        
        for i, (start, end) in enumerate(offsets):
            if start == end == 0 and i > 0:

                continue
            

            for span_start, span_end in assistant_spans:
                if start >= span_start and end <= span_end:
                    label_mask[i] = 1.0
                    break
        

        labels = input_ids.clone()
        

        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'label_mask': label_mask,
        }


class TrainingDataset(TorchDataset):
    """PyTorch Dataset for training.
    
    Wraps HuggingFace dataset with proper tokenization.
    Uses fixed shapes for CUDA graph compatibility.
    Caches tokenized data to disk for fast subsequent loads.
    Uses memory mapping to minimize RAM usage.
    """
    
    def __init__(
        self,
        hf_dataset: Dataset,
        tokenizer: ConversationTokenizer,
        cache_path: Optional[str] = None,
        num_proc: int = 8,
    ):
        self.tokenizer = tokenizer
        

        if cache_path and Path(cache_path).exists():
            print(f"Loading tokenized data from cache: {cache_path}")
            from datasets import load_from_disk
            self.dataset = load_from_disk(cache_path)
            self.dataset.set_format(type='torch', columns=['input_ids', 'labels', 'label_mask'])
            print(f"Loaded {len(self.dataset)} cached examples")
            return
        

        def process_batch(batch):
            outputs = {'input_ids': [], 'labels': [], 'label_mask': []}
            
            for conversations in batch['conversations']:
                if len(conversations) < 2:
                    continue
                
                try:
                    tokenized = tokenizer.tokenize_with_mask(conversations)

                    outputs['input_ids'].append(tokenized['input_ids'].numpy())
                    outputs['labels'].append(tokenized['labels'].numpy())
                    outputs['label_mask'].append(tokenized['label_mask'].numpy())
                except Exception:
                    continue
            
            return outputs

        print(f"Tokenizing {len(hf_dataset)} examples...")
        

        safe_num_proc = num_proc
        if safe_num_proc > 1:
            print(f"Using {safe_num_proc} processes for tokenization")
        

        self.dataset = hf_dataset.map(
            process_batch,
            batched=True,
            batch_size=1000,
            num_proc=safe_num_proc,
            writer_batch_size=100,
            remove_columns=hf_dataset.column_names,
            desc="Tokenizing",
        )
        

        self.dataset.set_format(type='torch', columns=['input_ids', 'labels', 'label_mask'])
        print(f"Tokenized {len(self.dataset)} examples")
        

        if cache_path:
            print(f"Saving tokenized dataset to: {cache_path}")
            self.dataset.save_to_disk(cache_path)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[idx]


def create_dataloaders(
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
    batch_size: int = 16,
    max_length: int = 2048,
    num_workers: int = 0,
    max_samples_per_dataset: Optional[int] = None,
    num_proc: int = 32,
    datasets: Optional[List[str]] = None,
    max_download_speed_mb: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Any]:
    """Create train and validation dataloaders.
    
    Args:
        tokenizer_name: HuggingFace tokenizer to use
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length
        num_workers: Number of DataLoader workers
        max_samples_per_dataset: Limit samples per dataset (None for all)
        max_download_speed_mb: Max download speed in MB/s (None for unlimited)
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        hf_tokenizer: The HuggingFace tokenizer (needed for model embedding resize)
    """

    print(f"Loading tokenizer: {tokenizer_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )
    

    conv_tokenizer = ConversationTokenizer(hf_tokenizer, max_length=max_length)
    

    config_args = {
        'max_length': max_length,
        'max_samples_per_dataset': max_samples_per_dataset,
        'num_proc': num_proc,
        'max_download_speed_mb': max_download_speed_mb,
    }
    if datasets:
        config_args['datasets'] = datasets
        
    config = DatasetConfig(**config_args)
    loader = DatasetLoader(config)
    datasets = loader.run()
    

    cache_dir = Path(config.merged_path) / "tokenized_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = str(cache_dir / f"train_{max_length}")
    val_cache = str(cache_dir / f"val_{max_length}")
    
    train_dataset = TrainingDataset(datasets['train'], conv_tokenizer, cache_path=train_cache, num_proc=num_proc)
    val_dataset = TrainingDataset(datasets['validation'], conv_tokenizer, cache_path=val_cache, num_proc=num_proc)
    





    

    use_persistent_workers = num_workers > 0


    prefetch_factor = 4 if num_workers > 0 else None
    


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    
    if num_workers > 0:
        print(f"DataLoader optimizations: persistent_workers=True, prefetch_factor={prefetch_factor}")
    
    return train_loader, val_loader, hf_tokenizer


if __name__ == "__main__":

    print("Testing dataset loading...")
    
    train_loader, val_loader, tokenizer = create_dataloaders(
        batch_size=4,
        max_length=512,
        num_workers=0,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    

    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Label mask shape: {batch['label_mask'].shape}")
    

    print(f"\nSample label mask (first 50 tokens): {batch['label_mask'][0, :50]}")
    print(f"Masked tokens: {batch['label_mask'][0].sum().item():.0f}/{batch['label_mask'][0].numel()}")
