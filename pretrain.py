"""
Pretraining Script for SmallLM
==============================
Train a language model from scratch on web-scale data.

Usage:
    python pretrain.py --config pretrain_config.yaml

Requirements:
    - ~270 GB disk space for sample-100BT
    - RTX 5090 or similar GPU
    - Stable internet connection for streaming
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yaml

import torch
import torch.nn as nn
from torch.optim import AdamW

from model import SmallLM, ModelConfig
from pretrain_dataset import create_pretrain_dataloader


# ============================================================================
# CUDA Optimizations
# ============================================================================
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    print("CUDA Optimizations Applied:")
    print("  - TF32 matmul: enabled")
    print("  - cuDNN TF32: enabled")
    print("  - cuDNN benchmark: enabled")


@dataclass
class PretrainConfig:
    """Pretraining configuration."""
    # Model
    vocab_size: int = 151936  # Qwen tokenizer size
    dim: int = 1280
    n_layers: int = 28
    n_heads: int = 20
    n_kv_heads: int = 5
    intermediate_size: int = 4480
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    
    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-100BT"
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    cache_dir: str = "./pretrain_cache"
    shuffle_buffer: int = 10000
    
    # Training
    batch_size: int = 4
    gradient_accumulation: int = 16  # Effective batch = 64
    max_steps: int = 500000  # ~100B tokens / (64 * 1024) ≈ steps needed
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # FP8/Precision
    dtype: str = "bfloat16"
    compile_model: bool = True
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5000
    output_dir: str = "./pretrain_outputs"
    
    @classmethod
    def from_yaml(cls, path: str) -> "PretrainConfig":
        with open(path) as f:
            config = yaml.safe_load(f)
        
        flat = {}
        for section in config.values():
            if isinstance(section, dict):
                flat.update(section)
            else:
                continue
        
        # Ensure float fields are actually floats (YAML sometimes parses scientific notation as strings)
        float_fields = ['learning_rate', 'min_lr', 'weight_decay', 'max_grad_norm', 
                        'adam_beta1', 'adam_beta2', 'adam_epsilon', 'rope_theta', 'rms_norm_eps']
        for field in float_fields:
            if field in flat and isinstance(flat[field], str):
                flat[field] = float(flat[field])
        
        return cls(**{k: v for k, v in flat.items() if hasattr(cls, k)})


class Pretrainer:
    """Main pretraining class."""
    
    def __init__(self, config: PretrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.tokens_seen = 0
        self.best_loss = float('inf')
        
        # Dtype
        self.dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        
        print(f"Pretraining on: {self.device}")
        print(f"Dtype: {self.dtype}")
    
    def create_model(self, vocab_size: int, compile: bool = True) -> SmallLM:
        """Create model, optionally compile it."""
        model_config = ModelConfig(
            vocab_size=vocab_size,
            dim=self.config.dim,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            n_kv_heads=self.config.n_kv_heads,
            intermediate_size=self.config.intermediate_size,
            max_seq_len=self.config.max_seq_len,
            rope_theta=self.config.rope_theta,
            rms_norm_eps=self.config.rms_norm_eps,
            tie_word_embeddings=self.config.tie_word_embeddings,
        )
        
        model = SmallLM(model_config)
        model = model.to(device=self.device, dtype=self.dtype)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        if compile and self.config.compile_model and torch.cuda.is_available():
            print("Compiling model...")
            model = torch.compile(model, mode="max-autotune")
            print("Model compiled!")
        
        return model
    
    def compile_model(self, model):
        """Compile model after checkpoint loading for CUDA graphs compatibility."""
        if self.config.compile_model and torch.cuda.is_available():
            print("Compiling model...")
            model = torch.compile(model, mode="max-autotune")
            print("Model compiled!")
        return model
    
    def get_lr(self, step: int) -> float:
        """Cosine learning rate with warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step / self.config.warmup_steps)
        
        progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr + (self.config.learning_rate - self.config.min_lr) * cosine
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        print("\n" + "="*60)
        print("PRETRAINING")
        print("="*60)
        print(f"Dataset: {self.config.dataset_name} ({self.config.dataset_config})")
        print(f"Max steps: {self.config.max_steps:,}")
        print(f"Batch size: {self.config.batch_size} x {self.config.gradient_accumulation} = {self.config.batch_size * self.config.gradient_accumulation}")
        print(f"Sequence length: {self.config.max_seq_len}")
        tokens_per_step = self.config.batch_size * self.config.gradient_accumulation * self.config.max_seq_len
        print(f"Tokens per step: {tokens_per_step:,}")
        print(f"Total tokens (estimated): {tokens_per_step * self.config.max_steps / 1e9:.1f}B")
        print("="*60 + "\n")
        
        # Create dataloader
        dataloader, tokenizer = create_pretrain_dataloader(
            dataset_name=self.config.dataset_name,
            dataset_config=self.config.dataset_config,
            tokenizer_name=self.config.tokenizer_name,
            max_length=self.config.max_seq_len,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle_buffer=self.config.shuffle_buffer,
            cache_dir=self.config.cache_dir,
        )
        
        # Create model - always defer compilation, we'll compile after everything is set up
        model = self.create_model(len(tokenizer), compile=False)
        
        # Load checkpoint if resuming (model weights only, before compilation)
        start_step = 0
        if resume_from:
            start_step = self.load_checkpoint(model, resume_from)
            print(f"Resuming training from step {start_step}")
        
        # Now compile model (CUDA graphs will be built with loaded state)
        model = self.compile_model(model)
        
        # Create optimizer AFTER compilation (fresh optimizer, LR scheduler handles correct rate)
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        
        # Training loop
        model.train()
        optimizer.zero_grad()
        
        start_time = time.time()
        
        data_iter = iter(dataloader)
        
        from tqdm import tqdm
        pbar = tqdm(range(start_step, self.config.max_steps), desc="Pretraining", initial=start_step, total=self.config.max_steps)
        
        # Token tracking based on actual batch size (no gradient accumulation for CUDA graphs)
        tokens_per_step = self.config.batch_size * self.config.max_seq_len
        
        for step in pbar:
            # Mark step begin for CUDA graphs - MUST be before any tensor ops
            if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()
            
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart data iterator (epoch boundary)
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            label_mask = batch['label_mask'].to(self.device)
            
            # Forward pass
            with torch.amp.autocast('cuda', dtype=self.dtype):
                logits, loss = model(input_ids, labels=labels, label_mask=label_mask)
            
            # Clone loss for logging before backward
            loss_value = loss.detach().clone().item()
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.max_grad_norm
            )
            
            # Update learning rate
            lr = self.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            self.global_step += 1
            self.tokens_seen += tokens_per_step
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = self.tokens_seen / elapsed
                
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'lr': f'{lr:.2e}',
                    'tok/s': f'{tokens_per_sec:.0f}',
                    'tokens': f'{self.tokens_seen/1e9:.2f}B',
                })
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 1:
                self.save_checkpoint(model, optimizer, loss_value)
        
        # Final save
        self.save_checkpoint(model, optimizer, loss_value, final=True)
        print(f"\nPretraining complete!")
        print(f"Total tokens: {self.tokens_seen/1e9:.2f}B")
        print(f"Model saved to: {self.output_dir}")
    
    def save_checkpoint(self, model, optimizer, loss, final=False):
        """Save checkpoint, keeping only the last 3 step checkpoints."""
        # Get underlying model if compiled
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': self.global_step,
            'tokens_seen': self.tokens_seen,
            'loss': loss,
            'config': self.config,
        }
        
        if final:
            path = self.output_dir / "pretrain_final.pt"
        else:
            path = self.output_dir / f"pretrain_step_{self.global_step}.pt"
        
        torch.save(checkpoint, path)
        print(f"\nCheckpoint saved: {path}")
        
        # Also save latest
        torch.save(checkpoint, self.output_dir / "pretrain_latest.pt")
        
        # Save best
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(checkpoint, self.output_dir / "pretrain_best.pt")
        
        # Cleanup old checkpoints - keep only the last 3 step checkpoints
        if not final:
            import glob
            step_checkpoints = sorted(
                glob.glob(str(self.output_dir / "pretrain_step_*.pt")),
                key=lambda x: int(x.split('_step_')[1].split('.pt')[0])
            )
            # Keep only the last 3
            for old_ckpt in step_checkpoints[:-3]:
                try:
                    os.remove(old_ckpt)
                    print(f"  Removed old checkpoint: {os.path.basename(old_ckpt)}")
                except OSError:
                    pass
    
    def load_checkpoint(self, model, checkpoint_path: str):
        """Load checkpoint and restore training state (model weights only for CUDA graphs compatibility)."""
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Register TrainingConfig as alias for PretrainConfig to handle checkpoints
        # saved by train.py (which uses TrainingConfig class)
        import sys
        sys.modules['__main__'].TrainingConfig = PretrainConfig
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model weights (before compilation)
        model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # Note: We don't load optimizer state - CUDA graphs are incompatible with loaded optimizer tensors
        # The LR scheduler will use the correct learning rate based on step number
        
        self.global_step = checkpoint['global_step']
        self.tokens_seen = checkpoint['tokens_seen']
        self.best_loss = checkpoint.get('loss', float('inf'))
        
        print(f"  Resumed from step: {self.global_step}")
        print(f"  Tokens seen: {self.tokens_seen/1e9:.2f}B")
        print(f"  Loss at checkpoint: {checkpoint.get('loss', 'N/A')}")
        print(f"  Note: Using fresh optimizer (CUDA graphs requirement)")
        
        return self.global_step


def main():
    parser = argparse.ArgumentParser(description="Pretrain SmallLM from scratch")
    parser.add_argument('--config', type=str, default='pretrain_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume from')
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = PretrainConfig.from_yaml(args.config)
    else:
        print(f"Config not found: {args.config}")
        print("Using default config...")
        config = PretrainConfig()
    
    # Run pretraining
    trainer = Pretrainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
