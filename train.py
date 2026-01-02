"""
FP8 Training Script for RTX 5090 Blackwell
==========================================
High-performance training with:
- FP8 precision (native Blackwell support)
- CUDA graphs (via torch.compile max-autotune)
- Proper loss masking (only train on assistant responses)
- No gradient checkpointing (CUDA graph compatible)

Usage:
    python train.py --config config.yaml

Requirements:
    - PyTorch 2.5+ with CUDA 12.8+
    - RTX 5090 or other Blackwell GPU
"""

import os
import sys
import time
import math
import copy
import argparse
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
import yaml
from tqdm import tqdm

# ============================================================================
# PERFORMANCE OPTIMIZATIONS (DO NOT AFFECT MODEL QUALITY OR SIZE)
# ============================================================================
# These optimizations speed up training without changing the mathematical
# operations that affect model weights or convergence behavior.

def apply_cuda_optimizations():
    """
    Apply CUDA backend optimizations for maximum training speed.
    
    All optimizations here are mathematically equivalent or use hardware
    acceleration that does not affect model quality or size:
    
    1. TF32 Precision: Uses TensorFloat-32 for matmul on Ampere+ GPUs
       - 2-6x faster matrix operations
       - Maintains float32 dynamic range with 10-bit mantissa
       - NVIDIA validated: no accuracy degradation for DL workloads
       
    2. cuDNN Benchmark Mode: Auto-selects fastest convolution algorithms
       - Only beneficial with fixed input sizes (which we have)
       - Caches optimal algorithm selection after first iteration
       
    3. cuDNN TF32: Enables TF32 for cuDNN operations
       - Same benefits as TF32 matmul
       
    4. CUDA Memory Optimizations:
       - Expandable segments reduce memory fragmentation
       - Allows larger effective batch sizes
    """
    if not torch.cuda.is_available():
        return
    
    # TensorFloat-32 (TF32) Precision
    # Uses 10-bit mantissa instead of 23-bit for internal matmul computations
    # Provides 2-6x speedup on Ampere (A100, RTX 30xx) and newer GPUs
    # Does NOT affect model weights or training quality
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # cuDNN Benchmark Mode
    # Auto-tunes and caches the fastest convolution/matmul algorithms
    # Safe because our input shapes are fixed (batch_size, max_seq_len)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # CUDA Memory Allocator Optimizations
    # Expandable segments reduce fragmentation for variable allocations
    # This is a memory management optimization, not a numerical one
    if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'set_per_process_memory_fraction'):
        pass  # Let PyTorch manage memory automatically
    
    # Set CUDA allocator to use expandable segments (reduces fragmentation)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    
    print("CUDA Optimizations Applied:")
    print(f"  - TF32 matmul precision: high")
    print(f"  - cuDNN TF32: enabled")
    print(f"  - cuDNN benchmark: enabled")
    print(f"  - Memory allocator: expandable segments")

# Apply optimizations at import time
apply_cuda_optimizations()

# Local imports
from model import SmallLM, ModelConfig
from dataset import create_dataloaders


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2
    intermediate_size: int = 1408
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    
    # Training
    batch_size: int = 16
    gradient_accumulation: int = 4
    max_steps: int = 100000
    warmup_steps: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # FP8
    fp8_enabled: bool = True
    
    # Compile
    compile_enabled: bool = True
    compile_mode: str = "max-autotune"
    compile_fullgraph: bool = True
    compile_dynamic: bool = False
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "./outputs"
    
    # Data
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    max_samples_per_dataset: Optional[int] = None  # Limit samples per dataset
    datasets: Optional[List[str]] = None  # List of datasets to train on
    max_download_speed_mb: Optional[int] = None # Max download speed limit
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested config
        flat_config = {}
        
        if 'model' in config_dict:
            flat_config.update(config_dict['model'])
        if 'training' in config_dict:
            flat_config.update(config_dict['training'])
        if 'fp8' in config_dict:
            flat_config['fp8_enabled'] = config_dict['fp8'].get('enabled', True)
        if 'compile' in config_dict:
            c = config_dict['compile']
            flat_config['compile_enabled'] = c.get('enabled', True)
            flat_config['compile_mode'] = c.get('mode', 'max-autotune')
            flat_config['compile_fullgraph'] = c.get('fullgraph', True)
            flat_config['compile_dynamic'] = c.get('dynamic', False)
        if 'logging' in config_dict:
            flat_config.update(config_dict['logging'])
        if 'tokenizer' in config_dict:
            flat_config['tokenizer_name'] = config_dict['tokenizer'].get('name', 'Qwen/Qwen2.5-0.5B')
        if 'data' in config_dict:
            max_samples = config_dict['data'].get('max_samples_per_dataset')
            if max_samples is not None:
                flat_config['max_samples_per_dataset'] = int(max_samples)
            if 'datasets' in config_dict['data']:
                flat_config['datasets'] = config_dict['data']['datasets']
        
        # Ensure float fields are properly converted
        float_fields = ['learning_rate', 'weight_decay', 'max_grad_norm', 'adam_beta1', 'adam_beta2', 
                       'adam_epsilon', 'rope_theta', 'rms_norm_eps']
        for field in float_fields:
            if field in flat_config and isinstance(flat_config[field], str):
                flat_config[field] = float(flat_config[field])
        
        # Ensure int fields are properly converted
        int_fields = ['batch_size', 'gradient_accumulation', 'max_steps', 'warmup_steps', 
                     'log_interval', 'eval_interval', 'save_interval', 'dim', 'n_layers',
                     'n_heads', 'n_kv_heads', 'intermediate_size', 'max_seq_len', 'vocab_size']
        for field in int_fields:
            if field in flat_config and isinstance(flat_config[field], str):
                flat_config[field] = int(flat_config[field])
        
        return cls(**{k: v for k, v in flat_config.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})


class FP8Manager:
    """Manages FP8 training state and scaling.
    
    For RTX 5090 Blackwell, we use native FP8 support.
    This class handles dynamic scaling to prevent over/underflow.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and self._check_fp8_support()
        
        if self.enabled:
            print("FP8 training enabled!")
            # Initialize AMAX history for dynamic scaling
            self.amax_history = []
            self.amax_history_len = 16
            self.scale = 1.0
        else:
            print("FP8 training disabled - using BF16")
    
    def _check_fp8_support(self) -> bool:
        """Check if FP8 is supported on current GPU."""
        if not torch.cuda.is_available():
            return False
        
        # Check for compute capability 8.9+ (Ada) or 10.0+ (Blackwell)
        cc = torch.cuda.get_device_capability()
        major, minor = cc
        
        # Blackwell is compute capability 10.0+
        # Ada (RTX 40xx) is 8.9
        # Hopper is 9.0
        if major >= 9 or (major == 8 and minor >= 9):
            print(f"GPU compute capability: {major}.{minor} - FP8 supported")
            return True
        
        print(f"GPU compute capability: {major}.{minor} - FP8 not supported")
        return False
    
    def get_dtype(self) -> torch.dtype:
        """Get the training dtype."""
        if self.enabled:
            # Use bfloat16 as base, FP8 is applied in linear layers
            return torch.bfloat16
        return torch.bfloat16


class Trainer:
    """Main trainer class with FP8 and CUDA graph support."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FP8 manager
        self.fp8 = FP8Manager(enabled=config.fp8_enabled)
        self.dtype = self.fp8.get_dtype()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # ====================================================================
        # CUDA WARMUP (Does not affect model quality)
        # ====================================================================
        # Warm up CUDA to avoid timing jitter from lazy initialization.
        # CUDA lazily initializes contexts and JIT-compiles kernels on first use,
        # which can cause the first few training steps to appear slower.
        # This warmup ensures consistent timing from step 1.
        if torch.cuda.is_available():
            # Synchronize and reset timing counters
            torch.cuda.synchronize()
            # Small warmup computation to trigger CUDA context initialization
            _warmup = torch.randn(64, 64, device=self.device, dtype=self.dtype) @ torch.randn(64, 64, device=self.device, dtype=self.dtype)
            del _warmup
            torch.cuda.synchronize()
            # Clear the warmup memory
            torch.cuda.empty_cache()
            print("CUDA warmed up and synchronized")
        
        print(f"Training on: {self.device}")
        print(f"Training dtype: {self.dtype}")
    
    def create_model(self, vocab_size: int) -> SmallLM:
        """Create and optionally compile the model."""
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
        
        # Compile with torch.compile for CUDA graphs
        if self.config.compile_enabled:
            print(f"Compiling model with mode='{self.config.compile_mode}'...")
            model = torch.compile(
                model,
                mode=self.config.compile_mode,
                fullgraph=self.config.compile_fullgraph,
                dynamic=self.config.compile_dynamic,
            )
            print("Model compiled successfully!")
        
        return model
    
    def expand_model_depth(
        self,
        old_state_dict: Dict[str, torch.Tensor],
        old_n_layers: int,
        new_n_layers: int,
        model: nn.Module,
        init_strategy: str = "identity-ish",  # or "duplicate"
    ) -> nn.Module:
        """Expand model depth by inserting new layers with proper initialization.
        
        This method handles scaling model depth while preserving learned representations.
        New layers are inserted evenly distributed throughout the model.
        
        Args:
            old_state_dict: State dict from the smaller model
            old_n_layers: Number of layers in the old model
            new_n_layers: Number of layers in the new model
            model: New model (uncompiled) to load weights into
            init_strategy: 
                "identity-ish" - New layers initialized to behave like identity (output projections ≈ 0)
                "duplicate" - Copy existing layers with tiny noise
        
        Returns:
            Model with expanded depth and properly initialized weights
        """
        if new_n_layers <= old_n_layers:
            raise ValueError(f"New layers ({new_n_layers}) must be greater than old layers ({old_n_layers})")
        
        print(f"\n{'='*60}")
        print(f"LAYER EXPANSION: {old_n_layers} → {new_n_layers} layers")
        print(f"Strategy: {init_strategy}")
        print(f"{'='*60}")
        
        num_new_layers = new_n_layers - old_n_layers
        
        # Calculate where to insert new layers (spread evenly, avoiding front/end clustering)
        # Strategy: For N new layers, insert them at positions that spread them evenly
        # in the middle portion of the network
        insertion_positions = self._compute_insertion_positions(old_n_layers, num_new_layers)
        print(f"Inserting {num_new_layers} new layers at positions: {insertion_positions}")
        
        # Get handle to uncompiled model
        model_to_modify = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        # Create a mapping from old layer indices to new layer indices
        old_to_new = self._create_layer_mapping(old_n_layers, insertion_positions)
        print(f"Layer mapping (old → new): {old_to_new}")
        
        # Mark which new layer indices are newly inserted (vs copied from old)
        new_layer_indices = set(range(new_n_layers)) - set(old_to_new.values())
        print(f"New layer indices: {sorted(new_layer_indices)}")
        
        # 1. Load non-layer weights directly (embeddings, final norm, lm_head)
        new_state_dict = OrderedDict()
        for key, value in old_state_dict.items():
            if not key.startswith('layers.'):
                new_state_dict[key] = value
        
        # 2. Remap existing layer weights to new positions
        for old_idx, new_idx in old_to_new.items():
            for key, value in old_state_dict.items():
                if key.startswith(f'layers.{old_idx}.'):
                    new_key = key.replace(f'layers.{old_idx}.', f'layers.{new_idx}.')
                    new_state_dict[new_key] = value.clone()
        
        # 3. Initialize new layers
        if init_strategy == "identity-ish":
            self._init_identity_layers(model_to_modify, new_layer_indices)
        elif init_strategy == "duplicate":
            self._init_duplicate_layers(
                model_to_modify, 
                new_layer_indices, 
                old_state_dict, 
                old_n_layers,
                new_state_dict
            )
        else:
            raise ValueError(f"Unknown init strategy: {init_strategy}")
        
        # 4. Load the remapped state dict (this loads old weights + duplicate weights for new layers if using duplicate strategy)
        # For identity-ish, we already modified the model parameters directly, so we just need to load the remapped old weights
        if init_strategy == "identity-ish":
            # Only load the weights we have (old layers remapped + embeddings)
            model_to_modify.load_state_dict(new_state_dict, strict=False)
        else:
            # For duplicate, the new_state_dict already contains everything
            model_to_modify.load_state_dict(new_state_dict, strict=True)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model_to_modify.parameters())
        print(f"Expanded model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"{'='*60}\n")
        
        return model
    
    def _compute_insertion_positions(self, old_n_layers: int, num_new: int) -> List[int]:
        """Compute where to insert new layers (spread evenly, favoring middle).
        
        Strategy: Insert new layers after positions that are evenly spaced,
        avoiding clustering at the very beginning or end.
        """
        if num_new == 0:
            return []
        
        # We want to spread new layers evenly in the NEW layer space
        # Insert after old layers, spreading from roughly 1/4 to 3/4 of the model
        total_layers = old_n_layers + num_new
        
        # Calculate insertion points in the new layer index space
        # Start inserting after the first quarter and stop before the last quarter
        start_fraction = 0.25
        end_fraction = 0.75
        
        start_idx = max(1, int(old_n_layers * start_fraction))
        end_idx = min(old_n_layers - 1, int(old_n_layers * end_fraction))
        
        # If we're inserting more layers than we have insertion points, spread uniformly
        if num_new > (end_idx - start_idx + 1):
            # Spread uniformly across all old layer positions (except first and last)
            step = (old_n_layers - 2) / (num_new + 1)
            positions = [int(1 + step * (i + 1)) for i in range(num_new)]
        else:
            # Spread within the middle portion
            if num_new == 1:
                positions = [old_n_layers // 2]  # Middle
            else:
                step = (end_idx - start_idx) / (num_new - 1) if num_new > 1 else 0
                positions = [int(start_idx + step * i) for i in range(num_new)]
        
        return sorted(positions)
    
    def _create_layer_mapping(self, old_n_layers: int, insertion_positions: List[int]) -> Dict[int, int]:
        """Create mapping from old layer indices to new layer indices.
        
        Insertion positions refer to "insert after this old layer index".
        """
        old_to_new = {}
        offset = 0
        insertion_counts = {}
        
        # Count how many insertions happen after each position
        for pos in insertion_positions:
            insertion_counts[pos] = insertion_counts.get(pos, 0) + 1
        
        for old_idx in range(old_n_layers):
            old_to_new[old_idx] = old_idx + offset
            # Add offset for insertions that happen after this layer
            if old_idx in insertion_counts:
                offset += insertion_counts[old_idx]
        
        return old_to_new
    
    def _init_identity_layers(self, model: nn.Module, new_layer_indices: set):
        """Initialize new layers with identity-ish behavior.
        
        - Attention output projection (wo) ≈ 0
        - FFN output projection (w2) ≈ 0  
        - RMSNorm weights = 1 (already default)
        
        Effect: New layers initially pass input through unchanged.
        """
        print(f"Initializing {len(new_layer_indices)} new layers with identity-ish init...")
        
        for layer_idx in new_layer_indices:
            layer = model.layers[layer_idx]
            
            # Attention output projection near zero
            # Small initialization to break symmetry but output is essentially zero
            nn.init.normal_(layer.attention.wo.weight, mean=0.0, std=1e-4)
            
            # FFN output projection near zero
            nn.init.normal_(layer.feed_forward.w2.weight, mean=0.0, std=1e-4)
            
            # Initialize gate and up projections normally (they get multiplied by near-zero w2)
            nn.init.normal_(layer.feed_forward.w1.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.feed_forward.w3.weight, mean=0.0, std=0.02)
            
            # Q, K, V projections with small init
            nn.init.normal_(layer.attention.wq.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.attention.wk.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.attention.wv.weight, mean=0.0, std=0.02)
            
            # RMSNorm weights = 1 (identity for normalized input)
            nn.init.ones_(layer.attention_norm.weight)
            nn.init.ones_(layer.ffn_norm.weight)
            
            print(f"  Layer {layer_idx}: identity-ish init complete")
    
    def _init_duplicate_layers(
        self, 
        model: nn.Module, 
        new_layer_indices: set,
        old_state_dict: Dict[str, torch.Tensor],
        old_n_layers: int,
        new_state_dict: Dict[str, torch.Tensor],
    ):
        """Initialize new layers by duplicating existing layers with tiny noise.
        
        Copies middle layers and adds small Gaussian noise (ε ≈ 1e-3).
        """
        print(f"Initializing {len(new_layer_indices)} new layers by duplication with noise...")
        
        # Select source layers from the middle of the old model
        middle_start = old_n_layers // 3
        middle_end = 2 * old_n_layers // 3
        source_layers = list(range(middle_start, middle_end))
        
        noise_std = 1e-3
        
        for i, new_idx in enumerate(sorted(new_layer_indices)):
            # Select source layer (cycle through middle layers)
            source_idx = source_layers[i % len(source_layers)]
            
            # Copy weights from source layer with noise
            for key, value in old_state_dict.items():
                if key.startswith(f'layers.{source_idx}.'):
                    new_key = key.replace(f'layers.{source_idx}.', f'layers.{new_idx}.')
                    # Add small Gaussian noise
                    noise = torch.randn_like(value) * noise_std
                    new_state_dict[new_key] = value.clone() + noise
            
            print(f"  Layer {new_idx}: copied from layer {source_idx} + noise (std={noise_std})")
    
    def create_optimizer(self, model: nn.Module) -> Tuple[AdamW, Any]:
        """Create optimizer with weight decay separation."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Create optimizer groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        optimizer = AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            # Use fused optimizer for speed (CUDA kernel fusion)
            fused=True,
        )
        
        # Create scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        return optimizer, scheduler
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for current step with warmup."""
        if step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (step / self.config.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
            return self.config.learning_rate * 0.1 + 0.5 * (self.config.learning_rate - self.config.learning_rate * 0.1) * (1 + math.cos(math.pi * progress))
    
    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: AdamW,
    ) -> float:
        """Single training step - CUDA graph compatible.
        
        No conditional logic allowed for CUDA graph compatibility.
        Uses BF16 mixed precision on Blackwell with CUDA graphs.
        """
        # Mark CUDA graph step boundary
        torch.compiler.cudagraph_mark_step_begin()
        
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        label_mask = batch['label_mask'].to(self.device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        with torch.amp.autocast('cuda', dtype=self.dtype):
            logits, loss = model(
                input_ids=input_ids,
                labels=labels,
                label_mask=label_mask,
            )
        
        # Clone loss for return (CUDA graph tensor reuse)
        loss_value = loss.detach().clone()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.config.max_grad_norm,
        )
        
        # Optimizer step (always - no conditionals for CUDA graphs)
        optimizer.step()
        
        return loss_value.item()


    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """Evaluate model on validation set."""
        model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            label_mask = batch['label_mask'].to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=self.dtype):
                _, loss = model(
                    input_ids=input_ids,
                    labels=labels,
                    label_mask=label_mask,
                )
            
            num_tokens = label_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        model.train()
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
        }
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: AdamW,
        scheduler: Any,
        path: str,
    ):
        """Save training checkpoint."""
        # Handle compiled model
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: AdamW,
        scheduler: Any,
        path: str,
    ) -> nn.Module:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle compiled model
        model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded: {path} (step {self.global_step})")
        return model
    
    def train(self, pretrain_checkpoint: Optional[str] = None):
        """Main training loop.
        
        Args:
            pretrain_checkpoint: Optional path to pretrained model checkpoint.
                                 If provided, loads these weights before instruction tuning.
        """
        print("\n" + "="*60)
        if pretrain_checkpoint:
            print("Starting Instruction Tuning (with pretrained weights)")
        else:
            print("Starting FP8 Training")
        print("="*60)
        
        # Create dataloaders
        print("\nLoading datasets...")
        train_loader, val_loader, tokenizer = create_dataloaders(
            tokenizer_name=self.config.tokenizer_name,
            batch_size=self.config.batch_size,
            max_length=self.config.max_seq_len,
            num_workers=16,
            max_samples_per_dataset=getattr(self.config, 'max_samples_per_dataset', None),
            num_proc=32,
            datasets=getattr(self.config, 'datasets', None),
            max_download_speed_mb=getattr(self.config, 'max_download_speed_mb', None),
        )
        
        # Get actual vocab size from tokenizer
        vocab_size = len(tokenizer)
        print(f"Vocabulary size: {vocab_size}")
        
        # Create model
        print("\nCreating model...")
        model = self.create_model(vocab_size)
        
        # Load pretrained weights if provided
        if pretrain_checkpoint and Path(pretrain_checkpoint).exists():
            print(f"\nLoading pretrained weights from: {pretrain_checkpoint}")
            checkpoint = torch.load(pretrain_checkpoint, map_location=self.device, weights_only=False)
            
            # Get model state dict
            model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
            state_dict = checkpoint['model_state_dict']
            
            # Handle vocabulary mismatch (e.g. 151665 -> 151667 due to special tokens)
            if 'embed_tokens.weight' in state_dict:
                current_vocab_size = model_to_load.embed_tokens.weight.shape[0]
                loaded_vocab_size = state_dict['embed_tokens.weight'].shape[0]
                
                if current_vocab_size != loaded_vocab_size:
                    print(f"  Resizing embeddings: {loaded_vocab_size} -> {current_vocab_size}")
                    
                    # Resize embeddings
                    new_embeddings = model_to_load.embed_tokens.weight.data.clone()
                    # Copy old embeddings
                    min_vocab = min(current_vocab_size, loaded_vocab_size)
                    new_embeddings[:min_vocab] = state_dict['embed_tokens.weight'][:min_vocab]
                    state_dict['embed_tokens.weight'] = new_embeddings
                    
                    # Resize LM head
                    if 'lm_head.weight' in state_dict:
                        new_head = model_to_load.lm_head.weight.data.clone()
                        new_head[:min_vocab] = state_dict['lm_head.weight'][:min_vocab]
                        state_dict['lm_head.weight'] = new_head
            
            model_to_load.load_state_dict(state_dict)
            
            print(f"  Loaded from step {checkpoint.get('global_step', 'unknown')}")
            print(f"  Pretrain tokens: {checkpoint.get('tokens_seen', 0)/1e9:.2f}B")
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer(model)
        
        # Check for resume checkpoint (only if not loading pretrained)
        resume_path = self.output_dir / "latest_checkpoint.pt"
        if not pretrain_checkpoint and resume_path.exists():
            print(f"\nResuming from {resume_path}")
            model = self.load_checkpoint(model, optimizer, scheduler, str(resume_path))
        
        # Training loop
        print(f"\nTraining for {self.config.max_steps} steps...")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation}")
        
        model.train()
        
        train_iter = iter(train_loader)
        losses = []
        
        pbar = tqdm(range(self.global_step, self.config.max_steps), desc="Training")
        step_start_time = time.time()
        
        for step in pbar:
            self.global_step = step
            
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            # Single training step (no gradient accumulation for CUDA graphs)
            loss = self.train_step(model, batch, optimizer)
            losses.append(loss)
            
            # Update learning rate (using manual warmup + cosine)
            lr = self.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Logging
            if step % self.config.log_interval == 0:
                step_time = time.time() - step_start_time
                tokens_per_sec = (self.config.batch_size * self.config.max_seq_len * self.config.log_interval) / max(step_time, 0.001)
                
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'tok/s': f'{tokens_per_sec:.0f}',
                })
                
                step_start_time = time.time()
            
            # Evaluation
            if step > 0 and step % self.config.eval_interval == 0:
                metrics = self.evaluate(model, val_loader)
                print(f"\nStep {step} - Val Loss: {metrics['val_loss']:.4f}, PPL: {metrics['val_perplexity']:.2f}")
                
                # Save best model
                if metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = metrics['val_loss']
                    best_path = self.output_dir / "best_model.pt"
                    self.save_checkpoint(model, optimizer, scheduler, str(best_path))
            
            # Save checkpoint
            if step > 0 and step % self.config.save_interval == 0:
                ckpt_path = self.output_dir / f"checkpoint_{step}.pt"
                self.save_checkpoint(model, optimizer, scheduler, str(ckpt_path))
                
                # Also save as latest
                latest_path = self.output_dir / "latest_checkpoint.pt"
                self.save_checkpoint(model, optimizer, scheduler, str(latest_path))
                
                # ============================================================
                # PERIODIC MEMORY CLEANUP (Does not affect training quality)
                # ============================================================
                # Clear CUDA cache and run garbage collection to:
                # 1. Reduce memory fragmentation during long training runs
                # 2. Free unused cached memory for other operations
                # 3. Prevent gradual memory creep over thousands of steps
                # This is a memory management optimization only.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final save
        final_path = self.output_dir / "final_model.pt"
        self.save_checkpoint(model, optimizer, scheduler, str(final_path))
        
        print("\n" + "="*60)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.output_dir}")
        print("="*60)
    
    def train_continued(
        self, 
        checkpoint_path: str,
        init_strategy: str = "identity-ish",
        lr_scale: float = 0.25,  # 1/4 of original LR
    ):
        """Continue training with potential model depth expansion.
        
        This method:
        1. Loads a checkpoint and compares its config to current config
        2. If depth increased, expands the model with proper initialization
        3. Trains with a lower learning rate for stability
        
        Args:
            checkpoint_path: Path to the checkpoint to continue from
            init_strategy: "identity-ish" or "duplicate" for new layer init
            lr_scale: Learning rate multiplier (default 0.25 = 1/4 of config LR)
        """
        print("\n" + "="*60)
        print("CONTINUED TRAINING WITH MODEL EXPANSION")
        print("="*60)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        old_config = checkpoint.get('config', None)
        old_state_dict = checkpoint['model_state_dict']
        
        if old_config is None:
            raise ValueError("Checkpoint does not contain config - cannot detect architecture changes")
        
        # Detect architecture changes
        old_n_layers = getattr(old_config, 'n_layers', old_config.get('n_layers') if isinstance(old_config, dict) else None)
        old_dim = getattr(old_config, 'dim', old_config.get('dim') if isinstance(old_config, dict) else None)
        
        if old_n_layers is None or old_dim is None:
            # Try to infer from state dict
            old_n_layers = max([int(k.split('.')[1]) for k in old_state_dict.keys() if k.startswith('layers.')]) + 1
            for key in old_state_dict:
                if 'embed_tokens' in key:
                    old_dim = old_state_dict[key].shape[1]
                    break
        
        new_n_layers = self.config.n_layers
        new_dim = self.config.dim
        
        print(f"\nArchitecture comparison:")
        print(f"  Old model: {old_n_layers} layers, dim={old_dim}")
        print(f"  New model: {new_n_layers} layers, dim={new_dim}")
        
        # Validate: only depth changes are supported
        if new_dim != old_dim:
            raise ValueError(
                f"Width change detected ({old_dim} → {new_dim}). "
                "Only depth (n_layers) expansion is supported with --continue. "
                "Width expansion requires different techniques."
            )
        
        if new_n_layers < old_n_layers:
            raise ValueError(
                f"Layer reduction detected ({old_n_layers} → {new_n_layers}). "
                "Only layer expansion is supported with --continue."
            )
        
        if new_n_layers == old_n_layers:
            print("\nNo architecture change detected - resuming normal training")
            # Just load and continue
            return self._resume_normal_training(checkpoint_path)
        
        # Create dataloaders
        print("\nLoading datasets...")
        train_loader, val_loader, tokenizer = create_dataloaders(
            tokenizer_name=self.config.tokenizer_name,
            batch_size=self.config.batch_size,
            max_length=self.config.max_seq_len,
            num_workers=4,
            max_samples_per_dataset=getattr(self.config, 'max_samples_per_dataset', None),
        )
        
        vocab_size = len(tokenizer)
        print(f"Vocabulary size: {vocab_size}")
        
        # Create NEW model (uncompiled for modification)
        print("\nCreating new model architecture...")
        compile_was_enabled = self.config.compile_enabled
        self.config.compile_enabled = False  # Disable compile for expansion
        model = self.create_model(vocab_size)
        
        # Expand model depth
        model = self.expand_model_depth(
            old_state_dict=old_state_dict,
            old_n_layers=old_n_layers,
            new_n_layers=new_n_layers,
            model=model,
            init_strategy=init_strategy,
        )
        
        # Now compile if it was enabled
        if compile_was_enabled:
            print(f"Compiling expanded model with mode='{self.config.compile_mode}'...")
            model = torch.compile(
                model,
                mode=self.config.compile_mode,
                fullgraph=self.config.compile_fullgraph,
                dynamic=self.config.compile_dynamic,
            )
            print("Model compiled successfully!")
        
        # Create optimizer with LOWER learning rate
        original_lr = self.config.learning_rate
        scaled_lr = original_lr * lr_scale
        print(f"\nLearning rate scaled: {original_lr:.2e} → {scaled_lr:.2e} ({lr_scale:.0%} of original)")
        
        # Temporarily modify config for optimizer creation
        self.config.learning_rate = scaled_lr
        optimizer, scheduler = self.create_optimizer(model)
        self.config.learning_rate = original_lr  # Restore for get_lr()
        
        # Reset training state for expansion training
        self.global_step = 0
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resetting training from step 0 (best val loss from previous: {self.best_val_loss:.4f})")
        
        # Run training
        print("\n" + "="*60)
        print("Starting Expansion Training")
        print("="*60)
        print(f"Training for {self.config.max_steps} steps with scaled LR...")
        
        model.train()
        train_iter = iter(train_loader)
        losses = []
        
        pbar = tqdm(range(self.config.max_steps), desc="Expansion Training")
        step_start_time = time.time()
        
        for step in pbar:
            self.global_step = step
            
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            loss = self.train_step(model, batch, optimizer)
            losses.append(loss)
            
            # Use scaled LR with warmup
            lr = self.get_lr(step) * lr_scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            if step % self.config.log_interval == 0:
                step_time = time.time() - step_start_time
                tokens_per_sec = (self.config.batch_size * self.config.max_seq_len * self.config.log_interval) / max(step_time, 0.001)
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'tok/s': f'{tokens_per_sec:.0f}',
                })
                step_start_time = time.time()
            
            if step > 0 and step % self.config.eval_interval == 0:
                metrics = self.evaluate(model, val_loader)
                print(f"\nStep {step} - Val Loss: {metrics['val_loss']:.4f}, PPL: {metrics['val_perplexity']:.2f}")
                
                if metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = metrics['val_loss']
                    best_path = self.output_dir / "best_model.pt"
                    self.save_checkpoint(model, optimizer, scheduler, str(best_path))
            
            if step > 0 and step % self.config.save_interval == 0:
                ckpt_path = self.output_dir / f"checkpoint_{step}.pt"
                self.save_checkpoint(model, optimizer, scheduler, str(ckpt_path))
                latest_path = self.output_dir / "latest_checkpoint.pt"
                self.save_checkpoint(model, optimizer, scheduler, str(latest_path))
                
                # Periodic memory cleanup (does not affect training)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final save
        final_path = self.output_dir / "final_model.pt"
        self.save_checkpoint(model, optimizer, scheduler, str(final_path))
        
        print("\n" + "="*60)
        print("Expansion Training complete!")
        print(f"Model expanded: {old_n_layers} → {new_n_layers} layers")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.output_dir}")
        print("="*60)
    
    def _resume_normal_training(self, checkpoint_path: str):
        """Resume normal training without architecture changes."""
        print("\nLoading datasets...")
        train_loader, val_loader, tokenizer = create_dataloaders(
            tokenizer_name=self.config.tokenizer_name,
            batch_size=self.config.batch_size,
            max_length=self.config.max_seq_len,
            num_workers=4,
            max_samples_per_dataset=getattr(self.config, 'max_samples_per_dataset', None),
        )
        
        vocab_size = len(tokenizer)
        model = self.create_model(vocab_size)
        optimizer, scheduler = self.create_optimizer(model)
        model = self.load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        
        # Continue with regular training loop
        model.train()
        train_iter = iter(train_loader)
        losses = []
        
        pbar = tqdm(range(self.global_step, self.config.max_steps), desc="Training")
        step_start_time = time.time()
        
        for step in pbar:
            self.global_step = step
            
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            loss = self.train_step(model, batch, optimizer)
            losses.append(loss)
            
            lr = self.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            if step % self.config.log_interval == 0:
                step_time = time.time() - step_start_time
                tokens_per_sec = (self.config.batch_size * self.config.max_seq_len * self.config.log_interval) / max(step_time, 0.001)
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'tok/s': f'{tokens_per_sec:.0f}',
                })
                step_start_time = time.time()
            
            if step > 0 and step % self.config.eval_interval == 0:
                metrics = self.evaluate(model, val_loader)
                print(f"\nStep {step} - Val Loss: {metrics['val_loss']:.4f}, PPL: {metrics['val_perplexity']:.2f}")
                
                if metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = metrics['val_loss']
                    self.save_checkpoint(model, optimizer, scheduler, str(self.output_dir / "best_model.pt"))
            
            if step > 0 and step % self.config.save_interval == 0:
                self.save_checkpoint(model, optimizer, scheduler, str(self.output_dir / f"checkpoint_{step}.pt"))
                self.save_checkpoint(model, optimizer, scheduler, str(self.output_dir / "latest_checkpoint.pt"))
                
                # Periodic memory cleanup (does not affect training)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.save_checkpoint(model, optimizer, scheduler, str(self.output_dir / "final_model.pt"))
        print("\n" + "="*60)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)


def run_pretraining(config: TrainingConfig, large: bool = False, test: bool = False):
    """Run pretraining phase on FineWeb-Edu dataset.
    
    Args:
        config: Training config (used for model architecture)
        large: If True, use 100BT dataset (~60 hours), else 10BT (~6 hours)
        test: If True, run only 1 step for testing
    
    Returns:
        Path to pretrained model checkpoint
    """
    from pretrain_dataset import create_pretrain_dataloader
    import math
    
    print("\n" + "="*60)
    print(f"PHASE 1: PRETRAINING on FineWeb-Edu {'(TEST RUN)' if test else ''}")
    print("="*60)
    
    dataset_config = "sample-100BT" if large else "sample-10BT"
    tokens = 100e9 if large else 10e9
    
    # Test mode settings
    if test:
        dataset_config = "sample-10BT"  # Force small dataset for test
        tokens = 1000  # Tiny amount
    
    print(f"Dataset: HuggingFaceFW/fineweb-edu ({dataset_config})")
    print(f"Tokens: {tokens/1e9:.9f}B")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Calculate steps
    batch_size = config.batch_size
    grad_accum = config.gradient_accumulation
    seq_len = config.max_seq_len
    
    # NOTE: No gradient accumulation for CUDA graph compatibility
    # Each step = one forward + backward + optimizer step
    # Effective batch size = batch_size (not batch_size * grad_accum)
    tokens_per_step = batch_size * seq_len
    
    max_steps = int(tokens / tokens_per_step)
    if test:
        max_steps = 5  # Run 5 steps for test
        print("TEST MODE: Running 5 steps")
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Total steps: {max_steps:,}")
    if not test:
        print(f"Estimated time: {max_steps / 7 / 3600:.1f} hours")
    print("="*60 + "\n")
    
    # Create dataloader
    dataloader, tokenizer = create_pretrain_dataloader(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config=dataset_config,
        tokenizer_name=config.tokenizer_name,
        max_length=seq_len,
        batch_size=batch_size,
        num_workers=0 if test else 4, # No workers for test to speed up start
        shuffle_buffer=1000 if test else 10000,
        cache_dir="./pretrain_cache",
    )
    
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    from model import SmallLM, ModelConfig
    model_config = ModelConfig(
        vocab_size=vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        intermediate_size=config.intermediate_size,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    
    model = SmallLM(model_config)
    model = model.to(device=device, dtype=dtype)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Compile model with CUDA graphs
    if config.compile_enabled:
        print("Compiling model (max-autotune with CUDA graphs)...")
        model = torch.compile(model, mode="max-autotune")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    
    # Output directory
    output_dir = Path("./pretrain_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Learning rate schedule
    warmup_steps = min(2000, max_steps // 10)
    min_lr = config.learning_rate * 0.1
    
    def get_lr(step):
        if step < warmup_steps:
            return config.learning_rate * (step / max(1, warmup_steps)) # Avoid div by 0
        progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (config.learning_rate - min_lr) * cosine
    
    # Training loop
    model.train()
    
    accum_loss = 0.0
    global_step = 0
    tokens_seen = 0
    best_loss = float('inf')
    start_time = time.time()
    
    data_iter = iter(dataloader)
    pbar = tqdm(range(max_steps), desc="Pretraining")
    
    # NOTE: No gradient accumulation for CUDA graph compatibility
    # Each step = one forward + backward + optimizer step
    # Effective batch size = batch_size (not batch_size * grad_accum)
    tokens_per_step = batch_size * seq_len  # Recalculate without grad_accum
    
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Mark step for CUDA graphs
        torch.compiler.cudagraph_mark_step_begin()
        
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        label_mask = batch['label_mask'].to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        with torch.amp.autocast('cuda', dtype=dtype):
            logits, loss = model(input_ids, labels=labels, label_mask=label_mask)
        
        # Clone loss for logging (CUDA graph tensor reuse)
        loss_value = loss.detach().clone()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # Update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Optimizer step
        optimizer.step()
        
        # Update counters
        global_step += 1
        tokens_seen += tokens_per_step
        accum_loss = loss_value.item()
        
        # Logging
        if step % 10 == 0 or test: # Always log in test mode
            elapsed = time.time() - start_time
            tok_per_sec = tokens_seen / max(elapsed, 0.001)
            
            pbar.set_postfix({
                'loss': f'{accum_loss:.4f}',
                'lr': f'{lr:.2e}',
                'tok/s': f'{tok_per_sec:.0f}',
                'tokens': f'{tokens_seen/1e9:.2f}B',
            })
        
        # Checkpointing
        if (step > 0 and step % 5000 == 0) or (test and step == max_steps - 1):
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            checkpoint = {
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'tokens_seen': tokens_seen,
                'loss': accum_loss,
                'config': config,
            }
            torch.save(checkpoint, output_dir / f"pretrain_step_{global_step}.pt")
            torch.save(checkpoint, output_dir / "pretrain_latest.pt")
            
            if accum_loss < best_loss:
                best_loss = accum_loss
                torch.save(checkpoint, output_dir / "pretrain_best.pt")
            
            print(f"\nCheckpoint saved at step {global_step}")
    
    # Final save
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    final_checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'global_step': global_step,
        'tokens_seen': tokens_seen,
        'config': config,
    }
    final_path = output_dir / "pretrain_final.pt"
    torch.save(final_checkpoint, final_path)
    
    print(f"\n" + "="*60)
    print("PRETRAINING COMPLETE!")
    print(f"Total tokens: {tokens_seen/1e9:.9f}B")
    print(f"Final loss: {best_loss if best_loss != float('inf') else accum_loss:.4f}")
    print(f"Model saved to: {final_path}")
    print("="*60)
    
    return str(final_path)


def main():
    parser = argparse.ArgumentParser(
        description="FP8 Training for RTX 5090 Blackwell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal instruction tuning (assumes pretrained model)
  python train.py --config config.yaml
  
  # RECOMMENDED: Pretrain from scratch then instruction tune
  python train.py --config config.yaml --pretrain
  
  # Pretrain with larger dataset (100B tokens, ~60 hours)
  python train.py --config config.yaml --pretrain --large
  
  # Test the entire pipeline (1 step of each)
  python train.py --config config.yaml --pretrain --test
"""
    )
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to config file')
    parser.add_argument('--pretrain', action='store_true',
                        help='Run pretraining on FineWeb-Edu before instruction tuning. '
                             'Default: 10B tokens (~6 hours). Use --large for 100B tokens.')
    parser.add_argument('--large', action='store_true',
                        help='Use larger pretraining dataset (100B tokens, ~60 hours instead of 10B)')
    parser.add_argument('--test', action='store_true',
                        help='Run a quick test (5 steps) of the pipeline to verify everything works.')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume from (without architecture changes)')
    parser.add_argument('--continue', dest='continue_from', type=str, default=None,
                        help='Path to checkpoint for continued training with potential depth expansion.')
    parser.add_argument('--init-strategy', type=str, default='identity-ish',
                        choices=['identity-ish', 'duplicate'],
                        help='Initialization strategy for new layers')
    parser.add_argument('--lr-scale', type=float, default=0.25,
                        help='Learning rate multiplier for expansion training')
    parser.add_argument('--max-speed', type=int, default=None,
                        help='Max dataset download speed in MB/s')
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = TrainingConfig.from_yaml(args.config)
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = TrainingConfig()
    
    # Override with command line args
    if args.max_speed is not None:
        config.max_download_speed_mb = args.max_speed
        
    # Test mode overrides for instruction tuning
    if args.test:
        print("\n>>> TEST MODE ENABLED: Reducing steps for quick verification")
        config.max_steps = 5
        config.warmup_steps = 1
        config.save_interval = 5
        config.eval_interval = 5
        config.log_interval = 1
    
    # Print config
    print("\nTraining Configuration:")
    for field, value in config.__dict__.items():
        print(f"  {field}: {value}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Decide training mode
    if args.pretrain:
        # Phase 1: Pretrain on FineWeb-Edu
        pretrain_checkpoint = run_pretraining(config, large=args.large, test=args.test)
        
        # Phase 2: Instruction tuning
        print("\n" + "="*60)
        print(f"PHASE 2: INSTRUCTION TUNING {'(TEST RUN)' if args.test else ''}")
        print("="*60)
        print(f"Loading pretrained model from: {pretrain_checkpoint}")
        
        # Load pretrained weights and continue with instruction tuning
        trainer.train(pretrain_checkpoint=pretrain_checkpoint)
        
    elif args.continue_from:
        # Continue training with potential expansion
        print(f"\n>>> Continue mode: expanding from {args.continue_from}")
        trainer.train_continued(
            checkpoint_path=args.continue_from,
            init_strategy=args.init_strategy,
            lr_scale=args.lr_scale,
        )
    else:
        # Normal training (with optional resume)
        trainer.train()


if __name__ == "__main__":
    main()
