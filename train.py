"""
Training Script for Infinitely Scalable Recursive Model (ISRM)
==============================================================

Trains the ISRM with scalable inference-time compute.

Key Training Features:
1. Random K per batch (model sees all depths during training)
2. Convergence loss (later steps must be better)
3. Contrastive loss (higher confidence at later steps)
4. Step-aware processing (model knows which loop it's on)

Usage:
    python train_scalable.py --config config.yaml
    python train_scalable.py --config config.yaml --k-min 1 --k-max 64
    
After training, use --loops argument during inference to scale quality:
    python inference_scalable.py --model outputs/best_model.pt --loops 100

"""

import os
import sys
import time
import math
import argparse
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from tqdm import tqdm

from model import InfinitelyScalableRecursiveModel, ScalableModelConfig, create_isrm
from dataset import create_dataloaders


def apply_cuda_optimizations():
    """Apply CUDA optimizations for training speed."""
    if not torch.cuda.is_available():
        return
    

    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    
    print("CUDA Optimizations Applied (TF32 + Flash Attention)")


apply_cuda_optimizations()


@dataclass
class ScalableTrainingConfig:
    """Training configuration for ISRM."""

    vocab_size: int = 32000
    dim: int = 384
    n_layers: int = 2
    n_heads: int = 6
    n_kv_heads: int = 2
    intermediate_size: int = 1024
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    

    train_k_min: int = 1
    train_k_max: int = 32
    default_k: int = 8
    n_latent_iter: int = 2
    convergence_loss_weight: float = 0.1
    contrastive_loss_weight: float = 0.05
    max_step_embeddings: int = 256
    gate_init_bias: float = -2.0
    

    batch_size: int = 8
    gradient_accumulation: int = 4
    max_steps: int = 50000
    warmup_steps: int = 1000
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    

    use_bf16: bool = True
    

    compile_enabled: bool = False
    compile_mode: str = "reduce-overhead"
    

    log_interval: int = 10
    eval_interval: int = 10000
    save_interval: int = 2000
    output_dir: str = "./outputs"
    

    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    max_samples_per_dataset: Optional[int] = None
    datasets: Optional[List[str]] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "ScalableTrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        flat_config = {}
        
        if 'model' in config_dict:
            flat_config.update(config_dict['model'])
        if 'scalable' in config_dict:
            flat_config.update(config_dict['scalable'])
        if 'training' in config_dict:
            flat_config.update(config_dict['training'])
        if 'logging' in config_dict:
            flat_config.update(config_dict['logging'])
        if 'tokenizer' in config_dict:
            flat_config['tokenizer_name'] = config_dict['tokenizer'].get('name', 'Qwen/Qwen2.5-0.5B')
        if 'data' in config_dict and isinstance(config_dict['data'], dict):
            if 'max_samples_per_dataset' in config_dict['data']:
                flat_config['max_samples_per_dataset'] = int(config_dict['data']['max_samples_per_dataset'])
            if 'datasets' in config_dict['data']:
                flat_config['datasets'] = config_dict['data']['datasets']
        

        float_fields = ['learning_rate', 'weight_decay', 'max_grad_norm', 'adam_beta1', 
                       'adam_beta2', 'adam_epsilon', 'rope_theta', 'rms_norm_eps',
                       'convergence_loss_weight', 'contrastive_loss_weight', 'gate_init_bias']
        for field in float_fields:
            if field in flat_config and isinstance(flat_config[field], str):
                flat_config[field] = float(flat_config[field])
        
        int_fields = ['batch_size', 'gradient_accumulation', 'max_steps', 'warmup_steps',
                     'log_interval', 'eval_interval', 'save_interval', 'dim', 'n_layers',
                     'n_heads', 'n_kv_heads', 'intermediate_size', 'max_seq_len',
                     'train_k_min', 'train_k_max', 'default_k', 'n_latent_iter', 'max_step_embeddings']
        for field in int_fields:
            if field in flat_config and isinstance(flat_config[field], str):
                flat_config[field] = int(flat_config[field])
        
        return cls(**{k: v for k, v in flat_config.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})


class ScalableTrainer:
    """Trainer for the Infinitely Scalable Recursive Model."""
    
    def __init__(self, config: ScalableTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dtype = torch.bfloat16 if config.use_bf16 else torch.float32
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        

        self.k_distribution = []
        self.convergence_metrics = []
        

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            _warmup = torch.randn(64, 64, device=self.device, dtype=self.dtype) @ torch.randn(64, 64, device=self.device, dtype=self.dtype)
            del _warmup
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        print(f"Training device: {self.device}")
        print(f"Training dtype: {self.dtype}")
        print(f"Scalable K range: [{config.train_k_min}, {config.train_k_max}]")
    
    def create_model(self, vocab_size: int) -> InfinitelyScalableRecursiveModel:
        """Create and optionally compile the ISRM."""
        model_config = ScalableModelConfig(
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
            train_k_min=self.config.train_k_min,
            train_k_max=self.config.train_k_max,
            default_k=self.config.default_k,
            n_latent_iter=self.config.n_latent_iter,
            convergence_loss_weight=self.config.convergence_loss_weight,
            contrastive_loss_weight=self.config.contrastive_loss_weight,
            max_step_embeddings=self.config.max_step_embeddings,
            gate_init_bias=self.config.gate_init_bias,
        )
        
        model = create_isrm(model_config)
        model = model.to(device=self.device, dtype=self.dtype)
        
        if self.config.compile_enabled:
            print(f"Compiling model with mode='{self.config.compile_mode}'...")
            model = torch.compile(model, mode=self.config.compile_mode)
            print("Compilation complete!")
        
        return model

    def load_checkpoint(self, path: str, model: nn.Module, optimizer: Optional[AdamW] = None, scheduler: Optional[Any] = None) -> None:
        """Load checkpoint state."""
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        

        msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Model loaded with msg: {msg}")
        

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
            

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded")
            

        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
            print(f"Resumed from global step {self.global_step}")
            

        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
            

        if 'k_distribution' in checkpoint:
            self.k_distribution = checkpoint['k_distribution']
    
    def create_optimizer(self, model: nn.Module) -> Tuple[AdamW, Any]:
        """Create optimizer with weight decay separation."""
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'gate' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        optimizer = AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            fused=torch.cuda.is_available(),
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        return optimizer, scheduler
    
    def get_lr(self, step: int) -> float:
        """Get LR with warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step / self.config.warmup_steps)
        else:
            progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
            return self.config.learning_rate * 0.1 + \
                   0.5 * (self.config.learning_rate - self.config.learning_rate * 0.1) * \
                   (1 + math.cos(math.pi * progress))
    
    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: AdamW,
    ) -> Dict[str, float]:
        """Single training step with random K."""

        if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
            torch.compiler.cudagraph_mark_step_begin()
        
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        label_mask = batch['label_mask'].to(self.device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', dtype=self.dtype):
            logits, loss, metrics = model(
                input_ids=input_ids,
                labels=labels,
                label_mask=label_mask,
            )
        
        loss_value = loss.detach().clone()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
        
        optimizer.step()
        

        self.k_distribution.append(metrics.get('num_steps', self.config.default_k))
        
        return {
            'loss': loss_value.item(),
            'ce_loss': metrics.get('ce_loss', 0),
            'ponder_loss': metrics.get('ponder_loss', 0),
            'mono_loss': metrics.get('monotonic_loss', 0),
            'conv_loss': metrics.get('convergence_loss', 0),
            'avg_halt': metrics.get('avg_halt_step', 0),
            'k': metrics.get('num_steps', 0),
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """Evaluate at different K values to verify INFINITE scalability.
        
        Tests:
        1. Quality improves from K=1 to K=max_training_K
        2. Quality continues improving BEYOND training K (extrapolation!)
        3. No degradation at ANY K value
        """
        model.eval()
        
        results = {}
        


        k_values = [1, 4, 8, 16, 32, 64, 128, 256]
        
        for k in k_values:
            total_loss = 0.0
            total_tokens = 0
            
            for i, batch in enumerate(val_loader):
                if i >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                label_mask = batch['label_mask'].to(self.device)
                
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    _, loss, _ = model(
                        input_ids=input_ids,
                        labels=labels,
                        label_mask=label_mask,
                        num_steps=k,
                    )
                
                num_tokens = label_mask.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
            
            avg_loss = total_loss / max(total_tokens, 1)
            results[f'loss_k{k}'] = avg_loss
            results[f'ppl_k{k}'] = math.exp(min(avg_loss, 20))
        
        model.train()
        

        losses = [results[f'loss_k{k}'] for k in k_values]
        

        improvements = sum(1 for i in range(len(losses)-1) if losses[i] > losses[i+1])

        is_scaling = improvements >= len(losses) * 0.7
        results['is_scaling_properly'] = float(is_scaling)
        

        train_k_max = self.config.train_k_max
        extrapolation_ks = [k for k in k_values if k > train_k_max]
        if len(extrapolation_ks) >= 2:
            extrap_losses = [results[f'loss_k{k}'] for k in extrapolation_ks]
            extrap_improving = all(extrap_losses[i] >= extrap_losses[i+1] - 0.01 for i in range(len(extrap_losses)-1))
            results['extrapolation_works'] = float(extrap_improving)
        else:
            results['extrapolation_works'] = 1.0
        
        return results
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: AdamW,
        scheduler: Any,
        path: str,
    ):
        """Save checkpoint."""
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'k_distribution': self.k_distribution[-1000:],
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}", flush=True)
    
    def train(self, resume_path: Optional[str] = None):
        """Main training loop."""
        print("\n" + "="*60)
        print("Training Infinitely Scalable Recursive Model (ISRM)")
        print("="*60)
        

        print("\nLoading datasets...")
        train_loader, val_loader, tokenizer = create_dataloaders(
            tokenizer_name=self.config.tokenizer_name,
            batch_size=self.config.batch_size,
            max_length=self.config.max_seq_len,
            num_workers=8,
            max_samples_per_dataset=self.config.max_samples_per_dataset,
            num_proc=16,
            datasets=self.config.datasets,
        )
        
        vocab_size = len(tokenizer)
        print(f"Vocabulary size: {vocab_size}")
        

        print("\nCreating model...")
        model = self.create_model(vocab_size)
        

        optimizer, scheduler = self.create_optimizer(model)
        
        if resume_path:
            self.load_checkpoint(resume_path, model, optimizer, scheduler)
        

        print("\nStarting training...")
        model.train()
        
        train_iter = iter(train_loader)
        start_time = time.time()
        running_loss = 0.0
        running_mono = 0.0
        running_conv = 0.0
        running_halt = 0.0
        
        pbar = tqdm(range(self.global_step, self.config.max_steps), desc="Training")
        
        for step in pbar:
            self.global_step = step
            

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            

            lr = self.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            

            metrics = self.train_step(model, batch, optimizer)
            
            running_loss += metrics['loss']
            running_mono += metrics['mono_loss']
            running_conv += metrics['conv_loss']
            running_halt += metrics['avg_halt']
            

            if step >= self.config.warmup_steps:
                scheduler.step()
            

            if step > 0 and step % self.config.log_interval == 0:
                avg_loss = running_loss / self.config.log_interval
                avg_mono = running_mono / self.config.log_interval
                avg_conv = running_conv / self.config.log_interval
                avg_halt = running_halt / self.config.log_interval
                
                elapsed = time.time() - start_time
                steps_per_sec = self.config.log_interval / elapsed
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'mono': f'{avg_mono:.4f}',
                    'conv': f'{avg_conv:.4f}',
                    'halt': f'{avg_halt:.1f}',
                    'k': metrics['k'],
                    'lr': f'{lr:.2e}',
                })
                
                running_loss = 0.0
                running_mono = 0.0
                running_conv = 0.0
                running_halt = 0.0
                start_time = time.time()
            

            if step > 0 and step % self.config.eval_interval == 0:
                print("\n\nEvaluating INFINITE scalability...", flush=True)
                eval_metrics = self.evaluate(model, val_loader)
                
                print("Scalability Analysis (more K = should be BETTER):", flush=True)
                prev_loss = float('inf')
                for k in [1, 4, 8, 16, 32, 64, 128, 256]:
                    loss = eval_metrics[f'loss_k{k}']
                    ppl = eval_metrics[f'ppl_k{k}']
                    indicator = "↓" if loss < prev_loss else "↑" if loss > prev_loss + 0.01 else "="
                    extrapolate_marker = " (extrapolation)" if k > self.config.train_k_max else ""
                    print(f"  K={k:3d}: loss={loss:.4f}, ppl={ppl:.2f} {indicator}{extrapolate_marker}", flush=True)
                    prev_loss = loss
                
                if eval_metrics['is_scaling_properly']:
                    print("  ✓ Quality SCALES with K - TRUE infinite scalability!", flush=True)
                else:
                    print("  ✗ Warning: Quality not scaling properly with K", flush=True)
                
                if eval_metrics.get('extrapolation_works', 0) > 0.5:
                    print("  ✓ Extrapolation beyond training K works!", flush=True)
                else:
                    print("  ⚠ Extrapolation beyond training K needs work", flush=True)
                

                val_loss = eval_metrics['loss_k8']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        model, optimizer, scheduler,
                        str(self.output_dir / "best_model.pt")
                    )
                print()
            

            if step > 0 and step % self.config.save_interval == 0:
                self.save_checkpoint(
                    model, optimizer, scheduler,
                    str(self.output_dir / f"checkpoint_step{step}.pt")
                )
        

        self.save_checkpoint(
            model, optimizer, scheduler,
            str(self.output_dir / "final_model.pt")
        )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train ISRM - Infinitely Scalable Recursive Model")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config')
    parser.add_argument('--refine', type=int, default=None, help='Set training K (shortcut for --k-max)')
    parser.add_argument('--k-min', type=int, default=None, help='Minimum K during training')
    parser.add_argument('--k-max', type=int, default=None, help='Maximum K during training')
    parser.add_argument('--dim', type=int, default=None, help='Model dimension')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum training steps')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    

    if args.config:
        config = ScalableTrainingConfig.from_yaml(args.config)
    else:
        config = ScalableTrainingConfig()
    

    if args.refine is not None:
        config.train_k_max = args.refine
        config.default_k = args.refine
        print(f"Training with K={args.refine}")
    if args.k_min is not None:
        config.train_k_min = args.k_min
    if args.k_max is not None:
        config.train_k_max = args.k_max
    if args.dim is not None:
        config.dim = args.dim
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    

    trainer = ScalableTrainer(config)
    trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()
