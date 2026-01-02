"""
Modern Small Transformer Model
==============================
A ~50M parameter LLaMA-style model with modern features:
- RoPE (Rotary Positional Embeddings)
- GQA (Grouped Query Attention)
- RMSNorm (Root Mean Square Normalization)
- SwiGLU (SiLU Gated Linear Unit)

Designed for FP8 training on RTX 5090 Blackwell with CUDA graphs.
No gradient checkpointing - keeps CUDA graphs clean.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2  # GQA
    intermediate_size: int = 1408
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads
    
    @property
    def n_rep(self) -> int:
        """Number of repetitions for GQA."""
        return self.n_heads // self.n_kv_heads


class RMSNorm(nn.Module):
    """Root Mean Square Normalization.
    
    More computationally efficient than LayerNorm and provides
    better numerical stability for FP8 training.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos and sin tensors.
    
    Returns (cos, sin) tensors for rotation - no complex numbers for CUDA graph compatibility.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Return cos and sin separately (no complex numbers)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors.
    
    Uses real-valued operations for CUDA graph compatibility.
    """
    # xq, xk: (B, S, H, D)
    # cos, sin: (S, D/2)
    
    # Split into even and odd components
    xq_r = xq[..., ::2]  # (B, S, H, D/2)
    xq_i = xq[..., 1::2]  # (B, S, H, D/2)
    xk_r = xk[..., ::2]
    xk_i = xk[..., 1::2]
    
    # Broadcast cos/sin: (S, D/2) -> (1, S, 1, D/2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # Apply rotation using real arithmetic
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos
    
    # Interleave back
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)
    
    return xq_out, xk_out



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for Grouped Query Attention.
    
    Takes (B, S, n_kv_heads, head_dim) -> (B, S, n_heads, head_dim)
    """
    if n_rep == 1:
        return x
    b, s, n_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(b, s, n_kv_heads, n_rep, head_dim)
    return x.reshape(b, s, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    """Multi-Head Attention with Grouped Query Attention (GQA).
    
    Uses fewer KV heads than Q heads for memory efficiency.
    CUDA graph compatible - no dynamic operations.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_rep
        self.head_dim = config.head_dim
        
        # Q, K, V projections
        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        
        # Project to Q, K, V
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE (using real-valued cos/sin)
        xq, xk = apply_rope(xq, xk, cos, sin)
        
        # Repeat K, V for GQA
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        
        # Transpose for attention: (B, H, S, D)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Scaled dot-product attention (uses Flash Attention if available)
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=mask is None,  # Use causal mask if no explicit mask
        )
        
        # Reshape back: (B, H, S, D) -> (B, S, H*D)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.
    
    Uses SiLU activation with gating for better performance.
    More efficient than standard FFN + ReLU.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.intermediate_size
        
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)  # Down
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)  # Up
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization.
    
    Architecture:
    - Pre-RMSNorm -> Attention -> Residual
    - Pre-RMSNorm -> SwiGLU FFN -> Residual
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with residual
        x = x + self.attention(self.attention_norm(x), cos, sin, mask)
        # FFN with residual
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class SmallLM(nn.Module):
    """Language Model (~800M parameters, configurable).
    
    A modern, efficient transformer designed for:
    - FP8 training on RTX 5090 Blackwell
    - CUDA graph compatibility
    - High-quality distillation from larger models
    
    Features:
    - RoPE for positional encoding
    - GQA for memory efficiency
    - RMSNorm for stability
    - SwiGLU for better FFN
    - Tied embeddings for parameter efficiency
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final norm and output projection
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Pre-compute RoPE cos/sin (registered as buffers for CUDA graph compatibility)
        cos, sin = precompute_rope_freqs(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        self._init_scaled_weights()
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled initialization.
        
        Using proper transformer initialization:
        - Linear layers: normal init with std=0.02
        - Embeddings: normal init with std=0.02 (standard for LLMs)
        """
        if isinstance(module, nn.Linear):
            # Standard transformer init
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_scaled_weights(self):
        """Apply scaling to residual output projections for deep networks.
        
        Uses a gentler scaling factor that prevents vanishing gradients:
        scale = sqrt(2 / n_layers) instead of 1 / sqrt(2 * n_layers)
        
        This keeps gradient flow healthy while still preventing explosion.
        """
        # Gentler scaling: sqrt(2/n) instead of 1/sqrt(2n)
        # For 32 layers: sqrt(2/32) ≈ 0.25 vs original 1/sqrt(64) ≈ 0.125
        scale_factor = math.sqrt(2.0 / self.config.n_layers)
        
        for layer in self.layers:
            # Scale attention output projection
            layer.attention.wo.weight.data.mul_(scale_factor)
            # Scale FFN output projection  
            layer.feed_forward.w2.weight.data.mul_(scale_factor)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (B, S)
            attention_mask: Optional attention mask (B, S) - not typically needed with causal attention
            labels: Target token IDs for loss computation (B, S)
            label_mask: Mask for which positions to compute loss on (B, S)
                       1 = compute loss, 0 = ignore (e.g., user prompts)
        
        Returns:
            logits: Output logits (B, S, V)
            loss: Cross-entropy loss if labels provided
        """
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        h = self.embed_tokens(input_ids)
        
        # Get position frequencies for this sequence length
        cos = self.rope_cos[:seqlen].to(device)
        sin = self.rope_sin[:seqlen].to(device)
        
        # Process through transformer blocks
        for layer in self.layers:
            h = layer(h, cos, sin)
        
        # Final norm and project to vocabulary
        h = self.norm(h)
        logits = self.lm_head(h)
        
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss computation
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            if label_mask is not None:
                # Only compute loss on masked positions (assistant responses)
                shift_mask = label_mask[..., 1:].contiguous().view(-1)
                
                # Compute loss for all positions
                loss = F.cross_entropy(
                    shift_logits,
                    shift_labels,
                    reduction='none'
                )
                
                # Apply mask and compute mean only over non-masked positions
                # Use torch.where to avoid data-dependent branching (CUDA graph compatible)
                masked_loss = loss * shift_mask
                num_tokens = shift_mask.sum() + 1e-8  # Add epsilon to avoid division by zero
                loss = masked_loss.sum() / num_tokens
            else:
                loss = F.cross_entropy(shift_logits, shift_labels)
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """Generate text autoregressively.
        
        Simple generation without KV cache for CUDA graph compatibility.
        """
        for _ in range(max_new_tokens):
            # Truncate if exceeding max length
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token_id in set(input_ids[i].tolist()):
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= repetition_penalty
                        else:
                            logits[i, token_id] *= repetition_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def count_parameters(self) -> int:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(config: Optional[ModelConfig] = None) -> SmallLM:
    """Create a model instance with default or custom config."""
    if config is None:
        config = ModelConfig()
    
    model = SmallLM(config)
    
    # Print model stats
    total, trainable = model.count_parameters()
    print(f"Model created:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Size estimate: {total * 4 / 1024 / 1024:.1f} MB (FP32)")
    print(f"  Size estimate: {total * 1 / 1024 / 1024:.1f} MB (FP8)")
    
    return model


if __name__ == "__main__":
    # Test model creation
    config = ModelConfig()
    model = create_model(config)
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    x = torch.randint(0, config.vocab_size, (2, 128), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 128), device=device)
    label_mask = torch.ones(2, 128, device=device)
    label_mask[:, :64] = 0  # Simulate masking user tokens
    
    logits, loss = model(x, labels=labels, label_mask=label_mask)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
