#!/usr/bin/env python3
"""
Convert PyTorch Model to GGUF
=============================
Converts the trained SmallLM model to GGUF format for use with llama.cpp.

Usage:
    python convert_to_gguf.py --model outputs/best_model.pt --output model.gguf
"""

import argparse
import struct
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from model import SmallLM, ModelConfig
from train import TrainingConfig  # Needed for checkpoint deserialization


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little endian
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_BF16 = 30


class GGUFWriter:
    """Writer for GGUF format files."""
    
    def __init__(self, output_path: str, arch: str = "llama"):
        self.output_path = output_path
        self.arch = arch
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, Dict[str, Any]] = {}
        
    def add_metadata(self, key: str, value: Any, value_type: Optional[int] = None):
        """Add metadata key-value pair."""
        self.metadata[key] = (value, value_type)
    
    def add_tensor(self, name: str, tensor: np.ndarray, tensor_type: int = GGML_TYPE_F16):
        """Add a tensor to the GGUF file."""
        self.tensors[name] = {
            "data": tensor,
            "type": tensor_type,
            "shape": tensor.shape,
        }
    
    def _write_string(self, f, s: str):
        """Write a GGUF string."""
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
    
    def _write_value(self, f, value: Any, value_type: Optional[int] = None):
        """Write a typed value."""
        if value_type is None:
            if isinstance(value, bool):
                value_type = GGUF_TYPE_BOOL
            elif isinstance(value, int):
                if value < 0:
                    value_type = GGUF_TYPE_INT64
                else:
                    value_type = GGUF_TYPE_UINT64
            elif isinstance(value, float):
                value_type = GGUF_TYPE_FLOAT32
            elif isinstance(value, str):
                value_type = GGUF_TYPE_STRING
            elif isinstance(value, (list, tuple)):
                value_type = GGUF_TYPE_ARRAY
            else:
                raise ValueError(f"Unknown value type: {type(value)}")
        
        f.write(struct.pack("<I", value_type))
        
        if value_type == GGUF_TYPE_UINT8:
            f.write(struct.pack("<B", value))
        elif value_type == GGUF_TYPE_INT8:
            f.write(struct.pack("<b", value))
        elif value_type == GGUF_TYPE_UINT16:
            f.write(struct.pack("<H", value))
        elif value_type == GGUF_TYPE_INT16:
            f.write(struct.pack("<h", value))
        elif value_type == GGUF_TYPE_UINT32:
            f.write(struct.pack("<I", value))
        elif value_type == GGUF_TYPE_INT32:
            f.write(struct.pack("<i", value))
        elif value_type == GGUF_TYPE_FLOAT32:
            f.write(struct.pack("<f", value))
        elif value_type == GGUF_TYPE_BOOL:
            f.write(struct.pack("<B", 1 if value else 0))
        elif value_type == GGUF_TYPE_STRING:
            self._write_string(f, value)
        elif value_type == GGUF_TYPE_UINT64:
            f.write(struct.pack("<Q", value))
        elif value_type == GGUF_TYPE_INT64:
            f.write(struct.pack("<q", value))
        elif value_type == GGUF_TYPE_FLOAT64:
            f.write(struct.pack("<d", value))
        elif value_type == GGUF_TYPE_ARRAY:
            if len(value) == 0:
                raise ValueError("Empty arrays not supported")
            # Determine element type
            elem = value[0]
            if isinstance(elem, bool):
                elem_type = GGUF_TYPE_BOOL
            elif isinstance(elem, int):
                elem_type = GGUF_TYPE_INT64
            elif isinstance(elem, float):
                elem_type = GGUF_TYPE_FLOAT32
            elif isinstance(elem, str):
                elem_type = GGUF_TYPE_STRING
            else:
                raise ValueError(f"Unknown array element type: {type(elem)}")
            
            f.write(struct.pack("<I", elem_type))
            f.write(struct.pack("<Q", len(value)))
            for elem in value:
                if elem_type == GGUF_TYPE_BOOL:
                    f.write(struct.pack("<B", 1 if elem else 0))
                elif elem_type == GGUF_TYPE_INT64:
                    f.write(struct.pack("<q", elem))
                elif elem_type == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack("<f", elem))
                elif elem_type == GGUF_TYPE_STRING:
                    self._write_string(f, elem)
    
    def _get_type_size(self, tensor_type: int) -> int:
        """Get bytes per element for tensor type."""
        sizes = {
            GGML_TYPE_F32: 4,
            GGML_TYPE_F16: 2,
            GGML_TYPE_BF16: 2,
            GGML_TYPE_Q8_0: 1,  # Approximate
            GGML_TYPE_Q4_0: 0.5,  # Approximate
        }
        return sizes.get(tensor_type, 2)
    
    def _convert_to_dtype(self, tensor: np.ndarray, tensor_type: int) -> np.ndarray:
        """Convert tensor to target dtype."""
        if tensor_type == GGML_TYPE_F32:
            return tensor.astype(np.float32)
        elif tensor_type == GGML_TYPE_F16:
            return tensor.astype(np.float16)
        elif tensor_type == GGML_TYPE_BF16:
            # NumPy doesn't have native bf16, so we do it manually
            # Convert to float32 first, then reinterpret as bf16
            f32 = tensor.astype(np.float32)
            # BF16 is just the top 16 bits of F32
            f32_bits = f32.view(np.uint32)
            bf16_bits = (f32_bits >> 16).astype(np.uint16)
            return bf16_bits
        else:
            return tensor.astype(np.float16)
    
    def write(self):
        """Write the GGUF file."""
        with open(self.output_path, "wb") as f:
            # Write header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))  # n_tensors
            f.write(struct.pack("<Q", len(self.metadata)))  # n_kv
            
            # Write metadata
            for key, (value, value_type) in self.metadata.items():
                self._write_string(f, key)
                self._write_value(f, value, value_type)
            
            # Calculate tensor data offset (must be aligned to 32 bytes)
            tensor_info_size = 0
            for name, info in self.tensors.items():
                tensor_info_size += 8 + len(name.encode("utf-8"))  # string length + string
                tensor_info_size += 4  # n_dims
                tensor_info_size += 8 * len(info["shape"])  # dimensions
                tensor_info_size += 4  # type
                tensor_info_size += 8  # offset
            
            current_pos = f.tell() + tensor_info_size
            padding = (32 - (current_pos % 32)) % 32
            data_start = current_pos + padding
            
            # Calculate tensor offsets
            tensor_offsets = {}
            current_offset = 0
            for name, info in self.tensors.items():
                tensor_offsets[name] = current_offset
                converted = self._convert_to_dtype(info["data"], info["type"])
                current_offset += converted.nbytes
                # Align to 32 bytes
                current_offset = ((current_offset + 31) // 32) * 32
            
            # Write tensor info
            for name, info in self.tensors.items():
                self._write_string(f, name)
                shape = info["shape"]
                f.write(struct.pack("<I", len(shape)))
                for dim in reversed(shape):  # GGUF stores in reverse order
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", info["type"]))
                f.write(struct.pack("<Q", tensor_offsets[name]))
            
            # Write padding
            f.write(b"\x00" * padding)
            
            # Write tensor data
            for name, info in self.tensors.items():
                converted = self._convert_to_dtype(info["data"], info["type"])
                f.write(converted.tobytes())
                # Pad to 32-byte alignment
                pad_size = ((converted.nbytes + 31) // 32) * 32 - converted.nbytes
                f.write(b"\x00" * pad_size)
        
        print(f"GGUF file written to: {self.output_path}")


def map_tensor_name(pytorch_name: str) -> str:
    """Map PyTorch tensor names to GGUF/llama.cpp names."""
    # Standard LLaMA-style naming for llama.cpp
    mappings = {
        "embed_tokens.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    
    if pytorch_name in mappings:
        return mappings[pytorch_name]
    
    # Layer mappings
    # layers.X.attention.wq.weight -> blk.X.attn_q.weight
    # layers.X.attention.wk.weight -> blk.X.attn_k.weight
    # layers.X.attention.wv.weight -> blk.X.attn_v.weight
    # layers.X.attention.wo.weight -> blk.X.attn_output.weight
    # layers.X.feed_forward.w1.weight -> blk.X.ffn_gate.weight
    # layers.X.feed_forward.w2.weight -> blk.X.ffn_down.weight
    # layers.X.feed_forward.w3.weight -> blk.X.ffn_up.weight
    # layers.X.attention_norm.weight -> blk.X.attn_norm.weight
    # layers.X.ffn_norm.weight -> blk.X.ffn_norm.weight
    
    if pytorch_name.startswith("layers."):
        parts = pytorch_name.split(".")
        layer_idx = parts[1]
        
        layer_mappings = {
            "attention.wq.weight": "attn_q.weight",
            "attention.wk.weight": "attn_k.weight",
            "attention.wv.weight": "attn_v.weight",
            "attention.wo.weight": "attn_output.weight",
            "feed_forward.w1.weight": "ffn_gate.weight",
            "feed_forward.w2.weight": "ffn_down.weight",
            "feed_forward.w3.weight": "ffn_up.weight",
            "attention_norm.weight": "attn_norm.weight",
            "ffn_norm.weight": "ffn_norm.weight",
        }
        
        subname = ".".join(parts[2:])
        if subname in layer_mappings:
            return f"blk.{layer_idx}.{layer_mappings[subname]}"
    
    # Unknown name, return as-is with warning
    print(f"  Warning: Unknown tensor name: {pytorch_name}")
    return pytorch_name


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load PyTorch checkpoint."""
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint


def convert_to_gguf(
    model_path: str,
    output_path: str,
    dtype: str = "bf16",
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
):
    """Convert PyTorch model to GGUF format."""
    
    # Load checkpoint
    checkpoint = load_checkpoint(model_path)
    
    # Extract config from checkpoint
    config_data = checkpoint.get("config")
    if config_data is None:
        print("Warning: No config found in checkpoint, using defaults")
        config = ModelConfig()
    else:
        config = ModelConfig(
            vocab_size=getattr(config_data, "vocab_size", 32000),
            dim=getattr(config_data, "dim", 1024),
            n_layers=getattr(config_data, "n_layers", 57),
            n_heads=getattr(config_data, "n_heads", 16),
            n_kv_heads=getattr(config_data, "n_kv_heads", 4),
            intermediate_size=getattr(config_data, "intermediate_size", 2816),
            max_seq_len=getattr(config_data, "max_seq_len", 1024),
            rope_theta=getattr(config_data, "rope_theta", 10000.0),
            rms_norm_eps=getattr(config_data, "rms_norm_eps", 1e-6),
            tie_word_embeddings=getattr(config_data, "tie_word_embeddings", True),
        )
    
    print(f"\nModel Configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  dim: {config.dim}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_kv_heads: {config.n_kv_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  rope_theta: {config.rope_theta}")
    print(f"  head_dim: {config.head_dim}")
    
    # Load tokenizer first to get actual vocab size
    print(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        # Add special tokens exactly as in dataset.py
        special_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        print(f"  Adding special tokens: {special_tokens}")
        num_added = tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        print(f"  Added {num_added} special tokens")
        
        vocab_size = tokenizer.vocab_size
        actual_vocab_size = len(tokenizer)
        print(f"  Tokenizer vocab size: {vocab_size} (len={actual_vocab_size})")
        
        # Override config vocab size if different
        if actual_vocab_size != config.vocab_size:
            print(f"  Warning: Config vocab size ({config.vocab_size}) != Tokenizer size ({actual_vocab_size})")
            
        # Extract tokens
        vocab = tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        id_to_token = {v: k for k, v in vocab.items()}
        
        all_tokens = []
        scores = []
        token_types = []
        
        print("  Extracting vocabulary...")
        for i in range(actual_vocab_size):
            token_str = id_to_token.get(i, "")
            all_tokens.append(token_str)
            scores.append(0.0)
            
            # Mark special tokens
            if token_str in special_tokens:
                token_types.append(3) # CONTROL
            else:
                token_types.append(1) # NORMAL
            
    except Exception as e:
        print(f"Warning: Failed to extract tokenizer vocab: {e}")
        # Fallback to config size if tokenizer fails
        actual_vocab_size = config.vocab_size
        all_tokens = []
        scores = []
        token_types = []
    
    # Get state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Create GGUF writer
    writer = GGUFWriter(output_path, arch="llama")
    
    # Set tensor type based on dtype
    if dtype == "bf16":
        tensor_type = GGML_TYPE_BF16
        print(f"\nOutput dtype: BF16")
    elif dtype == "f16":
        tensor_type = GGML_TYPE_F16
        print(f"\nOutput dtype: F16")
    elif dtype == "f32":
        tensor_type = GGML_TYPE_F32
        print(f"\nOutput dtype: F32")
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
    
    # Add metadata
    writer.add_metadata("general.architecture", "llama")
    writer.add_metadata("general.name", "SmallLM")
    writer.add_metadata("general.quantization_version", 2, GGUF_TYPE_UINT32)
    writer.add_metadata("general.file_type", 1 if dtype == "f16" else (32 if dtype == "bf16" else 0), GGUF_TYPE_UINT32)
    
    # LLaMA architecture parameters
    writer.add_metadata("llama.vocab_size", actual_vocab_size, GGUF_TYPE_UINT32)
    writer.add_metadata("llama.context_length", config.max_seq_len, GGUF_TYPE_UINT32)
    writer.add_metadata("llama.embedding_length", config.dim, GGUF_TYPE_UINT32)
    writer.add_metadata("llama.block_count", config.n_layers, GGUF_TYPE_UINT32)
    writer.add_metadata("llama.feed_forward_length", config.intermediate_size, GGUF_TYPE_UINT32)
    writer.add_metadata("llama.attention.head_count", config.n_heads, GGUF_TYPE_UINT32)
    writer.add_metadata("llama.attention.head_count_kv", config.n_kv_heads, GGUF_TYPE_UINT32)
    writer.add_metadata("llama.rope.dimension_count", config.head_dim, GGUF_TYPE_UINT32)
    writer.add_metadata("llama.attention.layer_norm_rms_epsilon", config.rms_norm_eps, GGUF_TYPE_FLOAT32)
    writer.add_metadata("llama.rope.freq_base", config.rope_theta, GGUF_TYPE_FLOAT32)
    
    # Add tokenizer metadata
    writer.add_metadata("tokenizer.ggml.model", "gpt2") 
    writer.add_metadata("tokenizer.ggml.bos_token_id", 1, GGUF_TYPE_UINT32)
    writer.add_metadata("tokenizer.ggml.eos_token_id", 2, GGUF_TYPE_UINT32) 
    writer.add_metadata("tokenizer.ggml.padding_token_id", 0, GGUF_TYPE_UINT32)
    writer.add_metadata("tokenizer.ggml.pre", "qwen2") # Important for Qwen tokenizer
    
    # Extract real merges
    print("  Extracting merges...")
    merges = []
    try:
        # Save tokenizer to temp location to get merges.txt
        temp_tok_dir = "temp_tokenizer_extract"
        if not os.path.exists(temp_tok_dir):
            tokenizer.save_pretrained(temp_tok_dir)
            
        merges_file = os.path.join(temp_tok_dir, "merges.txt")
        if os.path.exists(merges_file):
            with open(merges_file, "r", encoding="utf-8") as f:
                # Skip version line and read merges
                merges = [line.strip() for line in f.readlines()[1:] if line.strip()]
            print(f"  Loaded {len(merges)} merges")
        else:
            print("  Warning: merges.txt not found, using dummy.")
            merges = ["  t"]
            
    except Exception as e:
        print(f"  Warning: Failed to extract merges: {e}")
        merges = ["  t"]

    writer.add_metadata("tokenizer.ggml.merges", merges)
    
    if all_tokens:
        writer.add_metadata("tokenizer.ggml.tokens", all_tokens)
        writer.add_metadata("tokenizer.ggml.scores", scores)
        writer.add_metadata("tokenizer.ggml.token_type", token_types)

    # Convert and add tensors
    print(f"\nConverting {len(state_dict)} tensors...")
    
    # Track if we need to add output weights separately (for tied embeddings)
    has_lm_head = "lm_head.weight" in state_dict
    embed_weight = None
    
    for name, tensor in state_dict.items():
        # Skip rope buffers (they get recomputed)
        if "rope_" in name:
            continue
        
        # Convert to numpy
        np_tensor = tensor.float().numpy()
        
        # Map name
        gguf_name = map_tensor_name(name)
        
        # Store embed weight for potential tie
        if name == "embed_tokens.weight":
            embed_weight = np_tensor
        
        print(f"  {name} -> {gguf_name} {np_tensor.shape}")
        writer.add_tensor(gguf_name, np_tensor, tensor_type)
    
    # If embeddings are tied and lm_head wasn't in state_dict, add it
    if not has_lm_head and embed_weight is not None and config.tie_word_embeddings:
        print(f"  Adding tied output weights: output.weight {embed_weight.shape}")
        writer.add_tensor("output.weight", embed_weight, tensor_type)
    
    # Write the file
    writer.write()
    
    # Print stats
    file_size = Path(output_path).stat().st_size
    print(f"\nConversion complete!")
    print(f"  Output file: {output_path}")
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to GGUF format")
    parser.add_argument(
        "--model", 
        type=str, 
        default="outputs/best_model.pt",
        help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="model.gguf",
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bf16",
        choices=["f32", "f16", "bf16"],
        help="Output data type (default: bf16 to match training)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Tokenizer name (for metadata)"
    )
    
    args = parser.parse_args()
    
    convert_to_gguf(
        model_path=args.model,
        output_path=args.output,
        dtype=args.dtype,
        tokenizer_name=args.tokenizer,
    )


if __name__ == "__main__":
    main()
