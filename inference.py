"""
Inference Script for Infinitely Scalable Recursive Model (ISRM)
===============================================================

Run inference with configurable number of loops for quality scaling.

The key innovation: Quality scales MONOTONICALLY with compute!
- --loops 1    : Fast but lower quality
- --loops 8    : Good balance (default)
- --loops 32   : High quality
- --loops 100  : Best quality (more compute)
- --loops 500  : Even better (if you have time)

Usage:
    # Quick inference
    python inference_scalable.py --model outputs_scalable/best_model.pt --prompt "Hello" --loops 8
    
    # High quality inference
    python inference_scalable.py --model outputs_scalable/best_model.pt --prompt "Explain quantum computing" --loops 100
    
    # Interactive chat
    python inference_scalable.py --model outputs_scalable/best_model.pt --chat --loops 32
    
    # Debug refinement process (analysis + generate N tokens with greedy refinement)
    python inference_scalable.py --model outputs_scalable/best_model.pt --prompt "Hello" --debug-refinement 10
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Generator, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import InfinitelyScalableRecursiveModel, ScalableModelConfig

# Import training config to enable checkpoint loading
# Note: In the final unified structure, we should move ScalableTrainingConfig to model.py or config utils
# For now, we'll try to import from train.py if needed, or handle it dynamically.
try:
    from train import ScalableTrainingConfig
except ImportError:
    ScalableTrainingConfig = None


class ScalableInferenceEngine:
    """Inference engine for ISRM with configurable loops."""
    

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        device: Optional[str] = None,
        default_loops: int = 8,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.default_loops = default_loops
        

        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        special_tokens = [self.IM_START, self.IM_END]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        

        print(f"Loading model: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Default inference loops: {default_loops}")
        print(f"Device: {self.device}")
    
    def _load_model(self, path: str) -> InfinitelyScalableRecursiveModel:
        """Load ISRM from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        config_data = checkpoint.get('config')
        if config_data is None:
            model_config = ScalableModelConfig(vocab_size=len(self.tokenizer))
        else:

            def get_val(obj, key, default):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            
            model_config = ScalableModelConfig(
                vocab_size=len(self.tokenizer),
                dim=get_val(config_data, 'dim', 256),
                n_layers=get_val(config_data, 'n_layers', 2),
                n_heads=get_val(config_data, 'n_heads', 4),
                n_kv_heads=get_val(config_data, 'n_kv_heads', 2),
                intermediate_size=get_val(config_data, 'intermediate_size', 512),
                max_seq_len=get_val(config_data, 'max_seq_len', 1024),
                train_k_min=get_val(config_data, 'train_k_min', 1),
                train_k_max=get_val(config_data, 'train_k_max', 16),
                default_k=get_val(config_data, 'default_k', 8),
                n_latent_iter=get_val(config_data, 'n_latent_iter', 1),
                max_step_embeddings=get_val(config_data, 'max_step_embeddings', 256),
            )
        
        model = InfinitelyScalableRecursiveModel(model_config)
        
        state_dict = checkpoint['model_state_dict']
        

        if 'embed_tokens.weight' in state_dict:
            current_vocab = model.embed_tokens.weight.shape[0]
            loaded_vocab = state_dict['embed_tokens.weight'].shape[0]
            if current_vocab != loaded_vocab:
                print(f"Resizing embeddings: {loaded_vocab} -> {current_vocab}")
                new_emb = model.embed_tokens.weight.data.clone()
                min_vocab = min(current_vocab, loaded_vocab)
                new_emb[:min_vocab] = state_dict['embed_tokens.weight'][:min_vocab]
                state_dict['embed_tokens.weight'] = new_emb
                if 'lm_head.weight' in state_dict:
                    new_head = model.lm_head.weight.data.clone()
                    new_head[:min_vocab] = state_dict['lm_head.weight'][:min_vocab]
                    state_dict['lm_head.weight'] = new_head
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        return model
    
    def format_prompt(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """Format as chat prompt."""
        parts = []
        if system_prompt:
            parts.append(f"{self.IM_START}system\n{system_prompt}{self.IM_END}\n")
        parts.append(f"{self.IM_START}user\n{user_input}{self.IM_END}\n")
        parts.append(f"{self.IM_START}assistant\n")
        return ''.join(parts)
    
    def extract_response(self, full_text: str, prompt: str) -> str:
        """Extract assistant response."""
        response = full_text[len(prompt):]
        if self.IM_END in response:
            response = response.split(self.IM_END)[0]
        return response.strip()
    
    @torch.no_grad()
    def analyze_refinement(
        self,
        prompt: str,
        max_loops: int = 64,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze how quality improves with more loops.
        
        This is useful for understanding the model's scaling behavior.
        Returns metrics at each loop count INCLUDING actual CE loss.
        """
        formatted = self.format_prompt(prompt, system_prompt)
        input_ids = self.tokenizer.encode(formatted, return_tensors='pt', add_special_tokens=False).to(self.device)
        
        bsz, seqlen = input_ids.shape
        

        labels = input_ids.clone()
        
        inputs = self.model.embed_tokens(input_ids)
        outputs, latents = self.model.get_initial_states(bsz, seqlen, self.device)
        
        results = []
        
        for step in range(1, max_loops + 1):
            outputs, latents, metrics = self.model.single_refinement_step(inputs, outputs, latents, step)
            

            all_logits = self.model.lm_head(outputs)
            

            shift_logits = all_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            ).item()
            

            logits = all_logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            top_prob, top_idx = probs.max(dim=-1)
            top_token = self.tokenizer.decode([top_idx.item()])
            

            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).item()
            

            latent_alpha = metrics.get('latent_alpha', 0)
            output_alpha = metrics.get('output_alpha', 0)
            if hasattr(latent_alpha, 'item'):
                latent_alpha = latent_alpha.item()
            if hasattr(output_alpha, 'item'):
                output_alpha = output_alpha.item()
            
            results.append({
                'step': step,
                'top_token': top_token,
                'top_token_id': top_idx.item(),
                'confidence': top_prob.item(),
                'entropy': entropy,
                'loss': ce_loss,
                'latent_alpha': latent_alpha,
                'output_alpha': output_alpha,
            })
        
        return {
            'prompt': prompt,
            'refinement_steps': results,
            'summary': {
                'initial_entropy': results[0]['entropy'],
                'final_entropy': results[-1]['entropy'],
                'entropy_reduction': results[0]['entropy'] - results[-1]['entropy'],
                'initial_confidence': results[0]['confidence'],
                'final_confidence': results[-1]['confidence'],
                'confidence_gain': results[-1]['confidence'] - results[0]['confidence'],
                'initial_loss': results[0]['loss'],
                'final_loss': results[-1]['loss'],
                'loss_reduction': results[0]['loss'] - results[-1]['loss'],
            }
        }
    
    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        num_loops: int = None,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Generate with streaming output."""
        if num_loops is None:
            num_loops = self.default_loops
        
        formatted = self.format_prompt(prompt, system_prompt)
        input_ids = self.tokenizer.encode(formatted, return_tensors='pt', add_special_tokens=False).to(self.device)
        
        eos_token_id = self.tokenizer.convert_tokens_to_ids(self.IM_END)
        
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.model.config.max_seq_len:]
            bsz, seqlen = idx_cond.shape
            

            inputs = self.model.embed_tokens(idx_cond)
            outputs, latents = self.model.get_initial_states(bsz, seqlen, self.device)
            

            for step in range(1, num_loops + 1):
                outputs, latents, _ = self.model.single_refinement_step(inputs, outputs, latents, step)
            
            logits = self.model.lm_head(outputs)[:, -1, :]
            

            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token_id in set(input_ids[i].tolist()):
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= repetition_penalty
                        else:
                            logits[i, token_id] *= repetition_penalty
            
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            
            yield token_text
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        num_loops: int = None,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, str]:
        """Generate a response."""
        if stream:
            return self.generate_stream(
                prompt, max_new_tokens, num_loops,
                temperature, top_k, top_p,
                system_prompt=system_prompt
            )
        
        if num_loops is None:
            num_loops = self.default_loops
        
        formatted = self.format_prompt(prompt, system_prompt)
        input_ids = self.tokenizer.encode(formatted, return_tensors='pt', add_special_tokens=False).to(self.device)
        
        if self.device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    num_loops=num_loops,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.convert_tokens_to_ids(self.IM_END),
                )
        else:
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_loops=num_loops,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.tokenizer.convert_tokens_to_ids(self.IM_END),
            )
        
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        response = self.extract_response(full_text, formatted)
        
        return {'response': response, 'num_loops': num_loops}

    @torch.no_grad()
    def generate_debug_refinement(
        self,
        prompt: str,
        max_new_tokens: int = 1,
        num_loops: int = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate with greedy refinement for a fixed token budget."""
        if num_loops is None:
            num_loops = self.default_loops

        formatted = self.format_prompt(prompt, system_prompt)
        input_ids = self.tokenizer.encode(formatted, return_tensors='pt', add_special_tokens=False).to(self.device)
        generated_tokens = []

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.model.config.max_seq_len:]
            bsz, seqlen = idx_cond.shape

            inputs = self.model.embed_tokens(idx_cond)
            outputs, latents = self.model.get_initial_states(bsz, seqlen, self.device)

            for step in range(1, num_loops + 1):
                outputs, latents, _ = self.model.single_refinement_step(inputs, outputs, latents, step)

            logits = self.model.lm_head(outputs)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            generated_tokens.append(token_text)

        return ''.join(generated_tokens).strip()
    
    def chat(self, num_loops: int = None, stream: bool = True):
        """Interactive chat mode."""
        if num_loops is None:
            num_loops = self.default_loops
        
        print("\n" + "="*60)
        print("ISRM Interactive Chat - Infinitely Scalable Recursive Model")
        print(f"Using {num_loops} refinement loops per token")
        print("Type 'quit' to exit, 'loops N' to change loop count")
        print("="*60 + "\n")
        
        system_prompt = "You are a helpful AI assistant."
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower().startswith('loops '):
                try:
                    new_loops = int(user_input.split()[1])
                    num_loops = new_loops
                    print(f"Now using {num_loops} refinement loops")
                    continue
                except:
                    print("Usage: loops <number>")
                    continue
            
            print("\nAssistant: ", end="", flush=True)
            
            if stream:
                for token in self.generate_stream(user_input, num_loops=num_loops, system_prompt=system_prompt):
                    print(token, end="", flush=True)
                print("\n")
            else:
                result = self.generate(user_input, num_loops=num_loops, system_prompt=system_prompt)
                print(f"{result['response']}\n")


def main():
    parser = argparse.ArgumentParser(description="Inference with ISRM - Infinitely Scalable Recursive Model")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen2.5-0.5B', help='Tokenizer name')
    parser.add_argument('--prompt', type=str, default=None, help='Single prompt to process')
    parser.add_argument('--chat', action='store_true', help='Interactive chat mode')
    parser.add_argument('--loops', type=int, default=8, help='Number of refinement loops (quality scaling!)')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--stream', action='store_true', default=True, help='Enable streaming')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming')
    parser.add_argument(
        '--debug-refinement',
        type=int,
        nargs='?',
        const=1,
        default=0,
        metavar='N',
        help='Debug refinement: show loop analysis with N-token outputs per step (default N=1)',
    )
    args = parser.parse_args()
    
    device = "cpu" if args.cpu else None
    
    engine = ScalableInferenceEngine(
        model_path=args.model,
        tokenizer_name=args.tokenizer,
        device=device,
        default_loops=args.loops,
    )
    
    use_stream = args.stream and not args.no_stream
    
    if args.debug_refinement and args.prompt:
        print("\n" + "="*70)
        print("Refinement Analysis (ISRM)")
        print("="*70)

        print(f"\nPrompt: {args.prompt}\n")

        max_loops = args.loops
        analysis = engine.analyze_refinement(args.prompt, max_loops=max_loops)

        print("Step |   Loss   | Entropy | Confid | Output")
        print("-" * 70)

        for r in analysis['refinement_steps']:
            sample = engine.generate_debug_refinement(
                args.prompt,
                max_new_tokens=args.debug_refinement,
                num_loops=r['step'],
            )
            print(f"{r['step']:4d} | {r['loss']:8.4f} | {r['entropy']:7.4f} | {r['confidence']:.4f} | {sample}")

        print("\nSummary:")
        print(f"  Loss reduction: {analysis['summary']['loss_reduction']:.4f} ({analysis['summary']['initial_loss']:.4f} -> {analysis['summary']['final_loss']:.4f})")
        print(f"  Entropy reduction: {analysis['summary']['entropy_reduction']:.4f}")
        print(f"  Confidence gain: {analysis['summary']['confidence_gain']:.4f}")
        

        losses = [r['loss'] for r in analysis['refinement_steps']]
        is_monotonic = all(losses[i] >= losses[i+1] - 0.01 for i in range(len(losses)-1))
        if is_monotonic:
            print("  ✓ Loss MONOTONICALLY DECREASING - TRUE ISRM!")
        else:
            degradations = sum(1 for i in range(len(losses)-1) if losses[i+1] > losses[i] + 0.01)
            print(f"  ✗ Loss degraded {degradations} times")

        print("="*70)
        
    elif args.chat:
        engine.chat(num_loops=args.loops, stream=use_stream)
        
    elif args.prompt:
        print("\n" + "="*60)
        print(f"Generating with {args.loops} refinement loops...")
        print("="*60)
        
        if use_stream:
            print("\n[Response]:")
            for token in engine.generate_stream(
                args.prompt,
                max_new_tokens=args.max_tokens,
                num_loops=args.loops,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            ):
                print(token, end="", flush=True)
            print("\n" + "="*60)
        else:
            result = engine.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                num_loops=args.loops,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            print(f"\n[Response]:\n{result['response']}")
            print("="*60)
    else:
        print("Please provide --prompt or use --chat mode")


if __name__ == "__main__":
    main()
