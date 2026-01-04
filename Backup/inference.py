"""
Inference Script
================
Run inference with the trained model.

Usage:
    python inference.py --model outputs/best_model.pt --prompt "What is 2+2?"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Generator

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import SmallLM, ModelConfig
from train import TrainingConfig

# Allow unpickling pretrain checkpoints saved from pretrain.py (__main__.PretrainConfig).
class PretrainConfig:
    pass


class InferenceEngine:
    """Inference engine for the trained model."""
    

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    THINK_START = "<think>"
    THINK_END = "</think>"
    
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        

        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        

        special_tokens = [self.IM_START, self.IM_END, self.THINK_START, self.THINK_END]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        

        print(f"Loading model: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, path: str) -> SmallLM:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        

        config_data = checkpoint.get('config')
        if config_data is None:

            model_config = ModelConfig(vocab_size=len(self.tokenizer))
        else:
            model_config = ModelConfig(
                vocab_size=len(self.tokenizer),
                dim=getattr(config_data, 'dim', 512),
                n_layers=getattr(config_data, 'n_layers', 12),
                n_heads=getattr(config_data, 'n_heads', 8),
                n_kv_heads=getattr(config_data, 'n_kv_heads', 2),
                intermediate_size=getattr(config_data, 'intermediate_size', 1408),
                max_seq_len=getattr(config_data, 'max_seq_len', 2048),
            )
        

        model = SmallLM(model_config)
        

        state_dict = checkpoint['model_state_dict']
        if 'embed_tokens.weight' in state_dict:
            current_vocab_size = model.embed_tokens.weight.shape[0]
            loaded_vocab_size = state_dict['embed_tokens.weight'].shape[0]
            if current_vocab_size != loaded_vocab_size:
                print(f"Resizing embeddings: {loaded_vocab_size} -> {current_vocab_size}")
                new_embeddings = model.embed_tokens.weight.data.clone()
                min_vocab = min(current_vocab_size, loaded_vocab_size)
                new_embeddings[:min_vocab] = state_dict['embed_tokens.weight'][:min_vocab]
                state_dict['embed_tokens.weight'] = new_embeddings
                if 'lm_head.weight' in state_dict:
                    new_head = model.lm_head.weight.data.clone()
                    new_head[:min_vocab] = state_dict['lm_head.weight'][:min_vocab]
                    state_dict['lm_head.weight'] = new_head
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        
        return model
    
    def format_prompt(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """Format user input as a chat prompt."""
        parts = []
        
        if system_prompt:
            parts.append(f"{self.IM_START}system\n{system_prompt}{self.IM_END}\n")
        
        parts.append(f"{self.IM_START}user\n{user_input}{self.IM_END}\n")
        parts.append(f"{self.IM_START}assistant\n")
        
        return ''.join(parts)
    
    def extract_response(self, full_text: str, prompt: str) -> str:
        """Extract just the assistant response from generated text."""

        response = full_text[len(prompt):]
        

        if self.IM_END in response:
            response = response.split(self.IM_END)[0]
        
        return response.strip()
    
    def extract_thinking(self, response: str) -> tuple[str, str]:
        """Extract thinking and final response from assistant output."""
        if self.THINK_START in response and self.THINK_END in response:
            start = response.find(self.THINK_START) + len(self.THINK_START)
            end = response.find(self.THINK_END)
            thinking = response[start:end].strip()
            final = response[end + len(self.THINK_END):].strip()
            return thinking, final
        return "", response
    
    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Generate a response with streaming output.
        
        Yields:
            Each token as it's generated.
        """

        formatted_prompt = self.format_prompt(prompt, system_prompt)
        

        input_ids = self.tokenizer.encode(
            formatted_prompt,
            return_tensors='pt',
            add_special_tokens=False,
        ).to(self.device)
        
        eos_token_id = self.tokenizer.convert_tokens_to_ids(self.IM_END)
        

        for _ in range(max_new_tokens):

            idx_cond = input_ids[:, -self.model.config.max_seq_len:]
            

            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = self.model(idx_cond)
            else:
                logits, _ = self.model(idx_cond)
            
            logits = logits[:, -1, :]
            

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
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> dict:
        """Generate a response.
        
        Args:
            stream: If True, returns a generator that yields tokens.
        
        Returns:
            dict with 'thinking' and 'response' keys (or generator if streaming)
        """
        if stream:
            return self.generate_stream(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                system_prompt=system_prompt,
            )
        

        formatted_prompt = self.format_prompt(prompt, system_prompt)
        

        input_ids = self.tokenizer.encode(
            formatted_prompt,
            return_tensors='pt',
            add_special_tokens=False,
        ).to(self.device)
        

        if self.device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.convert_tokens_to_ids(self.IM_END),
                )
        else:
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.tokenizer.convert_tokens_to_ids(self.IM_END),
            )
        

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        

        response = self.extract_response(full_text, formatted_prompt)
        thinking, final_response = self.extract_thinking(response)
        
        return {
            'thinking': thinking,
            'response': final_response,
            'full_output': response,
        }
    
    def chat(self, stream: bool = True):
        """Interactive chat mode.
        
        Args:
            stream: If True, stream tokens as they're generated.
        """
        print("\n" + "="*60)
        print("Interactive Chat Mode")
        print("Type 'quit' to exit, 'clear' to clear history")
        mode_str = "(streaming)" if stream else "(non-streaming)"
        print(f"Mode: {mode_str}")
        print("="*60 + "\n")
        
        system_prompt = "You are a helpful AI assistant. Think step by step before answering."
        
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
            
            if user_input.lower() == 'clear':
                print("History cleared.")
                continue
            
            print("\nAssistant: ", end="", flush=True)
            
            if stream:

                full_response = ""
                for token in self.generate_stream(user_input, system_prompt=system_prompt):
                    print(token, end="", flush=True)
                    full_response += token
                print("\n")
            else:

                result = self.generate(
                    user_input,
                    system_prompt=system_prompt,
                )
                
                if result['thinking']:
                    print(f"\n[Thinking]: {result['thinking'][:200]}...")
                    print(f"\n{result['response']}\n")
                else:
                    print(f"{result['response']}\n")


def main():
    parser = argparse.ArgumentParser(description="Inference with trained model")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen2.5-0.5B', help='Tokenizer name')
    parser.add_argument('--prompt', type=str, default=None, help='Single prompt to process')
    parser.add_argument('--chat', action='store_true', help='Enter interactive chat mode')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--cpu', nargs='?', const=0, type=int, default=None, 
                        help='Force CPU usage. Optionally specify number of threads (e.g. --cpu 8)')
    parser.add_argument('--stream', action='store_true', default=True,
                        help='Enable streaming output (default: enabled)')
    parser.add_argument('--no-stream', action='store_true',
                        help='Disable streaming output')
    args = parser.parse_args()
    

    device_name = None
    if args.cpu is not None:
        device_name = "cpu"
        if args.cpu > 0:
            torch.set_num_threads(args.cpu)
            print(f"Setting CPU threads to {args.cpu}")
        else:
            print(f"Running on CPU with default threads ({torch.get_num_threads()})")
    

    engine = InferenceEngine(
        model_path=args.model,
        tokenizer_name=args.tokenizer,
        device=device_name,
    )
    

    use_stream = args.stream and not args.no_stream
    
    if args.chat:
        engine.chat(stream=use_stream)
    elif args.prompt:
        print("\n" + "="*60)
        
        if use_stream:

            print("[Response]:")
            for token in engine.generate_stream(
                args.prompt,
                max_new_tokens=args.max_tokens,
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
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            
            if result['thinking']:
                print(f"[Thinking]:\n{result['thinking']}\n")
            print(f"[Response]:\n{result['response']}")
            print("="*60)
    else:
        print("Please provide --prompt or use --chat mode")


if __name__ == "__main__":
    main()
