
import torch
import os
import shutil
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer
# We need to load the checkpoint which contains the config
# The config object is likely just a namespace or dataclass instance,
# but simply loading the dict from pt file is enough if we access attributes.

def export():

    checkpoint_path = "outputs/best_model.pt"
    print(f"Loading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint keys: {checkpoint.keys()}")
    

    t_config = checkpoint['config']
    

    print(f"Loaded config: {t_config}")
    


    vocab_size = getattr(t_config, 'vocab_size', 151667)

    if vocab_size == 32000 and "Qwen" in getattr(t_config, 'tokenizer_name', ""):
        print("Override vocab_size to 151667 for Qwen tokenizer")
        vocab_size = 151667
        
    hf_config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=getattr(t_config, 'dim', 1024),
        intermediate_size=getattr(t_config, 'intermediate_size', 2816),
        num_hidden_layers=getattr(t_config, 'n_layers', 57),
        num_attention_heads=getattr(t_config, 'n_heads', 16),
        num_key_value_heads=getattr(t_config, 'n_kv_heads', 4),
        max_position_embeddings=getattr(t_config, 'max_seq_len', 1024),
        rms_norm_eps=getattr(t_config, 'rms_norm_eps', 1e-6),
        rope_theta=getattr(t_config, 'rope_theta', 10000.0),
        bos_token_id=1, 
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=getattr(t_config, 'tie_word_embeddings', True)
    )
    
    print("Creating Qwen2ForCausalLM...")
    hf_model = Qwen2ForCausalLM(hf_config)
    

    state_dict = checkpoint['model_state_dict']
    

    new_state_dict = {}
    print("Mapping weights...")
    for key, value in state_dict.items():

        new_key = key
        

        if key.startswith("layers."):

            parts = key.split(".")
            idx = parts[1]
            module = parts[2]
            

            sub = ".".join(parts[3:])
            
            prefix = f"model.layers.{idx}"
            
            if module == "attention":
                if "wq" in parts: new_key = f"{prefix}.self_attn.q_proj.weight"
                elif "wk" in parts: new_key = f"{prefix}.self_attn.k_proj.weight"
                elif "wv" in parts: new_key = f"{prefix}.self_attn.v_proj.weight"
                elif "wo" in parts: new_key = f"{prefix}.self_attn.o_proj.weight"
            elif module == "feed_forward":
                if "w1" in parts: new_key = f"{prefix}.mlp.gate_proj.weight"
                elif "w2" in parts: new_key = f"{prefix}.mlp.down_proj.weight"
                elif "w3" in parts: new_key = f"{prefix}.mlp.up_proj.weight"
            elif module == "attention_norm":
                new_key = f"{prefix}.input_layernorm.weight"
            elif module == "ffn_norm":
                new_key = f"{prefix}.post_attention_layernorm.weight"
        
        elif key == "embed_tokens.weight":
            new_key = "model.embed_tokens.weight"
        elif key == "norm.weight":
            new_key = "model.norm.weight"
        elif key == "lm_head.weight":
            new_key = "lm_head.weight"
        elif key == "rope_cos" or key == "rope_sin":
            continue
            

        if new_key in hf_model.state_dict():
            target_shape = hf_model.state_dict()[new_key].shape
            if value.shape != target_shape:
                 print(f"Warning: Shape mismatch for {new_key}: {value.shape} vs {target_shape}")

                 pass
        
        new_state_dict[new_key] = value


    print("Loading mapped state dict into HF model...")
    missing, unexpected = hf_model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {len(missing)}")
    if missing:
        print(f"Sample missing: {missing[:5]}")
        print("Initializing missing keys to zero...")
        for name, param in hf_model.named_parameters():
            if name in missing:
                param.data.zero_()
                
    print(f"Unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"Sample unexpected: {unexpected[:5]}")


    output_dir = "hf_export"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"Saving to {output_dir}...")
    hf_model.save_pretrained(output_dir)
    

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    
    print("Export done.")

if __name__ == "__main__":
    from train import TrainingConfig
    export()
