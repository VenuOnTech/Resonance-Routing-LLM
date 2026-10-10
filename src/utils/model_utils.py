import torch
import torch.nn as nn

def get_dynamic_routing_layer(model: nn.Module, depth_pct: float = 0.10) -> nn.Module:
    """
    Dynamically finds the optimal early-stage layer for any causal LLM.
    Uses the 10% depth rule to capture structural intent before deep semantic processing.
    """
    try:
        total_layers = model.config.num_hidden_layers
    except AttributeError:
        # Fallback for models that use different config naming (e.g., older GPT models)
        total_layers = getattr(model.config, 'n_layer', 12)
        
    target_layer_idx = max(1, int(total_layers * depth_pct))
    
    # Try common HuggingFace architecture layer paths
    possible_paths = [
        f"model.layers.{target_layer_idx}",      # Modern: LLaMA, Mistral, TinyLlama, Qwen
        f"transformer.h.{target_layer_idx}",     # Legacy: GPT-2, GPT-Neo, DialoGPT
        f"transformer.layers.{target_layer_idx}" # Alternative standard
    ]
    
    for path in possible_paths:
        try:
            target_layer = model.get_submodule(path)
            print(f"[*] Dynamically attached to {path} (Depth: {target_layer_idx}/{total_layers})")
            return target_layer
        except AttributeError:
            continue
            
    raise ValueError(f"Could not automatically find layer structure for {model.config.model_type}")