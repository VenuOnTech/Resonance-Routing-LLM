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
        total_layers = getattr(model.config, 'n_layer', 22)
        
    target_layer_idx = max(1, int(total_layers * depth_pct))
    
    # HuggingFace standardizes causal LMs under the 'model.layers' attribute
    try:
        target_layer = model.get_submodule(f"model.layers.{target_layer_idx}")
        print(f"[*] Dynamically attached to Layer {target_layer_idx} (Total: {total_layers})")
        return target_layer
    except AttributeError:
        raise ValueError(f"Could not automatically find layer structure for {model.config.model_type}")