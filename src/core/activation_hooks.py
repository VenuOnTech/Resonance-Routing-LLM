import torch
from torch import nn

class LayerHook:
    """Intercepts and stores forward-pass tensor activations for resonance routing."""
    def __init__(self, target_layer: nn.Module):
        self.activation = None
        self.handle = target_layer.register_forward_hook(self._hook)
        
    def _hook(self, module, inputs, output):
        # HuggingFace layers sometimes return a tuple, sometimes a raw tensor.
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Mean-pool across the sequence length (dim=1 for 3D, dim=0 for 2D)
        if hidden_states.dim() == 3:
            self.activation = hidden_states.mean(dim=1).detach()
        elif hidden_states.dim() == 2:
            self.activation = hidden_states.mean(dim=0).unsqueeze(0).detach()
            
    def remove(self):
        self.handle.remove()