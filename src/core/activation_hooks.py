import torch
from torch import nn

class LayerHook:
    """Intercepts and stores forward-pass tensor activations for resonance routing."""
    def __init__(self, target_layer: nn.Module):
        self.activation = None
        self.handle = target_layer.register_forward_hook(self._hook)
        
    def _hook(self, module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        if hidden_states.dim() == 3:
            self.activation = hidden_states[:, -1, :].detach()
        elif hidden_states.dim() == 2:
            self.activation = hidden_states[-1, :].unsqueeze(0).detach()
            
    def remove(self):
        self.handle.remove()