import torch
from torch import nn

class LayerHook:
    """Intercepts and stores forward-pass tensor activations for resonance routing."""
    def __init__(self, target_layer: nn.Module):
        self.activation = None
        self.handle = target_layer.register_forward_hook(self._hook)
        
    def _hook(self, module, inputs, output):
        # Extract the last token's hidden state: shape (batch_size, hidden_dim)
        # HuggingFace transformers typically return a tuple where output[0] is the hidden state.
        self.activation = output[0][:, -1, :].detach()
        
    def remove(self):
        self.handle.remove()
