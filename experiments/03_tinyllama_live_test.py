import torch, sys, os
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.resonance_router import ResonanceRouter
from src.core.activation_hooks import LayerHook

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

router = ResonanceRouter()
hook = LayerHook(model.model.layers[2])

# Simulate training activations (replace with real LoRA fine-tuning acts)
dim = model.config.hidden_size
router.calibrate("Python_LoRA", torch.randn(100, dim))
router.calibrate("Medical_LoRA", torch.randn(100, dim))

inputs = tokenizer("Write a Python script for patient data.", return_tensors="pt")
with torch.no_grad(): model(**inputs)

task, _ = router(hook.activation)
print(f"Routed to: {task[0]}")
hook.remove()
