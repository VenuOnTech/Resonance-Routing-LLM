import torch, os, sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.activation_hooks import LayerHook
from src.utils.model_utils import get_dynamic_routing_layer

# You can now change this to any causal LLM (e.g., "meta-llama/Meta-Llama-3-8B")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Dynamically attach to the correct layer
target_layer = get_dynamic_routing_layer(model)
hook = LayerHook(target_layer)

D = model.config.hidden_size
print(f"[*] Model Hidden Dimension: {D}")

def extract_structural_fingerprints(dataset_name, text_column, n_samples=2000):
    print(f"\nStreaming {n_samples} samples from {dataset_name}...")
    data = load_dataset(dataset_name, split="train", streaming=True)
    
    activations = []
    count = 0
    if n_samples < D:
        print(f"WARNING: You requested {n_samples} samples, but the model dimension is {D}.")
        print("Using pseudo-inverse approximation. For full-rank precision, increase n_samples.")
    for row in data:
        if count >= n_samples: break
        
        inputs = tokenizer(row[text_column], return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad(): 
            model(**inputs)
        
        activations.append(hook.activation)
        count += 1
        
        # Added a progress tracker so you know it hasn't crashed
        if count % 10 == 0:
            print(f"Processed {count}/{n_samples} prompts...")
        
    return torch.cat(activations, dim=0)

python_acts = extract_structural_fingerprints("iamtarun/python_code_instructions_18k_alpaca", "prompt")
medical_acts = extract_structural_fingerprints("medalpaca/medical_meadow_medical_flashcards", "input")

os.makedirs("data/processed", exist_ok=True)
torch.save(python_acts, "data/processed/python_acts.pt")
torch.save(medical_acts, "data/processed/medical_acts.pt")

print("\nReal structural activations extracted and saved!")
hook.remove()