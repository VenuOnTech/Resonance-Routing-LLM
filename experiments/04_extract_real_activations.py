import torch, os, sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.activation_hooks import LayerHook

# 1. Initialize Model and Tokenizer
print("Loading TinyLlama...")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Attach your hook to Layer 2
hook = LayerHook(model.model.layers[2])

def extract_structural_fingerprints(dataset_name, text_column, n_samples=500):
    print(f"Streaming {n_samples} samples from {dataset_name}...")
    # streaming=True prevents downloading gigabytes of unnecessary data
    data = load_dataset(dataset_name, split="train", streaming=True)
    
    activations = []
    count = 0
    for row in data:
        if count >= n_samples: break
        
        # Tokenize and push through the model
        inputs = tokenizer(row[text_column], return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad(): 
            model(**inputs)
        
        # The hook intercepts the tensor shape. We save it.
        activations.append(hook.activation)
        count += 1
        
    return torch.cat(activations, dim=0)

# 2. Extract Data
python_acts = extract_structural_fingerprints("iamtarun/python_code_instructions_18k_alpaca", "prompt")
medical_acts = extract_structural_fingerprints("medalpaca/medical_meadow_medical_flashcards", "input")

# 3. Save to your repository's data folder
os.makedirs("data/processed", exist_ok=True)
torch.save(python_acts, "data/processed/python_acts.pt")
torch.save(medical_acts, "data/processed/medical_acts.pt")

print("Real structural activations extracted and saved!")
hook.remove()
