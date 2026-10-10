import torch, os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.resonance_router import ResonanceRouter
from src.core.activation_hooks import LayerHook

# 1. Load the Router and Calibrate with Real Data
router = ResonanceRouter()
python_acts = torch.load("data/processed/python_acts.pt", weights_only=True)
medical_acts = torch.load("data/processed/medical_acts.pt", weights_only=True)

print("Calibrating router with real HuggingFace activations...")
# Force 32-bit float calibration
router.calibrate("Python_LoRA", python_acts.float())
router.calibrate("Medical_LoRA", medical_acts.float())

# 2. Initialize TinyLlama
from src.utils.model_utils import get_dynamic_routing_layer

print("Loading model for inference...")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Dynamically attach for inference
target_layer = get_dynamic_routing_layer(model)
hook = LayerHook(target_layer)

# 3. The Ambiguous Prompt Test
test_prompts = [
    "Write a Python function to calculate patient BMI from a list of weights.",
    "Create a pandas dataframe script to filter clinical trial data.",
    "How do I use PyTorch to classify MRI brain scans?"
]

print("\n--- END-TO-END ROUTING RESULTS ---")
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        model(**inputs)
        
    # Cast the live 16-bit activation up to 32-bit before routing
    act = hook.activation.float()
    tasks, distances = router(act)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Routed To: {tasks[0]}")
    print(f"Mahalanobis Scores [Python, Medical]: [{distances[0][0]:.2f}, {distances[0][1]:.2f}]")

hook.remove()