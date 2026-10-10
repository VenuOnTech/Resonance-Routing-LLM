# Resonance-Routing-LLM 🧠⚡

[![CI Tests](https://github.com/VenuOnTech/Resonance-Routing-LLM/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/VenuOnTech/Resonance-Routing-LLM/actions/workflows/ci-tests.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An autonomous, zero-shot, training-free routing architecture for Multi-LoRA Large Language Models. 

Instead of relying on fragile keyword-matching (cosine similarity) or training computationally expensive classifier networks, this engine routes prompts by measuring the **Mahalanobis Distance** on the latent space covariance geometries of early-layer neural activations.

## 🚀 The Core Problem: The Semantic Trap
Standard Mixture-of-Experts (MoE) or Multi-LoRA routers suffer from the **Semantic Mean-Vector Trap**. If you ask a standard cosine-similarity router to *"Write a Python script to filter clinical patient data,"* it gets confused by the domain-specific vocabulary ("clinical", "patient") and misroutes the prompt to a Medical expert.

**Resonance-Routing solves this.** By intercepting the structural cognitive footprint at early layers (before the model focuses on vocabulary), the geometry recognizes the structural command ("Write a script") and routes it accurately to the coding expert, ignoring the deceptive medical nouns.

## ⚙️ Key Engineering Features

* **Universal Model Agnosticism:** Dynamically parses HuggingFace architectures (Llama, Mistral, Qwen, GPT-2) via `model_utils.py`. It automatically identifies the 10% network depth mark to hook the optimal structural routing layer, regardless of whether the base model uses `model.layers` or `transformer.h`.
* **Causal Last-Token Extraction:** Safely respects causal masking by strictly extracting the final token's hidden state (`[:, -1, :]`), capturing the fully aggregated attention context of the entire prompt sequence.
* **Rank-Safe Mathematical Fallbacks:** Replaces standard matrix inversion with the Moore-Penrose Pseudo-Inverse (`torch.linalg.pinv`). The router gracefully approximates geometric covariance even when the calibration sample size is smaller than the model's hidden dimension ($N < D$), preventing rank-deficiency crashes.
* **Zero-Shot Continual Learning:** Add new expert domains on the fly without backpropagation. Simply pass domain prompts through the model, calculate the mean vector ($\mu$) and inverse covariance matrix ($\Sigma^{-1}$), and save the footprint.
* **Production-Secured:** Patched against Remote Code Execution (RCE) `pickle` vulnerabilities by enforcing `weights_only=True` during PyTorch tensor deserialization.

## 📁 Repository Structure

```text
├── experiments/
│   ├── 01_2d_physics_proof.py          # Visual mathematical proof of Mahalanobis routing
│   ├── 02_1024d_benchmark.py           # High-dimensional tensor speed benchmarking
│   ├── 03_tinyllama_live_test.py       # Base model live-routing prototype
│   ├── 04_extract_real_activations.py  # Builds the covariance matrices from HF datasets
│   └── 05_evaluate_real_text.py        # Live zero-shot routing inference
├── src/
│   ├── core/
│   │   ├── activation_hooks.py         # PyTorch forward hooks for extraction
│   │   └── resonance_router.py         # Core Mahalanobis distance calculator
│   └── utils/
│       ├── metrics.py                  # Euclidean/Cosine baseline comparators
│       └── model_utils.py              # Dynamic LLM architecture parser
├── tests/
│   └── test_router_math.py             # Pytest unit tests for rank/matrix math
└── requirements.txt
```

## 🛠️ Quick Start
### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract Latent Footprints (Calibration)
Stream real datasets through any causal LLM to build the geometric covariance matrices. The script will automatically find the optimal extraction layer and adapt to the model's hidden dimensions.

```bash
python experiments/04_extract_real_activations.py
```
### 3. Run Live Routing Inference
Test the router against highly ambiguous prompts. Watch it slice through semantic traps using the pre-calibrated latent geometries.
```bash
python experiments/05_evaluate_real_text.py
```

## Example Output
```text
Calibrating router with real HuggingFace activations...  
Loading model for inference...  
[*] Dynamically attached to transformer.h.1 (Depth: 1/12)  
  
--- END-TO-END ROUTING RESULTS ---  
  
Prompt: 'Write a Python function to calculate patient BMI from a list of weights.'
Routed To: Python_LoRA
Mahalanobis Scores [Python, Medical]: [23.51, 107.83]

Prompt: 'How do I use PyTorch to classify MRI brain scans?'
Routed To: Medical_LoRA
Mahalanobis Scores [Python, Medical]: [23.47, 10.16]
```  
## 🧪 Testing  
This repository includes a fully automated Continuous Integration (CI) pipeline. To run the mathematical unit tests locally:

```bash
pytest tests/ -v
```
