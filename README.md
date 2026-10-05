# Resonance Routing LLM

Zero-shot autonomous routing for Multi-LoRA continual learning. This repository solves the "Mean-Vector Trap" in MoE by measuring the geometric shape of early activations (Mahalanobis Distance) rather than magnitude (Cosine/Euclidean).

## The Math
Instead of standard vector similarity, we route using structural resonance:
$D_k(x) = \sqrt{(x - \mu_k)^T \tilde{\Sigma}_k^{-1} (x - \mu_k)}$

## Quick Start
```bash
pip install -r requirements.txt
python experiments/02_1024d_benchmark.py
