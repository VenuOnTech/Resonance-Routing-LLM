import torch, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.resonance_router import ResonanceRouter

torch.manual_seed(42)
router = ResonanceRouter()

# Task A: Stretched highway
mean_a, cov_a = torch.tensor([5.0, 5.0]), torch.tensor([[4.0, 3.8], [3.8, 4.0]])
acts_a = torch.distributions.MultivariateNormal(mean_a, cov_a).sample((1000,))

# Task B: Tight sphere
mean_b, cov_b = torch.tensor([-5.0, -5.0]), torch.tensor([[1.0, 0.0], [0.0, 1.0]])
acts_b = torch.distributions.MultivariateNormal(mean_b, cov_b).sample((1000,))

router.calibrate("Python", acts_a)
router.calibrate("Medical", acts_b)

prompt = torch.tensor([[10.0, 10.0]])
tasks, dists = router(prompt)

print(f"Euclidean Distances -> Python: {torch.norm(prompt - mean_a):.2f}, Medical: {torch.norm(prompt - mean_b):.2f}")
print(f"Mahalanobis Distances -> Python: {dists[0][0]:.2f}, Medical: {dists[0][1]:.2f}")
print(f"Routed to: {tasks[0]}")
