import torch, sys
sys.path.append('..')
from src.core.resonance_router import ResonanceRouter
from src.utils.metrics import euclidean_distance

router = ResonanceRouter()
dim, N = 1024, 500
bA, bB = torch.randn(dim), torch.randn(dim)
bA, bB = bA/torch.norm(bA), bB/torch.norm(bB)

tA = torch.randn(N, dim) + torch.randn(N, 1)*15*bA
tB = torch.randn(N, dim) + torch.randn(N, 1)*15*bB
router.calibrate("A", tA)
router.calibrate("B", tB)

test = torch.ones(N, 1)*30*bA + torch.ones(N, 1)*5*bB
mah_tasks, _ = router(test)
eA = euclidean_distance(test, tA.mean(0, keepdim=True))
eB = euclidean_distance(test, tB.mean(0, keepdim=True))

m_acc = sum(1 for t in mah_tasks if t=='A') / N * 100
e_acc = sum(1 for a,b in zip(eA, eB) if a<b) / N * 100
print(f"Mahalanobis Acc: {m_acc}%\nEuclidean Acc: {e_acc}%")
