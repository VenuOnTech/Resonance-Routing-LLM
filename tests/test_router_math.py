import torch, pytest, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.resonance_router import ResonanceRouter

def test_calibration():
    router = ResonanceRouter()
    acts = torch.randn(50, 64)
    router.calibrate("A", acts)
    assert router.means["A"].shape == (64,)
    assert router.inv_covs["A"].shape == (64, 64)

def test_routing():
    router = ResonanceRouter()
    acts_a = torch.randn(100, 10) + 5
    acts_b = torch.randn(100, 10) - 5
    router.calibrate("A", acts_a)
    router.calibrate("B", acts_b)
    
    test_prompt = torch.ones(1, 10) * 5
    tasks, _ = router(test_prompt)
    assert tasks[0] == "A"
