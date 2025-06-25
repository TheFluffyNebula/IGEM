import pytest
from new_files import igem
from igem import IGEMPlugin
from new_files import core
import torch

def test_gem_qp():
    # Mock small memory matrix G and gradient vector g
    G = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    g = torch.tensor([0.5, -0.5])
    memory_strength = 0.1
    device = torch.device("cuda:0")

    g_proj = core.solve_quadprog(g, G, memory_strength, device)

    # Should return a tensor of same shape as g
    assert isinstance(g_proj, torch.Tensor), "Output is not a tensor"
    assert g_proj.shape == g.shape, f"Expected shape {g.shape}, got {g_proj.shape}"
    assert g_proj.dtype == torch.float32

def test_agem_qp():
    # Create dummy tensors
    reference_gradients = torch.tensor([1.0, 2.0, 3.0], device='cuda' if torch.cuda.is_available() else 'cpu')
    dotg = torch.tensor(7.0, device=reference_gradients.device)

    # Run function
    alpha2, elapsed = core.solve_agem_sgd(dotg, reference_gradients)

    # Expected value: 7 / (1^2 + 2^2 + 3^2) = 7 / 14 = 0.5
    expected = 0.5
    assert torch.isclose(alpha2, torch.tensor(expected, device=alpha2.device)), "alpha2 not computed correctly"
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0, "Elapsed time must be non-negative"
    
def test_igem_qp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize plugin with dummy args
    plugin = IGEMPlugin(
        patterns_per_exp=1,
        sgd_iterations=10,
        lr=0.1,
        use_adaptive_lr=False,
        use_warm_start=False,
        proj_metric=None
    )

    # Create fake G and g
    plugin.G = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)  # shape: (2, 2)
    g = torch.tensor([-1.0, -1.0], device=device)                     # shape: (2,)
    
    # Set memory strength
    plugin.memory_strength = 0.0

    # Call solve_dualsgd
    v_star = plugin.solve_dualsgd(t=2, dev=device, g=g, I=20)

    # v_star should be a 2D tensor (length 2)
    assert isinstance(v_star, torch.Tensor)
    assert v_star.shape == (2,)

    # All elements should be non-negative due to projection
    assert torch.all(v_star >= 0)

    # Optional: check approximate gradient descent convergence behavior
    assert torch.any(v_star > 0)

    # Stability test: calling again with same t should keep shape
    v_star2 = plugin.solve_dualsgd(t=2, dev=device, g=g, I=5)
    assert v_star2.shape == (2,)

test_gem_qp()
test_agem_qp()
test_igem_qp()
