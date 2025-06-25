from ..plugins.agem import AGEMPlugin
from ..plugins.igem import IGEMPlugin
from ..plugins.gem import GEMPlugin
import torch

def test_gem_qp():
    G = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    g = torch.tensor([0.5, -0.5])
    memory_strength = 0.1
    proj_interval = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    plugin = GEMPlugin(
        patterns_per_exp=1,
        memory_strength=memory_strength,
        proj_interval=proj_interval,
        proj_metric=None
    )

    plugin.G = G.to(device)
    g = g.to(device)

    # Call as static method (recommended)
    g_proj = plugin.solve_quadprog(plugin.G, g, plugin.memory_strength)

    assert isinstance(g_proj, torch.Tensor), "Output is not a tensor"
    assert g_proj.shape == g.shape, f"Expected shape {g.shape}, got {g_proj.shape}"
    assert g_proj.dtype == torch.float32

def test_agem_qp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plugin = AGEMPlugin(
        patterns_per_exp=1,
        sample_size=10,
        memory_strength=0.1,    # Add these two
        proj_interval=1,
        proj_metric=None
    )

    reference_gradients = torch.tensor([1.0, 2.0, 3.0], device=device)
    dotg = torch.tensor(7.0, device=device)

    alpha2, elapsed = plugin.solve_agem_sgd(dotg, reference_gradients)

    expected = 7.0 / (1**2 + 2**2 + 3**2)
    assert torch.isclose(alpha2, torch.tensor(expected, device=device)), "alpha2 not correct"
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0
    
def test_igem_qp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plugin = IGEMPlugin(
        patterns_per_exp=1,
        sgd_iterations=10,
        lr=0.1,
        use_adaptive_lr=False,
        use_warm_start=False,
        memory_strength=0.0,
        proj_interval=1,
        proj_metric=None
    )

    t = 2
    v = None
    plugin.G = torch.eye(t, device=device)
    plugin.GGT = torch.matmul(plugin.G, plugin.G.T)
    g = torch.tensor([-1.0, -1.0], device=device)

    v_star = plugin.solve_dualsgd(
        v=v,
        t=t,
        dev=device,
        G=plugin.G,
        g=g,
        GGT=plugin.GGT,
        I=plugin.sgd_iterations,
        lr=plugin.lr,
        memory_strength=plugin.memory_strength,
        use_adaptive_lr=plugin.use_adaptive_lr,
        use_warm_start=plugin.use_warm_start,
    )

    assert isinstance(v_star, torch.Tensor)
    assert v_star.shape == (t,)
    assert torch.all(v_star >= 0)
