import os
import numpy as np

from polymerfts import PropagatorSolver, _core


def _available_platforms():
    try:
        return set(_core.PlatformSelector().avail_platforms())
    except Exception:
        return set()


def _make_fields(n_grid: int, seed: int = 1234):
    rng = np.random.default_rng(seed)
    # Per testing guidelines: use field amplitude std ~ 5.
    w_a = rng.normal(0.0, 5.0, size=n_grid)
    w_b = rng.normal(0.0, 5.0, size=n_grid)
    return w_a, w_b


def _run_solver(platform: str, numerical_method: str = "rqm4",
                bc=None, chain_model: str = "continuous"):
    nx = [8, 8, 8]
    lx = [4.0, 4.0, 4.0]
    n_grid = int(np.prod(nx))

    if bc is None:
        bc = ["periodic"] * 6

    solver = PropagatorSolver(
        nx=nx,
        lx=lx,
        ds=0.02,
        bond_lengths={"A": 1.0, "B": 1.0},
        bc=bc,
        chain_model=chain_model,
        numerical_method=numerical_method,
        platform=platform,
        reduce_memory=False,
    )
    solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])

    w_a, w_b = _make_fields(n_grid)
    solver.compute_propagators({"A": w_a, "B": w_b})

    assert solver.check_total_partition(), "Partition check failed for exact method"

    solver.compute_concentrations()
    phi_a = solver.get_concentration("A")
    phi_b = solver.get_concentration("B")
    q0 = solver.get_partition_function(0)
    total_mean = solver.mean(phi_a + phi_b)

    return {
        "Q": q0,
        "phi_a": phi_a,
        "phi_b": phi_b,
        "total_mean": total_mean,
        "solver": solver,
    }


def test_cpu_cuda_equivalence_rqm4():
    platforms = _available_platforms()
    if "cpu-fftw" not in platforms or "cuda" not in platforms:
        # Skip when either backend is unavailable in the current build.
        return

    # Choose a likely-idle GPU without requiring NVML privileges.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    cpu = _run_solver("cpu-fftw", numerical_method="rqm4")
    cuda = _run_solver("cuda", numerical_method="rqm4")

    # Partition functions should match to near machine precision.
    assert np.isclose(cpu["Q"], cuda["Q"], rtol=5e-12, atol=1e-12), (
        f"Partition mismatch: cpu={cpu['Q']:.15e}, cuda={cuda['Q']:.15e}"
    )

    # Material conservation should hold on both platforms.
    assert abs(cpu["total_mean"] - 1.0) < 5e-12
    assert abs(cuda["total_mean"] - 1.0) < 5e-12

    # Concentrations should be very close across platforms.
    max_diff_a = float(np.max(np.abs(cpu["phi_a"] - cuda["phi_a"])))
    max_diff_b = float(np.max(np.abs(cpu["phi_b"] - cuda["phi_b"])))
    assert max(max_diff_a, max_diff_b) < 1e-9, (
        f"Concentration mismatch too large: A={max_diff_a:.3e}, B={max_diff_b:.3e}"
    )


def test_cpu_cuda_equivalence_reflecting_continuous():
    """Continuous chain + reflecting (DCT) BC must match between CPU and CUDA.

    Regression guard for the CUDA pseudo-spectral DCT path: this is the case the
    guardrail allows on CUDA, so it must stay correct.
    """
    platforms = _available_platforms()
    if "cpu-fftw" not in platforms or "cuda" not in platforms:
        return

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    bc = ["reflecting"] * 6
    cpu = _run_solver("cpu-fftw", bc=bc, chain_model="continuous")
    cuda = _run_solver("cuda", bc=bc, chain_model="continuous")

    assert np.isclose(cpu["Q"], cuda["Q"], rtol=5e-12, atol=1e-12), (
        f"Reflecting-BC partition mismatch: cpu={cpu['Q']:.15e}, cuda={cuda['Q']:.15e}"
    )
    max_diff_a = float(np.max(np.abs(cpu["phi_a"] - cuda["phi_a"])))
    max_diff_b = float(np.max(np.abs(cpu["phi_b"] - cuda["phi_b"])))
    assert max(max_diff_a, max_diff_b) < 1e-9, (
        f"Reflecting-BC concentration mismatch: A={max_diff_a:.3e}, B={max_diff_b:.3e}"
    )


def test_cuda_refuses_unsupported_bc():
    """The CUDA guardrail must refuse BC/chain combinations whose CUDA
    pseudo-spectral path is known to be numerically wrong, rather than silently
    returning incorrect results:

      - continuous + absorbing  (CUDA DST high-frequency-mode error)
      - discrete   + reflecting (CUDA DCT wrong for the discrete bond model)
      - discrete   + absorbing  (CUDA DST wrong for the discrete bond model)
    """
    platforms = _available_platforms()
    if "cuda" not in platforms:
        return

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    unsupported = [
        ("continuous", ["absorbing"] * 6),
        ("discrete", ["reflecting"] * 6),
        ("discrete", ["absorbing"] * 6),
    ]
    for chain_model, bc in unsupported:
        raised = False
        try:
            _run_solver("cuda", bc=bc, chain_model=chain_model)
        except NotImplementedError:
            raised = True
        assert raised, (
            f"CUDA guardrail failed to refuse unsupported combination: "
            f"chain_model={chain_model}, bc={bc[0]}"
        )
