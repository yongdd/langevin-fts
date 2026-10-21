import sys

import numpy as np
from scipy.ndimage import gaussian_filter

from polymerfts import PropagatorSolver, _core


def _available_platforms():
    return set(_core.PlatformSelector.avail_platforms())


def _get_platform():
    """Get available platform (cpu-mkl > cpu-fftw for 1D simulations)."""
    platforms = _available_platforms()
    if "cpu-mkl" in platforms:
        return "cpu-mkl"
    if "cpu-fftw" in platforms:
        return "cpu-fftw"
    return None


def _partition_for(ds: float, method: str, w_a: np.ndarray, platform: str) -> float:
    nx = [128]
    lx = [4.0]

    solver = PropagatorSolver(
        nx=nx,
        lx=lx,
        ds=ds,
        bond_lengths={"A": 1.0},
        bc=["periodic", "periodic"],
        chain_model="continuous",
        numerical_method=method,
        platform=platform,
        reduce_memory=False,
    )
    solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
    solver.compute_propagators({"A": w_a})
    assert solver.check_total_partition()
    return float(solver.get_partition_function(0))


def test_convergence_order_lite_rqm4_and_rk2():
    platform = _get_platform()
    if platform is None:
        return

    rng = np.random.default_rng(2025)
    # Per testing guidelines: use field amplitude std ~ 5. The field is smoothed
    # (Gaussian filter, as in TestStressNonPeriodic) so that the methods are in
    # their asymptotic regime at these ds; white noise on a 0.03 grid gives
    # apparent orders of 1.5 (RQM4) and 1.4 (RK2) instead of 4 and 2.
    w_a = gaussian_filter(rng.normal(0.0, 5.0, size=128), sigma=3.0, mode="wrap")
    w_a *= 5.0 / w_a.std()

    # Reference solution with a much smaller ds.
    q_ref = _partition_for(ds=0.0025, method="rqm4", w_a=w_a, platform=platform)

    ds_values = np.array([0.04, 0.02, 0.01], dtype=float)

    def estimate_order(method: str) -> float:
        qs = np.array([_partition_for(ds=float(ds), method=method, w_a=w_a, platform=platform) for ds in ds_values])
        errors = np.abs(qs - q_ref)
        # Guard against degenerate fits.
        errors = np.maximum(errors, 1e-30)
        slope, _ = np.polyfit(np.log(ds_values), np.log(errors), 1)
        return float(slope)

    order_rqm4 = estimate_order("rqm4")
    order_rk2 = estimate_order("rk2")

    # Lightweight but meaningful guardrails on expected convergence orders.
    assert order_rqm4 > 3.0, f"Unexpected RQM4 order: {order_rqm4:.3f}"
    assert 1.5 < order_rk2 < 2.5, f"Unexpected RK2 order: {order_rk2:.3f}"
    print(f"RQM4 order {order_rqm4:.3f}, RK2 order {order_rk2:.3f}")


if __name__ == "__main__":
    if _get_platform() is None:
        print("SKIP: no CPU platform in this build")
        sys.exit(77)  # ctest SKIP_RETURN_CODE
    test_convergence_order_lite_rqm4_and_rk2()
    print("Test passed, convergence orders")
