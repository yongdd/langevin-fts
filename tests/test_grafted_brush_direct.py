#!/usr/bin/env python3
"""
Compare 2nd-order vs 4th-order real-space solver using direct .so loading.

This test directly loads _core.so from different build directories
to properly compare 2nd-order and 4th-order Richardson extrapolation.
"""

import os
import sys
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

# Script that directly loads _core.so
TEST_SCRIPT = '''
import sys
import json
import numpy as np
import importlib.util

# Direct load of _core.so
core_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("_core", core_path)
_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_core)

# Now we need to manually create the PropagatorSolver-like functionality
# using the low-level _core interface

def run_test(nx, Lx, ds, x0, sigma0, N_steps):
    # Create factory
    factory = _core.PlatformSelector.create_factory("cpu-mkl", False)

    # Create molecules
    bond_lengths = {"A": 1.0}
    molecules = factory.create_molecules_information("continuous", ds, bond_lengths)

    # Add polymer with grafting point
    chain_length = N_steps * ds
    molecules.add_polymer(1.0, [["A", chain_length, 0, 1]], {0: "G"})

    # Create propagator optimizer
    optimizer = factory.create_propagator_computation_optimizer(molecules, True)

    # Create computation box with absorbing BCs
    bc = ["absorbing", "absorbing"]
    cb = factory.create_computation_box(nx=[nx], lx=[Lx], bc=bc)

    # Create real-space solver
    solver = factory.create_realspace_solver(cb, molecules, optimizer)

    # Setup grid
    dx = Lx / nx
    x = (np.arange(nx) + 0.5) * dx

    # Gaussian initial condition
    q_init = np.exp(-(x - x0)**2 / (2 * sigma0**2))
    q_init_integral = np.sum(q_init) * dx
    q_init = q_init / q_init_integral

    # Zero potential field
    w_field = np.zeros(nx)

    # Compute propagators
    solver.compute_propagators({"A": w_field}, q_init={"G": q_init})

    # Get final propagator
    q_final = solver.get_chain_propagator(0, 0, 1, N_steps)
    Q = solver.get_total_partition(0)

    return x.tolist(), q_final.tolist(), Q

def analytical_gaussian(x, s, x0, sigma0, L, b=1.0, n_terms=200):
    D = b**2 / 6.0
    q = np.zeros_like(x)
    x_fine = np.linspace(0, L, 1000)
    dx_fine = x_fine[1] - x_fine[0]
    q0 = np.exp(-(x_fine - x0)**2 / (2 * sigma0**2))

    for n in range(1, n_terms + 1):
        kn = n * np.pi / L
        an = (2.0 / L) * np.sum(q0 * np.sin(kn * x_fine)) * dx_fine
        q += an * np.sin(kn * x) * np.exp(-D * kn**2 * s)
    return q

params = json.loads(sys.argv[2])
x, q, Q = run_test(
    params["nx"], params["Lx"], params["ds"],
    params["x0"], params["sigma0"], params["N_steps"]
)

x_arr = np.array(x)
s_max = params["N_steps"] * params["ds"]
q_ana = analytical_gaussian(x_arr, s_max, params["x0"], params["sigma0"], params["Lx"])

dx = params["Lx"] / params["nx"]
q_arr = np.array(q)

# Normalize analytical to same integral
q_ana_int = np.sum(q_ana) * dx
q_num_int = np.sum(q_arr) * dx
if q_ana_int > 0:
    q_ana = q_ana * (q_num_int / q_ana_int)

l2_error = np.sqrt(np.sum((q_arr - q_ana)**2) * dx)
l2_ana = np.sqrt(np.sum(q_ana**2) * dx)
rel_l2 = l2_error / l2_ana if l2_ana > 0 else 0

result = {"x": x, "q": q, "q_analytical": q_ana.tolist(),
          "rel_l2_error": rel_l2, "Q_numerical": q_num_int, "Q": Q}
print(json.dumps(result))
'''


def run_with_core(core_path, params):
    """Run test with specified _core.so."""
    script_path = "/tmp/test_direct_core.py"
    with open(script_path, 'w') as f:
        f.write(TEST_SCRIPT)

    result = subprocess.run(
        [sys.executable, script_path, core_path, json.dumps(params)],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        print(f"Error with {core_path}:")
        print(result.stderr)
        return None

    for line in result.stdout.strip().split('\n'):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def main():
    print("=" * 70)
    print("Grafted Brush: 2nd-Order vs 4th-Order (Direct Loading)")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core_4th = os.path.join(base_dir, "build", "_core.so")
    core_2nd = os.path.join(base_dir, "build_2nd", "_core.so")

    # Check builds
    for name, path in [("4th-order", core_4th), ("2nd-order", core_2nd)]:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)
            mtime = os.path.getmtime(path)
            print(f"  {name}: {path} ({size:.1f} MB)")
        else:
            print(f"  {name}: NOT FOUND at {path}")

    # Test parameters - Gaussian at safe distance from boundary
    nx = 512
    Lx = 4.0
    x0 = 0.5      # At least 5*sigma from boundary for stability
    sigma0 = 0.1  # Moderate width
    s_max = 0.2

    ds_values = [0.1, 0.05, 0.025, 0.0125]

    print(f"\nParameters:")
    print(f"  Grid: nx = {nx}")
    print(f"  Domain: Lx = {Lx}")
    print(f"  Gaussian: x0 = {x0}, sigma = {sigma0}")
    print(f"  s_max = {s_max}")

    results = {"4th-order": {}, "2nd-order": {}}
    cores = {"4th-order": core_4th, "2nd-order": core_2nd}

    for method, core_path in cores.items():
        if not os.path.exists(core_path):
            print(f"\nSkipping {method}: build not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"{method}")
        print(f"{'=' * 70}")

        for ds in ds_values:
            N_steps = int(round(s_max / ds))
            print(f"\n  ds = {ds}, N_steps = {N_steps}")

            params = {
                "nx": nx, "Lx": Lx, "ds": ds,
                "x0": x0, "sigma0": sigma0, "N_steps": N_steps
            }

            result = run_with_core(core_path, params)
            if result:
                results[method][ds] = result
                print(f"    L2 rel error: {result['rel_l2_error']*100:.6f}%")
                print(f"    Q (integral): {result['Q_numerical']:.10f}")

    # Compare 2nd vs 4th at same ds
    print(f"\n{'=' * 70}")
    print("Direct Comparison: 2nd vs 4th Order")
    print(f"{'=' * 70}")

    for ds in ds_values:
        if ds in results["4th-order"] and ds in results["2nd-order"]:
            q_4th = np.array(results["4th-order"][ds]['q'])
            q_2nd = np.array(results["2nd-order"][ds]['q'])

            dx = Lx / nx
            diff = np.sqrt(np.sum((q_4th - q_2nd)**2) * dx)
            l2_4th = np.sqrt(np.sum(q_4th**2) * dx)

            err_4th = results["4th-order"][ds]['rel_l2_error']
            err_2nd = results["2nd-order"][ds]['rel_l2_error']

            print(f"\n  ds = {ds}:")
            print(f"    4th-order error: {err_4th*100:.6f}%")
            print(f"    2nd-order error: {err_2nd*100:.6f}%")
            print(f"    Difference (2nd vs 4th): {diff/l2_4th*100:.6f}%")

    # Convergence analysis
    print(f"\n{'=' * 70}")
    print("Convergence Order")
    print(f"{'=' * 70}")

    for method in ["4th-order", "2nd-order"]:
        if len(results[method]) >= 2:
            print(f"\n{method}:")
            ds_list = sorted(results[method].keys(), reverse=True)

            for i in range(len(ds_list) - 1):
                ds1, ds2 = ds_list[i], ds_list[i+1]
                e1 = results[method][ds1]['rel_l2_error']
                e2 = results[method][ds2]['rel_l2_error']

                if e2 > 1e-15 and e1 > 1e-15:
                    ratio = ds1 / ds2
                    order = np.log(e1 / e2) / np.log(ratio)
                    print(f"  ds={ds1:.4f} -> ds={ds2:.4f}: error {e1*100:.4f}% -> {e2*100:.4f}%, order ≈ {order:.2f}")

    # Plot
    plt.figure(figsize=(14, 5))

    colors = {"4th-order": "blue", "2nd-order": "red"}

    # Plot 1: Propagator profiles
    plt.subplot(1, 3, 1)
    for method in ["4th-order", "2nd-order"]:
        if results[method]:
            finest_ds = min(results[method].keys())
            r = results[method][finest_ds]
            plt.plot(r['x'], r['q'], '-', color=colors[method],
                     label=f'{method} (ds={finest_ds})', linewidth=1.5)

    if results.get("4th-order"):
        finest = min(results["4th-order"].keys())
        plt.plot(results["4th-order"][finest]['x'],
                 results["4th-order"][finest]['q_analytical'],
                 'k--', label='Analytical', linewidth=2, alpha=0.7)

    plt.xlabel('x')
    plt.ylabel('q(x,s)')
    plt.title(f'Propagator at s={s_max}')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # Plot 2: Difference between 2nd and 4th order
    plt.subplot(1, 3, 2)
    for ds in ds_values:
        if ds in results["4th-order"] and ds in results["2nd-order"]:
            q_4th = np.array(results["4th-order"][ds]['q'])
            q_2nd = np.array(results["2nd-order"][ds]['q'])
            x = results["4th-order"][ds]['x']
            diff = np.abs(q_4th - q_2nd)
            plt.semilogy(x, diff + 1e-16, '-', label=f'ds={ds}', linewidth=1.5)

    plt.xlabel('x')
    plt.ylabel('|q_4th - q_2nd|')
    plt.title('Difference: 4th vs 2nd Order')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # Plot 3: Convergence
    plt.subplot(1, 3, 3)
    markers = {"4th-order": "o", "2nd-order": "s"}
    for method in ["4th-order", "2nd-order"]:
        if results[method]:
            ds_arr = np.array(sorted(results[method].keys()))
            l2_errors = np.array([results[method][ds]['rel_l2_error'] for ds in ds_arr])
            plt.loglog(ds_arr, l2_errors, f'{markers[method]}-', color=colors[method],
                       label=method, linewidth=2, markersize=8)

    # Reference slopes
    ds_ref = np.array([0.01, 0.15])
    e_ref = 0.002
    plt.loglog(ds_ref, e_ref * (ds_ref/0.01)**2, 'k--', alpha=0.5, label='O(ds²)')
    plt.loglog(ds_ref, e_ref * (ds_ref/0.01)**4, 'k:', alpha=0.5, label='O(ds⁴)')

    plt.xlabel('ds')
    plt.ylabel('Relative L2 Error')
    plt.title('Convergence')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'grafted_brush_direct.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
