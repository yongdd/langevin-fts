# Quick Start Guide

This guide helps you run your first polymer field theory simulation.

## Prerequisites

Ensure you have:
1. Completed the installation (see [Installation.md](Installation.md))
2. Activated the conda environment:
   ```bash
   conda activate polymerfts
   ```

## Your First Simulation

### Running the Lamellar Example

```bash
cd examples/scft
python Lamella3D.py
```

This runs a Self-Consistent Field Theory (SCFT) simulation for an AB diblock copolymer in the lamellar phase.

### Understanding the Output

The simulation outputs:
- **Iteration count**: SCFT iteration number
- **Error**: Residual error (convergence metric)
- **Free energy**: Helmholtz free energy per chain

When the error drops below the tolerance (default 10⁻⁸), the simulation converges.

## Basic Parameter Structure

```python
params = {
    # Grid
    "nx": [32, 32, 32],           # Grid points
    "lx": [4.0, 4.0, 4.0],        # Box size (in units of bN^(1/2))

    # Chain model
    "chain_model": "continuous",   # or "discrete"
    "ds": 0.01,                    # Contour step size

    # Polymer
    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 20.0},        # χN parameter
    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": 0.5},  # For linear chains, v/u are optional
            {"type": "B", "length": 0.5}
        ]
    }],

    # Numerical method
    "numerical_method": "rqm4",    # "rqm4", "rk2", or "cn-adi2"

    # Box relaxation (required by SCFT; True relaxes lx with the stress)
    "box_is_altering": False,

    # Optimizer
    "optimizer": {
        "name": "am",
        "max_hist": 20,
        "start_error": 1e-2,
        "mix_min": 0.1,
        "mix_init": 0.1
    }
}
```

## Numerical Method Selection

| Method | Description | Best For |
|--------|-------------|----------|
| `rqm4` | 4th-order Richardson extrapolation (default) | Standard SCFT/FTS |
| `rk2` | 2nd-order Rasmussen-Kalosakas | Faster, lower accuracy |
| `cn-adi2` | 2nd-order Crank-Nicolson ADI | Brush with grafted delta-function |

See [NumericalMethods.md](../theory/NumericalMethods.md) for detailed benchmarks.

## Performance Tips

### Memory Saving Mode

For large systems that exceed GPU memory:

```python
params["reduce_memory"] = True
```

This stores only checkpoints and recomputes intermediate values, which cuts the memory that grows with the grid and the number of contour steps (for `examples/scft/Gyroid.py` on a 64³ grid the GPU memory falls from 1064 to 420 MiB) at the cost of 2-4x slower execution.

### Multi-threading Settings

Set these **before** importing polymerfts:

```python
import os

# Option 1: Disable OpenMP (single-threaded)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

# Option 2: Use multiple threads
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
```

`OMP_NUM_THREADS` is the number of propagators computed at the same time: CPU threads, or CUDA streams (at most 4) on the GPU. Option 1 therefore also makes a GPU run compute its propagators one after another; GPU runs should use Option 2 (the examples use 2 threads). With `reduce_memory` the propagators are always computed one at a time.

### GPU Selection

```bash
# Check GPU availability
nvidia-smi

# Select specific GPU
export CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting Convergence

If SCFT doesn't converge, reduce the mixing parameters:

```python
"optimizer": {
    "name": "am",
    "max_hist": 20,
    "mix_min": 0.01,      # Reduce from 0.1
    "mix_init": 0.01,     # Reduce from 0.1
    "start_error": 1e-2
}
```

## Making Your Own Variant

The simulation classes are written so that one file is enough to change. To
try a different iteration scheme, a new observable, or an extra term in the
free energy, copy the file you want to change next to your script and edit the
copy; the installed package keeps providing the C++ core and the shared modules.

1. Copy the file (`scft.py`, `lfts.py` or `clfts.py`):
   ```bash
   cp src/python/scft.py ~/my_project/my_scft.py
   ```
2. In the copy, point the imports at the installed package instead of the
   relative `.`:
   ```bash
   sed -i -E 's/^from \. import _core/from polymerfts import _core/; s/^from \.([a-z_]+) import/from polymerfts.\1 import/' ~/my_project/my_scft.py
   ```
   The result looks like
   ```python
   from polymerfts import _core
   from polymerfts.polymer_field_theory import SymmetricPolymerTheory
   from polymerfts.propagator_solver import PropagatorSolver
   from polymerfts.validation import validate_system_params, ValidationError
   from polymerfts.utils import (...)
   from polymerfts.smearing import Smearing
   ```
3. Use the copy in your script:
   ```python
   import my_scft as scft          # instead of: from polymerfts import scft
   calculation = scft.SCFT(params)
   ```

Everything else stays the same; no rebuild and no reinstall. Edit `run()` or
`__init__()` in the copy as you like, and keep several variants as separate
files (`my_scft_v2.py`, ...).

Other options:

- To change a single method, subclass instead of copying:
  `class MySCFT(scft.SCFT): def compute_concentrations(self, w, q_init=None): ...`
- To change the whole package, including C++ code, clone the repository,
  run `make` (no `make install`) and point `PYTHONPATH` at `build/python`;
  `make` re-stages the Python files in seconds after each edit.
- If the change is worth sharing, work on a git branch and open a pull request.

## Next Steps

- **Tutorials**: Start with notebooks in `tutorials/` (see `tutorials/README.md` for order)
- **Examples**: Explore `examples/scft/`, `examples/lfts/`, and `examples/clfts/` for various polymer systems
- **Parameters**: See [Parameters.md](../references/Parameters.md) for complete parameter reference
- **Theory**: Read [NumericalMethods.md](../theory/NumericalMethods.md) for algorithm details

## Validation

Verify your installation by comparing results:

### Continuous Chains
Results should match [PSCF](https://github.com/dmorse/pscfpp) within machine precision (for even contour steps).

### Cross-Platform Consistency
Results should be identical (~10⁻¹³) regardless of:
- Platform (`cuda`, `cpu-mkl`, or `cpu-fftw`)
- Memory mode (standard or reduce_memory)
