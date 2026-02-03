# Quick Start Guide

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

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

This stores only checkpoints and recomputes intermediate values, reducing memory by ~90% at the cost of 3-4x slower execution.

### Multi-threading Settings

Set these **before** importing polymerfts:

```python
import os

# Option 1: Disable OpenMP (single-threaded)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

# Option 2: Use multiple threads
os.environ["OMP_NUM_THREADS"] = "4"
```

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
    "mix_min": 0.01,      # Reduce from 0.1
    "mix_init": 0.01,     # Reduce from 0.1
    "start_error": 1e-2
}
```

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
