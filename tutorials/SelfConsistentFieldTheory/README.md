# Self-Consistent Field Theory (SCFT)

Find equilibrium polymer morphologies by solving saddle-point equations.

## Environment Setup

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"      # Single-threaded OpenMP
os.environ["OMP_NUM_THREADS"] = "1"      # Single-threaded FFTW
```

## Tutorials

Read the notebooks in the following order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `NaiveSCFT.ipynb` | Step-by-step SCFT implementation (educational) |
| 2 | `Cylinder.ipynb` | Production SCFT with Anderson mixing and box optimization |

## Key Concepts

- **Saddle-point**: Self-consistent solution where fields and concentrations are mutually consistent
- **Anderson mixing**: Accelerated iterative solver for field updates
- **Box optimization**: Finding the unit cell size that minimizes free energy
- **Free energy**: Helmholtz free energy functional evaluated at the saddle point
