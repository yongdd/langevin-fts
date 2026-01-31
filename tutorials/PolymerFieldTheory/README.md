# Polymer Field Theory Foundations

Understand the mathematical framework behind SCFT and L-FTS.

## Environment Setup

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
```

## Tutorials

Read the notebooks in the following order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `MonomerPotentialFields.ipynb` | Self-consistent field equations and χ matrix |
| 2 | `IncompressibleModel.ipynb` | Incompressible field theory with auxiliary fields |
| 3 | `CompressibleModel.ipynb` | Compressible field theory (ζN parameter) |

## Key Concepts

- **Monomer potential fields** $w_A$, $w_B$: Direct representation of chemical potentials
- **Auxiliary fields** $\Omega_-$, $\Omega_+$: Exchange and pressure-like fields
- **χN parameter**: Flory-Huggins interaction strength × chain length
- **ζN parameter**: Compressibility parameter × chain length
