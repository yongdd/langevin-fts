# Langevin Field-Theoretic Simulation (L-FTS)

Include compositional fluctuations beyond mean-field theory.

For detailed theory documentation, see [LangevinFieldTheoreticSimulation.md](../../docs/theory/LangevinFieldTheoreticSimulation.md).

## Environment Setup

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
```

## Tutorials

Read the notebooks in the following order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `NaiveLFTS.ipynb` | Step-by-step L-FTS implementation (educational) |
| 2 | `Lamellar.ipynb` | Production L-FTS with structure functions |

## Key Concepts

- **Langevin dynamics**: Stochastic evolution of fields with random noise
- **Invariant polymerization index** $\bar{N}$: Controls fluctuation strength ($\bar{N} \to \infty$ recovers SCFT)
- **Structure function** $S(k)$: Fourier-space correlation function for analyzing phase behavior
- **Partial saddle-point**: Real fields fluctuate while imaginary fields are at saddle point
