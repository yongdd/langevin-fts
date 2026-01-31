# Non-Interacting Chains in Potential Fields

Learn how to compute chain propagators and concentrations without self-consistent field theory.

> **See also:** [docs/theory/ChainPropagators.md](../../docs/theory/ChainPropagators.md) for the consolidated theory documentation.

## Environment Setup

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
```

## Tutorials

Read the notebooks in the following order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `Diblock.ipynb` | AB diblock copolymer basics: propagators, partition functions, concentrations |
| 2 | `NonPeriodicBC.ipynb` | Reflecting and absorbing boundary conditions for confined systems |
| 3 | `BranchedMultiArmStar.ipynb` | Multi-arm star polymers and computational optimization |
| 4 | `BranchedComb.ipynb` | Comb polymers with backbone and side chains |
| 5 | `Mixture.ipynb` | Polymer mixtures (multiple species) |
| 6 | `GraftingPoints.ipynb` | Grafted polymers with real-space solver |
| 7 | `NanoParticle.ipynb` | Polymers around impenetrable particles |

## Key Concepts

- **Propagator** $q(\mathbf{r}, s)$: Statistical weight of chain conformations ending at position $\mathbf{r}$
- **Partition function** $Q$: Total statistical weight (normalization factor)
- **Concentration** $\phi(\mathbf{r})$: Ensemble-averaged monomer density
