# Polymer Field Theory Tutorials

This directory contains tutorials for learning polymer field theory simulations using `polymerfts`.

## Quick Start

**New to polymer field theory?** Start here:

1. **`00_QuickStart.ipynb`** - Get started in 5 minutes! Learn the basics of `PropagatorSolver`.

## Learning Path

After the Quick Start, follow the tutorials in this order:

### 1. Non-Interacting Chains in Potential Fields

Learn how to compute chain propagators and concentrations without self-consistent field theory.

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | `Diblock.ipynb` | AB diblock copolymer basics: propagators, partition functions, concentrations |
| 2 | `NonPeriodicBC.ipynb` | **NEW!** Reflecting and absorbing boundary conditions for confined systems |
| 3 | `BranchedMultiArmStar.ipynb` | Multi-arm star polymers and computational optimization |
| 4 | `BranchedComb.ipynb` | Comb polymers with backbone and side chains |
| 5 | `Mixture.ipynb` | Polymer mixtures (multiple species) |
| 6 | `GraftingPoints.ipynb` | Grafted polymers with real-space solver |
| 7 | `NanoParticle.ipynb` | Polymers around impenetrable particles |

### 2. Polymer Field Theory Foundations

Understand the mathematical framework behind SCFT and L-FTS.

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | `MonomerPotentialFields.ipynb` | Self-consistent field equations and χ matrix |
| 2 | `IncompressibleModel.ipynb` | Incompressible field theory with auxiliary fields |
| 3 | `CompressibleModel.ipynb` | Compressible field theory (ζN parameter) |

### 3. Self-Consistent Field Theory (SCFT)

Find equilibrium polymer morphologies by solving saddle-point equations.

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | `NaiveSCFT.ipynb` | Step-by-step SCFT implementation (educational) |
| 2 | `Cylinder.ipynb` | Production SCFT with Anderson mixing and box optimization |

### 4. Langevin Field-Theoretic Simulation (L-FTS)

Include compositional fluctuations beyond mean-field theory.

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | `NaiveLFTS.ipynb` | Step-by-step L-FTS implementation (educational) |
| 2 | `Lamellar.ipynb` | Production L-FTS with structure functions |

## API Levels

The tutorials use two API levels:

### High-Level API (Recommended for beginners)

```python
from polymerfts import PropagatorSolver

solver = PropagatorSolver(
    nx=[64, 64], lx=[4.0, 4.0],
    bc=["periodic", "periodic", "reflecting", "reflecting"],
    ds=0.01,
    bond_lengths={"A": 1.0, "B": 1.0}
)
solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
solver.set_fields({"A": w_A, "B": w_B})
q_out = solver.advance(q_in, "A")
```

### Low-Level Factory API (For advanced control)

```python
import polymerfts

factory = polymerfts.PlatformSelector.create_factory("cpu-mkl", False)
cb = factory.create_computation_box(nx, lx, bc=bc)
molecules = factory.create_molecules_information("continuous", ds, bond_lengths)
molecules.add_polymer(1.0, blocks)
prop_opt = factory.create_propagator_computation_optimizer(molecules, True)
solver = factory.create_pseudospectral_solver(cb, molecules, prop_opt)
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Propagator** $q(\mathbf{r}, s)$ | Statistical weight of chain conformations ending at position $\mathbf{r}$ at contour $s$ |
| **Partition function** $Q$ | Normalization for all chain conformations |
| **Concentration** $\phi(\mathbf{r})$ | Ensemble-averaged monomer density |
| **Potential field** $w(\mathbf{r})$ | External or self-consistent field acting on monomers |

## Requirements

- Python 3.8+
- NumPy, Matplotlib
- `polymerfts` installed (`make install` from build directory)
- Jupyter notebook or JupyterLab

## Running Tutorials

```bash
# Option 1: VSCode (Recommended)
# Open the tutorials folder in VSCode with the Jupyter extension installed
code tutorials/

# Option 2: JupyterLab
cd tutorials
jupyter lab

# Option 3: Classic Jupyter Notebook
cd tutorials
jupyter notebook
```

## Additional Resources

- **Examples**: See `examples/scft/` and `examples/fts/` for production simulation scripts
- **Documentation**: Run `doxygen Doxyfile` in root directory for API docs
- **Deep Learning Extension**: https://github.com/yongdd/deep-langevin-fts
