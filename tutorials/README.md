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
| 2 | `NonPeriodicBC.ipynb` | Reflecting and absorbing boundary conditions for confined systems |
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

### High-Level API (Recommended)

```python
from polymerfts import PropagatorSolver

# Create solver
solver = PropagatorSolver(
    nx=[64, 64], lx=[4.0, 4.0],
    ds=0.01,
    bond_lengths={"A": 1.0, "B": 1.0},
    bc=["periodic"]*4,
    chain_model="continuous",
    method="pseudospectral",
    platform="cpu-mkl",
    reduce_memory_usage=False
)

# Add polymer
solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])

# Compute propagators
solver.compute_propagators({"A": w_A, "B": w_B})

# Get results
Q = solver.get_partition_function(polymer=0)
q = solver.get_propagator(polymer=0, v=0, u=1, step=50)
solver.compute_concentrations()
phi_A = solver.get_concentration("A")
```

## Notation and Units

### Dimensionless Units

All quantities in `polymerfts` are expressed in **dimensionless units** based on a reference polymer:

| Symbol | Name | Definition |
|--------|------|------------|
| $b$ | Reference segment length | Statistical segment length of reference monomer |
| $N$ | Reference chain length | Number of statistical segments in reference chain |
| $R_0 = b\sqrt{N}$ | Reference length scale | Unperturbed end-to-end distance |

**Length**: Measured in units of $R_0$. A box with `lx = [4.0]` is 4 times the polymer size.

**Contour**: The chain backbone is parameterized by $s \in [0, 1]$, where $s=0$ is one end and $s=1$ is the other. The step size `ds = 0.01` corresponds to $N = 100$ segments.

### Key Physical Quantities

| Symbol | Name | Physical Meaning |
|--------|------|------------------|
| $q(\mathbf{r}, s)$ | **Propagator** | Statistical weight of all chain conformations with segment $s$ at position $\mathbf{r}$ |
| $Q$ | **Partition function** | Total statistical weight of all chain conformations (normalization) |
| $\phi(\mathbf{r})$ | **Concentration** | Ensemble-averaged monomer density at $\mathbf{r}$ |
| $w(\mathbf{r})$ | **Potential field** | Dimensionless free energy penalty for placing a monomer at $\mathbf{r}$ |

### Boltzmann Weighting

The potential field affects chain statistics through:

$$\text{local weight} \propto e^{-w(\mathbf{r})}$$

- $w > 0$: Repulsive (monomers avoid this region)
- $w < 0$: Attractive (monomers prefer this region)
- $w = 0$: No bias (reference state)

## Requirements

- Python 3.8+
- NumPy, Matplotlib
- `polymerfts` installed (`make install` from build directory)
- Jupyter notebook or JupyterLab

## Environment Setup

For optimal performance, set these environment variables at the start of each notebook:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"      # Single-threaded OpenMP
os.environ["MKL_NUM_THREADS"] = "1"      # Single-threaded MKL
```

This prevents thread oversubscription when running multiple calculations.

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
