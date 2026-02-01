# Non-Periodic Boundary Conditions Examples

This folder contains examples demonstrating the use of non-periodic boundary conditions with the pseudo-spectral method for polymer field theory simulations.

## Background

Traditional pseudo-spectral methods use Fast Fourier Transforms (FFT) which impose periodic boundary conditions. For confined systems (thin films, surfaces, interfaces), non-periodic boundary conditions are needed:

- **Reflecting BC (Neumann)**: Zero flux at boundaries (∂q/∂n = 0)
  - Uses Discrete Cosine Transform (DCT-II/III)
  - Mass is conserved
  - Models impenetrable, non-reactive walls

- **Absorbing BC (Dirichlet)**: Zero value at boundaries (q = 0)
  - Uses Discrete Sine Transform (DST-II/III)
  - Mass decreases over time (absorbed at boundaries)
  - Models reactive surfaces

## Examples

### 1. CompareRealSpacePseudoSpectral.py
Compares propagator computation using two numerical methods:
- **Pseudo-spectral**: DCT/DST transforms (high accuracy for smooth solutions)
- **Real-space**: Crank-Nicolson finite differences (more general BCs)

Tests 1D, 2D, and 3D cases with reflecting, absorbing, and mixed BCs.
Both methods converge to the same solution as grid resolution increases.

### 2. ReflectingBC_Propagator1D.py
Demonstrates propagator computation with reflecting BC in 1D.
- Shows mass conservation with reflecting boundaries
- Gaussian initial condition diffuses but total mass is preserved

### 3. AbsorbingBC_Propagator1D.py
Demonstrates propagator computation with absorbing BC in 1D.
- Shows mass decay with absorbing boundaries
- Compares with reflecting BC for the same initial condition

### 4. MixedBC_Propagator2D.py
Demonstrates mixed boundary conditions in 2D:
- Reflecting BC in x-direction (side walls)
- Absorbing BC in y-direction (top/bottom surfaces)
- Models a thin film with impenetrable sides and reactive surfaces

### 5. DCT_DST_Comparison.py
Validates DCT/DST implementation against FFT:
- DCT on [0,L] should match FFT on [0,2L] with symmetric extension
- DST on [0,L] should match FFT on [0,2L] with antisymmetric extension
- Demonstrates mathematical equivalence

## Usage

```bash
cd devel/non_periodic_bc

# Run comparison test (real-space vs pseudo-spectral)
python CompareRealSpacePseudoSpectral.py

# Run individual examples
python ReflectingBC_Propagator1D.py
python AbsorbingBC_Propagator1D.py
python MixedBC_Propagator2D.py
python DCT_DST_Comparison.py
```

## API Usage

These examples use the high-level `PropagatorSolver` class which provides a simple interface for propagator computations.

### Creating a solver with non-periodic BC

```python
from polymerfts import PropagatorSolver
import numpy as np

# 1D with reflecting BC on both sides
solver = PropagatorSolver(
    nx=[32], lx=[4.0],
    ds=0.01,
    bond_lengths={"A": 1.0},
    chain_model="continuous",
    bc=["reflecting", "reflecting"]
)

# 2D with mixed BC (reflecting in x, absorbing in y)
solver = PropagatorSolver(
    nx=[32, 24], lx=[4.0, 3.0],
    ds=0.01,
    bond_lengths={"A": 1.0},
    chain_model="continuous",
    bc=["reflecting", "reflecting", "absorbing", "absorbing"]
)
```

### BC format

- 1D: `[x_low, x_high]`
- 2D: `[x_low, x_high, y_low, y_high]`
- 3D: `[x_low, x_high, y_low, y_high, z_low, z_high]`

Options: `"periodic"`, `"reflecting"`, `"absorbing"`

### Computing propagators

```python
# Add a homopolymer
solver.add_polymer(volume_fraction=1.0, blocks=[["A", 1.0, 0, 1]])

# Set potential fields (zero for free diffusion)
solver.set_fields({"A": np.zeros(32)})

# Advance propagator one segment
q_out = solver.advance(q_in, "A")

# Or propagate for multiple steps at once
q_final = solver.propagate(q_init, "A", n_steps=100)
```

### Choosing numerical method

```python
# Pseudo-spectral method (higher accuracy)
solver = PropagatorSolver(
    nx=[32], lx=[4.0], ds=0.01, bond_lengths={"A": 1.0},
    chain_model="continuous", bc=["reflecting", "reflecting"],
    numerical_method="rqm4"  # or "rk2" for 2nd-order
)

# Real-space method (Crank-Nicolson, more flexible)
solver = PropagatorSolver(
    nx=[32], lx=[4.0], ds=0.01, bond_lengths={"A": 1.0},
    chain_model="continuous", bc=["reflecting", "reflecting"],
    numerical_method="cn-adi2"
)
```

### Platform selection

```python
# Auto-select platform: cuda for 2D/3D, cpu-mkl for 1D
solver = PropagatorSolver(
    nx=[32, 24], lx=[4.0, 3.0], ds=0.01, bond_lengths={"A": 1.0},
    chain_model="continuous", bc=["periodic"]*4, platform="auto"
)

# Force CPU
solver = PropagatorSolver(
    nx=[32, 24], lx=[4.0, 3.0], ds=0.01, bond_lengths={"A": 1.0},
    chain_model="continuous", bc=["periodic"]*4, platform="cpu-fftw"
)

# Force CUDA
solver = PropagatorSolver(
    nx=[32, 24], lx=[4.0, 3.0], ds=0.01, bond_lengths={"A": 1.0},
    chain_model="continuous", bc=["periodic"]*4, platform="cuda"
)
```

## Notes

- The high-level `PropagatorSolver` class automatically handles platform selection and method fallback
- Both CUDA and CPU-FFTW support pseudo-spectral methods with DCT (reflecting BC) and DST (absorbing BC)
- Real-space methods (CN-ADI) are available on both platforms for all BC types
- Full SCFT iterations with non-periodic BC require additional considerations for field normalization and incompressibility
- The DCT-II/III and DST-II/III conventions follow scipy.fft

## Numerical Methods

Two methods are available for non-periodic boundary conditions:

1. **Pseudo-spectral (DCT/DST)**
   - Uses DCT for reflecting BC, DST for absorbing BC
   - High accuracy for smooth solutions (spectral convergence)
   - Efficient for large grids
   - Available on both CPU (FFTW) and CUDA

2. **Real-space (Crank-Nicolson)**
   - Uses finite differences with ADI splitting
   - Second-order accuracy in space and time
   - More flexible for complex boundary conditions
   - Available on both CPU and CUDA
