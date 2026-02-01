# Self-Consistent Field Theory (SCFT)

> **Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the self-consistent field theory (SCFT) implementation for finding saddle-point solutions in polymer field theory.

## Table of Contents

1. [Overview](#1-overview)
2. [Saddle-Point Equations](#2-saddle-point-equations)
3. [Iterative Methods](#3-iterative-methods)
4. [Box Optimization](#4-box-optimization)
5. [Implementation](#5-implementation)
6. [References](#6-references)

---

## 1. Overview

Self-consistent field theory (SCFT) finds the saddle-point of the effective Hamiltonian where the fields and concentrations are mutually consistent. At the saddle point:

- The functional derivatives of the Hamiltonian with respect to the fields vanish
- The fields satisfy the self-consistency equations
- The free energy is stationary

SCFT provides mean-field predictions for polymer phase behavior, morphology, and thermodynamic properties.

---

## 2. Saddle-Point Equations

### 2.1 Self-Consistency Conditions

For a system with monomer types $i = A, B, C, \ldots$, the self-consistent potential fields satisfy:

$$w_i(\mathbf{r}) = \sum_{j \neq i} \chi_{ij} N \phi_j(\mathbf{r}) + \xi(\mathbf{r})$$

where:
- $w_i(\mathbf{r})$: potential field for monomer type $i$
- $\chi_{ij}$: Flory-Huggins interaction parameter
- $\phi_j(\mathbf{r})$: concentration of monomer type $j$
- $\xi(\mathbf{r})$: Lagrange multiplier enforcing incompressibility

### 2.2 Field Residuals

The field residuals measure the deviation from self-consistency:

$$\mathbf{R}_w = X \boldsymbol{\phi} - P \mathbf{w}$$

where:
- $X$: the $\chi N$ matrix
- $P = I - \frac{\mathbf{e}\mathbf{e}^T X^{-1}}{\mathbf{e}^T X^{-1} \mathbf{e}}$: projection matrix

### 2.3 Example: AB Diblock Copolymer

For AB-type systems:

$$\mathbf{R}_w = \begin{pmatrix} 0 & \chi N \\ \chi N & 0 \end{pmatrix} \begin{pmatrix} \phi_A(\mathbf{r}) \\ \phi_B(\mathbf{r}) \end{pmatrix} - \begin{pmatrix} 1/2 & -1/2 \\ -1/2 & 1/2 \end{pmatrix} \begin{pmatrix} w_A(\mathbf{r}) \\ w_B(\mathbf{r}) \end{pmatrix}$$

Explicitly:

$$R_{w_A} = \chi N \phi_B(\mathbf{r}) - \frac{1}{2}(w_A(\mathbf{r}) - w_B(\mathbf{r}))$$

$$R_{w_B} = \chi N \phi_A(\mathbf{r}) + \frac{1}{2}(w_A(\mathbf{r}) - w_B(\mathbf{r}))$$

---

## 3. Iterative Methods

### 3.1 Simple Mixing (Gradient Descent)

The simplest approach updates fields using the residuals:

$$w_i(\mathbf{r}, \tau+1) = w_i(\mathbf{r}, \tau) + \lambda (R_w)_i$$

where $\lambda$ is the mixing parameter and $\tau$ is the iteration number.

**Advantages:**
- Simple to implement
- Robust for initial iterations

**Disadvantages:**
- Slow convergence near the saddle point
- Requires small $\lambda$ for stability

### 3.2 Anderson Mixing

Anderson mixing accelerates convergence by using history of previous iterations:

$$w^{(n+1)} = w^{(n)} + \lambda R_w^{(n)} - \sum_{i=1}^{m} c_i \left[ (w^{(n)} - w^{(n-i)}) + \lambda (R_w^{(n)} - R_w^{(n-i)}) \right]$$

where the coefficients $c_i$ are determined by minimizing:

$$\min_{c_i} \left\| R_w^{(n)} - \sum_{i=1}^{m} c_i (R_w^{(n)} - R_w^{(n-i)}) \right\|^2$$

**Key parameters:**
- `max_hist`: Maximum history length (typically 10-20)
- `start_error`: Error threshold to switch from simple mixing to Anderson mixing
- `mix_min`, `mix_init`: Minimum and initial mixing parameters

**Advantages:**
- Much faster convergence (often 10x fewer iterations)
- Robust for a wide range of systems

### 3.3 Convergence Criterion

The self-consistency error is defined as:

$$\epsilon = \sqrt{\frac{\sum_i \int |\mathbf{R}_{w_i}|^2 d\mathbf{r}}{1 + \sum_i \int |w_i|^2 d\mathbf{r}}}$$

The constant 1 in the denominator prevents divergence when the field magnitudes are small.

Iteration terminates when $\epsilon < \text{tolerance}$ (typically $10^{-7}$ to $10^{-9}$).

---

## 4. Box Optimization

### 4.1 Stress Residuals

The equilibrium unit cell minimizes the free energy. The stress residuals are:

$$\mathbf{R}_{stress} = -\frac{1}{Q} \frac{\partial Q}{\partial \boldsymbol{\theta}}$$

where $\boldsymbol{\theta} = [L_a, L_b, L_c, \alpha, \beta, \gamma]^T$ contains the unit cell parameters.

### 4.2 Box Update

During SCFT iteration with `box_is_altering=True`:

$$L_a^{(n+1)} = L_a^{(n)} - \eta \cdot \sigma_a$$

$$\alpha^{(n+1)} = \alpha^{(n)} - \eta \cdot \sigma_{bc}$$

where $\eta$ is the `scale_stress` parameter. At equilibrium, all stress components vanish.

See [StressTensor.md](StressTensor.md) for detailed stress calculation.

---

## 5. Implementation

### 5.1 Using the SCFT Class

```python
from polymerfts import scft

params = {
    "platform": "cuda",              # or "cpu-mkl", "cpu-fftw"
    "nx": [64, 64, 64],              # Grid points
    "lx": [4.0, 4.0, 4.0],           # Box size

    "chain_model": "continuous",     # or "discrete"
    "ds": 0.01,                      # Contour step (1/N_ref)

    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 20.0},

    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": 0.5},
            {"type": "B", "length": 0.5},
        ],
    }],

    "optimizer": {
        "name": "am",                # Anderson Mixing
        "max_hist": 20,
        "start_error": 1e-2,
        "mix_min": 0.1,
        "mix_init": 0.1,
    },

    "box_is_altering": True,         # Enable box optimization
    "scale_stress": 0.5,             # Stress scaling factor

    "max_iter": 2000,
    "tolerance": 1e-8,
}

# Initialize and run
calculation = scft.SCFT(params=params)
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Access results
phi_A = calculation.phi["A"]
free_energy = calculation.free_energy
final_box = calculation.cb.get_lx()

# Save results
calculation.save_results("fields.json")
```

### 5.2 Manual SCFT Implementation

For educational purposes, here's a simplified SCFT loop:

```python
from polymerfts import PropagatorSolver
import numpy as np

# Create solver
solver = PropagatorSolver(
    nx=[64, 64], lx=[2.0, 2.0], ds=0.01,
    bond_lengths={"A": 1.0, "B": 1.0},
)
solver.add_polymer(volume_fraction=1.0,
                   blocks=[["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])

# Initialize fields
w = {"A": np.sin(...), "B": -np.sin(...)}
chi_n = 20.0
lambda_mix = 0.5

for iteration in range(1000):
    # Compute concentrations
    solver.compute_propagators(w)
    solver.compute_concentrations()
    phi_A = solver.get_concentration("A")
    phi_B = solver.get_concentration("B")

    # Compute residuals
    R_A = chi_n * phi_B - 0.5 * (w["A"] - w["B"])
    R_B = chi_n * phi_A + 0.5 * (w["A"] - w["B"])

    # Check convergence
    error = np.sqrt(np.mean(R_A**2) + np.mean(R_B**2))
    if error < 1e-7:
        break

    # Update fields
    w["A"] += lambda_mix * R_A
    w["B"] += lambda_mix * R_B
```

### 5.3 Key Output

| Property | Access | Description |
|----------|--------|-------------|
| Concentration | `calculation.phi["A"]` | Monomer density field |
| Free energy | `calculation.free_energy` | Helmholtz free energy per chain |
| Partition function | `solver.get_partition_function(p)` | Single-chain partition function |
| Box size | `calculation.cb.get_lx()` | Final box dimensions |
| Stress | `calculation.stress` | Stress tensor components |

---

## 6. References

1. Ceniceros, H. D. & Fredrickson, G. H. "Numerical solution of polymer self-consistent field theory." *Multiscale Model. Simul.* **2**, 452-474 (2004).

2. Stasiak, P. & Matsen, M. W. "Efficiency of pseudo-spectral algorithms with Anderson mixing for the SCFT of periodic block-copolymer phases." *Eur. Phys. J. E* **34**, 110 (2011).

3. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).

4. Thompson, R. B., Rasmussen, K. Ø. & Lookman, T. "Improved convergence in block copolymer self-consistent field theory by Anderson mixing." *J. Chem. Phys.* **120**, 31-34 (2004).
