# Chain Propagators and Polymer Statistics

> **Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes how to compute chain propagators, partition functions, and concentrations for polymers in external potential fields.

## Table of Contents

1. [Overview](#1-overview)
2. [Modified Diffusion Equation](#2-modified-diffusion-equation)
3. [Linear Polymers](#3-linear-polymers)
4. [Branched Polymers](#4-branched-polymers)
5. [Polymer Mixtures](#5-polymer-mixtures)
6. [Boundary Conditions](#6-boundary-conditions)
7. [Implementation](#7-implementation)
8. [References](#8-references)

---

## 1. Overview

Chain propagators describe the statistical weight of polymer conformations in external potential fields. They are fundamental to polymer field theory calculations including SCFT and L-FTS.

**Key quantities:**

| Quantity | Symbol | Description |
|----------|--------|-------------|
| Propagator | $q(\mathbf{r}, s)$ | Statistical weight of chain conformations ending at position $\mathbf{r}$ after contour length $s$ |
| Partition function | $Q$ | Total statistical weight (normalization factor) |
| Concentration | $\phi(\mathbf{r})$ | Ensemble-averaged monomer density |

**Physical interpretation of potential fields:**

The monomer potential field $w_\alpha(\mathbf{r})$ determines local Boltzmann weighting:

$$\text{local weight} \propto e^{-w_\alpha(\mathbf{r})}$$

- $w < 0$: **Attractive** — monomers are favored here
- $w > 0$: **Repulsive** — monomers avoid this region

---

## 2. Modified Diffusion Equation

### 2.1 Continuous Chain Model

For continuous Gaussian chains, the propagator satisfies the modified diffusion equation:

$$\frac{\partial q(\mathbf{r}, s)}{\partial s} = \frac{b^2}{6} \nabla^2 q(\mathbf{r}, s) - w(\mathbf{r}) q(\mathbf{r}, s)$$

where:
- $b$: statistical segment length
- $w(\mathbf{r})$: potential field
- $s$: contour variable (0 to 1 for full chain)

### 2.2 Discrete Chain Model

For discrete chains (freely-jointed chain model), the propagator satisfies the Chapman-Kolmogorov equation:

$$q(\mathbf{r}, n+1) = e^{-w(\mathbf{r}) \Delta s} \int g(\mathbf{r} - \mathbf{r}') q(\mathbf{r}', n) \, d\mathbf{r}'$$

where $g(\mathbf{r})$ is the bond function (Gaussian distribution for bead-spring model).

### 2.3 Initial Conditions

**Free chain end:**
$$q(\mathbf{r}, 0) = 1$$

**Junction point (branched polymers):**
$$q^{v \rightarrow u}(\mathbf{r}, 0) = \prod_{k \neq u} q^{k \rightarrow v}(\mathbf{r}, f_k)$$

where the product is over all blocks connected to node $v$ except the one leading to $u$.

---

## 3. Linear Polymers

### 3.1 AB Diblock Copolymer

For an AB diblock with graph structure `0--A--1--B--2`:

**Forward propagators:**
- $q^{0 \rightarrow 1}(\mathbf{r}, s)$: Through A block, $s \in [0, f_A]$
- $q^{1 \rightarrow 2}(\mathbf{r}, s)$: Through B block, $s \in [0, f_B]$

**Backward propagators:**
- $q^{2 \rightarrow 1}(\mathbf{r}, s)$: Through B block (reverse), $s \in [0, f_B]$
- $q^{1 \rightarrow 0}(\mathbf{r}, s)$: Through A block (reverse), $s \in [0, f_A]$

### 3.2 Partition Function

$$Q = \frac{1}{V} \int d\mathbf{r} \, q^{0 \rightarrow 1}(\mathbf{r}, s) \cdot q^{1 \rightarrow 0}(\mathbf{r}, f_A - s)$$

The partition function can be computed at any point along the chain (result is independent of $s$).

**Interpretation:**
- $Q > 1$: Potential field favors certain conformations
- $Q < 1$: Potential field restricts conformations
- $Q = 1$: No external field ($w = 0$)

### 3.3 Concentrations

The ensemble-averaged concentration is computed by combining forward and backward propagators:

$$\phi_A(\mathbf{r}) = \frac{1}{Q} \int_0^{f_A} ds \, q^{0 \rightarrow 1}(\mathbf{r}, s) \cdot q^{1 \rightarrow 0}(\mathbf{r}, f_A - s)$$

$$\phi_B(\mathbf{r}) = \frac{1}{Q} \int_0^{f_B} ds \, q^{1 \rightarrow 2}(\mathbf{r}, s) \cdot q^{2 \rightarrow 1}(\mathbf{r}, f_B - s)$$

**Normalization:** $\frac{1}{V} \int \phi_\alpha(\mathbf{r}) \, d\mathbf{r} = f_\alpha$ (block fraction)

---

## 4. Branched Polymers

### 4.1 Graph Representation

Branched polymers are represented as graphs where:
- **Nodes**: Junction points and chain ends
- **Edges**: Polymer blocks connecting nodes

### 4.2 Multi-arm Star Polymer

A 3-arm AB star polymer:

```
       B(4)           B(5)           B(6)
        |              |              |
       A(1)           A(2)           A(3)
         \             |             /
          \            |            /
           -------- (0) center ----
```

**Block definition:**
```python
blocks = [
    ["A", f_A, 0, 1], ["B", f_B, 1, 4],  # Arm 1
    ["A", f_A, 0, 2], ["B", f_B, 2, 5],  # Arm 2
    ["A", f_A, 0, 3], ["B", f_B, 3, 6],  # Arm 3
]
```

**Concentrations:**

$$\phi_A(\mathbf{r}) = \frac{1}{Q} \sum_{(v,u) \in \{(0,1),(0,2),(0,3)\}} \int_0^{f_A} ds \, q^{v \rightarrow u}(\mathbf{r}, s) \cdot q^{u \rightarrow v}(\mathbf{r}, f_A - s)$$

### 4.3 Comb Polymer

A comb polymer with backbone and side chains:

```
    B(3)    B(5)    B(7)
     |       |       |
A(0)-A(1)---A(2)---A(4)---A(6)-A(8)
```

The library automatically handles propagator dependencies using dynamic programming to avoid redundant calculations.

---

## 5. Polymer Mixtures

### 5.1 Multiple Species

For a mixture of polymer species $p = 0, 1, 2, \ldots$:

**Partition functions:** Each species has its own partition function $Q_p$.

**Concentrations:** Total concentration is the weighted sum over all species:

$$\phi_\alpha(\mathbf{r}) = \sum_p \frac{\bar{\phi}_p}{\alpha_p Q_p} \int ds \, q_p^{v \rightarrow u}(\mathbf{r}, s) \cdot q_p^{u \rightarrow v}(\mathbf{r}, f - s)$$

where:
- $\bar{\phi}_p$: volume fraction of species $p$
- $\alpha_p = N_p / N_{ref}$: relative chain length

### 5.2 Example: Diblock + Homopolymer Mixture

```python
# AB diblock (80% volume fraction)
solver.add_polymer(volume_fraction=0.8,
                   blocks=[["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])

# A homopolymer (20% volume fraction)
solver.add_polymer(volume_fraction=0.2,
                   blocks=[["A", 0.5, 0, 1]])
```

---

## 6. Boundary Conditions

### 6.1 Types

| Type | Transform | Physical Meaning |
|------|-----------|------------------|
| Periodic | FFT | Bulk system |
| Reflecting | DCT | Impenetrable wall (zero flux) |
| Absorbing | DST | Reactive surface (zero concentration) |

### 6.2 BC Format

Boundary conditions are specified as a list `[x_low, x_high, y_low, y_high, z_low, z_high]`:

```python
# Periodic in x/y, reflecting in z (confined film)
bc = ["periodic", "periodic", "periodic", "periodic", "reflecting", "reflecting"]
```

---

## 7. Implementation

### 7.1 Using PropagatorSolver

```python
from polymerfts import PropagatorSolver
import numpy as np

# Create solver
solver = PropagatorSolver(
    nx=[64, 64],
    lx=[5.0, 5.0],
    ds=0.01,
    bond_lengths={"A": 1.0, "B": 1.0},
    bc=["periodic"] * 4,
    chain_model="continuous",
    method="pseudospectral",
)

# Add AB diblock copolymer
solver.add_polymer(
    volume_fraction=1.0,
    blocks=[["A", 0.7, 0, 1], ["B", 0.3, 1, 2]]
)

# Set potential fields
w_A = np.sin(np.linspace(0, 2*np.pi, 64))
w_B = -w_A
solver.compute_propagators({"A": w_A.flatten(), "B": w_B.flatten()})

# Get partition function
Q = solver.get_partition_function(polymer=0)

# Compute and get concentrations
solver.compute_concentrations()
phi_A = solver.get_concentration("A")
phi_B = solver.get_concentration("B")
```

### 7.2 Accessing Individual Propagators

With `reduce_memory=False`, individual propagator steps can be accessed:

```python
# Get propagator at specific contour step
# polymer=0, from node v=0 to node u=1, at step s
q = solver.get_propagator(polymer=0, v=0, u=1, step=50)
```

### 7.3 Key Methods

| Method | Description |
|--------|-------------|
| `add_polymer(volume_fraction, blocks)` | Add polymer species |
| `compute_propagators(fields)` | Compute all propagators |
| `get_partition_function(polymer)` | Get $Q$ for polymer species |
| `compute_concentrations()` | Compute all concentrations |
| `get_concentration(monomer_type)` | Get $\phi_\alpha(\mathbf{r})$ |
| `get_propagator(polymer, v, u, step)` | Get propagator at step |

---

## 8. References

1. Matsen, M. W. "The standard Gaussian model for block copolymer melts." *J. Phys.: Condens. Matter* **14**, R21 (2002).

2. Fredrickson, G. H. *The Equilibrium Theory of Inhomogeneous Polymers*. Oxford University Press (2006).

3. Park, S. J., Yong, D., Kim, Y. & Kim, J. U. "Numerical implementation of pseudo-spectral method in self-consistent mean field theory for discrete polymer chains." *J. Chem. Phys.* **150**, 234901 (2019).

4. Yong, D. & Kim, J. U. "Dynamic Programming for Chain Propagator Computation of Branched Block Copolymers in Polymer Field Theory Simulations." *J. Chem. Theory Comput.* **2025**, 21, 3676.
