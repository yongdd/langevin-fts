# Stress Tensor Calculation in Polymer Field Theory

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the calculation of stress in self-consistent field theory (SCFT), following the perturbation theory developed by Tyler and Morse [1]. The stress tensor is essential for optimizing the unit cell dimensions to find equilibrium periodic structures.

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Unit Cell Geometry](#2-unit-cell-geometry)
3. [Derivative of k²](#3-derivative-of-k²)
4. [Implementation](#4-implementation)
5. [Box Optimization](#5-box-optimization)
6. [References](#6-references)

---

## 1. Theoretical Foundation

### Free Energy and Stress

The stress tensor measures the response of the free energy to changes in lattice parameters. Tyler and Morse [1] showed that for a self-consistent solution of the SCFT equations, the derivative of the free energy with respect to any unit cell parameter $\theta_i$ takes a simple form:

$$\frac{dF}{d\theta_i} = -k_B T \frac{\partial \ln Q}{\partial \theta_i}$$

where $Q$ is the single-chain partition function and the partial derivative is evaluated with the field values held fixed on the computational grid.

At self-consistency, the functional derivative $\delta F / \delta w = 0$ vanishes, so the stress is determined entirely by how the partition function responds to geometric changes in the simulation box.

### Chain Propagator

The polymer behavior is governed by the operator:

$$\hat{H} = -\frac{b^2}{6} \nabla^2 + w(\mathbf{r})$$

In Fourier space, the propagator step between monomers involves a factor of:

$$\exp\left(-\frac{b^2 \Delta s}{6} k^2\right)$$

When a unit cell parameter $\theta_i$ changes, the value of $k^2$ changes. The derivative of this propagator pulls down the factor:

$$-\frac{b^2 \Delta s}{6} \frac{\partial k^2}{\partial \theta_i}$$

---

## 2. Unit Cell Geometry

### Cell Matrix and Metric Tensor

The unit cell is defined by a matrix $\mathbf{h} = [\mathbf{a}, \mathbf{b}, \mathbf{c}]$, where $\mathbf{a}$, $\mathbf{b}$, $\mathbf{c}$ are the Bravais lattice vectors with lengths $L_a$, $L_b$, $L_c$.

The angles between lattice vectors follow the standard crystallographic convention:
- $\alpha$: angle between $\mathbf{b}$ and $\mathbf{c}$
- $\beta$: angle between $\mathbf{a}$ and $\mathbf{c}$
- $\gamma$: angle between $\mathbf{a}$ and $\mathbf{b}$

The **metric tensor** encodes both lengths and angles:

$$g_{ij} = (\mathbf{h}^T \mathbf{h})_{ij}$$

For a general triclinic cell:

$$g = \begin{pmatrix} L_a^2 & L_a L_b \cos\gamma & L_a L_c \cos\beta \\ L_a L_b \cos\gamma & L_b^2 & L_b L_c \cos\alpha \\ L_a L_c \cos\beta & L_b L_c \cos\alpha & L_c^2 \end{pmatrix}$$

### Reciprocal Space

The reciprocal lattice vectors are:

$$\mathbf{k} = 2\pi (\mathbf{h}^{-1})^T \mathbf{m}$$

where $\mathbf{m} = (m_1, m_2, m_3)$ are integer Miller indices.

The squared wavevector magnitude using the inverse metric tensor is:

$$k^2 = (2\pi)^2 \mathbf{m}^T g^{-1} \mathbf{m}$$

---

## 3. Derivative of k²

### Deformation Vector

For numerical implementation, we define the **deformation vector**:

$$\mathbf{v} = 2\pi g^{-1} \mathbf{m}$$

**Important:** The components $v_a$, $v_b$, $v_c$ have units of $1/L^2$, not $1/L$ like Cartesian wavevector components.

The squared wavevector magnitude can be written as:

$$k^2 = 2\pi \, \mathbf{m}^T \mathbf{v} = 2\pi (m_a v_a + m_b v_b + m_c v_c)$$

### Derivative Formulas

When varying a unit cell parameter $\theta_i$, $k^2$ changes through the inverse metric tensor. Using the identity $\frac{\partial g^{-1}}{\partial \theta} = -g^{-1} \frac{\partial g}{\partial \theta} g^{-1}$:

$$\frac{\partial k^2}{\partial \theta_i} = -(2\pi)^2 \mathbf{m}^T \left( g^{-1} \frac{\partial g}{\partial \theta_i} g^{-1} \right) \mathbf{m}$$

For a general triclinic cell, varying $L_a$ affects $g_{aa}$, $g_{ab}$, and $g_{ac}$:

$$\frac{\partial k^2}{\partial L_a} = -2(v_a^2 L_a + v_a v_b L_b \cos\gamma + v_a v_c L_c \cos\beta)$$

For the angle $\gamma$, only $g_{ab} = L_a L_b \cos\gamma$ depends on $\gamma$:

$$\frac{\partial k^2}{\partial \gamma} = 2 v_a v_b L_a L_b \sin\gamma$$

### Summary Table

The formulas use the **deformation vector** $\mathbf{v} = 2\pi g^{-1} \mathbf{m}$ (units: $1/L^2$):

| Parameter $\theta_i$ | Derivative $\frac{\partial k^2}{\partial \theta_i}$ |
|---------------------|---------------------------------------------------|
| $L_a$ | $-2(v_a^2 L_a + v_a v_b L_b \cos\gamma + v_a v_c L_c \cos\beta)$ |
| $L_b$ | $-2(v_b^2 L_b + v_a v_b L_a \cos\gamma + v_b v_c L_c \cos\alpha)$ |
| $L_c$ | $-2(v_c^2 L_c + v_a v_c L_a \cos\beta + v_b v_c L_b \cos\alpha)$ |
| $\alpha$ | $2 v_b v_c L_b L_c \sin\alpha$ |
| $\beta$ | $2 v_a v_c L_a L_c \sin\beta$ |
| $\gamma$ | $2 v_a v_b L_a L_b \sin\gamma$ |

---

## 4. Implementation

### Stress Array Convention

The stress is stored as a 6-component array (Voigt notation):

| Index | Component | Parameter |
|-------|-----------|-----------|
| 0 | $\sigma_a$ | $L_a$ |
| 1 | $\sigma_b$ | $L_b$ |
| 2 | $\sigma_c$ | $L_c$ |
| 3 | $\sigma_{ab}$ | $\gamma$ |
| 4 | $\sigma_{ac}$ | $\beta$ |
| 5 | $\sigma_{bc}$ | $\alpha$ |

For 2D systems, indices 0, 1 are used (and 3 for non-orthogonal cells). For 1D systems, only index 0 is used.

### Chain Model Dependence

The bond factor $\Phi(k)$ differs between chain models:

| Chain Model | Bond Factor $\Phi(k)$ |
|-------------|----------------------|
| Continuous | $1$ |
| Discrete | $\exp(-b^2 k^2 \Delta s/6)$ |
| Half-bond | $\exp(-b^2 k^2 \Delta s/12)$ |

### Boundary Conditions

| Boundary Condition | Transform | Wavenumber |
|-------------------|-----------|------------|
| Periodic | FFT | $(2\pi m/L)^2$ |
| Reflecting | DCT-II | $(\pi m/L)^2$ |
| Absorbing | DST-II | $(\pi m/L)^2$ |

Non-periodic boundary conditions require orthogonal grids.

### Computation Steps

**Step 1:** Compute the kernel in Fourier space:

$$\text{Kern}(\mathbf{k}) = q(\mathbf{k}) \cdot q^\dagger(-\mathbf{k}) \cdot \Phi(k)$$

**Step 2:** Accumulate stress over all wavevectors:

$$\sigma_i \propto \sum_{\mathbf{k}} \text{Kern}(\mathbf{k}) \cdot \frac{\partial k^2}{\partial \theta_i}$$

**Step 3:** Integrate over contour (Simpson's rule for continuous chains).

---

## 5. Box Optimization

During SCFT iteration with `box_is_altering=True`, the lattice parameters are updated using gradient descent:

$$L_a^{(n+1)} = L_a^{(n)} - \eta \cdot \sigma_a, \quad L_b^{(n+1)} = L_b^{(n)} - \eta \cdot \sigma_b, \quad L_c^{(n+1)} = L_c^{(n)} - \eta \cdot \sigma_c$$

$$\alpha^{(n+1)} = \alpha^{(n)} - \eta \cdot \sigma_{bc}, \quad \beta^{(n+1)} = \beta^{(n)} - \eta \cdot \sigma_{ac}, \quad \gamma^{(n+1)} = \gamma^{(n)} - \eta \cdot \sigma_{ab}$$

where $\eta$ is the `scale_stress` parameter. At equilibrium, all stress components vanish.

Tyler and Morse [1] proposed iterating the unit cell parameters simultaneously with the SCFT equations, which typically converges in approximately the same number of iterations as solving in a fixed unit cell.

---

## 6. References

1. Tyler, C. A. & Morse, D. C. "Stress in self-consistent-field theory." *Macromolecules* **36**, 8184-8188 (2003).

2. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).

3. Beardsley, T. M. & Matsen, M. W. "Fluctuation correction for the order–disorder transition of diblock copolymer melts." *J. Chem. Phys.* **154**, 124902 (2021).
