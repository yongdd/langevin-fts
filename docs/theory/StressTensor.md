# Stress Tensor Calculation in Polymer Field Theory

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

At self-consistency, the functional derivative $\delta F / \delta \omega = 0$ vanishes, so the stress is determined entirely by how the partition function responds to geometric changes in the simulation box.

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

The unit cell is defined by a matrix $\mathbf{h} = [\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3]$, where $\mathbf{a}_i$ are the Bravais lattice vectors with lengths $L_i = |\mathbf{a}_i|$.

The angles between lattice vectors follow the standard crystallographic convention:
- $\alpha$: angle between $\mathbf{a}_2$ and $\mathbf{a}_3$
- $\beta$: angle between $\mathbf{a}_1$ and $\mathbf{a}_3$
- $\gamma$: angle between $\mathbf{a}_1$ and $\mathbf{a}_2$

The **metric tensor** encodes both lengths and angles:

$$g_{ij} = \mathbf{a}_i \cdot \mathbf{a}_j = (\mathbf{h}^T \mathbf{h})_{ij}$$

For a general triclinic cell:

$$g = \begin{pmatrix} L_1^2 & L_1 L_2 \cos\gamma & L_1 L_3 \cos\beta \\ L_1 L_2 \cos\gamma & L_2^2 & L_2 L_3 \cos\alpha \\ L_1 L_3 \cos\beta & L_2 L_3 \cos\alpha & L_3^2 \end{pmatrix}$$

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

**Important:** The components $v_i$ have units of $1/L^2$, not $1/L$ like Cartesian wavevector components.

The squared wavevector magnitude can be written as:

$$k^2 = 2\pi \, \mathbf{m}^T \mathbf{v} = 2\pi (m_1 v_1 + m_2 v_2 + m_3 v_3)$$

### Derivative Formulas

When varying a unit cell parameter $\theta_i$, $k^2$ changes through the inverse metric tensor. Using the identity $\frac{\partial g^{-1}}{\partial \theta} = -g^{-1} \frac{\partial g}{\partial \theta} g^{-1}$:

$$\frac{\partial k^2}{\partial \theta_i} = -(2\pi)^2 \mathbf{m}^T \left( g^{-1} \frac{\partial g}{\partial \theta_i} g^{-1} \right) \mathbf{m}$$

For a general triclinic cell, varying $L_1$ affects $g_{11}$, $g_{12}$, and $g_{13}$:

$$\frac{\partial k^2}{\partial L_1} = -2(v_1^2 L_1 + v_1 v_2 L_2 \cos\gamma + v_1 v_3 L_3 \cos\beta)$$

For the angle $\gamma$, only $g_{12} = L_1 L_2 \cos\gamma$ depends on $\gamma$:

$$\frac{\partial k^2}{\partial \gamma} = 2 v_1 v_2 L_1 L_2 \sin\gamma$$

### Summary Table

The formulas use the **deformation vector** $\mathbf{v} = 2\pi g^{-1} \mathbf{m}$ (units: $1/L^2$):

| Parameter $\theta_i$ | Derivative $\frac{\partial k^2}{\partial \theta_i}$ |
|---------------------|---------------------------------------------------|
| $L_1$ | $-2(v_1^2 L_1 + v_1 v_2 L_2 \cos\gamma + v_1 v_3 L_3 \cos\beta)$ |
| $L_2$ | $-2(v_2^2 L_2 + v_1 v_2 L_1 \cos\gamma + v_2 v_3 L_3 \cos\alpha)$ |
| $L_3$ | $-2(v_3^2 L_3 + v_1 v_3 L_1 \cos\beta + v_2 v_3 L_2 \cos\alpha)$ |
| $\alpha$ | $2 v_2 v_3 L_2 L_3 \sin\alpha$ |
| $\beta$ | $2 v_1 v_3 L_1 L_3 \sin\beta$ |
| $\gamma$ | $2 v_1 v_2 L_1 L_2 \sin\gamma$ |

---

## 4. Implementation

### Stress Array Convention

The stress is stored as a 6-component array (Voigt notation):

| Index | Component | Parameter |
|-------|-----------|-----------|
| 0 | $\sigma_1$ | $L_1$ |
| 1 | $\sigma_2$ | $L_2$ |
| 2 | $\sigma_3$ | $L_3$ |
| 3 | $\sigma_{12}$ | $\gamma$ |
| 4 | $\sigma_{13}$ | $\beta$ |
| 5 | $\sigma_{23}$ | $\alpha$ |

For 2D systems, indices 0, 1, 2 are used. For 1D systems, only index 0 is used.

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

$$L_i^{(n+1)} = L_i^{(n)} - \eta \cdot \sigma_i$$

$$\alpha^{(n+1)} = \alpha^{(n)} - \eta \cdot \sigma_{23}, \quad \beta^{(n+1)} = \beta^{(n)} - \eta \cdot \sigma_{13}, \quad \gamma^{(n+1)} = \gamma^{(n)} - \eta \cdot \sigma_{12}$$

where $\eta$ is the `scale_stress` parameter. At equilibrium, all stress components vanish.

Tyler and Morse [1] proposed iterating the unit cell parameters simultaneously with the SCFT equations, which typically converges in approximately the same number of iterations as solving in a fixed unit cell.

---

## 6. References

1. Tyler, C. A. & Morse, D. C. "Stress in self-consistent-field theory." *Macromolecules* **36**, 8184-8188 (2003).

2. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).

3. Beardsley, T. M. & Matsen, M. W. "Fluctuation correction for the order–disorder transition of diblock copolymer melts." *J. Chem. Phys.* **154**, 124902 (2021).
