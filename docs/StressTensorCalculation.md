# Stress Tensor Calculation in Polymer Field Theory

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the calculation of stress in self-consistent field theory (SCFT), following the perturbation theory developed by Tyler and Morse [1]. The stress tensor is essential for optimizing the unit cell dimensions to find equilibrium periodic structures.

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Unit Cell Geometry](#2-unit-cell-geometry)
3. [Stress Derivation](#3-stress-derivation)
4. [Implementation Details](#4-implementation-details)
5. [Box Optimization Algorithm](#5-box-optimization-algorithm)

---

## 1. Theoretical Foundation

### Free Energy and Stress

The stress tensor measures the response of the free energy to changes in lattice parameters. Tyler and Morse [1] showed that for a self-consistent solution of the SCFT equations, the derivative of the free energy with respect to any unit cell parameter $\theta_i$ takes a remarkably simple form:

$$\frac{dF}{d\theta_i} = -k_B T \frac{\partial \ln Q}{\partial \theta_i}$$

where $Q$ is the single-chain partition function and the partial derivative is evaluated under an **affine deformation** of the chemical potential fields (i.e., $\omega(\tilde{\mathbf{r}})$ is held fixed while the unit cell parameters change, where $\tilde{\mathbf{r}}$ is the dimensionless coordinate).

The free energy in SCFT is a functional of the chemical potential fields $\omega$:

$$\frac{F}{k_B T} = -\ln Q - \frac{N}{V} \int d\mathbf{r} \left[ \omega_\alpha(\mathbf{r}) \phi_\alpha(\mathbf{r}) - \frac{1}{2} \chi_{\alpha\beta} \phi_\alpha(\mathbf{r}) \phi_\beta(\mathbf{r}) \right]$$

When differentiating with respect to a unit cell parameter $\theta_i$, one might expect contributions from both:
1. The explicit dependence of $Q$ on the unit cell geometry
2. The implicit dependence through changes in the $\omega$ fields

However, Tyler and Morse showed that at self-consistency, the functional derivative $\delta F / \delta \omega = 0$ vanishes, eliminating the second contribution. The stress is thus determined entirely by how the partition function responds to geometric changes in the simulation box.

### Gaussian Chain Propagator

The polymer behavior is governed by the operator:

$$\hat{H} = -\frac{b^2}{6} \nabla^2 + w(\mathbf{r})$$

In Fourier space, the $\nabla^2$ operator simply becomes $-k^2$. The "bond transition" or "propagator" step between monomers involves a factor of:

$$\exp\left(-\frac{b^2 \Delta s}{6} k^2\right)$$

Whenever you change a unit cell parameter $\theta_i$ (length or angle), you are changing the "size" of the grid in reciprocal space, which changes the value of $k^2$. By the chain rule, the derivative of this propagator will always pull down the factor:

$$-\frac{b^2 \Delta s}{6} \frac{\partial k^2}{\partial \theta_i}$$

---

## 2. Unit Cell Geometry

To handle general unit cells (orthogonal, monoclinic, triclinic, etc.), we use the cell matrix and metric tensor formalism.

### Cell Matrix

The unit cell is defined by a matrix $\mathbf{h} = [\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3]$, where $\mathbf{a}_i$ are the Bravais lattice vectors (column vectors) with lengths $L_i = |\mathbf{a}_i|$.

**Coordinate transformation:**

$$\mathbf{r} = \mathbf{h} \tilde{\mathbf{r}}$$

where $\tilde{\mathbf{r}} \in [0,1]^3$ is the fractional (scaled) coordinate vector.

**Volume:**

$$V = \det(\mathbf{h})$$

### Angle Conventions

The angles between lattice vectors follow the standard crystallographic convention:
- $\alpha$: angle between $\mathbf{a}_2$ and $\mathbf{a}_3$
- $\beta$: angle between $\mathbf{a}_1$ and $\mathbf{a}_3$
- $\gamma$: angle between $\mathbf{a}_1$ and $\mathbf{a}_2$

### Metric Tensor

The **real-space metric tensor** encodes both lengths and angles:

$$g_{ij} = \mathbf{a}_i \cdot \mathbf{a}_j = (\mathbf{h}^T \mathbf{h})_{ij}$$

For a general triclinic cell:

$$g = \begin{pmatrix} L_1^2 & L_1 L_2 \cos\gamma & L_1 L_3 \cos\beta \\ L_1 L_2 \cos\gamma & L_2^2 & L_2 L_3 \cos\alpha \\ L_1 L_3 \cos\beta & L_2 L_3 \cos\alpha & L_3^2 \end{pmatrix}$$

### Reciprocal Space

The reciprocal lattice vectors are:

$$\mathbf{k} = 2\pi (\mathbf{h}^{-1})^T \mathbf{m}$$

where $\mathbf{m} = (m_1, m_2, m_3)$ are integer Miller indices.

**Wavevector magnitude squared:**

The squared magnitude using the inverse metric tensor is:

$$k^2 = (2\pi)^2 \mathbf{m}^T g^{-1} \mathbf{m} = g^{-1}_{11} k_1^2 + g^{-1}_{22} k_2^2 + g^{-1}_{33} k_3^2 + 2g^{-1}_{12} k_1 k_2 + 2g^{-1}_{13} k_1 k_3 + 2g^{-1}_{23} k_2 k_3$$

where we define $k_i = 2\pi m_i$ for convenience.

**Laplacian in scaled coordinates:**

$$\nabla^2 = g^{-1}_{ij} \frac{\partial^2}{\partial \tilde{r}_i \partial \tilde{r}_j}$$

---

## 3. Stress Derivation

### General Formula in Fourier Space

The partition function derivative with respect to any unit cell parameter $\theta$ is computed in Fourier space. We define:
- $C = \frac{b^2 \Delta s}{6(2\pi)^3 V}$ (normalization constant)
- $\Phi(k) = \exp(-b^2 k^2 \Delta s/6)$ (bond transition function for discrete chains)
- $\text{Kern}(\mathbf{k}) = q(\mathbf{k}) q^\dagger(-\mathbf{k}) \Phi(k)$ (kernel function)

**Discrete chain model:**

$$\frac{\partial(Q/V)}{\partial\theta} = -C \sum_{\text{bonds}} \int d\mathbf{k} \, \text{Kern}(\mathbf{k}) \, \frac{\partial k^2}{\partial\theta}$$

**Continuous chain model:**

$$\frac{\partial(Q/V)}{\partial\theta} = -\frac{b^2}{6(2\pi)^3 V} \int_0^1 ds \int d\mathbf{k} \, q(\mathbf{k}, s) \, q^\dagger(\mathbf{k}, s) \, \frac{\partial k^2}{\partial\theta}$$

where $q(\mathbf{k})$ and $q^\dagger(-\mathbf{k})$ are Fourier transforms of the forward and backward propagators.

The key quantity to compute is $\frac{\partial k^2}{\partial \theta}$ for each unit cell parameter $\theta$.

### Metric Tensor Approach

When varying a unit cell parameter $\theta$ (length or angle), $k^2$ changes through the inverse metric tensor $g^{-1}_{ij}$. The procedure is:

1. Calculate how the parameter change affects the metric tensor $g$
2. Use $\frac{\partial g^{-1}}{\partial \theta} = -g^{-1} \frac{\partial g}{\partial \theta} g^{-1}$ to find how $g^{-1}$ changes
3. Compute $\frac{\partial k^2}{\partial \theta}$ from the change in $g^{-1}$

### Derivative with Respect to Length

**Orthogonal case (example):**

For a rectangular system with $\theta = L_x$:

$$k^2 = k_x^2 + k_y^2 + k_z^2 = \left(\frac{2\pi n_x}{L_x}\right)^2 + k_y^2 + k_z^2$$

The derivative is:

$$\frac{\partial k^2}{\partial L_x} = \frac{\partial}{\partial L_x}\left(\frac{2\pi n_x}{L_x}\right)^2 = 2\left(\frac{2\pi n_x}{L_x}\right)\left(-\frac{2\pi n_x}{L_x^2}\right) = -\frac{2}{L_x} k_x^2$$

**Non-orthogonal case:**

For a general triclinic cell, varying $L_1$ affects $g_{11}$, $g_{12}$, and $g_{13}$:

$$\frac{\partial k^2}{\partial L_1} = -2(k_1^2 L_1 + k_1 k_2 L_2 \cos\gamma + k_1 k_3 L_3 \cos\beta)$$

### Derivative with Respect to Angle

To find $\frac{\partial k^2}{\partial \gamma}$, we use the matrix identity for the derivative of an inverse:

$$\frac{\partial g^{-1}}{\partial \gamma} = -g^{-1} \frac{\partial g}{\partial \gamma} g^{-1}$$

Since $k^2 = (2\pi)^2 \mathbf{m}^T g^{-1} \mathbf{m}$, we have:

$$\frac{\partial k^2}{\partial \gamma} = -(2\pi)^2 \mathbf{m}^T \left( g^{-1} \frac{\partial g}{\partial \gamma} g^{-1} \right) \mathbf{m}$$

For the angle $\gamma$ between $\mathbf{a}_1$ and $\mathbf{a}_2$, only $g_{12} = L_1 L_2 \cos\gamma$ depends on $\gamma$:

$$\frac{\partial g_{12}}{\partial \gamma} = -L_1 L_2 \sin\gamma$$

Expanding the matrix product and letting $\mathbf{k} = 2\pi g^{-1} \mathbf{m}$ be the reciprocal wavevector:

$$\frac{\partial k^2}{\partial \gamma} = 2 k_1 k_2 (L_1 L_2 \sin\gamma)$$

**Note:** Here $k_1$ and $k_2$ are the components of the reciprocal vector $\mathbf{k}$ in the non-orthogonal basis.

### Summary of Stress Parameters

| Parameter $\theta_i$ | Influence on Metric Tensor $g$ | Derivative Term $\frac{\partial k^2}{\partial \theta_i}$ |
|---------------------|-------------------------------|--------------------------------------------------------|
| $L_1$ | Changes $g_{11}, g_{12}, g_{13}$ | $-2(k_1^2 L_1 + k_1 k_2 L_2 \cos\gamma + k_1 k_3 L_3 \cos\beta)$ |
| $L_2$ | Changes $g_{22}, g_{12}, g_{23}$ | $-2(k_2^2 L_2 + k_1 k_2 L_1 \cos\gamma + k_2 k_3 L_3 \cos\alpha)$ |
| $L_3$ | Changes $g_{33}, g_{13}, g_{23}$ | $-2(k_3^2 L_3 + k_1 k_3 L_1 \cos\beta + k_2 k_3 L_2 \cos\alpha)$ |
| $\alpha$ (between $L_2, L_3$) | Changes off-diagonal $g_{23}$ | $2 k_2 k_3 L_2 L_3 \sin\alpha$ |
| $\beta$ (between $L_1, L_3$) | Changes off-diagonal $g_{13}$ | $2 k_1 k_3 L_1 L_3 \sin\beta$ |
| $\gamma$ (between $L_1, L_2$) | Changes off-diagonal $g_{12}$ | $2 k_1 k_2 L_1 L_2 \sin\gamma$ |

---

## 4. Implementation Details

### Stress Array Convention

The stress is stored as a 6-component array following Voigt notation:

| Index | Component | Drives Optimization of |
|-------|-----------|------------------------|
| 0 | $\sigma_1$ | $L_1$ (length of $\mathbf{a}_1$) |
| 1 | $\sigma_2$ | $L_2$ (length of $\mathbf{a}_2$) |
| 2 | $\sigma_3$ | $L_3$ (length of $\mathbf{a}_3$) |
| 3 | $\sigma_{12}$ | $\gamma$ (angle between $\mathbf{a}_1$ and $\mathbf{a}_2$) |
| 4 | $\sigma_{13}$ | $\beta$ (angle between $\mathbf{a}_1$ and $\mathbf{a}_3$) |
| 5 | $\sigma_{23}$ | $\alpha$ (angle between $\mathbf{a}_2$ and $\mathbf{a}_3$) |

For 2D systems, the layout is [σ₁, σ₂, σ₁₂, 0, 0, 0], so indices 0, 1, and 2 are used. For 1D systems, only index 0 is used.

### Fourier Basis Arrays

The code precomputes arrays that decompose $k^2$ into components (with $k_i = 2\pi m_i$).

**Diagonal basis arrays** (`fourier_basis_x`, `fourier_basis_y`, `fourier_basis_z`) — stored values include the $(2\pi)^2$ factor:

$$\text{fourier-basis-x}[m] = (2\pi)^2 \, g^{-1}_{11} \, m_1^2 = g^{-1}_{11} k_1^2$$

$$\text{fourier-basis-y}[m] = (2\pi)^2 \, g^{-1}_{22} \, m_2^2 = g^{-1}_{22} k_2^2$$

$$\text{fourier-basis-z}[m] = (2\pi)^2 \, g^{-1}_{33} \, m_3^2 = g^{-1}_{33} k_3^2$$

**Cross-term basis arrays** (`fourier_basis_xy`, `fourier_basis_xz`, `fourier_basis_yz`):

$$\text{fourier-basis-xy}[m] = 2 (2\pi)^2 \, g^{-1}_{12} \, m_1 m_2 = 2 g^{-1}_{12} k_1 k_2$$

$$\text{fourier-basis-xz}[m] = 2 (2\pi)^2 \, g^{-1}_{13} \, m_1 m_3 = 2 g^{-1}_{13} k_1 k_3$$

$$\text{fourier-basis-yz}[m] = 2 (2\pi)^2 \, g^{-1}_{23} \, m_2 m_3 = 2 g^{-1}_{23} k_2 k_3$$

The factor of 2 in cross-terms accounts for the symmetric sum. The cross-term arrays are directly used in the angle stress calculation:

$$\sigma_{12} \propto L_1 L_2 \sin\gamma \cdot \sum_{\mathbf{k}} \text{Kern}(\mathbf{k}) \cdot \text{fourier-basis-xy}$$

### Chain Model Dependence

The bond factor $\Phi(k)$ differs between chain models:

| Chain Model | Bond Factor $\Phi(k)$ | Reason |
|-------------|----------------------|--------|
| Continuous | $1$ | Diffusion propagator already in propagator computation |
| Discrete | $\exp(-b^2 k^2 \Delta s/6)$ | Explicit bond connectivity |
| Half-bond | $\exp(-b^2 k^2 \Delta s/12)$ | Chain ends or junctions |

### Boundary Condition Types

| Boundary Condition | Transform | Wavenumber Factor |
|-------------------|-----------|-------------------|
| Periodic | FFT | $(2\pi m/L)^2$ |
| Reflecting (Neumann) | DCT-II | $(\pi m/L)^2$ |
| Absorbing (Dirichlet) | DST-II | $(\pi m/L)^2$ |

**Note:** Non-periodic boundary conditions currently require orthogonal grids.

### Implementation Steps

**Step 1: Per-Bond/Segment Contribution**

Compute the kernel product in Fourier space:

$$\text{Kern}(\mathbf{k}) = q(\mathbf{k}) \cdot q^\dagger(-\mathbf{k}) \cdot \Phi(k)$$

**Step 2: Stress Accumulation**

For each stress component, accumulate over all wavevectors:

$$\sigma_i \propto \sum_{\mathbf{k}} \text{Kern}(\mathbf{k}) \cdot (\text{corresponding basis array})$$

**Step 3: Contour Integration** (Simpson's rule)

$$\frac{\partial Q}{\partial \theta} \propto \sum_{\text{segments}} w_i \, S_\theta(i)$$

where $w_i$ are Simpson's rule weights.

---

## 5. Box Optimization Algorithm

During SCFT iteration with `box_is_altering=True`, the lattice parameters are updated using gradient descent:

**For lengths:**

$$L_i^{(n+1)} = L_i^{(n)} - \eta \cdot \sigma_i, \quad i = 1, 2, 3$$

**For angles:**

$$\alpha^{(n+1)} = \alpha^{(n)} - \eta \cdot \sigma_{23}$$

$$\beta^{(n+1)} = \beta^{(n)} - \eta \cdot \sigma_{13}$$

$$\gamma^{(n+1)} = \gamma^{(n)} - \eta \cdot \sigma_{12}$$

where $\eta$ is the `scale_stress` parameter. At equilibrium, all stress components vanish:

$$\sigma_i = 0, \quad \sigma_{ij} = 0$$

### Simultaneous Iteration

Tyler and Morse [1] proposed iterating the unit cell parameters simultaneously with the SCFT equations, rather than using nested optimization loops. This approach:

1. Adds the unit cell parameters $\theta$ (lengths and angles) to the unknowns
2. Adds the zero-stress conditions $\partial \ln Q / \partial \theta = 0$ to the residual equations
3. Updates both fields and unit cell parameters in each iteration

This simultaneous iteration typically converges in approximately the same number of iterations as solving the SCFT equations in a fixed unit cell.

### Code Example

```python
# In SCFT iteration loop
if box_is_altering:
    stress = propagator_solver.get_stress()
    # Update lengths (indices 0, 1, 2)
    for i in range(dim):
        lx[i] -= scale_stress * stress[i]
    # Update angles (indices 3, 4, 5 for gamma, beta, alpha)
    if dim >= 2:
        angles[0] -= scale_stress * stress[3]  # gamma
    if dim == 3:
        angles[1] -= scale_stress * stress[4]  # beta
        angles[2] -= scale_stress * stress[5]  # alpha
```

---

## References

1. Tyler, C. A. & Morse, D. C. "Stress in self-consistent-field theory." *Macromolecules* **36**, 8184-8188 (2003).

2. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).

3. Beardsley, T. M. & Matsen, M. W. "Fluctuation correction for the order-disorder transition of diblock copolymer melts." *J. Chem. Phys.* **154**, 124902 (2021).
