# Stress Tensor Calculation in Polymer Field Theory

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the calculation of stress in self-consistent field theory (SCFT), following the perturbation theory developed by Tyler and Morse [1]. The stress tensor is essential for optimizing the unit cell dimensions to find equilibrium periodic structures.

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Unit Cell Geometry](#2-unit-cell-geometry)
3. [Stress Formula](#3-stress-formula)
4. [Implementation Details](#4-implementation-details)
5. [Box Optimization Algorithm](#5-box-optimization-algorithm)

---

## 1. Theoretical Foundation

The stress tensor measures the response of the free energy to changes in lattice parameters. Tyler and Morse [1] showed that for a self-consistent solution of the SCFT equations, the derivative of the free energy with respect to any unit cell parameter $\theta_i$ takes a remarkably simple form:

$$\frac{dF}{d\theta_i} = -k_B T \frac{\partial \ln Q}{\partial \theta_i}$$

where $Q$ is the single-chain partition function and the partial derivative is evaluated under an **affine deformation** of the chemical potential fields (i.e., $\omega(\tilde{\boldsymbol{r}})$ is held fixed while the unit cell parameters change, where $\tilde{\boldsymbol{r}}$ is the dimensionless coordinate).

### Why This Result is Non-Trivial

The free energy in SCFT is a functional of the chemical potential fields $\omega$:

$$\frac{F}{k_B T} = -\ln Q - \frac{N}{V} \int d\boldsymbol{r} \left[ \omega_\alpha(\boldsymbol{r}) \phi_\alpha(\boldsymbol{r}) - \frac{1}{2} \chi_{\alpha\beta} \phi_\alpha(\boldsymbol{r}) \phi_\beta(\boldsymbol{r}) \right]$$

When differentiating with respect to a unit cell parameter $\theta_i$, one might expect contributions from both:
1. The explicit dependence of $Q$ on the unit cell geometry
2. The implicit dependence through changes in the $\omega$ fields

However, Tyler and Morse showed that at self-consistency, the functional derivative $\delta F / \delta \omega = 0$ vanishes, eliminating the second contribution. The stress is thus determined entirely by how the partition function responds to geometric changes in the simulation box.

---

## 2. Unit Cell Geometry

To handle general unit cells (orthogonal, monoclinic, triclinic, etc.), we use the cell matrix and metric tensor formalism.

### Cell Matrix

The unit cell is defined by a matrix $\boldsymbol{h} = [\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3]$, where $\boldsymbol{a}_i$ are the Bravais lattice vectors (column vectors) with lengths $L_i = |\boldsymbol{a}_i|$.

**Coordinate transformation:**

$$\boldsymbol{r} = \boldsymbol{h} \tilde{\boldsymbol{r}}$$

where $\tilde{\boldsymbol{r}} \in [0,1]^3$ is the fractional (scaled) coordinate vector.

**Volume:**

$$V = \det(\boldsymbol{h})$$

### Angle Conventions

The angles between lattice vectors follow the standard crystallographic convention:
- $\alpha$: angle between $\boldsymbol{a}_2$ and $\boldsymbol{a}_3$
- $\beta$: angle between $\boldsymbol{a}_1$ and $\boldsymbol{a}_3$
- $\gamma$: angle between $\boldsymbol{a}_1$ and $\boldsymbol{a}_2$

### Metric Tensor

The **real-space metric tensor** encodes both lengths and angles:

$$g_{ij} = \boldsymbol{a}_i \cdot \boldsymbol{a}_j = (\boldsymbol{h}^T \boldsymbol{h})_{ij}$$

For a general triclinic cell:

$$g = \begin{pmatrix} L_1^2 & L_1 L_2 \cos\gamma & L_1 L_3 \cos\beta \\ L_1 L_2 \cos\gamma & L_2^2 & L_2 L_3 \cos\alpha \\ L_1 L_3 \cos\beta & L_2 L_3 \cos\alpha & L_3^2 \end{pmatrix}$$

### Reciprocal Space

The reciprocal lattice vectors are:

$$\boldsymbol{k} = 2\pi (\boldsymbol{h}^{-1})^T \boldsymbol{m}$$

where $\boldsymbol{m} = (m_1, m_2, m_3)$ are integer Miller indices.

**Wavevector magnitude squared:**

We define wavevector components $k_i = 2\pi m_i$ for convenience. The squared magnitude is:

$$k^2 = |\boldsymbol{k}|^2 = g^{-1}_{11} k_1^2 + g^{-1}_{22} k_2^2 + g^{-1}_{33} k_3^2 + 2g^{-1}_{12} k_1 k_2 + 2g^{-1}_{13} k_1 k_3 + 2g^{-1}_{23} k_2 k_3$$

**Laplacian in scaled coordinates:**

$$\nabla^2 = g^{-1}_{ij} \frac{\partial^2}{\partial \tilde{r}_i \partial \tilde{r}_j}$$

### Orthogonal vs Non-Orthogonal Systems

| Feature | Orthogonal | Non-Orthogonal |
|---------|------------|----------------|
| Geometry | $L_1, L_2, L_3$ | Matrix $\boldsymbol{h}$ (3 lengths + 3 angles) |
| Metric tensor | Diagonal: $\text{diag}(L_1^2, L_2^2, L_3^2)$ | Full symmetric matrix $g_{ij}$ |
| Laplacian | $\partial_1^2 + \partial_2^2 + \partial_3^2$ | $g^{-1}_{ij} \partial_i \partial_j$ |
| $k^2$ | $k_1^2/L_1^2 + k_2^2/L_2^2 + k_3^2/L_3^2$ | $g^{-1}_{ij} k_i k_j$ |

---

## 3. Stress Formula

### General Formula in Fourier Space

The partition function derivative with respect to any unit cell parameter $\theta$ is computed in Fourier space. We define:
- $C = \frac{b^2 \Delta s}{6(2\pi)^3 V}$ (normalization constant)
- $\Phi(k) = \exp(-b^2 k^2 \Delta s/6)$ (bond transition function for discrete chains)
- $\text{Kern}(\boldsymbol{k}) = q(\boldsymbol{k}) q^\dagger(-\boldsymbol{k}) \Phi(k)$ (kernel function)

**Discrete chain model:**

$$\frac{\partial(Q/V)}{\partial\theta} = -C \sum_{\text{bonds}} \int d\boldsymbol{k} \, \text{Kern}(\boldsymbol{k}) \, \frac{\partial k^2}{\partial\theta}$$

**Continuous chain model:**

$$\frac{\partial(Q/V)}{\partial\theta} = -\frac{b^2}{6(2\pi)^3 V} \int_0^1 ds \int d\boldsymbol{k} \, q(\boldsymbol{k}, s) \, q^\dagger(\boldsymbol{k}, s) \, \frac{\partial k^2}{\partial\theta}$$

where $q(\boldsymbol{k})$ and $q^\dagger(-\boldsymbol{k})$ are Fourier transforms of the forward and backward propagators.

### Derivatives with Respect to Lengths

For the lattice lengths $L_1, L_2, L_3$:

$$\frac{\partial(Q/V)}{\partial L_1} = -C \sum_{\text{bonds}} \int d\boldsymbol{k} \, \text{Kern}(\boldsymbol{k}) \, \frac{\partial k^2}{\partial L_1}$$

The derivative $\partial k^2 / \partial L_i$ depends on the specific form of the metric tensor.

### Derivatives with Respect to Angles

For the angles, the derivatives of $k^2$ take simple forms:

$$\frac{\partial k^2}{\partial \alpha} = 2 k_2 k_3 (L_2 L_3 \sin\alpha)$$

$$\frac{\partial k^2}{\partial \beta} = 2 k_1 k_3 (L_1 L_3 \sin\beta)$$

$$\frac{\partial k^2}{\partial \gamma} = 2 k_1 k_2 (L_1 L_2 \sin\gamma)$$

These can be rewritten using the Fourier basis cross-term arrays (see Section 4):

$$\frac{\partial(Q/V)}{\partial \alpha} = -C L_2 L_3 \sin\alpha \sum_{\text{bonds}} \int d\boldsymbol{k} \, \text{Kern}(\boldsymbol{k}) \cdot 2 k_2 k_3$$

$$\frac{\partial(Q/V)}{\partial \beta} = -C L_1 L_3 \sin\beta \sum_{\text{bonds}} \int d\boldsymbol{k} \, \text{Kern}(\boldsymbol{k}) \cdot 2 k_1 k_3$$

$$\frac{\partial(Q/V)}{\partial \gamma} = -C L_1 L_2 \sin\gamma \sum_{\text{bonds}} \int d\boldsymbol{k} \, \text{Kern}(\boldsymbol{k}) \cdot 2 k_1 k_2$$

### Stress Parameters

| Parameter $\theta$ | Physical Meaning | Derivative Term |
|-------------------|------------------|-----------------|
| $L_1$ | Length of $\boldsymbol{a}_1$ | $\partial k^2/\partial L_1$ |
| $L_2$ | Length of $\boldsymbol{a}_2$ | $\partial k^2/\partial L_2$ |
| $L_3$ | Length of $\boldsymbol{a}_3$ | $\partial k^2/\partial L_3$ |
| $\gamma$ | Angle between $\boldsymbol{a}_1$ and $\boldsymbol{a}_2$ | $2 k_1 k_2 (L_1 L_2 \sin\gamma)$ |
| $\beta$ | Angle between $\boldsymbol{a}_1$ and $\boldsymbol{a}_3$ | $2 k_1 k_3 (L_1 L_3 \sin\beta)$ |
| $\alpha$ | Angle between $\boldsymbol{a}_2$ and $\boldsymbol{a}_3$ | $2 k_2 k_3 (L_2 L_3 \sin\alpha)$ |

### Cell Matrix Update Rule

$$\boldsymbol{h}^{\text{new}} = \boldsymbol{h}^{\text{old}} - \lambda \frac{\partial F}{\partial \boldsymbol{h}}$$

To maintain a specific crystal symmetry (e.g., monoclinic), only the relevant components of $\boldsymbol{h}$ are updated.

---

## 4. Implementation Details

### Stress Array Convention

The stress is stored as a 6-component array following Voigt notation:

| Index | Component | Drives Optimization of |
|-------|-----------|------------------------|
| 0 | $\sigma_1$ | $L_1$ (length of $\boldsymbol{a}_1$) |
| 1 | $\sigma_2$ | $L_2$ (length of $\boldsymbol{a}_2$) |
| 2 | $\sigma_3$ | $L_3$ (length of $\boldsymbol{a}_3$) |
| 3 | $\sigma_{12}$ | $\gamma$ (angle between $\boldsymbol{a}_1$ and $\boldsymbol{a}_2$) |
| 4 | $\sigma_{13}$ | $\beta$ (angle between $\boldsymbol{a}_1$ and $\boldsymbol{a}_3$) |
| 5 | $\sigma_{23}$ | $\alpha$ (angle between $\boldsymbol{a}_2$ and $\boldsymbol{a}_3$) |

For 2D systems, the layout is [σ₁, σ₂, σ₁₂, 0, 0, 0], so indices 0, 1, and 2 are used. For 1D systems, only index 0 is used.

### Fourier Basis Arrays

The code precomputes arrays that decompose $k^2$ into components. With $k_i = 2\pi m_i$ where $m_i$ are integer Miller indices:

$$k^2 = g^{-1}_{11} k_1^2 + g^{-1}_{22} k_2^2 + g^{-1}_{33} k_3^2 + 2g^{-1}_{12} k_1 k_2 + 2g^{-1}_{13} k_1 k_3 + 2g^{-1}_{23} k_2 k_3$$

**Diagonal basis arrays** — stored values include the $(2\pi)^2$ factor:

$$\texttt{fourier\_basis\_x}[m] = (2\pi)^2 \, g^{-1}_{11} \, m_1^2 = g^{-1}_{11} k_1^2$$
$$\texttt{fourier\_basis\_y}[m] = (2\pi)^2 \, g^{-1}_{22} \, m_2^2 = g^{-1}_{22} k_2^2$$
$$\texttt{fourier\_basis\_z}[m] = (2\pi)^2 \, g^{-1}_{33} \, m_3^2 = g^{-1}_{33} k_3^2$$

**Cross-term basis arrays:**

$$\texttt{fourier\_basis\_xy}[m] = 2 (2\pi)^2 \, g^{-1}_{12} \, m_1 m_2 = 2 g^{-1}_{12} k_1 k_2$$
$$\texttt{fourier\_basis\_xz}[m] = 2 (2\pi)^2 \, g^{-1}_{13} \, m_1 m_3 = 2 g^{-1}_{13} k_1 k_3$$
$$\texttt{fourier\_basis\_yz}[m] = 2 (2\pi)^2 \, g^{-1}_{23} \, m_2 m_3 = 2 g^{-1}_{23} k_2 k_3$$

The factor of 2 in cross-terms accounts for the symmetric sum. The cross-term arrays are directly used in the angle stress calculation:

$$\sigma_{12} \propto L_1 L_2 \sin\gamma \cdot \sum_{\boldsymbol{k}} \text{Kern}(\boldsymbol{k}) \cdot \texttt{fourier\_basis\_xy}$$

### Boundary Condition Types

| Boundary Condition | Transform | Wavenumber Factor |
|-------------------|-----------|-------------------|
| Periodic | FFT | $(2\pi m/L)^2$ |
| Reflecting (Neumann) | DCT-II | $(\pi m/L)^2$ |
| Absorbing (Dirichlet) | DST-II | $(\pi m/L)^2$ |

**Note:** Non-periodic boundary conditions currently require orthogonal grids.

### Chain Model Dependence

The bond factor $\Phi(k)$ differs between chain models:

| Chain Model | Bond Factor $\Phi(k)$ | Reason |
|-------------|----------------------|--------|
| Continuous | $1$ | Diffusion propagator already in propagator computation |
| Discrete | $\exp(-b^2 k^2 \Delta s/6)$ | Explicit bond connectivity |
| Half-bond | $\exp(-b^2 k^2 \Delta s/12)$ | Chain ends or junctions |

### Implementation Steps

**Step 1: Per-Bond/Segment Contribution**

Compute the kernel product in Fourier space:

$$\text{Kern}(\boldsymbol{k}) = q(\boldsymbol{k}) \cdot q^\dagger(-\boldsymbol{k}) \cdot \Phi(k)$$

**Step 2: Stress Accumulation**

For each stress component, accumulate over all wavevectors:

$$\sigma_i \propto \sum_{\boldsymbol{k}} \text{Kern}(\boldsymbol{k}) \cdot (\text{corresponding basis array})$$

**Step 3: Contour Integration** — Simpson's rule

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
