# Stress Tensor Calculation in Polymer Field Theory

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the calculation of stress in self-consistent field theory (SCFT), following the perturbation theory developed by Tyler and Morse [1]. The stress tensor is essential for optimizing the unit cell dimensions to find equilibrium periodic structures.

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Perturbation Theory for Partition Function](#2-perturbation-theory-for-partition-function)
3. [Stress Definition and Array Convention](#3-stress-definition-and-array-convention)
4. [Fourier-Space Implementation](#4-fourier-space-implementation)
5. [Fourier Basis Arrays](#5-fourier-basis-arrays)
6. [Non-Orthogonal Box Corrections](#6-non-orthogonal-box-corrections)
7. [Chain Model Dependence](#7-chain-model-dependence)
8. [Box Optimization Algorithm](#8-box-optimization-algorithm)

---

## 1. Theoretical Foundation

The stress tensor measures the response of the free energy to changes in lattice parameters. Tyler and Morse [1] showed that for a self-consistent solution of the SCFT equations, the derivative of the free energy with respect to any unit cell parameter $\theta_i$ takes a remarkably simple form:

$$\frac{dF}{d\theta_i} = -k_B T \frac{\partial \ln Q}{\partial \theta_i}$$

where $Q$ is the single-chain partition function and the partial derivative is evaluated under an **affine deformation** of the chemical potential fields (i.e., $\omega(\tilde{\mathbf{r}})$ is held fixed while the unit cell parameters change, where $\tilde{\mathbf{r}}$ is the dimensionless coordinate).

### Why This Result is Non-Trivial

The free energy in SCFT is a functional of the chemical potential fields $\omega$:

$$\frac{F}{k_B T} = -\ln Q - \frac{N}{V} \int d\mathbf{r} \left[ \omega_\alpha(\mathbf{r}) \phi_\alpha(\mathbf{r}) - \frac{1}{2} \chi_{\alpha\beta} \phi_\alpha(\mathbf{r}) \phi_\beta(\mathbf{r}) \right]$$

When differentiating with respect to a unit cell parameter $\theta_i$, one might expect contributions from both:
1. The explicit dependence of $Q$ on the unit cell geometry
2. The implicit dependence through changes in the $\omega$ fields

However, Tyler and Morse showed that at self-consistency, the functional derivative $\delta F / \delta \omega = 0$ vanishes, eliminating the second contribution. The stress is thus determined entirely by how the partition function responds to geometric changes in the simulation box.

---

## 2. Perturbation Theory for Partition Function

The partition function derivative is computed via first-order perturbation theory. The chain propagator $q(\mathbf{r}, s)$ satisfies the modified diffusion equation with Hamiltonian operator:

$$\hat{H}_\alpha = -\frac{b_\alpha^2}{6} \nabla^2 + \omega_\alpha(\mathbf{r})$$

where $b_\alpha$ is the statistical segment length for block $\alpha$.

### Affine Deformation

Under an affine deformation that changes the unit cell parameters while keeping $\omega(\tilde{\mathbf{r}})$ fixed in dimensionless coordinates, the Hamiltonian operator changes as:

$$\delta \hat{H}_\alpha = -\frac{b_\alpha^2}{6} \delta \nabla^2$$

The potential term $\omega_\alpha(\mathbf{r})$ does not contribute because it is held fixed in the dimensionless representation.

### Laplacian in Dimensionless Coordinates

The Laplacian in dimensionless coordinates $\tilde{\mathbf{r}}$ (where $\mathbf{r} = \sum_\mu \tilde{r}_\mu \mathbf{a}_\mu$ and $\mathbf{a}_\mu$ are Bravais lattice vectors) is:

$$\nabla^2 = \sum_{\mu,\nu=1}^{3} \frac{\mathbf{b}_\mu \cdot \mathbf{b}_\nu}{(2\pi)^2} \frac{\partial^2}{\partial \tilde{r}_\mu \partial \tilde{r}_\nu}$$

where $\mathbf{b}_\mu$ are reciprocal lattice vectors satisfying the orthonormality relation $\mathbf{a}_\lambda \cdot \mathbf{b}_\mu = 2\pi \delta_{\lambda\mu}$.

### Perturbation of Partition Function

The partition function can be expressed as a product of propagators (Green's functions):

$$Q = \langle 0 | \hat{G}_B(N_B) \hat{G}_{B-1}(N_{B-1}) \cdots \hat{G}_1(N_1) | 0 \rangle$$

where $|0\rangle$ represents the uniform initial condition and $\hat{G}_\alpha(s)$ is the propagator for block $\alpha$.

The perturbation of $Q$ can be expressed as a sum over all blocks:

$$\delta Q = \sum_{\alpha=1}^{B} \langle q^\dagger(s_\alpha) | \delta \hat{G}_\alpha(N_\alpha) | q(s_{\alpha-1}) \rangle$$

where $|q(s)\rangle$ and $|q^\dagger(s)\rangle$ are the forward and backward partition functions at contour position $s$.

### Matrix Elements

The perturbation of the propagator satisfies:

$$\left[ \frac{\partial}{\partial s} + \hat{H}_\alpha \right] \delta \hat{G}_\alpha = -\delta \hat{H}_\alpha \hat{G}_\alpha$$

which yields:

$$\delta \hat{G}_\alpha(s) = -\int_0^s ds' \, \hat{G}_\alpha(s-s') \delta \hat{H}_\alpha \hat{G}_\alpha(s')$$

For eigenstates $|\psi_i^\alpha\rangle$ of $\hat{H}_\alpha$ with eigenvalues $E_i^\alpha$, the matrix elements are:

$$\langle \psi_i^\alpha | \delta \hat{G}_\alpha(s) | \psi_j^\alpha \rangle = -\langle \psi_i^\alpha | \delta \hat{H}_\alpha | \psi_j^\alpha \rangle \frac{e^{-E_i^\alpha s} - e^{-E_j^\alpha s}}{E_j^\alpha - E_i^\alpha}$$

---

## 3. Stress Definition and Array Convention

We define stress components corresponding to derivatives with respect to lattice parameters:

$$\sigma_i = -\frac{\partial (\beta F/n)}{\partial L_i}$$

where $L_i$ are the lattice lengths ($L_a$, $L_b$, $L_c$), and:

$$\sigma_{ij} = -\frac{\partial (\beta F/n)}{\partial \gamma_{ij}}$$

where $\gamma_{ij}$ are the lattice angles. Here $\beta F/n$ is the dimensionless free energy per reference chain.

### Stress Array Convention

The stress is stored as a 6-component array following Voigt notation:

| Index | Component | Drives Optimization of |
|-------|-----------|------------------------|
| 0 | $\sigma_a$ | $L_a$ (length a) |
| 1 | $\sigma_b$ | $L_b$ (length b) |
| 2 | $\sigma_c$ | $L_c$ (length c) |
| 3 | $\sigma_{ab}$ | $\gamma$ (angle between a and b) |
| 4 | $\sigma_{ac}$ | $\beta$ (angle between a and c) |
| 5 | $\sigma_{bc}$ | $\alpha$ (angle between b and c) |

For 2D systems, only indices 0, 1, and 2 (for $\sigma_{ab}$) are used.
For 1D systems, only index 0 is used.

---

## 4. Fourier-Space Implementation

The stress calculation in the pseudo-spectral method proceeds in three steps: (1) computing per-segment contributions in Fourier space, (2) integrating along the chain contour, and (3) applying normalization factors.

### Step 1: Per-Segment Stress Contribution

For each contour position $s$ along a polymer block, the code computes a Fourier-space sum using the forward propagator $q(\mathbf{r}, s)$ and backward propagator $q^\dagger(\mathbf{r}, s)$:

$$S_i(s) = \sum_{\mathbf{k}} b^2 \cdot B(\mathbf{k}) \cdot \text{Re}\left[\hat{q}(\mathbf{k}, s) \cdot \hat{q}^{\dagger *}(\mathbf{k}, s)\right] \cdot \mathcal{B}_i(\mathbf{k})$$

where:
- $\hat{q}$, $\hat{q}^\dagger$ are Fourier transforms of forward and backward propagators
- $B(\mathbf{k})$ is the Boltzmann bond factor (model-dependent, see [Section 7](#7-chain-model-dependence))
- $b$ is the statistical segment length
- $\mathcal{B}_i(\mathbf{k})$ is the "Fourier basis" array (see [Section 5](#5-fourier-basis-arrays))

For **real-valued fields**, the conjugate symmetry $\hat{q}(-\mathbf{k}) = \hat{q}^*(\mathbf{k})$ is used. For **complex-valued fields**, the code maintains a mapping from $\mathbf{k}$ to $-\mathbf{k}$ indices.

### Step 2: Contour Integration

The per-segment contributions are integrated along the chain contour using **Simpson's rule**:

$$\frac{\partial Q}{\partial \theta_i} = \int_0^1 S_i(s) \, ds \approx \sum_{n=0}^{N} w_n \, S_i(s_n)$$

where $w_n$ are Simpson's rule weights and $N$ is the number of contour steps.

### Step 3: Normalization

The final stress is obtained by normalizing the integrated result:

$$\sigma_i = -\frac{1}{3 L_i M^2 / \Delta s} \cdot \frac{\partial Q}{\partial \theta_i}$$

where:
- $L_i$ is the box dimension in direction $i$
- $M$ is the total number of grid points
- $\Delta s$ is the contour step size
- The factor of 3 comes from the $b^2/6$ coefficient in the diffusion equation (combined with the factor of 2 from the derivative of $|\mathbf{k}|^2$)

---

## 5. Fourier Basis Arrays

The code precomputes "Fourier basis" arrays that encode $\partial |\mathbf{k}|^2 / \partial \theta_i$. The wavevector magnitude squared is:

$$|\mathbf{k}|^2 = (2\pi)^2 G^*_{ij} n_i n_j$$

where $G^*_{ij}$ is the reciprocal metric tensor and $n_i$ are integer Miller indices.

### Diagonal Basis Arrays (for length derivatives)

$$\texttt{fourier\_basis\_x} = (2\pi)^2 G^*_{00} n_0^2$$
$$\texttt{fourier\_basis\_y} = (2\pi)^2 G^*_{11} n_1^2$$
$$\texttt{fourier\_basis\_z} = (2\pi)^2 G^*_{22} n_2^2$$

### Cross-term Basis Arrays (for angle derivatives)

$$\texttt{fourier\_basis\_xy} = 2(2\pi)^2 G^*_{01} n_0 n_1$$
$$\texttt{fourier\_basis\_xz} = 2(2\pi)^2 G^*_{02} n_0 n_2$$
$$\texttt{fourier\_basis\_yz} = 2(2\pi)^2 G^*_{12} n_1 n_2$$

The factor of 2 in cross-terms accounts for the symmetric sum $G^*_{01} n_0 n_1 + G^*_{10} n_1 n_0 = 2 G^*_{01} n_0 n_1$.

### Weight Factors for Real-to-Complex FFT

For real-valued fields using half-complex storage (storing only $k_z \geq 0$), interior modes ($k_z \neq 0$ and $2k_z \neq N_z$) are multiplied by a weight factor of 2 to account for their conjugate partners.

### Non-Periodic Boundary Conditions

For non-periodic boundary conditions (reflecting or absorbing), the wavenumber factors are:

$$\texttt{xfactor} = \left(\frac{\pi}{L}\right)^2$$

instead of $(2\pi/L)^2$ for periodic boundaries. Cross-terms are zero since non-periodic systems must be orthogonal.

---

## 6. Non-Orthogonal Box Corrections

For non-orthogonal crystal systems, the derivative $\partial |\mathbf{k}|^2 / \partial L_a$ involves not only $G^*_{00}$ but also the cross-terms $G^*_{01}$ and $G^*_{02}$ (since these depend on $L_a$). This follows from the chain rule applied to the reciprocal metric tensor elements.

### Derivative of Reciprocal Lattice Vectors

To compute how the reciprocal metric changes with lattice parameters, Tyler and Morse [1] showed that the variation of reciprocal basis vectors is:

$$\delta \mathbf{b}_\mu = -\frac{1}{2\pi} \sum_\nu (\mathbf{b}_\mu \cdot \delta \mathbf{a}_\nu) \mathbf{b}_\nu$$

This leads to:

$$\delta(\mathbf{b}_\mu \cdot \mathbf{b}_\nu) = \delta \mathbf{b}_\mu \cdot \mathbf{b}_\nu + \mathbf{b}_\mu \cdot \delta \mathbf{b}_\nu$$

### 3D Stress Components

For 3D non-orthogonal boxes, the stress components are computed as:

$$\sigma_a \propto \sum_{\mathbf{k}} \left( \texttt{fourier\_basis\_x} + \frac{1}{2}\texttt{fourier\_basis\_xy} + \frac{1}{2}\texttt{fourier\_basis\_xz} \right) \cdot (\cdots)$$

$$\sigma_b \propto \sum_{\mathbf{k}} \left( \texttt{fourier\_basis\_y} + \frac{1}{2}\texttt{fourier\_basis\_xy} + \frac{1}{2}\texttt{fourier\_basis\_yz} \right) \cdot (\cdots)$$

$$\sigma_c \propto \sum_{\mathbf{k}} \left( \texttt{fourier\_basis\_z} + \frac{1}{2}\texttt{fourier\_basis\_xz} + \frac{1}{2}\texttt{fourier\_basis\_yz} \right) \cdot (\cdots)$$

### 2D Stress Components

For 2D non-orthogonal boxes:

$$\sigma_a \propto \sum_{\mathbf{k}} \left( \texttt{fourier\_basis\_x} + \frac{1}{2}\texttt{fourier\_basis\_xy} \right) \cdot (\cdots)$$

$$\sigma_b \propto \sum_{\mathbf{k}} \left( \texttt{fourier\_basis\_y} + \frac{1}{2}\texttt{fourier\_basis\_xy} \right) \cdot (\cdots)$$

### Correction Factor Explanation

The factor of $\frac{1}{2}$ appears because the cross-term basis arrays already include a factor of 2 (see [Section 5](#5-fourier-basis-arrays)).

### Shear Stress

Shear stress (for angle optimization) uses the cross-term basis directly:

$$\sigma_{ab} \propto \sum_{\mathbf{k}} \texttt{fourier\_basis\_xy} \cdot (\cdots)$$

### Orthogonal Systems

For **orthogonal systems** ($\alpha = \beta = \gamma = 90°$), all cross-terms vanish ($G^*_{01} = G^*_{02} = G^*_{12} = 0$), and the corrections are not needed. The stress components simplify to:

$$\sigma_a \propto \sum_{\mathbf{k}} \texttt{fourier\_basis\_x} \cdot (\cdots)$$

---

## 7. Chain Model Dependence

The Boltzmann bond factor $B(\mathbf{k})$ differs between chain models:

### Continuous Chain Model

$$B(\mathbf{k}) = 1$$

For continuous chains, the diffusion propagator $\exp(-b^2 |\mathbf{k}|^2 \Delta s / 6)$ is already incorporated into the propagator computation through the modified diffusion equation. No additional bond factor is needed in the stress calculation.

### Discrete Chain Model

$$B(\mathbf{k}) = \exp\left(-\frac{b^2 |\mathbf{k}|^2 \Delta s}{6}\right)$$

For discrete chains, this is the Fourier transform of the Gaussian bond function:

$$g(\mathbf{R}) = \left(\frac{3}{2\pi a^2}\right)^{3/2} \exp\left(-\frac{3|\mathbf{R}|^2}{2a^2}\right)$$

The bond factor must be explicitly included in stress calculations because it represents the connectivity between discrete segments.

### Half-Bond Length

For stress calculations at chain ends or junctions where only half a bond contributes, the code uses:

$$B_{1/2}(\mathbf{k}) = \exp\left(-\frac{b^2 |\mathbf{k}|^2 \Delta s}{12}\right)$$

---

## 8. Box Optimization Algorithm

During SCFT iteration with `box_is_altering=True`, the lattice parameters are updated using gradient descent:

$$L_i^{(n+1)} = L_i^{(n)} - \eta \cdot \sigma_i$$

$$\gamma_{ij}^{(n+1)} = \gamma_{ij}^{(n)} - \eta \cdot \sigma_{ij}$$

where $\eta$ is the `scale_stress` parameter. At equilibrium, all stress components vanish:

$$\sigma_i = 0, \quad \sigma_{ij} = 0$$

### Simultaneous Iteration

Tyler and Morse [1] proposed iterating the unit cell parameters simultaneously with the SCFT equations, rather than using nested optimization loops. This approach:

1. Adds the unit cell parameters $\theta_i$ to the unknowns
2. Adds the zero-stress conditions $\partial \ln Q / \partial \theta_i = 0$ to the residual equations
3. Updates both fields and unit cell parameters in each iteration

This simultaneous iteration typically converges in approximately the same number of iterations as solving the SCFT equations in a fixed unit cell, making it highly efficient for finding equilibrium structures.

### Implementation in This Code

The stress is computed after propagator calculation using `compute_stress()` and retrieved via `get_stress()`. The box dimensions are then updated:

```python
# In SCFT iteration loop
if box_is_altering:
    stress = propagator_solver.get_stress()
    for i in range(dim):
        lx[i] -= scale_stress * stress[i]
```

---

## References

1. Tyler, C. A. & Morse, D. C. "Stress in self-consistent-field theory." *Macromolecules* **36**, 8184-8188 (2003).

2. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).
