# Pseudo-Spectral Method for Polymer Field Theory in General Crystal Systems

This document provides the mathematical derivation of the pseudo-spectral method used in this software for solving the modified diffusion equation in polymer self-consistent field theory (SCFT), with particular emphasis on supporting non-orthogonal crystal systems.

## Table of Contents

1. [Modified Diffusion Equation](#1-modified-diffusion-equation)
2. [Pseudo-Spectral Method](#2-pseudo-spectral-method)
3. [Lattice Vectors and Metric Tensors](#3-lattice-vectors-and-metric-tensors)
4. [Fourier Transform in Non-Orthogonal Systems](#4-fourier-transform-in-non-orthogonal-systems)
5. [Boltzmann Weight Calculation](#5-boltzmann-weight-calculation)
6. [RQM4: 4th-Order Accuracy via Richardson Extrapolation](#6-rqm4-4th-order-accuracy-via-richardson-extrapolation)
7. [ETDRK4: Alternative 4th-Order Method](#7-etdrk4-alternative-4th-order-method)
8. [Stress Tensor Calculation](#8-stress-tensor-calculation)
9. [Crystal System Constraints](#9-crystal-system-constraints)
10. [Cell-Averaged Bond Function](#10-cell-averaged-bond-function)

---

## 1. Modified Diffusion Equation

In polymer field theory, the chain propagator q(r,s) satisfies the modified diffusion equation:

$$\frac{\partial q(\mathbf{r}, s)}{\partial s} = \frac{b^2}{6} \nabla^2 q(\mathbf{r}, s) - w(\mathbf{r}) q(\mathbf{r}, s)$$

where:
- $q(\mathbf{r}, s)$ is the chain propagator at position $\mathbf{r}$ and contour variable $s$
- $b$ is the statistical segment length
- $w(\mathbf{r})$ is the potential field
- $s \in [0, 1]$ parameterizes position along the chain contour

The initial condition is typically $q(\mathbf{r}, 0) = 1$ for a free chain end.

---

## 2. Pseudo-Spectral Method

### 2.1 Continuous Chain Model

The pseudo-spectral method solves the modified diffusion equation by operator splitting. The basic **Rasmussen-Kalosakas (RK) step** [2] for a contour step $\Delta s$ consists of:

1. **Potential half-step** (real space):
   $$q^* = \exp\left(-\frac{w \Delta s}{2}\right) q^n$$

2. **Diffusion step** (Fourier space):
   $$\hat{q}^{**} = \exp\left(-\frac{b^2 k^2 \Delta s}{6}\right) \hat{q}^*$$

3. **Potential half-step** (real space):
   $$q^{n+1} = \exp\left(-\frac{w \Delta s}{2}\right) q^{**}$$

where $\hat{q}$ denotes the Fourier transform of $q$, and $k^2 = |\mathbf{k}|^2$ is the squared magnitude of the wavevector.

This symmetric splitting yields **$O(\Delta s^2)$ global accuracy**. For higher accuracy, this library applies **RQM4** (Ranjan-Qin-Morse 4th-order method using Richardson extrapolation, see [Section 6](#6-rqm4-4th-order-accuracy-via-richardson-extrapolation)) to achieve **$O(\Delta s^4)$ accuracy** by combining full and half steps:

$$q^{n+1} = \frac{4 q(\Delta s/2, \Delta s/2) - q(\Delta s)}{3}$$

### 2.2 Discrete Chain Model

For discrete chains, the propagator is computed using **recursive integral equations** (Chapman-Kolmogorov equations) rather than solving a differential equation. This library implements the **N-1 bond model** from Park et al. (2019) [5] and Matsen & Beardsley (2021) [11].

**Units and Conventions:**
- **Unit length**: $R_0 = aN^{1/2}$, where $a$ is the statistical segment length and $N$ is the polymerization index
- **Contour step size**: $\Delta s = 1/N$
- **Segment indices**: $i = 1, 2, \ldots, N$ (N segments total)
- A polymer chain has **N segments** (monomers) connected by **N-1 bonds**

The propagator evolution from segment $i$ to segment $i+1$ follows:

1. **Bond convolution** (Fourier space):
   $$\hat{q}^*(\mathbf{k}) = \hat{g}(\mathbf{k}) \cdot \hat{q}_i(\mathbf{k})$$

2. **Full-segment Boltzmann weight** (real space):
   $$q_{i+1}(\mathbf{r}) = \exp(-w(\mathbf{r}) \Delta s) \cdot q^*(\mathbf{r})$$

with **initial condition**:
$$q_1(\mathbf{r}) = \exp(-w(\mathbf{r}) \Delta s)$$

where $g(\mathbf{R})$ is the **bond function** representing the probability distribution of bond vectors. For the bead-spring (Gaussian) model:

$$g(\mathbf{R}) = \left(\frac{3}{2\pi a^2}\right)^{3/2} \exp\left(-\frac{3|\mathbf{R}|^2}{2a^2}\right)$$

Its Fourier transform is:

$$\hat{g}(\mathbf{k}) = \exp\left(-\frac{a^2 |\mathbf{k}|^2}{6}\right)$$

**Physical Interpretation:**

The recursion relation computes the statistical weight of finding segment $i+1$ at position $\mathbf{r}$:
1. First, the bond convolution accounts for all possible positions of segment $i$ connected to segment $i+1$ by a bond
2. Then, the Boltzmann weight accounts for the field acting on segment $i+1$

This formulation makes the discrete chain computation as fast as the continuous chain with $O(M \log M)$ complexity per step.

---

## 3. Lattice Vectors and Metric Tensors

### 3.1 Real-Space Lattice Vectors

For a general crystal system, the simulation box is defined by three lattice vectors $\mathbf{a}$, $\mathbf{b}$, $\mathbf{c}$ with lengths $L_a$, $L_b$, $L_c$ and angles:
- $\alpha$: angle between $\mathbf{b}$ and $\mathbf{c}$
- $\beta$: angle between $\mathbf{a}$ and $\mathbf{c}$
- $\gamma$: angle between $\mathbf{a}$ and $\mathbf{b}$

Using the standard crystallographic convention, we orient:
- $\mathbf{a}$ along the x-axis
- $\mathbf{b}$ in the xy-plane
- $\mathbf{c}$ in general position

The lattice vectors in Cartesian coordinates are:

$$\mathbf{a} = \begin{pmatrix} L_a \\ 0 \\ 0 \end{pmatrix}$$

$$\mathbf{b} = \begin{pmatrix} L_b \cos\gamma \\ L_b \sin\gamma \\ 0 \end{pmatrix}$$

$$\mathbf{c} = \begin{pmatrix} L_c \cos\beta \\ L_c \frac{\cos\alpha - \cos\beta \cos\gamma}{\sin\gamma} \\ L_c \frac{\sqrt{\sin^2\gamma - \cos^2\alpha - \cos^2\beta + 2\cos\alpha\cos\beta\cos\gamma}}{\sin\gamma} \end{pmatrix}$$

### 3.2 Real-Space Metric Tensor

The metric tensor $G_{ij}$ relates the dot product of lattice vectors:

$$G_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j$$

where $\mathbf{e}_1 = \mathbf{a}$, $\mathbf{e}_2 = \mathbf{b}$, $\mathbf{e}_3 = \mathbf{c}$.

Explicitly:

$$G = \begin{pmatrix} L_a^2 & L_a L_b \cos\gamma & L_a L_c \cos\beta \\ L_a L_b \cos\gamma & L_b^2 & L_b L_c \cos\alpha \\ L_a L_c \cos\beta & L_b L_c \cos\alpha & L_c^2 \end{pmatrix}$$

### 3.3 Unit Cell Volume

The volume of the unit cell is:

$$V = \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) = L_a L_b L_c \sqrt{1 - \cos^2\alpha - \cos^2\beta - \cos^2\gamma + 2\cos\alpha\cos\beta\cos\gamma}$$

### 3.4 Reciprocal Lattice Vectors

The reciprocal lattice vectors are defined by:

$$\mathbf{a}^* = \frac{2\pi}{V}(\mathbf{b} \times \mathbf{c}), \quad \mathbf{b}^* = \frac{2\pi}{V}(\mathbf{c} \times \mathbf{a}), \quad \mathbf{c}^* = \frac{2\pi}{V}(\mathbf{a} \times \mathbf{b})$$

These satisfy the orthonormality relation:

$$\mathbf{e}_i \cdot \mathbf{e}_j^* = 2\pi \delta_{ij}$$

### 3.5 Reciprocal Metric Tensor

The reciprocal metric tensor $G_{ij}^*$ is computed from the reciprocal lattice vectors:

$$G_{ij}^* = \mathbf{e}_i^* \cdot \mathbf{e}_j^*$$

For practical computation, we use the symmetric storage:

$$G^* = \begin{pmatrix} G_{00}^* & G_{01}^* & G_{02}^* \\ G_{01}^* & G_{11}^* & G_{12}^* \\ G_{02}^* & G_{12}^* & G_{22}^* \end{pmatrix}$$

The reciprocal metric is related to the inverse of the real-space metric:

$$G^* = (2\pi)^2 G^{-1}$$

---

## 4. Fourier Transform in Non-Orthogonal Systems

### 4.1 Wavevector Representation

In a periodic system with grid dimensions $N_x \times N_y \times N_z$, the discrete wavevector is:

$$\mathbf{k} = n_1 \mathbf{a}^* + n_2 \mathbf{b}^* + n_3 \mathbf{c}^*$$

where $n_i$ are integers in the range $[-N_i/2, N_i/2)$.

### 4.2 Wavevector Magnitude

The squared magnitude of the wavevector is computed using the reciprocal metric tensor:

$$|\mathbf{k}|^2 = G_{ij}^* n_i n_j = G_{00}^* n_1^2 + G_{11}^* n_2^2 + G_{22}^* n_3^2 + 2G_{01}^* n_1 n_2 + 2G_{02}^* n_1 n_3 + 2G_{12}^* n_2 n_3$$

For **orthogonal systems** ($\alpha = \beta = \gamma = 90°$), the cross-terms vanish:

$$|\mathbf{k}|^2 = \left(\frac{2\pi n_1}{L_a}\right)^2 + \left(\frac{2\pi n_2}{L_b}\right)^2 + \left(\frac{2\pi n_3}{L_c}\right)^2$$

For **non-orthogonal systems**, the cross-terms contribute and must be included.

### 4.3 FFT Implementation

The standard FFT computes:

$$\hat{q}(\mathbf{n}) = \sum_{\mathbf{m}} q(\mathbf{m}) e^{-2\pi i \mathbf{n} \cdot \mathbf{m} / \mathbf{N}}$$

This remains unchanged for non-orthogonal systems. The non-orthogonality enters only through the Boltzmann weight calculation via $|\mathbf{k}|^2$.

---

## 5. Boltzmann Weight Calculation

### 5.1 Diffusion Boltzmann Factor

The Boltzmann factor for the diffusion step is:

$$B(\mathbf{k}) = \exp\left(-\frac{b^2 |\mathbf{k}|^2 \Delta s}{6}\right)$$

where $|\mathbf{k}|^2$ is computed using the reciprocal metric tensor as shown above.

### 5.2 Implementation

For each wavevector index $(n_1, n_2, n_3)$:

```
k_sq = G*_00 * n1^2 + G*_11 * n2^2 + G*_22 * n3^2
     + 2 * G*_01 * n1 * n2
     + 2 * G*_02 * n1 * n3
     + 2 * G*_12 * n2 * n3

boltz_bond[idx] = exp(-b^2 * k_sq * ds / 6)
```

---

## 6. RQM4: 4th-Order Accuracy via Richardson Extrapolation

**RQM4** (Ranjan-Qin-Morse 4th-order) is named after Ranjan, Qin, and Morse [6], who extended the Rasmussen-Kalosakas algorithm to 4th-order accuracy using Richardson extrapolation. This method combines the simplicity of the RK algorithm with higher-order temporal accuracy. A comprehensive benchmark comparing RQM4 with other pseudo-spectral algorithms is provided by Stasiak and Matsen [7].

### 6.1 Rasmussen-Kalosakas (RK) Algorithm

The pseudo-spectral method described in Section 2.1 is known as the **Rasmussen-Kalosakas (RK) algorithm** [2]. For the modified diffusion equation with Hamiltonian operator:

$$\hat{H} = -\frac{b^2}{6}\nabla^2 + w(\mathbf{r})$$

the formal solution for one step is:

$$q(\mathbf{r}, s_{n+1}) = e^{-\hat{H}\Delta s} q(\mathbf{r}, s_n)$$

The RK algorithm approximates this using symmetric operator splitting:

$$e^{-\hat{H}\Delta s} \approx e^{-\frac{1}{2}w\Delta s} \cdot e^{\frac{b^2}{6}\nabla^2 \Delta s} \cdot e^{-\frac{1}{2}w\Delta s}$$

This **symmetric Strang splitting** has several important properties:
- The diffusion operator is applied in Fourier space (diagonal)
- The potential operators are applied in real space (diagonal)
- The symmetry makes the algorithm **time-reversible**

### 6.2 RQM4: Richardson Extrapolation for 4th-Order Accuracy

**RQM4** uses Richardson extrapolation to eliminate the leading-order error by combining results from different step sizes. For one full step from $s_n$ to $s_{n+1} = s_n + \Delta s$:

1. **Full step**: Apply RK once with step size $\Delta s$:
   $$q_{full} = q_{RK}(s_{n+1}; \Delta s)$$

2. **Half steps**: Apply RK twice with step size $\Delta s/2$:
   $$q_{half} = q_{RK}(s_{n+1}; \Delta s/2)$$

The extrapolated result is:

$$q^{n+1} = \frac{4 q_{half} - q_{full}}{3}$$

### 6.3 Computational Cost

RQM4 requires **3 RK applications** per effective step:
- 1 full step
- 2 half steps

This triples the computational cost but provides two orders of magnitude better accuracy, which is typically more efficient than reducing $\Delta s$ by a factor of 10.

### 6.4 Implementation Summary

For each contour step in continuous chain propagation:

```
# Full step with Δs
q_full = RK_step(q_n, Δs)

# Two half steps with Δs/2
q_temp = RK_step(q_n, Δs/2)
q_half = RK_step(q_temp, Δs/2)

# Richardson extrapolation
q_{n+1} = (4 * q_half - q_full) / 3
```

where `RK_step(q, ds)` implements the symmetric operator splitting from Section 2.1.

---

## 7. ETDRK4: Alternative 4th-Order Method

### 7.1 Overview

**Exponential Time Differencing Runge-Kutta 4th-order (ETDRK4)** is an alternative to RQM4 for achieving 4th-order temporal accuracy. Originally developed by Cox and Matthews [9], ETDRK4 treats the linear diffusion term exactly while using a 4th-order Runge-Kutta scheme for the nonlinear potential term. Song, Liu, and Zhang [8] demonstrated that ETDRK4 can be more than 10 times more accurate than RQM4 at the same contour resolution, requiring fewer contour steps to achieve the same accuracy. The performance advantage is particularly pronounced at high segregation strengths.

### 7.2 Algorithm Formulation

The modified diffusion equation can be written as:

$$\frac{\partial \hat{q}}{\partial s} = c \hat{q} + \hat{N}(q)$$

where:
- $c = -\frac{b^2 k^2}{6}$ is the diffusion eigenvalue (diagonal in Fourier space)
- $\hat{N}(q) = \mathcal{F}[-w \cdot q]$ is the nonlinear potential term

The ETDRK4 algorithm advances from $s_n$ to $s_{n+1} = s_n + h$ through four stages:

**Stage a:**
$$\hat{a} = E_2 \hat{q}_n + \alpha \hat{N}_n$$

**Stage b:**
$$\hat{b} = E_2 \hat{q}_n + \alpha \hat{N}_a$$

**Stage c:**
$$\hat{c} = E_2 \hat{a} + \alpha (2\hat{N}_b - \hat{N}_n)$$

**Final combination:**
$$\hat{q}_{n+1} = E \hat{q}_n + f_1 \hat{N}_n + f_2 (\hat{N}_a + \hat{N}_b) + f_3 \hat{N}_c$$

where:
- $E = e^{ch}$ (full-step exponential)
- $E_2 = e^{ch/2}$ (half-step exponential)
- $\alpha$, $f_1$, $f_2$, $f_3$ are coefficients derived from $\varphi$-functions

### 7.3 Phi Functions

The ETDRK4 coefficients are expressed using $\varphi$-functions:

$$\varphi_1(z) = \frac{e^z - 1}{z}$$

$$\varphi_2(z) = \frac{e^z - 1 - z}{z^2}$$

$$\varphi_3(z) = \frac{e^z - 1 - z - z^2/2}{z^3}$$

The coefficients are:
- $\alpha = h \cdot \varphi_1(ch/2)$
- $f_1 = h \cdot (\varphi_1 - 3\varphi_2 + 4\varphi_3)$
- $f_2 = h \cdot 2(\varphi_2 - 2\varphi_3)$
- $f_3 = h \cdot (-\varphi_2 + 4\varphi_3)$

where all $\varphi$-functions are evaluated at $z = ch$.

### 7.4 Kassam-Trefethen Stabilization

For small $|ch|$, the $\varphi$-functions suffer from **catastrophic cancellation** due to subtraction of nearly equal terms. Kassam and Trefethen [10] proposed evaluating these functions using **contour integration**:

$$\varphi_n(z) \approx \frac{1}{M} \sum_{m=0}^{M-1} \varphi_n(z + r e^{2\pi i m/M})$$

With $M = 32$ points on a circle of radius $r = 1$, this achieves approximately 15 digits of accuracy for all $z$.

### 7.5 Comparison: ETDRK4 vs RQM4

| Property | RQM4 | ETDRK4 |
|----------|------|--------|
| Order | 4th | 4th |
| FFT pairs per step | 6 | 8 |
| Coefficient storage | 2 arrays | 6 arrays |
| Stability | L-stable | L-stable |
| Implementation | Simple | More complex |

**Benchmark Results (32³ grid, single step):**

| Metric | RQM4 | ETDRK4 |
|--------|------|--------|
| Max relative difference | — | $3.16 \times 10^{-5}$ |
| After 10 steps | — | $2.91 \times 10^{-4}$ |

Both methods produce nearly identical results, confirming equivalent 4th-order accuracy.

### 7.6 Implementation

ETDRK4 is implemented in:
- `CpuSolverPseudoETDRK4` (CPU/MKL)
- `CudaSolverPseudoETDRK4` (CUDA)
- `ETDRK4Coefficients` (shared coefficient computation)

For the RQM4 method, see:
- `CpuSolverPseudoRQM4` (CPU/MKL)
- `CudaSolverPseudoRQM4` (CUDA)

The coefficients are precomputed once during initialization and reused for all propagator steps.

### 7.7 Runtime Selection

Both RQM4 and ETDRK4 can be selected at runtime using the `numerical_method` parameter:

```python
params = {
    # ... other parameters ...
    "numerical_method": "rqm4"   # RQM4 (Ranjan-Qin-Morse 4th-order)
    # or
    "numerical_method": "etdrk4"  # ETDRK4 (Exponential Time Differencing RK4)
}
```

---

## 8. Stress Tensor Calculation

### 8.1 Stress Definition

The stress tensor measures the response of the free energy to deformation of the unit cell:

$$\sigma_{ij} = \frac{1}{V} \frac{\partial F}{\partial \epsilon_{ij}}$$

where $\epsilon_{ij}$ is the strain tensor.

### 8.2 Stress from Propagators

For polymer field theory, the stress contribution from chain statistics is computed in Fourier space. The stress tensor has 6 independent components (symmetric tensor):

$$\sigma = \begin{pmatrix} \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\ \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\ \sigma_{xz} & \sigma_{yz} & \sigma_{zz} \end{pmatrix}$$

### 8.3 Fourier-Space Stress Calculation

The stress is computed from the chain propagators:

$$\sigma_{ij} = -\frac{b^2}{V} \sum_{\mathbf{k}} \mathcal{F}_{ij}(\mathbf{k}) \cdot \hat{q}_1(\mathbf{k}) \hat{q}_2(-\mathbf{k})$$

where:
- $\hat{q}_1$, $\hat{q}_2$ are Fourier transforms of forward and backward propagators
- $\mathcal{F}_{ij}(\mathbf{k})$ is the Fourier basis for stress component $(i,j)$

### 8.4 Fourier Basis Arrays

The Fourier basis arrays encode the derivative of $|\mathbf{k}|^2$ with respect to strain:

**Diagonal components:**
$$\mathcal{F}_{xx} = k_x^2, \quad \mathcal{F}_{yy} = k_y^2, \quad \mathcal{F}_{zz} = k_z^2$$

**Off-diagonal components (cross-terms):**
$$\mathcal{F}_{xy} = 2 k_x k_y, \quad \mathcal{F}_{xz} = 2 k_x k_z, \quad \mathcal{F}_{yz} = 2 k_y k_z$$

Here, $k_x$, $k_y$, $k_z$ are the Cartesian components of the wavevector:

$$k_x = n_1 a_x^* + n_2 b_x^* + n_3 c_x^*$$

and similarly for $k_y$, $k_z$.

### 8.5 Stress Components and Box Optimization

The stress components drive optimization of different lattice parameters:

| Stress Component | Lattice Parameter |
|------------------|-------------------|
| $\sigma_{xx}$ | $L_a$ (length a) |
| $\sigma_{yy}$ | $L_b$ (length b) |
| $\sigma_{zz}$ | $L_c$ (length c) |
| $\sigma_{xy}$ | $\gamma$ (angle between a and b) |
| $\sigma_{xz}$ | $\beta$ (angle between a and c) |
| $\sigma_{yz}$ | $\alpha$ (angle between b and c) |

---

## 9. Crystal System Constraints

Different crystal systems impose constraints on the lattice parameters:

### 9.1 Orthorhombic/Tetragonal/Cubic

$$\alpha = \beta = \gamma = 90°$$

Cross-terms in $|\mathbf{k}|^2$ vanish. Off-diagonal stress components are zero at equilibrium.

**Constraints:**
- Orthorhombic: $L_a \neq L_b \neq L_c$ (3 independent lengths)
- Tetragonal: $L_a = L_b \neq L_c$ (2 independent lengths)
- Cubic: $L_a = L_b = L_c$ (1 independent length)

### 9.2 Hexagonal/Trigonal

$$\alpha = \beta = 90°, \quad \gamma = 120°$$

Cross-term $G_{01}^*$ is non-zero.

**Constraints:**
- $L_a = L_b \neq L_c$ (2 independent lengths)
- $\gamma$ fixed at 120°

### 9.3 Monoclinic

$$\alpha = \gamma = 90°, \quad \beta \neq 90°$$

Cross-term $G_{02}^*$ is non-zero.

**Constraints:**
- $L_a$, $L_b$, $L_c$ all independent (3 lengths)
- $\beta$ is a free parameter (1 angle)
- Off-diagonal stress $\sigma_{xz}$ drives $\beta$ optimization

### 9.4 Triclinic

$$\alpha, \beta, \gamma \text{ arbitrary}$$

All cross-terms may be non-zero.

**Constraints:**
- All 6 parameters ($L_a$, $L_b$, $L_c$, $\alpha$, $\beta$, $\gamma$) are independent
- All off-diagonal stress components may be non-zero

---

## 10. Cell-Averaged Bond Function

### 10.1 Motivation

The standard Gaussian bond function in the discrete chain model is strictly positive in Fourier space:

$$\hat{g}(\mathbf{k}) = \exp\left(-\frac{b^2 |\mathbf{k}|^2 \Delta s}{6}\right)$$

However, when this is transformed to real space on a finite grid, the resulting bond function can have **negative values** near grid points far from the origin due to aliasing effects. This is physically problematic since the bond function represents a probability density and should be non-negative.

### 10.2 Cell-Averaging Approach

To ensure non-negativity, we apply **cell-averaging** (also called **sinc filtering**) to the bond function. Instead of evaluating the bond function at discrete grid points, we average over each grid cell. This is equivalent to multiplying the Fourier-space bond function by a sinc filter:

**For periodic boundary conditions:**

$$\hat{g}_{avg}(\mathbf{k}) = \hat{g}(\mathbf{k}) \cdot \prod_{d=1}^{D} \text{sinc}\left(\frac{\pi n_d}{N_d}\right)$$

where $\text{sinc}(x) = \sin(x)/x$, $n_d$ is the Fourier mode index, and $N_d$ is the number of grid points in dimension $d$.

**For mixed boundary conditions (reflecting or absorbing):**

$$\hat{g}_{avg}(\mathbf{k}) = \hat{g}(\mathbf{k}) \cdot \prod_{d=1}^{D} \text{sinc}\left(\frac{\pi n_d}{2N_d}\right)$$

The factor of 2 in the denominator accounts for the implicit doubling of the domain in DCT/DST transforms.

### 10.3 End-to-End Distance Correction

A naive application of the sinc filter changes the chain statistics. The sinc filter adds variance $\Delta x^2/12$ per dimension to the bond distribution (where $\Delta x = L/N$ is the grid spacing). This inflates the end-to-end distance:

$$\langle R^2 \rangle_{naive} = N b^2 + \frac{N \cdot L^2}{12 N^2} \cdot D = N b^2 + \frac{D L^2}{12 N}$$

For coarse grids, this can cause errors exceeding 60%.

**Correction:** To preserve the correct chain statistics, we modify the Gaussian exponent to compensate for the sinc filter's contribution:

$$\hat{g}_{corrected}(\mathbf{k}) = \exp\left(-\frac{b^2 |\mathbf{k}|^2 \Delta s}{6} + \sum_{d=1}^{D} \frac{\pi^2 n_d^2}{6 N_d^2}\right) \cdot \prod_{d=1}^{D} \text{sinc}\left(\frac{\pi n_d}{N_d}\right)$$

The correction term $+\pi^2 n_d^2/(6 N_d^2)$ exactly cancels the variance added by the sinc filter, preserving:

$$\langle R^2 \rangle = N b^2$$

### 10.4 Mathematical Derivation

The variance of the sinc-averaged Gaussian in 1D is:

$$\langle \Delta x^2 \rangle = \frac{b^2 \Delta s}{3} + \frac{\Delta x^2}{12}$$

To recover the original variance $b^2 \Delta s/3$, we use an effective segment length:

$$b_{eff}^2 = b^2 - \frac{\Delta x^2}{4 \Delta s}$$

In Fourier space with $k = 2\pi n/L$, this corresponds to adding:

$$\frac{b^2 k^2 \Delta s}{6} \to \frac{b^2 k^2 \Delta s}{6} - \frac{\pi^2 n^2}{6 N^2}$$

The negative sign in the exponent becomes a positive correction term.

### 10.5 API Usage

Cell-averaging can be toggled at runtime:

```python
# Enable cell-averaged bond function
solver.set_cell_averaged_bond(True)

# Disable (return to standard bond function)
solver.set_cell_averaged_bond(False)
```

**Default:** Cell-averaging is **disabled** by default to maintain backward compatibility.

### 10.6 When to Use Cell-Averaging

Cell-averaging is recommended when:
- Using coarse grids where aliasing may cause negative bond function values
- Physical non-negativity of the bond function is important for the application
- Simulating grafted polymer brush systems where absorbing boundary conditions are used at the grafting surface

Cell-averaging is **not needed** when:
- Using fine grids (large $N$) where aliasing is negligible
- Only the partition function and concentrations are needed (not the real-space bond function itself)

### 10.7 References

The cell-averaging approach for discrete chain models is discussed in:
- Park, S. J., Yong, D., Kim, Y. & Kim, J. U. "Numerical implementation of pseudo-spectral method in self-consistent mean field theory for discrete polymer chains." *J. Chem. Phys.* **150**, 234901 (2019).

---

## References

1. Matsen, M. W. "The standard Gaussian model for block copolymer melts." *J. Phys.: Condens. Matter* **14**, R21 (2002).

2. Rasmussen, K. O. & Kalosakas, G. "Improved numerical algorithm for exploring block copolymer mesophases." *J. Polym. Sci. B: Polym. Phys.* **40**, 1777 (2002).

3. Tzeremes, G., Rasmussen, K. O., Lookman, T. & Saxena, A. "Efficient computation of the structural phase behavior of block copolymers." *Phys. Rev. E* **65**, 041806 (2002).

4. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).

5. Park, S. J., Yong, D., Kim, Y. & Kim, J. U. "Numerical implementation of pseudo-spectral method in self-consistent mean field theory for discrete polymer chains." *J. Chem. Phys.* **150**, 234901 (2019).

6. Ranjan, A., Qin, J. & Morse, D. C. "Linear response and stability of ordered phases of block copolymer melts." *Macromolecules* **41**, 942-954 (2008).

7. Stasiak, P. & Matsen, M. W. "Efficiency of pseudo-spectral algorithms with Anderson mixing for the SCFT of periodic block-copolymer phases." *Eur. Phys. J. E* **34**, 110 (2011).

8. Song, J. Q., Liu, Y. X. & Zhang, H. D. "An efficient algorithm for self-consistent field theory calculations of complex self-assembled structures of block copolymer melts." *Chinese J. Polym. Sci.* **36**, 488-496 (2018).

9. Cox, S. M. & Matthews, P. C. "Exponential time differencing for stiff systems." *J. Comput. Phys.* **176**, 430-455 (2002).

10. Kassam, A.-K. & Trefethen, L. N. "Fourth-order time-stepping for stiff PDEs." *SIAM J. Sci. Comput.* **26**, 1214-1233 (2005).

11. Matsen, M. W. & Beardsley, T. M. "Field-theoretic simulations for block copolymer melts using the partial saddle-point approximation." *Polymers* **13**, 2437 (2021).
