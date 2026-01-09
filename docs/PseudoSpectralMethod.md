# Pseudo-Spectral Method for Polymer Field Theory in General Crystal Systems

This document provides the mathematical derivation of the pseudo-spectral method used in this software for solving the modified diffusion equation in polymer self-consistent field theory (SCFT), with particular emphasis on supporting non-orthogonal crystal systems.

## Table of Contents

1. [Modified Diffusion Equation](#1-modified-diffusion-equation)
2. [Pseudo-Spectral Method](#2-pseudo-spectral-method)
3. [Lattice Vectors and Metric Tensors](#3-lattice-vectors-and-metric-tensors)
4. [Fourier Transform in Non-Orthogonal Systems](#4-fourier-transform-in-non-orthogonal-systems)
5. [Boltzmann Weight Calculation](#5-boltzmann-weight-calculation)
6. [Richardson Extrapolation for 4th-Order Accuracy](#6-richardson-extrapolation-for-4th-order-accuracy)
7. [Stress Tensor Calculation](#7-stress-tensor-calculation)
8. [Crystal System Constraints](#8-crystal-system-constraints)

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

This symmetric splitting yields **$O(\Delta s^2)$ global accuracy**. For higher accuracy, this library applies **Richardson extrapolation** (see [Section 6](#6-richardson-extrapolation-for-4th-order-accuracy)) to achieve **$O(\Delta s^4)$ accuracy** by combining full and half steps:

$$q^{n+1} = \frac{4 q(\Delta s/2, \Delta s/2) - q(\Delta s)}{3}$$

### 2.2 Discrete Chain Model

For discrete chains, the propagator is computed using **recursive integral equations** (Chapman-Kolmogorov equations) rather than solving a differential equation. This library implements the **N-1 bond model** from Park et al. (2019).

In this model:
- A polymer chain has **N segments** connected by **N-1 bonds**
- The contour step size is $\Delta s = 1/(N-1)$
- The natural end-to-end distance is $R_0 = a\sqrt{N-1}$

The propagator evolution from segment $s$ to $s + \Delta s$ follows:

1. **Half-segment Boltzmann weight** (real space):
   $$q^* = \exp\left(-\frac{w \Delta s}{2}\right) q^n$$

2. **Bond convolution** (Fourier space):
   $$\hat{q}^{**} = \hat{g}(\mathbf{k}) \cdot \hat{q}^*$$

3. **Half-segment Boltzmann weight** (real space):
   $$q^{n+1} = \exp\left(-\frac{w \Delta s}{2}\right) q^{**}$$

where $g(\mathbf{R})$ is the bond function. For the bead-spring model:

$$g(\mathbf{R}) = \left(\frac{3}{2\pi a^2}\right)^{3/2} \exp\left(-\frac{3|\mathbf{R}|^2}{2a^2}\right)$$

Its Fourier transform is:

$$\hat{g}(\mathbf{k}) = \exp\left(-\frac{a^2 |\mathbf{k}|^2}{6}\right)$$

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

$$\mathbf{e}_i \cdot \mathbf{e}^*_j = 2\pi \delta_{ij}$$

### 3.5 Reciprocal Metric Tensor

The reciprocal metric tensor $G^*_{ij}$ is computed from the reciprocal lattice vectors:

$$G^*_{ij} = \mathbf{e}^*_i \cdot \mathbf{e}^*_j$$

For practical computation, we use the symmetric storage:

$$G^* = \begin{pmatrix} G^*_{00} & G^*_{01} & G^*_{02} \\ G^*_{01} & G^*_{11} & G^*_{12} \\ G^*_{02} & G^*_{12} & G^*_{22} \end{pmatrix}$$

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

$$|\mathbf{k}|^2 = G^*_{ij} n_i n_j = G^*_{00} n_1^2 + G^*_{11} n_2^2 + G^*_{22} n_3^2 + 2G^*_{01} n_1 n_2 + 2G^*_{02} n_1 n_3 + 2G^*_{12} n_2 n_3$$

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

## 6. Richardson Extrapolation for 4th-Order Accuracy

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

### 6.2 Error Analysis of the RK Algorithm

For symmetric operator splitting, the local truncation error per step is $O(\Delta s^3)$, leading to a **global error of $O(\Delta s^2)$** after $1/\Delta s$ steps.

The exact propagator can be expanded as:

$$q(s + \Delta s) = q(s) + \Delta s \cdot f_1 + \Delta s^2 \cdot f_2 + \Delta s^3 \cdot f_3 + O(\Delta s^4)$$

The RK algorithm gives:

$$q_{RK}(s + \Delta s; \Delta s) = q(s) + \Delta s \cdot f_1 + \Delta s^2 \cdot f_2 + \Delta s^3 \cdot g_3 + O(\Delta s^4)$$

where $g_3 \neq f_3$ represents the leading-order error.

### 6.3 Richardson Extrapolation

**Richardson extrapolation** eliminates the leading-order error by combining results from different step sizes. For one full step from $s_n$ to $s_{n+1} = s_n + \Delta s$:

1. **Full step**: Apply RK once with step size $\Delta s$:
   $$q_{full} = q_{RK}(s_{n+1}; \Delta s)$$

2. **Half steps**: Apply RK twice with step size $\Delta s/2$:
   $$q_{half} = q_{RK}(s_{n+1}; \Delta s/2)$$

The extrapolated result is:

$$q^{n+1} = \frac{4 q_{half} - q_{full}}{3}$$

### 6.4 Why 4th-Order Accuracy?

For a generic $O(\Delta s^2)$ integrator, Richardson extrapolation would yield only $O(\Delta s^3)$ accuracy. However, the RK algorithm achieves **$O(\Delta s^4)$ accuracy** due to its **time-reversibility** [6].

For reversible integrators, the error expansion contains only **even powers** of $\Delta s$:

$$q_{RK}(s + \Delta s; \Delta s) = q_{exact}(s + \Delta s) + c_2 \Delta s^2 + c_4 \Delta s^4 + O(\Delta s^6)$$

The $O(\Delta s^3)$ term vanishes identically due to time-reversal symmetry.

Applying Richardson extrapolation:
- $q_{full}$ has error $c_2 \Delta s^2 + c_4 \Delta s^4 + O(\Delta s^6)$
- $q_{half}$ has error $c_2 (\Delta s/2)^2 + c_4 (\Delta s/2)^4 + O(\Delta s^6) = \frac{c_2}{4}\Delta s^2 + \frac{c_4}{16}\Delta s^4 + O(\Delta s^6)$

The extrapolation:

$$\frac{4 q_{half} - q_{full}}{3} = q_{exact} + \frac{4 \cdot \frac{c_2}{4} - c_2}{3}\Delta s^2 + \frac{4 \cdot \frac{c_4}{16} - c_4}{3}\Delta s^4 + O(\Delta s^6)$$

$$= q_{exact} + 0 \cdot \Delta s^2 - \frac{3c_4}{16}\Delta s^4 + O(\Delta s^6)$$

The $O(\Delta s^2)$ error is eliminated, and because the $O(\Delta s^3)$ term was already zero, the result has **$O(\Delta s^4)$ global accuracy**.

### 6.5 Computational Cost

Richardson extrapolation requires **3 RK applications** per effective step:
- 1 full step
- 2 half steps

This triples the computational cost but provides two orders of magnitude better accuracy, which is typically more efficient than reducing $\Delta s$ by a factor of 10.

### 6.6 Implementation Summary

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

## 7. Stress Tensor Calculation

### 7.1 Stress Definition

The stress tensor measures the response of the free energy to deformation of the unit cell:

$$\sigma_{ij} = \frac{1}{V} \frac{\partial F}{\partial \epsilon_{ij}}$$

where $\epsilon_{ij}$ is the strain tensor.

### 7.2 Stress from Propagators

For polymer field theory, the stress contribution from chain statistics is computed in Fourier space. The stress tensor has 6 independent components (symmetric tensor):

$$\sigma = \begin{pmatrix} \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\ \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\ \sigma_{xz} & \sigma_{yz} & \sigma_{zz} \end{pmatrix}$$

### 7.3 Fourier-Space Stress Calculation

The stress is computed from the chain propagators:

$$\sigma_{ij} = -\frac{b^2}{V} \sum_{\mathbf{k}} \mathcal{F}_{ij}(\mathbf{k}) \cdot \hat{q}_1(\mathbf{k}) \hat{q}_2(-\mathbf{k})$$

where:
- $\hat{q}_1$, $\hat{q}_2$ are Fourier transforms of forward and backward propagators
- $\mathcal{F}_{ij}(\mathbf{k})$ is the Fourier basis for stress component $(i,j)$

### 7.4 Fourier Basis Arrays

The Fourier basis arrays encode the derivative of $|\mathbf{k}|^2$ with respect to strain:

**Diagonal components:**
$$\mathcal{F}_{xx} = k_x^2, \quad \mathcal{F}_{yy} = k_y^2, \quad \mathcal{F}_{zz} = k_z^2$$

**Off-diagonal components (cross-terms):**
$$\mathcal{F}_{xy} = 2 k_x k_y, \quad \mathcal{F}_{xz} = 2 k_x k_z, \quad \mathcal{F}_{yz} = 2 k_y k_z$$

Here, $k_x$, $k_y$, $k_z$ are the Cartesian components of the wavevector:

$$k_x = n_1 a^*_x + n_2 b^*_x + n_3 c^*_x$$

and similarly for $k_y$, $k_z$.

### 7.5 Stress Components and Box Optimization

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

## 8. Crystal System Constraints

Different crystal systems impose constraints on the lattice parameters:

### 8.1 Orthorhombic/Tetragonal/Cubic

$$\alpha = \beta = \gamma = 90°$$

Cross-terms in $|\mathbf{k}|^2$ vanish. Off-diagonal stress components are zero at equilibrium.

**Constraints:**
- Orthorhombic: $L_a \neq L_b \neq L_c$ (3 independent lengths)
- Tetragonal: $L_a = L_b \neq L_c$ (2 independent lengths)
- Cubic: $L_a = L_b = L_c$ (1 independent length)

### 8.2 Hexagonal/Trigonal

$$\alpha = \beta = 90°, \quad \gamma = 120°$$

Cross-term $G^*_{01}$ is non-zero.

**Constraints:**
- $L_a = L_b \neq L_c$ (2 independent lengths)
- $\gamma$ fixed at 120°

### 8.3 Monoclinic

$$\alpha = \gamma = 90°, \quad \beta \neq 90°$$

Cross-term $G^*_{02}$ is non-zero.

**Constraints:**
- $L_a$, $L_b$, $L_c$ all independent (3 lengths)
- $\beta$ is a free parameter (1 angle)
- Off-diagonal stress $\sigma_{xz}$ drives $\beta$ optimization

### 8.4 Triclinic

$$\alpha, \beta, \gamma \text{ arbitrary}$$

All cross-terms may be non-zero.

**Constraints:**
- All 6 parameters ($L_a$, $L_b$, $L_c$, $\alpha$, $\beta$, $\gamma$) are independent
- All off-diagonal stress components may be non-zero

---

## References

1. Matsen, M. W. "The standard Gaussian model for block copolymer melts." *J. Phys.: Condens. Matter* **14**, R21 (2002).

2. Rasmussen, K. O. & Kalosakas, G. "Improved numerical algorithm for exploring block copolymer mesophases." *J. Polym. Sci. B: Polym. Phys.* **40**, 1777 (2002).

3. Tzeremes, G., Rasmussen, K. O., Lookman, T. & Saxena, A. "Efficient computation of the structural phase behavior of block copolymers." *Phys. Rev. E* **65**, 041806 (2002).

4. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).

5. Park, S. J., Yong, D., Kim, Y. & Kim, J. U. "Numerical implementation of pseudo-spectral method in self-consistent mean field theory for discrete polymer chains." *J. Chem. Phys.* **150**, 234901 (2019).

6. Ranjan, A., Qin, J. & Morse, D. C. "Linear response and stability of ordered phases of block copolymer melts." *Macromolecules* **41**, 942-954 (2008).
