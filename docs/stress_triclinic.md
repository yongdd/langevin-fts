# Stress Computation for Non-Orthogonal (Triclinic) Lattices

This document describes the mathematical derivation and implementation of stress tensor computation for triclinic unit cells in the polymer field theory code.

## Lattice Definition

We use the standard crystallographic convention for the direct lattice vectors:

$$
\mathbf{a} = (L_1, 0, 0)
$$

$$
\mathbf{b} = (L_2 \cos\gamma, \, L_2 \sin\gamma, \, 0)
$$

$$
\mathbf{c} = \left( L_3 \cos\beta, \, L_3 \frac{\cos\alpha - \cos\beta \cos\gamma}{\sin\gamma}, \, L_3 \frac{\sqrt{1 - \cos^2\alpha - \cos^2\beta - \cos^2\gamma + 2\cos\alpha \cos\beta \cos\gamma}}{\sin\gamma} \right)
$$

where:
- $L_1, L_2, L_3$ are the lattice lengths
- $\alpha$ is the angle between $\mathbf{b}$ and $\mathbf{c}$
- $\beta$ is the angle between $\mathbf{a}$ and $\mathbf{c}$
- $\gamma$ is the angle between $\mathbf{a}$ and $\mathbf{b}$

The unit cell volume is:

$$
V = L_1 L_2 L_3 \sqrt{1 - \cos^2\alpha - \cos^2\beta - \cos^2\gamma + 2\cos\alpha \cos\beta \cos\gamma}
$$

For 2D, we use only $\mathbf{a}$ and $\mathbf{b}$ with angle $\gamma$, giving $V = L_1 L_2 \sin\gamma$.

## Reciprocal Lattice Vectors

The reciprocal lattice vectors (without the $2\pi$ factor) are:

$$
\mathbf{a}^* = \frac{\mathbf{b} \times \mathbf{c}}{V}, \quad
\mathbf{b}^* = \frac{\mathbf{c} \times \mathbf{a}}{V}, \quad
\mathbf{c}^* = \frac{\mathbf{a} \times \mathbf{b}}{V}
$$

For 2D with $\mathbf{a} = (L_1, 0)$ and $\mathbf{b} = (L_2 \cos\gamma, L_2 \sin\gamma)$:

$$
\mathbf{a}^* = \left( \frac{1}{L_1}, \, -\frac{\cot\gamma}{L_1} \right)
$$

$$
\mathbf{b}^* = \left( 0, \, \frac{\csc\gamma}{L_2} \right)
$$

## Wavevector in Cartesian Coordinates

The wavevector for Miller indices $(m_1, m_2, m_3)$ is:

$$
\mathbf{k} = 2\pi (m_1 \mathbf{a}^* + m_2 \mathbf{b}^* + m_3 \mathbf{c}^*)
$$

In Cartesian components for 2D:

$$
k_x = \frac{2\pi m_1}{L_1}
$$

$$
k_y = 2\pi \left( -\frac{m_1 \cot\gamma}{L_1} + \frac{m_2 \csc\gamma}{L_2} \right)
$$

## Stress Computation Overview

The stress is computed from the derivative of the Hamiltonian with respect to lattice parameters. The chain entropy contribution involves sums of the form:

$$
H = \sum_{\mathbf{k}} f(|\mathbf{k}|^2)
$$

where $f$ depends on the propagator products and Boltzmann factors. To compute $\partial H / \partial(\text{lattice parameter})$, we need $\partial(|\mathbf{k}|^2) / \partial(\text{lattice parameter})$.

We define the Cartesian $\mathbf{k} \otimes \mathbf{k}$ sums:

$$
S_{xx} = \sum_{\mathbf{k}} c(\mathbf{k}) \, k_x^2, \quad
S_{yy} = \sum_{\mathbf{k}} c(\mathbf{k}) \, k_y^2, \quad
S_{zz} = \sum_{\mathbf{k}} c(\mathbf{k}) \, k_z^2
$$

$$
S_{xy} = \sum_{\mathbf{k}} c(\mathbf{k}) \, k_x k_y, \quad
S_{xz} = \sum_{\mathbf{k}} c(\mathbf{k}) \, k_x k_z, \quad
S_{yz} = \sum_{\mathbf{k}} c(\mathbf{k}) \, k_y k_z
$$

where $c(\mathbf{k})$ is the coefficient from the propagator stress calculation.

## Derivation of Length Derivatives

### $\partial(|\mathbf{k}|^2)/\partial L_1$

From $k_x = 2\pi m_1 / L_1$:

$$
\frac{\partial k_x}{\partial L_1} = -\frac{2\pi m_1}{L_1^2} = -\frac{k_x}{L_1}
$$

From $k_y = 2\pi(-m_1 \cot\gamma / L_1 + m_2 \csc\gamma / L_2)$:

$$
\frac{\partial k_y}{\partial L_1} = \frac{2\pi m_1 \cot\gamma}{L_1^2} = -\frac{k_x \cos\gamma}{L_1 \sin\gamma}
$$

Therefore:

$$
\frac{\partial(|\mathbf{k}|^2)}{\partial L_1} = 2 k_x \frac{\partial k_x}{\partial L_1} + 2 k_y \frac{\partial k_y}{\partial L_1} = -\frac{2 k_x^2}{L_1} - \frac{2 k_x k_y \cos\gamma}{L_1 \sin\gamma}
$$

Summing over all $\mathbf{k}$ with coefficients:

$$
\frac{\partial H}{\partial L_1} = -\frac{2 S_{xx}}{L_1} - \frac{2 S_{xy} \cos\gamma}{L_1 \sin\gamma}
$$

After including the normalization factor $\mathcal{N}$ (which absorbs the factor of $-2$):

$$
\boxed{ \frac{\partial H}{\partial L_1} = \frac{1}{\mathcal{N}} \left( \frac{S_{xx}}{L_1} - \frac{S_{xy} \cos\gamma}{L_1 \sin\gamma} \right) }
$$

### $\partial(|\mathbf{k}|^2)/\partial L_2$

From $k_y = 2\pi(-m_1 \cot\gamma / L_1 + m_2 \csc\gamma / L_2)$:

$$
\frac{\partial k_y}{\partial L_2} = -\frac{2\pi m_2 \csc\gamma}{L_2^2}
$$

Using $k_y \sin\gamma = 2\pi(-m_1 \cos\gamma / L_1 + m_2 / L_2)$, we can show:

$$
\frac{m_2}{L_2} = \frac{k_y \sin\gamma + k_x \cos\gamma}{2\pi}
$$

After careful derivation:

$$
\frac{\partial(|\mathbf{k}|^2)}{\partial L_2} = -\frac{2 k_y^2}{L_2} + \frac{2 k_x k_y \cos\gamma}{L_2 \sin\gamma}
$$

The **opposite sign** on the cross-term compared to $\partial/\partial L_1$ arises from the asymmetric dependence of $k_y$ on $L_1$ and $L_2$.

$$
\boxed{ \frac{\partial H}{\partial L_2} = \frac{1}{\mathcal{N}} \left( \frac{S_{yy}}{L_2} + \frac{S_{xy} \cos\gamma}{L_2 \sin\gamma} \right) }
$$

### 3D Extension

For 3D, similar analysis gives:

$$
\boxed{ \frac{\partial H}{\partial L_1} = \frac{1}{\mathcal{N}} \left( \frac{S_{xx}}{L_1} - \frac{S_{xy} \cos\gamma}{L_1 \sin\gamma} - \frac{S_{xz} \cos\beta}{L_1} \right) }
$$

$$
\boxed{ \frac{\partial H}{\partial L_2} = \frac{1}{\mathcal{N}} \left( \frac{S_{yy}}{L_2} + \frac{S_{xy} \cos\gamma}{L_2 \sin\gamma} - \frac{S_{yz} \cos\alpha}{L_2} \right) }
$$

$$
\boxed{ \frac{\partial H}{\partial L_3} = \frac{1}{\mathcal{N}} \left( \frac{S_{zz}}{L_3} - \frac{S_{xz} \cos\beta}{L_3} - \frac{S_{yz} \cos\alpha}{L_3} \right) }
$$

## Derivation of Angle Derivatives

### $\partial(|\mathbf{k}|^2)/\partial\gamma$ for 2D

Since $k_x = 2\pi m_1 / L_1$ does not depend on $\gamma$:

$$
\frac{\partial(|\mathbf{k}|^2)}{\partial\gamma} = \frac{\partial(k_y^2)}{\partial\gamma} = 2 k_y \frac{\partial k_y}{\partial\gamma}
$$

From $k_y = 2\pi(-m_1 \cot\gamma / L_1 + m_2 \csc\gamma / L_2)$:

$$
\frac{\partial k_y}{\partial\gamma} = 2\pi \left( \frac{m_1 \csc^2\gamma}{L_1} - \frac{m_2 \csc\gamma \cot\gamma}{L_2} \right)
$$

Expressing $m_1/L_1$ and $m_2/L_2$ in terms of $k_x$ and $k_y$:

$$
\frac{m_1}{L_1} = \frac{k_x}{2\pi}, \quad \frac{m_2}{L_2} = \frac{k_y \sin\gamma + k_x \cos\gamma}{2\pi}
$$

Substituting and simplifying:

$$
\frac{\partial k_y}{\partial\gamma} = \frac{k_x \sin\gamma - k_y \cos\gamma}{\sin\gamma}
$$

Therefore:

$$
\frac{\partial(k_y^2)}{\partial\gamma} = 2 k_y \cdot \frac{k_x \sin\gamma - k_y \cos\gamma}{\sin\gamma} = 2 k_x k_y - 2 k_y^2 \cot\gamma
$$

Summing over all $\mathbf{k}$:

$$
\frac{\partial H}{\partial\gamma} = 2 S_{xy} - 2 S_{yy} \cot\gamma
$$

After normalization (absorbing the factor of 2):

$$
\boxed{ \frac{\partial H}{\partial\gamma} = \frac{1}{\mathcal{N}} \left( S_{xy} - S_{yy} \cot\gamma \right) }
$$

### 3D Angle Derivatives

For 3D, the same structure applies:

$$
\boxed{ \frac{\partial H}{\partial\gamma} = \frac{1}{\mathcal{N}} \left( S_{xy} - S_{yy} \cot\gamma \right) }
$$

$$
\boxed{ \frac{\partial H}{\partial\beta} = \frac{1}{\mathcal{N}} \left( S_{xz} - S_{zz} \cot\beta \right) }
$$

$$
\boxed{ \frac{\partial H}{\partial\alpha} = \frac{1}{\mathcal{N}} \left( S_{yz} - S_{zz} \cot\alpha \right) }
$$

**Note on 3D accuracy:** In a fully general triclinic system, changing one angle also affects other lattice components through the **c** vector:

$$
\mathbf{c} = \left( L_3 \cos\beta, \, L_3 \frac{\cos\alpha - \cos\beta \cos\gamma}{\sin\gamma}, \, \frac{V}{L_1 L_2 \sin\gamma} \right)
$$

This coupling means that $\partial k_z/\partial\gamma \neq 0$ in 3D (unlike in 2D), introducing additional terms involving $S_{xz}$, $S_{yz}$, and $S_{zz}$. The current implementation uses the simplified formulas above, which achieve ~1-2% accuracy for the angle derivatives. The errors arise from neglecting these cross-coupling terms.

## Normalization Factor

The normalization factor is:

$$
\mathcal{N} = -\frac{3 M^2}{\Delta s}
$$

where $M$ is the total number of grid points and $\Delta s$ is the contour step size.

## Summary of Formulas

### Lattice Length Derivatives

| Derivative | Formula |
|:----------:|:--------|
| $\dfrac{\partial H}{\partial L_1}$ | $\dfrac{1}{\mathcal{N}} \left( \dfrac{S_{xx}}{L_1} - \dfrac{S_{xy} \cos\gamma}{L_1 \sin\gamma} - \dfrac{S_{xz} \cos\beta}{L_1} \right)$ |
| $\dfrac{\partial H}{\partial L_2}$ | $\dfrac{1}{\mathcal{N}} \left( \dfrac{S_{yy}}{L_2} + \dfrac{S_{xy} \cos\gamma}{L_2 \sin\gamma} - \dfrac{S_{yz} \cos\alpha}{L_2} \right)$ |
| $\dfrac{\partial H}{\partial L_3}$ | $\dfrac{1}{\mathcal{N}} \left( \dfrac{S_{zz}}{L_3} - \dfrac{S_{xz} \cos\beta}{L_3} - \dfrac{S_{yz} \cos\alpha}{L_3} \right)$ |

### Angle Derivatives

| Derivative | Formula |
|:----------:|:--------|
| $\dfrac{\partial H}{\partial\gamma}$ | $\dfrac{1}{\mathcal{N}} \left( S_{xy} - S_{yy} \cot\gamma \right)$ |
| $\dfrac{\partial H}{\partial\beta}$ | $\dfrac{1}{\mathcal{N}} \left( S_{xz} - S_{zz} \cot\beta \right)$ |
| $\dfrac{\partial H}{\partial\alpha}$ | $\dfrac{1}{\mathcal{N}} \left( S_{yz} - S_{zz} \cot\alpha \right)$ |

## Validation Results

### 2D Tests ($L_1 = 3.2$, $L_2 = 4.1$, $\gamma = 115°$)

| Stress Component | Relative Error |
|:----------------:|:--------------:|
| $\partial H/\partial L_1$ | $\sim 10^{-8}$ |
| $\partial H/\partial L_2$ | $\sim 10^{-8}$ |
| $\partial H/\partial\gamma$ | $\sim 10^{-5}$ (factor = 1.0) |

The 2D implementation is exact because the formula correctly accounts for all dependencies.

### 3D Tests - Near-Orthogonal ($\alpha = 89.5°$, $\beta = 90.5°$, $\gamma = 90.5°$)

| Stress Component | Relative Error | Factor |
|:----------------:|:--------------:|:------:|
| $\partial H/\partial L_1$ | $\sim 10^{-5}$ | — |
| $\partial H/\partial L_2$ | $\sim 10^{-5}$ | — |
| $\partial H/\partial L_3$ | $\sim 0.05\%$ | — |
| $\partial H/\partial\gamma$ | $\sim 1.4\%$ | 0.986 |
| $\partial H/\partial\beta$ | $\sim 0.9\%$ | 0.991 |
| $\partial H/\partial\alpha$ | $\sim 0.4\%$ | 0.996 |

### 3D Tests - Large Deviations ($\alpha = 85°$, $\beta = 95°$, $\gamma = 100°$)

| Stress Component | Relative Error | Factor |
|:----------------:|:--------------:|:------:|
| $\partial H/\partial L_1$ | $\sim 0.13\%$ | — |
| $\partial H/\partial L_2$ | $\sim 0.02\%$ | — |
| $\partial H/\partial L_3$ | $\sim 3.4\%$ | — |
| $\partial H/\partial\gamma$ | $\sim 4.9\%$ | 0.951 |
| $\partial H/\partial\beta$ | $\sim 14.5\%$ | 0.855 |
| $\partial H/\partial\alpha$ | $\sim 6.0\%$ | 0.940 |

## Accuracy Limitations

The current implementation achieves:

- **2D**: Exact results for all angle deviations (factor = 1.0)
- **3D near-orthogonal** (angles within ±1° of 90°): 1-2% accuracy for angle derivatives
- **3D large deviations** (angles 5-10° from 90°): 5-15% accuracy for angle derivatives

The degraded accuracy for large 3D deviations arises from the simplified formulas that neglect cross-coupling terms in the general triclinic **c** vector. In particular:

1. The **β derivative** has the largest error because $k_z$ depends on both $\alpha$ and $\beta$ through the $c_y$ component
2. The **$L_3$ derivative** shows ~3% error because the z-direction is coupled to all three angles

For most polymer physics applications with nearly orthogonal boxes, the current implementation provides sufficient accuracy. For highly triclinic systems, additional correction terms would be needed.

## Implementation Files

The stress normalization is implemented in the `compute_stress()` method of:

- `src/platforms/cpu/CpuComputationDiscrete.cpp`
- `src/platforms/cpu/CpuComputationContinuous.cpp`
- `src/platforms/cuda/CudaComputationDiscrete.cu`
- `src/platforms/cuda/CudaComputationContinuous.cu`
