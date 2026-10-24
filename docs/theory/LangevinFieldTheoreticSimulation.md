# Langevin Field-Theoretic Simulation (L-FTS and CL-FTS)

This document describes Langevin field-theoretic simulation (L-FTS), which includes compositional fluctuations beyond mean-field theory, and complex Langevin simulation (CL-FTS), which samples every field without the partial saddle-point approximation.

## Table of Contents

1. [Overview](#1-overview)
2. [Partial Saddle-Point Approximation](#2-partial-saddle-point-approximation)
3. [Langevin Dynamics](#3-langevin-dynamics)
4. [Saddle-Point Iteration for Imaginary Fields](#4-saddle-point-iteration-for-imaginary-fields)
5. [Complex Langevin Simulation (CL-FTS)](#5-complex-langevin-simulation-cl-fts)
6. [Structure Functions](#6-structure-functions)
7. [Implementation](#7-implementation)
8. [References](#8-references)

---

## 1. Overview

L-FTS is a simulation method that incorporates compositional fluctuations into polymer field theory. Unlike SCFT, which finds a single saddle-point solution, L-FTS samples field configurations according to the Boltzmann distribution, capturing:

- **Fluctuation effects**: Order-disorder transition shifts, composition fluctuations
- **Correlation functions**: Structure factors, scattering functions
- **Thermodynamic averages**: Ensemble-averaged properties

The key insight is the **partial saddle-point approximation (PSPA)**: real fields fluctuate via Langevin dynamics while imaginary fields are kept at their saddle points. Sections 2-4 describe L-FTS. CL-FTS (Section 5) lets the imaginary fields fluctuate as well, at the cost of complex fields and possible numerical instabilities. The structure functions of Section 6 apply to both methods.

---

## 2. Partial Saddle-Point Approximation

### 2.1 Partition Function

The canonical partition function in field-theoretic representation:

```math
\mathcal{Z} \propto \int \{\mathcal{D}\Omega_i\} \exp(-\beta H[\{\Omega_i\}])
```

In PSPA, we split auxiliary fields into real ($`\Omega_i`$) and imaginary ($`\Omega_j`$) components:

```math
\mathcal{Z} \propto \int \{\mathcal{D}\Omega_i\} \exp(-\beta H[\{\Omega_i\}, \{\Omega_j^*\}])
```

where $`\{\Omega_j^*\}`$ are saddle-point solutions for imaginary fields:

```math
\left.\frac{\delta H[\{\Omega_i\}, \{\Omega_j\}]}{\delta \Omega_j}\right|_{\Omega_j = \Omega_j^*} = 0
```

### 2.2 Example: AB Diblock Copolymer

For AB-type systems:
- **Real field**: $`\Omega_-`$ (exchange field, composition fluctuations)
- **Imaginary field**: $`\Omega_+`$ (pressure-like field, incompressibility)

The exchange field $`\Omega_-`$ fluctuates according to Langevin dynamics, while $`\Omega_+`$ is kept at its saddle point to enforce incompressibility.

---

## 3. Langevin Dynamics

### 3.1 Continuous Langevin Equation

The real fields evolve according to:

```math
\frac{\partial \Omega_i(\mathbf{r}, \tau)}{\partial \tau} = -\lambda_i \frac{\delta H}{\delta \Omega_i(\mathbf{r}, \tau)} + \eta(\mathbf{r}, \tau)
```

with noise statistics:
- $`\langle \eta(\mathbf{r}, \tau) \rangle = 0`$
- $`\langle \eta(\mathbf{r}, \tau) \eta(\mathbf{r}', \tau') \rangle = 2\lambda_i k_B T \, \delta(\mathbf{r} - \mathbf{r}') \delta(\tau - \tau')`$

Here $`\lambda_i`$ is the relative step size (mobility) of field $`i`$, `dt_scaling` in `lfts.py`, not an eigenvalue: for the first $`M-1`$ auxiliary fields it is the absolute value of the field's eigenvalue in [PolymerFieldTheory.md](PolymerFieldTheory.md#23-effective-hamiltonian) divided by the largest absolute eigenvalue among them, and it is 1 for the last field $`\Omega_M`$. For AB systems $`\lambda_- = \lambda_+ = 1`$.

### 3.2 Discretized Equation (Per Chain Unit)

Discretizing with cell volume $`\Delta V`$ and time step $`\delta\tau`$:

```math
\Omega_i(\mathbf{r}, \tau + \delta\tau) = \Omega_i(\mathbf{r}, \tau) - \lambda_i \frac{\delta H}{\delta \Omega_i} \delta\tau N + \sigma_i \mathcal{N}(\mathbf{r}, \tau)
```

where:
- $`\mathcal{N}`$: Gaussian noise with zero mean and unit variance
- $`\sigma_i^2 = \lambda_i \frac{2 \delta\tau N}{\Delta V / R_0^3 \cdot \sqrt{\bar{N}}}`$
- $`\bar{N} = b^6 \rho_0^2 N`$: invariant polymerization index

### 3.3 Euler Method

The simplest discretization (first-order in $`\delta\tau`$):

```math
\Omega_-(\mathbf{r}, \tau + \delta\tau) = \Omega_-(\mathbf{r}, \tau) - \lambda_- \left[\Phi_-(\mathbf{r}) + \frac{2}{\chi N}\Omega_-(\mathbf{r})\right] \delta\tau N + \sigma \mathcal{N}(\mathbf{r}, \tau)
```

### 3.4 Leimkuhler-Matthews Method

A more accurate second-order method that averages noise over two time steps:

```math
\Omega_i(\mathbf{r}, \tau + \delta\tau) = \Omega_i(\mathbf{r}, \tau) - \lambda_i \frac{\delta H}{\delta \Omega_i} \delta\tau N + \frac{\sigma_i}{2}\left[\mathcal{N}(\mathbf{r}, \tau) + \mathcal{N}(\mathbf{r}, \tau + \delta\tau)\right]
```

**Advantages:**
- Better sampling of the target distribution
- More stable for larger time steps
- Reduced systematic bias

---

## 4. Saddle-Point Iteration for Imaginary Fields

### 4.1 Simple Mixing

Update the pressure-like field using gradient descent; $`k`$ counts the saddle-point iterations within one Langevin step:

```math
\Omega_+^{(k+1)}(\mathbf{r}) = \Omega_+^{(k)}(\mathbf{r}) + \alpha \left[\phi_A(\mathbf{r}) + \phi_B(\mathbf{r}) - 1\right]
```

where $`\alpha`$ is the mixing parameter of the compressor (`mix_init`, adapted as in SCFT simple mixing), not the time step $`\delta\tau N`$.

### 4.2 Anderson Mixing

Accelerates convergence using history of previous iterations (see [SelfConsistentFieldTheory.md](SelfConsistentFieldTheory.md)).

### 4.3 Linear Response

Uses linear response theory to predict the saddle-point directly:

```math
\Omega_+^* \approx \Omega_+^{(0)} + \chi_{\Omega_+}^{-1} \cdot (\phi_+ - 1)
```

where $`\chi_{\Omega_+}`$ is the response function.

### 4.4 Available Compressors

| Name | Description |
|------|-------------|
| `am` | Anderson Mixing |
| `lr` | Linear Response |
| `lram` | Linear Response + Anderson Mixing (hybrid) |

---

## 5. Complex Langevin Simulation (CL-FTS)

CL-FTS drops the partial saddle-point approximation: every auxiliary field fluctuates, including the imaginary fields that L-FTS keeps at their saddle points. The fields then become complex, and so do the concentrations, the partition functions $`Q_p`$ and $`H`$. When the dynamics is stable, trajectory averages converge to the averages of the full field theory (References 2 and 4). `CLFTS` uses the compressible model of [PolymerFieldTheory.md](PolymerFieldTheory.md#3-compressible-model) and needs a finite `zeta_n` (default 100).

### 5.1 Complex Langevin Equations

As in [PolymerFieldTheory.md](PolymerFieldTheory.md#43-sign-convention-for-imaginary-fields), $`H`$ is minimized with respect to the real fields and maximized with respect to the imaginary fields at the saddle point; for AB, $`\Omega_-`$ is real and $`\Omega_+`$ imaginary. An imaginary field is integrated along the imaginary axis, $`\Omega_j = i w_j`$ with real $`w_j`$. Written for $`\Omega_j`$, the real Langevin step of $`w_j`$ reverses the sign of the force and makes the noise imaginary:

```math
\Omega_i(\mathbf{r}, \tau + \delta\tau) = \Omega_i(\mathbf{r}, \tau) - \lambda_i \frac{\delta H}{\delta \Omega_i} \delta\tau N + \sigma_i \mathcal{N}_i(\mathbf{r}, \tau) \qquad \text{(real fields)}
```

```math
\Omega_j(\mathbf{r}, \tau + \delta\tau) = \Omega_j(\mathbf{r}, \tau) + \lambda_j \frac{\delta H}{\delta \Omega_j} \delta\tau N + i \sigma_j \mathcal{N}_j(\mathbf{r}, \tau) \qquad \text{(imaginary fields)}
```

The derivatives are those of [PolymerFieldTheory.md](PolymerFieldTheory.md#32-effective-hamiltonian), evaluated with the complex concentrations (smeared, when smearing is used). $`\mathcal{N}_i`$ is real Gaussian noise of unit variance, and $`\sigma_i`$ is that of Section 3.2. The mobility $`\lambda_i`$ (`dt_scaling` in `clfts.py`) follows Section 3.1 with the eigenvalues of $`U = X + \zeta N \, \mathbf{e}\mathbf{e}^T`$. A field with a zero eigenvalue does not enter $`H`$; it has $`\lambda_i = 0`$ and stays fixed.

LM (Section 3.4) is the default integrator and averages the noise over two steps; `langevin.integrator: "euler"` selects Section 3.3. For an AB diblock, these updates are Eqs. (13)-(16) of Reference 10. The time step is fixed. Reference 10 ran LM with $`\delta\tau N = 2`$ in CL-FTS and $`\delta\tau N = 8`$ in L-FTS.

### 5.2 Hot Spots and Dynamical Stabilization

The imaginary part of a real field can grow without bound at a few grid points ("hot spots"), most often at small $`\bar{N}`$ and with contact interactions (Reference 8). `CLFTS` stops with a `ValidationError` in either of two cases:

- the fields, the concentrations or $`H`$ become non-finite;
- the spatial mean of $`\sum_i \phi_i - 1`$ exceeds $`5 \times 10^{-8}`$ in magnitude, which is usually the first sign of a hot spot.

Two remedies are available:

- **Dynamical stabilization** (Reference 8, `dynamic_stabilization`): the force on each real field gets the term $`i \alpha_{ds} \mathrm{Im}[\Omega_i]`$, which damps its imaginary part.
  - It biases the sampling toward stronger segregation: a slightly lower $`(\chi N)_{\mathrm{ODT}}`$ and a higher $`S(k^*)`$.
  - Reference 8 used $`\alpha_{ds} = 0.001`$ at $`\bar{N} = 10^6`$-$`10^7`$ and 0.01-0.02 at $`\bar{N} = 10^5`$.
  - Reference 10 found structure functions more sensitive to $`\alpha_{ds}`$ than the order-disorder transition, and computed them without stabilization.
- **Smearing** ([`smearing`](../references/Parameters.md#smearing)): finite-range interactions damp the high-$`k`$ fluctuations from which hot spots grow (Reference 10).

[DYNAMICAL_STABILIZATION.md](../../examples/clfts/DYNAMICAL_STABILIZATION.md) and [SMEARING.md](../../examples/clfts/SMEARING.md) in `examples/clfts/` give recommended values.

### 5.3 Renormalization and Calibration

The UV divergence of the grid makes the effective interaction $`\chi \approx z_\infty \chi_b`$ differ from the bare $`\chi_b`$ given by `chi_n`. CL-FTS has a second correction: the fluctuations of the imaginary (pressure) field also change the chain size. The bare segment length $`a_b`$ is therefore chosen so that $`R_0 = a N^{1/2}`$ in the limit $`\chi_b \to 0`$. In L-FTS and SCFT, $`a_b = a`$. For discrete chains with smearing function $`\Gamma(k)`$, Reference 10 (Eqs. 26 and 29) gives

```math
\frac{a_b}{a} = \left[ 1 + \frac{a^2}{72 (2\pi)^3 \rho_0} \int \frac{k^2 \, d\mathbf{k}}{s^2(k) \left[ 1 + t(k)/\zeta \right]} \right]^{-1}, \qquad z_\infty = 1 - \frac{1}{(2\pi)^3 \rho_0} \int \frac{d\mathbf{k}}{t(k) \left[ 1 + t(k)/\zeta \right]}
```

where:

- $`s(k) = \sinh(k^2 a^2/12)`$ and $`t(k) = \tanh(k^2 a^2/12) / \Gamma^2(k)`$;
- the integrals run over $`-\pi/\Delta_\nu \le k_\nu \le \pi/\Delta_\nu`$;
- $`\rho_0`$ is the monomer density, with $`\rho_0 a^3 = (\bar{N}/N)^{1/2}`$;
- $`\zeta N`$ is `zeta_n`.

The same $`z_\infty`$ applies to L-FTS of the compressible model, and as $`\zeta \to \infty`$ the factor $`1 + t/\zeta`$ goes to 1. `scripts/analysis/renormalization.py` ([Tools](../references/Tools.md#3-χn-renormalization)) evaluates both formulas from a fields file or from options. For continuous chains, it uses their limit $`N \to \infty`$, where $`\tanh`$ and $`\sinh`$ become linear.

To use the calibration, multiply all `segment_lengths` by $`a_b/a`$ and set `chi_n` to the $`\chi_b N`$ of the target $`z_\infty \chi_b N`$. Reference 10 adds a quadratic term, $`\chi = z_\infty \chi_b + c_2 (z_\infty \chi_b)^2`$, fitted to renormalized one-loop theory; $`c_2`$ differs between CL-FTS and L-FTS. Section 6.4 gives an example.

### 5.4 Observables

- **Hamiltonian:** $`H`$ and $`\partial H / \partial (\chi N)`$ are complex and recorded as real and imaginary parts (`H_history_real`, `H_history_imag`, `dH_history_*_real`, `dH_history_*_imag`). Their ensemble averages are real, so the imaginary parts average to zero within the statistical error. `drift_max_history` includes the stabilization term.
- **Structure functions:** they use all fields with $`\lambda_k \neq 0`$ (Section 6.1) and full complex Fourier transforms, and are recorded as `structure_function_*_real` and `structure_function_*_imag`. In the compressible model $`S_{AA}`$ also contains the total-density fluctuations. For AB, the composition structure function is $`(S_{AA} + S_{BB} - 2 S_{AB})/4`$ (Section 6.4).
- **Fields:** `fields_*.mat` stores the complex fields and concentrations as `_real`/`_imag` pairs, and runs restart from these files as in L-FTS.

### 5.5 Using the CLFTS Class

`CLFTS(params, random_seed)` takes the parameters of the L-FTS example in Section 7.1 without `saddle` and `compressor`. It adds `zeta_n`, `dynamic_stabilization` and, optionally, `smearing` and `electrostatics` (see [CL-FTS Parameters](../references/Parameters.md#cl-fts-parameters)). [examples/clfts/Lamella.py](../../examples/clfts/Lamella.py) and [examples/clfts/ChargedDiblock.py](../../examples/clfts/ChargedDiblock.py) are complete scripts.

### 5.6 Charged Systems

Monomer type $`i`$ may carry the valence $`z_i`$ per segment (`electrostatics`). Small ions are molecules of one segment, so the compressibility term counts them like the other species.

**Coulomb interaction.** The charges interact through the smeared charge density and the Coulomb kernel:

```math
c(\mathbf{r}) = \sum_i z_i \, (\Gamma_i * \phi_i)(\mathbf{r}), \qquad \frac{\beta U_{el}}{C V / R_0^3} = \frac{E}{2V} \int c \, (-\nabla^2)^{-1} c \, d\mathbf{r}
```

Here $`\Gamma_i`$ is the smearing function of type $`i`$; one Gaussian length per type plays the role of a Born radius. The coupling in units of the reference chain is

```math
E = 4\pi l_B \rho_0 N R_0^2 = 4\pi (l_B/R_0) N^2 \bar{N}^{1/2}
```

**Electrostatic potential.** A Hubbard-Stratonovich transformation decouples the interaction with an electrostatic potential $`\psi`$. Like the pressure field, $`\psi`$ is an imaginary-type field stored on the rotated contour. Each type feels $`z_i \psi`$, so the propagator input becomes $`\Gamma_i * (w_i + z_i \psi)`$, and

```math
\frac{\beta H_{el}}{C V / R_0^3} = \frac{1}{2 E V} \int \psi \, \nabla^2 \psi \, d\mathbf{r}, \qquad \frac{\delta H}{\delta \psi} = \frac{\nabla^2 \psi}{E} + c
```

The saddle point is the Poisson equation $`-\nabla^2 \psi = E c`$. The uniform mode of $`\psi`$ is zero, because the system is electroneutral.

**Update of ψ.** $`\psi`$ moves like an imaginary field (Section 5.1) with the mobility $`s`$ (`psi_dt_scaling`). Its linear part $`\nabla^2 \psi / E`$ relaxes at the rate $`s k^2/E`$, which is stiff at large $`k`$. This part is therefore integrated exactly for each wavevector, by exponential time differencing:

```math
\hat\psi_{\mathbf{k}}(\tau + \delta\tau) = a_k \hat\psi_{\mathbf{k}}(\tau) + (1 - a_k) \frac{E}{k^2} \hat c_{\mathbf{k}} + i \sqrt{\frac{1 - a_k^2}{2 (k^2/E) \, \delta\tau N}} \, \hat{\xi}_{\mathbf{k}}, \qquad a_k = e^{-s (k^2/E) \delta\tau N}
```

In this step:

- $`\hat\xi_{\mathbf{k}}`$ is the Fourier transform of real Gaussian noise with the standard deviation $`\sigma`$ of Section 3.2, drawn from a separate random-number stream (seed + 1). The noise of the auxiliary fields is therefore the same as without charges.
- $`\hat c_{\mathbf{k}}`$ is the charge density of the state before the step, as for the forces of the auxiliary fields.
- As $`k \to 0`$, the step becomes the Euler step with mobility $`s`$.

The stationary variance of the linear modes does not depend on $`\delta\tau`$, while the response to the charge density stays explicit.

**Observables.**
- **H:** the recorded $`H`$ includes $`H_{el}`$.
- **Structure functions:** Section 6.1 gives $`S_{ij}`$ of all types, counterions included. For composition modes without an auxiliary field, such as two types that differ only in charge, $`\psi`$ completes the estimator ([Structure Function Estimators](StructureFunctionEstimators.md#5-modes-without-an-auxiliary-field)). The charge structure function is $`S_{zz} = \sum_{ij} z_i z_j S_{ij}`$. The files also hold `charge_structure_function`, the structure function of the smeared charge density. It combines two estimators that use $`\psi`$ with the weight of least variance at each wavevector, and it is much less noisy where screening is strong ([Section 6](StructureFunctionEstimators.md#6-charge-structure-function)).
- **RPA:** the RPA of Section 6.4 adds $`(E/k^2)\, z z^T`$ to the interaction matrix.

**Tests.**
- `TestClftsElectrostatics` checks:
  - that zero valences, or zero coupling, reproduce CL-FTS without charges bit for bit;
  - that the forces of $`\psi`$ and of the auxiliary fields are the derivatives of $`H`$;
  - that a continued run reproduces an uninterrupted one bit for bit.
- `TestChargedStructureFunctionRPA` compares two systems with the RPA:
  - a charged diblock with counterions;
  - a polyampholyte melt whose charge mode has no auxiliary field.

**Limitations.** The implementation supports periodic boxes only and no random copolymers. L-FTS keeps imaginary-type fields at their saddle point, so it does not have this fluctuating $`\psi`$.

---

## 6. Structure Functions

### 6.1 Definition

The structure function is computed using response theory (chemical potential-density correlation):

```math
S_{ij}(\mathbf{k}) = V \sqrt{\bar{n}} \langle \tilde{\mu}_i(\mathbf{k}) \tilde{\phi}_j(-\mathbf{k}) \rangle, \qquad \mu_i(\mathbf{r}) = \sum_{k} \frac{A_{ik}\, \Omega_k(\mathbf{r})}{\lambda_k}
```

where $`\tilde{\mu}_i(\mathbf{k})`$ and $`\tilde{\phi}_j(\mathbf{k})`$ are the Fourier transforms (normalized by the number of grid points) of the chemical potential field $`\mu_i`$ and of the concentration field. The sum runs over the auxiliary fields with $`\lambda_k \neq 0`$ (L-FTS: the first $`M-1`$ of them; CL-FTS: all of them). $`S_{ij}`$ is in units of $`\rho_0 N`$; in the disordered phase at weak fluctuations it approaches the random-phase approximation of [Section 6.4](#64-rpa-reference).

[Structure Function Estimators](StructureFunctionEstimators.md) covers the rest:
- **Derivation:** the shift of the Gaussian integral over the fields;
- **Smearing:** the division by $`\Gamma_i`$, and the projections that L-FTS records;
- **Charges:** the completion by $`\psi`$ for composition modes without an auxiliary field, such as two types that differ only in charge;
- **Charge structure function**, and a comparison with BioFTS.

### 6.2 Computing Structure Functions

In L-FTS, structure functions are computed by:
1. Fourier transforming the chemical potential fields $`\mu_i(\mathbf{r})`$
2. Fourier transforming the concentration fields $`\phi_j(\mathbf{r})`$
3. Computing $`\tilde{\mu}_i(\mathbf{k}) \tilde{\phi}_j(-\mathbf{k})`$ from the concentrations of the same fields
4. Averaging over Langevin trajectory
5. Multiplying by $`V \sqrt{\bar{n}}`$ normalization factor

### 6.3 Physical Interpretation

- **Peak position**: Characteristic length scale of microphase separation
- **Peak height**: Degree of ordering (diverges at spinodal)
- **Low-$`k`$ behavior**: Long-wavelength fluctuations

### 6.4 RPA Reference

In the disordered phase at weak fluctuations (large $`\bar{N}`$, $`\chi N`$ well below the order-disorder transition) the structure function approaches the random-phase approximation (RPA) of Leibler (Reference 1). `tests/TestStructureFunctionRPA.py` compares L-FTS and CL-FTS runs, with and without smearing, against the formulas below, and `tests/validation/validate_fts_sampling.py` uses the same reference for the sampling studies of [Numerical Validation](../internals/NumericalValidation.md).

Lengths are in units of $`R_0`$ and $`S_{ij}`$ is in units of $`\rho_0 N`$ as in Section 6.1, i.e. $`S_{ij}`$ is Leibler's per-monomer structure function divided by $`N`$.

**Ideal-chain correlations.** For a block of contour fraction $`f`$ whose monomer type has the relative segment length $`\epsilon`$, let $`x = \epsilon^2 k^2 / 6`$ ($`k^2 R_g^2`$ of a whole chain of that type) and

```math
g(f, x) = \frac{2 \left( f x + e^{-f x} - 1 \right)}{x^2}, \qquad h(f, x) = \frac{1 - e^{-f x}}{x}
```

$`g(1, x)`$ is the Debye function and $`g(f, 0) = f^2`$. For a melt of AB diblocks with A fraction $`f`$ the single-chain correlation matrix is

```math
G = \begin{pmatrix} g_{AA} & g_{AB} \\ g_{AB} & g_{BB} \end{pmatrix}, \qquad g_{AA} = g(f, x_A), \quad g_{BB} = g(1-f, x_B), \quad g_{AB} = h(f, x_A) \, h(1-f, x_B)
```

With equal segment lengths, $`g_{AB} = [g(1, x) - g(f, x) - g(1-f, x)]/2`$ and $`g_{AA} + g_{BB} + 2 g_{AB} = g(1, x)`$. These are continuous-chain functions; for the discrete chain model the contour integrals become sums over segments.

**Compressible model.** With $`X`$ the $`\chi N`$ matrix (zero diagonal), $`\mathbf{e} = [1, \ldots, 1]^T`$ and $`U = X + \zeta N \, \mathbf{e}\mathbf{e}^T`$ the interaction matrix of [Polymer Field Theory](PolymerFieldTheory.md),

```math
S(\mathbf{k}) = \left[ G^{-1} + U \right]^{-1}, \qquad U_{AB} = \begin{pmatrix} \zeta N & \zeta N + \chi N \\ \zeta N + \chi N & \zeta N \end{pmatrix}
```

**Incompressible model.** The constraint $`\sum_i \delta\phi_i = 0`$ is the limit $`\zeta N \to \infty`$:

```math
S(\mathbf{k}) = Q \left[ Q^T \left( G^{-1} + X \right) Q \right]^{-1} Q^T
```

where the $`M - 1`$ columns of $`Q`$ are orthonormal and orthogonal to $`\mathbf{e}`$. For AB, $`Q = [1, -1]^T / \sqrt{2}`$ gives Leibler's result

```math
\frac{1}{S_{AA}(k)} = F(x) - 2\chi N, \qquad F(x) = \frac{g_{AA} + g_{BB} + 2 g_{AB}}{g_{AA} g_{BB} - g_{AB}^2}, \qquad S_{BB} = -S_{AB} = S_{AA}
```

which is his $`N/S(q) = F(x) - 2\chi N`$.

**Smearing.** The interactions act on the smeared densities, so $`U`$ (or $`X`$) is multiplied by $`\Gamma(\mathbf{k})^2`$, while the recorded structure function is that of the densities (Section 6.1):

```math
S(\mathbf{k}) = \left[ G^{-1} + \Gamma(\mathbf{k})^2 \, U \right]^{-1}, \qquad \frac{1}{S_{AA}(k)} = F(x) - 2\chi N \, \Gamma(k)^2 \quad \text{(incompressible AB)}
```

with $`\Gamma(k)^2 = \exp(-a^2 k^2)`$ for Gaussian smearing. This holds in a periodic box, where the recorded $`S_{ij}`$ is divided by $`\Gamma`$.

**Charges.** With one smearing function per type and the valences of Section 5.6:

```math
S(\mathbf{k}) = \left[ G^{-1} + D_\Gamma \left( U + \frac{E}{k^2} \, z z^T \right) D_\Gamma \right]^{-1}, \qquad D_\Gamma = \mathrm{diag}(\Gamma_1, \ldots, \Gamma_M)
```

Here $`G`$ is block-diagonal over the molecules: each molecule contributes its single-chain correlations, weighted by its volume fraction divided by its length (in units of $`N`$). A counterion of one segment adds $`\phi_C / N`$.

**Validity.** The RPA leaves out fluctuation corrections, of relative order $`\bar{N}^{-1/2}`$, which grow as $`\chi N`$ approaches the order-disorder transition. Grid and contour discretization change the high wavevectors, so comparisons use the lowest modes; finite discretization and fluctuation corrections must be distinguished from sampling uncertainty.

**Example.** The conditions of Fig. 3 of Reference 10: symmetric AB diblock ($`f = 0.5`$), discrete chains with $`N = 90`$, $`\zeta N = 90`$, $`\bar{N} = 10^5`$, $`32^3`$ grid with $`L = 4.5 R_0`$, CUDA backend, with contact interactions or Gaussian smearing of $`a_{\mathrm{int}} = 0.02 R_0`$. The bare $`\chi_b N`$ is set so that the effective $`z_\infty \chi_b N = 9`$, with $`z_\infty`$ from Eq. (29) of Reference 10 (0.9548 and 0.9596; the compressible model has the factor $`1/(1 + t/\zeta)`$ in the integrand), and CL-FTS uses the renormalized bare segment length of its Eq. (26) ($`a_b/a = 0.9929`$ and 0.9931). L-FTS treats $`W_+`$ in the partial saddle-point approximation of the same compressible model. Both methods use the LM integrator (L-FTS: $`\Delta t = 8`$; CL-FTS: $`\Delta t = 2`$, $`\alpha_{ds} = 0.01`$) for 15000 steps from zero fields, and the first 5000 steps are discarded. The figure shows $`S = (S_{AA} + S_{BB} - 2 S_{AB})/4`$, the structure function of $`\phi_A - \phi_B`$ used by Reference 10, averaged over shells of equal $`\vert\mathbf{k}\vert`$; error bars are standard errors from two 5000-step blocks. The dotted curve is the incompressible Gaussian-chain RPA at $`\chi N = 9`$.

![L-FTS, CL-FTS and RPA structure functions](../figures/structure_function_rpa.png)

At the peak ($`k R_0 = 4.63`$), L-FTS gives $`0.306 \pm 0.002`$ (contact) and $`0.295 \pm 0.002`$ (smeared), and CL-FTS gives $`0.316 \pm 0.004`$ and $`0.304 \pm 0.004`$. These agree within about 1 % with the peaks of Fig. 3 of Reference 10 (read from the figure: 0.308 and 0.297 for L-FTS, 0.313 and 0.302 for CL-FTS). Both methods lie below the RPA peak (0.332 at the same $`k`$), which leaves out fluctuation corrections. Reference 10 instead compares the peaks with renormalized one-loop theory and absorbs the remaining difference into a nonlinear $`\chi`$; its coefficient is larger for CL-FTS than for L-FTS (Table I of Reference 10), consistent with CL-FTS lying slightly above L-FTS. The dynamical stabilization used here, which Reference 10 avoided for structure functions, also raises $`S(k^*)`$ slightly (Section 5.2). In CL-FTS, $`S_{AA}`$ also contains a quarter of the total-density fluctuations, about 0.003 at the peak, so it differs from $`S`$.

---

## 7. Implementation

### 7.1 Using the LFTS Class

```python
from polymerfts import lfts
import numpy as np

params = {
    "platform": "cuda",              # or "cpu-mkl", "cpu-fftw"
    "nx": [40, 40, 40],              # Grid points
    "lx": [4.36, 4.36, 4.36],        # Box size

    "chain_model": "discrete",       # or "continuous"
    "ds": 1/90,                      # Contour step (1/N_Ref)

    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 17.0},

    "polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": 0.4},
            {"type": "B", "length": 0.6},
        ],
    }],

    "langevin": {
        "max_step": 10000,           # Total Langevin steps
        "dt": 8.0,                   # Time step (delta_tau * N_Ref)
        "nbar": 10000,               # Invariant polymerization index
    },

    "recording": {
        "dir": "data_simulation",
        "checkpoint_interval": 1000,     # Save fields every N steps
        "sample_interval": 10,    # Compute S(k) every N steps
        "statistics_save_interval": 1000,  # Save S(k) every N steps
    },

    "saddle": {
        "max_iter": 100,
        "tolerance": 1e-4,
    },

    "compressor": {
        "name": "lram",              # Linear Response + Anderson Mixing
        "max_hist": 20,
        "start_error": 5e-1,
        "mix_min": 0.01,
        "mix_init": 0.01,
    },

    "verbose_level": 1,
}

# Initialize and run
random_seed = 12345
simulation = lfts.LFTS(params=params, random_seed=random_seed)
simulation.run(initial_fields={"A": w_A, "B": w_B})
```

### 7.2 Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `nbar` | Invariant polymerization index $`\bar{N}`$ | 100 - 10000 |
| `dt` | Time step $`\delta\tau N`$ | 1.0 - 10.0 |
| `max_step` | Total Langevin steps | 10000 - 100000 |
| `tolerance` | Incompressibility tolerance | 1e-4 |

### 7.3 Noise Amplitude

The noise amplitude depends on system parameters:

```math
\sigma = \sqrt{\frac{2 \cdot \delta\tau N}{\Delta V \cdot \sqrt{\bar{N}}}}
```

where $`\Delta V = V/n_{\mathrm{grid}}`$ is the volume of one grid cell; $`V`$ is the volume of the simulation cell, which includes the angle factor of a non-orthogonal cell ($`\Delta V = \Delta x \Delta y \Delta z`$ only for an orthogonal box).

### 7.4 Output Files

| File | Description |
|------|-------------|
| `fields_NNNNNN.mat` | Potential fields and concentrations |
| `structure_function_NNNNNN.mat` | Structure functions $`S_{AA}`$, $`S_{AB}`$, $`S_{BB}`$ |

---

## 8. References

1. Leibler, L. "Theory of Microphase Separation in Block Copolymers." *Macromolecules* **13**, 1602 (1980).

2. Ganesan, V. & Fredrickson, G. H. "Field-theoretic polymer simulations." *Europhys. Lett.* **55**, 814 (2001).

3. Lennon, E. M., Katsov, K. & Fredrickson, G. H. "Free Energy Evaluation in Field-Theoretic Polymer Simulations." *Phys. Rev. Lett.* **101**, 138302 (2008).

4. Delaney, K. T. & Fredrickson, G. H. "Recent Developments in Fully Fluctuating Field-Theoretic Simulations of Polymer Melts and Solutions." *J. Phys. Chem. B* **120**, 7615 (2016).

5. Beardsley, T. M. & Matsen, M. W. "Fluctuation correction for the order–disorder transition of diblock copolymer melts." *J. Chem. Phys.* **154**, 124902 (2021).

6. Matsen, M. W. & Beardsley, T. M. "Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation." *Polymers* **13**, 2437 (2021).

7. Vorselaars, B. "Efficient Langevin and Monte Carlo sampling algorithms: The case of field-theoretic simulations." *J. Chem. Phys.* **158**, 114117 (2023).

8. Willis, J. D. & Matsen, M. W. "Stabilizing complex-Langevin field-theoretic simulations for block copolymer melts." *J. Chem. Phys.* **161**, 244903 (2024).

9. Morse, D. C., Yong, D. & Chen, K. "Polymer Field Theory for Multimonomer Incompressible Models: Symmetric Formulation and ABC Systems." *Macromolecules* **58**, 816 (2025).

10. Matsen, M. W., Ramírez, J., Willis, J. D. & Pina, P. D. "Accessing the universal phase behavior of block copolymer melts with complex-Langevin field-theoretic simulations." *J. Chem. Phys.* **164**, 014905 (2026).
