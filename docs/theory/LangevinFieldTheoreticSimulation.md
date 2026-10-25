# Langevin Field-Theoretic Simulation

This document describes what the two Langevin field-theoretic simulations of polymerfts share: the Langevin dynamics, the recorded observables, the structure functions and the renormalization of the interaction. [PartialSaddlePointApproximation.md](PartialSaddlePointApproximation.md) describes L-FTS with the partial saddle-point approximation, and [ComplexLangevinSimulation.md](ComplexLangevinSimulation.md) describes complex Langevin simulation (CL-FTS).

## Table of Contents

1. [Overview](#1-overview)
2. [Langevin Dynamics](#2-langevin-dynamics)
3. [Observables and Output Files](#3-observables-and-output-files)
4. [Structure Functions](#4-structure-functions)
5. [Renormalization and Calibration](#5-renormalization-and-calibration)
6. [References](#6-references)

---

## 1. Overview

Field-theoretic simulations (FTS) include the fluctuations of the fields of polymer field theory. Unlike SCFT, which finds a single saddle-point solution, they sample field configurations according to their Boltzmann weight, capturing:

- **Fluctuation effects**: Order-disorder transition shifts, composition fluctuations
- **Correlation functions**: Structure factors, scattering functions
- **Thermodynamic averages**: Ensemble-averaged properties

polymerfts samples the fields with Langevin dynamics in two ways:

| | L-FTS | CL-FTS |
|---|---|---|
| Fluctuating fields | The real auxiliary fields; the imaginary fields stay at their saddle point | Every auxiliary field, which makes the fields complex |
| Model | Incompressible or compressible | Compressible (finite `zeta_n`) |
| Main caveat | The partial saddle-point approximation | Correct convergence of complex Langevin dynamics |
| Class | `LFTS` | `CLFTS` |
| Method document | [PartialSaddlePointApproximation.md](PartialSaddlePointApproximation.md) | [ComplexLangevinSimulation.md](ComplexLangevinSimulation.md) |

Both use the Langevin dynamics of Section 2, record the observables of Section 3 and the structure functions of Section 4, and need the renormalization of Section 5 for a comparison with experiments or other simulations.

---

## 2. Langevin Dynamics

### 2.1 Continuous Langevin Equation

In both methods the real fields evolve according to:

```math
\frac{\partial \Omega_i(\mathbf{r}, \tau)}{\partial \tau} = -\lambda_i \frac{\delta H}{\delta \Omega_i(\mathbf{r}, \tau)} + \eta(\mathbf{r}, \tau)
```

with noise statistics:
- $`\langle \eta(\mathbf{r}, \tau) \rangle = 0`$
- $`\langle \eta(\mathbf{r}, \tau) \eta(\mathbf{r}', \tau') \rangle = 2\lambda_i k_B T \, \delta(\mathbf{r} - \mathbf{r}') \delta(\tau - \tau')`$

Here $`\lambda_i`$ is the relative step size (mobility) of field $`i`$, `dt_scaling` in `lfts.py`, not an eigenvalue: for the first $`M-1`$ auxiliary fields it is the absolute value of the field's eigenvalue in [PolymerFieldTheory.md](PolymerFieldTheory.md#23-effective-hamiltonian) divided by the largest absolute eigenvalue among them, and it is 1 for the last field $`\Omega_M`$. For AB systems $`\lambda_- = \lambda_+ = 1`$. CL-FTS takes the eigenvalues of $`U = X + \zeta N \, \mathbf{e}\mathbf{e}^T`$ and also moves the imaginary fields ([CL-FTS, Section 2](ComplexLangevinSimulation.md#2-complex-langevin-equations)).

### 2.2 Discretized Equation (Per Chain Unit)

Discretizing with cell volume $`\Delta V`$ and time step $`\delta\tau`$:

```math
\Omega_i(\mathbf{r}, \tau + \delta\tau) = \Omega_i(\mathbf{r}, \tau) - \lambda_i \frac{\delta H}{\delta \Omega_i} \delta\tau N + \sigma_i \mathcal{N}(\mathbf{r}, \tau)
```

where:
- $`\mathcal{N}`$: Gaussian noise with zero mean and unit variance
- $`\sigma_i^2 = \lambda_i \frac{2 \delta\tau N}{\Delta V / R_0^3 \cdot \sqrt{\bar{N}}}`$
- $`\bar{N} = b^6 \rho_0^2 N`$: invariant polymerization index

### 2.3 Euler Method

The simplest discretization (first-order in $`\delta\tau`$):

```math
\Omega_-(\mathbf{r}, \tau + \delta\tau) = \Omega_-(\mathbf{r}, \tau) - \lambda_- \left[\Phi_-(\mathbf{r}) + \frac{2}{\chi N}\Omega_-(\mathbf{r})\right] \delta\tau N + \sigma \mathcal{N}(\mathbf{r}, \tau)
```

### 2.4 Leimkuhler-Matthews Method

A more accurate second-order method that averages noise over two time steps:

```math
\Omega_i(\mathbf{r}, \tau + \delta\tau) = \Omega_i(\mathbf{r}, \tau) - \lambda_i \frac{\delta H}{\delta \Omega_i} \delta\tau N + \frac{\sigma_i}{2}\left[\mathcal{N}(\mathbf{r}, \tau) + \mathcal{N}(\mathbf{r}, \tau + \delta\tau)\right]
```

**Advantages:**
- Better sampling of the target distribution
- More stable for larger time steps
- Reduced systematic bias

LM is the default integrator of both methods; `langevin.integrator: "euler"` selects Section 2.3, and [Extensions](../internals/Extensions.md#langevin-integrators) describes how to add an integrator.

### 2.5 Noise Amplitude

The noise amplitude depends on system parameters:

```math
\sigma = \sqrt{\frac{2 \cdot \delta\tau N}{\Delta V \cdot \sqrt{\bar{N}}}}
```

where $`\Delta V = V/n_{\mathrm{grid}}`$ is the volume of one grid cell; $`V`$ is the volume of the simulation cell, which includes the angle factor of a non-orthogonal cell ($`\Delta V = \Delta x \Delta y \Delta z`$ only for an orthogonal box).

---

## 3. Observables and Output Files

Both methods write their measurements into `recording.dir` at the intervals of [`recording`](../references/Parameters.md#recording): `sample_interval` for the samples, `statistics_save_interval` for the files of averages and histories, and `checkpoint_interval` for the fields. [Tools, Section 5](../references/Tools.md#5-file-format-reference) describes the file formats.

| File | Contents |
|------|----------|
| `fields_NNNNNN.mat` | Fields, concentrations and auxiliary fields, with the random-number state and the pending measurements; runs continue from these files |
| `dH_NNNNNN.mat` | Histories of $`H`$, $`\partial H / \partial (\chi N)`$ and of the largest drift |
| `structure_function_NNNNNN.mat` | Structure functions $`S_{ij}`$ averaged over the samples of the interval (Section 4) |
| `run_status.json` | Completion or failure and the last accepted step |

- **Hamiltonian:** $`H`$ and $`\partial H / \partial (\chi N)`$ are recorded at every sample (`H_history`, `dH_history_*`). In CL-FTS they are complex and recorded as real and imaginary parts (`H_history_real`, `H_history_imag`, `dH_history_*_real`, `dH_history_*_imag`). Their ensemble averages are real, so the imaginary parts average to zero within the statistical error.
- **Drift:** `drift_max_history` holds the largest scaled force of each sampled step, including dynamical stabilization in CL-FTS. CL-FTS also records the largest drift of every step for each auxiliary field (`drift_max_step_history`, one column per field) and, with charges, that of $`\psi`$ (`psi_drift_max_history` at the sampled steps, `psi_drift_max_step_history` at every step); checkpoints keep the steps since the last statistics file. The linear part $`\nabla^2 \psi / E`$ of the force of $`\psi`$ is integrated exactly and would dominate the maximum at large $`k`$, so $`\psi`$ is recorded apart. [CL-FTS, Section 3](ComplexLangevinSimulation.md#3-correct-convergence) explains their use.
- **Structure functions:** L-FTS uses the first $`M-1`$ auxiliary fields and real Fourier transforms. CL-FTS uses all fields with $`\lambda_k \neq 0`$ and full complex Fourier transforms, recorded as `structure_function_*_real` and `structure_function_*_imag`; in its compressible model $`S_{AA}`$ also contains the total-density fluctuations. For AB, the composition structure function is $`(S_{AA} + S_{BB} - 2 S_{AB})/4`$ (Section 4.4).
- **Fields:** CL-FTS stores the complex fields and concentrations as `_real`/`_imag` pairs.

---

## 4. Structure Functions

### 4.1 Definition

The structure function is the correlation of the segment densities. In units of $`\rho_0 N`$,

```math
S_{ij}(\mathbf{k}) = \frac{1}{\rho_0 N V} \left\langle \hat{\rho}_i(\mathbf{k}) \, \hat{\rho}_j(-\mathbf{k}) \right\rangle, \qquad \mathbf{k} \neq 0
```

where $`\hat{\rho}_i(\mathbf{r})`$ is the microscopic density of the segments of type $`i`$ and $`\rho_0`$ is the segment density. In the disordered phase at weak fluctuations, $`S_{ij}`$ approaches the random-phase approximation of [Section 4.4](#44-rpa-reference).

### 4.2 Estimator

A simulation samples fields, not particles. In a field configuration the chains do not interact, and the concentration $`\phi_i`$ is their average density, so $`\langle \tilde{\phi}_i \tilde{\phi}_j \rangle`$ misses the correlation within a chain. At fixed particles, each auxiliary field is Gaussian with a mean proportional to the density it couples to. The correlation of a field with a concentration is therefore proportional to $`S_{ij}`$ itself ([Structure Function Estimators, Section 2](StructureFunctionEstimators.md#2-the-gaussian-shift-identity)). Combined into the chemical potential fields of the monomer types, this gives $`S_{ij}`$ as an average over the sampled fields alone:

```math
S_{ij}(\mathbf{k}) = \frac{V \sqrt{\bar{N}}}{\Gamma_i(\mathbf{k})} \left\langle \tilde{\mu}_i(\mathbf{k}) \, \tilde{\phi}_j(-\mathbf{k}) \right\rangle, \qquad \mu_i(\mathbf{r}) = \sum_{k:\, \lambda_k \neq 0} \frac{A_{ik}\, \Omega_k(\mathbf{r})}{\lambda_k}
```

- Tildes denote Fourier transforms normalized by the number of grid points. $`\Gamma_i`$ is the smearing function of type $`i`$ ([Polymer Field Theory, Section 4](PolymerFieldTheory.md#4-smeared-interactions); $`\Gamma_i = 1`$ for contact interactions).
- The identity needs fluctuating fields, so the sum runs over the sampled fields with $`\lambda_k \neq 0`$: the first $`M-1`$ in L-FTS, all of them in CL-FTS.
- Each sample takes the fields and the concentrations of the same accepted step, and the recorded $`S_{ij}`$ is the average over the trajectory.

The estimator equals the definition when the sampled fields cover every fluctuating composition mode. [Structure Function Estimators](StructureFunctionEstimators.md) gives the derivation and the remaining cases:
- **Projections:** what L-FTS records in compressible models, and what is recorded for modes without an auxiliary field;
- **Charges:** the completion by $`\psi`$ for such modes, e.g. two types that differ only in charge;
- **Charge structure function**, and a comparison with BioFTS.

### 4.3 Physical Interpretation

- **Peak position**: Characteristic length scale of microphase separation
- **Peak height**: Degree of ordering (diverges at spinodal)
- **Low-$`k`$ behavior**: Long-wavelength fluctuations

### 4.4 RPA Reference

In the disordered phase at weak fluctuations (large $`\bar{N}`$, $`\chi N`$ well below the order-disorder transition) the structure function approaches the random-phase approximation (RPA) of Leibler (Reference 1). `tests/TestStructureFunctionRPA.py` compares L-FTS and CL-FTS runs, with and without smearing, against the formulas below, and `tests/validation/validate_fts_sampling.py` uses the same reference for the sampling studies of [Numerical Validation](../internals/NumericalValidation.md).

Lengths are in units of $`R_0`$ and $`S_{ij}`$ is in units of $`\rho_0 N`$ as in Section 4.1, i.e. $`S_{ij}`$ is Leibler's per-monomer structure function divided by $`N`$.

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

**Smearing.** The interactions act on the smeared densities, so $`U`$ (or $`X`$) is multiplied by $`\Gamma(\mathbf{k})^2`$, while the recorded structure function is that of the densities (Sections 4.1 and 4.2):

```math
S(\mathbf{k}) = \left[ G^{-1} + \Gamma(\mathbf{k})^2 \, U \right]^{-1}, \qquad \frac{1}{S_{AA}(k)} = F(x) - 2\chi N \, \Gamma(k)^2 \quad \text{(incompressible AB)}
```

with $`\Gamma(k)^2 = \exp(-a^2 k^2)`$ for Gaussian smearing. This holds in a periodic box, where the recorded $`S_{ij}`$ is divided by $`\Gamma`$.

**Charges.** With one smearing function per type and the valences of [Polymer Field Theory, Section 5](PolymerFieldTheory.md#5-electrostatic-interactions):

```math
S(\mathbf{k}) = \left[ G^{-1} + D_\Gamma \left( U + \frac{E}{k^2} \, z z^T \right) D_\Gamma \right]^{-1}, \qquad D_\Gamma = \mathrm{diag}(\Gamma_1, \ldots, \Gamma_M)
```

Here $`G`$ is block-diagonal over the molecules: each molecule contributes its single-chain correlations, weighted by its volume fraction divided by its length (in units of $`N`$). A counterion of one segment adds $`\phi_C / N`$.

**Validity.** The RPA leaves out fluctuation corrections, of relative order $`\bar{N}^{-1/2}`$, which grow as $`\chi N`$ approaches the order-disorder transition. Grid and contour discretization change the high wavevectors, so comparisons use the lowest modes; finite discretization and fluctuation corrections must be distinguished from sampling uncertainty.

**Example.** The conditions of Fig. 3 of Reference 9: symmetric AB diblock ($`f = 0.5`$), discrete chains with $`N = 90`$, $`\zeta N = 90`$, $`\bar{N} = 10^5`$, $`32^3`$ grid with $`L = 4.5 R_0`$, CUDA backend, with contact interactions or Gaussian smearing of $`a_{\mathrm{int}} = 0.02 R_0`$. The bare $`\chi_b N`$ is set so that the effective $`z_\infty \chi_b N = 9`$, with $`z_\infty`$ from Eq. (29) of Reference 9 (0.9548 and 0.9596; the compressible model has the factor $`1/(1 + t/\zeta)`$ in the integrand), and CL-FTS uses the renormalized bare segment length of its Eq. (26) ($`a_b/a = 0.9929`$ and 0.9931). L-FTS treats $`W_+`$ in the partial saddle-point approximation of the same compressible model. Both methods use the LM integrator (L-FTS: $`\Delta t = 8`$; CL-FTS: $`\Delta t = 2`$, $`\alpha_{ds} = 0.01`$) for 15000 steps from zero fields, and the first 5000 steps are discarded. The figure shows $`S = (S_{AA} + S_{BB} - 2 S_{AB})/4`$, the structure function of $`\phi_A - \phi_B`$ used by Reference 9, averaged over shells of equal $`\vert\mathbf{k}\vert`$; error bars are standard errors from two 5000-step blocks. The dotted curve is the incompressible Gaussian-chain RPA at $`\chi N = 9`$.

![L-FTS, CL-FTS and RPA structure functions](../figures/structure_function_rpa.png)

At the peak ($`k R_0 = 4.63`$), L-FTS gives $`0.306 \pm 0.002`$ (contact) and $`0.295 \pm 0.002`$ (smeared), and CL-FTS gives $`0.316 \pm 0.004`$ and $`0.304 \pm 0.004`$. These agree within about 1 % with the peaks of Fig. 3 of Reference 9 (read from the figure: 0.308 and 0.297 for L-FTS, 0.313 and 0.302 for CL-FTS). Both methods lie below the RPA peak (0.332 at the same $`k`$), which leaves out fluctuation corrections. Reference 9 instead compares the peaks with renormalized one-loop theory and absorbs the remaining difference into a nonlinear $`\chi`$; its coefficient is larger for CL-FTS than for L-FTS (Table I of Reference 9), consistent with CL-FTS lying slightly above L-FTS. The dynamical stabilization used here, which Reference 9 avoided for structure functions, also raises $`S(k^*)`$ slightly ([CL-FTS, Section 4](ComplexLangevinSimulation.md#4-hot-spots-and-dynamical-stabilization)). In CL-FTS, $`S_{AA}`$ also contains a quarter of the total-density fluctuations, about 0.003 at the peak, so it differs from $`S`$.

---

## 5. Renormalization and Calibration

### 5.1 Effective Interaction and Chain Size

The ultraviolet (UV) fluctuations of the grid renormalize the interaction: to linear order, the effective Flory-Huggins parameter is $`\chi N = z_\infty \chi_b N`$, where the bare $`\chi_b N`$ is `chi_n` (Reference 5). This holds for L-FTS and CL-FTS, with and without compressibility. CL-FTS has a second correction: the fluctuations of the imaginary (pressure) field also change the chain size. The bare segment length $`a_b`$ is therefore chosen so that $`R_0 = a N^{1/2}`$ in the limit $`\chi_b \to 0`$. In L-FTS and SCFT, $`a_b = a`$.

**Discrete chains.** Eqs. (29) and (26) of Reference 9 give, in units of $`R_0`$ with $`a^2 = 1/N`$ and the monomer density $`\rho_0 = \bar{N}^{1/2} N`$,

```math
z_\infty = 1 - \frac{1}{(2\pi)^3 \rho_0} \int \frac{d\mathbf{k}}{t(k) \left[ 1 + t(k)/\zeta \right]}, \qquad \frac{a_b}{a} = \left[ 1 + \frac{a^2}{72 (2\pi)^3 \rho_0} \int \frac{k^2 \, d\mathbf{k}}{s^2(k) \left[ 1 + t(k)/\zeta \right]} \right]^{-1}
```

In these formulas:

- $`t(k) = \tanh(k^2 a^2/12)/\Gamma^2(k)`$ and $`s(k) = \sinh(k^2 a^2/12)`$;
- $`\Gamma(k)`$ is the smearing function, and $`\zeta N`$ is `zeta_n`;
- the integrals run over $`-\pi/\Delta_\nu \le k_\nu \le \pi/\Delta_\nu`$.

In the incompressible limit $`\zeta \to \infty`$ the factor $`1 + t/\zeta`$ goes to 1. For $`\zeta \to \infty`$ and $`\Gamma = 1`$, $`z_\infty`$ equals the bond sum of Reference 5, $`1 - (1 + 2\sum_i P_i)/(\rho_0 \Delta V)`$, which the former `renormalization_discrete.m` evaluated.

**Continuous chains.** Continuous chains use the limit $`N \to \infty`$ of the discrete formulas at a fixed grid in units of $`R_0`$. In this limit, $`\tanh u`$ and $`\sinh u`$ both become $`u = k^2/(12N)`$:

```math
z_\infty = 1 - \frac{1}{(2\pi)^3 \sqrt{\bar{N}}} \int \frac{12 \, \Gamma^2(k) \, d\mathbf{k}}{k^2 \left[ 1 + k^2/(12 \zeta N \Gamma^2(k)) \right]}, \qquad \frac{a_b}{a} = \left[ 1 + \frac{1}{(2\pi)^3 \sqrt{\bar{N}}} \int \frac{2 \, d\mathbf{k}}{k^2 \left[ 1 + k^2/(12 \zeta N \Gamma^2(k)) \right]} \right]^{-1}
```

Both corrections grow without bound as the grid is refined, as $`z_\infty`$ of discrete chains with contact interactions does.

### 5.2 Calibration

`scripts/analysis/renormalization.py` ([Tools, Section 3](../references/Tools.md#3-χn-renormalization)) evaluates both corrections from a fields file or from options. To use the calibration, set `chi_n` to the $`\chi_b N`$ of the target $`z_\infty \chi_b N`$ and, in CL-FTS, multiply all `segment_lengths` by $`a_b/a`$. Reference 9 adds a quadratic term, $`\chi = z_\infty \chi_b + c_2 (z_\infty \chi_b)^2`$, fitted to renormalized one-loop theory; $`c_2`$ differs between CL-FTS and L-FTS. Section 4.4 gives an example.

### 5.3 Earlier RPA Calibration

The former `renormalization_rpa.m` integrated the athermal RPA of Reference 4 instead. That integral also contains a finite term of order $`\bar{N}^{-1/2}`$ from $`k \lesssim 1/R_0`$, which Eq. (29) does not include because it is derived for $`N \to \infty`$. For example, for a $`32^3`$ grid with $`L = 4 R_0`$ and $`\bar{N} = 10^4`$, the former script gave $`z_\infty = 0.858`$, while the formula above gives 0.813.

---

## 6. References

1. Leibler, L. "Theory of Microphase Separation in Block Copolymers." *Macromolecules* **13**, 1602 (1980).

2. Ganesan, V. & Fredrickson, G. H. "Field-theoretic polymer simulations." *Europhys. Lett.* **55**, 814 (2001).

3. Lennon, E. M., Katsov, K. & Fredrickson, G. H. "Free Energy Evaluation in Field-Theoretic Polymer Simulations." *Phys. Rev. Lett.* **101**, 138302 (2008).

4. Vorselaars, B., Stasiak, P. & Matsen, M. W. "Field-Theoretic Simulation of Block Copolymers at Experimentally Relevant Molecular Weights." *Macromolecules* **48**, 9071 (2015).

5. Beardsley, T. M. & Matsen, M. W. "Calibration of the Flory-Huggins interaction parameter in field-theoretic simulations." *J. Chem. Phys.* **150**, 174902 (2019).

6. Beardsley, T. M. & Matsen, M. W. "Fluctuation correction for the order–disorder transition of diblock copolymer melts." *J. Chem. Phys.* **154**, 124902 (2021).

7. Vorselaars, B. "Efficient Langevin and Monte Carlo sampling algorithms: The case of field-theoretic simulations." *J. Chem. Phys.* **158**, 114117 (2023).

8. Morse, D. C., Yong, D. & Chen, K. "Polymer Field Theory for Multimonomer Incompressible Models: Symmetric Formulation and ABC Systems." *Macromolecules* **58**, 816 (2025).

9. Matsen, M. W., Ramírez, J., Willis, J. D. & Pina, P. D. "Accessing the universal phase behavior of block copolymer melts with complex-Langevin field-theoretic simulations." *J. Chem. Phys.* **164**, 014905 (2026).
