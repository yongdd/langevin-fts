# Structure Function Estimators in Field-Theoretic Simulations

A field-theoretic simulation (L-FTS or CL-FTS) samples fields, not particles. A particle correlation function such as the structure function is therefore measured as the average of an expression of the fields along the Langevin trajectory, an *estimator*. This is the term of statistics and Monte Carlo methods: several estimators can have the same mean but different variances. Field-theoretic literature often calls these expressions *operators* (density, chemical potential or pressure operators; Reference 4). This document is self-contained. It sets up the model (Section 1), derives the identity behind every estimator used in polymerfts (Section 2), and builds three estimators from it:

- the estimator of the auxiliary fields (Section 3) and its limits (Section 4);
- its completion by the electrostatic potential for composition modes without an auxiliary field (Section 5);
- the charge structure function (Section 6).

Section 7 gives the tests and an independent comparison with BioFTS.

## Table of Contents

1. [Model and Notation](#1-model-and-notation)
2. [The Gaussian Shift Identity](#2-the-gaussian-shift-identity)
3. [Auxiliary-Field Estimator](#3-auxiliary-field-estimator)
4. [Fields at Their Saddle Point](#4-fields-at-their-saddle-point)
5. [Modes Without an Auxiliary Field](#5-modes-without-an-auxiliary-field)
6. [Charge Structure Function](#6-charge-structure-function)
7. [Validation](#7-validation)
8. [References](#8-references)

---

## 1. Model and Notation

**Units and normalization.**
- $`M`$ monomer types $`i = 1, \ldots, M`$ form chains of segments with the reference chain length $`N`$.
- $`\rho_0`$ is the segment density. Lengths are in units of $`R_0 = b N^{1/2}`$, and $`C = \rho_0 R_0^3 / N = \bar{N}^{1/2}`$.
- $`\hat\rho_i(\mathbf{r})`$ is the microscopic density of the segments of type $`i`$. A tilde denotes a Fourier transform; for a field on the grid it is normalized by the number of grid points.

**Structure function.** In units of $`\rho_0 N`$ (Leibler's per-monomer structure function divided by $`N`$), the structure function is

```math
S_{ij}(\mathbf{k}) = \frac{1}{\rho_0 N V} \left\langle \hat\rho_i(\mathbf{k}) \, \hat\rho_j(-\mathbf{k}) \right\rangle, \qquad \mathbf{k} \neq 0
```

For a melt of ideal homopolymers, $`S_{AA}(k)`$ is the Debye function $`2(x + e^{-x} - 1)/x^2`$ with $`x = k^2 R_0^2/6`$.

**Interactions.** The interactions act between smeared densities $`\Gamma_i * \hat\rho_i`$, where $`\Gamma_i(\mathbf{k})`$ is the smearing function of type $`i`$ ($`\Gamma_i = 1`$ for contact interactions).
- **Short range:** the interaction matrix $`U`$ is the $`\chi N`$ matrix plus, for the compressible model, $`\zeta N`$ in every entry. Its eigenvalues are $`\lambda_k`$, and its eigenvectors $`\mathbf{a}_k`$ are the columns of $`A`$, normalized so that $`A A^T = M I`$:

```math
U = \frac{1}{M} \sum_{k} \lambda_k \, \mathbf{a}_k \mathbf{a}_k^T, \qquad \frac{1}{2} \hat\rho^T U \hat\rho = \sum_k \frac{\lambda_k}{2M} \, y_k^2, \qquad y_k = \sum_l A_{lk} \, \Gamma_l \hat\rho_l
```

- **Charges** (optional): type $`i`$ carries the valence $`z_i`$ per segment. The Coulomb energy of the smeared charge density $`c = \sum_l z_l \Gamma_l \hat\rho_l`$ is $`(E/2k^2)\vert \tilde c(\mathbf{k}) \vert^2`$ per wavevector, with $`E = 4\pi l_B \rho_0 N R_0^2`$ in units of the reference chain.

**Fields.** The Hubbard-Stratonovich transformation replaces each mode $`k`$ of $`U`$ by an auxiliary field $`\Omega_k`$, with Gaussian weight $`(M/2\lambda_k)\,\Omega_k^2`$, and the Coulomb energy by the electrostatic potential $`\psi`$, with Gaussian weight $`(k^2/2E)\vert\tilde\psi\vert^2`$. Fields of negative weight are integrated along the imaginary axis; CL-FTS stores them on the rotated contour. The chains feel the potentials

```math
w_i = \Gamma_i * \Big( \sum_k A_{ik} \Omega_k + z_i \psi \Big)
```

A mode with $`\lambda_k = 0`$ does not interact and has no field.

**Concentrations.** In a field configuration $`w`$ the chains do not interact. Their average density is the concentration computed by the simulation: $`\langle \hat\rho_i \rangle_w = \rho_0 \phi_i[w]`$.

## 2. The Gaussian Shift Identity

**Why ⟨φφ⟩ is not the structure function.** A particle average $`\langle F \rangle`$ becomes the field average of $`\langle F \rangle_w`$. For a product of two densities,

```math
\langle \hat\rho_i \hat\rho_j \rangle_w = \rho_0^2 \, \phi_i[w] \, \phi_j[w] + (\text{correlation within a chain in the field } w)
```

The second term is a single-chain response function and is expensive in an arbitrary field. So $`\langle \phi_i \phi_j \rangle`$ alone is not $`S_{ij}`$. An estimator should contain the particle densities only linearly, because $`\langle \hat\rho_j \rangle_w = \rho_0 \phi_j[w]`$ is exact.

**The identity.** Fix the particle configuration. The field $`\Omega_k`$ then appears only in its Gaussian weight and in its coupling to $`y_k`$. Completing the square shows that $`\Omega_k`$ is Gaussian with a mean proportional to $`y_k`$. For fields on the imaginary contour, the same algebra holds by analytic continuation. For any observable $`F`$ of the particles, this gives

```math
\langle \Omega_k \, F \rangle = \frac{\lambda_k}{M} \, \langle y_k \, F \rangle
```

**From particles to fields.** Integrating out the particles at fixed fields replaces $`F`$ by $`\langle F \rangle_w`$. For $`F = \hat\rho_j(-\mathbf{k})`$, which is linear in the densities, this is $`\rho_0 \phi_j`$. With the normalization of the field theory per reference chain, the identity becomes an average over the sampled fields alone:

```math
V \sqrt{\bar{N}} \left\langle \tilde\Omega_k(\mathbf{k}) \, \tilde\phi_j(-\mathbf{k}) \right\rangle = \frac{\lambda_k}{M} \sum_l A_{lk} \, \Gamma_l(\mathbf{k}) \, S_{lj}(\mathbf{k})
```

The intrachain correlation is contained in $`S_{lj}`$ on the right, without being computed. The electrostatic potential obeys the same identity, with $`\lambda_k/M \to E/k^2`$ and $`\mathbf{a}_k \to \mathbf{z}`$:

```math
V \sqrt{\bar{N}} \left\langle \frac{k^2}{E} \tilde\psi(\mathbf{k}) \, \tilde\phi_j(-\mathbf{k}) \right\rangle = \sum_l z_l \, \Gamma_l(\mathbf{k}) \, S_{lj}(\mathbf{k})
```

**Field-field form.** At fixed particles, $`\Omega_k`$ fluctuates around its mean $`(\lambda_k/M)\, y_k`$ with the Gaussian variance $`-\lambda_k/M`$ (in the same normalization). This variance is positive for real fields ($`\lambda_k < 0`$) and negative on the rotated contour of imaginary fields. Hence

```math
\mathbf{a}_k^T \Gamma S \Gamma \, \mathbf{a}_k = \left(\frac{M}{\lambda_k}\right)^2 \left[ V \sqrt{\bar{N}} \left\langle \tilde\Omega_k(\mathbf{k}) \, \tilde\Omega_k(-\mathbf{k}) \right\rangle + \frac{\lambda_k}{M} \right], \qquad \mathbf{z}^T \Gamma S \Gamma \, \mathbf{z} = \left(\frac{k^2}{E}\right)^2 \left[ V \sqrt{\bar{N}} \left\langle \tilde\psi(\mathbf{k}) \, \tilde\psi(-\mathbf{k}) \right\rangle + \frac{E}{k^2} \right]
```

The mixed and the field-field forms have the same mean, but different variances (Section 7).

**In the literature.**
- For an AB melt, the identity written as a correlation of two fields is the estimator $`S = \rho_0 \langle W_- W_- \rangle/(V\chi_b^2) - \rho_0/(2\chi_b)`$, Eq. (21) of Reference 5, which cites References 1 and 2.
- Reference 4 treats field-theoretic simulations and their operators in general, for polymers and other fluids.
- BioFTS evaluates the mixed form $`i G_0 \langle \tilde\rho \, \Psi \rangle`$ in each interaction channel, including its Coulomb channel.

## 3. Auxiliary-Field Estimator

Combine the fields into the chemical potential fields of the monomer types:

```math
\mu_i(\mathbf{r}) = \sum_{k:\, \lambda_k \neq 0} \frac{A_{ik} \, \Omega_k(\mathbf{r})}{\lambda_k}, \qquad S_{ij}(\mathbf{k}) = \frac{V \sqrt{\bar{N}}}{\Gamma_i(\mathbf{k})} \left\langle \tilde\mu_i(\mathbf{k}) \, \tilde\phi_j(-\mathbf{k}) \right\rangle
```

By the identity of Section 2,

```math
V \sqrt{\bar{N}} \langle \tilde\mu_i \, \tilde\phi_j(-\mathbf{k}) \rangle = \sum_{k:\, \lambda_k \neq 0} \frac{A_{ik}}{M} \sum_l A_{lk} \, \Gamma_l S_{lj} = (P \, \Gamma S)_{ij}, \qquad P = \frac{1}{M} \sum_{k:\, \lambda_k \neq 0} \mathbf{a}_k \mathbf{a}_k^T
```

Since $`A A^T = M I`$, the projector $`P`$ is the identity when every mode has $`\lambda_k \neq 0`$, and the estimator then gives $`\Gamma_i S_{ij}`$ exactly.

**Implementation details.**
- **Smearing:** the recorded $`S_{ij}`$ is divided by $`\Gamma_i`$, the smearing function of the first type of the pair; modes with $`\Gamma_i < 10^{-12}`$ are recorded as 0. With walls, $`\Gamma_i`$ is defined in the cosine basis while $`S_{ij}`$ is a Fourier transform, so the estimator is recorded without the division.
- **Complex fields:** the fields of CL-FTS are complex, so $`\tilde\phi_j(-\mathbf{k})`$ is computed as the conjugate of the transform of $`\bar\phi_j`$. For real fields it is $`\tilde\phi_j^*(\mathbf{k})`$.
- **Same state:** the fields and the concentrations of one sample belong to the same accepted step.
- **AB melts:** on average, the estimator equals $`V\sqrt{\bar{N}}\,\langle \vert\tilde{W}_-(\mathbf{k})\vert^2\rangle/(\chi N)^2 - 1/(2\chi N)`$.

## 4. Fields at Their Saddle Point

The identity needs a fluctuating field. L-FTS keeps imaginary fields at their saddle point (partial saddle-point approximation), so its sum runs over the first $`M-1`$ fields with $`\lambda_k \neq 0`$: all but the pressure field (incompressible model) or all but the field of the largest eigenvalue (compressible model).

- **Real fields:** when these $`M-1`$ fields are real, the estimator gives $`P S`$ with $`P`$ of their eigenvectors. In incompressible models $`\sum_i \delta\phi_i = 0`$, so $`P S = S`$.
- **Compressible L-FTS:** the component along the eigenvector of the largest eigenvalue, close to the total density, is not recorded.
- **An imaginary field among the first M−1:** it also stays at its saddle point, and L-FTS warns. Its term then holds the correlation of the field-averaged concentrations without the intrachain part, and the estimator is not the structure function.

CL-FTS samples every field.

## 5. Modes Without an Auxiliary Field

A composition mode with $`\lambda_k = 0`$ has no field, so the estimator of Section 3 gives $`P \Gamma S`$ with $`P`$ projecting it out. Such modes arise between monomer types that interact identically. An example is a polyampholyte whose types E and K differ only in charge: E − K is such a mode.

With screened charges (`screening_length`, [Polymer Field Theory, Section 5.4](PolymerFieldTheory.md#54-screened-coulomb-interaction)), $`k^2/E`$ becomes $`(k^2 + \kappa^2)/E`$, the inverse of the kernel, in this section and in Section 6.

**Completion with ψ.** With charges, $`\psi`$ measures the component of $`\Gamma S`$ along $`\mathbf{z}`$ (Section 2), while the auxiliary fields measure $`P \Gamma S`$. Subtract from the $`\psi`$ component what $`P`$ already contains. The remainder is the component along

```math
\mathbf{u} = (I - P)\, \mathbf{z}, \qquad \mathbf{u}^T \Gamma S = \mathbf{z}^T \Gamma S - \mathbf{z}^T P \, \Gamma S = V \sqrt{\bar{N}} \Big\langle \Big[ \frac{k^2}{E} \tilde\psi - \sum_l z_l \tilde\mu_l \Big] \tilde\phi_j(-\mathbf{k}) \Big\rangle
```

Adding this component along $`\mathbf{u}`$ to the chemical potential fields,

```math
\tilde\mu_i \;\to\; \tilde\mu_i + \frac{u_i}{\mathbf{u} \cdot \mathbf{u}} \Big[ \frac{k^2}{E} \tilde\psi - \sum_l z_l \tilde\mu_l \Big]
```

makes the estimator of Section 3 give $`(P + \mathbf{u}\mathbf{u}^T/\mathbf{u}\cdot\mathbf{u})\, \Gamma S`$.

**When this is exact.** The result equals $`\Gamma S`$ when the zero-eigenvalue modes are spanned by $`\mathbf{u}`$, e.g. a single such mode with a charge difference. With several zero-eigenvalue modes, only their charge direction is completed.

**When nothing changes.** If $`\mathbf{u} = 0`$, the charges lie along modes that already have fields, and the estimator is unchanged. This is the case without charges, with $`E = 0`$, and when every mode has a field. At $`\mathbf{k} = 0`$ the term with $`\tilde\psi`$ is left out: this mode is not a structure function in the canonical ensemble, and with screening $`\kappa^2 \tilde\psi(0)/E`$ would carry a net charge.

**In polymerfts.** CL-FTS applies the completion when it has electrostatics and $`\mathbf{u} \neq 0`$ (`CLFTS._psi_probe`). The completion follows from the identity of Section 2 and was derived for polymerfts; it is not taken from a publication.

## 6. Charge Structure Function

The structure function of the smeared charge density follows from the recorded $`S_{ij}`$, or directly from $`\psi`$ by the identity of Section 2 with $`\tilde c = \sum_l z_l \Gamma_l \tilde\phi_l`$:

```math
\sum_{ij} z_i \Gamma_i \, S_{ij} \, \Gamma_j z_j = V \sqrt{\bar{N}} \left\langle \frac{k^2}{E} \tilde\psi(\mathbf{k}) \, \tilde c(-\mathbf{k}) \right\rangle
```

Divided by $`\Gamma^2`$ for a common smearing function, this is $`S_{zz} = \sum_{ij} z_i z_j S_{ij}`$.

**Two forms with complementary noise.** The same quantity $`S_c = \sum_{ij} z_i \Gamma_i S_{ij} \Gamma_j z_j`$ has a mixed form and a field-field form (Section 2). Per sample, they are

```math
M = \frac{k^2}{E} \, V \sqrt{\bar{N}} \, \tilde\psi(\mathbf{k}) \, \tilde c(-\mathbf{k}), \qquad F = \left(\frac{k^2}{E}\right)^2 \left[ V \sqrt{\bar{N}} \, \tilde\psi(\mathbf{k}) \, \tilde\psi(-\mathbf{k}) + \frac{E}{k^2} \right]
```

Their variances differ with the strength of screening at the wavevector, $`(E/k^2)\, S_c`$:

- **Strong screening** ($`(E/k^2) S_c \to 1`$, low $`k`$): the bracket of $`F`$ is close to its constant, so $`F`$ has far less variance than $`M`$.
- **Weak screening** (large $`k`$): $`S_c`$ is a small difference between two large terms of $`F`$, and $`M`$ is better (Section 7).

**Combined estimator.** Since both forms have the mean $`S_c`$, their difference $`D = F - M`$ has mean zero; it is a control variate. For any weight $`w`$, $`\langle M \rangle + w \langle D \rangle`$ is an estimator of $`S_c`$. Its variance is least for

```math
w = -\frac{\mathrm{Cov}(M, D)}{\mathrm{Var}(D)}
```

CL-FTS with Coulomb coupling accumulates $`M`$, $`D`$, $`M D`$ and $`D^2`$ at each wavevector. Each `structure_function_*.mat` then contains:
- `charge_structure_function_real` and `_imag`: $`\langle M \rangle + w \langle D \rangle`$ in units of $`\rho_0 N`$, with $`w`$ from the covariances of the same statistics interval; the mode $`\mathbf{k} = 0`$ is 0.
- `charge_structure_function_weight`: $`w`$. A weight of 0 is the mixed form, 1 the field-field form.
- `charge_structure_function_mixed_real`, `_imag` and `charge_structure_function_field_field_real`, `_imag`: the two forms alone, $`\langle M \rangle`$ and $`\langle F \rangle`$.

Details:
- **Smearing:** $`S_c`$ is the structure function of the smeared charge density; with a common smearing function, $`S_{zz} = S_c/\Gamma^2`$.
- **Gaussian regime:** at weak fluctuations the charge density responds linearly to $`\psi`$, $`\tilde c \approx -G\tilde\psi`$ with the single-chain response $`G`$. Then $`M`$ and $`F`$ are both linear in $`\tilde\psi(\mathbf{k})\tilde\psi(-\mathbf{k})`$, and the weight removes this term. What remains is the RPA computed from $`G`$, with almost no variance. In this regime the combined estimator therefore does not test how well $`\psi`$ is sampled; the two forms alone do. In the polyampholyte of `TestChargedStructureFunctionRPA`, a $`\psi`$ noise amplitude scaled by 2 or 1/2 left the combined estimator at the RPA to $`10^{-3}`$, while the mixed form gave 4 and 1/4 of it.
- **Approximation in w:** the weight uses the variance of single samples. Neighboring samples are correlated, so $`w`$ is close to, but not exactly, the weight of least variance of the time average. $`\langle M \rangle + w \langle D \rangle`$ remains unbiased up to the statistical error of $`w`$.
- **Time step:** with the Leimkuhler-Matthews noise of its step (integrators that declare `averaged_noise`, such as the default `lm`), $`\psi`$ samples Gaussian fluctuations exactly for any time step ([CL-FTS theory, Section 6](ComplexLangevinSimulation.md#6-charged-systems)), so both forms are free of time-step error in the Gaussian regime. With the Euler integrator, $`\psi`$ takes fresh noise, and the explicit charge density changes the variance of $`\psi`$ at first order in $`\gamma \delta t`$, with $`\gamma = s k^2/E`$ and $`s`$ = `psi_dt_scaling`. Where the charge density follows $`\psi`$ alone, the field-field form then becomes $`\langle F \rangle \approx S_c \left[1 - \tanh(\gamma\delta t/2)\right]`$, a large error where screening is weak. In the polyampholyte of `TestChargedStructureFunctionRPA` ($`E = 300`$, $`\delta t = 1`$), this step gave $`\langle F \rangle/S_c`$ from 0.98 at the lowest shell to about 0.7 at the eighth and the mixed form 2 % above $`S_c`$; with the averaged noise, both agree with the RPA within their errors (0.98 ± 0.03 and 1.005 ± 0.004 on average). The combination stays at the RPA in the Gaussian regime either way.
- **Restarts:** the pending sums are stored in the checkpoints and continue after a restart.

## 7. Validation

**RPA tests.** In the disordered phase at weak fluctuations (large $`\bar{N}`$), the structure function approaches the random-phase approximation (RPA). With the single-chain correlation matrix $`G`$ and $`D_\Gamma = \mathrm{diag}(\Gamma_1, \ldots, \Gamma_M)`$,

```math
S = \left[ G^{-1} + D_\Gamma \left( U + \frac{E}{k^2} \mathbf{z}\mathbf{z}^T \right) D_\Gamma \right]^{-1}
```

The regression tests compare the estimators with this formula:
- `TestStructureFunctionRPA`: L-FTS and CL-FTS of an AB diblock melt, with and without smearing.
- `TestChargedStructureFunctionRPA`:
  - a charged diblock with counterions, where every mode has a field;
  - a polyampholyte melt (E5K5E5K5, no $`\chi`$), whose charge mode has none.

  In the polyampholyte, the completed estimator gives the charge structure function of the RPA. Without the completion, $`S_{EE} + S_{KK} - 2 S_{EK}`$ is close to zero.

**Comparison with BioFTS.** BioFTS (Reference 3) samples the same bead-spring model with the same Gaussian smearing, an Edwards excluded volume $`v`$ and a Coulomb channel. Its estimator is $`i G_0 \langle \tilde\rho \Psi \rangle`$ per channel, and it advances the fields semi-implicitly.

Conditions (bead units, $`b = 1`$):
- chains of $`N = 20`$ beads at bead density $`\rho_b = 2`$, with $`v = 0.05`$ and smearing length $`a = b/\sqrt{6}`$;
- a periodic $`16^3`$ grid of spacing $`a`$;
- a homopolymer solution, and a polyampholyte E5K5E5K5 with $`l_B = 0.5`$.

The mapping to polymerfts is $`R_0 = b\sqrt{N}`$, $`\bar{N} = \rho_b^2 N`$, $`\zeta N = v \rho_b N`$ and $`E = 4\pi l_B \rho_b N^2`$.

Each code made four independent runs of 35000 steps after equilibration; BioFTS at $`\delta t = 2 \times 10^{-3}`$ made two runs of 18000 steps. The results compare the smeared structure functions on shells of equal $`\vert\mathbf{k}\vert`$:

| Quantity | Agreement between the codes | Remark |
|---|---|---|
| Density, homopolymer | within 1-2 % | close to the RPA |
| Density, polyampholyte | within 1-2 % | up to 28 % above the RPA at the lowest shell; both codes show the same fluctuation correction |
| Charge, polyampholyte | within 1-3 % at all but the lowest shell | at the lowest shell, BioFTS gives 0.121 ± 0.004 at $`\delta t = 5 \times 10^{-4}`$ and 0.132 ± 0.003 at $`2 \times 10^{-3}`$; polymerfts gives 0.138 ± 0.002 |

For the charge structure function, polymerfts used the estimator of Section 6. BioFTS approaches polymerfts as its time step decreases, as expected from the bias of semi-implicit stepping. Both codes use complex Langevin dynamics, so their agreement checks the implementations and the estimators, not whether the dynamics converges to the averages of the field theory ([CL-FTS theory, Section 3](ComplexLangevinSimulation.md#3-correct-convergence)).

**Estimators on the same trajectory.** Different estimators of the same structure function can be compared on identical samples. Their difference in noise is then the difference between the estimators alone, not between the sampling dynamics.

Conditions:
- the polyampholyte above on a $`32^3`$ grid, in CL-FTS on CUDA ($`\delta\tau N = 0.5`$);
- four independent runs of 17000 steps after equilibration, with every step sampled;
- the noise is the standard deviation of 1000-step block means (68 blocks).

Compared estimators:
- **Density:** the mixed form of Section 3, the field-field form of Section 2 for the mode with $`\lambda_k \neq 0`$, and $`V\sqrt{\bar{N}} \langle \tilde\phi \tilde\phi \rangle`$ alone.
- **Charge:** the mixed form of Section 6 and the field-field form of $`\psi`$.

Results on the shells $`k R_0 = 2.2`$-$`7.5`$:

| Structure function | Means of the two correct forms | Variance, field-field / mixed |
|---|---|---|
| Density | agree within 1.4 % (at most 2.2 combined standard errors) | 0.70-0.81 at the three lowest shells, 1.5-13 at larger $`k`$ |
| Charge | agree within 2.7 %; the field-field values lie 1.5 % lower on average (at most 2.6 combined standard errors) | 0.002-0.05 |
| Density, $`\langle \phi \phi \rangle`$ alone | negative (e.g. −0.32 against 0.45 at the lowest shell): the intrachain correlation is missing | — |

**Which form has less noise.**
- **Density:** the mixed form is better overall.
- **Charge:** in this strongly screened polyampholyte, the field-field form has 20-600 times less variance. Screening suppresses the fluctuations of $`\psi`$ about its Gaussian part, so $`V\sqrt{\bar{N}}\langle\tilde\psi\tilde\psi\rangle + E/k^2`$ fluctuates little. The mixed form, in contrast, multiplies $`\psi`$ by the fluctuating charge density.

## 8. References

1. Müller, M. & Schmid, F. "Incorporating Fluctuations and Dynamics in Self-Consistent Field Theories for Polymer Blends." In *Advanced Computer Simulation Approaches for Soft Matter Sciences II* (Springer, Berlin, 2005).

2. Spencer, R. K. W. & Matsen, M. W. "Critical Point of Symmetric Binary Homopolymer Blends." *Macromolecules* **49**, 6116 (2016).

3. Wessén, J., Das, S., Pal, T. & Chan, H. S. "Analytical Formulation and Field-Theoretic Simulation of Sequence-Specific Phase Separation of Protein-Like Heteropolymers with Short- and Long-Spatial-Range Interactions." *J. Phys. Chem. B* **126**, 9222 (2022). Code: <https://github.com/jwessen/BioFTS>.

4. Fredrickson, G. H. & Delaney, K. T. *Field-Theoretic Simulations in Soft Matter and Quantum Fluids* (Oxford University Press, Oxford, 2023), International Series of Monographs on Physics 173.

5. Matsen, M. W., Ramírez, J., Willis, J. D. & Pina, P. D. "Accessing the universal phase behavior of block copolymer melts with complex-Langevin field-theoretic simulations." *J. Chem. Phys.* **164**, 014905 (2026).
