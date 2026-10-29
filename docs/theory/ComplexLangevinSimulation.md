# Complex Langevin Field-Theoretic Simulation (CL-FTS)

This document describes complex Langevin field-theoretic simulation (CL-FTS), which samples every auxiliary field without the partial saddle-point approximation of [L-FTS](PartialSaddlePointApproximation.md). The Langevin integrators, the observables and output files, the structure functions and the renormalization that CL-FTS shares with L-FTS are described in [LangevinFieldTheoreticSimulation.md](LangevinFieldTheoreticSimulation.md).

## Table of Contents

1. [Overview](#1-overview)
2. [Complex Langevin Equations](#2-complex-langevin-equations)
3. [Correct Convergence](#3-correct-convergence)
4. [Hot Spots and Dynamical Stabilization](#4-hot-spots-and-dynamical-stabilization)
5. [Using the CLFTS Class](#5-using-the-clfts-class)
6. [Charged Systems](#6-charged-systems)
7. [Chemical Potential and Osmotic Pressure](#7-chemical-potential-and-osmotic-pressure)
8. [Gibbs Ensemble](#8-gibbs-ensemble)
9. [References](#9-references)

---

## 1. Overview

CL-FTS drops the partial saddle-point approximation: every auxiliary field fluctuates, including the imaginary fields that L-FTS keeps at their saddle points. The fields then become complex, and so do the concentrations, the partition functions $`Q_p`$ and $`H`$. The method assumes that the stationary averages of this complex dynamics are the averages of the full field theory (References 1 and 3). `CLFTS` uses the compressible model of [PolymerFieldTheory.md](PolymerFieldTheory.md#3-compressible-model) and needs a finite `zeta_n` (default 100).

---

## 2. Complex Langevin Equations

As in [PolymerFieldTheory.md](PolymerFieldTheory.md#73-sign-convention-for-imaginary-fields), $`H`$ is minimized with respect to the real fields and maximized with respect to the imaginary fields at the saddle point; for AB, $`\Omega_-`$ is real and $`\Omega_+`$ imaginary. An imaginary field is integrated along the imaginary axis, $`\Omega_j = i w_j`$ with real $`w_j`$. Written for $`\Omega_j`$, the real Langevin step of $`w_j`$ reverses the sign of the force and makes the noise imaginary:

```math
\Omega_i(\mathbf{r}, \tau + \delta\tau) = \Omega_i(\mathbf{r}, \tau) - \lambda_i \frac{\delta H}{\delta \Omega_i} \delta\tau N + \sigma_i \mathcal{N}_i(\mathbf{r}, \tau) \qquad \text{(real fields)}
```

```math
\Omega_j(\mathbf{r}, \tau + \delta\tau) = \Omega_j(\mathbf{r}, \tau) + \lambda_j \frac{\delta H}{\delta \Omega_j} \delta\tau N + i \sigma_j \mathcal{N}_j(\mathbf{r}, \tau) \qquad \text{(imaginary fields)}
```

The derivatives are those of [PolymerFieldTheory.md](PolymerFieldTheory.md#32-effective-hamiltonian), evaluated with the complex concentrations (smeared, when smearing is used). $`\mathcal{N}_i`$ is real Gaussian noise of unit variance, and $`\sigma_i`$ is that of [Langevin dynamics, Section 2.2](LangevinFieldTheoreticSimulation.md#22-discretized-equation-per-chain-unit). The mobility $`\lambda_i`$ (`dt_scaling` in `clfts.py`) follows [Langevin dynamics, Section 2.1](LangevinFieldTheoreticSimulation.md#21-continuous-langevin-equation) with the eigenvalues of $`U = X + \zeta N \, \mathbf{e}\mathbf{e}^T`$. A field with a zero eigenvalue does not enter $`H`$; it has $`\lambda_i = 0`$ and stays fixed.

LM ([Langevin dynamics, Section 2.4](LangevinFieldTheoreticSimulation.md#24-leimkuhler-matthews-method)) is the default integrator and averages the noise over two steps; `langevin.integrator: "euler"` selects [Langevin dynamics, Section 2.3](LangevinFieldTheoreticSimulation.md#23-euler-method) and `"etd"` the exponential time differencing of Section 2.1. For an AB diblock, these updates are Eqs. (13)-(16) of Reference 6. The time step is fixed. Reference 6 ran LM with $`\delta\tau N = 2`$ in CL-FTS and $`\delta\tau N = 8`$ in L-FTS.

### 2.1 Exponential Time Differencing

LM steps diverge when the time step exceeds about twice the inverse of the fastest relaxation rate of the fields. The `etd` integrator removes this limit for the linear response of the homogeneous state: it integrates that part of the forces exactly and keeps the rest explicit (exponential time differencing, Reference 7).

**Linear response.** Write the update of Section 2 as $`\Omega_i \to \Omega_i - \lambda_i F_i \, \delta\tau N + n_i \sigma \mathcal{N}_i`$, with $`F_i = \pm\delta H/\delta\Omega_i`$ and the noise direction $`n_i`$ (1 for real fields, $`i`$ for imaginary fields). Before the run, `CLFTS` measures the Jacobian $`J_{ij}(\mathbf{k}) = \partial \hat F_i(\mathbf{k}) / \partial \hat\Omega_j(\mathbf{k})`$ at the homogeneous state $`\Omega = 0`$ (a uniform shift of the fields does not change the concentrations of canonical chains; grand-canonical chains have the density $`e^{\beta\mu_p}\alpha_p`$ there rather than that of the saddle point, which changes only the linear part that the step integrates exactly).
- A kick $`i\epsilon`$ at one grid point of field $`j`$ has the Fourier amplitude $`i\epsilon`$ at every wavevector. In a homogeneous periodic box the concentrations do not couple different wavevectors at linear order, so the imaginary part of the change of the forces, divided by $`\epsilon`$, gives the column $`j`$ of $`J(\mathbf{k})`$ at all $`\mathbf{k}`$ from one computation of the concentrations (a complex-step derivative, Reference 8, with an error of order $`\epsilon^2`$).
- The forces of the unkicked state are subtracted. Transforms of real data, for example FFTs of sizes with large prime factors, leave imaginary rounding errors of order $`10^{-15}`$, which a division by $`\epsilon`$ alone would amplify. With $`\epsilon = 10^{-6}`$ the error of $`J`$ stays below about $`10^{-7}`$ of its size.
- $`J`$ contains the quadratic terms of $`H`$, the response of the chains of any architecture and the smearing, all as the solver computes them, but not the dynamical stabilization of Section 4. With `electrostatics`, $`\psi`$ is one more field, with the force $`-\delta H/\delta\psi`$ (Section 6).

**Step.** In the variables $`x_i = \Omega_i / n_i`$ the noise is real, the force is $`D^{-1} F`$ and its linear part is $`K x`$, with $`D = \mathrm{diag}(n_i)`$ and $`K = D^{-1} J D`$. With the mobilities $`\Lambda = \mathrm{diag}(\lambda_i)`$ and $`K' = \Lambda^{1/2} K \Lambda^{1/2}`$, each wavevector takes the step

```math
\hat x(\tau + \delta\tau) = \hat x(\tau) - \delta\tau N \, \Lambda^{1/2} \varphi_1(K' \delta\tau N) \Lambda^{1/2} D^{-1} \hat F + \Lambda^{1/2} \varphi_1(K' \delta\tau N)^{1/2} \, \frac{\hat\xi + \hat\xi'}{2}, \qquad \varphi_1(z) = \frac{1 - e^{-z}}{z}
```

It is the LM step with the effective mobility $`\delta\tau N \, \Lambda^{1/2} \varphi_1(K'\delta\tau N) \Lambda^{1/2}`$ in place of $`\delta\tau N \, \Lambda`$, and $`\hat\xi`$, $`\hat\xi'`$ are the noises of the current and the previous step, as in LM. The full force is used in every step; $`K`$ only sets the mobility, which is constant for a given time step.
- For a linear force the deterministic part is the exact flow $`e^{-\Lambda K \delta\tau N} \hat x`$, and the stationary covariance is $`\langle \hat x \hat x^T \rangle \propto K^{-1}`$, the Gaussian statistics of $`e^{-H}`$, for any time step. The step of $`\psi`$ in Section 6 is the same construction for one field with $`K = (k^2 + \kappa^2)/E`$.
- As $`\delta\tau \to 0`$, or for $`J = 0`$, it becomes the LM step, so the stationary averages tend to the same limit. A constant mobility does not change the stationary distribution of the continuous dynamics; at a finite time step the errors are, as for LM, of second order in $`\delta\tau`$, because the noise is averaged over two steps as in LM.
- $`K'`$ is complex symmetric; the integrator rejects a $`J`$ that is not. The forces of continuous chains are the gradient of the discretized $`H`$ only up to the contour error, so the measured $`J`$ is symmetric only to $`O(\Delta s^2)`$ (`rk2` and `rqm4` alike), notably where the segment lengths differ; CLFTS passes its symmetric part, and rejects a relative asymmetry above $`10^{-2}`$, which only a far too coarse contour step gives. Real and imaginary fields couple where the chains respond to both, for example in a diblock with unequal blocks, and $`K`$ is then complex rather than real. The exact integral of the linear dynamics over one step then mixes the noises the fields received during the step, so the noise of a real field gets an imaginary part at that wavevector. This is the time integration of the same complex Langevin equations, not a kernel that changes them. Any $`B`$ with $`B B^T = \varphi_1`$ gives the right covariance. The integrator takes the Cholesky factorization without complex conjugation, or the symmetric square root where a small pivot would make the Cholesky factor much larger than $`\varphi_1^{1/2}`$.
- Where $`K'`$ has eigenvalues with a negative real part, the homogeneous state is unstable at that wavevector, as near the order-disorder transition and in ordered phases. These eigenvalues are set to zero: their directions take the LM step, while the other directions of the wavevector keep the exact step. Along the unstable directions the force is held by the nonlinear response of the fields, which the homogeneous state does not give, so the time step there is limited as for LM.

**Use.** `etd` helps where the linear part of the forces limits the time step: in disordered states, where the fields fluctuate about the homogeneous state, and where fields relax at very different rates, for example at small $`\chi N`$ (the quadratic term $`2/\chi N`$ of the exchange field), for the pressure field at large $`\zeta N`$, and for $`\psi`$. Near the order-disorder transition and in ordered phases, the unstable directions limit the time step, so time-step convergence has to be checked there; at strong coupling too. In a 1D symmetric diblock (continuous chains, 32 grid points, $`L = 4 R_0`$) at $`\chi N = 11`$, $`\zeta N = 100`$ and $`\bar N = 10^4`$, where two wavevectors are unstable, `etd` agrees with LM at $`\delta\tau N = 2`$ and 4 ($`\langle H \rangle = 2.82`$, largest drift 0.6). At $`\delta\tau N = 8`$, beyond the LM limit of 4.9 of this system, two of three runs of 400 steps went to a wrong state ($`\langle H \rangle = -30`$ and $`-15`$) with drifts up to 110; the drift records (Section 3.1) show it. In an ordered lamellar phase (3D, $`32^3`$ grid points, $`L = 4.21 R_0`$, discrete chains with $`N = 90`$, $`\chi N = 15`$, $`\zeta N = 100`$, $`\bar N = 10^5`$, Gaussian smearing $`a = 0.02 R_0`$, $`\alpha_{ds} = 0.01`$, 40000 steps on an A10 GPU) `etd` gave no larger time step than LM: both ran at $`\delta\tau N = 0.5`$, 1 and 2 and diverged within 30 steps at 4, and a step of `etd` costs as much as one of LM (17 ms at $`32^3`$ and 99 ms at $`64^3`$ grid points with the same spacing). There the time step is limited by the nonlinear dynamics around the ordered structure, not by the linear part, so LM is the better choice. The drift and noise matrices are Fourier multipliers of `PropagatorSolver.fourier` (in GPU memory with CUDA), and a step applies both in one call. The integrator needs a periodic box and is not available in L-FTS. Each simulation keeps its own copy of the integrator, which holds the response of its system.

**Tests.** `TestClftsEtd` checks, with a synthetic linear force that couples a real field to two imaginary fields, the exact flow and the exact stationary covariance of every mode at time steps 8 and 300 times the stability limit of LM steps; that a zero Jacobian gives the LM step; that at an unstable wavevector only the unstable direction loses its exact step; the noise factor where the Cholesky pivot is small; the rejection of an asymmetric Jacobian; that the measured Jacobian equals central differences of the forces for a charged diblock with counterions (with and without screening) and on a 67-point grid; the Gaussian statistics of $`\psi`$ in a run at a time step beyond its LM limit; a charged lamella with strong screening that diverged when unstable wavevectors took the LM step for all fields; and that continued runs reproduce uninterrupted ones bit for bit. `TestStructureFunctionRPA` runs the disordered AB diblock of its CL-FTS case with `etd` at $`\delta\tau N = 16`$, where LM steps diverge.

**Validation.** In that diblock (continuous chains, 1D, 32 grid points, $`\chi N = 6`$, $`\bar N = 10^6`$, cpu-fftw, 20000 steps), LM diverges at $`\delta\tau N = 4`$ with $`\zeta N = 100`$, while `etd` gives $`S_{AA}`$ within 1 % of the RPA (mean over the first eight modes) at $`\delta\tau N`$ from 1 to 64, with $`\zeta N = 100`$ and 1000. The order of the time-step error is measured by `tests/validation/validate_etd_order.py`:

overdamped Langevin dynamics of independent trajectories, the grid points of one field, advanced by the integrators of the package, with the stationary density $`e^{-U}`$ as the reference. For $`U = x^2/2 + x^4/4`$ ($`10^6`$ trajectories over 200 time units, standard errors from 20 groups of trajectories), the error of $`\langle x^2 \rangle = 0.4679`$ in units of $`10^{-4}`$ is:

| Integrator | $`\delta\tau = 0.025`$ | 0.05 | 0.1 | 0.2 |
|---|---:|---:|---:|---:|
| Euler | 68.5 ± 0.3 | 142.0 ± 0.3 | 313.0 ± 0.2 | diverges |
| ETD with fresh noise | 28.5 ± 0.2 | 59.2 ± 0.2 | 133.0 ± 0.3 | diverges |
| LM | −0.4 ± 0.3 | −1.2 ± 0.2 | −6.1 ± 0.3 | diverges |
| `etd` | −0.2 ± 0.3 | −1.3 ± 0.3 | −5.8 ± 0.2 | −20.6 ± 0.3 |

The errors of Euler and of ETD with fresh noise double with $`\delta\tau`$ (first order), those of LM and `etd` grow by factors of about four (second order) and are 20 to 50 times smaller; "diverges" means that some trajectories run away in the cubic force. With the stiff force $`F = 20x + 2\tanh x`$ ($`2 \times 10^5`$ trajectories, 400 time units), LM diverges from $`\delta\tau = 0.1`$ on, while the error of `etd` stays at the statistical level, $`5 \times 10^{-6}`$ of $`\langle x^2 \rangle = 0.046`$, up to $`\delta\tau = 0.4`$; ETD with fresh noise is off by $`1.7 \times 10^{-4}`$ there.

---

## 3. Correct Convergence

A stable, bounded trajectory does not guarantee that the stationary averages of the complex dynamics are those of the field theory (Section 1): complex Langevin dynamics can converge to wrong averages (Reference 2). The justification of the method requires the distribution of the drift to fall off at least exponentially at large magnitude (Reference 4). Large drifts occur, for example, near zeros of $`Q_p`$, where the concentrations can diverge.

### 3.1 Drift Records

`drift_max_step_history` ([Observables](LangevinFieldTheoreticSimulation.md#3-observables-and-output-files)) records the largest drift of every step for each auxiliary field, and `psi_drift_max_step_history` that of the electrostatic potential (Section 6). The end of a run prints a summary of their distribution, and `scripts/analysis/drift_tail.py` analyzes the records of a run ([Tools](../references/Tools.md#6-cl-fts-drift-distribution)).

Near a simple zero of a partition function the drift grows as the inverse distance $`r`$ from the zero in the complex plane of its field amplitude. If the samples reach the zero with a finite density, the fraction of steps with a drift above $`u`$ falls off as $`S(u) \propto r^2 \propto u^{-2}`$ far out; a density that vanishes at the zero gives a steeper power law. Either is a power-law tail instead of an exponential one. In one-pole toy models with such a zero (References 4 and 10), run in their wrong regime with $`10^7`$ steps or more, the log-log slopes of $`S`$ between 3 and 100 times the median drift were about $`-2`$ to $`-3`$, and approached $`-2`$ only in the last decade sampled. Near the boundary of their correct regime about $`10^{-4}`$ of the steps lay beyond ten times the median, and too few beyond that for a slope: a run of $`10^7`$ steps cannot classify such a case. A finite run cannot show that a tail is exponential, so results in a new regime need an independent reference.

**Partition ratios.** A canonical single-site species, a discrete chain of one segment (a solvent or counterions) or a point ion, has $`Q_s = \langle e^{-u_s} \rangle`$, the grid mean of the Boltzmann weight of its one site, and its term $`-n_s \ln Q_s`$ of $`H`$ has a pole where $`Q_s = 0`$. There the force $`n_s e^{-u_s}/Q_s`$ of the species grows without bound. `partition_ratio_step_history` records at every step the ratio $`\vert Q_s \vert / \langle \vert e^{-u_s} \vert \rangle`$ of each such species, with the columns named by `partition_ratio_species_json`. The ratio is 1 for real fields and approaches 0 as the complex weights cancel in the mean. The end of a run prints the smallest ratio and the fractions of steps below 0.1 and 0.01, and `drift_tail.py` summarizes the records. Grand-canonical species enter $`H`$ as $`-e^{\beta\mu} Q_s`$, without a pole, and are not recorded.

**Checks that do not show correctness.** Imaginary parts of real observables that average to zero, and Schwinger-Dyson identities (Section 7) that hold within the errors, are necessary but not sufficient. In the one-pole model of Reference 10 at $`\beta = 1.6`$ (`tests/scripts/TestDriftTail.py`), complex Langevin dynamics misses $`\mathrm{Re}\langle z^2 \rangle`$ by 438 standard errors in a long run (8000 walkers), while $`\mathrm{Im}\langle z^2 \rangle = -0.0007 \pm 0.0015`$; in this model the imaginary part vanishes by the symmetry $`z \to -\bar z`$ of the dynamics, correct or not. In the shorter run of the test (1000 walkers, 57 standard errors off), the Schwinger-Dyson identities of $`z`$ and $`z^2`$, $`\langle K \rangle = 0`$ and $`\langle 2 + 2 z K \rangle = 0`$ with the drift $`K`$, hold within 1.7 standard errors.

### 3.2 Test Against Exact Averages

A model small enough to be summed exactly shows how the tail and the bias are related. Two cells of a box of length $`R_0`$ with reflecting walls hold nine beads: a chain of four beads with valence $`-1`$ each and its first bead fixed in the first cell, two counterions of valence $`+1`$, one ion of valence $`+3`$, one of valence $`-1`$ and one neutral bead. Here $`C = 9/4`$ ($`\bar{N} \approx 5`$), all smearing radii are $`0.1 R_0`$ and $`\chi N = 2`$ between the chain and the neutral bead. The averages follow exactly from the $`2^8`$ particle configurations of the same finite action. Its complex Langevin dynamics was integrated for $`\zeta N`$ from 0.25 to 4 and $`E`$ from 0 to 100 with a reduced implementation of the equations of Section 2; its densities and drifts agree with those of the native solver within $`10^{-13}`$. Each setting has 2048 independent trajectories, Heun steps of $`\delta\tau N = 6.25 \times 10^{-4}`$ and 100 time units after 20 of equilibration, and the largest drift of every step is recorded.

![Distribution of the largest drift and the bias of complex Langevin averages in a model with exact averages](../figures/cl_drift_tail_exact.png)

The comparison covers the counterion, polymer and trivalent-ion densities in the first cell, the mean height of the chain and the mean of one field amplitude; max $`\vert z \vert`$ is the largest of $`\vert \mathrm{CL} - \mathrm{exact} \vert`$ over them, in standard errors. $`S(10\times)`$ is the fraction of steps with a drift above ten times its median. The dots in panel (a) mark it on each curve; panel (b) plots it for every setting against the bias, with the settings of panel (a) in their colors, and the dashed line at $`3 \times 10^{-5}`$ is the same boundary in both panels.

| $`\zeta N`$ | $`E`$ | max $`\vert z \vert`$ | $`S(10\times)`$ | CL averages |
|---:|---:|---:|---:|---|
| 0.25 | 0 | 0.66 | $`1.1 \times 10^{-6}`$ | agree |
| 0.5 | 0 | 0.35 | $`2.7 \times 10^{-6}`$ | agree |
| 1 | 0 | 0.55 | $`1.1 \times 10^{-5}`$ | agree |
| 2 | 0 | 2.98 | $`3.6 \times 10^{-5}`$ | differ |
| 4 | 0 | 6.47 | $`1.7 \times 10^{-4}`$ | differ |
| 0.5 | 4 | 0.36 | $`7.4 \times 10^{-6}`$ | agree |
| 0.5 | 10 | 1.43 | $`2.9 \times 10^{-5}`$ | agree |
| 0.25 | 10 | 1.98 | $`1.9 \times 10^{-5}`$ | agree |
| 0.5 | 25 | 5.97 | $`1.2 \times 10^{-4}`$ | differ |
| 0.5 | 100 | 19.80 | $`3.3 \times 10^{-4}`$ | differ |
| 4 | 100 | 27.11 | $`6.5 \times 10^{-4}`$ | differ |

- **Tail:** every setting has the $`u^{-2}`$ tail of a partition-function zero beyond about ten times the median drift (panel a). Strictly, the criterion of Reference 4 is violated in all of them.
- **Bias:** it grows with the weight of the tail (panel b). Where at most $`3 \times 10^{-5}`$ of the steps have a drift above ten times the median, the averages agree with the exact ones within two standard errors, although the interactions shift them by up to about 120 standard errors. From $`3.6 \times 10^{-5}`$ on they differ, by up to 27 standard errors.
- **Regimes:** CL-FTS converges within the statistical errors without charges at $`\zeta N \le 1`$, and with charges up to $`E = 10`$ at $`\zeta N = 0.25`$–$`0.5`$. It fails at $`\zeta N = 4`$ for every tested $`E`$, already without charges, and at $`E \ge 25`$ for every tested $`\zeta N`$. The LM integrator and a half time step give the same agreement.

The thresholds belong to this model, whose $`\bar{N}`$ is very small; they are not criteria for production runs. In a production run, a heavier power-law tail of the drift means a larger bias, and only an independent reference shows whether the bias matters.

---

## 4. Hot Spots and Dynamical Stabilization

The imaginary part of a real field can grow without bound at a few grid points ("hot spots"), most often at small $`\bar{N}`$ and with contact interactions (Reference 5). `CLFTS` stops with a `ValidationError` in either of two cases:

- the fields, the concentrations or $`H`$ become non-finite;
- the spatial mean of $`\sum_i \phi_i - 1`$ exceeds $`5 \times 10^{-8}`$ in magnitude, which is usually the first sign of a hot spot.

Two remedies are available:

- **Dynamical stabilization** (Reference 5, `dynamic_stabilization`): the force on each real field gets the term $`i \alpha_{ds} \mathrm{Im}[\Omega_i]`$, which damps its imaginary part.
  - It biases the sampling toward stronger segregation: a slightly lower $`(\chi N)_{\mathrm{ODT}}`$ and a higher $`S(k^*)`$.
  - Reference 5 used $`\alpha_{ds} = 0.001`$ at $`\bar{N} = 10^6`$-$`10^7`$ and 0.01-0.02 at $`\bar{N} = 10^5`$.
  - Reference 6 found structure functions more sensitive to $`\alpha_{ds}`$ than the order-disorder transition, and computed them without stabilization.
- **Smearing** ([`smearing`](../references/Parameters.md#smearing), [model](PolymerFieldTheory.md#4-smeared-interactions)): finite-range interactions damp the high-$`k`$ fluctuations from which hot spots grow (Reference 6).

[DYNAMICAL_STABILIZATION.md](../../examples/block_copolymer/clfts/DYNAMICAL_STABILIZATION.md) and [SMEARING.md](../../examples/block_copolymer/clfts/SMEARING.md) in `examples/block_copolymer/clfts/` give recommended values.

---

## 5. Using the CLFTS Class

`CLFTS(params, random_seed)` takes the parameters of the [L-FTS example](PartialSaddlePointApproximation.md#4-using-the-lfts-class) without `saddle` and `compressor`. It adds `zeta_n`, `dynamic_stabilization` and, optionally, `smearing` and `electrostatics` (see [CL-FTS Parameters](../references/Parameters.md#cl-fts-parameters)). [examples/block_copolymer/clfts/Lamella.py](../../examples/block_copolymer/clfts/Lamella.py), [examples/charged/clfts/ChargedDiblock.py](../../examples/charged/clfts/ChargedDiblock.py) and [examples/charged/clfts/SaltReservoir.py](../../examples/charged/clfts/SaltReservoir.py) are complete scripts. For a comparison with experiments or other simulations, set `chi_n` and the bare segment lengths as in [Renormalization and calibration](LangevinFieldTheoreticSimulation.md#5-renormalization-and-calibration); CL-FTS needs both corrections, $`z_\infty`$ and $`a_b/a`$.

---

## 6. Charged Systems

[Polymer Field Theory, Section 5](PolymerFieldTheory.md#5-electrostatic-interactions) defines the charged model: the valences $`z_i`$ per segment (`electrostatics`), the smeared charge density $`c`$, the coupling $`E`$, the electrostatic potential $`\psi`$ and, optionally, the screening $`\kappa`$ of a background electrolyte (`screening_length` $`= 1/\kappa`$; $`\kappa = 0`$ without it), with

```math
\frac{\beta H_{el}}{C V / R_0^3} = \frac{1}{2 E V} \int \psi \, (\nabla^2 - \kappa^2) \, \psi \, d\mathbf{r}, \qquad \frac{\delta H}{\delta \psi} = \frac{(\nabla^2 - \kappa^2) \psi}{E} + c
```

CL-FTS samples $`\psi`$ together with the auxiliary fields and stores it on the rotated contour, like the other imaginary-type fields. Without screening the uniform mode of $`\psi`$ is zero and the system must be electroneutral. With screening the uniform mode fluctuates, and the explicit species may carry a net charge. A charged grand-canonical species also makes the uniform mode fluctuate: it then is the Donnan potential that keeps the system neutral on average ([Polymer Field Theory, Section 6](PolymerFieldTheory.md#6-grand-canonical-species)).

**Update of ψ.** $`\psi`$ moves like an imaginary field (Section 2) with the mobility $`s`$ (`psi_dt_scaling`). Its linear part $`(\nabla^2 - \kappa^2) \psi / E`$ relaxes at the rate $`\gamma_k = s (k^2 + \kappa^2)/E`$, which is stiff at large $`k`$. This part is therefore integrated exactly for each wavevector, by exponential time differencing. Like the Leimkuhler-Matthews integrator of the auxiliary fields ([Langevin dynamics, Section 2.4](LangevinFieldTheoreticSimulation.md#24-leimkuhler-matthews-method)), $`\psi`$ takes the mean of the current and the previous noise. It follows the noise scheme the integrator declares (`averaged_noise`, see [Extensions](../internals/Extensions.md#langevin-integrators)): averaged with the Leimkuhler-Matthews integrator, fresh with the Euler integrator. A custom integrator must declare it, even one that declares `linear_response` and so receives $`\psi`$ (below):

```math
\hat\psi_{\mathbf{k}}(\tau + \delta\tau) = a_k \hat\psi_{\mathbf{k}}(\tau) + (1 - a_k) \frac{E}{k^2 + \kappa^2} \hat c_{\mathbf{k}} + i \sqrt{\frac{1 - a_k}{4 ((k^2 + \kappa^2)/E) \, \delta\tau N}} \, \big(\hat{\xi}_{\mathbf{k}} + \hat{\xi}'_{\mathbf{k}}\big), \qquad a_k = e^{-\gamma_k \delta\tau N}
```

In this step:

- $`\hat\xi_{\mathbf{k}}`$ is the Fourier transform of real Gaussian noise with the standard deviation $`\sigma`$ of [Langevin dynamics, Section 2.2](LangevinFieldTheoreticSimulation.md#22-discretized-equation-per-chain-unit), drawn from a separate random-number stream, the jumped stream of the auxiliary fields (`PCG64.jumped()`). The noise of the auxiliary fields is therefore the same as without charges, and runs with neighboring seeds do not share a stream. $`\hat\xi'_{\mathbf{k}}`$ is the noise of the previous step (zero at the first step); checkpoints store it as `psi_noise_prev`.
- $`\hat c_{\mathbf{k}}`$ is the charge density of the state before the step, as for the forces of the auxiliary fields.
- The step is the Leimkuhler-Matthews step of $`\psi`$ with the time step $`(1 - a_k)/\gamma_k < \delta\tau N`$ of each mode, so it is stable for any $`\delta\tau`$. As $`\gamma_k \to 0`$ it becomes the Leimkuhler-Matthews step with mobility $`s`$.
- Without screening the uniform mode is set to zero after the step, unless a charged species is grand canonical. The uniform mode then takes the step with $`\gamma_0 = 0`$ and relaxes only through the charge of the grand-canonical species, slowly where they are dilute; `psi_dt_scaling` speeds it up.

The Leimkuhler-Matthews step samples a Gaussian distribution exactly for any stable time step and any mobilities. For Gaussian fluctuations, the statistics of $`\psi`$ and of the auxiliary fields therefore do not depend on $`\delta\tau`$, although the charge density is explicit.

With the Euler integrator, $`\psi`$ takes fresh noise, $`i \sqrt{(1 - a_k^2)/(2 ((k^2 + \kappa^2)/E)\, \delta\tau N)}\, \hat\xi_{\mathbf{k}}`$. The linear modes alone then keep their exact variance, but the explicit charge density changes it at first order in $`\gamma_k \delta\tau N`$. Where the charge density follows $`\psi`$ as $`\hat c = -R \hat\psi`$, the variance is too large by the factor

```math
\frac{1 + \epsilon}{1 + \frac{2 a_k}{1 + a_k} \epsilon - \frac{1 - a_k}{1 + a_k} \epsilon^2} \approx 1 + \epsilon \tanh\frac{\gamma_k \delta\tau N}{2}, \qquad \epsilon = \frac{R E}{k^2 + \kappa^2}
```

Before the Leimkuhler-Matthews noise was introduced, this was the step of every integrator. In the polyampholyte of `TestChargedStructureFunctionRPA` ($`E = 300`$, $`\delta\tau N = 1`$) it put the charge structure functions from $`\psi`$ about 2 % above the RPA; with the averaged noise they agree within 0.5 % ([Structure Function Estimators, Section 6](StructureFunctionEstimators.md#6-charge-structure-function)).

**With the `etd` integrator.** The integrator of Section 2.1 advances $`\psi`$ together with the auxiliary fields, as the last field with the force $`-\delta H/\delta\psi`$, its mobility $`s`$ and its noise stream. Its Jacobian then contains the screening of $`\psi`$ by the charges and the coupling of $`\psi`$ to the auxiliary fields, which the step above keeps explicit, so the Gaussian statistics of all fields are exact for any time step. Without screening the uniform mode of $`\psi`$ is set to zero after each step, unless a charged species is grand canonical.

**Observables.**
- **H:** the recorded $`H`$ includes $`H_{el}`$.
- **Structure functions:** The estimator of [Structure functions, Section 4.2](LangevinFieldTheoreticSimulation.md#42-estimator) gives $`S_{ij}`$ of all types, counterions included. For composition modes without an auxiliary field, such as two types that differ only in charge, $`\psi`$ completes the estimator ([Structure Function Estimators](StructureFunctionEstimators.md#5-modes-without-an-auxiliary-field)). The charge structure function is $`S_{zz} = \sum_{ij} z_i z_j S_{ij}`$. The files also hold `charge_structure_function`, the structure function of the smeared charge density. It combines two estimators that use $`\psi`$ with the weight of least variance at each wavevector, and it is much less noisy where screening is strong ([Section 6](StructureFunctionEstimators.md#6-charge-structure-function)).
- **RPA:** the RPA of [Structure functions, Section 4.4](LangevinFieldTheoreticSimulation.md#44-rpa-reference) adds $`E/(k^2 + \kappa^2)\, z z^T`$ to the interaction matrix.
- **Estimators:** the estimators that use $`\psi`$ ([Structure Function Estimators](StructureFunctionEstimators.md#5-modes-without-an-auxiliary-field)) take $`(k^2 + \kappa^2)/E`$ in place of $`k^2/E`$.

**Tests.**
- `TestClftsElectrostatics` checks:
  - that zero valences, or zero coupling, reproduce CL-FTS without charges bit for bit;
  - that the forces of $`\psi`$ and of the auxiliary fields are the derivatives of $`H`$;
  - that the step of $`\psi`$ without its noise is the exact flow of its force, and that its stationary variance is exact, also with a linear charge response and far beyond the stability limit of Euler steps;
  - that a continued run reproduces an uninterrupted one bit for bit;
  - with screening, that the force of $`\psi`$ is the derivative of $`H`$ along the uniform mode too, that the stationary variance of every mode is exact, the uniform mode included, and that a net charge is accepted and moves the uniform mode only with screening.
- `TestChargedStructureFunctionRPA` compares two systems with the RPA:
  - a charged diblock with counterions;
  - a polyampholyte melt whose charge mode has no auxiliary field.

**Checked range.** The tests compare with the RPA where fluctuations are weak. The comparison with BioFTS ([Structure Function Estimators, Section 7](StructureFunctionEstimators.md#7-validation)) extends to a polyampholyte whose structure function lies up to 28 % above the RPA. BioFTS also uses complex Langevin dynamics, so this agreement checks the implementation, not the convergence of the method (Section 3). Against an exact Monte Carlo simulation of the same lattice model (point ions on the grid sites with the smeared Coulomb pair potential of the box), CL-FTS of point ions agrees in a bulk salt up to $`l_B \kappa = 4.8`$, for salt at uncharged walls up to $`l_B/a = 2`$, and for counterions between charged plates up to the coupling parameter $`\Xi \approx 0.6`$ of Netz; at $`\Xi = 0.85`$ the contact density is 4 % too high even as $`\delta\tau \to 0`$, and at $`\Xi \ge 1.6`$ the profile is qualitatively wrong, so CL-FTS does not reach the strong-coupling regime of counterions. The step of $`\psi`$ has an $`O(\delta\tau)`$ bias that grows with the charge response of the ions: beyond weak coupling these systems needed $`\delta\tau N \le 0.25`$ ([Example validation](../internals/ExampleValidation.md#charged-systems)). In the small model of Section 3.2, whose averages can be summed exactly, CL-FTS agrees with the exact averages up to $`E = 10`$ with weak excluded volume and differs at $`E \ge 25`$; at $`E = 100`$ and $`\zeta N = 4`$ the counterion density is about 3 % off, and a smaller time step and other integrators do not remove the difference. At strong coupling, compare with an independent reference where one exists and inspect the tails of `drift_max_step_history` and `psi_drift_max_step_history` (Section 3.1).

**Closed boxes and fixed charges.** In a box with walls on every side the transforms of $`\psi`$ are the cosine series of `PropagatorSolver.fourier`, which diagonalize $`(k^2 + \kappa^2)/E`$ with no normal derivative of $`\psi`$ at the walls; the step above then integrates each cosine mode exactly, and $`H_{el}`$ is half the grid mean of $`\psi`$ times the backward series of $`-(k^2 + \kappa^2)\psi(\mathbf{k})/E`$. The estimators with $`\psi`$ and the charge structure function need a periodic box, as do the osmotic pressure and `etd`. A fixed charge density $`c_f`$ adds $`\langle \psi c_f \rangle`$ to $`H`$ and $`c_f`$ to $`c`$; held at its grid values when the box dilates, it adds $`-\langle \psi c_f \rangle`$ to the osmotic pressure. `TestConfinedElectrostatics` checks the forces against finite differences of $`H`$ in a closed box with a fixed charge and point ions, that the mean-field limit reaches the Poisson-Boltzmann solution between charged layers (solved independently with `scipy.fft.dct`), bitwise restarts in the closed box, and the pressure with a fixed charge in a periodic box.

**Ions with $`\chi`$.** Point ions have neither volume nor $`\chi`$. Ions that the monomers solvate selectively are molecules of one segment in `polymers` (discrete chains), whose monomer type has a valence in `charges` and its pairs in `chi_n` ([Parameters](../references/Parameters.md#electrostatics)). Like a solvent they take volume and feel the auxiliary fields, and CL-FTS records their partition ratios (Section 3.1).

**Point ions alone.** The model needs a polymer species as the reference chain. A system of point ions alone therefore takes a neutral solvent of one segment (discrete chains, `ds` 1, one block of length 1) that the ions do not feel. Give it `zeta_n: 0` and an empty `chi_n` (`{}`). Its auxiliary field then has a zero eigenvalue and is left out, so the solvent stays an ideal gas of uniform density. $`\psi`$ and the ions take the same steps as with $`\zeta N > 0`$, bit for bit with the `lm` integrator. With $`\zeta N > 0`$ the field of the solvent fluctuates more strongly the fewer solvent molecules a grid cell holds, $`C \Delta V/R_0^3`$ for the cell volume $`\Delta V`$. Below about 0.01 molecules per cell, the weights of the solvent lose their phase coherence and its partition ratio (Section 3.1) falls towards zero. In the box of [CounterionsBetweenPlates.py](../../examples/charged/clfts/CounterionsBetweenPlates.py) with $`\zeta N = 1`$ and $`\delta\tau N = 1`$, runs at 0.024 and 0.015 molecules per cell completed 1000 steps, while runs at 0.0098 and 0.0049 diverged after 327 and 171 steps.

**Limitations.** No random copolymers, no box that mixes periodic and wall axes, and no impenetrable regions: the simulation classes reject a `mask`. L-FTS keeps imaginary-type fields at their saddle point, so it does not have this fluctuating $`\psi`$.

---

## 7. Chemical Potential and Osmotic Pressure

The canonical partition function holds the ideal terms $`\prod_p (V Q_p)^{n_p}/n_p!`$ of the chains and the Gaussian integrals over the fields, normalized by the same integrals without the chains (the Hubbard-Stratonovich transformation). Derivatives of $`\ln Z`$ give operators whose averages over the complex Langevin samples are the chemical potentials and the osmotic pressure. They are recorded with $`H`$ at every sample (`sample_interval`) in the `dH` files: the chemical potentials always, the pressure with `recording.pressure`.

**Grand-canonical species.** With `chemical_potential` in place of the amount of a polymer or a point ion, its term of $`H`$ is $`-e^{\beta\mu} Q`$ per reference chain ([Polymer Field Theory, Section 6](PolymerFieldTheory.md#6-grand-canonical-species)). The numbers of chains and ions then fluctuate: `number_density_history_real` and `_imag` hold, in the columns of the chemical potentials, the number per reference chain of every species, $`\bar\phi_p/\alpha_p`$ or $`N c_s`$ of canonical species and the operator $`e^{\beta\mu} Q`$ of grand-canonical ones. Their averages give the composition in equilibrium with a reservoir of the given chemical potentials. Material conservation is checked for the canonical chains.

**Chemical potential.** $`H`$ depends on the number of chains $`n_p`$ only through $`-n_p \ln Q_p`$, so

```math
\beta\mu_p = \ln\frac{\bar\phi_p}{\alpha_p} - \ln Q_p
```

per chain, in the convention of `SCFT.compute_chemical_potentials` ([Phase Equilibrium](PhaseEquilibrium.md#restoring-chemical-potentials)), which leaves out the constant $`\ln(\rho_0/N)`$ common to all species; at the saddle point the operator equals the SCFT value. `chemical_potential_history_real` and `_imag` hold one row of all species per sample. The average of the real part is the chemical potential; it does not depend on the branch of the complex logarithm. A species without chains has $`-\infty`$. Point ions follow the polymers with $`\beta\mu_s = \ln(N c_s) - \ln Q_s`$ ([Polymer Field Theory, Section 5.5](PolymerFieldTheory.md#55-point-ions)). Without screening $`\psi`$ has no uniform mode, so for charged species only electroneutral combinations of chemical potentials are physical. A grand-canonical species records its given chemical potential.

**Osmotic pressure.** $`\beta\Pi = \partial \ln Z / \partial V`$ at fixed $`n_p`$ (at fixed $`\mu`$ for grand-canonical species) is evaluated with the grid values of the fields held fixed while the box is dilated in its shape, so every wavevector scales as $`k^2 \propto V^{-2/d}`$. Per $`C/R_0^3 = \rho_0/N`$:

```math
\frac{\beta\Pi R_0^3}{C} = \sum_p \frac{\bar\phi_p}{\alpha_p} - \frac{1}{d} \sum_a L_a \frac{\partial h}{\partial L_a} - \sum_i \left\langle \phi_i \left( \frac{\partial \Gamma_i}{\partial \ln V} * (w_i + z_i \psi) \right) \right\rangle - h_\Omega + \frac{n_\Omega \, n_g}{2 C V / R_0^3} + \Pi_\psi
```

The terms are:
- the ideal gas of the chains; a grand-canonical polymer contributes $`e^{\beta\mu_p} Q_p`$, from its term of $`H`$, in place of $`\bar\phi_p/\alpha_p`$;
- the conformations of the chains, from the stress of the solver: $`h = -\sum_p (\bar\phi_p/\alpha_p) \ln Q_p`$ at fixed smeared fields, with $`-e^{\beta\mu_p} Q_p`$ for grand-canonical polymers;
- the smearing functions, which follow the box, with $`\partial\Gamma/\partial \ln V = -(2/d) \, k^2 \, d\Gamma/dk^2`$ and the unsmeared concentrations $`\phi_i`$;
- the field part of $`H`$ per reference chain, $`h_\Omega = \sum_i \langle h_{2,i} \Omega_i^2 + h_{1,i} \Omega_i \rangle + h_{\mathrm{const}}`$ (the quadratic, linear and constant terms of [Polymer Field Theory, Section 3.2](PolymerFieldTheory.md#32-effective-hamiltonian)), which scales with $`V`$;
- the normalization of the Gaussian integral over each of the $`n_\Omega`$ auxiliary fields with a nonzero eigenvalue on $`n_g`$ grid points: it cancels the grid-dependent part of $`h_\Omega`$, which is $`n_g/(2 C V/R_0^3)`$ per field without chains;
- $`\psi`$: its terms of $`H`$ change by $`(1 - 2/d) H_k + H_\kappa`$, the parts of $`H_{el}`$ from $`k^2`$ and from $`\kappa^2`$, and its normalization adds $`(1 - (2/d) \, k^2/(k^2 + \kappa^2))/2`$ per $`C V/R_0^3`$ chains for every mode with $`k^2 + \kappa^2 > 0`$:

```math
\Pi_\psi = -\left[\left(1 - \frac{2}{d}\right) H_k + H_\kappa\right] + \frac{1}{2 C V/R_0^3} \sum_{k^2 + \kappa^2 > 0} \left(1 - \frac{2}{d} \frac{k^2}{k^2 + \kappa^2}\right)
```

- point ions: their ideal gas $`\sum_s N c_s`$ ($`e^{\beta\mu_s} Q_s`$ if grand canonical) and the dilation of their smearing functions, $`-\sum_s \langle c_s(\mathbf{r}) \, ((\partial \Gamma_s/\partial \ln V) * z_s \psi) \rangle`$, as for the monomer types; they have no conformations.

At the homogeneous saddle point the operator is the Flory-Huggins osmotic pressure $`\sum_p \bar\phi_p/\alpha_p + \bar{\boldsymbol\phi}^T X \bar{\boldsymbol\phi}/2 + (\zeta N/2)\big((\sum_i \bar\phi_i)^2 - 1\big)`$ as $`C \to \infty`$. A screened system with the net charge $`q = \sum_i z_i \bar\phi_i`$ per segment has $`\psi_0 = E q/\kappa^2`$ there and adds the mean-field Yukawa energy $`E q^2/(2\kappa^2)`$.

**Grid resolution.** The dilation keeps the number of grid points, so the grid spacing changes with the box. When the smearing is resolved by the grid ($`a \gtrsim \Delta x/2`$), the free energy does not depend on the grid spacing and the operator is the pressure of the continuum model. A model whose interactions end at the grid, such as point charges on a lattice, has a free energy that depends on $`\Delta x`$: the operator then differs from the pressure at fixed grid spacing, which sets the coexistence of two phases on one lattice (in a salt-free coacervate of Lee et al. at one loop, $`-1.87`$ against $`-11.07`$ for the electrostatic term). For such models compare phases on one lattice through the chemical potentials, which hold the box and the grid fixed (Gibbs-Duhem: $`dp = c\,d\mu`$). The same applies to `GibbsEnsemble`, whose boxes of different volume share the number of grid points.

`recording.pressure: True` records `pressure_history_real` and `_imag` in the virial form below. Each sample needs the stress of the solver, which costs about as much as one more computation of the propagators. The pressure needs a periodic box and free chains. With continuous chains it carries the contour error of the stress ([Stress Tensor](StressTensor.md)). With `rk2` this error is of order $`\Delta s^2`$ and grows when one contour step spans more than the smearing length: for the A50C50 coacervate of Danielsen et al. ($`a = 0.082 R_0`$, $`\Delta s = 0.01`$, a step of $`0.1 R_0`$) the `rk2` pressure is 0.05 to 0.07 too low per $`C/R_0^3`$. About a third of it comes from the discretized chains, the rest from the contour quadrature of the stress, which is not the exact derivative of the `rk2` $`\ln Q`$. `rqm4` is accurate to $`10^{-4}`$ there. With `rk2`, `CLFTS` warns when the contour step $`b_i \sqrt{\Delta s}\, R_0`$ of a monomer type exceeds its smearing length, or with no smearing.

**Virial form.** Integration by parts over the $`n_g`$ grid values of an auxiliary field gives the Schwinger-Dyson identity

```math
\left\langle \frac{C V}{R_0^3} \, \overline{\Omega_i \frac{\partial H}{\partial \Omega_i}} \right\rangle = n_g, \qquad \frac{\partial H}{\partial \Omega_i} = 2 h_{2,i} \Omega_i + h_{1,i} + \sum_j A_{ji} \phi_j
```

where the bar is the grid mean and $`\phi_j`$ are the smeared concentrations. Every mode of $`\psi`$ with $`k^2 + \kappa^2 > 0`$ obeys $`\langle (C V/R_0^3) \, \psi(\mathbf{k}) F(-\mathbf{k}) \rangle = n_g^2`$, with $`F = \delta H/\delta \psi`$ and unnormalized transforms. `CLFTS.pressure_virial` adds $`\tfrac{1}{2}[\overline{\Omega_i \, \partial H/\partial \Omega_i} - n_g/(C V/R_0^3)]`$ for every auxiliary field, and the same terms of $`\psi`$ with the weight $`w_k = 1 - (2/d) \, k^2/(k^2 + \kappa^2)`$ of each mode. These terms have zero average. They replace the quadratic field terms and the normalizations, $`-\sum_i \langle h_{2,i} \Omega_i^2 + h_{1,i} \Omega_i \rangle + n_\Omega n_g/(2 C V/R_0^3)`$ and the Gaussian part of $`\Pi_\psi`$, by

```math
\frac{1}{2} \sum_i \overline{\Omega_i \Big( \sum_j A_{ji} \phi_j - h_{1,i} \Big)} + \frac{1}{2 n_g^2} \sum_{k^2 + \kappa^2 > 0} w_k \, \psi(\mathbf{k}) \, c(-\mathbf{k})
```

with the charge density $`c`$ of Section 6: the smeared charges of the chains and the point ions and the fixed charges. This is the virial (scaled-field) form of the operator (Reference 11). `CLFTS.pressure` keeps the dilation operator, the volume derivative at fixed grid values for each state, which `TestClftsThermodynamics` checks state by state.

Both forms have the same average. The Gaussian terms of the dilation operator fluctuate by about $`\sqrt{n_g/2}/(C V/R_0^3)`$ per field and sample, which grows with the number of grid points per chain, and the virial form removes this fluctuation:
- In the charged diblock of `TestClftsThermodynamics` (discrete chains with $`N = 20`$, $`\chi N = 9`$, $`\zeta N = 30`$, $`8 \times 6 \times 7`$ grid points, $`C V/R_0^3 = 60`$, `screening_length` $`0.3 R_0`$), the standard deviation per sample of the real part drops from 0.41 to 0.078. Over 6000 steps at $`\delta\tau N = 0.5`$ the standard error drops from 0.023 to 0.0035.
- In the homopolymer solution of [HomopolymerSolutionBook.py](../../examples/blend/clfts/HomopolymerSolutionBook.py) ($`\zeta N = 1`$), run at $`L = 1.6 R_g`$ with $`16^3`$ grid points, the standard deviation drops from 10.3 to 0.013; at the script's $`L = 3.2 R_g`$ with $`32^3`$ points the dilation form fluctuates by about 4.
- Where the chains respond strongly to a field, as to the pressure field at large $`\zeta N`$, the term $`\Omega_i \sum_j A_{ji} \phi_j`$ fluctuates more and the gain is smaller.
- The imaginary part, whose average is zero, keeps the fluctuation of the uniform mode of each imaginary-type field, whose linear term remains in both forms.

At the homogeneous saddle point all forces vanish, so the virial form gives the Flory-Huggins pressure without subtracting the normalizations. The identities hold for the exact stationary distribution. The $`O(\delta\tau)`$ bias of the integrator, or dynamical stabilization, which is not part of $`H`$, can make the added terms average to a small nonzero value. In the diblock above they averaged to $`0.034 \pm 0.021`$, $`0.038 \pm 0.023`$ and $`0.005 \pm 0.026`$ at $`\delta\tau N`$ = 0.5, 0.2 and 0.1, zero within two standard errors. An identity that holds does not show that complex Langevin dynamics converges to the right averages (Section 3.1).

**Discretization.** Like $`H`$, both averages depend on the grid and the smearing, which regularize the field theory; compare states computed with the same discretization.

**Restarts.** Checkpoints hold the pending samples. A checkpoint without them, written by an earlier version or by a run without `recording.pressure`, gives NaN for those samples in the next file.

**Tests.** `TestClftsThermodynamics` checks:
- for random complex fields on a 3D grid with discrete chains, that the pressure operator without the ideal term and the normalizations equals $`-\partial H/\partial V`$ at fixed grid values and numbers of chains to $`10^{-8}`$: without smearing, with Gaussian and sigmoidal smearing, and with smeared charges with and without screening (`TestClftsPointIons` adds point ions at fixed numbers of ions, `TestClftsGrandCanonical` a grand-canonical homopolymer and ions at fixed chemical potentials);
- the Flory-Huggins chemical potentials and pressure at the homogeneous saddle point, for the dilation operator after removing the normalizations and for the virial form as it is;
- in CL runs of 2000 steps with and without screening, that the terms of the virial form average to zero within four standard errors and that the virial form fluctuates less than half as much;
- that the records equal the operators recomputed from the saved state, and that continued runs reproduce them bit for bit.

**Validation of the pressure.** Where the fluctuations are Gaussian, the average pressure is the Flory-Huggins value plus the one-loop correction $`(1/C)\, \partial \ln Z_1 / \partial V`$ with

```math
\ln Z_1 = -\frac{1}{2} \sum_{\mathbf{k}} \ln \det \left( I + J_0^{-1} R \right)
```

where $`J_0`$ is the quadratic part of the Jacobian of Section 2.1 and $`R = J - J_0`$ the response of the chains, which scales as $`1/V`$ at fixed numbers of chains; the derivative is taken by central differences of the measured Jacobians. Two systems were run with `etd` on cpu-fftw, discarding the first fifth of each run:
- a symmetric AB diblock melt, continuous chains, $`\chi N = 6`$, $`\zeta N = 100`$, 1D with 32 grid points and $`L = 4 R_0`$, $`\delta\tau N = 4`$, 200000 steps;
- the charged diblock of Section 6 with A valence 0.2 and counterions, discrete chains ($`N = 20`$), $`\chi N = 6`$, $`\zeta N = 50`$, $`E = 300`$, `screening_length` $`1/\kappa = 0.5 R_0`$, Gaussian smearing lengths 0.15, 0.15 and 0.2, 2D with $`16^2`$ grid points and $`L = 3 R_0`$, $`\delta\tau N = 2`$, 100000 steps.

| System | $`\bar N`$ | $`\langle \Pi \rangle - \Pi_{FH}`$ | One-loop | Ratio |
|---|---:|---:|---:|---:|
| Diblock | $`10^3`$ | −0.08796 | −0.08772 | 1.003 ± 0.005 |
| Diblock | $`10^4`$ | −0.02790 | −0.02774 | 1.006 ± 0.003 |
| Charged | $`10^3`$ | −0.1306 | −0.1291 | 1.011 ± 0.008 |
| Charged | $`10^4`$ | −0.04107 | −0.04084 | 1.006 ± 0.008 |

The errors are standard errors of eight block means. The normalization terms are larger than the correction itself (0.25 for the diblock at $`\bar N = 10^3`$), so the agreement also checks them.

**Comparison with BioFTS.** BioFTS (jwessen/BioFTS, commit b82f6e01, with the four corrections of its issue 1, among them the noise of its semi-implicit step, without which its $`H`$ and pressure are off by 3 to 4 per reference chain at $`\delta t = 4\times10^{-3}`$) samples the same discrete Gaussian chains with Gaussian smearing in an implicit solvent with the Edwards excluded volume $`v`$ and a Coulomb channel. With $`N`$ beads at the bead density $`\rho_b`$ the mapping is $`R_0 = bN^{1/2}`$, $`\rho_0 = \rho_b`$, $`\bar N = \rho_b^2 N b^6`$, $`\zeta N = v\rho_b N`$, $`E = 4\pi l_B\rho_b N R_0^2`$ and $`c_s = \rho_s/\rho_0`$. At fixed numbers of chains the compressibility term differs from $`(v/2)\rho^2`$ by terms linear in the amounts, so $`\beta\mu_p = \beta\mu_p^{\rm BioFTS} - \ln(\rho_0 b^3/N) - \zeta N`$, $`\beta\mu_s = \beta\mu_s^{\rm BioFTS} - \ln(\rho_0 b^3/N)`$ and $`\beta\Pi R_0^3/C = \beta\Pi^{\rm BioFTS}N/\rho_0 - \zeta N/2`$ (BioFTS has no pressure operator; it was taken as a central difference of its $`H`$ at fixed grid values). On the same fields the two codes give the same partition functions, concentrations, $`H`$, chemical potentials and pressure to $`10^{-12}`$, and Euler-Maruyama steps with the same normal deviates give the same trajectory. In a homopolymer solution and the polyampholyte E5K5E5K5 without salt, with canonical and with grand-canonical point ions ($`N = 20`$, $`\rho_b = 2`$, $`v = 0.05`$, $`l_B = 0.5b`$, $`a = b/\sqrt6`$, $`16^3`$ grid of spacing $`a`$), the chemical potentials of chains and ions, the uptake of grand-canonical salt, $`H`$ and the structure functions agree within their errors (chain chemical potentials within $`4\times10^{-4}`$, for fluctuation corrections of 0.35 to 2.3), and so do the osmotic pressures. Evaluated with `CLFTS.pressure_virial` on the fields of both codes, the polyampholyte with canonical salt gives $`\beta\Pi R_0^3/C = 3.7299 \pm 0.0004`$ in polymerfts (128 runs, `lm` at $`\delta\tau N`$ = 0.1, 0.2 and 0.4 and `etd` at 0.2) and $`3.7298 \pm 0.0004`$ in BioFTS (96 runs at $`\delta\tau N`$ = 0.08 to 0.32); neither code changes with its time step. With grand-canonical ions, without salt and in the homopolymer solution the pressures differ by $`0.0000 \pm 0.0011`$, $`-0.0007 \pm 0.0010`$ and $`-0.00001 \pm 0.00007`$. The dilation operator of the same runs agrees within its errors, which are about ten times larger: its run averages scatter by 0.03, against 0.004 for the virial form.

**Comparison with polycomp.** polycomp ([rotskoff-group/polycomp](https://github.com/rotskoff-group/polycomp), commit 2789769e, with the corrections of its issue 23) samples smeared implicit-solvent models of continuous chains with lengths in units of $`R_g`$. With the total segment density $`\rho_{\rm tot}`$ as $`\rho_0`$ it is the compressible model of [Polymer Field Theory, Section 3](PolymerFieldTheory.md#3-compressible-model):
- lengths $`L = L_{\rm pc}/\sqrt 6`$ and $`a = a_{\rm pc}/\sqrt 6`$, $`C = 6^{d/2}\rho_{\rm tot}`$, $`X + \zeta N\,\mathbf e\mathbf e^T = \rho_{\rm tot}A`$, $`E = 6\rho_{\rm tot}E_{\rm pc}`$ and point-ion concentrations $`c_s/(N\rho_{\rm tot})`$;
- the terms of $`(\zeta N/2)(\sum_i\phi_i-1)^2`$ linear in the fixed amounts shift $`\beta\mu_p`$ by $`-\zeta N\alpha_p-\ln\rho_{\rm tot}`$, $`\beta\mu_s`$ by $`-\ln\rho_{\rm tot}`$ and $`\beta\Pi R_0^3/C`$ by $`-\zeta N/2`$ from polycomp's values divided by $`\rho_{\rm tot}`$.

On the same complex fields the two codes give the same $`Q_p`$, $`Q_s`$, $`\beta\mu`$ and $`H`$ to $`10^{-14}`$. Their mean-field 1D lamellae of a neutral and a charged diblock with point ions agree to $`10^{-6}`$ once polycomp's trapezoidal contour sum is extrapolated in $`\Delta s^2`$. In CL-FTS of a 3D homopolymer solution and a polyelectrolyte solution with point ions ($`24^3`$ points, $`L = 4.5R_0`$, $`a = 0.25R_0`$, $`\bar N = 10^3`$, $`\zeta N = 10`$, $`N = 20`$, $`z = -0.25`$, $`E = 500`$), with polycomp extrapolated to zero time step:

| Quantity | polymerfts | polycomp | One loop |
|---|---:|---:|---:|
| $`\beta\Pi R_0^3/C`$, homopolymer solution | 0.98083 ± 0.00020 | 0.98093 ± 0.00003 | 0.98095 |
| $`\beta\Pi R_0^3/C`$, polyelectrolyte | 9.97248 ± 0.00021 | 9.97253 ± 0.00003 | 9.97262 |
| $`\beta\mu_p`$, homopolymer solution | 0.03840 ± 0.00002 | 0.03843 ± 0.00004 | 0.03838 |
| $`\beta\mu_p`$, polyelectrolyte | 0.04036 ± 0.00001 | 0.04033 ± 0.00004 | 0.04036 |
| $`\beta\mu_+`$, polyelectrolyte | 1.947073 ± 0.000001 | 1.947069 ± 0.000004 | 1.947072 |

$`H`$ and the density structure functions agree within the errors at 12 wavevector shells. The fluctuation corrections are resolved at 100–600 standard errors. polycomp's first-order ETD step shows a time-step bias that polymerfts, with the Jacobian of Section 2.1, does not.

polycomp's ETD step also takes the linear coefficient of a real-type field with the wrong sign. A real-type field is one with $`\gamma = 1`$ and $`B < 0`$, such as the exchange field of a blend. Without a response of the chains, one step maps the field to $`(2 - e^x)`$ times itself, with $`x`$ the rate times $`\vert 1/B \vert`$. The step diverges for $`x > \ln 3`$ and below it inflates the variance by $`(e^x + 1)/(3 - e^x)`$. polycomp measured 1.106, 1.950 and 13.25 at $`x`$ = 0.1, 0.5 and 1, and with the sign corrected 0.993 to 0.996 up to $`x = 4`$. The implicit-solvent systems above have no real-type field and are not affected.

---

## 8. Gibbs Ensemble

`GibbsEnsemble` (`gibbs_ensemble.py`) finds coexisting phases with two CL-FTS boxes, one per phase, that exchange chains, point ions and volume at fixed totals (Reference 9). Each box keeps the reference density on average, so its volume follows from its chains, $`V_b = \sum_p \alpha_p n_p^b`$, with the numbers of chains $`n_p^b`$ and of ions $`n_s^b = N c_s^b V_b`$ in units of $`C/R_0^3`$ per unit volume. A chain of species $`p`$ that moves from box II to box I takes its volume along, and the free energy changes by $`\tilde\mu_p^I - \tilde\mu_p^{II}`$ with the exchange potential

```math
\tilde\mu_p = \beta\mu_p - \alpha_p \frac{\beta\Pi R_0^3}{C}, \qquad \tilde\mu_s = \beta\mu_s
```

for a point ion, which has no volume. Both are the averages of the real parts of the operators of Section 7 over one cycle of Langevin steps of each box. After the cycle the amounts move down the difference:

```math
\Delta n^I = -r \, M_P \left(\tilde\mu^I - \tilde\mu^{II}\right), \qquad \Delta n^{II} = -\Delta n^I, \qquad M = \mathrm{diag}\!\left(\frac{n^I n^{II}}{n^I + n^{II}}\right)
```

with the rate $`r`$ (`exchange_rate`). For ideal species $`M`$ makes the step relative to the amounts and the relaxation rate of every species $`r`$; a step moves at most half of the smaller amount of a species. Without charges $`M_P = M`$. With charged species, no screening and only canonical ions each box stays neutral: $`M_P = M - M\mathbf{z}\mathbf{z}^T M/(\mathbf{z}^T M \mathbf{z})`$ with the charges $`\mathbf{z}`$ of the species (the valence of an ion, $`N \sum_i z_i \alpha_{p,i}`$ of a chain) moves only neutral combinations, and the arbitrary uniform potential of each box drops out of the step. Grand-canonical point ions have the same chemical potential in both boxes and are not exchanged, so the boxes are in contact with one salt reservoir. If they are charged, the uniform potential of each box is its Donnan potential against the reservoir ([Polymer Field Theory, Section 6](PolymerFieldTheory.md#6-grand-canonical-species)); then $`M_P = M`$, every species moves on its own and the canonical charge of a box may change. At coexistence the exchange potentials agree (in every neutral combination when only those move); the pressures need not agree, since each box is held at the reference density. Every species needs a positive amount in both boxes, since a species absent from a box could not enter it.

**Cycles.** Each cycle builds both boxes from the input with their volume fractions, ion concentrations and box lengths, which scale with $`V_b^{1/d}`$ at the same grid. A box continues from the fields, $`\psi`$ and the previous noises of the Leimkuhler-Matthews step (of the auxiliary fields and of $`\psi`$) at the end of its previous cycle (its fields file), with a random seed of its own for each cycle, and the step numbers continue. In canonical CL-FTS the uniform mode of the pressure field relaxes only at the rate $`2/(\chi N + 2\zeta N)`$ per unit time, and it shifts the chemical potentials and the pressure. The uniform modes of the fields therefore move with the homogeneous saddle point, $`w_i = \sum_j \chi_{ij} N \bar\phi_j`$ and the homogeneous $`\psi_0`$, when the composition changes, and a box without initial fields starts at its homogeneous saddle point. `skip_steps` leaves the first steps of each cycle out of the averages. The console line of each cycle shows the largest difference of the exchange potentials (in a neutral system without its component along $`\mathbf{z}`$, the gauge). `gibbs_history.json` holds the settings and every cycle, and `continue_run()` continues an interrupted run from it and the fields files as the uninterrupted run would. `summary()` averages the second half of the cycles.

**Use.** The averages of each cycle carry statistical errors, so the amounts fluctuate about coexistence with an amplitude that grows with $`r`$ and falls with the length of the cycles; average them over many cycles and check that the result does not depend on $`r`$ and `cycle_steps`. Near a critical point the relaxation slows down, since the free energy changes little along the composition difference. The boxes must be large enough to hold a homogeneous phase; a box that phase separates, or forms a microphase, does not represent one phase. [GibbsEnsemble.py](../../examples/blend/clfts/GibbsEnsemble.py) runs a binary blend.

**Limitations.** Canonical free chains only (no grand-canonical or grafted polymers), periodic boxes with the grid of the input, and the reference density in every box, which is exact as $`\zeta N \to \infty`$. Grand-canonical point ions are allowed.

**Tests.** `TestGibbsEnsemble` checks, in the mean-field limit ($`\bar N = 10^{28}`$), that a symmetric blend at $`\chi N = 3`$ reaches the Flory-Huggins binodal within $`2 \times 10^{-6}`$; that for charged chains with counterions and salt as canonical point ions, started at the coexistence of the Flory-Huggins free energy with ideal ions solved independently, the exchange potentials of the boxes agree in every neutral combination within $`10^{-6}`$, and that from distant boxes the boxes stay neutral and approach it; that charged chains with grand-canonical salt reach the Donnan coexistence of the same free energy, solved independently with the Donnan potentials of the boxes, where every exchange potential agrees within $`10^{-6}`$; the input checks; and that a run continued by `continue_run()` reproduces the uninterrupted one bit for bit.

---

## 9. References

1. Ganesan, V. & Fredrickson, G. H. "Field-theoretic polymer simulations." *Europhys. Lett.* **55**, 814 (2001).

2. Aarts, G., Seiler, E. & Stamatescu, I.-O. "Complex Langevin method: When can it be trusted?" *Phys. Rev. D* **81**, 054508 (2010).

3. Delaney, K. T. & Fredrickson, G. H. "Recent Developments in Fully Fluctuating Field-Theoretic Simulations of Polymer Melts and Solutions." *J. Phys. Chem. B* **120**, 7615 (2016).

4. Nagata, K., Nishimura, J. & Shimasaki, S. "Argument for justification of the complex Langevin method and the condition for correct convergence." *Phys. Rev. D* **94**, 114515 (2016).

5. Willis, J. D. & Matsen, M. W. "Stabilizing complex-Langevin field-theoretic simulations for block copolymer melts." *J. Chem. Phys.* **161**, 244903 (2024).

6. Matsen, M. W., Ramírez, J., Willis, J. D. & Pina, P. D. "Accessing the universal phase behavior of block copolymer melts with complex-Langevin field-theoretic simulations." *J. Chem. Phys.* **164**, 014905 (2026).

7. Cox, S. M. & Matthews, P. C. "Exponential time differencing for stiff systems." *J. Comput. Phys.* **176**, 430 (2002).

8. Squire, W. & Trapp, G. "Using complex variables to estimate derivatives of real functions." *SIAM Rev.* **40**, 110 (1998).

9. Riggleman, R. A. & Fredrickson, G. H. "Field-theoretic simulations in the Gibbs ensemble." *J. Chem. Phys.* **132**, 024104 (2010).

10. Aarts, G., Seiler, E., Sexty, D. & Stamatescu, I.-O. "Complex Langevin dynamics and zeroes of the fermion determinant." *J. High Energy Phys.* **2017**(05), 044 (2017).

11. Fredrickson, G. H. & Delaney, K. T. *Field-Theoretic Simulations in Soft Matter and Quantum Fluids.* Oxford University Press (2023).
