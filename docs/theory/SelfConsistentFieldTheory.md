# Self-Consistent Field Theory (SCFT)

This document describes the self-consistent field theory (SCFT) implementation for finding saddle-point solutions in polymer field theory.

## Table of Contents

1. [Overview](#1-overview)
2. [Saddle-Point Equations](#2-saddle-point-equations)
3. [Iterative Methods](#3-iterative-methods)
4. [Box Optimization](#4-box-optimization)
5. [Implementation](#5-implementation)
6. [References](#6-references)

---

## 1. Overview

Self-consistent field theory (SCFT) finds the saddle-point of the effective Hamiltonian where the fields and concentrations are mutually consistent. At the saddle point:

- The functional derivatives of the Hamiltonian with respect to the fields vanish
- The fields satisfy the self-consistency equations
- The free energy is stationary

SCFT provides mean-field predictions for polymer phase behavior, morphology, and thermodynamic properties.

---

## 2. Saddle-Point Equations

### 2.1 Self-Consistency Conditions

For a system with monomer types $`i = A, B, C, \ldots`$, the self-consistent potential fields satisfy:

```math
w_i(\mathbf{r}) = \sum_{j \neq i} \chi_{ij} N \phi_j(\mathbf{r}) + \xi(\mathbf{r})
```

where:
- $`w_i(\mathbf{r})`$: potential field for monomer type $`i`$
- $`\chi_{ij}`$: Flory-Huggins interaction parameter
- $`\phi_j(\mathbf{r})`$: concentration of monomer type $`j`$
- $`\xi(\mathbf{r})`$: Lagrange multiplier enforcing incompressibility

### 2.2 Field Residuals

The field residuals measure the deviation from self-consistency. They have a composition part and a pressure part:

```math
\mathbf{R}_w = P_\perp \left( X \boldsymbol{\phi} - \mathbf{w} \right) + \kappa \, \mathbf{e} \left( \sum_i \phi_i - 1 \right), \qquad P_\perp = I - \frac{\mathbf{e}\mathbf{e}^T}{M}, \qquad \kappa = \frac{\max(\max_k \vert\lambda_k\vert, 1)}{2}
```

where:
- $`X`$: the $`\chi N`$ matrix
- $`\mathbf{e} = (1, \ldots, 1)`$
- $`\lambda_k`$: the eigenvalues of $`P_\perp X P_\perp`$ (the $`\lambda_i`$ of the auxiliary fields, [PolymerFieldTheory.md](PolymerFieldTheory.md))

$`\mathbf{R}_w = 0`$ exactly at the saddle point: $`P_\perp (X \boldsymbol{\phi} - \mathbf{w}) = 0`$ means $`\mathbf{w} = X \boldsymbol{\phi} + \xi \mathbf{e}`$ for some $`\xi(\mathbf{r})`$, and the second part requires $`\sum_i \phi_i = 1`$. Where $`\sum_i \phi_i = 1`$ the composition part is the residual of the exchange fields of the symmetric theory, $`\Omega_k \leftarrow (\lambda_k \Phi_k + S_k)/M`$, written for the monomer fields; it needs no inverse of $`X`$. The pressure part raises the pressure where the total density is too high. Its scale $`\kappa`$ follows the coupling between the composition and the pressure, which grows with $`\chi N`$ ($`\kappa = \chi N / 2`$ for AB); it does not depend on $`\mathbf{e}^T X^{-1} \mathbf{e}`$, which can be negative or close to zero (e.g. ABC with $`\chi_{AC} N`$ large), and $`X`$ may be singular (a zero pair in a three-monomer system, chemically identical monomers). The spatial mean of each component is subtracted, $`R_{w_i} \leftarrow R_{w_i} - \langle R_{w_i} \rangle`$; the field means are set separately (Section 1.3 of [PolymerFieldTheory.md](PolymerFieldTheory.md#13-scft-iteration-incompressible)).

### 2.3 Example: AB Diblock Copolymer

For AB-type systems $`\lambda = -\chi N`$ and $`\kappa = \chi N / 2`$ (for $`\chi N \geq 1`$), so

```math
R_{w_A} = \chi N \phi_B(\mathbf{r}) - \frac{1}{2}(w_A(\mathbf{r}) - w_B(\mathbf{r})) - \frac{\chi N}{2}
```

```math
R_{w_B} = \chi N \phi_A(\mathbf{r}) + \frac{1}{2}(w_A(\mathbf{r}) - w_B(\mathbf{r})) - \frac{\chi N}{2}
```

(the constants drop out with the subtracted means).

### 2.4 Compressible Model ($`\zeta N \neq 0`$)

When a finite compressibility parameter $`\zeta N`$ is introduced, the incompressibility constraint is replaced by a penalty:

```math
U_{comp} = \frac{\zeta N}{2V} \int d\mathbf{r} \left[ \sum_i \phi_i(\mathbf{r}) - 1 \right]^2
```

The pressure field becomes a known function of the concentrations rather than an unknown Lagrange multiplier:

```math
\xi(\mathbf{r}) = \zeta N \left[ \sum_i \phi_i(\mathbf{r}) - 1 \right]
```

The self-consistency conditions become:

```math
w_i(\mathbf{r}) = \sum_{j \neq i} \chi_{ij} N \phi_j(\mathbf{r}) + \zeta N \left[ \sum_j \phi_j(\mathbf{r}) - 1 \right]
```

#### Field Residuals (Compressible)

A direct residual formulation using $`R_w = (X + \zeta N \, \mathbf{e}\mathbf{e}^T) \boldsymbol{\phi} - \zeta N \, \mathbf{e} - \mathbf{w}`$ is mathematically correct but numerically stiff. The effective interaction matrix $`X + \zeta N \, \mathbf{e}\mathbf{e}^T`$ has eigenvalue $`\chi N + 2\zeta N`$ (e.g., 213.27 for $`\chi N = 13.27`$, $`\zeta N = 100`$), requiring very small mixing rates for convergence.

Instead, the stiff $`\zeta N`$ term, which lies along $`\mathbf{e}`$, is rescaled with the $`\kappa`$ of Section 2.2:

```math
\mathbf{R} = P_\perp \mathbf{R}_w + \frac{\kappa}{\kappa + \zeta N} \, P_\parallel \mathbf{R}_w, \qquad P_\parallel = \frac{\mathbf{e}\mathbf{e}^T}{M}, \qquad R_i \leftarrow R_i - \langle R_i \rangle
```

$`\mathbf{R} = 0`$ exactly when $`\mathbf{R}_w = 0`$, so the fixed point is unchanged. For $`\zeta N \to \infty`$ the residual becomes the incompressible one of Section 2.2, $`P_\perp (X \boldsymbol{\phi} - \mathbf{w}) + \kappa \, \mathbf{e} (\sum_j \phi_j - 1)`$, and the iteration converges as fast as in the incompressible model; for $`\zeta N \to 0`$ it is $`\mathbf{R}_w`$ itself.

The field means have no gauge freedom in the compressible model; they are set directly after each field update:

```math
\langle w_i \rangle \leftarrow \sum_j X_{ij} \langle \phi_j \rangle + \xi
```

where $`\xi = \zeta N \left( \langle \sum_j \phi_j \rangle - 1 \right)`$ and $`\langle \phi_j \rangle`$ is fixed by the chain compositions. This pinning runs for every compressible run.

#### Example: AB Diblock (Compressible)

At convergence with $`\phi_A = \phi_B \approx 0.5`$ and $`\xi \approx 0`$:

```math
\langle w_A \rangle = \langle w_B \rangle = \frac{\chi N}{2} = 6.635
```

```math
\langle W_+ \rangle = \frac{\langle w_A \rangle + \langle w_B \rangle}{2} = \frac{\chi N}{2} \approx 6.64
```

using the implemented normalization $`\boldsymbol{\Omega} = O^T \mathbf{w} / M`$ with $`M = 2`$ (see [PolymerFieldTheory.md](PolymerFieldTheory.md)).

---

## 3. Iterative Methods

### 3.1 Simple Mixing (Gradient Descent)

The simplest approach updates fields using the residuals:

```math
w_i(\mathbf{r}, \tau+1) = w_i(\mathbf{r}, \tau) + \lambda (R_w)_i
```

where $`\lambda`$ is the mixing parameter and $`\tau`$ is the iteration number.

**Advantages:**
- Simple to implement
- Robust for initial iterations

**Disadvantages:**
- Slow convergence near the saddle point
- Requires small $`\lambda`$ for stability

### 3.2 Anderson Mixing

Anderson mixing accelerates convergence by using history of previous iterations:

```math
w^{(n+1)} = w^{(n)} + R_w^{(n)} - \sum_{i=1}^{m} c_i \left[ (w^{(n)} - w^{(n-i)}) + (R_w^{(n)} - R_w^{(n-i)}) \right]
```

where the coefficients $`c_i`$ are determined by minimizing:

```math
\min_{c_i} \left\| R_w^{(n)} - \sum_{i=1}^{m} c_i (R_w^{(n)} - R_w^{(n-i)}) \right\|^2
```

**Key parameters:**
- `max_hist`: Maximum history length (typically 10-20)
- `start_error`: Error threshold to switch from simple mixing to Anderson mixing
- `mix_min`, `mix_init`: Minimum and initial mixing parameter $`\lambda`$ of the simple-mixing phase (before `start_error` is reached); $`\lambda`$ is multiplied by 0.7 (not below `mix_min`) when the error grows and by 1.01 otherwise. The Anderson step above uses no mixing parameter.

**Advantages:**
- Much faster convergence (often 10x fewer iterations)
- Robust for a wide range of systems

### 3.3 Convergence Criterion

The self-consistency error is defined as:

```math
\epsilon = \sqrt{\frac{\sum_i \int |\mathbf{R}_{w_i}|^2 d\mathbf{r}}{1 + \sum_i \int |w_i|^2 d\mathbf{r}}}
```

The constant 1 in the denominator prevents divergence when the field magnitudes are small.

Iteration terminates when $`\epsilon < \text{tolerance}`$ (typically $`10^{-7}`$ to $`10^{-9}`$). With `relax_cell=True`, the Euclidean norm of the stress along the relaxed lattice parameters (the derivatives with respect to the box lengths and, when angles are optimized, the angles) is added to $`\epsilon`$ before this test. The $`n`$ lengths tied by the crystal system (a = b or a = b = c) contribute one component, the sum of their derivatives divided by $`\sqrt{n}`$ (§4.2).

---

## 4. Box Optimization

### 4.1 Stress Residuals

The equilibrium unit cell minimizes the free energy. The stress residuals are:

```math
\mathbf{R}_{stress} = -\frac{\partial H}{\partial \boldsymbol{\theta}}
```

where $`\boldsymbol{\theta} = [L_a, L_b, L_c, \gamma, \beta, \alpha]^T`$ contains the unit cell parameters. Note the angle ordering: the API (`get_stress()`) returns the derivatives $`+\partial H / \partial \boldsymbol{\theta}`$ as the array $`[\partial H/\partial L_a, \partial H/\partial L_b, \partial H/\partial L_c, \partial H/\partial \gamma, \partial H/\partial \beta, \partial H/\partial \alpha]`$ (in 2D, $`\partial H/\partial \gamma`$ is stored at index 2). In general $`\partial H / \partial \boldsymbol{\theta} = -\sum_p (\bar{\phi}_p / \alpha_p) \, (1/Q_p) \, \partial Q_p / \partial \boldsymbol{\theta}`$, which reduces to $`-(1/Q) \, \partial Q / \partial \boldsymbol{\theta}`$ for a single polymer species with $`\alpha = 1`$.

### 4.2 Box Update

During SCFT iteration with `relax_cell=True`:

```math
L_a^{(n+1)} = L_a^{(n)} - \eta \cdot \frac{\partial H}{\partial L_a}
```

```math
\alpha^{(n+1)} = \alpha^{(n)} - \eta \cdot \frac{\partial H}{\partial \alpha}
```

where $`\eta`$ is the `stress_scale` parameter. At equilibrium, all stress components vanish. In the implementation, the scaled stress residuals are concatenated with the field residuals and passed through the field optimizer (Anderson mixing); the plain gradient-descent update above corresponds to the simple-mixing limit.

**Implementation details:**
- Box size changes are limited to 10% per iteration
- Angle changes are limited to 5° per iteration
- Stress can be computed adaptively (`stress_interval: "adaptive"`) or at fixed intervals (default: 1)
- Crystal system constraints (orthorhombic, tetragonal, cubic, hexagonal, monoclinic, triclinic) are automatically enforced
- Lengths tied by the crystal system move together along the mean of their derivatives, so the relaxation and its convergence test reach the optimum of the tied cell. A structure without the symmetry of the tie (for example a = b of `"Tetragonal"` without a space group) can have tied lengths whose derivatives are nonzero and opposite at that optimum; no cell of that crystal system is then free of stress, and SCFT prints a warning that names a crystal system with free lengths
- Stress computation supports periodic, reflecting, and absorbing boundary conditions with pseudo-spectral methods (real fields); the real-space CN-ADI2 solver does not support stress. `relax_cell=True` therefore requires a pseudo-spectral method; with non-periodic boundaries only box lengths are optimized (angles stay at 90°).

See [StressTensor.md](StressTensor.md) for detailed stress calculation.

---

## 5. Implementation

### 5.1 Using the SCFT Class

```python
from polymerfts import scft

params = {
    "platform": "cuda",              # or "cpu-mkl", "cpu-fftw"
    "nx": [64, 64, 64],              # Grid points
    "lx": [4.0, 4.0, 4.0],           # Box size

    "chain_model": "continuous",     # or "discrete"
    "ds": 0.01,                      # Contour step (1/N_Ref)

    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 20.0},

    "polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": 0.5},
            {"type": "B", "length": 0.5},
        ],
    }],

    "optimizer": {
        "name": "am",                # Anderson Mixing
        "max_hist": 20,
        "start_error": 1e-1,
        "mix_min": 0.1,
        "mix_init": 0.1,
    },

    "relax_cell": True,         # Enable box optimization
    "stress_scale": 0.5,             # Stress scaling factor

    "max_iter": 2000,
    "tolerance": 1e-8,
}

# Initialize and run
calculation = scft.SCFT(params=params)
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Access results
phi_A = calculation.phi["A"]
free_energy = calculation.free_energy
final_box = calculation.cb.get_lx()

# Save results
calculation.save_results("fields.json")
```

`run` also accepts a progress callback and can return the results as one object:

```python
def monitor(info):                      # info: IterationInfo
    print(info.iteration, info.error_level, info.energy, info.lx)
    return False if info.iteration == 200 else None   # False stops the run

result = calculation.run(initial_fields={"A": w_A, "B": w_B},
                         on_iteration=monitor, return_result=True)
result.converged, result.free_energy, result.phi["A"]   # fields on the full grid
calculation.run(initial_fields=result.w)               # restart from the solution
```

`SCFTResult.phi` and `SCFTResult.w` are dictionaries of full-grid arrays, also with a space group, so `result.w` can start the next run of a parameter sweep.

### 5.2 Manual SCFT Implementation

For educational purposes, here's a simplified SCFT loop:

```python
from polymerfts import PropagatorSolver
import numpy as np

# Create solver
solver = PropagatorSolver(
    nx=[64, 64], lx=[2.0, 2.0], ds=0.01,
    bond_lengths={"A": 1.0, "B": 1.0},
    chain_model="continuous",
    bc=["periodic"] * 4,
)
solver.add_polymer(volume_fraction=1.0,
                   blocks=[["A", 0.5, 0, 1],   # [monomer_type, length, v, u]
                           ["B", 0.5, 1, 2]])  # v, u are node indices

# Initialize fields (flattened arrays matching the 64x64 grid)
x = np.linspace(0, 2*np.pi, 64, endpoint=False)
X, _ = np.meshgrid(x, x, indexing="ij")
w = {"A": 5.0*np.sin(X).flatten(), "B": -5.0*np.sin(X).flatten()}
chi_n = 20.0
lambda_mix = 0.5

for iteration in range(1000):
    # Compute concentrations
    solver.compute_propagators(w)
    solver.compute_concentrations()
    phi_A = solver.get_concentration("A")
    phi_B = solver.get_concentration("B")

    # Compute residuals
    R_A = chi_n * phi_B - 0.5 * (w["A"] - w["B"])
    R_B = chi_n * phi_A + 0.5 * (w["A"] - w["B"])
    R_A -= np.mean(R_A)   # the uniform part (chi_n/2) does not vanish at the saddle point
    R_B -= np.mean(R_B)

    # Check convergence
    error = np.sqrt(np.mean(R_A**2) + np.mean(R_B**2))
    if error < 1e-7:
        break

    # Update fields
    w["A"] += lambda_mix * R_A
    w["B"] += lambda_mix * R_B
```

### 5.3 Key Output

| Property | Access | Description |
|----------|--------|-------------|
| Concentration | `calculation.phi["A"]` | Monomer density field |
| Relative free energy | `calculation.free_energy` | Difference from the homogeneous state at the same composition; zero for a disordered bulk state; used in the iteration log and progress callbacks |
| Full CE free energy | `calculation.free_energy_absolute` | Full canonical Helmholtz free energy $`\beta F_{\mathrm{tot}}/(\rho_0V/N)`$, including ideal mixing entropy; equals `free_energy + free_energy_homogeneous` |
| Homogeneous state | `calculation.free_energy_homogeneous` | Full homogeneous Flory-Huggins value at the same composition |
| Chemical potential | `result.chemical_potentials[p]` | $`\beta\mu_p`$ per polymer molecule; saddle-point means restored, with $`\langle\xi\rangle=0`$ in the incompressible model |
| Grand potential | `result.grand_potential` | $`F-\sum_p(\bar\phi_p/\alpha_p)\beta\mu_p`$ using the full CE F, in the same pressure convention |
| Coexistence quantities | `result.coexistence_quantities(reference_polymer=0)` | Exchange chemical potentials per reference-chain volume and the reduced grand potential with the reference polymer's chemical potential set to zero |
| Chain composition | `result.polymer_lengths`, `result.polymer_volume_fractions` | Relative chain lengths $`\alpha_p`$ and polymer volume fractions $`\bar\phi_p`$ |
| Partition function | `solver.get_partition_function(p)` | Single-chain partition function |
| Box size | `calculation.cb.get_lx()` | Final box dimensions |
| Stress | `result.stress` | Stress components; available via `result = calculation.run(..., return_result=True)` (set only when `relax_cell=True`), or via `solver.compute_stress()` / `solver.get_stress()` |

The final summary prints both relative and full free energies. For phase comparisons at different compositions, use `free_energy_absolute` and the reference-species quantities described in [Phase Equilibrium from Canonical SCFT](PhaseEquilibrium.md), which also explains evaluation at prescribed chemical potentials using CE calculations. `SCFTResult` is obtained with `run(..., return_result=True)`; the thermodynamic values are also stored on the `SCFT` object and by `save_results()`. Chemical potentials are unavailable for grafted chains or supplied initial propagators. Bulk coexistence requires converged, incompressible, periodic states.

### 5.4 Additional Features

The SCFT class supports several advanced features not covered in the basic examples above:

| Feature | Parameter | Description |
|---------|-----------|-------------|
| Space group symmetry | `"space_group": {"symbol": "Ia-3d"}` | Exploit crystallographic symmetry (beta) |
| Smearing | `"smearing": {...}` | Finite-range interactions |
| Random copolymers | `{"monomer_fractions": {"A": 0.3, "B": 0.7}}` | Statistical copolymer blocks |
| Non-orthogonal cells | `"angles": [90, 90, 120]` | Monoclinic/triclinic unit cells |
| Memory saving | `"reduce_memory": True` | Checkpoint-based propagator storage |
| Stress interval | `"stress_interval": 3` or `"adaptive"` | Control stress computation frequency |

See [Parameters.md](../references/Parameters.md) for complete parameter reference.

---

## 6. References

1. Ceniceros, H. D. & Fredrickson, G. H. "Numerical solution of polymer self-consistent field theory." *Multiscale Model. Simul.* **2**, 452-474 (2004).

2. Stasiak, P. & Matsen, M. W. "Efficiency of pseudo-spectral algorithms with Anderson mixing for the SCFT of periodic block-copolymer phases." *Eur. Phys. J. E* **34**, 110 (2011).

3. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).

4. Thompson, R. B., Rasmussen, K. Ø. & Lookman, T. "Improved convergence in block copolymer self-consistent field theory by Anderson mixing." *J. Chem. Phys.* **120**, 31-34 (2004).
