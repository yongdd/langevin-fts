# Phase Equilibrium from Canonical SCFT

`SCFT` solves the canonical ensemble (CE): the polymer volume fractions are fixed while the fields and, optionally, the unit cell relax. Chemical potentials and grand potentials are obtained from this solution without solving grand-canonical field equations. The coexistence solver adjusts the compositions of two CE calculations until their exchange chemical potentials and reduced grand potentials agree.

## Free energy and units

Write $`n_{\mathrm{ref}}=\rho_0 V/N`$, $`\alpha_p=N_p/N`$, $`c_p=\bar\phi_p/\alpha_p`$, and $`X_{ij}=\chi_{ij}N`$. The full CE free energy stored in `free_energy_absolute` is $`F=\beta F_{\mathrm{tot}}/n_{\mathrm{ref}}`$; chemical potentials $`m_p=\beta\mu_p`$ are per polymer molecule. For free chains, $`Q_p`$ is normalized by volume.

```math
F=\sum_p c_p\left[\ln\left(\frac{c_p}{Q_p}\right)-1\right]
+\left\langle\frac12\boldsymbol\phi^T X\boldsymbol\phi-\mathbf w\cdot\boldsymbol\phi\right\rangle.
```

The homogeneous limit is the full Flory-Huggins expression

```math
F_{\mathrm{hom}}=\sum_p c_p(\ln c_p-1)+\frac12\bar{\boldsymbol\phi}^T X\bar{\boldsymbol\phi}.
```

`free_energy_homogeneous` stores this value. `free_energy` stores the relative value

```math
\Delta F=F-F_{\mathrm{hom}},
\qquad F=\Delta F+F_{\mathrm{hom}}.
```

The relative value is zero in a disordered bulk state. The iteration log and progress callbacks report this relative value; the final summary prints both relative and full values. Use `free_energy_absolute` for composition derivatives, common tangents and grand-canonical transforms: the homogeneous term depends nonlinearly on composition and cannot be discarded. `homogeneous_thermodynamics()` uses the same keys, with `free_energy = 0` and `free_energy_absolute = free_energy_homogeneous`. At zero polymer volume fraction the ideal free-energy contribution is zero, while the species chemical potential is $`-\infty`$.

## Restoring chemical potentials

The CE iteration shifts each incompressible monomer field to zero mean. This leaves concentrations and $`F`$ unchanged but changes $`Q_p`$. Chemical potentials must therefore be evaluated after restoring the saddle-point means.

Choose $`\langle\xi\rangle=0`$. Let $`\ell_{pi}`$ be the total relative contour length of monomer i in polymer p, including the fractional composition of random blocks, so $`\sum_i\ell_{pi}=\alpha_p`$. The mean-field correction is

```math
b_i=(X\bar{\boldsymbol\phi})_i-\langle w_i\rangle,
\qquad
\ln Q_p^{\mathrm{saddle}}=\ln Q_p-\sum_i\ell_{pi}b_i.
```

Consequently,

```math
m_p=\ln c_p-\ln Q_p+\sum_i\ell_{pi}b_i.
```

`SCFT.compute_chemical_potentials()` applies this correction analytically. Optional `w` and `partition_functions` arguments must describe the same CE state; `w` may be `SCFT.w` or a dict of fields such as `SCFTResult.w`, which is on the full grid when a space group is used. No extra propagator calculation is needed. `SCFTResult.chemical_potentials` stores the values for the final snapshot, and `grand_potential` stores

```math
\Omega=F-\sum_p c_p m_p.
```

Here F is always `free_energy_absolute`, including the homogeneous value. For a converged state with positive polymer volume fractions:

```python
import numpy as np

result = calculation.run(initial_fields=fields, return_result=True)
c = np.asarray(result.polymer_volume_fractions) / result.polymer_lengths
omega = result.free_energy_absolute - c @ np.asarray(result.chemical_potentials)
assert np.isclose(omega, result.grand_potential)
```

The fugacity convention is $`z_p=e^{m_p}`$ and $`\bar\phi_p=\alpha_p z_p Q_p^{\mathrm{saddle}}`$. Thus this CE transform equals the grand-canonical expression

```math
\Omega=-\sum_p z_p Q_p^{\mathrm{saddle}}
+\left\langle\frac12\boldsymbol\phi^T X\boldsymbol\phi
-\mathbf w^{\mathrm{saddle}}\cdot\boldsymbol\phi\right\rangle.
```

The saved fields and partition functions retain the CE working field means. When evaluating the expression above, use the restored fields and partition functions; `chemical_potentials` and `grand_potential` already include that correction.

For compressible SCFT, the mean-field target additionally contains $`\zeta N(\sum_i\bar\phi_i-1)`$. The coexistence solver below is restricted to incompressible systems; a compressible phase-equilibrium calculation also needs the total density as a variable.

## Reference species and coexistence

A common pressure shift $`w_i\to w_i+C`$ changes $`m_p\to m_p+\alpha_p C`$ and $`\Omega\to\Omega-C`$, while leaving $`F`$ invariant. In particular, comparing the grand potentials of two CE states with $`\langle\xi\rangle=0`$ alone does not impose coexistence.

Choose a reference polymer r and form

```math
g_p=\frac{m_p}{\alpha_p}-\frac{m_r}{\alpha_r},
\qquad g_r=0,
\qquad
\Psi=F-\sum_{p\ne r}g_p\bar\phi_p
=\Omega+\frac{m_r}{\alpha_r}.
```

Both $`g_p`$ and $`\Psi`$ are independent of the common pressure shift. $`\Psi`$ is the grand potential after setting the reference polymer's chemical potential to zero. For a binary blend with $`x=\bar\phi_1`$ and reference polymer 0, $`g_1=dF/dx`$ and $`\Psi=F-xg_1`$. Two phases A and B coexist when

```math
g_1^A(x_A)=g_1^B(x_B),
\qquad
\Psi^A(x_A)=\Psi^B(x_B).
```

These are the common-tangent slope and intercept conditions. For an AB diblock of length 1 and a homopolymer of length $`\alpha`$, Matsen's chemical-potential variable obeys $`\beta\mu=\alpha g_h`$; his $`F_{\mathrm{gc}}=F_c-\mu n_h`$ is the reference-species transformation above.

```python
result = calculation.run(initial_fields=fields, return_result=True)
quantities = result.coexistence_quantities(reference_polymer=0)
g = quantities["exchange_chemical_potentials"]  # includes the reference entry, zero
psi = quantities["reduced_grand_potential"]
```

To express the same state in the GCE convention with $`m_r=0`$, use

```math
m_p^{\mathrm{GCE}}=\alpha_p g_p,
\qquad \Omega^{\mathrm{GCE}}=\Psi.
```

```python
alpha = np.asarray(result.polymer_lengths)
mu_gce = alpha * np.asarray(g)  # beta mu per molecule; reference entry is zero
fugacity = np.exp(mu_gce)
omega_gce = psi
```

For another prescribed reference value $`m_r^*`$, the corresponding values are $`m_p^*=\alpha_p(g_p+m_r^*/\alpha_r)`$ and $`\Omega^*=\Psi-m_r^*/\alpha_r`$. Matching the pressure convention is necessary when comparing grand-potential values.

## Binary coexistence solver

`find_binary_coexistence` accepts two functions of x, the volume fraction of polymer 1. Each function returns a converged `SCFTResult` for polymer fractions `[1-x, x]`, or an analytic state from `homogeneous_thermodynamics`. Both must use the same physical model and species order. No GCE field solver is invoked.

The following example finds the two homogeneous phases of a blend with different chain lengths:

```python
import numpy as np
from polymerfts import homogeneous_thermodynamics, find_binary_coexistence

def disordered(x):
    return homogeneous_thermodynamics(
        volume_fractions=[1-x, x],
        polymer_lengths=[1.0, 2.0],
        monomer_fractions=np.eye(2),  # polymer p's fractions of monomer i
        interaction_matrix=[[0.0, 2.0], [2.0, 0.0]],  # chi N
    )

coexistence = find_binary_coexistence(
    disordered, disordered, initial_compositions=[0.06, 0.85],
    composition_bounds=((0.01, 0.25), (0.65, 0.98)),
    tolerance=1e-10,
)
print(coexistence["compositions"])  # approximately [0.0595665046, 0.8490591390]
```

For a modulated phase, the callback builds CE parameters for `[1-x, x]`, runs `SCFT`, and returns its result. Reuse the preceding solution's fields and cell as initial guesses to follow that phase branch. `examples/scft/CanonicalCoexistence.py` demonstrates a CE SCFT callback paired with an analytic homogeneous callback for a homopolymer blend.

Use a preliminary CE composition scan to select the phase branches and narrow their bounds. The solver caches scalar thermodynamic evaluations, not full field arrays. It verifies both coexistence residuals against `tolerance`; optimizer step convergence alone is insufficient. An unconverged CE state, nonperiodic boundaries, a compressible model, `electrostatics` (whose point ions and potential gauge need exchange potentials of neutral combinations) or an incorrect callback composition is rejected. Callbacks are responsible for following the intended morphology. Changing morphology or finding another local CE minimum requires a different branch initialization.

The solver finds a common tangent for the specified phase branches. Compare other candidate phases with that tangent to establish global stability. Coincident compositions, critical points and congruent transitions require a separate treatment; `min_separation` excludes coincident solutions. The returned `evaluations` counts actual evaluations of each phase, including finite-difference evaluations used by the optimizer. `tolerance` bounds the coexistence residuals; spatial and contour discretization errors require independent refinement. In continuous-chain calculations, numerical derivatives of F and the chemical potentials agree up to these discretization errors. Expensive composition scans and coexistence calculations should run through SLURM.

## Prescribed chemical potentials using CE calculations

For a binary system, prescribed molecular chemical potentials $`m_0^*,m_1^*`$ determine a target exchange value

```math
g^*=\frac{m_1^*}{\alpha_1}-\frac{m_0^*}{\alpha_0}.
```

Keep each field calculation canonical and adjust x until $`g_1(x)=g^*`$. This outer composition solve supplies the composition that a GCE calculation would return on the same phase branch. The following code uses the `disordered(x)` callback defined above; replace it with a CE SCFT callback to evaluate a modulated phase.

```python
from scipy.optimize import brentq
from polymerfts import coexistence_quantities

alpha = np.array([1.0, 2.0])
target_mu = np.array([0.0, 0.12])  # beta mu per polymer molecule
target_g = target_mu[1] / alpha[1] - target_mu[0] / alpha[0]

def residual(x):
    q = coexistence_quantities(disordered(x), reference_polymer=0)
    return q["exchange_chemical_potentials"][1] - target_g

# A bracket on the low-x branch, chosen from a preliminary composition scan.
x = brentq(residual, 0.01, 0.25, xtol=1e-12)
state = disordered(x)
q = coexistence_quantities(state, reference_polymer=0)
omega = q["reduced_grand_potential"] - target_mu[0] / alpha[0]
```

Repeat the calculation for every candidate phase branch at the same target chemical potentials and compare their grand potentials; the smallest value selects the stable branch among those evaluated. A bracket must contain a root on the intended branch, and SCFT callbacks must return converged states. The CE result's own `chemical_potentials` use its mean-pressure convention; the reference-species conversion above expresses that result at the prescribed values. No GCE field iteration is required.

## Reference

M. W. Matsen, [New Fast SCFT Algorithm Applied to Binary Diblock Copolymer/Homopolymer Blends](https://doi.org/10.1021/ma0349377), *Macromolecules* **36**, 9647–9657 (2003).

On p. 9650, eq. (47) obtains the blend chemical-potential variable from a canonical solution, and eq. (49) gives the canonical-to-grand-canonical free-energy transform. Page 9655 describes an ordered/disordered coexistence procedure that varies the ordered-phase composition using canonical SCFT calculations, evaluates the chemical potential with eq. (47), and matches the two phases' grand-canonical free energies at that chemical potential.
