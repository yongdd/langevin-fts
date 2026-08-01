# Charged-Polymer FTS: Theory and Implementation Plan

> Working notes (devel). Formulation of polyelectrolyte solutions within this
> codebase's multi-monomer field theory, following the smeared-charge model of
> Wang [1] and Villet–Delaney–Fredrickson [2,3], written in the conventions of
> *Macromolecules* **2025**, 58, 816 (the neutral multi-monomer theory this
> library implements).

## 1. Model

Species $i = 1, \dots, S$ (here: polymer P, solvent S, counter-ion C, salt
ions SP/SM). Each species is a (possibly very short) bead-spring chain of the
discrete or continuous model already implemented. Two interaction channels on
top of the neutral theory:

1. **Flory–Huggins** $\chi_{ij}$ among a subset of species (the
   `chi_monomers`; here P and S). Species outside this subset (the small
   ions) have $\chi = 0$ with everything.
2. **Coulomb**, between smeared charge densities. Segment of species $i$
   carries valence $z_i$ (per segment), distributed as a normalized Gaussian
   $$h_i(\mathbf r) = \frac{1}{(2\pi a_i^2)^{3/2}} e^{-r^2/2a_i^2},
     \qquad \hat h_i(\mathbf k) = e^{-a_i^2 k^2/2},$$
   with smearing radius $a_i$ (params `radiuses`, in $R_0$ units after
   nondimensionalization). Smearing regularizes the Coulomb self-energy and
   makes the field theory UV-convergent [1,2]; the point-charge limit
   $a_i \to 0$ is singular and must not be taken on the grid.

The microscopic charge density is
$$\hat\rho_c(\mathbf r) = \sum_i z_i \,(h_i * \hat\rho_i)(\mathbf r),$$
with $\hat\rho_i$ the microscopic segment density of species $i$. The Coulomb
energy is
$$\beta U_C = \frac{l_B}{2} \int\!\!\int
   \frac{\hat\rho_c(\mathbf r)\,\hat\rho_c(\mathbf r')}{|\mathbf r-\mathbf r'|}
   \, d\mathbf r\, d\mathbf r',
   \qquad l_B = \frac{e^2}{4\pi \varepsilon k_B T}$$
(Bjerrum length; a uniform dielectric $\varepsilon$ is assumed — no
dielectric contrast in this first version).

**Incompressibility**: $\sum_i \hat\phi_i(\mathbf r) = 1$ as in the neutral
theory (or compressible with $\zeta N$).

**Global electroneutrality** is required. With per-segment valences $z_i$
and equal segment volumes the charge density is $\rho_0 \sum_i z_i \phi_i$, so
the constraint is
$$\sum_i z_i\, \bar\phi_i = 0$$
where $\bar\phi_i$ is the overall segment volume fraction (NO division by
chain length — that would be per-chain counting, wrong for per-segment
valences). The counter-ion fraction
is therefore NOT a free parameter: for P with $z_P = +1$ and fraction
$\phi_P$, one needs $\phi_C = \phi_P z_P/|z_C|$ (plus balanced salt
$z_{SP}\phi_{SP} + z_{SM}\phi_{SM} = 0$). *The current `Lamella.py`
placeholder sets $\phi_C = 0$ with charged P — that configuration is not
electroneutral and the $k=0$ Coulomb mode would diverge; the driver must
compute $\phi_C$ from $\phi_P$.*

## 2. Units and code conventions

Same as the neutral theory: lengths in $R_0 = a_{\rm Ref} N_{\rm Ref}^{1/2}$,
fields per **reference chain** ($w_{\rm code} = N\,w_{\rm per\,segment}$),
densities in $\rho_0$. Define the chain number density
$C = \rho_0 R_0^3 / N = \sqrt{\bar N}$ (as in L-FTS).

Nondimensionalize with $\mathbf x = \mathbf r/R_0$ and the dimensionless
smeared charge density per segment
$c(\mathbf x) = \sum_i z_i (h_i * \phi_i)(\mathbf x)$ ($h_i$ Gaussian of
width $a_i$ in $R_0$ units, $\phi_i$ volume fractions). The total Coulomb
energy in $k_BT$ is then (note the $C^2$ — Coulomb is quadratic in
$\rho_0$, unlike the χ block where $\chi N \propto \rho_0$ leaves one $C$):
$$\beta U_C = \frac{C^2 E_0}{8\pi}\int\!\!\int
   \frac{c(\mathbf x)\,c(\mathbf x')}{|\mathbf x-\mathbf x'|}\,
   d\mathbf x\, d\mathbf x'
 = \frac{C^2 E_0}{2} \sum_{\mathbf k} \frac{|\hat c_{\mathbf k}|^2}{k^2},
 \qquad E_0 \equiv \frac{4\pi l_B N^2}{R_0}.$$
For the per-chain-normalized Hamiltonian used throughout this codebase, the
convenient coupling is
$$\boxed{\;E \;\equiv\; C\,E_0 \;=\; \frac{4\pi l_B N^2 C}{R_0}
   \;=\; 4\pi l_B\, \rho_0 N R_0^2\;}$$
(density-DEPENDENT: $E = \sqrt{\bar N}\,E_0$). Caveat: the literature's
"electrostatic strength" [2,3] is the density-independent $E_0$, usually in
$R_g = R_0/\sqrt6$ units — convert when comparing. New scalar param:
`"bjerrum_e"` $= E$ (or `"bjerrum_length"` $= l_B/R_0$ with $E$ computed
internally from `nbar`).

## 3. Field theory: Hubbard–Stratonovich structure

The identity-resolution and HS transforms factor into three independent
blocks:

**(a) χ block — unchanged.** The $\chi_{ij}$ quadratic form over the
`chi_monomers` subset is diagonalized exactly as in the existing
`SymmetricPolymerTheory` (projected $\chi N$ matrix, eigenvalues
$\lambda_k$, exchange fields real for $\lambda_k<0$, imaginary for
$\lambda_k>0$). Species outside the subset simply do not appear in this
block. Dimension: $S_\chi \times S_\chi$ with $S_\chi = $ len(chi_monomers)
(2 here).

**(b) Pressure block — unchanged.** Incompressibility over ALL $S$ species
gives the pressure field $w_+$ (imaginary type), seen identically by every
species.

**(c) Coulomb block — new.** HS on the positive-definite Coulomb quadratic
form introduces the electrostatic potential $\psi(\mathbf x)$:
$$e^{-\beta U_C} = \int \mathcal D\psi\;
  \exp\!\left[ - C \int d\mathbf x\, \frac{|\nabla\psi|^2}{2E}
               \;+\; i\, C \int d\mathbf x\; c(\mathbf x)\,\psi(\mathbf x) \right]
  \Big/ \mathcal N,$$
which reproduces $\exp[-\tfrac{C^2E_0}{2}\sum_k |\hat c_k|^2/k^2]$ exactly
because $E = CE_0$ (Gaussian integral over $\psi$; this is why the
density-dependent $E$ of §2 is the natural coupling here). $\psi$ is the
per-reference-chain potential $N\beta e\varphi_{\rm phys}$ — the factor $N$
is absorbed into $\psi$ as for all code fields. Because it couples with
$+i$ to a real density, $\psi$ is an **imaginary-type field** (the Coulomb
kernel $4\pi l_B/k^2$ is positive definite, exactly analogous to
$\lambda > 0$ eigenvalues): it lands with the $w_+$ family — see §5.

## 4. Hamiltonian and single-chain problem

Collecting (a)–(c), the per-chain effective Hamiltonian is
$$\frac{H[\{w_k\}, w_+, \psi]}{C\,k_BT V/R_0^3}
 = \underbrace{h_{\rm const} + \sum_k \left[ A_k \langle w_k\rangle + B_k \langle w_k^2\rangle\right]}_{\text{neutral (existing)}}
 \;+\; \frac{1}{V}\int \frac{|\nabla \psi|^2}{2E} d\mathbf x
 \;-\; \sum_p \frac{\bar\phi_p}{\alpha_p} \ln Q_p[\{W_i\}]$$
with the same $h$-coefficients as the neutral theory. The only change to the
single-chain problem is the **one-body potential of each species**:
$$\boxed{\;W_i(\mathbf x) \;=\; W_i^{\chi}(\mathbf x) \;+\; i\,w_+(\mathbf x)
   \;+\; i\, z_i\, (h_i * \psi)(\mathbf x)\;}$$
where $W_i^{\chi}$ is the usual `matrix_a` mapping of the exchange fields
(zero for species outside `chi_monomers`). No explicit $N$ multiplies the
electrostatic term: $\psi$ is already per-reference-chain and $z_i$ is
per-segment — the code's $w = N w_{\rm seg}$ convention is carried entirely
by $\psi$. The smearing appears as a
convolution of $\psi$, NOT of the propagator — cheap in k-space:
$\widehat{h_i * \psi} = \hat h_i(k)\hat\psi(k)$.

Concentrations: standard $\phi_i$ from propagators; the **smeared** charge
density that sources the Poisson equation is
$c = \sum_i z_i\, h_i * \phi_i$ (a second convolution, again diagonal in k).

## 5. Forces and saddle conditions

Functional derivatives (per the code's normalization):

- Exchange/pressure fields: unchanged from the neutral theory.
- Electrostatic field:
$$\frac{1}{C}\frac{\delta H}{\delta \psi(\mathbf x)}
  = -\frac{\nabla^2 \psi}{E} \;-\; c(\mathbf x)
  \qquad\Longrightarrow\qquad
  \text{saddle: } -\nabla^2\psi^* = E\, c(\mathbf x)$$
i.e. the (smeared) **Poisson equation**; with the ideal-gas ions responding
through their Boltzmann factors this is the fluctuating-field generalization
of Poisson–Boltzmann. In k-space the saddle solve is diagonal:
$$\hat\psi^*(\mathbf k) = \frac{E\,\hat c(\mathbf k)}{k^2}, \qquad k \ne 0,$$
and the $k=0$ mode is fixed by electroneutrality ($\hat c(0) = 0$ holds
exactly per configuration in the canonical ensemble once §1's constraint
holds; set $\hat\psi(0)=0$ as gauge).

**Debye sanity check** (1-segment ions, $z=\pm1$, ideal): linearizing gives
$\kappa^2 R_0^2 = E\,\bar\phi_{\rm ion}/N = 4\pi l_B \rho_0
\bar\phi_{\rm ion} R_0^2$ — the physical Debye constant. Any factor error
in $E$ or $W_i$ shows up here first; keep this as a unit test.

**Field classification for L-FTS** (recommended first implementation):
treat $\psi$ like $w_+$ — a *partial-saddle* field solved to tolerance at
every Langevin step, while only the real exchange field(s) receive Langevin
noise. This is the standard approach in charged L-FTS [3] and requires no
new stochastic machinery: the compressor loop gains a second, *linear* solve
(the Poisson equation above), which unlike the $w_+$ iteration is exact in
one k-space division per iteration of the outer loop.

Convergence detail: $\psi$ and $w_+$ couple through the densities, so the
practical scheme is: within the existing saddle iteration, after each
propagator solve, (i) update $\psi$ exactly from the current $c(\mathbf x)$,
(ii) take the usual Anderson-mixing step on $w_+$. Both residuals go into
the stopping criterion.

## 6. Discretization notes

- All new operations are diagonal in k-space: $\hat h_i(k) = e^{-a_i^2k^2/2}$
  multiplications and the $E/k^2$ Poisson solve. Use the same FFT objects and
  the deformation-vector $|\mathbf k|^2$ tables (`Pseudo`) already present —
  the periodic-BC $k^2$ including the reciprocal metric is available; oblique
  cells therefore work automatically.
- The Gaussian smearing widths must satisfy $a_i \gtrsim dx/2$; below that
  the smearing is under-resolved and the self-energy regularization is grid
  dependent (same class of issue as the bond-function study in
  `devel/spring_bead_bond/`).
- Structure function / observables: the charge–charge structure factor
  $S_{cc}(k)$ comes for free from $\hat c(k)$; worth recording alongside the
  existing $S(k)$.
- Stress (box optimization) with electrostatics: the Maxwell-stress term
  $\partial/\partial L\, \int |\nabla\psi|^2/2E$ must be added if
  `box_is_altering` is ever used — defer; raise for now.

## 7. Parameter dictionary (target)

```python
"chi_monomers": ["P", "S"],          # species entering the chi block
"charges":  {"P": 1.0, "S": None, "C": -1.0, "SP": +1, "SM": -1.0},
"radiuses": {"P": 0.025, "S": None, "C": 0.025, "SP": 0.025, "SM": 0.025},
"bjerrum_e": 10000.0,                # E = 4 pi l_B N^2 / R0   (NEW - was missing)
# molecules: counter-ion fraction computed from electroneutrality, not free
```

`None` charge = neutral species (no $\psi$ coupling, no smearing needed).

## 8. Implementation plan (on CURRENT mainline, not the Feb fork)

The Jan–Feb fork predates PropagatorSolver/validation; rebase the ideas, not
the code:

1. `polymer_field_theory.py`: keep `SymmetricPolymerTheory` neutral and
   untouched; add a `ChargedPolymerTheory` wrapper that owns
   (chi-subset SPT) + charges/radii/E and produces per-species $W_i$ from
   (exchange fields, $w_+$, $\psi$). NOTE this is more than a thin wrapper:
   running the FULL 5-species matrix through SPT would produce zero
   eigenvalues and a singular `matrix_a` (χ only couples P,S), so the
   wrapper must own the $W_i$ assembly for non-χ species (pressure + ψ
   only), which touches how `lfts.py` builds per-monomer fields.
2. `lfts.py`: accept the new params; inside the saddle loop add the exact
   k-space $\psi$ update (§5) and include the electrostatic energy term in
   $H$; electroneutrality validated at init (ValidationError). Keep $\psi$
   OUT of `aux_fields_imag_idx` and the Anderson-mixing compressor state
   (the AM object is sized to the imaginary-field count at construction) —
   $\psi$ has its own exact solve and needs no mixing.
   **Collision warning**: a global `Smearing` class already exists
   (`src/python/smearing.py`, the `"smearing"` param) that Gaussian-filters
   ALL fields; the per-species $h_i$ here is a different object — do not
   route it through that class, and forbid combining both params.
3. Ions as short chains work today (`length=0.01` ⇒ 1 discrete segment);
   no C++ changes required for the first version — everything new lives in
   Python/k-space on fields the C++ solvers never see internally.
4. Validation ladder:
   a. $E \to 0$ reduces exactly to the neutral multi-monomer run.
   b. Salt-only (no polymer): compare $S_{cc}(k)$ against the
      Debye–Hückel/RPA result with smeared charges,
      $S_{cc}^{-1} = [S_{cc}^{\rm ideal}]^{-1} + E/k^2$ where
      $S_{cc}^{\rm ideal}$ is the ideal structure factor of the SMEARED
      charge density (it contains the $\hat h_i^2$ factors — with a bare
      ideal $S$ one must write $\hat h^2 S/(1 + (E/k^2)\hat h^2 S)$; the
      $\hat h^2$ cannot be dropped). $k \to 0$ recovers the Debye
      $\kappa^2$ of §5.
   c. Uncharged-P limit vs existing L-FTS Lamella.
   d. Literature anchor: polyelectrolyte solution structure factor of
      Refs. [2,3].

## References

1. Z.-G. Wang, *Phys. Rev. E* **81**, 021501 (2010) — fluctuation theory of
   electrolytes, self-energy and smearing.
2. M. V. Villet, G. H. Fredrickson, *J. Chem. Phys.* **141**, 224115 (2014) —
   CL-FTS with Gaussian-regularized (UV-convergent) models.
2b. R. A. Riggleman, R. Kumar, G. H. Fredrickson, *J. Chem. Phys.* **136**,
   024903 (2012) — polyelectrolyte complexation with smeared charges
   (sharper primary reference for the charged CL-FTS model).
3. K. T. Delaney, G. H. Fredrickson, *J. Phys. Chem. B* **120**, 7615 (2016) —
   recent developments in fully fluctuating polyelectrolyte FTS.
4. *Macromolecules* **2025**, 58, 816 — the neutral multi-monomer framework
   (this library).
