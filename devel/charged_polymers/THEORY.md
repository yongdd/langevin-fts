# Charged-Polymer FTS: Theory and Implementation Plan

> Working notes (devel). Formulation of polyelectrolyte solutions within this
> codebase's multi-monomer field theory, following the smeared-charge model of
> Wang [1] and Villet–Delaney–Fredrickson [2,3], written in the conventions of
> *Macromolecules* **2025**, 58, 816 (the neutral multi-monomer theory this
> library implements). **Compressible (finite $\zeta N$) model with
> per-species density smearing** — the fully UV-regularized formulation.

## 1. Model

Species $i = 1, \dots, S$ (here: polymer P, solvent S, counter-ion C, salt
ions SP/SM). Each species is a (possibly very short) bead-spring chain of the
discrete or continuous model already implemented.

**Smeared densities.** Every species carries a normalized Gaussian shape
function
$$h_i(\mathbf r) = \frac{1}{(2\pi a_i^2)^{3/2}} e^{-r^2/2a_i^2},
  \qquad \hat h_i(\mathbf k) = e^{-a_i^2 k^2/2},$$
of radius $a_i$ (params `radiuses`, in $R_0$ units). $a_i$ is the
**smearing length** (Fredrickson school: width of the "shape function" or
form factor of the species [2,3]); for charged species Wang [1] identifies
it with the ion's **Born radius**, since the Gaussian self-energy
$u_{\rm self} = z_i^2 l_B/(2\sqrt{\pi}\, a_i)$ is the Born solvation
energy of an ion of that size. ALL interactions are written in terms of the
smeared volume fractions
$$\bar\phi_i \equiv h_i * \phi_i .$$
Smearing regularizes both the contact (excluded-volume/χ) and the Coulomb
self-interactions, making the fluctuating field theory UV-convergent
[1,2,3]. The point-particle limit $a_i \to 0$ is singular for a compressible
fluctuating model and must not be taken. ($a_i = $ `None` ⇒ $h_i = \delta$;
acceptable only for testing or in mean-field.)

Three interaction channels:

1. **Flory–Huggins**: $\chi N_{ij}$ pairs (here only P,S — all pairs
   involving ions are zero). No separate `chi_monomers` machinery is needed
   in the compressible model: the $\chi N$ matrix is simply $S \times S$
   with zero rows/columns for the ions (see §3).
2. **Compressibility (Helfand)**: instead of the incompressibility
   constraint, a finite penalty
   $$\beta U_{\zeta} = \frac{\rho_0 \zeta N}{2 N} \int
     \Big( \sum_i \bar\phi_i(\mathbf r) - 1 \Big)^2 d\mathbf r$$
   with the code's existing `zeta_n` parameter. The incompressible theory is
   the $\zeta N \to \infty$ limit. In a fluctuating simulation the smeared
   $\bar\phi_i$ (not raw $\phi_i$) must enter here, or the contact penalty
   is UV-divergent.
3. **Coulomb**: segment of species $i$ carries per-segment valence $z_i$;
   the smeared charge density is
   $$c(\mathbf r) = \sum_i z_i\, \bar\phi_i(\mathbf r)
     \qquad (\text{same } h_i \text{ as above}),$$
   and
   $$\beta U_C = \frac{l_B \rho_0^2}{2} \int\!\!\int
      \frac{c(\mathbf r)\, c(\mathbf r')}{|\mathbf r-\mathbf r'|}
      \, d\mathbf r\, d\mathbf r',
      \qquad l_B = \frac{e^2}{4\pi \varepsilon k_B T}$$
   (Bjerrum length; uniform dielectric $\varepsilon$, no dielectric
   contrast in this first version).

**Global electroneutrality** is still required (compressibility does not fix
charge). With per-segment valences $z_i$ and equal segment volumes:
$$\sum_i z_i\, \bar\phi_i^{\,\rm tot} = 0$$
over overall volume fractions (NO division by chain length — that would be
per-chain counting, wrong for per-segment valences). The counter-ion
fraction is NOT a free parameter: for P with $z_P = +1$ and fraction
$\phi_P$, one needs $\phi_C = \phi_P z_P/|z_C|$ (plus balanced salt
$z_{SP}\phi_{SP} + z_{SM}\phi_{SM} = 0$). *The current `Lamella.py`
placeholder sets $\phi_C = 0$ with charged P — not electroneutral; the
$k=0$ Coulomb mode diverges. The driver must compute $\phi_C$ from
$\phi_P$.*

## 2. Units and code conventions

Same as the neutral theory: lengths in $R_0 = a_{\rm Ref} N_{\rm Ref}^{1/2}$,
fields per **reference chain** ($w_{\rm code} = N\,w_{\rm per\,segment}$),
densities in $\rho_0$. Chain number density
$C = \rho_0 R_0^3 / N = \sqrt{\bar N}$.

Nondimensionalize with $\mathbf x = \mathbf r/R_0$. The Coulomb energy in
$k_BT$ becomes (note the $C^2$ — Coulomb is quadratic in $\rho_0$, unlike
the χ/ζ block where $\chi N, \zeta N \propto \rho_0$ leave one $C$):
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

Two independent HS blocks (the compressible model has NO separate pressure
constraint):

**(a) χ+ζ block — the existing compressible SPT, over ALL $S$ species.**
The quadratic form of channels 1+2 is
$$\tfrac12\, \bar{\boldsymbol\phi}^{\mathsf T}
  \big( \zeta N\, \mathbf J + \boldsymbol\chi N \big) \bar{\boldsymbol\phi}
  \;-\; \zeta N\, \mathbf 1^{\mathsf T}\bar{\boldsymbol\phi} + \tfrac12 \zeta N,$$
with $\mathbf J$ the all-ones matrix — exactly the matrix
$u = \zeta N\,\mathbf J + \chi N$ that `SymmetricPolymerTheory`'s
compressible branch (`zeta_n` set) already diagonalizes, including the
linear term (its `vector_large_s`). Ion rows of $\chi N$ are zero. Spectrum
for the 5-species example:
- one large positive eigenvalue $\approx S\zeta N$ (the "pressure-like"
  mode) — imaginary-type;
- the χ-split P–S pair: one negative (exchange, **real** Langevin field),
  one positive (imaginary-type);
- pure ion-composition modes orthogonal to $\mathbf 1$ with no χ:
  eigenvalue **0** — ideal mixing in those channels; the corresponding
  auxiliary fields are identically zero and SPT's existing zero-eigenvalue
  handling (Gram–Schmidt + warning) applies. This is physical, not an
  error: those composition fluctuations are free.

Because $\zeta N\,\mathbf J$ lifts the would-be singularity of the χ-only
matrix, **the full $S$-species SPT can be used directly** — the
chi-subset wrapper of the incompressible draft is no longer needed.

**(b) Coulomb block — new.** HS on the positive-definite Coulomb quadratic
form introduces the electrostatic potential $\psi(\mathbf x)$:
$$e^{-\beta U_C} = \int \mathcal D\psi\;
  \exp\!\left[ - C \int d\mathbf x\, \frac{|\nabla\psi|^2}{2E}
               \;+\; i\, C \int d\mathbf x\; c(\mathbf x)\,\psi(\mathbf x) \right]
  \Big/ \mathcal N,$$
which reproduces $\exp[-\tfrac{C^2E_0}{2}\sum_k |\hat c_k|^2/k^2]$ exactly
because $E = CE_0$ (Gaussian integral over $\psi$). $\psi$ is the
per-reference-chain potential $N\beta e\varphi_{\rm phys}$. Coupling with
$+i$ to a real density and a positive-definite kernel ($4\pi l_B/k^2$,
analogous to $\lambda > 0$): $\psi$ is an **imaginary-type field**,
partial-saddled like the positive-eigenvalue SPT fields — see §5.

## 4. Hamiltonian and single-chain problem

The per-chain effective Hamiltonian is
$$\frac{H[\{w_k\}, \psi]}{C\,k_BT V/R_0^3}
 = \underbrace{h_{\rm const} + \sum_k \left[ A_k \langle w_k\rangle + B_k \langle w_k^2\rangle\right]}_{\text{neutral compressible (existing SPT coefficients)}}
 \;+\; \frac{1}{V}\int \frac{|\nabla \psi|^2}{2E} d\mathbf x
 \;-\; \sum_p \frac{\bar\phi_p}{\alpha_p} \ln Q_p[\{W_i\}]$$
with the $h$-coefficients of the existing compressible SPT. The one-body
potential of each species picks up the smearing convolution on EVERY term
(the interactions couple to $\bar\phi_i = h_i*\phi_i$, so the functional
derivative w.r.t. $\phi_i$ convolves back with $h_i$):
$$\boxed{\;W_i(\mathbf x) \;=\; h_i * \Big[ W_i^{\rm SPT}
   \;+\; i\, z_i\, \psi \Big](\mathbf x)\;}$$
where $W_i^{\rm SPT}$ is the usual `matrix_a` mapping of the auxiliary
fields (now including the pressure-like ζ mode; ion rows couple only to
that mode and to $\psi$). No explicit $N$ multiplies the electrostatic
term: $\psi$ is already per-reference-chain and $z_i$ is per-segment — the
code's $w = N w_{\rm seg}$ convention is carried entirely by $\psi$. The
smearing is a k-space multiplication
$\hat h_i(k)\,[\widehat{W^{\rm SPT}_i} + i z_i \hat\psi](k)$ — cheap, per
species, applied once per saddle iteration; the propagator solvers are
untouched.

**Rotated contour / stored real field.** As with the positive-eigenvalue
SPT modes, the code stores the REAL array $\psi_s = -i\psi$ (the saddle
lies on the imaginary $\psi$ axis). Under this substitution the one-body
term becomes $+z_i \psi_s$ (the $i$ disappears — this is what the code
adds to $W_i^{\rm SPT}$), and the explicit quadratic term flips sign:
$$\frac{1}{V}\int \frac{|\nabla\psi|^2}{2E}\, d\mathbf x
  \;\longrightarrow\;
  -\frac{1}{V}\int \frac{|\nabla\psi_s|^2}{2E}\, d\mathbf x
  \;=\; -\frac{1}{2V}\int c\,\psi_s\, d\mathbf x$$
(the last equality at the Poisson-consistent $\psi_s$). This mirrors the
negated $B_k$ coefficients of the imaginary SPT fields. Note the
$+c\psi_s$ coupling itself lives inside $-\ln Q$ via $W_i$; the explicit
term recorded in $H$ is only the (negative) gradient piece.

Concentrations: standard $\phi_i$ from propagators; the smeared densities
that enter all interaction terms and the Poisson source are
$\bar\phi_i = h_i * \phi_i$, $c = \sum_i z_i \bar\phi_i$ (again diagonal
in k).

## 5. Forces and saddle conditions

- SPT auxiliary fields: unchanged from the neutral compressible theory —
  functional derivatives evaluated with the SMEARED densities
  $\bar\phi_i$ in place of $\phi_i$. Negative-eigenvalue (exchange) fields
  receive Langevin noise; positive-eigenvalue fields are partial-saddled by
  the existing compressor loop; zero-eigenvalue fields stay zero.
- Electrostatic field:
$$\frac{1}{C}\frac{\delta H}{\delta \psi(\mathbf x)}
  = -\frac{\nabla^2 \psi}{E} \;-\; c(\mathbf x)
  \qquad\Longrightarrow\qquad
  \text{saddle: } -\nabla^2\psi^* = E\, c(\mathbf x),$$
the (smeared) **Poisson equation**; with the ideal-gas ions responding
through their Boltzmann factors this is the fluctuating-field
generalization of Poisson–Boltzmann. In k-space the solve is diagonal and
exact:
$$\hat\psi^*(\mathbf k) = \frac{E\,\hat c(\mathbf k)}{k^2}, \qquad k \ne 0,$$
and the $k=0$ mode is fixed by electroneutrality ($\hat c(0) = 0$ holds
exactly per configuration in the canonical ensemble once §1's constraint
holds; set $\hat\psi(0)=0$ as gauge).

**Debye sanity check** (1-segment ions, $z=\pm1$, ideal): linearizing gives
$\kappa^2 R_0^2 = E\,\bar\phi_{\rm ion}/N = 4\pi l_B \rho_0
\bar\phi_{\rm ion} R_0^2$ — the physical Debye constant. Any factor error
in $E$ or $W_i$ shows up here first; keep this as a unit test.

**CL-FTS (the real implementation): $\psi$ fluctuates.** L-FTS holds
imaginary-type fields at partial saddle, so it can never capture $\psi$
fluctuations — the charged theory belongs in CL-FTS, where $\psi$ evolves
with complex Langevin dynamics exactly like the pressure field $W_+$
(rotated storage, imaginary-direction noise, "+$\Lambda$" drift):
$$\psi \;\leftarrow\; \psi + \left(\frac{\nabla^2\psi}{E} + c\right)
  \Delta t\, s_\psi \;+\; i\,\mathcal N(0,\sigma)\sqrt{s_\psi},$$
whose drift fixed point is the Poisson equation. Three discrete-scheme
facts (all found the hard way):
- The $\nabla^2/E$ part is stiff at high $k$: explicit Euler is UNSTABLE
  ($k^2\Delta t/E$ exceeds the stability limit), and semi-implicit Euler,
  while stable, SUPPRESSES the stationary variance by $1/(1+2\kappa\Delta
  t)$ per mode — this silently wrecks fluctuation thermodynamics (the
  one-loop $\langle\partial H/\partial E\rangle$ came out $2$–$4\times$
  too small). The correct treatment is **ETD (exact Ornstein–Uhlenbeck
  integration per k-mode)**:
  $$\hat\psi' = a_k\hat\psi + (1-a_k)\frac{E\hat c}{k^2}
    + i\,\sigma\sqrt{\frac{1-a_k^2}{2\gamma_k\Delta t}}\;\hat\xi,
    \qquad a_k = e^{-\gamma_k \Delta t s_\psi},\;\; \gamma_k = k^2/E,$$
  which reproduces the exact per-mode stationary variance of the linear
  part for ANY $\Delta t$; the $c[\psi]$ screening stays explicit
  (per-step rate $\le \sum_i z_i^2\bar\phi_i\,\Delta t$). The $k=0$ mode
  is gauged to zero every step.
- The EXCHANGE field's explicit mass is $\sim 1/\chi N$, so small-$\chi$
  systems (e.g. the pure-salt validation) need $\Delta t < \chi N$;
  violating this diverges regardless of the charge sector.
- The explicit Hamiltonian term must be evaluated ANALYTICALLY off the
  saddle: $+(1/2EV)\int \psi\nabla^2\psi$ (equal to
  $-\tfrac12\langle c\psi\rangle$ only when Poisson holds).

At $E=0$ the $\psi$ stiffness is infinite: $\psi$ stays pinned at zero.

**L-FTS prototype (mean-field $\psi$): scheme per saddle iteration** —
after each propagator solve, (i) build $\bar\phi_i$, $c$ (k-space
$\hat h_i$ multiplications), (ii) update $\psi$ toward the Poisson
solution (below), (iii) take the usual Anderson-mixing step on the
imaginary SPT fields. Both residuals enter the stopping criterion. $\psi$
stays OUT of `aux_fields_imag_idx` and the AM compressor state.

**Stabilized $\psi$ update (important).** Replacing $\psi$ by the bare
solution $E\hat c/k^2$ every iteration is an undamped fixed-point sweep
whose linearized gain is
$$g(k) \;=\; \frac{E}{k^2}\sum_i z_i^2\,\bar\phi_i\,\hat h_i^2(k),$$
which exceeds 1 at long wavelengths for realistic $E$ (e.g. $E=25$,
$k_{\min}=2\pi/L$) — the joint saddle loop then diverges. Use instead the
preconditioned Newton update with the ideal (local) screening estimate
$S_{\rm scr}(k) = \sum_i z_i^2 \bar\phi_i \hat h_i^2(k)$:
$$\hat\psi \;\leftarrow\; \hat\psi
  \;+\; \frac{E\,\hat c - k^2\,\hat\psi}{\,k^2 + E\,S_{\rm scr}(k)\,},
  \qquad \hat\psi(0)=0 .$$
The fixed point is unchanged ($-\nabla^2\psi = E c$, monitored via the
residual $-\nabla^2\psi/E - c$), but the screening term in the denominator
damps exactly the modes the bare sweep amplifies; for ideal 1-segment ions
the update is a true Newton step. $S_{\rm scr}$ only sets the convergence
rate, so the segment-fraction estimate suffices for polymeric charges.

## 6. Discretization notes

- All new operations are diagonal in k-space: $\hat h_i(k)$ multiplications
  and the $E/k^2$ Poisson solve. Use the existing FFT objects and the
  deformation-vector $|\mathbf k|^2$ tables (`Pseudo`) — oblique cells work
  automatically.
- **No real-space $\nabla\psi$ is ever needed.** The electrostatic energy
  is evaluated by Parseval, and at the saddle the substitution
  $\hat\psi = E\hat c/k^2$ eliminates the gradient entirely:
  $$\frac{1}{2E}\int |\nabla\psi|^2 d\mathbf x
    = \frac{E}{2}\,\frac{V}{M^2}\sum_{\mathbf k \ne 0} w_t\,
      \frac{|\hat c_{\mathbf k}|^2}{k^2}
    = \frac12 \int c\,\psi\, d\mathbf x$$
  (unnormalized `rfftn`; $w_t$ = 2 interior / 1 edge modes, as in
  `wtmd.py`). Only $h_i * \psi$ requires an inverse transform. A real-space
  gradient (component-wise $ik_\alpha \hat\psi$) first becomes necessary
  for the deferred Maxwell-stress term. The identity above gives the
  MAGNITUDE; on the rotated contour the term entering the recorded $H$ is
  $-\tfrac{1}{2V}\int c\,\psi_s$ (see §4) — the code evaluates it as
  `-0.5*mean(c*psi)`.
- Smearing widths must satisfy $a_i \gtrsim dx/2$ **per species** — the
  binding constraint is $\min_i a_i$; below that the regularization becomes
  grid-dependent (same issue class as the bond-function study). With finite
  $\zeta N$ this applies to ALL species, not just charged ones.
- Per-species radii compose automatically: the effective $i$–$j$ pair
  kernel is $\hat h_i(k)\hat h_j(k) = e^{-(a_i^2+a_j^2)k^2/2}$, i.e. an
  effective pair smearing width $\sqrt{a_i^2+a_j^2}$ — Gaussian widths add
  in quadrature, so mixed radii (small ions, fat polymer segments) need no
  special treatment. Consequence for validation: with unequal radii the
  RPA/Debye–Hückel comparison must keep the per-species $\hat h_i^2$
  factors separate (they only factor out as a common $\hat h^2$ when all
  radii are equal); the $k \to 0$ Debye limit is radius-independent
  ($\hat h_i(0) = 1$), which is why it is the safe unit test.
- Charge–charge structure factor $S_{cc}(k)$ from $\hat c(k)$: record
  alongside the existing $S(k)$.
- Stress with electrostatics (Maxwell term from
  $\partial_L \int |\nabla\psi|^2/2E$) and with the $\hat h_i(k)$ cell
  dependence: defer; raise on `box_is_altering=True`.

## 7. Parameter dictionary (target)

```python
"zeta_n": 100.0,                     # compressible Helfand penalty (existing param)
"charges":  {"P": 1.0, "S": None, "C": -1.0, "SP": +1, "SM": -1.0},
"radiuses": {"P": 0.025, "S": 0.025, "C": 0.025, "SP": 0.025, "SM": 0.025},
"bjerrum_e": 10000.0,                # E = 4 pi l_B rho_0 N R0^2   (NEW)
"chi_n": {"P,S": 50},                # full-matrix SPT; ion rows zero
# NO "chi_monomers" needed; NO incompressibility constraint
# molecules: counter-ion fraction computed from electroneutrality, not free
```

`None` charge = neutral species (no $\psi$ coupling). `radiuses` now apply
to ALL interactions (χ, ζ, Coulomb) of that species — in the compressible
fluctuating model neutral species need smearing too.

## 8. Implementation (devel-only; CL-FTS is the primary implementation)

**Where the code lives (2026-08-02)**: entirely in THIS folder — mainline
`src/` is untouched.
- `electrostatics.py`: per-species smearing kernels (real AND complex
  fields), coupling $E$, electrostatic Hamiltonian, electroneutrality
  check, plus the $\psi$ Newton update used by the L-FTS prototype.
- `clfts_charged.py` — **the primary implementation**:
  `ChargedCLFTS(polymerfts.clfts.CLFTS)` with $\psi$ as a fully
  fluctuating CL field (§5): overrides `compute_concentrations` (complex
  $W_i = h_i*(W^{\rm SPT}_i + z_i\psi)$) and `run` (co-evolves $\psi$
  semi-implicitly, adds $-\tfrac12\langle c\psi\rangle$ to $H$, saves
  $\psi$ in checkpoints). $\psi$ noise uses a separate PCG64 stream
  (seed+1) so the neutral reduction stays bit-exact.
- `lfts_charged.py` — mean-field-$\psi$ PROTOTYPE kept for validation
  only: its deterministic saddle makes the sampling-free Debye–Hückel
  linear-response test possible (`test_debye_huckel.py`), which pins the
  $E$ normalization/kernels shared with the CL class.

L-FTS samples only the REAL exchange fields and holds imaginary-type
fields at partial saddle, so $\psi$ there is mean-field
(Poisson–Boltzmann level) — that is why CL-FTS is the real home of the
theory. The compressible choice simplifies everything versus the
incompressible draft:

1. `polymer_field_theory.py`: **unchanged** — the compressible branch
   (`zeta_n`) over the full $S$-species matrix already produces the right
   eigen-system, including the zero-eigenvalue ion modes (existing
   handling).
2. Relation to the existing global `Smearing` class: that class implements
   the SPECIAL CASE of one common radius applied to all fields — the
   prototype bypasses it (adapter), never applying both to the same field.
3. Ions as short chains work today (`length=0.01` ⇒ 1 discrete segment);
   no C++ changes — everything new is k-space Python.
4. Validation ladder:
   a. $E \to 0$, uniform $a_i$: must reproduce the existing neutral
      compressible run with global smearing exactly.
   b. $\zeta N \to \infty$, $E \to 0$: approach the incompressible neutral
      results.
   c. Salt-only (no polymer): $S_{cc}(k)$ against smeared Debye–Hückel/RPA,
      $S_{cc}^{-1} = [S_{cc}^{\rm ideal}]^{-1} + E/k^2$ where
      $S_{cc}^{\rm ideal}$ is the ideal structure factor of the SMEARED
      charge density (contains the $\hat h_i^2$ factors — with a bare ideal
      $S$ one must write $\hat h^2 S/(1 + (E/k^2)\hat h^2 S)$; the
      $\hat h^2$ cannot be dropped). $k \to 0$ recovers the Debye
      $\kappa^2$ of §5.
   d. Literature anchor: polyelectrolyte solution structure of Refs. [2,3]
      (mind the $E_0$ vs $E$ and $R_g$ vs $R_0$ conversions of §2).

**Validation status (2026-08-02)**:
- (a) Neutral reduction PASSED bit-exactly for BOTH classes
  (`test_neutral_reduction.py` for `ChargedLFTS`,
  `test_neutral_reduction_clfts.py` for `ChargedCLFTS`): $z=0$ with
  uniform $a_i$ reproduces the neutral run with global smearing to the
  last bit over 10 Langevin steps.
- (c) Debye–Hückel PASSED. Deterministic linear response
  (`test_debye_huckel.py`, via the L-FTS prototype's saddle): perturbing
  $w_-$ by $\varepsilon\cos(k_0x)$ reproduces
  $|\hat c| = \varepsilon\hat h^2/(1+E\hat h^2/k_0^2)$ and
  $\hat\psi = E\hat c/k_0^2$ to $10^{-6}$ for
  $E \in \lbrace 0, 6.25, 25\rbrace$ over six $k_0$ — pins the $E$
  normalization, $\hat h^2$ pair kernel and $W_i$ assembly without any
  sampling.
- Stochastic $S_{cc}$: with the exact single-mode Gaussian model (orig
  variables: $H_k = Bw^2 + \kappa\psi_o^2 - u_2(w+i\psi_o)^2$,
  $B = 1/\chi N$, $u_2 = \hat h^2/2$, $\kappa = k^2/2E$), the
  parameter-free predictions match: L-FTS
  $r(k) = \kappa^2(B-u_2)/[(\kappa+u_2)(B\kappa+Bu_2-u_2\kappa-2u_2^2)]$
  agrees with 20k-step runs to 1–3% except the lowest-$k$ shell; CL-FTS
  with the ANALYTIC estimator $\langle c_k c_{-k}\rangle$ (NOT
  $\langle|c_k|^2\rangle$, which is not an observable in CL) follows
  $r(k) = (\kappa-B)(B-u_2)/[(B-u_2)(\kappa+u_2)+u_2^2]$ including its
  characteristic sign flip at $k = \sqrt{2EB}$; the strongest-signal
  shells agree to a few % ($E=25$, $k_{\min}$: $-3.08$ vs $-3.07$).
  Residual deviations were pinned down quantitatively: (i) a finite-$dt$
  integrator bias — a synthetic single-mode replica of the exact discrete
  scheme reproduces the $dt{=}0.2$ run values and converges to the
  continuum prediction as $dt \to 0$, and real $dt{=}0.05$ runs move onto
  the synthetic curve (e.g. $k{=}4.71$: run $-2.164$ vs synthetic
  $-2.172$, continuum $-2.31$); (ii) shells where the Gaussian signal
  crosses zero are dominated by a nonlinear (beyond-Gaussian) background
  that does not shrink with statistics — they are not usable for this
  comparison.
- **One-loop DH thermodynamics PASSED (the CL-only validation)**: the
  analytic observable $\langle\partial H/\partial E\rangle = -\langle
  H_{\rm exp}\rangle/E$ measured over $E \in \lbrace 2.5, 6.25, 12.5,
  25, 50\rbrace$ (two $\Delta t$, linearly extrapolated) matches the
  exact-lattice Gaussian prediction $-(1/2\sqrt{\bar n}V)\sum_{k\ne0}
  (\kappa/E)(B-u_2)/\det(k)$ — the smeared generalization of the
  Debye–Hückel limiting law $-\kappa_D^3/12\pi$ — to $\le 0.1\%$ at every
  coupling (ratios 0.999–1.001). This free energy comes entirely from
  $\psi$ FLUCTUATIONS and is identically zero in the partial-saddle
  L-FTS. (Script: dh_salt_runs/test_oneloop_dh.py.)
- Pitfalls recorded the hard way: $\chi N = 2$ IS the mean-field spinodal
  of the symmetric 1-segment salt (RPA breaks); cosine amplitudes on the
  cell-centered grid carry an $e^{ik\,dx/2}$ phase — compare $|\hat c|$;
  the exchange field's explicit mass $1/\chi N$ caps the CL time step at
  $\Delta t < \chi N$; semi-implicit $\psi$ integration biases per-mode
  variances (use ETD, above).

## References

1. Z.-G. Wang, *Phys. Rev. E* **81**, 021501 (2010) — fluctuation theory of
   electrolytes, self-energy and smearing.
2. M. V. Villet, G. H. Fredrickson, *J. Chem. Phys.* **141**, 224115 (2014) —
   CL-FTS with Gaussian-regularized (UV-convergent) models.
3. R. A. Riggleman, R. Kumar, G. H. Fredrickson, *J. Chem. Phys.* **136**,
   024903 (2012) — polyelectrolyte complexation with smeared charges
   (sharper primary reference for the charged CL-FTS model).
4. K. T. Delaney, G. H. Fredrickson, *J. Phys. Chem. B* **120**, 7615 (2016) —
   perspective on fully fluctuating polyelectrolyte FTS.
5. *Macromolecules* **2025**, 58, 816 — the neutral multi-monomer framework
   (this library).
