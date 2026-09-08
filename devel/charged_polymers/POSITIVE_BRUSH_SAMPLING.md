# Positive 1D brush sampling: implementation record

Actual N=100 stochastic results are recorded separately in
[POSITIVE_BRUSH_RESULTS.md](POSITIVE_BRUSH_RESULTS.md). The equations below
define exactly which finite model was sampled; they are not a claim of
equivalence to the existing CL code or to the full 3D PRL measure.

The requested continuation is stochastic sampling. The earlier Fig. 2–3
LDA SCFT sweep did not carry out that request. This document records the
replacement calculation; the SCFT files are retained as separate references.

The initial implementation keeps the previously requested one-dimensional
restriction. Its explicit finite-volume action has polymer bead coordinates
z only, and integrates the electrostatic Gaussian field exactly. Transverse
charge fluctuations are absent, so it is not the full three-dimensional PRL
measure even when N, b, sigma, v, chi and charge radii have their PRL values.
The transverse area A determines the strength of fluctuations and must be
reported and varied. No LDA self-energy is inserted into the sampler.

For M cells, dz=L/M, K=A dz/v an integer, p_j polymer occupancies and
s_j=K-p_j solvent occupancies, the positive weight is

$$
\pi\propto B(\mathbf R)\,e^{-U_e-U_\chi}
\prod_j\frac{1}{s_j!c_j!}
\frac{a_+^{n_{+,j}}a_-^{n_{-,j}}}{n_{+,j}!n_{-,j}!},
\qquad U_\chi=\frac{\chi}{K}\sum_j p_j(K-p_j).
$$

Polymer bead paths are labeled and B contains Gaussian bonds and the graft
source. The C occupancies have fixed total n_C=n_chain N. Salt occupancies
obey sum n_- = z_+ sum n_+. The a's are dimensionless single-cell activities,
not measured bulk concentrations. Solvent positions and the pressure
constraint have been summed explicitly through s_j=K-p_j and its factorial.

If F_ej is the cumulative Gaussian charge shape of a particle in cell j at
edge e, Q_e=sum_j F_ej[-p_j+c_j+z_+ n_+,j-n_-,j], then

$$U_e=\frac{2\pi\ell_B\,dz}{A}\sum_e Q_e^2$$

with zero normal electric field at the boundaries. The periodic bulk
reference instead uses (Q_e-mean Q)^2. Each update preserves exact charge
neutrality. This is an explicitly discretized planar electrostatic action;
Gaussian smearing, boundary conditions and self terms must stay the same
in its checks and in sampling.

An optional attractive real field x gives an exactly equivalent joint
particle/field sampler:

$$
e^{(\chi/K)\sum_j(p_j-\bar p)^2}
\propto\int_{\sum x=0} dx\,
e^{-K\sum_j x_j^2/(4\chi)+\sum_j p_jx_j},
\quad
x\mid p\sim\mathcal N\left(2\chi(p-\bar p)/K,\;(2\chi/K)\Pi\right).
$$

The fully integrated particle sampler provides a second implementation of
the same finite action. They must agree within sampling uncertainty.

Implemented checks: exact small-cell grand partition sums; detailed balance of
polymer/solvent exchanges, ion redistribution, neutral salt insertion and
deletion; direct versus incrementally updated electrostatic energy; exact
local P/S occupancy and total charge; independent seeds, equilibration,
autocorrelation, grid and area sensitivity. Bulk activities must be calibrated
against a separately sampled pure-salt reference before labeling an axis
rho_b [M]. Completion, uncertainties and remaining model limits are recorded
in the results document, not inferred from the number of sweeps.

The sampler mixes same-cell neutral salt insertions/deletions with proposals
that place the ions independently across the box. For the latter birth,
the reverse move selects one cation and z anions uniformly without replacement;
the Metropolis factor is

$$R_+=\exp(-\Delta U_e)\frac{(a_+a_-^z)M^{z+1}}
{(N_++1)\prod_{k=1}^z(N_-+k)}.$$

The full proposal ratio includes repeated-cell combinatorics; these cancel
against the occupancy factorials. This avoids waiting for rare same-cell
coincidences in dilute multivalent salt. Detailed balance of both proposal
types is tested independently against the complete finite-state weight.

Production FFT/C++/CUDA, the existing CL drift, and the earlier nine-bead
sampler are protected. Evidence directory:
`/home/yongdd/polymer/dh_salt_runs/positive_brush_20260907/`.
