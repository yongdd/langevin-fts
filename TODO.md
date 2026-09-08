1. Parallel Tempering
6. Packaging with Conda, including scft.py and lfts.py
8. GPU bandwidth test
9. UnitTest: Anderson Mixing
10. Validation Check for Pseudo Parameters
16. Process Bar
20. Graphic User Interface
27. Syntax check of Keys in Molecules 
31. Throws Exception in openMP block
35. Write UnitTest: TestComparePropagatorKey
37. Reduce the number of split in scheduler
40. Figure out why two GPUs performance keeps changing in A100
42. Check if two GPUs can be utilized for CudaAndersonMixing
43. Operator Overloading for class Array
44. Check if CUDA cub works in CUDA 12.0 and higher
46. Replace "new" keyword to a safe memory allocation
47. Add tests for example files
48. Check if CUDA can be installed using Conda
50. Check if cuFFT callback function is applicable
51. Figure out why 'std::map<std::string, std::map<int, double *>>' causes errors in 'omp parallel for'
52. Figure out why 'propagator_half_steps[key].find(0) == propagator_half_steps[key].end()' fixes above problem
53. And a option to print a data file in a human-readable text.
54. Extend scft.py to cover singular chi matrix
55. Add tests for stress computation with complex type
56. Check get_negative_frequency_mapping() and compute_single_segment_stress()
57. Add tests for Anderson mixing with complex type
58. Figure out why 'os.environ' does not work under Python and openmp installation with conda-forge
59. Parameter sweep / continuation (use previous solution as initial guess for phase diagram construction)
60. Grand canonical ensemble (GCE) - core functions implemented (get_stress_gce, get_total_concentration_gce), needs SCFT/LFTS high-level API
61. Documentation: Python API reference (Sphinx + autodoc for PropagatorSolver, SCFT, LFTS classes)
62. Documentation: File formats (fields_*.mat, structure_function_*.mat, YAML/JSON config)
63. Documentation: Examples guide (detailed README for examples/scft/, examples/lfts/)
64. SCFT preconditioner for Anderson Mixing (left-preconditioning failed; try right-preconditioning or modified AM inner product)
65. Investigate discrete FFT/DCT/DST propagator accuracy at large spatial dynamic range (deferred, 2026-09-06).

    Native C++/CUDA propagation is left unchanged by user decision. The experimental
    FFT + selective real-space recovery was withdrawn; do not enable it by default.
    Reproducer: N=50, nz=704, Lz=6, absorbing z boundaries, per-chain W=100 for z<3
    and zero elsewhere; Gaussian graft source centered at 1/sqrt(50), width 0.2.
    The audited CUDA spectral calculation returned Q approximately -1.60e-18 and
    1.70% mass error; an independent same-boundary reference gave Q=+4.66676e-28
    and mean height <z>=3.076019224. Inverse-transform roundoff contaminating small
    propagator tails is the diagnosed issue; it remains unresolved in the native path.
    Before proposing a change, establish error control, threshold/precision convergence,
    complex linearity and forward/backward consistency, density versus d(log Q)/dw,
    long-run CL statistics, and CPU/CUDA performance and memory costs. Distinguish
    the same finite spectral bond operator from a different real-space wall model.
    Reproducers and historical experiment results:
    `/home/yongdd/polymer/dh_salt_runs/audit_20260906/audit_strong_field.py` and
    `/home/yongdd/polymer/dh_salt_runs/fft_repair_20260906/`.
66. Resolve charged CL discrepancies against exact finite-lattice particles (2026-09-07).

    The 24-run benchmark in `devel/charged_polymers/EXACT_CL_VALIDATION.md` passes
    mass/partition and final-state conditional-density checks, but charged CL
    averages differ from exhaustive positive particle sums. At E=100, dt=0.0025,
    the cell-0 counterion density is 0.21308439 +/- 0.00112541 versus exact
    0.20718911. E=0 controls agree within estimated uncertainty.
    Distinguish integration bias, rare-event sampling, and possible incorrect CL
    convergence before treating target PRL simulations as validated. Preserve the
    finite action and native FFT; changing the model cannot validate this benchmark.

    Follow-up: 21 trajectories and direct original-contour field quadrature are in
    `devel/charged_polymers/CL_CONVERGENCE_DIAGNOSIS.md`. Quadrature agrees with the
    particle sum to 2.8e-15. At dt=0.00015625, longer LM/ETD, Heun and adaptive Euler
    still give C density near 0.2132 versus exact 0.20719. Entire field moments also
    disagree. Large drifts occur near analytically located Q_P or Q_SP zeros;
    native and independent densities agree at the drift extremes. Longer E=0
    controls also reveal a smaller bias. These results strongly support incorrect
    CL convergence in this fixture. A validated sampling formulation or validated
    parameter regime is still needed; no production FFT or drift patch was made.

    Further diagnostic: `devel/charged_polymers/CL_POLE_RESOLUTION.md` exactly
    integrates the repulsive fields, leaving a positive one-real-field marginal
    of the SAME finite action. Twelve real-Langevin runs recover all 19 reference
    observables within 1.36 estimated SE. Original CL remains biased even though
    its tested low-order stationarity identities pass; near-zero measurements
    do not account directly for the density bias. This supplies a finite-model
    control, not a scalable production repair or a complete pole-boundary proof.
    Original CL validation and PRL reproduction remain open; FFT is unchanged.

    1D follow-up: `devel/charged_polymers/ONED_PRL_REPRODUCTION.md` implements
    particle/real-field Gibbs updates without exhaustive enumeration, tested on
    4/16/32 cells of the same finite action. It also recalculates PRL Fig. 1 at
    two grid resolutions. Current LDA curves still differ from digitized paper
    values beyond grid/raster errors; ensemble/box and nonlocal-theory differences
    remain. A working positive sampler for this small system does not validate
    the existing CL dynamics or reproduce the complete PRL theory.

67. Resolve the remaining PRL Fig. 2-3 discrepancies (2026-09-07).

    See `devel/charged_polymers/FIG23_PRL_REPRODUCTION.md` and the added LaTeX
    report pages. Fixed-parameter, canonical-C smoothed LDA gives collapse and
    reexpansion, but the low-salt onset differs substantially from the paper.
    Establish the paper's Gibbs-height prescription, box/ensemble and chosen
    nonlocal Green-function/I_ex treatment before claiming quantitative reproduction.
    The paper explicitly neglects u_P^ex in its brush calculations; its omission
    alone is not an established explanation for the discrepancy.
    Fig. 3's monovalent potential raster differs from Eq. (9), with M converted
    to nm^-3, by approximately the 0.6022 conversion factor. Confirm the figure's
    normalization with original data/code; do not silently rescale computed data.

    The separate Fig. 2-3 Python runner also exposes a NON-FFT low-salt issue:
    legacy real-space bead-slice underflow plus its 1e-300 floor/global mass
    normalization can mask an O(0.2) density error. The research runner uses
    extended-range positive convolution, a common partition function, and an
    independent all-log reference. It also includes the canonical PB rank-one
    normalization derivative. These remain isolated research implementations;
    production FFT, CL drift and `prl_scft.py` were not changed in this campaign.
