# Numerical validation and reproducibility

Keep the existing calibrated test parameters and tolerances. A passing solver comparison is evidence for that case, not a general proof of physical validity. Use independent reference calculations and refinement alongside regression fixtures.

## Summary

**Regression tests** (details in [Regression tests](#regression-tests)):

| Area | Tests | Checked against |
|---|---|---|
| Propagators and contour integration | `TestPropagatorAnalytic`, `TestConvergenceOrderLite`, `TestPartitionFunctionCheck` | Analytic solutions, refinement orders, forward/backward consistency, material conservation |
| Stress and cells | `TestStress*`, `TestScftObliqueSpaceGroups` | Finite differences of the free energy, constrained cells, monoclinic and triclinic groups |
| Platforms and transforms | `TestSpaceGroupPlatformEquivalence`, `TestStressComplexPlatforms`, `TestCnAdiLongLines`, `TestCrysFFT*`, `TestFourierTransform` | CUDA against the CPU, NumPy/SciPy and fresh solvers |
| Structure functions | `TestStructureFunctionRPA`, `TestChargedStructureFunctionRPA`, `TestStructureFunctionRecords` | RPA, recomputation from the saved fields |
| Charged systems | `TestClftsElectrostatics`, `TestClftsPointIons`, `TestClftsGrandCanonical`, `TestScftElectrostatics`, `TestConfinedElectrostatics` | Finite differences of H, Poisson-Boltzmann, Debye-Hückel statistics, mean-field limits |
| CL-FTS integrators and operators | `TestClftsEtd`, `TestClftsThermodynamics`, `TestGibbsEnsemble` | Exact linear dynamics, −dH/dV, Flory-Huggins |
| CL convergence diagnostics | `TestDriftStepHistory`, `tests/scripts/TestDriftTail.py` | Synthetic tails, the one-pole model with a known answer |
| Restarts, failures, extensions, examples | `TestLftsRun`, `TestFtsReliability`, `TestCheckpointValidation`, `TestExtensions`, `TestExamples`, `TestSamplingStatistics` | Bitwise continuation, corrupt checkpoints, extension contracts, known variances |

**Published results and other codes** (details in [Comparison with published results and other codes](#comparison-with-published-results-and-other-codes) and [Differences and limits](#differences-and-limits)):

| Reference | Method | Result |
|---|---|---|
| PSCF (pscfpp v1.4.0) | SCFT | Agrees to machine precision |
| Matsen et al. 2026 | L-FTS, CL-FTS | Agrees: structure-function peaks within about 1 % |
| BioFTS | CL-FTS | Agrees within the errors; the same operators to 10⁻¹² |
| polycomp | SCFT, CL-FTS | Agrees within the errors; the same operators to 10⁻¹⁴ |
| Fredrickson and Delaney 2023 | CL-FTS | Agrees within the reading precision (0.1–0.2 %) |
| Lequieu 2024 | CL-FTS | Agrees, except the excess pressure of the polyampholyte (not confirmed) |
| Netz 2001 | SCFT, CL-FTS | Agrees; CL-FTS up to Ξ ≈ 0.6 |
| Wang 2010; Agrawal and Wang 2022 | CL-FTS | Agrees with the one-loop theory (0.1 %) and exact Monte Carlo |
| Lee et al. 2008 | CL-FTS | Structure factors agree; the phase diagram of Fig. 6 is not reproduced (explained) |
| McCarty et al. 2019 | CL-FTS | Agrees, except the density structure function, 3–8 % low (undecided), and two state points inside the two-phase region (explained) |
| Danielsen et al. 2019 | CL-FTS | Converged values 3.4–5 % lower; with the paper's contour step the gap closes at E = 4000 and shrinks to 2 % at E = 2000 (explained) |

The cause of each difference from a published result is marked here and in [Differences and limits](#differences-and-limits):
- **Explained**: the cause is identified and supported by a calculation.
- **Not confirmed**: a likely cause that the published information cannot confirm.
- **Undecided**: the published data cannot tell whether the difference is real.

## Comparison with published results and other codes

Results of the same model computed by other codes or reported in papers, reproduced with polymerfts. Agreement between two complex Langevin codes checks the implementations, not whether complex Langevin dynamics converges to the averages of the field theory ([CL-FTS theory, Section 3](../theory/ComplexLangevinSimulation.md#3-correct-convergence)).

| Reference | System | Agreement | Details |
|---|---|---|---|
| PSCF (pscfpp v1.4.0) | SCFT of the phases of the phase library | Densities and fields within machine precision; free energies after the conversion of reference-chain units and species standard states | [phases/README.md](../../examples/block_copolymer/scft/phases/README.md) |
| Matsen, Ramírez, Willis and Pina, *J. Chem. Phys.* **2026**, 164, 014905 (Fig. 3) | L-FTS and CL-FTS structure functions of a symmetric diblock melt (discrete chains, N = 90, N̄ = 10⁵), contact and smeared interactions | Peak heights within about 1 % | [Langevin FTS theory, Section 4.4](../theory/LangevinFieldTheoreticSimulation.md#44-rpa-reference) |
| BioFTS (Wessén, Das, Pal and Chan, *J. Phys. Chem. B* **2022**, 126, 9222; commit b82f6e01 with the corrections of its issue 1) | Homopolymer solution and the polyampholyte E5K5E5K5 without salt and with canonical and grand-canonical point ions (N = 20, Edwards excluded volume, Coulomb channel) | Same fields: all operators to 10⁻¹²; same noise: the same trajectory. CL-FTS: chemical potentials of chains and ions, salt uptake, H and the structure functions (within 0.3–0.9 %) agree within the errors, as does the osmotic pressure in the virial form (polyampholyte with salt: 3.7299 ± 0.0004 against 3.7298 ± 0.0004 over time steps δτN = 0.08–0.4; all four systems within 0.7 standard errors) | [CL-FTS theory, Section 7](../theory/ComplexLangevinSimulation.md#7-chemical-potential-and-osmotic-pressure), [Structure Function Estimators, Section 7](../theory/StructureFunctionEstimators.md#7-validation) |
| Fredrickson and Delaney, *Field-Theoretic Simulations in Soft Matter and Quantum Fluids* (Oxford, 2023), Figs. 5.6, 5.7, 5.8 and 7.14 | Homopolymer in implicit good solvent (three box sizes) and a disordered diblock melt, CL-FTS | Excess chemical potentials and pressures within the reading precision of the figures (about 0.1–0.2 %) | [Example validation](ExampleValidation.md#blends-and-coexistence), `HomopolymerSolutionBook.py` |
| Lequieu, *Macromolecules* **2024**, 57, 10870 (Figs. 1, 2, 4 and 5) | Pressure and chemical potentials of soft-bead homopolymer solutions, diblock melts, a polyampholyte solution and a polyelectrolyte gel with explicit solvent and salt, which the paper computes by molecular dynamics and CL-FTS | Chemical potentials within 0.003–0.02 kT (fluctuation corrections 0.04–9.6 kT) and pressures within about one standard error, except the polyampholyte excess pressure, 2 % of its fluctuation correction below the paper and dependent on the time step; a Gaussian-core fluid agrees with an exact particle Monte Carlo simulation | [Example validation](ExampleValidation.md#block-copolymers), `SoftBeadDiblockMelt.py` |
| Netz, *Eur. Phys. J. E* **2001**, 5, 557 (Eqs. 63–69 and the loop expansion, Eqs. 41–47) | Counterions between two charged plates | SCFT: the Poisson-Boltzmann profile and pressure within 10⁻³ and 3×10⁻⁵, converging as Δx². CL-FTS at Ξ ≤ 0.2: the loop expansion evaluated with the Green function of the closed box within 1–4 % of the fluctuation correction, and an exact lattice Monte Carlo within its errors (agreement up to Ξ ≈ 0.6; CL-FTS fails at Ξ ≥ 0.85) | [Example validation](ExampleValidation.md#charged-systems), `scft/CounterionsBetweenPlates.py`, `clfts/CounterionsBetweenPlates.py` |
| Wang, *Phys. Rev. E* **2010**, 81, 021501; Agrawal and Wang, *Phys. Rev. Lett.* **2022**, 129, 228001 | Born self-energy and Debye-Hückel screening of Gaussian-smeared ions in a bulk salt | CL-FTS within 0.1 % of the one-loop theory at weak coupling; the papers' closed forms differ from the model's one loop by their approximations (point-charge screening, aκ → 0), which the comparison quantifies; an exact Monte Carlo (Widom insertion) agrees up to l_Bκ = 4.8 | [Example validation](ExampleValidation.md#charged-systems), `SmearedSaltSelfEnergy.py` |
| polycomp (rotskoff-group/polycomp, commit 2789769e with the corrections of its issue 23) | Homopolymer and polyelectrolyte solutions with point ions (3D CL-FTS) and neutral and charged diblock lamellae (1D mean field), smeared implicit-solvent models of continuous chains | Q, μ and H on the same complex fields to 10⁻¹⁴; mean-field lamellae to 10⁻⁶; CL-FTS pressures and chemical potentials within the statistical errors (fluctuation corrections resolved at 100–600 standard errors), also H and the density structure functions | [CL-FTS theory, Section 7](../theory/ComplexLangevinSimulation.md#7-chemical-potential-and-osmotic-pressure) |
| Lee, Popov and Fredrickson, *J. Chem. Phys.* **2008**, 128, 224908 (Figs. 4 and 5) | Density and charge structure factors of a salt-free polycation/polyanion solution (complex coacervate) in implicit solvent, CL-FTS | Within the precision of the figures at E = 900–19600, where S_nn rises up to 2.7 times above the RPA and S_cc falls 2–4.5 % below it. The phase diagram of Fig. 6 is not reproduced: its hysteresis points are metastability limits of a small box (see the example's notes) | [Example validation](ExampleValidation.md#charged-systems), `ComplexCoacervate.py` |
| McCarty, Delaney, Danielsen, Fredrickson and Shea, *J. Phys. Chem. Lett.* **2019**, 10, 1644 (Fig. 2, Fig. 3B and Fig. S3) | CL-FTS of sequence-defined polyampholytes (sequences sv10–sv30, discrete chains N = 50, smeared charges) in implicit solvent, mapped onto the compressible model with the reference density equal to the polymer density | Chemical potentials at infinite dilution within 0.01–0.07 kT of the paper's CL-FTS, at weak and at strong coupling, where they lie up to about 20 kT below the RPA; chemical potentials and osmotic pressures of finite concentrations and the binodal of sv30 (dense branch 4.91 against 4.93) within the precision of the figures; the density structure function S_ρ(2π/L) at ρb³ = 1.47 2.9 % below the paper's point for sv30 (2.5 standard errors) and 4–8 % below for sv10–sv25 (the paper's points are given without errors). Exact single-chain Monte Carlo agrees with the CL-FTS chemical potential extrapolated to an infinite box. The curves of Fig. 2 labelled l_B = 0.33 b were computed at l_B = 0.195 b | [Example validation](ExampleValidation.md#charged-systems), `PolyampholyteSolution.py` |
| Danielsen, McCarty, Shea, Delaney and Fredrickson, *Proc. Natl. Acad. Sci. USA* **2019**, 116, 8224 (Fig. 1) | Binodals of the block polyampholyte A50C50 in implicit solvent (B = 1), CL-FTS | With the paper's contour integration, second-order operator splitting at Δs = 0.01 (`rk2`; its SI, Section 2), the dense branch is C = 1.284 ± 0.007 against 1.31 at E = 2000 and 2.025 ± 0.011 against 2.01 at E = 4000. Converged contour integration (`rqm4`) puts it at 1.241 ± 0.003 and 1.942 ± 0.004, 5 % and 3.4 % below the paper; the box (4 and 6 R_g), the grid (24³ to 48³ points), the time step (δτN = 0.25 to 2) and Δs = 0.005 change it by at most 1.4 %. All values include the pressure of the dilute branch, about 0.004 at E = 2000. The one-loop theory reproduces the paper's RPA lines | [Example validation](ExampleValidation.md#charged-systems) |

## Differences and limits

This section collects the differences from the published results, with their causes as far as they are known, and the limits of CL-FTS that the comparisons above showed; the linked pages give the details.

**Exact references.** Complex Langevin averages were compared with exact averages of the same model in:
- the two-cell model of [CL-FTS theory, Section 3.2](../theory/ComplexLangevinSimulation.md#32-test-against-exact-averages), whose $`2^8`$ particle configurations are summed exactly; it relates the tail of the drift distribution to the bias;
- the one-pole model of Aarts et al. (`tests/scripts/TestDriftTail.py`, [CL-FTS theory, Section 3.1](../theory/ComplexLangevinSimulation.md#31-drift-records));
- Monte Carlo simulations of the same lattice models, point ions on the grid sites: a bulk salt up to $`l_B \kappa = 4.8`$, salt at uncharged walls up to $`l_B/a = 2`$ and counterions between charged plates up to $`\Xi \approx 0.6`$ ([CL-FTS theory, Section 6](../theory/ComplexLangevinSimulation.md#6-charged-systems));
- a Monte Carlo simulation of the particles of a Gaussian-core fluid, off the lattice (Lequieu);
- single-chain Monte Carlo of isolated McCarty polyampholytes, which the CL-FTS chemical potentials match once extrapolated in $`1/V`$ to an infinite box.

These cover small models and weak to moderate coupling. Agreement with another complex Langevin code checks the implementation, not the convergence of the method ([CL-FTS theory, Section 3](../theory/ComplexLangevinSimulation.md#3-correct-convergence)).

**Differences from published results.**

| Comparison | Difference | Cause | Details |
|---|---|---|---|
| Lequieu 2024 | Excess pressure of the polyampholyte 0.075 below the paper (2 % of its fluctuation correction), changing with the time step | Not confirmed. With the `etd` integrator polymerfts gives the paper's 1.388 at $`\delta\tau N = 1`$ and 1.332 at 0.5, so the paper's value may keep a time-step bias. The paper used another integrator (Euler-Maruyama predictor-corrector, time step 0.01–0.1, mobilities 1) and gives neither the step of this system, an extrapolation to zero step, nor errors | [Example validation](ExampleValidation.md#block-copolymers) |
| Danielsen et al. 2019 | Dense branch of the A50C50 binodal with converged contour integration 5 % (E = 2000) and 3.4 % (E = 4000) below the paper's CL-FTS | Explained: the paper integrated the contour by second-order operator splitting at Δs = 0.01 (its SI, Section 2). There the `rk2` pressure is 0.05 to 0.07 low, an error of order Δs² that is large because one contour step (0.1 R₀) is longer than the smearing length (0.082 R₀); with `rk2` at Δs = 0.01 polymerfts gives 1.284 ± 0.007 and 2.025 ± 0.011 against the paper's 1.31 and 2.01. The gap closes at E = 4000 and leaves 2 % at E = 2000, about the scatter of the paper's markers | [Example validation](ExampleValidation.md#charged-systems) |
| Lee et al. 2008, Fig. 6 | Phase diagram not reproduced: the paper's superheating and supercooling points are limits of metastability in a box of $`(4 R_g)^3`$, not the binodal | Explained: the paper's points are not the binodal; the CL-FTS binodal (C = 17.3 ± 0.1) lies close to the one-loop binodal on the same lattice (17.7) | [Example validation](ExampleValidation.md#charged-systems) |
| McCarty et al. 2019 | Density structure function S_ρ(2π/L) at ρb³ = 1.47 2.9 % (sv30) to 4–8 % (sv10–sv25) below the paper's points, which have no error bars | Undecided: the paper's points have no error bars; all sequences lie below them, sv30 by 2.5 standard errors of polymerfts, so the offset may be common to all; not investigated further | [Example validation](ExampleValidation.md#charged-systems) |
| McCarty et al. 2019, two-phase points | Chemical potential of sv30 at l_B = 0.195 b and ρb³ = 0.015 and 0.059, inside the two-phase region, differs from the paper (βμ_ex ≈ −15 at ρb³ = 0.059) | Explained: the boxes hold 0.8 and 3 chains on average, which aggregate, while the paper's values follow the homogeneous solution | [Example validation](ExampleValidation.md#charged-systems) |

**Limits of CL-FTS found in these comparisons.**
- Counterions between charged plates: at $`\Xi = 0.85`$ the contact density is 4 % too high even as $`\delta\tau \to 0`$, and at $`\Xi \ge 1.6`$ the profile is qualitatively wrong. The strong-coupling regime of counterions is out of reach ([CL-FTS theory, Section 6](../theory/ComplexLangevinSimulation.md#6-charged-systems)).
- The step of $`\psi`$ has an $`O(\delta\tau)`$ bias that grows with the charge response of the ions; beyond weak coupling the point-ion systems needed $`\delta\tau N \le 0.25`$ (same section).
- A one-segment species that couples to an auxiliary field, such as the reference solvent of point ions with $`\zeta N > 0`$, drives its partition function towards zero below about 0.01 molecules per grid cell, and the run diverges; such a solvent takes `zeta_n: 0` (same section, "Point ions alone"). Point ions, which couple only to $`\psi`$, run at 0.002–0.006 ions per cell in `clfts/SmearedSaltSelfEnergy.py` and `clfts/CounterionsBetweenPlates.py`.

## Regression tests

| Check | Existing or added test | Scope |
|---|---|---|
| Reaction and diffusion against an analytic solution | `TestPropagatorAnalytic` | Constant potential with a nonzero Fourier-mode initial condition; RQM4/RK2 on available backends |
| Contour refinement | `TestConvergenceOrderLite` | RQM4 and RK2 asymptotic orders on a nonuniform field |
| Forward/backward consistency | `TestPartitionFunctionCheck`, propagator tests | Every contour pairing yields the same partition function |
| Material conservation | Propagator tests, `TestSpaceGroupPlatformEquivalence` | Volume-averaged total concentration; expand reduced fields or use orbit weights |
| Analytical-limit structure factors | `TestStructureFunctionRPA`, `TestChargedStructureFunctionRPA` | Disordered Gaussian-fluctuation regime, L-FTS/CL-FTS with and without smearing, and charged CL-FTS with counterions; RPA formulas in [Section 4.4 of the Langevin FTS theory](../theory/LangevinFieldTheoreticSimulation.md#44-rpa-reference) |
| Recorded structure functions | `TestStructureFunctionRecords` | One sample of S_ij against a recomputation from the saved fields on every platform: L-FTS of an ABC triblock in 3D (periodic with smearing, reflecting walls) and CL-FTS with per-type smearing (periodic, reflecting walls, whose record holds Fourier coefficients) |
| Electrostatics | `TestClftsElectrostatics` | Neutral reductions and equal per-type smearing lengths (bitwise), forces of ψ and the auxiliary fields against finite differences of H, the ψ step against the exact flow of its force and the stationary variance of ψ (without charges and with a linear charge response), recorded S_ij (per-type smearing) and ψ drift against a recomputation from the saved state, restarts with ψ and its noise stream; with screening, the ψ force along the uniform mode, the stationary variance of every mode and a net charge |
| Point ions | `TestClftsPointIons` | Automatic counterions and input checks, forces of ψ and the auxiliary fields against finite differences of H with the ion term, the linear charge response of the ions, ion chemical potentials and records, bitwise restarts, the osmotic pressure against -dH/dV at fixed numbers of ions (with and without screening), and the Debye-Hückel statistics of ψ with the ion screening (etd, with and without screening) |
| Grand-canonical species | `TestClftsGrandCanonical` | Input checks, canonical and grand-canonical homopolymer on the same fields, forces of ψ (uniform mode included) and the auxiliary fields against finite differences of H with -e^(βμ) Q, the Donnan potential, Flory-Huggins density, chemical potentials, numbers of chains and ions and pressure in the mean-field limit (LM and etd), the osmotic pressure against -dH/dV at fixed chemical potentials (with and without screening), bitwise restarts |
| Mean-field electrostatics | `TestScftElectrostatics` | SCFT with ψ from the Poisson equation: neutral reduction, strong-screening limit against the neutral diblock with χ_eff (1/κ² convergence and Richardson extrapolation), stress against dF/dL with canonical and grand-canonical ions, Donnan and screened homogeneous states, canonical against grand-canonical ions, a fixed charge with `axis_ordering` against the standard order, platform agreement |
| Gibbs ensemble | `TestGibbsEnsemble` | Mean-field limit: Flory-Huggins binodal of a blend, exchange potentials of charged chains with canonical ions at an independently solved coexistence (neutral combinations) and the approach to it from distant boxes, charged chains with grand-canonical salt at an independently solved Donnan coexistence and the approach to it; input checks; bitwise continuation |
| Closed boxes and fixed charges | `TestConfinedElectrostatics` | ψ of SCFT and of the CL-FTS mean-field limit against an independent Poisson-Boltzmann solution between charged layers in a closed box (cosine basis), CL-FTS forces against finite differences of H with walls, a fixed charge and point ions, bitwise restarts, SCFT stress and CL-FTS pressure with a fixed charge in a periodic box, SCFT stress in a closed box against dF/dL, charged chains grafted to a wall |
| Linear-response integrator | `TestClftsEtd` | Exact flow and exact stationary covariance for a synthetic linear force that couples real and imaginary fields, far beyond the LM stability limit; LM step for a zero Jacobian; the complex-step Jacobian of a charged system against central differences of the forces (ψ included); bitwise restarts; a field dropped at a zero eigenvalue stays fixed |
| Thermodynamic operators | `TestClftsThermodynamics` | Osmotic pressure against -dH/dV at fixed grid values (smearing types, screened and unscreened charges), Flory-Huggins chemical potentials and pressure at the homogeneous saddle point (dilation and virial forms), the Schwinger-Dyson terms of the virial form averaging to zero in CL runs, records against a recomputation from the saved state, bitwise restarts of pending records |
| Fourier transform | `TestFourierTransform` | `PropagatorSolver.fourier` on every platform against NumPy/SciPy: transforms of real and complex fields (periodic in 1D-3D, cosine and sine series in 1D-3D, a cosine axis next to a sine axis), \|k\|² on orthogonal and oblique cells (minimal alias) and after a box change, diagonal and matrix multipliers summed over terms, real-field multipliers and coefficients that are not Hermitian, several real fields with an odd number of grid points; CUDA against the CPU |
| Stress derivatives | `TestStress1D`, `TestStressSpaceGroup`, `TestStressNonPeriodic*`, `TestStressAliasTies` | Lengths, angles, constrained cell deformations and boundary conditions |
| CPU/GPU agreement | `TestSpaceGroupPlatformEquivalence`, `TestStressComplexPlatforms`, `TestCnAdiLongLines`, `TestCrysFFTCellChange` | Same input fields, not independently translated/rotated converged phases; masks with every continuous method; CN-ADI lines beyond the CUDA shared-memory limit; CrysFFT solvers after a cell change (one axis alone) against fresh solvers |
| Monoclinic and triclinic groups | `TestScftObliqueSpaceGroups` | SCFT with P2/m (unique axes b and c), P2_1/c, C2/c and P-1: the free angles of box relaxation, all six stress components against the full grid on every platform, and relaxed cylinders against the same relaxation without a group |
| CrysFFT selection | `TestCrysFFTSelector` | The CrysFFT each space group, grid and cell takes; equivalence tests alone pass whatever is chosen |
| Examples | `TestExamples` | Every script under `examples/` reads its inputs and builds its simulation (stopped at its first run) |
| Restart equivalence | `TestLftsRun`, `TestLftsDetails`, `TestFtsReliability` | RNG, previous noise, fields, WTMD and pending measurements across recording boundaries |
| Failure handling and compatibility | `TestFtsReliability`, `TestCheckpointValidation` | Initial/terminal failure, unchanged RNG/extension/WTMD state on corrupt restarts, atomic-write failure, earlier WTMD checkpoints loaded into a reused simulation |
| Extension contracts | `TestExtensions` | Harmonic stochastic moments, read-only observers and sampling intervals, multistage forces, extension-state restart and method registry on available backends |
| Sampling error estimates | `TestSamplingStatistics` | Independent Gaussian draws and an AR(1) process with known asymptotic variance |
| CL convergence diagnostics | `TestDriftStepHistory`, `TestClftsPointIons`, `tests/scripts/TestDriftTail.py` | Per-step drift and partition-ratio records (definition, bitwise restarts, checkpoint checks); `drift_tail.py` on synthetic tails and on the one-pole model of Aarts et al. (2017), where it separates the wrong regime (power-law tail, Re⟨z²⟩ wrong while Im⟨z²⟩ and the Schwinger-Dyson identities vanish) from the correct one |

The Python sampling/restart tests do not replace long statistical checks. Existing RPA tolerances remain unchanged, except that the polyampholyte charge structure function of `TestChargedStructureFunctionRPA` must lie within 1.5 % of the RPA on average (it was 5 %), so that the former ψ step, 2 % above, fails; that test estimates standard errors from block means and requires its tolerances to exceed 2 of them. Its continuum reference applies in its specified weak-fluctuation regime; finite contour/grid errors and finite fluctuation corrections must be distinguished from sampling uncertainty.

## Running regression validation

After building, `ctest --test-dir build -L numerical --output-on-failure` runs the selected numerical regressions. Basic and full test selections remain available. Missing CPU backends can skip CPU-specific tests. The GitHub-hosted workflow builds without CUDA, on Linux (all tests) and on Windows with MSVC (the basic tests); its class-name consistency check does not execute GPU numerics.

Run the selected tests and the unmodified continuous-chain Gyroid example on
allocated compute resources. Set up the environment and GPU allocation using
your site's scheduler, and set `PYTHONPATH` to the staged build package when
validating uninstalled changes. For example, inside the allocation:

```bash
ctest --test-dir /path/to/polymerfts/build -L numerical --output-on-failure \
    --output-junit /path/to/report/numerical.xml
cd /path/to/report
python /path/to/polymerfts/examples/block_copolymer/scft/Gyroid.py > gyroid.log 2>&1
```

For CPU/GPU agreement, the build must include both backends and the job must
have a visible GPU. Preserve JUnit results, the CMake cache, environment details,
GPU information and the Gyroid convergence result in a distinct report directory
per job. Without a visible GPU, a build with the CUDA backend fails every test
with a CUDA part, C++ and Python alike, after its CPU parts have run: a skipped
CUDA part would read as validated GPU agreement. Submit long checks and parameter
sweeps through the scheduler, one configuration per job. Keep cluster-specific
launchers, paths and module settings outside the repository.

To run the full suite, omit `-L numerical` from the CTest command after building
the required targets.

## Statistical refinement

The test harness `tests/validation/validate_fts_sampling.py` runs one configuration and writes post-equilibration Hamiltonian estimates, Monte Carlo standard errors, effective sample sizes, block structure factors and continuum RPA references. CL-FTS also reports imaginary observables. Scaled drift maxima are summarized by quantiles, including the maximum. The full histories and run status remain in the adjacent `_data` directory. Existing output directories are rejected to prevent mixing independent runs.

`tests/validation/validate_etd_order.py` measures the time-step error of stationary averages for the Langevin integrators (Euler, LM, `etd` and, for comparison, exponential time differencing with fresh noise) on independent one-dimensional trajectories with an exactly known $`\langle x^2 \rangle`$, one configuration per invocation; `--collect` prints the table of a file of results ([CL-FTS theory, Section 2.1](../theory/ComplexLangevinSimulation.md#21-exponential-time-differencing)).

```bash
python tests/validation/validate_fts_sampling.py --model lfts --dt 0.5 \
    --seed 7 --steps 20000 --burn-in 4000 --block 1000 \
    --output /path/to/report/lfts-dt05-seed7.json
```

Run this command inside an allocated compute job with resources appropriate to
the selected backend. For GPU sampling, add `--platform cuda` and request a GPU
through the scheduler. The script draws chain diagrams inside its output
directory. It never changes the calibrated tests.

Compare separate jobs along these axes:

- Time step: three decreasing values. Keep comparable post-equilibration Langevin time; increase step counts and burn-in when reducing `dt`.
- Grid: increase `--nx` at fixed physical box and model; then refine `--ds` separately.
- Independent realizations: repeat each comparison with multiple `--seed` values. Account for both within-run autocorrelation and between-run variation.
- Saddle accuracy: tighten `--saddle-tolerance`; report `saddle_failures` and `noise_resamples`. The default retry policy consumes one new noise per accepted proposal attempt, even when its saddle solve is repeated. Explicit resampling studies assess the former policy and cannot establish that it is unbiased merely because failure is rare.
- CL-FTS: vary `--alpha` toward zero, inspect imaginary observables and drift tails, and cross-check in regimes where an independent reference is applicable. Bounded trajectories alone do not establish correct sampling, and agreement with another complex Langevin code checks the implementation, not the convergence of the method.

The ESS estimator uses an initial positive sequence of paired autocorrelations and conservatively caps ESS at the number of samples. Structure-factor uncertainty is estimated from saved block averages, so at least four full post-burn-in blocks are required. These are approximate error estimates, not automatic equilibrium or complex-Langevin correctness certificates. Persistent shifts under refinement require investigation even when one comparison falls within its error bars.

An initial CPU validation on 2026-10-22 used 20 separate jobs: L-FTS time steps 0.5/0.25/0.125, CL-FTS time steps 1/0.5/0.25, two independent seeds per time step, 32/64 grid points, contour steps 0.02/0.01, and CL stabilization 0/0.01/0.05. The box length was 4, chi N was 6 and nbar was one million; post-equilibration Langevin time was held fixed within each model's time-step comparison. All completed. The four-mode mean ratio S_AA/RPA ranged from 0.9755 to 1.0168; every individual mode was within 8 percent of RPA. No saddle failures occurred, so this pilot does not assess noise-selection bias under actual failures. The raw Hamiltonian differed between grids; agreement of these low-wavevector structure factors does not establish a continuum limit for every observable or validate strong-fluctuation CL-FTS.

## Provenance and continuation

Simulation files contain `simulation_metadata_json`; FTS also writes `run_status.json` after success or an exception. Metadata identifies the loaded native extension's embedded build version/commit/modification flag, compiler and toolkit flags, binary and Python-source hashes, runtime numerical-library versions, selected backend/method/precision, CPU/GPU information, threading environment, input hash and seed. Initial fields have a separate hash. A resumed run records its parent checkpoint and preserves the originating seed.

The embedded Git information (commit and whether `CMakeLists.txt`, `cmake/` or `src/` differ from it) is refreshed at every build; an unchanged state recompiles nothing. Hashes identify the actual installed artifacts even when a build is dirty or the simulation runs outside a checkout. Preserve the corresponding source/wheel and environment/build files; recording versions alone does not recreate them. `run_status.json` records the final execution outcome; periodic checkpoints describe an in-progress run at their saved accepted step.

Versioned checkpoints compare effective model and sampling settings before mutating the RNG. Output location, final step and backend are not part of physical compatibility. Same-backend deterministic regression checks auxiliary fields, concentrations and pending observables exactly; cross-backend continuations can differ by floating-point rounding, and so can `cpu-fftw` continuations in a new process, because FFTW chooses its transform plans by timing them (the restart tests continue within one process). Unversioned files require explicit legacy opt-in and cannot recover measurements that were never stored.

## Performance measurements

Keep accuracy tests and performance measurements separate. Propagator timings
and fixed-iteration SCFT timings measure different work; neither alone measures
time to convergence. Record provenance and input hashes with timing results.

Report problem geometry, architecture, contour discretization, field input, hardware, selected backend/method, precision, compilation options, threads/streams, reduced basis and memory mode. State warm-up, repetitions, timing boundaries and variability. Compare total SCFT time at matched residual/accuracy, and sampling efficiency at matched bias and Monte Carlo error. Cross-project comparisons also match physical units, ensemble, bead/bond conventions, free-energy offsets and Fourier normalization.
