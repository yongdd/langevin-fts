# Examples

Example scripts, grouped first by physical system and then by simulation method:

- `scft/`: self-consistent field theory, the mean-field saddle point of the field-theoretic Hamiltonian.
- `lfts/`: Langevin field-theoretic simulation, compositional fluctuations with the pressure-like fields at their saddle point.
- `clfts/`: complex Langevin simulation, in which every field fluctuates. It is needed where the partial saddle-point approximation of L-FTS does not hold, for example for charged systems.

| System | SCFT | L-FTS | CL-FTS |
|--------|------|-------|--------|
| [Block copolymers](#block-copolymers-block_copolymer) | `block_copolymer/scft/` | `block_copolymer/lfts/` | `block_copolymer/clfts/` |
| [Blends and coexistence](#blends-and-coexistence-blend) | `blend/scft/` | `blend/lfts/` | `blend/clfts/` |
| [Charged systems](#charged-systems-charged) | `charged/scft/` | — | `charged/clfts/` |
| [Tools](#tools-tools) | `tools/` (transforms and extension APIs) | | |

## Block copolymers (`block_copolymer/`)

| Example | Description |
|---------|-------------|
| `scft/Lamella3D.py` | AB diblock lamellar phase |
| `scft/Cylinder3D.py` | Cylindrical phase |
| `scft/Gyroid.py` | Gyroid phase with box optimization |
| `scft/GyroidNoBoxChange.py` | Gyroid without box optimization (discrete chains, space group) |
| `scft/ABA_Triblock1D.py` | Symmetric ABA triblock copolymer |
| `scft/ABC_Triblock_Sphere3D.py` | ABC triblock terpolymer spheres |
| `scft/Star3Arms1D.py` | 3-arm star polymer (1D) |
| `scft/Star9ArmsGyroid.py` | 9-arm star polymer gyroid |
| `scft/BottleBrushLamella3D.py` | Bottle-brush copolymer |
| `scft/phase_diagram/` | Phase diagram of a system in two parameters by continuation and boundary root finding; AB diblock and ABA triblock set up (GPU) |
| `scft/crystal_systems/` | Box relaxation in non-orthogonal cells (`crystal_system`): cylinders in hexagonal and oblique 2D cells and in a triclinic cell, a lamella in a monoclinic cell, and the free energy against the angle of fixed cells (`AngleScan2D.py`) |
| `lfts/Lamella.py` | Lamellar phase with fluctuations (discrete chains) |
| `lfts/Gyroid.py` | Gyroid with fluctuations (discrete chains) |
| `lfts/ABC_Triblock_Sphere3D.py` | ABC triblock with fluctuations |
| `clfts/Lamella.py` | Lamellar phase from complex Langevin dynamics; `DYNAMICAL_STABILIZATION.md` and `SMEARING.md` explain its stabilization options |

**Phase library** (`scft/phases/`, space-group symmetry):

- **Lamellar**: `Lamella1D.py`
- **Cylindrical**: `Cylinder2D.py`
- **Gyroid**: `SG.py` (single gyroid), `DG.py` (double gyroid)
- **Spherical**: `BCC.py`, `FCC.py`, `HCP_Hexagonal.py`, `SC.py`, `A15.py`, `Sigma.py`
- **Other**: `DD.py` (double diamond), `PL_Hexagonal.py` (perforated lamella), `Fddd.py`, `SD.py`, `SP.py`, `DP.py`

## Blends and coexistence (`blend/`)

| Example | Description |
|---------|-------------|
| `scft/MixtureBlockRandom.py` | Blend of a block copolymer and a random copolymer |
| `scft/CanonicalCoexistence.py` | Binodal of a homopolymer blend from canonical SCFT and analytic Flory-Huggins states (`find_binary_coexistence`) |
| `lfts/MixtureBlockRandom.py` | The same blend with fluctuations |
| `clfts/GibbsEnsemble.py` | Coexisting phases of a homopolymer blend from two CL-FTS boxes that exchange chains and volume (`GibbsEnsemble`) |

### Validation

`TestExamples` builds every example (inputs, input checks and construction) but does not run it. What the tests check beyond that, and what has not been checked:

- **`scft/MixtureBlockRandom.py`, `lfts/MixtureBlockRandom.py`**
  - Checked: the random copolymer in the homogeneous SCFT thermodynamics, against Flory-Huggins (`TestPhaseEquilibrium`).
  - Not checked: the ordered blend against a reference.
- **`scft/CanonicalCoexistence.py`**
  - Checked (`TestPhaseEquilibrium`): the coexistence solver against the common tangent of the Flory-Huggins free energy; homogeneous SCFT states against Flory-Huggins within 10⁻¹²; the exchange chemical potential of a modulated diblock/homopolymer blend against dF/dφ. The binodal of this example is therefore the Flory-Huggins binodal.
  - Not checked: a binodal with a modulated phase against an independent calculation.
- **`clfts/GibbsEnsemble.py`**
  - Checked (`TestGibbsEnsemble`, small boxes of a symmetric blend at χN = 3 with N = 20): in the mean-field limit N̄ = 10²⁸, the Flory-Huggins binodal within 2×10⁻⁶ and the coexistence of charged chains with canonical ions and with grand-canonical salt (Donnan potentials) against independent solutions within 10⁻⁶; at N̄ = 10⁴, bitwise continuation of a run.
  - Checked (validation runs with the settings of the example): in the mean-field limit the boxes reach the Flory-Huggins binodal. At N̄ = 10⁴, where fluctuations shift the binodal by −0.014, the Gibbs ensemble agrees with an independent route: canonical CL-FTS runs of one box at fixed compositions, whose exchange chemical potential βμ_A − βμ_B vanishes at the binodal of this symmetric blend.
  - Not checked: published fluctuating binodals; the dependence on `exchange_rate` and `cycle_steps`; finite-size effects (both routes use the same box).
  - The method follows Riggleman and Fredrickson (*J. Chem. Phys.* **2010**, 132, 024104), but each box is held at the reference density, so the pressures of the boxes need not agree ([CL-FTS theory, Section 8](../docs/theory/ComplexLangevinSimulation.md#8-gibbs-ensemble)).

![Compositions of the Gibbs boxes and the exchange chemical potential of canonical CL-FTS](../docs/figures/gibbs_ensemble_blend.png)

| Volume fraction of A in the A-rich phase | Value |
|------------------------------------------|-------|
| Flory-Huggins binodal | 0.92928 |
| `GibbsEnsemble`, N̄ = 10²⁸ (after 40 cycles) | 0.92928 |
| `GibbsEnsemble`, N̄ = 10⁴, 400 cycles (second half) | 0.91497 ± 0.0001 |
| `GibbsEnsemble`, N̄ = 10⁴, the example (40 cycles) | 0.9149 |
| Canonical CL-FTS, N̄ = 10⁴: zero of βμ_A − βμ_B | 0.91497 ± 0.00013 |

Settings of the example: 16 × 16 grid, boxes of 2 R₀ × 2 R₀ (the area of each box stays within about 1 % of 4 R₀²), discrete chains with N = 32, χN = 3, ζN = 100, δτN = 0.5. The canonical runs take 200,000 steps at each of eight compositions φ = 0.86–0.96, recorded in 20 blocks of which the first is left out. Their exchange chemical potential is fitted with ln(φ/(1 − φ)) plus a quadratic in φ (largest residual 9×10⁻⁶). All compositions use the same random seed, so their noise is correlated: the zero is fitted block by block, and its error is the spread of the 19 zeros. The error of the Gibbs ensemble comes from blocks of 20 cycles.

## Charged systems (`charged/`)

Smeared Coulomb interactions with an electrostatic potential ψ (`electrostatics`), in SCFT and CL-FTS; L-FTS does not take `electrostatics`.

| Example | Description |
|---------|-------------|
| `scft/ChargedDiblockLamella.py` | Charged diblock lamella with point counterions: period and free energy over a range of added salt, against the neutral diblock |
| `scft/PolyelectrolyteBrush.py` | Charged brush grafted to a charged wall (`fixed_charge`) of a closed box with point ions: height and Donnan potential over a range of salt (real-space `cn-adi2` method) |
| `clfts/ChargedDiblock.py` | Charged diblock melt with counterions as one-segment molecules |
| `clfts/SaltReservoir.py` | Charged diblock melt in contact with a salt reservoir (grand-canonical point ions): uptake of ions against the Donnan equilibrium |

### Validation

`TestExamples` builds every example but does not run it. SCFT solves a deterministic mean-field problem. CL-FTS adds the question whether complex Langevin sampling converges to the correct averages, which has been checked for charged systems only where the fluctuations are weak ([CL-FTS theory, Section 6](../docs/theory/ComplexLangevinSimulation.md#6-charged-systems), "Checked range").

- **`scft/ChargedDiblockLamella.py`**
  - Checked (`TestScftElectrostatics`, mostly with essentially the system of this example: charged AB diblock with Na⁺ Cl⁻ point ions, discrete chains, N = 50, E = 2000): zero valences reproduce neutral SCFT; the stress equals dF/dL with canonical and with grand-canonical ions; the Donnan potential of homogeneous states; canonical and grand-canonical ions reach the same state; the platforms agree. With the oppositely charged blocks and continuous chains described below: under strong screening the free energy and the relaxed period approach those of the neutral diblock with χ_eff N within 10⁻⁵.
  - Checked (validation runs): the strong-screening limit below, and the convergence of the periods that the example prints in the grid. With 64, 128 and 256 grid points they agree to seven digits: 4.35597, 4.38527, 4.43950, 4.58712 and 4.78701 R₀ for c_salt = 0, 0.01, 0.03, 0.1 and 0.3, and 4.97436 R₀ for the neutral diblock. The contour step is fixed by the discrete chains (N = 50).
  - Not checked: the periods and free energies of the unscreened lamellae against another code or published results.

![Free energy and period of a charged lamella under strong screening](../docs/figures/charged_lamella_screening.png)

Strong screening: blocks with valences +0.5 and −0.5, continuous chains (Δs = 1/50), 64 grid points, χN = 25 and E = 10/λ² for screening lengths λ = 1/κ = 0.02 to 0.005 R₀. The Coulomb term then acts as the contact interaction −E (z_A − z_B)²/(2κ²) = −5, and the free energy and the relaxed period approach those of the neutral diblock at χ_eff N = 20 linearly in λ² (the ratio of the differences at λ = 0.01 and 0.005 is 3.99).

- **`scft/PolyelectrolyteBrush.py`**
  - Checked (`TestConfinedElectrostatics`, with discrete chains and pseudo-spectral propagators): ψ in a closed box against an independent Poisson-Boltzmann solution between fixed charge layers (10⁻⁹); the closed-box stress against dF/dL; charged chains grafted to a wall converge to the Poisson equation.
  - Checked (validation runs, figure and table below): (a) with `cn-adi2`, the method of this example, ψ between the charged layers of that test agrees with the independent Poisson-Boltzmann solution within 2.8×10⁻¹¹ (|ψ| ≤ 5.5); (b) the brush heights converge at second order in Δx and Δs, and those of the example are 0.9 % (c_salt = 0.1) to 1.6 % (c_salt = 0.001) above the extrapolated values; in a 4 R₀ box, where the pseudo-spectral `rqm4` still runs at high salt, `rqm4` with 512 points and Δs = 0.0025 agrees with the extrapolated `cn-adi2` heights within 3×10⁻⁵ R₀ (c_salt = 0.1 and 0.03), although its iteration stalls at errors of 5×10⁻⁸ to 10⁻⁷ and its check of the partition functions fails at c_salt = 0.01; (c) `rqm4` in the example's box puts a spurious second layer of A segments at x ≈ 4 R₀ and negative φ_A between the brush and that layer (down to −0.0014) and near the far wall (down to −0.0005); its iteration stalls at an error of 10⁻⁵ already at c_salt = 0.1, and at c_salt = 0.03 the check of the partition functions fails. The propagators of the grafted chains fall by many orders of magnitude across the box: the true φ_A drops from 0.37 to 4×10⁻¹⁷ at the far wall of the 6 R₀ box (c_salt = 0.1), but only to 10⁻⁷ in the 4 R₀ box. The round-off of the pseudo-spectral transforms, about 10⁻¹⁶ of the largest value of a propagator, grows into the spurious density where the propagators are small; the real-space `cn-adi2` method does not have it.
  - Not checked: the electrostatics with `cn-adi2` in a ctest; the brush heights against brush theory (osmotic and salted regimes) or published profiles.

![Poisson-Boltzmann check, convergence of the brush height and pseudo-spectral failure](../docs/figures/polyelectrolyte_brush_validation.png)

| Brush height ⟨x⟩ [R₀], example box | c_salt = 0.1 | c_salt = 0.001 |
|------------------------------------|-------------:|---------------:|
| Δx = 0.094 (64 points), Δs = 0.02 | 1.05194 | 1.28034 |
| Δx = 0.047 (128 points), Δs = 0.01 (the example) | 1.02733 | 1.23055 |
| Δx = 0.023 (256 points), Δs = 0.005 | 1.02049 | 1.21642 |
| Δx = 0.012 (512 points), Δs = 0.0025 | 1.01874 | 1.21278 |
| Richardson extrapolation | 1.01816 | 1.21157 |

Every run converged to the error 10⁻⁸ of the example, with relative partition-function errors below 10⁻¹³. The potential difference between the walls converges only at first order, since it is read in the first cell, which also holds the substrate charge: the example's −0.648 and −2.833 lie 5 % and 2 % from the extrapolated −0.681 and −2.881 (c_salt = 0.1 and 0.001).

- **`clfts/ChargedDiblock.py`**
  - Checked: the forces of ψ and of the auxiliary fields against derivatives of H, the exact step of ψ and its stationary variance, the neutral limits and bitwise restarts (`TestClftsElectrostatics`); the structure functions of a charged diblock with one-segment counterions against the RPA at N̄ = 10⁶ (`TestChargedStructureFunctionRPA`, figure below).
  - Not checked: the regime of the example (N̄ = 10⁴, χN = 12 from a lamellar start, E ≈ 10³), where the fluctuations are not Gaussian. No reference exists there, and whether complex Langevin sampling converges to the correct averages at this coupling has not been established.

![Structure functions of a charged diblock with counterions against the RPA](../docs/figures/charged_structure_function_rpa.png)

System of `TestChargedStructureFunctionRPA`: discrete AB diblocks with N = 20, f = 0.5, valence 0.2 on A and one-segment counterions of valence −1, χN = 6, ζN = 100, E = 10⁴, N̄ = 10⁶, 32 × 32 grid with L = 4 R₀, Gaussian smearing 0.1 (A, B) and 0.15 (counterions); 30,000 steps, 29 blocks after the first. Mean ratios to the RPA over the eight shells: S_AA 1.000 ± 0.009, S_BB 1.002 ± 0.006, S_AB 0.996 ± 0.010, charge structure function 1.008 ± 0.008. The Coulomb interaction lowers the charge structure function at small k to a few percent of its value without it.

- **`clfts/SaltReservoir.py`**
  - Checked: the forces with the grand-canonical terms, canonical and grand-canonical species on the same fields, the mean-field limit (Donnan potential, numbers of ions, chemical potentials, osmotic pressure), the pressure against −dH/dV and bitwise restarts (`TestClftsGrandCanonical`); the Debye-Hückel statistics of ψ with point ions at weak coupling (`TestClftsPointIons`).
  - Checked (validation run of the example, figure and table below): the ion concentrations lie below the Donnan values, by the same 4.0×10⁻⁵ per segment for both ions (neutrality fixes their difference), and the one-loop theory of ψ on the same grid predicts this shift to 0.5 %, within about two standard errors. In it each ion is weighted by ⟨e^(−z_s Γ_s ψ/N)⟩ = e^(−z_s² σ²/(2N²)), where σ² is the variance of the smeared potential from the Gaussian theory of ψ: (k² + κ²)/E plus the screening by the point ions and the RPA charge response of the chains, summed over the 32³ modes of the grid. The unscreened part, σ² = 20.1, is the self-energy of the smeared ions; screening lowers it to σ² = 14.3. The weight then enters the Donnan equilibrium.
  - Not checked: stronger coupling, where the one-loop theory no longer suffices.

![Ion concentrations of the salt-reservoir example against Donnan and one-loop theory](../docs/figures/salt_reservoir_ions.png)

| Ion concentration per segment | Na⁺ | Cl⁻ |
|-------------------------------|----:|----:|
| Donnan (mean field) | 0.0309017 | 0.0809017 |
| One-loop theory | 0.0308621 | 0.0808621 |
| CL-FTS (the example, 9 blocks of 2000 steps) | 0.0308619 ± 0.0000001 | 0.0808619 ± 0.0000001 |

## Tools (`tools/`)

| Example | Description |
|---------|-------------|
| `FourierTransform.py` | `PropagatorSolver.fourier`: `forward`/`backward` transforms, `k_squared`, multipliers stored with `set_multiplier` and applied with `apply` (smearing, the exact step of a linear Langevin equation, a matrix multiplier, complex fields), each checked against NumPy |
| `CustomExtensions.py` | A stateful observable (`callbacks=`) and a multistage integrator (`integrator=`) for L-FTS without editing the package ([Extensions](../docs/internals/Extensions.md)) |

## Running Examples

Run a script from its directory, which holds its input files; the scripts write their output into the current directory:

```bash
cd examples/block_copolymer/scft
python Lamella3D.py
```

For GPU acceleration (recommended for 2D/3D):
```bash
# CUDA is used automatically for 2D/3D simulations if available
python Gyroid.py
```

## Tips

- **Start small**: Use small grid sizes (e.g., 32x32x32) for testing
- **Box optimization**: Use `relax_cell: True` to optimize box dimensions
- **Convergence issues**: Reduce `mix_min` and `mix_init` in optimizer settings
- **Memory**: Set `reduce_memory: True` for large systems

## References

#### L-FTS of Block Copolymer Melts (block_copolymer/lfts/Lamella.py)
+ T. M. Beardsley, R. K. W. Spencer, and M. W. Matsen, Computationally Efficient Field-Theoretic Simulations for Block Copolymer Melts, *Macromolecules* **2019**, 52, 8840

#### Discrete Chain Model (block_copolymer/lfts/Gyroid.py)
+ T. M. Beardsley, and M. W. Matsen, Fluctuation correction for the order-disorder transition of diblock copolymer melts, *J. Chem. Phys.* **2021**, 154, 124902

#### Complex Langevin FTS (block_copolymer/clfts/Lamella.py)
+ V. Ganesan, and G. H. Fredrickson, Field-theoretic polymer simulations, *Europhys. Lett.* **2001**, 55, 814
+ K. T. Delaney, and G. H. Fredrickson, Recent Developments in Fully Fluctuating Field-Theoretic Simulations of Polymer Melts and Solutions, *J. Phys. Chem. B* **2016**, 120, 7615
+ J. D. Willis, and M. W. Matsen, Stabilizing complex-Langevin field-theoretic simulations for block copolymer melts, *J. Chem. Phys.* **2024**, 161, 244903
+ M. W. Matsen, J. Ramírez, J. D. Willis, and P. D. Pina, Accessing the universal phase behavior of block copolymer melts with complex-Langevin field-theoretic simulations, *J. Chem. Phys.* **2026**, 164, 014905

#### ABA Triblock (block_copolymer/scft/ABA_Triblock1D.py)
+ M. W. Matsen, and R. B. Thompson, Equilibrium behavior of symmetric ABA triblock copolymer melts, *J. Chem. Phys* **1999**, 111, 7139

#### ABC Triblock (block_copolymer/scft/ABC_Triblock_Sphere3D.py)
+ S. J. Park, F. S. Bates, and Kevin D. Dorfman, Complex Phase Behavior in Binary Blends of AB Diblock Copolymer and ABC Triblock Terpolymer, *Macromolecules* **2023**, 56, 1278

#### Multi-Arm Star-Shaped Polymers (block_copolymer/scft/Star9ArmsGyroid.py)
+ M. W. Matsen, and M. Schick, Microphase Separation in Starblock Copolymer Melts, *Macromolecules* **1994**, 27, 6761

#### BottleBrush (block_copolymer/scft/BottleBrushLamella3D.py)
+ S. J. Park, G. K. Cheong, F. S. Bates, and K. D. Dorfman, Stability of the Double Gyroid Phase in Bottlebrush Diblock, *Macromolecules* **2021**, 54, 9063
