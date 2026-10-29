# Examples

Example scripts, grouped first by physical system and then by simulation method:

- `scft/`: self-consistent field theory, the mean-field saddle point of the field-theoretic Hamiltonian.
- `lfts/`: Langevin field-theoretic simulation, compositional fluctuations with the pressure-like fields at their saddle point.
- `clfts/`: complex Langevin simulation, in which every field fluctuates; needed where the partial saddle-point approximation of L-FTS does not hold, for example for charged systems.

| System | SCFT | L-FTS | CL-FTS |
|--------|------|-------|--------|
| [Block copolymers](#block-copolymers-block_copolymer) | `block_copolymer/scft/` | `block_copolymer/lfts/` | `block_copolymer/clfts/` |
| [Blends and coexistence](#blends-and-coexistence-blend) | `blend/scft/` | `blend/lfts/` | `blend/clfts/` |
| [Charged systems](#charged-systems-charged) | `charged/scft/` | — | `charged/clfts/` |
| [Tools](#tools-tools) | `tools/` (transforms and extension APIs) | | |

What the tests and validation runs check for each example, with the compared numbers and figures, is in [Validation of the Examples](../docs/internals/ExampleValidation.md). [Numerical Validation](../docs/internals/NumericalValidation.md) covers the validation of the library as a whole: the regression tests, the published results and other codes that polymerfts reproduces, and the differences from them.

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
| `scft/phase_diagram/` | Phase diagrams by continuation and boundary root finding (AB diblock, ABA triblock; GPU) |
| `scft/crystal_systems/` | Box relaxation in hexagonal, oblique, monoclinic and triclinic cells; free energy against the cell angle |
| `lfts/Lamella.py` | Lamellar phase with fluctuations (discrete chains) |
| `lfts/Gyroid.py` | Gyroid with fluctuations (discrete chains) |
| `lfts/ABC_Triblock_Sphere3D.py` | ABC triblock with fluctuations |
| `clfts/Lamella.py` | Lamellar phase from complex Langevin dynamics; stabilization options in `DYNAMICAL_STABILIZATION.md` and `SMEARING.md` |
| `clfts/SoftBeadDiblockMelt.py` | Pressure and chemical potential of a soft-bead diblock melt (Lequieu 2024) |

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
| `scft/CanonicalCoexistence.py` | Binodal of a homopolymer blend (`find_binary_coexistence`) |
| `lfts/MixtureBlockRandom.py` | The same blend with fluctuations |
| `clfts/GibbsEnsemble.py` | Coexisting phases of a blend from two CL-FTS boxes (`GibbsEnsemble`) |
| `clfts/HomopolymerSolutionBook.py` | Homopolymer in good solvent: chemical potential and pressure (Fredrickson and Delaney 2023) |

## Charged systems (`charged/`)

Smeared Coulomb interactions (`electrostatics`) in SCFT and CL-FTS.

| Example | Description |
|---------|-------------|
| `scft/ChargedDiblockLamella.py` | Charged diblock lamella with point counterions over a range of salt |
| `scft/PolyelectrolyteBrush.py` | Charged brush on a charged wall of a closed box with point ions (`cn-adi2`) |
| `scft/CounterionsBetweenPlates.py` | Poisson-Boltzmann counterions between charged plates (Netz 2001) |
| `clfts/ChargedDiblock.py` | Charged diblock melt with counterions |
| `clfts/SaltReservoir.py` | Charged melt in contact with a salt reservoir (grand-canonical point ions) |
| `clfts/CounterionsBetweenPlates.py` | Counterions between charged plates beyond Poisson-Boltzmann (Netz 2001) |
| `clfts/SmearedSaltSelfEnergy.py` | Self-energy and screening of smeared point ions (Wang 2010; Agrawal and Wang 2022) |
| `clfts/PolyampholyteSolution.py` | Polyampholyte in implicit solvent (McCarty et al. 2019); maps an implicit-solvent model onto the compressible model |
| `clfts/ComplexCoacervate.py` | Salt-free polycation/polyanion solution (Lee, Popov and Fredrickson 2008) |

## Tools (`tools/`)

| Example | Description |
|---------|-------------|
| `FourierTransform.py` | `PropagatorSolver.fourier`: transforms and multipliers, checked against NumPy |
| `CustomExtensions.py` | A custom observable (`callbacks=`) and integrator (`integrator=`) for L-FTS ([Extensions](../docs/internals/Extensions.md)) |

## Running Examples

Run a script from its directory, which holds its input files; the scripts write their output into the current directory. CUDA is used automatically for 2D/3D simulations if available.

```bash
cd examples/block_copolymer/scft
python Lamella3D.py
```

## Tips

- **Start small**: Use small grid sizes (e.g., 32x32x32) for testing
- **Box optimization**: Use `relax_cell: True` to optimize box dimensions
- **Convergence issues**: Reduce `mix_min` and `mix_init` in optimizer settings
- **Memory**: Set `reduce_memory: True` for large systems

## References

- T. M. Beardsley, R. K. W. Spencer, and M. W. Matsen, Computationally Efficient Field-Theoretic Simulations for Block Copolymer Melts, *Macromolecules* **2019**, 52, 8840 (`block_copolymer/lfts/Lamella.py`)
- T. M. Beardsley, and M. W. Matsen, Fluctuation correction for the order-disorder transition of diblock copolymer melts, *J. Chem. Phys.* **2021**, 154, 124902 (`block_copolymer/lfts/Gyroid.py`)
- V. Ganesan, and G. H. Fredrickson, Field-theoretic polymer simulations, *Europhys. Lett.* **2001**, 55, 814 (`block_copolymer/clfts/Lamella.py`)
- K. T. Delaney, and G. H. Fredrickson, Recent Developments in Fully Fluctuating Field-Theoretic Simulations of Polymer Melts and Solutions, *J. Phys. Chem. B* **2016**, 120, 7615 (`block_copolymer/clfts/Lamella.py`)
- J. D. Willis, and M. W. Matsen, Stabilizing complex-Langevin field-theoretic simulations for block copolymer melts, *J. Chem. Phys.* **2024**, 161, 244903 (`block_copolymer/clfts/Lamella.py`)
- M. W. Matsen, J. Ramírez, J. D. Willis, and P. D. Pina, Accessing the universal phase behavior of block copolymer melts with complex-Langevin field-theoretic simulations, *J. Chem. Phys.* **2026**, 164, 014905 (`block_copolymer/clfts/Lamella.py`)
- M. W. Matsen, and R. B. Thompson, Equilibrium behavior of symmetric ABA triblock copolymer melts, *J. Chem. Phys* **1999**, 111, 7139 (`block_copolymer/scft/ABA_Triblock1D.py`)
- S. J. Park, F. S. Bates, and Kevin D. Dorfman, Complex Phase Behavior in Binary Blends of AB Diblock Copolymer and ABC Triblock Terpolymer, *Macromolecules* **2023**, 56, 1278 (`block_copolymer/scft/ABC_Triblock_Sphere3D.py`)
- M. W. Matsen, and M. Schick, Microphase Separation in Starblock Copolymer Melts, *Macromolecules* **1994**, 27, 6761 (`block_copolymer/scft/Star9ArmsGyroid.py`)
- R. R. Netz, Electrostatistics of counter-ions at and between planar charged walls: From Poisson-Boltzmann to the strong-coupling theory, *Eur. Phys. J. E* **2001**, 5, 557 (`charged/scft/CounterionsBetweenPlates.py`, `charged/clfts/CounterionsBetweenPlates.py`)
- Z.-G. Wang, Fluctuation in electrolyte solutions: The self energy, *Phys. Rev. E* **2010**, 81, 021501 (`charged/clfts/SmearedSaltSelfEnergy.py`)
- N. R. Agrawal, and R. Wang, Self-Consistent Description of Vapor-Liquid Interface in Ionic Fluids, *Phys. Rev. Lett.* **2022**, 129, 228001 (`charged/clfts/SmearedSaltSelfEnergy.py`)
- G. H. Fredrickson, and K. T. Delaney, *Field-Theoretic Simulations in Soft Matter and Quantum Fluids*, Oxford University Press, **2023** (`blend/clfts/HomopolymerSolutionBook.py`)
- J. Lequieu, Quantitative Equivalence and Performance Comparison of Particle and Field-Theoretic Simulations, *Macromolecules* **2024**, 57, 10870 (`block_copolymer/clfts/SoftBeadDiblockMelt.py`)
- J. Lee, Y. O. Popov, and G. H. Fredrickson, Complex coacervation: A field theoretic simulation study of polyelectrolyte complexation, *J. Chem. Phys.* **2008**, 128, 224908 (`charged/clfts/ComplexCoacervate.py`)
- J. McCarty, K. T. Delaney, S. P. O. Danielsen, G. H. Fredrickson, and J.-E. Shea, Complete Phase Diagram for Liquid–Liquid Phase Separation of Intrinsically Disordered Proteins, *J. Phys. Chem. Lett.* **2019**, 10, 1644 (`charged/clfts/PolyampholyteSolution.py`)
- S. J. Park, G. K. Cheong, F. S. Bates, and K. D. Dorfman, Stability of the Double Gyroid Phase in Bottlebrush Diblock, *Macromolecules* **2021**, 54, 9063 (`block_copolymer/scft/BottleBrushLamella3D.py`)
