# Examples

This directory contains example scripts demonstrating polymer field theory simulations.

## SCFT (`scft/`)

Mean-field saddle point of the field-theoretic Hamiltonian.

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
| `scft/MixtureBlockRandom.py` | Block + random copolymer mixture |

**Phase library** (`scft/phases/`, space-group symmetry):

- **Lamellar**: `Lamella1D.py`
- **Cylindrical**: `Cylinder2D.py`
- **Gyroid**: `SG.py` (single gyroid), `DG.py` (double gyroid)
- **Spherical**: `BCC.py`, `FCC.py`, `HCP_Hexagonal.py`, `SC.py`, `A15.py`, `Sigma.py`
- **Other**: `DD.py` (double diamond), `PL_Hexagonal.py` (perforated lamella), `Fddd.py`, `SD.py`, `SP.py`, `DP.py`

## L-FTS (`lfts/`)

Compositional fluctuations via Langevin dynamics.

| Example | Description |
|---------|-------------|
| `lfts/Lamella.py` | Lamellar phase with fluctuations |
| `lfts/Gyroid.py` | Gyroid with fluctuations |
| `lfts/MixtureBlockRandom.py` | Mixture with fluctuations |
| `lfts/ABC_Triblock_Sphere3D.py` | ABC triblock with fluctuations |

## CL-FTS (`clfts/`)

Complex Langevin dynamics, for systems where the partial saddle-point approximation of L-FTS is not adequate.

| Example | Description |
|---------|-------------|
| `clfts/Lamella.py` | Complex Langevin dynamics |

## Running Examples

From the repository root or examples directory:

```bash
cd examples/scft
python Lamella3D.py
```

For GPU acceleration (recommended for 2D/3D):
```bash
# CUDA is used automatically for 2D/3D simulations if available
python Gyroid.py
```

## Tips

- **Start small**: Use small grid sizes (e.g., 32x32x32) for testing
- **Box optimization**: Use `box_is_altering: True` to optimize box dimensions
- **Convergence issues**: Reduce `mix_min` and `mix_init` in optimizer settings
- **Memory**: Set `reduce_memory: True` for large systems

## References

#### Continuous Chain Model (lfts/Lamella.py)
+ T. M. Beardsley, R. K. W. Spencer, and M. W. Matsen, Computationally Efficient Field-Theoretic Simulations for Block Copolymer Melts, *Macromolecules* **2019**, 52, 8840

#### Discrete Chain Model (lfts/Gyroid.py)
+ T. M. Beardsley, and M. W. Matsen, Fluctuation correction for the order-disorder transition of diblock copolymer melts, *J. Chem. Phys.* **2021**, 154, 124902

#### Complex Langevin FTS (clfts/Lamella.py)
+ V. Ganesan, and G. H. Fredrickson, Field-theoretic polymer simulations, *Europhys. Lett.* **2001**, 55, 814
+ K. T. Delaney, and G. H. Fredrickson, Recent Developments in Fully Fluctuating Field-Theoretic Simulations of Polymer Melts and Solutions, *J. Phys. Chem. B* **2016**, 120, 7615
+ J. D. Willis, and M. W. Matsen, Stabilizing complex-Langevin field-theoretic simulations for block copolymer melts, *J. Chem. Phys.* **2024**, 161, 244903
+ M. W. Matsen, J. Ramírez, J. D. Willis, and P. D. Pina, Accessing the universal phase behavior of block copolymer melts with complex-Langevin field-theoretic simulations, *J. Chem. Phys.* **2026**, 164, 014905

#### ABA Triblock (scft/ABA_Triblock1D.py)
+ M. W. Matsen, and R. B. Thompson, Equilibrium behavior of symmetric ABA triblock copolymer melts, *J. Chem. Phys* **1999**, 111, 7139

#### ABC Triblock (scft/ABC_Triblock_Sphere3D.py)
+ S. J. Park, F. S. Bates, and Kevin D. Dorfman, Complex Phase Behavior in Binary Blends of AB Diblock Copolymer and ABC Triblock Terpolymer, *Macromolecules* **2023**, 56, 1278

#### Multi-Arm Star-Shaped Polymers (scft/Star9ArmsGyroid.py)
+ M. W. Matsen, and M. Schick, Microphase Separation in Starblock Copolymer Melts, *Macromolecules* **1994**, 27, 6761

#### BottleBrush (scft/BottleBrushLamella3D.py)
+ S. J. Park, G. K. Cheong, F. S. Bates, and K. D. Dorfman, Stability of the Double Gyroid Phase in Bottlebrush Diblock, *Macromolecules* **2021**, 54, 9063
