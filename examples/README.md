# Examples

This directory contains example scripts demonstrating polymer field theory simulations.

## Getting Started

### Recommended Learning Path

**1. Start with SCFT (Self-Consistent Field Theory)**

SCFT finds the mean-field saddle point of the field-theoretic Hamiltonian. Begin here to understand the basics.

| Example | Description | Complexity |
|---------|-------------|------------|
| `scft/Lamella3D.py` | AB diblock lamellar phase - simplest starting point | Beginner |
| `scft/Cylinder3D.py` | Cylindrical phase | Beginner |
| `scft/Gyroid.py` | Gyroid phase with box optimization | Intermediate |
| `scft/GyroidNoBoxChange.py` | Gyroid without box optimization | Beginner |

**2. Complex Polymer Architectures**

After mastering simple diblocks, explore more complex chain architectures.

| Example | Description | Complexity |
|---------|-------------|------------|
| `scft/ABA_Triblock1D.py` | Symmetric ABA triblock copolymer | Intermediate |
| `scft/ABC_Triblock_Sphere3D.py` | ABC triblock terpolymer spheres | Intermediate |
| `scft/Star3Arms1D.py` | 3-arm star polymer (1D) | Intermediate |
| `scft/Star9ArmsGyroid.py` | 9-arm star polymer gyroid | Advanced |
| `scft/BottleBrushLamella3D.py` | Bottle-brush copolymer | Advanced |

**3. Polymer Mixtures and Random Copolymers**

Explore multi-component systems and random copolymers.

| Example | Description | Complexity |
|---------|-------------|------------|
| `scft/MixtureBlockRandom.py` | Block + random copolymer mixture | Intermediate |
| `scft/AdamRandomToGyroid.py` | Using ADAM optimizer for random to gyroid transition | Advanced |

**4. SCFT Phase Library**

Reference implementations for various morphologies (in `scft/phases/`):

- **Lamellar**: `Lamella1D.py`
- **Cylindrical**: `Cylinder2D.py`
- **Gyroid**: `SG.py` (single gyroid), `DG.py` (double gyroid)
- **Spherical**: `BCC.py`, `FCC.py`, `HCP_Hexagonal.py`, `SC.py`, `A15.py`, `Sigma.py`
- **Other**: `DD.py` (double diamond), `PL_Hexagonal.py` (perforated lamella), `Fddd.py`, `SD.py`, `SP.py`, `DP.py`

**5. L-FTS (Langevin Field-Theoretic Simulation)**

L-FTS includes thermal fluctuations via Langevin dynamics. Start after mastering SCFT.

| Example | Description | Complexity |
|---------|-------------|------------|
| `lfts/Lamella.py` | Lamellar phase with fluctuations | Intermediate |
| `lfts/Gyroid.py` | Gyroid with fluctuations | Advanced |
| `lfts/MixtureBlockRandom.py` | Mixture with fluctuations | Advanced |
| `lfts/ABC_Triblock_Sphere3D.py` | ABC triblock with fluctuations | Advanced |

**6. Complex Langevin FTS**

For systems where standard L-FTS has convergence issues.

| Example | Description | Complexity |
|---------|-------------|------------|
| `clfts/Lamella.py` | Complex Langevin dynamics | Advanced |

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
+ M. W. Matsen, T. M. Beardsley, and B. Vorselaars, Field-theoretic simulations for block copolymer melts using the partial saddle-point approximation, *J. Chem. Phys.* **2026**, 164, 014905

#### ABA Triblock (scft/ABA_Triblock1D.py)
+ M. W. Matsen, and R. B. Thompson, Equilibrium behavior of symmetric ABA triblock copolymer melts, *J. Chem. Phys* **1999**, 111, 7139

#### ABC Triblock (scft/ABC_Triblock_Sphere3D.py)
+ S. J. Park, F. S. Bates, and Kevin D. Dorfman, Complex Phase Behavior in Binary Blends of AB Diblock Copolymer and ABC Triblock Terpolymer, *Macromolecules* **2023**, 56, 1278

#### Multi-Arm Star-Shaped Polymers (scft/Star9ArmsGyroid.py)
+ M. W. Matsen, and M. Schick, Microphase Separation in Starblock Copolymer Melts, *Macromolecules* **1994**, 27, 6761

#### BottleBrush (scft/BottleBrushLamella3D.py)
+ S. J. Park, G. K. Cheong, F. S. Bates, and K. D. Dorfman, Stability of the Double Gyroid Phase in Bottlebrush Diblock, *Macromolecules* **2021**, 54, 9063
