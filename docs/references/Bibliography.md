# References

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document lists the key publications underlying the implementation of this polymer field theory simulation library.

## Primary Citation

If you use this software in your research, please cite:

> D. Yong and J. U. Kim, "Dynamic Programming for Chain Propagator Computation of Branched Block Copolymers in Polymer Field Theory Simulations", *J. Chem. Theory Comput.* **2025**, 21, 3676

## Theory

### Standard Gaussian Model
M. W. Matsen, "The standard Gaussian model for block copolymer melts", *J. Phys.: Condens. Matter* **2002**, 14, R21

## Chain Models

### Discrete Chain Model
S. J. Park, D. Yong, Y. Kim, and J. U. Kim, "Numerical implementation of pseudo-spectral method in self-consistent mean field theory for discrete polymer chains", *J. Chem. Phys.* **2019**, 150, 234901

### Multi-Monomer Polymer Field Theory
D. C. Morse, D. Yong, and K. Chen, "Polymer Field Theory for Multimonomer Incompressible Models: Symmetric Formulation and ABC Systems", *Macromolecules* **2025**, 58, 816

## Numerical Methods

### RQM4 (Richardson Extrapolation)
A. Ranjan, J. Qin, and D. C. Morse, "Linear Response and Stability of Ordered Phases of Block Copolymer Melts", *Macromolecules* **2008**, 41, 942-954

### RK2 (Rasmussen-Kalosakas)
K. Ø. Rasmussen and G. Kalosakas, "Improved numerical algorithm for exploring block copolymer mesophases", *J. Polym. Sci. B: Polym. Phys.* **2002**, 40, 1777

G. Tzeremes, K. Ø. Rasmussen, T. Lookman, and A. Saxena, "Efficient computation of the structural phase behavior of block copolymers", *Phys. Rev. E* **2002**, 65, 041806

### CN-ADI (Crank-Nicolson ADI)
J. Crank and P. Nicolson, "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type", *Proc. Cambridge Phil. Soc.* **1947**, 43, 50-67

D. W. Peaceman and H. H. Rachford, "The numerical solution of parabolic and elliptic differential equations", *J. Soc. Indust. Appl. Math.* **1955**, 3, 28-41

### Material Conservation
D. Yong and J. U. Kim, "Material conservation in polymer field theory simulations", *Phys. Rev. E* **2017**, 96, 063312

### Pseudo-Spectral Algorithm Benchmarks
P. Stasiak and M. W. Matsen, "Efficiency of pseudo-spectral algorithms with Anderson mixing for the SCFT of periodic block-copolymer phases", *Eur. Phys. J. E* **2011**, 34, 110

J. Q. Song, Y. X. Liu, and H. D. Zhang, "An efficient algorithm for self-consistent field theory calculations of complex self-assembled structures of block copolymer melts", *Chinese J. Polym. Sci.* **2018**, 36, 488-496

## Implementation

### CUDA Implementation
G. K. Cheong, A. Chawla, D. C. Morse, and K. D. Dorfman, "Open-source code for self-consistent field theory calculations of block polymer phase behavior on graphics processing units", *Eur. Phys. J. E* **2020**, 43, 15

D. Yong, Y. Kim, S. Jo, D. Y. Ryu, and J. U. Kim, "Order-to-Disorder Transition of Cylinder-Forming Block Copolymer Films Confined within Neutral Interfaces", *Macromolecules* **2021**, 54, 11304

## Field-Theoretic Simulations

### Langevin FTS (L-FTS)
M. W. Matsen and T. M. Beardsley, "Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation", *Polymers* **2021**, 13, 2437

### Complex Langevin FTS (CL-FTS)
V. Ganesan and G. H. Fredrickson, "Field-theoretic polymer simulations", *Europhys. Lett.* **2001**, 55, 814

K. T. Delaney and G. H. Fredrickson, "Recent Developments in Fully Fluctuating Field-Theoretic Simulations of Polymer Melts and Solutions", *J. Phys. Chem. B* **2016**, 120, 7615

J. D. Willis and M. W. Matsen, "Stabilizing complex-Langevin field-theoretic simulations for block copolymer melts", *J. Chem. Phys.* **2024**, 161, 244903

## Field Update Algorithms

### L-FTS Field Updates
B. Vorselaars, "Efficient Langevin and Monte Carlo sampling algorithms: the case of field-theoretic simulations", *J. Chem. Phys.* **2023**, 158, 114117

### SCFT Field Updates
A. Arora, D. C. Morse, F. S. Bates, and K. D. Dorfman, "Accelerating self-consistent field theory of block polymers in a variable unit cell", *J. Chem. Phys.* **2017**, 146, 244902
