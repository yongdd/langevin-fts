# Related Software

Other software for polymer field theory. The summaries follow each project's own
documentation as of October 2026; see the linked pages for current features,
installation and citation requests.

| Software | Methods | Implementation | License |
|----------|---------|----------------|---------|
| [PSCF](#pscf) | SCFT, PS-FTS | C++, CUDA | GPL-3.0 |
| [PSCF+](#pscf-1) | SCFT | C++, CUDA | GPL-3.0 |
| [Matsen and Beardsley's FTS programs](#matsen-and-beardsleys-fts-programs) | L-FTS, CL-FTS, WTMD for diblock melts | C++, CUDA | — |
| [SUSHI (OCTA)](#sushi-octa) | SCFT, GRPA, hybrid method | GPU and MPI parallel | free and open (OCTA) |
| [OpenFTS](#openfts) | SCFT, CL-FTS, RPA, Gibbs ensemble | C++, Python interface, CUDA, HIP | private (contact the developer) |
| [polycomp](#polycomp) | CL-FTS, polyelectrolytes | Python, CUDA (CuPy) | MIT |
| [BioFTS](#biofts) | CL-FTS, SCFT, biomolecular condensates | Python | MIT |

---

## PSCF

<https://github.com/dmorse/pscfpp> · manual: <https://dmorse.github.io/pscfpp-man>

The C++/CUDA successor of the Fortran PSCF program from David Morse's group at the
University of Minnesota. It performs SCFT and PS-FTS of incompressible mixtures of block
polymers, homopolymers and solvents, including acyclic branched polymers, with
continuous Gaussian or discrete bead-spring chains. `pscf_r1d` solves 1D problems
in Cartesian, cylindrical or spherical coordinates by finite differences; `pscf_rpc`
(CPU) and `pscf_rpg` (GPU) treat periodic systems pseudo-spectrally, with unit-cell
optimization, all 230 space groups, masks for thin films, parameter sweeps, and
Brownian dynamics (Leimkuhler-Matthews, predictor-corrector) or Monte Carlo
sampling for FTS.

## PSCF+

<https://github.com/qwcsu/PSCFplus>

An SCFT package by Qiang (David) Wang and Juntong He at Colorado State University,
built on the GPU framework of PSCF. It adds discrete-chain models with finite-range
non-bonded interactions, Richardson-extrapolated pseudo-spectral methods of several
orders (REPS-*K*), a "slice" algorithm that reduces the propagator memory, reuse of
equivalent propagator calculations from the chain architecture, discrete cosine
transforms for phases of the Pmmm supergroup, support for athermal species, and
automated calculation along a path (ACAP) combined with phase-boundary searches.

Reference: J. He and Q. Wang, *J. Chem. Theory Comput.* **21**, 9879 (2025).

## Matsen and Beardsley's FTS Programs

<https://www.tbeardsley.com/projects> · code: <https://github.com/tmbeardsley>

Compact C++/CUDA programs from the group of M. W. Matsen at the University of Waterloo,
developed by T. M. Beardsley, for AB diblock copolymer melts of discrete bead-spring
chains, each with a GPU version and some with a CPU version: `lfts_gpu`/`lfts_cpu`
(PS-FTS with Langevin dynamics), `clfts_gpu` (CL-FTS of a compressible melt) and
`wtmd_lfts_gpu`/`wtmd_lfts_cpu` (well-tempered metadynamics for L-FTS).

## SUSHI (OCTA)

<https://octa.jp/components/sushi/>

SUSHI (Simulation Utilities for Soft and Hard Interfaces) is a component of OCTA,
free and open software developed in a Japanese industry-government-academia project
conducted at Nagoya University from 1998 to 2002. It computes equilibrium and
non-equilibrium structures of polymer blends and block copolymers with SCFT, the
generalized random phase approximation (GRPA) and a hybrid method. It treats linear
and branched polymers of any topology, copolymers of any sequence (block, random,
tapered) and polymers grafted
on solid surfaces, with applications to micelles, interfaces, polymer brushes,
adsorption, thin films and reaction-induced phase separation. Hydrodynamic effects
can be included, and calculations run in parallel on GPUs and with MPI.

## OpenFTS

<https://www.lequieulab.org/openfts-docs/>

A C++11 field-theoretic simulation package with a Python interface, developed by
Joshua Lequieu at Drexel University. It runs on single or multiple CPUs (OpenMP),
NVIDIA GPUs (CUDA) and AMD GPUs (HIP) in single or double precision, and provides
variable-cell SCFT, CL-FTS, RPA, the Gibbs ensemble and
automated time-step optimization, for continuous and discrete chains. The OpenFTS
repository is private; access and licensing are arranged with Joshua Lequieu.

## polycomp

<https://github.com/rotskoff-group/polycomp> · documentation:
<https://rotskoff-group.github.io/polycomp>

A Python CL-FTS package from Grant Rotskoff's group at Stanford University, with a
focus on polyelectrolytes. It runs on CUDA GPUs through CuPy.

Reference: E. K. Pert, P. J. Hurst, R. M. Waymouth and G. M. Rotskoff,
*J. Chem. Phys.* **162**, 074902 (2025).

## BioFTS

<https://github.com/jwessen/BioFTS>

A Python package by Jonas Wessén (Department of Biochemistry, University of Toronto)
for CL-FTS of general polymer field theories, aimed at liquid-liquid phase separation
of intrinsically disordered proteins and other biomolecular condensates. Interactions
are Yukawa-type (for example, electrostatics) or contact (excluded volume) potentials;
species can be canonical or grand canonical, and switching off the noise gives SCFT.

Reference: J. Wessén, S. Das, T. Pal and H. S. Chan, *J. Phys. Chem. B* **126**,
9222 (2022).
