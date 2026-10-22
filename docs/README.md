# Documentation

## Getting Started

| Document | Description |
|----------|-------------|
| [Installation.md](getting-started/Installation.md) | Installing the library |
| [QuickStart.md](getting-started/QuickStart.md) | Running your first simulation |
| [SimulationGUI.md](getting-started/SimulationGUI.md) | Browser editor for preparing simulation scripts over SSH (beta) |
| [PscfConversion.md](getting-started/PscfConversion.md) | Export periodic PSCF input, import fields and generate polymerfts SCFT scripts |
| [Features.md](getting-started/Features.md) | Overview of library capabilities |

## Theory

| Document | Description |
|----------|-------------|
| [ChainPropagators.md](theory/ChainPropagators.md) | Chain propagators, partition functions, and concentrations |
| [PolymerFieldTheory.md](theory/PolymerFieldTheory.md) | Field theory formulations (monomer, auxiliary fields) |
| [SelfConsistentFieldTheory.md](theory/SelfConsistentFieldTheory.md) | SCFT saddle-point equations and iterative methods |
| [LangevinFieldTheoreticSimulation.md](theory/LangevinFieldTheoreticSimulation.md) | L-FTS with fluctuations and structure functions |
| [NumericalMethods.md](theory/NumericalMethods.md) | Pseudo-spectral and real-space solvers |
| [StressTensor.md](theory/StressTensor.md) | Stress tensor calculation |
| [SpaceGroup.md](theory/SpaceGroup.md) | Space group symmetry |
| [WellTemperedMetadynamics.md](theory/WellTemperedMetadynamics.md) | Well-Tempered Metadynamics for enhanced sampling |

## Internals

| Document | Description |
|----------|-------------|
| [DeveloperGuide.md](internals/DeveloperGuide.md) | Layers, class hierarchies, ownership, one SCFT iteration, where to change what, development practice |
| [FFTImplementation.md](internals/FFTImplementation.md) | FFT, DCT, DST, CrysFFT implementation |
| [PropagatorSystem.md](internals/PropagatorSystem.md) | Propagator identification and scheduling |
| [MemoryManagement.md](internals/MemoryManagement.md) | Memory optimization and checkpointing |
| [CudaMPS.md](internals/CudaMPS.md) | CUDA Multi-Process Service (MPS) for running multiple simulations on one GPU |

## References

| Document | Description |
|----------|-------------|
| [Parameters.md](references/Parameters.md) | Complete parameter reference |
| [Tools.md](references/Tools.md) | Utility scripts for visualization and data conversion |
| [Bibliography.md](references/Bibliography.md) | Publication references |
| [Errata.md](references/Errata.md) | Corrections to the publications on the algorithms of this library |
