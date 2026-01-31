# Polymer Field Theory Simulations with Python

A high-performance library for polymer field theory simulations: Self-Consistent Field Theory (SCFT), Langevin Field-Theoretic Simulation (L-FTS), and Complex Langevin FTS (CL-FTS). Core routines are implemented in C++/CUDA and exposed through Python interfaces.

**License**: Apache License 2.0

## Features

- Arbitrary acyclic branched polymers and their mixtures
- Continuous and discrete chain models
- Pseudo-spectral and real-space solvers (RQM4, RK2, CN-ADI2)
- Periodic, reflecting, and absorbing boundary conditions
- CPU (FFTW) and GPU (NVIDIA CUDA) platforms

For details, see [docs/Features.md](docs/Features.md).

## Installation

### Quick Start (Conda)

```bash
conda env create -f environment.yml
conda activate polymerfts
git clone https://github.com/yongdd/langevin-fts.git
cd langevin-fts && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOLYMERFTS_USE_FFTW=ON
make -j8 && make install && ctest -L basic
```

> **FFTW License**: FFTW backend (`-DPOLYMERFTS_USE_FFTW=ON`) is GPL-licensed. Distributing binaries with FFTW requires GPL compliance.

### Other Options

- **Docker**: Pre-built images available for CPU and GPU
- **pip**: `pip install .`

For detailed instructions, troubleshooting, and dependencies, see [docs/Installation.md](docs/Installation.md).

## Getting Started

```bash
conda activate polymerfts
cd examples/scft
python Lamella3D.py
```

Tutorials are in the `tutorials/` folder. Examples are in `examples/scft/`, `examples/lfts/`, and `examples/clfts/`.

## Documentation

| Document | Description |
|----------|-------------|
| [Installation.md](docs/Installation.md) | Installation guide and troubleshooting |
| [UserGuide.md](docs/UserGuide.md) | User guide and parameter reference |
| [DeveloperGuide.md](docs/DeveloperGuide.md) | Developer guide and contributing |
| [Features.md](docs/Features.md) | Complete feature list |
| [NumericalMethodsPerformance.md](docs/NumericalMethodsPerformance.md) | Benchmark comparisons |
| [References.md](docs/References.md) | Publication references |
| `tutorials/` | Jupyter notebooks explaining theory and usage |

## Numerical Methods

| Method | Type | Description |
|--------|------|-------------|
| `rqm4` | Pseudo-spectral | 4th-order Richardson extrapolation (default) |
| `rk2` | Pseudo-spectral | 2nd-order Rasmussen-Kalosakas |
| `cn-adi2` | Real-space | 2nd-order Crank-Nicolson ADI |

**Note**: `rqm4` is the default. See [benchmarks](docs/NumericalMethodsPerformance.md) for details.


## Citation

If you use this software, please cite:

> D. Yong and J. U. Kim, "Dynamic Programming for Chain Propagator Computation of Branched Block Copolymers in Polymer Field Theory Simulations", *J. Chem. Theory Comput.* **2025**, 21, 3676

## Contributing

Contributions are welcome. See [DeveloperGuide.md](docs/DeveloperGuide.md) for guidelines.

## References

See [References.md](docs/References.md) for the complete list of publications.
