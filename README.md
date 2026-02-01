# Polymer Field Theory Simulations with Python (PolymerFTS)

A high-performance library for polymer field theory simulations: Self-Consistent Field Theory (SCFT), Langevin Field-Theoretic Simulation (L-FTS), and Complex Langevin FTS (CL-FTS). Core routines are implemented in C++/CUDA and exposed through Python interfaces.

**License**: Apache License 2.0

## Features

- SCFT, L-FTS, and CL-FTS (beta) simulations
- Arbitrary acyclic branched polymers and their mixtures
- Continuous and discrete chain models
- CPU (MKL, FFTW) and GPU (NVIDIA CUDA) platforms

For details, see [Features.md](docs/getting-started/Features.md).

## Installation

### Quick Start (Conda)

```bash
git clone https://github.com/yongdd/langevin-fts.git
cd langevin-fts
conda env create -f environment.yml
conda activate polymerfts
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 && make install && ctest -L basic
```

> **CPU Backend**: MKL is enabled by default. To use FFTW instead, add `-DPOLYMERFTS_USE_FFTW=ON` (GPL license).

For detailed instructions, troubleshooting, and dependencies, see [Installation.md](docs/getting-started/Installation.md).

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
| [Getting Started](docs/getting-started/) | Installation, quick start, and features |
| [Theory](docs/theory/) | Chain propagators, field theory, SCFT, L-FTS, numerical methods |
| [Internals](docs/internals/) | Architecture, FFT implementation, propagator system |
| [References](docs/references/) | Parameter reference, bibliography, tools |
| `tutorials/` | Jupyter notebooks explaining theory and usage |

## Numerical Methods

| Method | Type | Description |
|--------|------|-------------|
| `rqm4` | Pseudo-spectral | 4th-order Richardson extrapolation (default) |
| `rk2` | Pseudo-spectral | 2nd-order Rasmussen-Kalosakas |
| `cn-adi2` | Real-space | 2nd-order Crank-Nicolson ADI |

**Note**: `rqm4` is the default. See [benchmarks](docs/theory/NumericalMethods.md) for details.

## Citation

If you use this software, please cite:

> D. Yong and J. U. Kim, "Dynamic Programming for Chain Propagator Computation of Branched Block Copolymers in Polymer Field Theory Simulations", *J. Chem. Theory Comput.* **2025**, 21, 3676

## Contributing

Contributions are welcome. See [DeveloperGuide.md](docs/internals/DeveloperGuide.md) for development guidelines.

## References

See [Bibliography.md](docs/references/Bibliography.md) for the complete list of publications.
