# Polymer Field Theory Simulations with Python (PolymerFTS)

A high-performance library for polymer field theory simulations: Self-Consistent Field Theory (SCFT), Langevin Field-Theoretic Simulation (L-FTS), and Complex Langevin FTS (CL-FTS). Core routines are implemented in C++/CUDA and exposed through Python interfaces.

**License**: Apache License 2.0

## Features

- SCFT, L-FTS, and CL-FTS (beta) simulations
- Arbitrary acyclic branched polymers, random copolymers, and their mixtures
- Continuous and discrete chain models
- Dynamic-programming reuse of equivalent propagators in branched polymers and mixtures
- SCFT unit-cell relaxation for orthogonal and non-orthogonal cells, and space-group symmetry (beta)
- Well-tempered metadynamics (L-FTS) and charged polymers with smeared Coulomb interactions (CL-FTS)
- CPU (MKL, FFTW) and GPU (NVIDIA CUDA) platforms

For details, see [Features.md](docs/getting-started/Features.md).

## Installation

### Quick Start (Conda)

```bash
git clone https://github.com/yongdd/langevin-fts.git
cd langevin-fts
conda env create -f environment.yml
conda activate polymerfts
```

For GPU support, put CUDA 12.2+ `nvcc` on PATH before building the wheel; CUDA is
detected automatically. The Conda environment does not include the toolkit.
On Windows, follow the [Windows notes](docs/getting-started/Installation.md#windows).

Choose one CPU backend.

**MKL**

```bash
conda install --override-channels -c conda-forge --freeze-installed mkl-devel=2026.1.0
CMAKE_PREFIX_PATH="$CONDA_PREFIX" python -m pip wheel . --no-deps --wheel-dir dist \
    -Ccmake.define.POLYMERFTS_USE_MKL=ON -Ccmake.define.POLYMERFTS_USE_FFTW=OFF
```

**FFTW**

```bash
conda install --override-channels -c conda-forge --freeze-installed fftw=3.3.11
CMAKE_PREFIX_PATH="$CONDA_PREFIX" python -m pip wheel . --no-deps --wheel-dir dist \
    -Ccmake.define.POLYMERFTS_USE_MKL=OFF -Ccmake.define.POLYMERFTS_USE_FFTW=ON
```

Install and verify the wheel:

```bash
python -m pip install --no-deps --force-reinstall dist/polymerfts-0.10.0-*.whl
python -c "from polymerfts import PlatformSelector; print(PlatformSelector.avail_platforms())"
```

Uninstall with `python -m pip uninstall polymerfts`.

FFTW-enabled binary distributions must comply with its
[GPL terms](docs/getting-started/Installation.md#gpl-license-warning-fftw).

For detailed instructions, troubleshooting, and dependencies, see [Installation.md](docs/getting-started/Installation.md).

## Getting Started

```bash
conda activate polymerfts
cd examples/scft
python Lamella3D.py
```

Tutorials are in the `tutorials/` folder. Examples are in `examples/scft/`, `examples/lfts/`, and `examples/clfts/`.

## Documentation

| Directory | Description |
|----------|-------------|
| [Getting Started](docs/getting-started/) | Installation, quick start, and features |
| [Theory](docs/theory/) | Chain propagators, field theory, SCFT, L-FTS, numerical methods |
| [Internals](docs/internals/) | Architecture, FFT implementation, propagator system |
| [References](docs/references/) | Parameter reference, bibliography, tools |
| `tutorials/` | Jupyter notebooks explaining theory and usage |

## Citation

If you use this software, please cite:

> D. Yong and J. U. Kim, "Dynamic Programming for Chain Propagator Computation of Branched Block Copolymers in Polymer Field Theory Simulations", *J. Chem. Theory Comput.* **2025**, 21, 3676
