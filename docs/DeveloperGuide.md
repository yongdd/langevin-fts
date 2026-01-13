# Developer Guide

This document provides guidance for developers who want to understand or contribute to the polymer field theory simulation library.

## Architecture Overview

### Chain Propagator Computation
The core algorithm for chain propagator computation is described in:
> D. Yong and J. U. Kim, *J. Chem. Theory Comput.* **2025**, 21, 3676

### Platform Abstraction (Abstract Factory Pattern)
This program is designed to run on different platforms (MKL and CUDA). There is a family of classes for each platform. To produce instances of these classes for a given platform, the **abstract factory pattern** is adopted.

### CUDA Programming
The CUDA implementation utilizes `streams` and `cuFFT` for parallel computation.

Reference: [NVIDIA Streams and Concurrency](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)

### Python Binding
`pybind11` is used to generate Python interfaces for the C++ classes.

Reference: [pybind11 Documentation](https://pybind11.readthedocs.io/en/stable/index.html)

## Code Structure

```
src/
├── common/           # Platform-independent code
│   ├── PropagatorComputation.h   # Chain propagator computation
│   ├── ComputationBox.h          # Simulation grid and FFT
│   ├── Polymer.h                 # Polymer chain definitions
│   ├── Molecules.h               # Polymer mixtures
│   └── AndersonMixing.h          # SCFT solver
├── platforms/
│   ├── cpu/          # Intel MKL implementations
│   └── cuda/         # NVIDIA CUDA implementations
├── python/           # Python simulation modules
│   ├── scft.py
│   ├── lfts.py
│   └── clfts.py
└── pybind11/         # Python bindings
```

## Building and Testing

### Rebuilding After C++ Changes
Changes to `src/common/*.cpp` or `src/platforms/*/*.cpp|.cu` require rebuilding:
```bash
cd build && make -j8 && make install
```

### Python Changes
Changes to `src/python/*.py` take effect after `make install` from build directory. No recompilation needed.

### Running Tests
```bash
cd build
make test           # Run all tests
ctest -V            # Verbose output
ctest -R Pseudo     # Run tests matching "Pseudo"
```

## Platform Implementation

When adding new computational features, you must implement them for both platforms unless explicitly platform-specific:
- `src/platforms/cpu/` - MKL-based CPU implementations
- `src/platforms/cuda/` - CUDA GPU implementations

### Memory Management
C++ code uses raw pointers. Ensure proper allocation/deallocation in constructors/destructors.

### Propagator Computation Optimizer
The `PropagatorComputationOptimizer` automatically detects redundant calculations using hash tables of `PropagatorCode` objects. Avoid manual optimization.

## Adding New Features

### New Monomer Types or Interactions
Modify only the parameter dictionary - the code supports arbitrary numbers of monomer types. The `SymmetricPolymerTheory` class handles interaction matrix eigendecomposition automatically.

### New Numerical Methods
1. Implement the solver in both `cpu/` and `cuda/` directories
2. Add the method to the factory classes
3. Update parameter validation in Python modules
4. Add tests and benchmarks

## Validation Requirements

Results must match:
- **PSCF** ([github.com/dmorse/pscfpp](https://github.com/dmorse/pscfpp)) for continuous chains with even contour steps
- **Previous FTS studies** for discrete AB diblock (*Polymers* **2021**, 13, 2437)
- Results should be **identical across platforms** (CUDA vs MKL) within machine precision

Verify by running the same parameters on both platforms.

## Contributing Guidelines

### What We Accept
- Python scripts for specific polymer morphologies
- Modified versions of `scft.py`, `lfts.py`, etc.
- Bug fixes and performance improvements

### Requirements
- Contributions should contain sample results, test code, or desired outputs
- There should be relevant published literature
- Code does not have to be optimal or excellent
- **Contributed code must not contain GPL-licensed code**

### What to Avoid
- Updating C++/CUDA code directly (contact maintainers instead)
- Exclusive code from your lab without permission
- GPL-licensed code

### Submitting Contributions
If you want to add new features to the C++ code but find it difficult to modify, send sample code written in any programming language to the maintainers.

**Note**: This library is updated without considering compatibility with previous versions. Contributed code will be managed to work with updated versions.

## Code Style

### C++
- Use C++20 features where appropriate
- Follow existing naming conventions
- Document public APIs

### Python
- Follow PEP 8 style guidelines
- Add docstrings for public functions
- Include type hints where practical

## Debugging Tips

### Segmentation Faults
```bash
ulimit -s unlimited
export OMP_STACKSIZE=1G
```

### CUDA Debugging
- Use `cuda-memcheck` for memory errors
- Check return values of CUDA API calls
- Use `nvprof` or Nsight for profiling

### Platform Consistency Issues
If CUDA and MKL give different results:
1. Check floating-point operation order
2. Verify FFT normalization
3. Check memory alignment
