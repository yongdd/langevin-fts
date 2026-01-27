# User Guide

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document provides guidance for using the polymer field theory simulation library.

## Getting Started

1. Activate the virtual environment:
   ```bash
   conda activate polymerfts
   ```

2. To learn how to use the library, read the files in the `tutorials/` folder.

3. Run example simulations:
   ```bash
   cd examples/scft
   python Lamella3D.py
   ```

## Units and Conventions

The unit of length in this library is $bN^{1/2}$ for both `Continuous` and `Discrete` chain models, where:
- $b$ is a reference statistical segment length
- $N$ is a reference polymerization index

The fields acting on chains are defined as **per reference chain** potential instead of **per reference segment** potential. The same notation is used in [*Macromolecules* **2013**, 46, 8037].

To obtain the **per reference segment** potential, multiply `ds` to each field.

## Numerical Methods

You can select the numerical algorithm for propagator computation at runtime using the `numerical_method` parameter:

| Method | Solver Type | Description |
|--------|-------------|-------------|
| `rqm4` | Pseudo-spectral | RQM4: 4th-order Richardson extrapolation (default) |
| `rk2` | Pseudo-spectral | RK2: 2nd-order Rasmussen-Kalosakas operator splitting |
| `cn-adi2` | Real-space | CN-ADI2: 2nd-order Crank-Nicolson ADI |

**Note**: `rqm4` is the default numerical method for `scft.py`, `lfts.py`, and `clfts.py`. It offers the best performance among 4th-order methods. See [NumericalMethodsPerformance.md](NumericalMethodsPerformance.md) for benchmark comparisons.

Example:
```python
params = {
    # ... other parameters ...
    "numerical_method": "rqm4"  # or "rk2", "cn-adi2"
}
scft = SCFT(params)
```

## Performance Tips

- **Memory saving**: Set `reduce_memory=True` if memory is insufficient to run your simulation. However, the execution time increases by several times. The method is based on the idea used in [pscfplus](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/SavMem.pdf).

- **CUDA multi-threading**: The CUDA version also uses multiple CPUs. Each CPU is responsible for a CUDA computation stream. Allocate as many CPUs as `OMP_NUM_THREADS` when submitting a job.

- **Single CPU core**: To run a simulation using only 1 CPU core, set `os.environ["OMP_MAX_ACTIVE_LEVELS"]="0"` in the Python script before imports.

## Using SCFT, L-FTS, and CL-FTS Modules

The `scft.py`, `lfts.py`, and `clfts.py` modules are implemented on top of the `_core` library and `polymer_field_theory.py`.

### Examples

- SCFT examples: `examples/scft/`
- L-FTS examples: `examples/lfts/`
- CL-FTS examples: `examples/clfts/`

### SCFT Tips

If your SCFT calculation does not converge, adjust the Anderson Mixing parameters:
```python
"optimizer": {
    "name": "am",
    "mix_min": 0.01,      # reduce from default
    "mix_init": 0.01,     # reduce from default
    "start_error": 1e-2   # reduce from default
}
```

### Platform Selection

The default platform is:
- `cuda` for 2D and 3D simulations
- `cpu-fftw` for 1D simulations

### L-FTS Notes

In `lfts.py`, the structure function is computed under the assumption that $\left<w({\bf k})\right>\left<\phi(-{\bf k})\right>$ is zero.

### CL-FTS Notes

In `clfts.py`, the fields are complex-valued and the full FFT is used for structure function calculations.

### Deep Learning Extension

If your goal is to use deep learning boosted L-FTS, use the sample scripts from the [DL-FTS repository](https://github.com/yongdd/deep-langevin-fts). You can easily turn on/off deep learning from the scripts.

## Validation

Open-source software has no warranty. Make sure that this program reproduces the results of previous SCFT and FTS studies and produces reasonable results.

### Continuous Chain Model
For acyclic branched polymers adopting the `Continuous` model with an even number of contour steps, the results must be identical to those of [PSCF](https://github.com/dmorse/pscfpp) within machine precision.

### Discrete Chain Model
For AB diblock copolymers adopting the `Discrete` model, the results must be identical to those of the code in [*Polymers* **2021**, 13, 2437].

### Cross-Platform Consistency
Results must be identical within machine precision regardless of:
- Platform (CUDA or FFTW)
- Use of superposition (`aggregate_propagator_computation`)
- Use of the memory saving option (`reduce_memory`)

After changing these settings, run a few iterations with the same simulation parameters and verify identical results.

## Additional Tools

MATLAB and Python tools for visualization and renormalization are included in the `tools/` folder.

## Parameter Reference

Simulations are configured via Python dictionaries with keys:

| Parameter | Description |
|-----------|-------------|
| `nx` | Grid points (list: [nx, ny, nz]) |
| `lx` | Box size (list: [lx, ly, lz]) |
| `chain_model` | `"discrete"` or `"continuous"` |
| `ds` | Contour discretization (typically 1/N_Ref) |
| `segment_lengths` | Relative statistical segment lengths |
| `chi_n` | Flory-Huggins interaction parameters × N_Ref |
| `distinct_polymers` | Polymer architectures and volume fractions |
| `platform` | `"cuda"` or `"cpu-fftw"` (auto-selected by default) |
| `numerical_method` | `"rqm4"`, `"rk2"`, or `"cn-adi2"` |
| `reduce_memory` | `True` to enable memory saving mode |
