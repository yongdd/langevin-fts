# Numerical Methods: Performance and Accuracy

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document summarizes performance and accuracy considerations for the numerical methods currently supported by the library.

## Available Methods

All numerical methods are selectable at runtime using the `numerical_method` parameter.

### Pseudo-Spectral Methods

| Method | Order | Description | Reference |
|--------|-------|-------------|-----------|
| **RQM4** | 4th | Richardson extrapolation with Ranjan-Qin-Morse 2008 parameters | *Macromolecules* 41, 942-954 (2008) |
| **RK2** | 2nd | Rasmussen-Kalosakas operator splitting (no Richardson extrapolation) | *J. Polym. Sci. B* 40, 1777 (2002) |

> **Note**: RK2 for continuous chains is mathematically equivalent to the **N-bond model** for discrete chains described in Park et al., *J. Chem. Phys.* 150, 234901 (2019).

### Real-Space Methods

| Method | Order | Description |
|--------|-------|-------------|
| **CN-ADI2** | 2nd | Crank-Nicolson Alternating Direction Implicit |

## Usage Example

```python
from polymerfts import scft

params = {
    "nx": [32, 32, 32],
    "lx": [3.3, 3.3, 3.3],
    "ds": 0.01,
    "chain_model": "continuous",
    "numerical_method": "rqm4",  # or "rk2", "cn-adi2"
    # ... other parameters
}

calculation = scft.SCFT(params=params)
```

## Performance Notes

- **RQM4** provides 4th-order accuracy in contour step size at a moderate cost per step.
- **RK2** is faster per step but only 2nd-order accurate; useful for quick iterations.
- **CN-ADI2** enables non-periodic boundary conditions in real space at 2nd-order accuracy.

For up-to-date performance benchmarking, run the benchmarks in `tests/` on your target hardware.
