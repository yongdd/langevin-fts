# Smearing Functions for Finite-Range Interactions

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the smearing functions implemented in the PolymerFTS library for field-theoretic simulations of polymer melts.

## Overview

In standard field-theoretic formulations, the Flory-Huggins interaction is treated as a point contact interaction, leading to an ultraviolet (UV) divergence that makes the theory ill-defined in the continuum limit. **Smearing** introduces finite-range interactions by convolving fields with a smearing function $\Gamma(\mathbf{r})$, which regularizes the UV behavior and enables meaningful results independent of the spatial discretization.

## Theory

### Smeared Fields

The smeared concentration field is defined as a convolution:

$$\phi_\Gamma(\mathbf{r}) = \int \phi(\mathbf{r}')\Gamma(\mathbf{r} - \mathbf{r}')d\mathbf{r}'$$

In Fourier space, this convolution becomes a simple multiplication:

$$\tilde{\phi}_\Gamma(\mathbf{k}) = \tilde{\phi}(\mathbf{k}) \cdot \tilde{\Gamma}(\mathbf{k})$$

where $\tilde{\Gamma}(\mathbf{k})$ is the Fourier transform of the smearing function, normalized such that $\tilde{\Gamma}(0) = 1$.

### Finite-Range Interaction

The smearing function defines an effective finite-range interaction potential:

$$u_{\text{int}}(\mathbf{r}) = \int \Gamma(|\mathbf{r} - \mathbf{r}'|)\Gamma(\mathbf{r}')d\mathbf{r}'$$

In Fourier space: $\tilde{u}_{\text{int}}(\mathbf{k}) = |\tilde{\Gamma}(\mathbf{k})|^2$

## Supported Smearing Types

### Gaussian Smearing

The Gaussian smearing function in real space:

$$\Gamma(\mathbf{r}) = \frac{1}{(2\pi a_{\text{int}}^2)^{3/2}} \exp\left(-\frac{r^2}{2a_{\text{int}}^2}\right)$$

In Fourier space:

$$\tilde{\Gamma}(\mathbf{k}) = \exp\left(-\frac{a_{\text{int}}^2 k^2}{2}\right)$$

**Parameter:**
- `a_int`: Smearing length scale in units of $R_0 = b\sqrt{N}$ (reference end-to-end distance)

**Configuration:**
```python
params = {
    # ... other parameters ...
    "smearing": {
        "type": "gaussian",
        "a_int": 0.02  # Recommended: <= 0.02 for universality
    }
}
```

### Sigmoidal Smearing

The sigmoidal smearing function is defined directly in Fourier space:

$$\tilde{\Gamma}(\mathbf{k}) = C_0 \left[1 - \tanh\left(\frac{k - k_{\text{int}}}{\Delta k_{\text{int}}}\right)\right]$$

where $C_0$ is a normalization constant ensuring $\tilde{\Gamma}(0) = 1$:

$$C_0 = \frac{1}{1 - \tanh(-k_{\text{int}}/\Delta k_{\text{int}})}$$

**Parameters:**
- `k_int`: Cutoff wavenumber in units of $1/R_0$
- `dk_int`: Transition width (optional, default: 5.0)

**Configuration:**
```python
params = {
    # ... other parameters ...
    "smearing": {
        "type": "sigmoidal",
        "k_int": 20.0,  # Recommended: >= 20 for universality
        "dk_int": 5.0   # Transition width
    }
}
```

## Universality Requirements

For results to be **universal** (independent of spatial discretization and matching standard Gaussian chain statistics), the smearing parameters must satisfy certain constraints:

| Smearing Type | Universality Condition | Physical Meaning |
|---------------|----------------------|------------------|
| Gaussian | $a_{\text{int}} \lesssim 0.02 R_0$ | Interaction range much smaller than chain size |
| Sigmoidal | $k_{\text{int}} \gtrsim 20/R_0$ | Cutoff wavenumber much larger than chain inverse size |

These conditions ensure that the smearing does not significantly alter the polymer chain statistics at length scales relevant to the physics of interest.

## Role of Smearing

Smearing serves two purposes depending on the simulation type:

### UV Regularization (SCFT, L-FTS, CL-FTS)

In the continuum limit, the point-contact Flory-Huggins interaction leads to ultraviolet (UV) divergence. Smearing introduces a finite interaction range, making the theory well-defined and ensuring that results are independent of spatial discretization. This applies to all simulation methods: SCFT, L-FTS, and CL-FTS.

### CL-FTS Stabilization

In CL-FTS, smearing additionally prevents numerical instabilities known as "hot spots" — localized regions where field values grow uncontrollably. Hot spots arise from high-frequency fluctuations that can destabilize the simulation. Smearing damps these high-frequency modes:

- **Gaussian smearing**: Exponential decay $\exp(-a_{\text{int}}^2 k^2/2)$ strongly suppresses high-$k$ modes
- **Sigmoidal smearing**: Sharp cutoff at $k \approx k_{\text{int}}$ eliminates modes above the cutoff

### Combined with Dynamical Stabilization

For optimal stability in CL-FTS, smearing can be combined with dynamical stabilization (see [DYNAMICAL_STABILIZATION.md](DYNAMICAL_STABILIZATION.md)):

```python
params = {
    # ... other parameters ...
    "smearing": {
        "type": "gaussian",
        "a_int": 0.02
    },
    "alpha_ds": 0.01  # Dynamical stabilization parameter
}
```

## Implementation Details

The `Smearing` class in `src/python/smearing.py` handles all smearing operations:

```python
from polymerfts import Smearing

# Create smearing object
smear = Smearing(nx=[32, 32, 32],
                 lx=[4.0, 4.0, 4.0],
                 smearing_params={"type": "gaussian", "a_int": 0.02})

# Check if smearing is enabled
print(smear.enabled)  # True

# Apply to single field
w_smeared = smear.apply(w_field)

# Apply to dictionary of fields
fields_smeared = smear.apply_to_dict({"A": w_A, "B": w_B})
```

### Automatic Integration

When smearing is specified in the simulation parameters, the `SCFT`, `LFTS`, and `CLFTS` classes automatically:
1. Initialize the smearing object
2. Apply smearing to potential fields before propagator computation: $w_i^{\text{smeared}} = \Gamma * w_i$
3. Apply smearing to concentration fields before force/residual computation: $\phi_i^{\text{smeared}} = \Gamma * \phi_i$

## Choosing Smearing Parameters

### General Guidelines

1. **Start with default values**: `a_int = 0.02` (Gaussian) or `k_int = 20.0` (Sigmoidal)

2. **Check universality**: Run simulations with different smearing parameters and verify results converge

3. **Balance stability vs. accuracy**:
   - Stronger smearing (larger `a_int` or smaller `k_int`) = more stable but less accurate
   - Weaker smearing = more accurate but potentially less stable

### Typical Parameter Ranges

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| `a_int` (Gaussian) | 0.01 - 0.05 | Use 0.02 for standard universality |
| `k_int` (Sigmoidal) | 15 - 30 | Use 20 for standard universality |
| `dk_int` (Sigmoidal) | 3 - 10 | Controls sharpness of cutoff |

## References

1. Delaney, K. T. & Fredrickson, G. H. "Recent Developments in Fully Fluctuating Field-Theoretic Simulations of Polymer Melts and Solutions." *J. Phys. Chem. B* **120**, 7615-7634 (2016).

2. Matsen, M. W., Willis, J. D. & Delaney, K. T. "Accessing the universal phase behavior of block copolymer melts with complex-Langevin field-theoretic simulations." *J. Chem. Phys.* **164**, 014905 (2026).

## See Also

- [DYNAMICAL_STABILIZATION.md](DYNAMICAL_STABILIZATION.md) - Dynamical stabilization for CL-FTS
- `src/python/smearing.py` - Smearing class implementation
- `examples/scft/` - SCFT example scripts
- `examples/lfts/` - L-FTS example scripts
- `examples/clfts/` - CL-FTS example scripts
