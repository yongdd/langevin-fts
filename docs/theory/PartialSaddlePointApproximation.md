# Langevin Field-Theoretic Simulation with the Partial Saddle-Point Approximation (L-FTS)

This document describes L-FTS, which samples the real auxiliary fields with Langevin dynamics and keeps the imaginary fields at their saddle point. The Langevin dynamics, the observables, the structure functions and the renormalization that L-FTS shares with CL-FTS are described in [LangevinFieldTheoreticSimulation.md](LangevinFieldTheoreticSimulation.md).

## Table of Contents

1. [Overview](#1-overview)
2. [Partial Saddle-Point Approximation](#2-partial-saddle-point-approximation)
3. [Saddle-Point Iteration for Imaginary Fields](#3-saddle-point-iteration-for-imaginary-fields)
4. [Using the LFTS Class](#4-using-the-lfts-class)
5. [Key Parameters](#5-key-parameters)
6. [References](#6-references)

---

## 1. Overview

The key insight of L-FTS is the **partial saddle-point approximation (PSPA)**: real fields fluctuate via Langevin dynamics ([Langevin dynamics](LangevinFieldTheoreticSimulation.md#2-langevin-dynamics)) while imaginary fields are kept at their saddle points (References 1 and 2). Each Langevin step therefore solves the saddle-point equations of the imaginary fields (Section 3). [CL-FTS](ComplexLangevinSimulation.md) lets the imaginary fields fluctuate as well, at the cost of complex fields and possible numerical instabilities.

---

## 2. Partial Saddle-Point Approximation

### 2.1 Partition Function

The canonical partition function in field-theoretic representation:

```math
\mathcal{Z} \propto \int \{\mathcal{D}\Omega_i\} \exp(-\beta H[\{\Omega_i\}])
```

In PSPA, we split auxiliary fields into real ($`\Omega_i`$) and imaginary ($`\Omega_j`$) components:

```math
\mathcal{Z} \propto \int \{\mathcal{D}\Omega_i\} \exp(-\beta H[\{\Omega_i\}, \{\Omega_j^*\}])
```

where $`\{\Omega_j^*\}`$ are saddle-point solutions for imaginary fields:

```math
\left.\frac{\delta H[\{\Omega_i\}, \{\Omega_j\}]}{\delta \Omega_j}\right|_{\Omega_j = \Omega_j^*} = 0
```

### 2.2 Example: AB Diblock Copolymer

For AB-type systems:
- **Real field**: $`\Omega_-`$ (exchange field, composition fluctuations)
- **Imaginary field**: $`\Omega_+`$ (pressure-like field, incompressibility)

The exchange field $`\Omega_-`$ fluctuates according to Langevin dynamics, while $`\Omega_+`$ is kept at its saddle point to enforce incompressibility.

---

---

## 3. Saddle-Point Iteration for Imaginary Fields

### 3.1 Simple Mixing

Update the pressure-like field using gradient descent; $`k`$ counts the saddle-point iterations within one Langevin step:

```math
\Omega_+^{(k+1)}(\mathbf{r}) = \Omega_+^{(k)}(\mathbf{r}) + \alpha \left[\phi_A(\mathbf{r}) + \phi_B(\mathbf{r}) - 1\right]
```

where $`\alpha`$ is the mixing parameter of the compressor (`mix_init`, adapted as in SCFT simple mixing), not the time step $`\delta\tau N`$.

### 3.2 Anderson Mixing

Accelerates convergence using history of previous iterations (see [SelfConsistentFieldTheory.md](SelfConsistentFieldTheory.md)).

### 3.3 Linear Response

Uses linear response theory to predict the saddle-point directly:

```math
\Omega_+^* \approx \Omega_+^{(0)} + \chi_{\Omega_+}^{-1} \cdot (\phi_+ - 1)
```

where $`\chi_{\Omega_+}`$ is the response function. It is measured once, at $`\Omega = 0`$, from a small kick of $`\Omega_+`$ at one grid point, and applied in the Fourier basis that makes it diagonal: the FFT in a periodic box and the cosine series (DCT-II) along reflecting walls, where the box is half of a periodic box of twice its size that holds the mirror image of the kick. Next to absorbing walls the concentrations at $`\Omega = 0`$ are not uniform, so no Fourier basis makes the response diagonal; `lr` and `lram` are refused there.

### 3.4 Available Compressors

| Name | Description |
|------|-------------|
| `am` | Anderson Mixing |
| `lr` | Linear Response |
| `lram` | Linear Response + Anderson Mixing (hybrid) |

---

## 4. Using the LFTS Class

```python
from polymerfts import lfts
import numpy as np

params = {
    "platform": "cuda",              # or "cpu-mkl", "cpu-fftw"
    "nx": [40, 40, 40],              # Grid points
    "lx": [4.36, 4.36, 4.36],        # Box size

    "chain_model": "discrete",       # or "continuous"
    "ds": 1/90,                      # Contour step (1/N_Ref)

    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 17.0},

    "polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": 0.4},
            {"type": "B", "length": 0.6},
        ],
    }],

    "langevin": {
        "max_step": 10000,           # Total Langevin steps
        "dt": 8.0,                   # Time step (delta_tau * N_Ref)
        "nbar": 10000,               # Invariant polymerization index
    },

    "recording": {
        "dir": "data_simulation",
        "checkpoint_interval": 1000,     # Save fields every N steps
        "sample_interval": 10,    # Compute S(k) every N steps
        "statistics_save_interval": 1000,  # Save S(k) every N steps
    },

    "saddle": {
        "max_iter": 100,
        "tolerance": 1e-4,
    },

    "compressor": {
        "name": "lram",              # Linear Response + Anderson Mixing
        "max_hist": 20,
        "start_error": 5e-1,
        "mix_min": 0.01,
        "mix_init": 0.01,
    },

    "verbose_level": 1,
}

# Initialize and run
random_seed = 12345
simulation = lfts.LFTS(params=params, random_seed=random_seed)
simulation.run(initial_fields={"A": w_A, "B": w_B})
```

---

## 5. Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `nbar` | Invariant polymerization index $`\bar{N}`$ | 100 - 10000 |
| `dt` | Time step $`\delta\tau N`$ | 1.0 - 10.0 |
| `max_step` | Total Langevin steps | 10000 - 100000 |
| `tolerance` | Incompressibility tolerance | 1e-4 |

[L-FTS Parameters](../references/Parameters.md#l-fts-parameters) lists every option.

---

## 6. References

1. Matsen, M. W. & Beardsley, T. M. "Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation." *Polymers* **13**, 2437 (2021).

2. Beardsley, T. M. & Matsen, M. W. "Fluctuation correction for the order–disorder transition of diblock copolymer melts." *J. Chem. Phys.* **154**, 124902 (2021).

3. Vorselaars, B. "Efficient Langevin and Monte Carlo sampling algorithms: The case of field-theoretic simulations." *J. Chem. Phys.* **158**, 114117 (2023).
