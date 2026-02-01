# Langevin Field-Theoretic Simulation (L-FTS)

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes Langevin field-theoretic simulation (L-FTS), which includes compositional fluctuations beyond mean-field theory.

## Table of Contents

1. [Overview](#1-overview)
2. [Partial Saddle-Point Approximation](#2-partial-saddle-point-approximation)
3. [Langevin Dynamics](#3-langevin-dynamics)
4. [Saddle-Point Iteration for Imaginary Fields](#4-saddle-point-iteration-for-imaginary-fields)
5. [Structure Functions](#5-structure-functions)
6. [Implementation](#6-implementation)
7. [References](#7-references)

---

## 1. Overview

L-FTS is a simulation method that incorporates compositional fluctuations into polymer field theory. Unlike SCFT, which finds a single saddle-point solution, L-FTS samples field configurations according to the Boltzmann distribution, capturing:

- **Fluctuation effects**: Order-disorder transition shifts, composition fluctuations
- **Correlation functions**: Structure factors, scattering functions
- **Thermodynamic averages**: Ensemble-averaged properties

The key insight is the **partial saddle-point approximation (PSPA)**: real fields fluctuate via Langevin dynamics while imaginary fields are kept at their saddle points.

---

## 2. Partial Saddle-Point Approximation

### 2.1 Partition Function

The canonical partition function in field-theoretic representation:

$$\mathcal{Z} \propto \int \{\mathcal{D}\Omega_i\} \exp(-\beta H[\{\Omega_i\}])$$

In PSPA, we split auxiliary fields into real ($\Omega_i$) and imaginary ($\Omega_j$) components:

$$\mathcal{Z} \propto \int \{\mathcal{D}\Omega_i\} \exp(-\beta H[\{\Omega_i\}, \{\Omega_j^*\}])$$

where $\{\Omega_j^*\}$ are saddle-point solutions for imaginary fields:

$$\left.\frac{\delta H[\{\Omega_i\}, \{\Omega_j\}]}{\delta \Omega_j}\right|_{\Omega_j = \Omega_j^*} = 0$$

### 2.2 Example: AB Diblock Copolymer

For AB-type systems:
- **Real field**: $\Omega_-$ (exchange field, composition fluctuations)
- **Imaginary field**: $\Omega_+$ (pressure-like field, incompressibility)

The exchange field $\Omega_-$ fluctuates according to Langevin dynamics, while $\Omega_+$ is kept at its saddle point to enforce incompressibility.

---

## 3. Langevin Dynamics

### 3.1 Continuous Langevin Equation

The real fields evolve according to:

$$\frac{\partial \Omega_i(\mathbf{r}, \tau)}{\partial \tau} = -\lambda_i \frac{\delta H}{\delta \Omega_i(\mathbf{r}, \tau)} + \eta(\mathbf{r}, \tau)$$

with noise statistics:
- $\langle \eta(\mathbf{r}, \tau) \rangle = 0$
- $\langle \eta(\mathbf{r}, \tau) \eta(\mathbf{r}', \tau') \rangle = 2\lambda_i k_B T \, \delta(\mathbf{r} - \mathbf{r}') \delta(\tau - \tau')$

### 3.2 Discretized Equation (Per Chain Unit)

Discretizing with cell volume $\Delta V$ and time step $\delta\tau$:

$$\Omega_i(\mathbf{r}, \tau + \delta\tau) = \Omega_i(\mathbf{r}, \tau) - \lambda_i \frac{\delta H}{\delta \Omega_i} \delta\tau N + \sigma_i \mathcal{N}(\mathbf{r}, \tau)$$

where:
- $\mathcal{N}$: Gaussian noise with zero mean and unit variance
- $\sigma_i^2 = \lambda_i \frac{2 \delta\tau N}{\Delta V / R_0^3 \cdot \sqrt{\bar{N}}}$
- $\bar{N} = b^6 \rho_0^2 N$: invariant polymerization index

### 3.3 Euler Method

The simplest discretization (first-order in $\delta\tau$):

$$\Omega_-(\mathbf{r}, \tau + \delta\tau) = \Omega_-(\mathbf{r}, \tau) - \lambda_- \left[\Phi_-(\mathbf{r}) + \frac{2}{\chi N}\Omega_-(\mathbf{r})\right] \delta\tau N + \sigma \mathcal{N}(\mathbf{r}, \tau)$$

### 3.4 Leimkuhler-Matthews Method

A more accurate second-order method that averages noise over two time steps:

$$\Omega_i(\mathbf{r}, \tau + \delta\tau) = \Omega_i(\mathbf{r}, \tau) - \lambda_i \frac{\delta H}{\delta \Omega_i} \delta\tau N + \frac{\sigma_i}{2}\left[\mathcal{N}(\mathbf{r}, \tau) + \mathcal{N}(\mathbf{r}, \tau + \delta\tau)\right]$$

**Advantages:**
- Better sampling of the target distribution
- More stable for larger time steps
- Reduced systematic bias

---

## 4. Saddle-Point Iteration for Imaginary Fields

### 4.1 Simple Mixing

Update the pressure-like field using gradient descent:

$$\Omega_+(\mathbf{r}, \tau + \delta\tau) = \Omega_+(\mathbf{r}, \tau) + \lambda_+ \left[\phi_A(\mathbf{r}) + \phi_B(\mathbf{r}) - 1\right] \delta\tau N$$

### 4.2 Anderson Mixing

Accelerates convergence using history of previous iterations (see [SelfConsistentFieldTheory.md](SelfConsistentFieldTheory.md)).

### 4.3 Linear Response

Uses linear response theory to predict the saddle-point directly:

$$\Omega_+^* \approx \Omega_+^{(0)} + \chi_{\Omega_+}^{-1} \cdot (\phi_+ - 1)$$

where $\chi_{\Omega_+}$ is the response function.

### 4.4 Available Compressors

| Name | Description |
|------|-------------|
| `am` | Anderson Mixing |
| `lr` | Linear Response |
| `lram` | Linear Response + Anderson Mixing (hybrid) |

---

## 5. Structure Functions

### 5.1 Definition

The structure function measures density-density correlations in Fourier space:

$$S_{ij}(\mathbf{k}) = \frac{1}{V} \int d\mathbf{r} \int d\mathbf{r}' \, e^{-i\mathbf{k} \cdot (\mathbf{r} - \mathbf{r}')} \langle \delta\hat{\rho}_i(\mathbf{r}) \delta\hat{\rho}_j(\mathbf{r}') \rangle$$

where $\delta\hat{\rho}_i(\mathbf{r}) = \hat{\rho}_i(\mathbf{r}) - \langle \hat{\rho}_i(\mathbf{r}) \rangle$.

### 5.2 Computing Structure Functions

In L-FTS, structure functions are computed by:
1. Fourier transforming the concentration fields
2. Computing $|\tilde{\phi}(\mathbf{k})|^2$
3. Averaging over Langevin trajectory
4. Spherically averaging in $k$-space

### 5.3 Physical Interpretation

- **Peak position**: Characteristic length scale of microphase separation
- **Peak height**: Degree of ordering (diverges at spinodal)
- **Low-$k$ behavior**: Long-wavelength fluctuations

---

## 6. Implementation

### 6.1 Using the LFTS Class

```python
from polymerfts import lfts
import numpy as np

params = {
    "platform": "cuda",              # or "cpu-mkl", "cpu-fftw"
    "nx": [40, 40, 40],              # Grid points
    "lx": [4.36, 4.36, 4.36],        # Box size

    "chain_model": "discrete",       # or "continuous"
    "ds": 1/90,                      # Contour step (1/N_ref)

    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 17.0},

    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": 0.4},
            {"type": "B", "length": 0.6},
        ],
    }],

    "langevin": {
        "max_step": 10000,           # Total Langevin steps
        "dt": 8.0,                   # Time step (delta_tau * N_ref)
        "nbar": 10000,               # Invariant polymerization index
    },

    "recording": {
        "dir": "data_simulation",
        "recording_period": 1000,     # Save fields every N steps
        "sf_computing_period": 10,    # Compute S(k) every N steps
        "sf_recording_period": 1000,  # Save S(k) every N steps
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

### 6.2 Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `nbar` | Invariant polymerization index $\bar{N}$ | 100 - 10000 |
| `dt` | Time step $\delta\tau N$ | 1.0 - 10.0 |
| `max_step` | Total Langevin steps | 10000 - 100000 |
| `tolerance` | Incompressibility tolerance | 1e-4 |

### 6.3 Noise Amplitude

The noise amplitude depends on system parameters:

$$\sigma = \sqrt{\frac{2 \cdot \delta\tau N}{\Delta V \cdot \sqrt{\bar{N}}}}$$

where $\Delta V = \Delta x \Delta y \Delta z$ is the cell volume.

### 6.4 Output Files

| File | Description |
|------|-------------|
| `fields_NNNNNN.mat` | Potential fields and concentrations |
| `structure_function_NNNNNN.mat` | Structure functions $S_{AA}$, $S_{AB}$, $S_{BB}$ |

---

## 7. References

1. Lennon, E. M., Katsov, K. & Fredrickson, G. H. "Free Energy Evaluation in Field-Theoretic Polymer Simulations." *Phys. Rev. Lett.* **101**, 138302 (2008).

2. Matsen, M. W. & Beardsley, T. M. "Field-Theoretic Simulations for Block Copolymer Melts Using the Partial Saddle-Point Approximation." *Polymers* **13**, 2437 (2021).

3. Vorselaars, B. "Efficient Langevin and Monte Carlo sampling algorithms: The case of field-theoretic simulations." *J. Chem. Phys.* **158**, 114117 (2023).

4. Beardsley, T. M. & Matsen, M. W. "Fluctuation correction for the order–disorder transition of diblock copolymer melts." *J. Chem. Phys.* **154**, 124902 (2021).

5. Morse, D. C., Yong, D. & Chen, K. "Polymer Field Theory for Multimonomer Incompressible Models: Symmetric Formulation and ABC Systems." *Macromolecules* **58**, 816 (2025).
