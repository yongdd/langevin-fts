# Well-Tempered Metadynamics (WTMD)

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

Well-tempered metadynamics (WTMD) is an enhanced sampling technique that overcomes energy barriers between competing phases by adding a history-dependent bias potential. This enables efficient location of phase transitions such as the order-disorder transition (ODT) in block copolymer melts.

## Table of Contents

1. [Overview](#1-overview)
2. [Theory](#2-theory)
3. [Order Parameter](#3-order-parameter)
   - [3.1 Definition (Continuum)](#31-definition-continuum)
   - [3.2 Discrete Formula](#32-discrete-formula)
   - [3.3 Low-Pass Filter](#33-low-pass-filter)
   - [3.4 Discretization Rules](#34-discretization-rules)
   - [3.5 Implementation](#35-implementation)
4. [Bias Potential Update](#4-bias-potential-update)
5. [Free Energy Estimation](#5-free-energy-estimation)
6. [Implementation](#6-implementation)
7. [Parameters](#7-parameters)
8. [Usage](#8-usage)
9. [Limitations](#9-limitations)
10. [References](#10-references)

---

## 1. Overview

In standard L-FTS, the system can become trapped in metastable states, making it difficult to accurately locate phase transitions. WTMD addresses this by:

1. Adding a bias potential $U(\Psi)$ that depends on an order parameter $\Psi$
2. Periodically depositing Gaussian "hills" to discourage revisiting explored regions
3. Gradually reducing hill heights to achieve convergence to the free energy surface

**Key advantages:**
- Overcomes energy barriers between phases (e.g., lamellar ↔ disordered)
- Provides direct estimate of free energy $F(\Psi)$
- Locates ODT with high precision (e.g., $\Delta\chi N \sim 0.01$)

---

## 2. Theory

### 2.1 Modified Hamiltonian

The WTMD simulation uses a modified Hamiltonian:

$$H = H_f + U(\Psi)$$

where $H_f$ is the field-theoretic Hamiltonian and $U(\Psi)$ is the bias potential.

### 2.2 Modified Langevin Dynamics

The bias contributes an additional force term to the Langevin equation:

$$W_-(\mathbf{r}, \tau + \delta\tau) = W_-(\mathbf{r}, \tau) - \frac{\beta}{\rho_0} \frac{\delta H_f}{\delta W_-(\mathbf{r})} \delta\tau - \frac{dU}{dW_-(\mathbf{r})} \delta\tau + \mathcal{N}(0, \sigma_\tau)$$

The bias force is computed using the chain rule:

$$\frac{dU}{dW_-(\mathbf{r})} = U'(\Psi) \frac{d\Psi}{dW_-(\mathbf{r})}$$

---

## 3. Order Parameter

### 3.1 Definition (Continuum)

The order parameter is defined in the continuum as a weighted $\ell$-norm of the Fourier transform of the exchange field:

$$\Psi = \frac{N}{V} \left( \frac{V}{(2\pi)^3} \int f(k) |W_-(k)|^\ell \, dk \right)^{1/\ell}$$

where:
- $W_-(k) = \mathcal{F}[W_-(\mathbf{r})]$ is the Fourier transform of the exchange field
- $\ell = 4$ is the recommended exponent (empirically determined)
- $f(k)$ is a low-pass filter function

This continuum definition is the fundamental formula from which the discrete implementation is derived.

### 3.2 Discrete Formula

The discrete formula is derived from the continuum definition (Section 3.1) by applying the discretization rules in Section 3.4:

$$\Psi = \frac{1}{M} \left( \sum_k f_k |\tilde{W}_k|^\ell w_t \right)^{1/\ell}$$

where $\tilde{W}_k = \text{DFT}[W_-]$, $M = n_x n_y n_z$ is the total number of grid points, and $w_t$ is the weight for `rfftn` (2 for interior k-points, 1 for edge k-points).

> **Note:** The supplementary code of the original paper (Beardsley & Matsen 2022) uses $1/M^2$ normalization:
> $$\Psi_{\text{paper}} = \frac{1}{M^2} \left( \sum_k f_k |\tilde{W}_k|^\ell w_t \right)^{1/\ell}$$
> Our implementation uses $1/M$ normalization instead, which makes $\Psi$ an intensive quantity—consistent regardless of the number of natural periods in the simulation box.

**Multi-monomer systems:** For systems with more than two monomer types, the "exchange field" is the auxiliary field with the **most negative eigenvalue** (i.e., largest magnitude among negative eigenvalues). This is determined automatically from the eigenvalue decomposition of the interaction matrix (same as [deep-langevin-fts](https://github.com/yongdd/deep-langevin-fts)).

### 3.3 Low-Pass Filter

The filter suppresses high-wavevector noise using a smooth sigmoid:

$$f(k) = \frac{1}{1 + \exp(12(k/k_c - 1))}$$

The cutoff $k_c$ is typically set to 1.4 times the peak position of the disordered-state structure function $S(k)$.

### 3.4 Discretization Rules

**Key relationships for discretization (continuum → discrete):**

| Continuum | Discrete |
|-----------|----------|
| $\frac{V}{(2\pi)^3} \int d\mathbf{k}$ | $\sum_k$ |
| $W_-(\mathbf{k})$ | $\frac{V}{M} \tilde{W}_k$ where $\tilde{W}_k = \text{DFT}[W_-]$ |
| $\mathcal{F}^{-1}[\cdot]$ | $\frac{M}{V} \text{IDFT}[\cdot]$ |

**Field convention:** $w_{\text{code}} = N \cdot w_{\text{paper}}$ (per reference chain vs per segment)

### 3.5 Implementation

The implementation matches [deep-langevin-fts](https://github.com/yongdd/deep-langevin-fts) exactly.

**Order parameter:**
$$\Psi = \frac{1}{M} \left( \sum_k f_k |\tilde{W}_k|^\ell w_t \right)^{1/\ell}$$

```python
self.w_aux_k = np.fft.rfftn(np.reshape(w_aux[self.exchange_idx], self.nx))
psi = np.sum(np.power(np.absolute(self.w_aux_k), self.l) * self.fk * self.wt)
psi = np.power(psi, 1.0 / self.l) / self.M
```

**Bias field:**
$$\text{bias} = V \cdot U'(\Psi) \cdot \frac{M^{2-\ell}}{V} \Psi^{1-\ell} \cdot \text{irfftn}\left[ f_k |\tilde{W}_k|^{\ell-2} \tilde{W}_k \right]$$

```python
up_hat = np.interp(psi, self.psi_range, self.up)
dpsi_dwk = np.power(np.absolute(self.w_aux_k), self.l - 2.0) * np.power(psi, 1.0 - self.l) * self.w_aux_k * self.fk
dpsi_dwr = np.fft.irfftn(dpsi_dwk, self.nx) * np.power(self.M, 2.0 - self.l) / self.V
bias = np.reshape(self.V * up_hat * dpsi_dwr, self.M)
langevin[self.langevin_idx] += bias
```

---

## 4. Bias Potential Update

### 4.1 Histogram-Based Update

The implementation uses a histogram-based approach with FFT convolution for efficient updates. Order parameters are collected during simulation and processed at `update_freq` intervals.

**Normalization constant:**
$$CV = \sqrt{\bar{N}} \cdot V$$

where $\bar{N}$ is the invariant polymerization index (`nbar`) and $V$ is the box volume.

**Gaussian kernels (precomputed in Fourier space):**
```python
X = dpsi * np.concatenate([np.arange((bins+1)//2), np.arange(bins//2) - bins//2]) / sigma_psi
u_kernel = np.fft.rfft(np.exp(-0.5 * X**2))
up_kernel = np.fft.rfft(-X / sigma_psi * np.exp(-0.5 * X**2))
```

### 4.2 Statistics Update

At every `update_freq` steps, the bias potential is updated using FFT convolution:

```python
# Compute histogram of order parameters
hist, _ = np.histogram(order_parameter_history, bins=psi_range_hist, density=True)
hist_k = np.fft.rfft(hist)

# Compute updates via FFT convolution
amplitude = np.exp(-CV * u / dT) / CV
gaussian = np.fft.irfft(hist_k * u_kernel, bins) * dpsi
du = amplitude * gaussian
dup = amplitude * np.fft.irfft(hist_k * up_kernel, bins) * dpsi - CV * up / dT * du

# Update bias potential and derivative
u += du
up += dup
```

### 4.3 Well-Tempering

The exponential prefactor $\exp(-CV \cdot U(\Psi)/\Delta T)$ causes hill heights to decrease in frequently visited regions. This ensures:

1. Early stages: Large hills for rapid exploration
2. Late stages: Small hills for convergence to free energy

---

## 5. Free Energy Estimation

### 5.1 Free Energy from Bias

Once the system reaches a well-tempered state, the free energy is:

$$F(\Psi; \chi) = -\frac{T + \Delta T}{\Delta T} U(\Psi) + \text{constant}$$

### 5.2 Histogram and Phase Transition

The probability distribution of $\Psi$ is:

$$P(\Psi) \propto \exp(-\beta F(\Psi; \chi))$$

This histogram typically shows:
- **Two peaks**: One for each phase (ordered/disordered)
- **Peak areas**: Equal at the phase transition

### 5.3 Linear Extrapolation

If the simulation is run close to the transition, $\chi$ can be adjusted using:

$$F(\Psi; \chi_b + \Delta\chi_b) = F(\Psi; \chi_b) + \frac{\partial F}{\partial \chi_b} \Delta\chi_b$$

The partial derivative is computed during simulation:

$$\frac{\partial F}{\partial \chi_b} = \left\langle \frac{\partial H}{\partial \chi_b} \right\rangle_\Psi = \frac{\rho_0 V}{4\beta} - \frac{\rho_0}{\beta \chi_b^2} \left\langle \int W_-^2(\mathbf{r}) d\mathbf{r} \right\rangle_\Psi$$

---

## 6. Implementation

### 6.1 Class Structure

The WTMD class is initialized automatically by LFTS when the `wtmd` parameter is provided. The class requires `nbar`, `eigenvalues`, and `real_fields_idx` from the simulation context.

```python
from polymerfts.wtmd import WTMD

wtmd = WTMD(
    nx=[32, 32, 32],           # Grid dimensions
    lx=[4.0, 4.0, 4.0],        # Box dimensions
    nbar=10000,                # Invariant polymerization index
    eigenvalues=eigenvalues,   # From field theory
    real_fields_idx=real_idx,  # Indices of real auxiliary fields
    l=4,                       # Order parameter exponent
    kc=6.02,                   # Wavenumber cutoff
    dT=5.0,                    # Well-tempering factor (ΔT/T)
    sigma_psi=0.16,            # Gaussian width
    psi_min=0.0,               # Minimum Ψ
    psi_max=10.0,              # Maximum Ψ
    dpsi=2e-3,                 # Bin width
    update_freq=1000,          # Statistics update frequency
    recording_period=10000,    # Data recording frequency
)
```

### 6.2 Key Methods

| Method | Description |
|--------|-------------|
| `compute_order_parameter(step, w_aux)` | Calculate order parameter $\Psi$ from auxiliary fields |
| `add_bias_to_langevin(psi, langevin)` | Add bias force to Langevin dynamics |
| `store_order_parameter(psi, dH)` | Store $\Psi$ and $dH/d\chi$ for statistics |
| `update_statistics()` | Update $U(\Psi)$ and $U'(\Psi)$ via FFT convolution |
| `write_data(filename)` | Save statistics to .mat file |
| `get_free_energy()` | Return $(\Psi, F(\Psi))$ arrays |

### 6.3 Integration with LFTS

WTMD is integrated into the LFTS class via the `wtmd` parameter:

```python
params = {
    # ... standard LFTS parameters ...
    "langevin": {
        "max_step": 5000000,
        "dt": 8.0,
        "nbar": 10000,
    },

    "wtmd": {
        "ell": 4,              # l (order parameter exponent)
        "kc": 6.02,            # Wavenumber cutoff
        "delta_t": 5.0,        # dT (well-tempering factor)
        "sigma_psi": 0.16,     # Gaussian width
        "psi_min": 0.0,
        "psi_max": 10.0,
        "dpsi": 2e-3,          # Bin width
        "update_freq": 1000,   # Statistics update frequency
        "recording_period": 10000,
    },
}

simulation = LFTS(params=params)
simulation.run(initial_fields=w)

# Access free energy after simulation
psi_bins, free_energy = simulation.wtmd.get_free_energy()
```

### 6.4 Langevin Loop Integration

The WTMD methods are called in the following order during the Langevin loop:

```python
# Before Langevin update
psi = wtmd.compute_order_parameter(langevin_step, w_aux)
wtmd.add_bias_to_langevin(psi, w_lambda)

# ... Langevin update and saddle point ...

# After successful saddle point convergence
dH = mpt.compute_h_deriv_chin(chi_n, w_aux)
wtmd.store_order_parameter(psi, dH)

if langevin_step % wtmd.update_freq == 0:
    wtmd.update_statistics()

if langevin_step % wtmd.recording_period == 0:
    wtmd.write_data("wtmd_statistics_%06d.mat" % langevin_step)
```

---

## 7. Parameters

### 7.1 Order Parameter Parameters

| Config Key | Internal Name | Default | Description |
|------------|---------------|---------|-------------|
| `ell` | `l` | 4 | Norm exponent $\ell$. Use 4 for L/C phases, 2 for S/G phases |
| `kc` | `kc` | 6.02 | Wavenumber cutoff $k_c$. Set to ~1.4× peak of $S(k)$ |

### 7.2 Bias Potential Parameters

| Config Key | Internal Name | Default | Description |
|------------|---------------|---------|-------------|
| `sigma_psi` | `sigma_psi` | 0.16 | Gaussian width $\sigma_\Psi$. Larger = smoother $U(\Psi)$ |
| `delta_t` | `dT` | 5.0 | Well-tempering factor $\Delta T/T$. Similar to barrier height |
| `update_freq` | `update_freq` | 1000 | Langevin steps between statistics updates |
| `recording_period` | `recording_period` | 10000 | Langevin steps between data file outputs |

### 7.3 Histogram Parameters

| Config Key | Internal Name | Default | Description |
|------------|---------------|---------|-------------|
| `psi_min` | `psi_min` | 0.0 | Minimum $\Psi$ for bias histogram |
| `psi_max` | `psi_max` | 10.0 | Maximum $\Psi$ for bias histogram |
| `dpsi` | `dpsi` | 2e-3 | Bin width $d\Psi$ (number of bins = `(psi_max - psi_min) / dpsi`) |

### 7.4 Parameter Selection Guidelines

From Beardsley & Matsen (2022):

1. **Initial runs**: Scan across ODT to bracket transition and estimate $\Psi$ values
2. **$\sigma_\Psi$**: Should allow migration of ~$\sigma_\Psi$ between hill depositions
3. **$\Delta T$**: Start with underestimate; increase if needed (keeping previous $U(\Psi)$)
4. **Convergence**: $U(\Psi)$ shape stabilizes after ~$10^6$ Langevin steps

---

## 8. Usage

### 8.1 Locating ODT

```python
# 1. Run WTMD simulation near expected ODT
params["chi_n"] = {"A,B": 13.15}  # Near expected ODT
simulation = LFTS(params=params)
simulation.run(initial_fields=w)

# 2. Get free energy and histogram
psi_bins, F = simulation.wtmd.get_free_energy()
P = np.exp(-F)  # Unnormalized probability

# 3. Find peaks and compare areas
# ODT is where peak areas are equal
```

### 8.2 Typical Workflow

1. **Initial scan**: Run standard L-FTS across ODT to bracket transition
2. **Parameter tuning**: Determine $\Psi$ values for each phase, set $\sigma_\Psi$
3. **WTMD run**: Run $5 \times 10^6$ Langevin steps near ODT
4. **Extrapolation**: Use linear extrapolation if needed to refine ODT

### 8.3 Checkpoint and Resume

```python
# WTMD state is automatically saved in wtmd_statistics_*.mat files
# at every recording_period steps

# Resume simulation from checkpoint
simulation.continue_run("fields_100000.mat")
```

---

## 9. Limitations

### 9.1 Order Parameter Sensitivity

WTMD works well when $\Psi$ clearly distinguishes competing phases:

| Transition | Effectiveness | Notes |
|------------|---------------|-------|
| **Lamellar ↔ Disorder** | Excellent | Clear morphology change |
| **Cylindrical ↔ Disorder** | Good | Some overlap in $\Psi$ |
| **Spherical ↔ Disorder** | Poor | Similar morphologies |
| **Gyroid ↔ Disorder** | Poor | Similar networks + competing phases |

### 9.2 Competing Phases

In the complex phase window (gyroid region), multiple phases may compete:
- Gyroid (G)
- Cylindrical (C)
- Perforated lamellar (PL)
- Fddd-like

A single order parameter may not distinguish all these phases.

### 9.3 Potential Solutions

1. **Alternative order parameters**: Machine learning could help design better $\Psi$
2. **Multiple order parameters**: Extend WTMD to 2D or higher
3. **Sensitivity to Bragg peaks**: Order parameters based on structure function peaks

---

## 10. References

1. **Original WTMD method:**
   A. Barducci, G. Bussi, and M. Parrinello, "Well-tempered metadynamics: A smoothly converging and tunable free-energy method," *Phys. Rev. Lett.* **100**, 020603 (2008).

2. **WTMD for FTS (primary reference):**
   T. M. Beardsley and M. W. Matsen, "Well-tempered metadynamics applied to field-theoretic simulations of diblock copolymer melts," *J. Chem. Phys.* **157**, 114902 (2022).

3. **WTMD for particle-based simulations:**
   T. Ghasimakbari and D. C. Morse, "Order-disorder transitions and free energies in asymmetric diblock copolymers," *Macromolecules* **53**, 7399-7409 (2020).

4. **L-FTS implementation:**
   M. W. Matsen and T. M. Beardsley, "Field-theoretic simulations for block copolymer melts using the partial saddle-point approximation," *Polymers* **13**, 2437 (2021).

5. **Review:**
   G. Bussi and A. Laio, "Using metadynamics to explore complex free-energy landscapes," *Nat. Rev. Phys.* **2**, 200-212 (2020).
