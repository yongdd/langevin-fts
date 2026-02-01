# Well-Tempered Metadynamics (WTMD)

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

Well-tempered metadynamics (WTMD) is an enhanced sampling technique that overcomes energy barriers between competing phases by adding a history-dependent bias potential. This enables efficient location of phase transitions such as the order-disorder transition (ODT) in block copolymer melts.

## Table of Contents

1. [Overview](#1-overview)
2. [Theory](#2-theory)
3. [Order Parameter](#3-order-parameter)
   - [3.1 Definition](#31-definition)
   - [3.2 Low-Pass Filter](#32-low-pass-filter)
   - [3.3 Discretization](#33-discretization)
   - [3.4 Implementation](#34-implementation)
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

### 3.1 Definition

The order parameter is a weighted $\ell$-norm of the Fourier transform of the exchange field:

$$\Psi = \frac{N}{V} \left( \frac{V}{(2\pi)^3} \int f(k) |W_-(k)|^\ell \, dk \right)^{1/\ell}$$

where:
- $W_-(k) = \mathcal{F}[W_-(\mathbf{r})]$ is the Fourier transform of the exchange field
- $\ell = 4$ is the recommended exponent (empirically determined)
- $f(k)$ is a low-pass filter function

**Discrete implementation** (used in `wtmd.py`):

$$\Psi = \left( \frac{1}{M} \sum_k f(k) |\tilde{W}_-(k)|^\ell \right)^{1/\ell}$$

where $\tilde{W}_-(k) = \text{DFT}[W_-]$ and $M = n_x n_y n_z$ is the total number of grid points.

> **Note:** The supplementary code of the original paper (Beardsley & Matsen 2022) uses $1/M^2$ normalization:
> $$\Psi_{\text{paper}} = \left( \frac{1}{M^2} \sum_k f(k) |\tilde{W}_-(k)|^\ell \right)^{1/\ell}$$
> Our implementation uses $1/M$ normalization instead, which makes $\Psi$ an intensive quantity—consistent regardless of the number of natural periods in the simulation box.

**Multi-monomer systems:** For systems with more than two monomer types, the "exchange field" is the auxiliary field with the **most negative eigenvalue** (i.e., largest magnitude among negative eigenvalues). This is determined automatically from the eigenvalue decomposition of the interaction matrix (same as [deep-langevin-fts](https://github.com/yongdd/deep-langevin-fts)).

### 3.2 Low-Pass Filter

The filter suppresses high-wavevector noise. The paper uses a smooth sigmoid:

$$f(k) = \frac{1}{1 + \exp(12(k/k_c - 1))}$$

The cutoff $k_c$ is typically set to 1.4 times the peak position of the disordered-state structure function $S(k)$.

**Simplified implementation** (used in `wtmd.py`):

$$f(k) = \begin{cases} 1 & \text{if } k < k_c \\ 0 & \text{otherwise} \end{cases}$$

### 3.3 Discretization

**Key relationships for discretization:**

| Continuum | Discrete |
|-----------|----------|
| $\frac{V}{(2\pi)^3} \int d\mathbf{k}$ | $\sum_k$ |
| $W_-(\mathbf{k})$ | $\frac{V}{M} \tilde{W}_k$ where $\tilde{W}_k = \text{DFT}[W_-]$ |
| $\mathcal{F}^{-1}[\cdot]$ | $\frac{M}{V} \text{IDFT}[\cdot]$ |

**Field convention:** $w_{\text{code}} = N \cdot w_{\text{paper}}$ (per reference chain vs per segment)

### 3.4 Implementation

The implementation matches [deep-langevin-fts](https://github.com/yongdd/deep-langevin-fts) exactly.

**Order parameter:**
$$\Psi = \frac{1}{M} \left( \sum_k f_k |\tilde{W}_k|^\ell w_t \right)^{1/\ell}$$

```python
psi_sum = np.sum(np.abs(w_k)**self.ell * self.fk * self.wt)
psi = psi_sum**(1.0 / self.ell) / self.n_grid
```

**Bias field:**
$$\text{bias} = V \cdot U'(\Psi) \cdot \frac{M^{2-\ell}}{V} \Psi^{1-\ell} \cdot \text{irfftn}\left[ f_k |\tilde{W}_k|^{\ell-2} \tilde{W}_k \right]$$

```python
dpsi_dwk = np.abs(w_k)**(self.ell - 2) * psi**(1.0 - self.ell) * w_k * self.fk
dpsi_dw = np.fft.irfftn(dpsi_dwk, s=self.nx) * self.n_grid**(2.0 - self.ell) / self.volume
bias_field = self.volume * du_dpsi * dpsi_dw.flatten()
```

---

## 4. Bias Potential Update

### 4.1 Gaussian Hill Deposition

Starting from $U(\Psi) = 0$, Gaussian hills are periodically added at the current order parameter value $\hat{\Psi}$:

$$\beta \delta U(\Psi) = \exp\left(-\frac{U(\Psi)}{k_B \Delta T}\right) \exp\left(-\frac{(\hat{\Psi} - \Psi)^2}{2\sigma_\Psi^2}\right)$$

Key features:
- **Gaussian width** $\sigma_\Psi$: Controls smoothness of $U(\Psi)$
- **Well-tempering factor** $\Delta T$: Controls decay rate of hill heights

### 4.2 Well-Tempering

The exponential prefactor $\exp(-U(\Psi)/k_B\Delta T)$ causes hill heights to decrease in frequently visited regions. This ensures:

1. Early stages: Large hills for rapid exploration
2. Late stages: Small hills for convergence to free energy

### 4.3 Derivative Update

The derivative $U'(\Psi)$ is updated simultaneously:

$$\delta U'(\Psi) = \left( \frac{\hat{\Psi} - \Psi}{\sigma_\Psi^2} - \frac{U'(\Psi)}{k_B \Delta T} \right) \delta U(\Psi)$$

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

```python
from polymerfts import WTMD

wtmd = WTMD(
    nx=[32, 32, 32],      # Grid dimensions
    lx=[4.0, 4.0, 4.0],   # Box dimensions
    ell=4,                # Order parameter exponent
    sigma_psi=40.0,       # Gaussian width
    delta_t=5.0,          # Well-tempering factor
    kc=6.02,              # Wavenumber cutoff
    update_freq=1000,     # Hill deposition frequency
)
```

### 6.2 Key Methods

| Method | Description |
|--------|-------------|
| `get_psi(w_minus)` | Calculate order parameter $\Psi$ |
| `get_bias_field(w_minus)` | Calculate bias force $-dU/dW_-$ |
| `update_bias(psi)` | Add Gaussian hill at current $\Psi$ |
| `get_free_energy()` | Return $(Ψ, F(Ψ))$ arrays |
| `save(filename)` / `load(filename)` | Save/restore state |

### 6.3 Integration with LFTS

WTMD is integrated into the LFTS class via the `wtmd` parameter:

```python
params = {
    # ... standard LFTS parameters ...

    "wtmd": {
        "ell": 4,
        "sigma_psi": 40.0,
        "delta_t": 5.0,
        "kc": 6.02,
        "update_freq": 1000,
    },
}

simulation = LFTS(params=params)
simulation.run(initial_fields=w)

# Access free energy after simulation
psi_bins, free_energy = simulation.wtmd.get_free_energy()
```

---

## 7. Parameters

### 7.1 Order Parameter Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `ell` | $\ell$ | 4 | Norm exponent. Use 4 for L/C phases, 2 for S/G phases |
| `kc` | $k_c$ | 6.02 | Wavenumber cutoff. Set to ~1.4× peak of $S(k)$ |

### 7.2 Bias Potential Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `sigma_psi` | $\sigma_\Psi$ | 40.0 | Gaussian width. Larger = smoother $U(\Psi)$ |
| `delta_t` | $\Delta T/T$ | 5.0 | Well-tempering factor. Similar to barrier height |
| `update_freq` | - | 1000 | Langevin steps between hill depositions |

### 7.3 Histogram Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `psi_min` | 0.0 | Minimum $\Psi$ for bias histogram |
| `psi_max` | 500.0 | Maximum $\Psi$ for bias histogram |
| `n_bins` | 5000 | Number of bins for bias histogram |

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
# Save WTMD state
simulation.wtmd.save("wtmd_state.npz")

# Or use LFTS checkpoint (includes WTMD state)
# Automatically saved in fields_*.mat files

# Resume
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
