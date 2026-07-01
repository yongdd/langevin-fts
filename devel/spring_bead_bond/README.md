# Spring-bead bond function: comparison of discretization methods

> Implements the DTFT cell-average bond function from Park et al., *J. Chem. Phys.*
> **2019**, 150, 234901 (appendix) for the **spring-bead (Gaussian) model**, and
> compares it against the Fourier-sampling scheme that the code actually uses.

## 1. What this is

This studies three ways to discretize the **bond function** (the single-`ds`-step
transition probability) of a discrete/continuous chain. Central question:
*"Is the code's Fourier sampling correct, or is the paper's real-space cell-average
better?"*

The bond multiplier being discretized (in Fourier space):
$$\tilde g(k) = \exp\!\left(-\frac{a^2\,ds}{6}\,k^2\right) = \exp\!\left(-\frac{a_{\mathrm{step}}^2}{6}k^2\right)$$

## 2. The three methods

| | Method | Start → transform | Definition |
|---|--------|-------------------|------------|
| **(1)** | **Fourier-sampled [code]** | Fourier sample → IFFT → real | sample $\tilde g(k)=e^{-a_{\mathrm{step}}^2k^2/6}$ on the grid k-points |
| **(2)** | real-space sampled | real sample → FFT → Fourier | sample the real-space Gaussian $g(z)$ on the grid, then normalize |
| **(3)** | paper cell-average | Fourier coeffs (formula) → IFFT → real | DTFT cell-average (Eq. 30, BS replacement): alias sum + sinc cell window |

- **(1) and (3)** both *build in Fourier space, then inverse-transform to real space*
  (their starting point is Fourier). The difference is that (3) adds an alias sum and a
  sinc cell window.
- **(2)** is the opposite direction (*sample in real space, forward FFT to Fourier*).
- **(1) is what the code (`src/common/Pseudo.cpp`) actually uses.**

## 3. Units (the most confusing part)

- Length unit: **$R_0 = aN^{1/2} = 1$** (a = statistical segment length, **not** 1; $a=R_0/\sqrt N$).
- per-bond step length: **$a_{\mathrm{step}} = a\sqrt{ds} = R_0/\sqrt N$**. With ds=1/100, N=100 → $a_{\mathrm{step}}=0.1\,R_0$.
- grid: **dx = 0.1 $R_0$** → **dx/a_step = 1.0**.
- Theory:
  - one bond: $\langle x^2\rangle = a_{\mathrm{step}}^2/3$
  - full chain (N bonds): $\langle x^2\rangle = a^2/3$, $\langle r^2\rangle = R_0^2 = 1$ (since $N\cdot ds=1$)

> ⚠️ Early on I plugged in $a_{\mathrm{step}}$ a factor of $\sqrt N$ too small, which made the
> full chain come out as 100/3. The correct value is **full chain $\langle r^2\rangle = 1\,R_0^2$**.

## 4. Key results

### Shape of the bond function — `bond_fourier_vs_real_dx0.1R0.png`
- **Fourier space**: (1) matches the continuous Gaussian exactly. (2) lies above it (aliasing),
  (3) lies below (sinc window).
- **Real space**: (1) has negative ringing, (2) is the narrowest positive bump, (3) is the broadest.
- **Lesson**: correctness of a bond function must be judged by its *Fourier multiplier*, not its
  real-space shape. (1) has negative ringing in real space, yet its multiplier is exact, so the
  propagator is exact.

> **Note on the truncated tail.** At dx/a_step=1.0 the single-bond multiplier at the Nyquist edge is
> $\exp(-\tfrac{\pi^2}{6}(a_{\mathrm{step}}/dx)^2) = 0.19$ — i.e. the Gaussian tail in k-space is simply
> cut off at Nyquist, not zero. To push it near zero one needs a finer grid:
> dx/a_step ≲ 0.6 gives <1%, ≲0.3 gives <1e-8 (nx ≈ 100 / 200 for L=6$R_0$). This truncation only
> affects the *single bond*; the **full-chain** multiplier $e^{-a^2k_{\max}^2/6}$ is already ~1e-72 at
> Nyquist (decays N× faster in k), so chain statistics stay accurate.

### Chain-statistics accuracy — `chain_x2_3d.png`
Propagate a delta source N=100 times and measure cumulative $\langle r^2\rangle$ (isotropic dx=0.1$R_0$):

| Method | full chain $\langle r^2\rangle/R_0^2$ | accuracy |
|--------|------|----------|
| (1) Fourier [code] | **1.000** | **100 %** ✅ |
| (2) real-space | 0.964 | 96 % |
| (3) paper cell-avg | 1.244 | 124 % |

- **(1) Fourier is accurate regardless of grid resolution / anisotropy** (still 100% on anisotropic coarse grids).
- **(2) real-space**: in coarse directions the multiplier flattens to 1.0, losing the step → collapses to 34%
  on an anisotropic box.
- **(3) paper cell-avg**: cell-averaging adds the cell variance $dx^2/12$, so it **always overshoots**.
  Overshoot ratio $=1+\tfrac14(dx/a_{\mathrm{step}})^2$ → 25% at dx/a_step=1. (The paper itself notes that an
  Eq. 24b correction is needed.)

### Inaccuracy at small n
For small n the propagator width ($\sim\sqrt n\,a_{\mathrm{step}}$) is smaller than dx (sub-grid), so
$\langle r^2\rangle$ is underestimated (~92–97% at n=1). As n grows the width exceeds dx and it becomes
well resolved (self-correcting). This is a **spatial-grid (Nyquist)** effect, not a contour-order effect,
so it is common to all three methods.

## 5. Cross-check against the real solver

`../StatisticalSegmentLength.py` (ported to the current API), run with the real C++ solver, matches this
numpy calculation **to the printed digits** (e.g. asymmetric diblock, n=100 → 190.0 = 190.0). This confirms
the numpy Fourier method reproduces the code exactly.

> **Note**: this test uses zero field (w=0), so RQM4 (4th order), RK2 (2nd order), and the plain numpy
> per-step product are **all identical**. Pure diffusion $e^{-a^2 ds\,k^2/6}$ is exact for any ds; the
> 4th-order advantage (reduced operator-splitting error) only appears when a field is present.

## 5b. Avoiding negative values

The negative ringing of method (1) in real space is **harmless** in the standard
pseudo-spectral method (the propagator works in Fourier space). But if positivity is
genuinely required (e.g. real-space convolution), the question is which method to use.

The relevant scale is the **per-bond step size $\sqrt{ds}\,R_0$** (= 0.1 $R_0$ for ds=1/100).
Negatives and accuracy depend on **dx compared to this step size**, not on box size.
Table for ds=1/100 (single-bond `min(g)` and `<x²>/theory`; "corrected" = pre-narrow the
Gaussian by $a_{\mathrm{eff}}^2=a_{\mathrm{step}}^2-dx^2/4$ then cell-average):

| dx [$R_0$] | (1) Fourier | (2) real-space | (3) cell-avg | corrected cell-avg |
|------------|-------------|----------------|--------------|--------------------|
| 0.025 (=step/4) | ≈0, 1.00 | +, **1.00** | +, 1.02 | +, 1.00 |
| 0.05  (=step/2) | -7e-4, 1.00 | +, 1.00 | +, 1.06 | +, 1.00 |
| 0.10  (=step)   | **-0.1**, 0.98 | +, 0.96 | +, **1.24** | +, 0.98 |
| 0.125 | -0.2, 0.96 | +, 0.76 | +, 1.32 | +, 0.78 |
| 0.15  | -0.1, 0.94 | +, 0.43 | +, 1.31 | +, 0.33 |

**Key facts:**
- **dx ≤ step/4 (= 0.025 $R_0$)**: method (1) Fourier negatives vanish (≈ -1e-12, machine
  roundoff) **and** it is exact. The per-bond multiplier at Nyquist is ~1e-12, i.e. the k-space
  Gaussian tail fully decays inside the Brillouin zone, so there is nothing to truncate.
  At this resolution **real-space and Fourier both give exactly $\langle x^2\rangle = 1/300\,R_0^2$**
  (single bond) — all methods converge, positive and accurate.
- **dx ≈ step (= 0.1 $R_0$)**: Fourier has large negatives. Among positive options, the corrected
  cell-average (98%) is best, raw real-space (96%) loses the step, raw cell-average overshoots (124%).
- **dx > step (≥ 0.125 $R_0$)**: no positive method is accurate (real-space collapses);
  positivity and correct $\langle x^2\rangle$ are fundamentally in tension on coarse grids.

**Rule of thumb:** the cleanest way to avoid negatives is to refine the grid to
**dx ≲ (per-bond step)/4 = $\tfrac14\sqrt{ds}\,R_0$** — then the standard Fourier method (1) is
both positive and exact, and no special scheme is needed. The cost is a finer grid
(e.g. dx=0.025$R_0$ → nx=240 for lx=6$R_0$; 64× the cells of dx=0.1$R_0$ in 3D).
real-space sampling is **not** the best route to positivity: it is positive but loses the step
whenever dx ≳ step.

### Corrected cell-average (positive + correct second moment)

If the grid is fixed at dx ≈ step and positivity is required, the best positive option is a
**variance-corrected cell-average**. The raw cell-average (method 3) overshoots because averaging
a Gaussian over a cell of width dx adds the box variance $dx^2/12$:
$$\langle x^2\rangle_{\text{cell-avg}} = \frac{a_{\mathrm{step}}^2}{3} + \frac{dx^2}{12}.$$
The fix is to **pre-narrow the Gaussian** by exactly that amount before cell-averaging, i.e. use an
effective bond length $a_{\mathrm{eff}}$ such that $a_{\mathrm{eff}}^2/3 + dx^2/12 = a_{\mathrm{step}}^2/3$:
$$\boxed{a_{\mathrm{eff}}^2 = a_{\mathrm{step}}^2 - \frac{dx^2}{4}}$$
Then $g_i = \tfrac{1}{dx}\int_{\text{cell}} \mathcal N(0,a_{\mathrm{eff}}^2/3)\,dz$ (closed form via erf).
The result is **strictly positive** (integral of a positive Gaussian) **and** has the correct
second moment $\langle x^2\rangle = a_{\mathrm{step}}^2/3$.

- Intuition: "anticipate that cell-averaging broadens by $dx^2/12$, so start from a narrower
  Gaussian" — i.e. deconvolve the box filter in advance.
- Accuracy: exact at fine grids (dx ≲ step/1.3); ~98% at dx ≈ step (the residual is because the
  $dx^2/12$ box variance is the *continuous* prediction, while the measured moment is the
  *discrete* sum).
- Limit: requires $a_{\mathrm{eff}}^2>0$, i.e. **dx < 2·step**. For coarser grids the correction is
  impossible (would need negative variance), and no positive method is accurate.
- This is **not** an explicit formula in Park 2019; that paper only states a "slight adjustment to
  satisfy Eq. 24b" is recommended. The closed form $a_{\mathrm{eff}}^2=a_{\mathrm{step}}^2-dx^2/4$
  here is a specific derivation of that adjustment for the spring-bead Gaussian (in the spirit of
  the material-conservation / fixed-rms constraints of Yong & Kim, *Phys. Rev. E* **2017**, 96, 063312).

## 6. Conclusion

**The code's Fourier sampling (method 1) is correct.** It reproduces the chain statistic $\langle r^2\rangle$
exactly, which is why `StatisticalSegmentLength.py` is accurate. The paper's real-space cell-average is meant
for the FJC delta-shell in the dx≪a regime; in this codebase's regime (spring-bead, sub-grid bond, dx≳a_step)
it is actually unsuitable (real-space loses the step, cell-average overshoots).

## Files

| File | Description |
|------|-------------|
| `verify_bond_dtft.py` | implementation of the three methods + verification/plot helpers (1D·3D, DTFT Eq. 30 BS replacement) |
| `bond_fourier_vs_real_dx0.1R0.png` | bond function compared in Fourier and real space |
| `chain_x2_3d.png` | 3D cumulative $\langle r^2\rangle$ accuracy (isotropic dx=0.1$R_0$) |

## References
- Park et al., *J. Chem. Phys.* **2019**, 150, 234901 (DTFT cell-average, appendix Eqs. A1–A8, Eq. 30)
- Code: `src/common/Pseudo.cpp` (`update_boltz_bond_periodic_impl`, $e^{-b^2 \cdot 4\pi^2 \cdot ds/6 \cdot |q|^2}$)
- Verification script: `devel/StatisticalSegmentLength.py`
