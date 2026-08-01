# WTMD Simulation Results

**Date**: 2026-02-07
**Status**: Cancelled at ~40% completion

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| System | AB diblock copolymer (Lamella) |
| f (A-fraction) | 4/9 = 0.444 |
| χN | 17.148912 |
| Grid | 40 × 40 × 40 |
| Chain model | Discrete (N=90) |
| dt | 8.0 |
| nbar | 10000 |
| Target steps | 5,000,000 |

### WTMD Parameters

| Parameter | Value |
|-----------|-------|
| ℓ (norm) | 4 |
| kc (cutoff) | 6.02 |
| ΔT/T | 5.0 |
| σ_Ψ | 0.16 |
| Ψ range | 0.0 ~ 10.0 |
| dΨ (bin width) | 1e-3 |

## Jobs Summary

| Job ID | Directory | update_freq | Final Step | Progress |
|--------|-----------|-------------|------------|----------|
| 115949 | wtmd/ | 1000 | 2,000,000 | 40% |
| 115952 | wtmd_100/ | 100 | 1,900,000 | 38% |
| 115951 | wtmd_200/ | 200 | 1,900,000 | 38% |
| 115950 | wtmd_500/ | 500 | 1,900,000 | 38% |

Total runtime: ~3 days 9 hours

## Results

### Bias Potential U(Ψ)

| update_freq | Most visited Ψ | U(Ψ) range |
|-------------|----------------|------------|
| 1000 | 3.26 | 0 ~ 22.6 |
| 100 | 1.62 | 0 ~ 33.7 |
| 200 | 3.27 | 0 ~ 30.4 |
| 500 | 3.23 | 0 ~ 26.1 |

### Convergence Analysis

U(Ψ) at fixed Ψ values over time (update_freq=1000):

| Step | U(Ψ=2.0) | U(Ψ=3.0) |
|------|----------|----------|
| 100,000 | 7.2 | 7.3 |
| 500,000 | 13.9 | 15.4 |
| 1,000,000 | 16.5 | 18.1 |
| 1,500,000 | 18.8 | 20.7 |
| 1,900,000 | 20.2 | 21.6 |

**Observation**: U(Ψ) is still slowly increasing but has stabilized sufficiently for analysis. ~2M steps provides adequate sampling for comparing with deep-langevin-fts.

### Free Energy F(Ψ) — CORRECTED by the full analysis of 2026-08-01

Converting the bias with the well-tempered relation F(Ψ) = −(1 + T/ΔT)·U(Ψ)
= −1.2·U(Ψ) reveals a **double well**, not a monotonic decrease:

- Disordered basin at Ψ ≈ 1.62, lamellar basin at Ψ ≈ 3.26
- Basin free-energy difference ΔF = F_dis − F_ord = +0.5 ± 0.6 kT across the
  four runs — zero within run-to-run scatter: **the simulation sits at the ODT**
- Barrier between the basins: 3.3–4.0 kT (measured from the ordered side)
- Sampled range is Ψ ∈ [0.82, 4.3]; the 0–10 grid is mostly unvisited

Extrapolating with the stored ⟨∂H/∂χ⟩(Ψ) = I₁/I₀ (≈ −685 in the disordered
basin, −730 in the ordered one; slope difference ≈ 45 kT per unit χN):

**χN_ODT ≈ 17.14 ± 0.02**  (f = 4/9, discrete N = 90, n̄ = 10⁴)

Internal consistency: the bias-derived F (−1.2·U) and the histogram-derived F
(−6·ln I₀) agree to ≲0.05 kT over the well-sampled region. The inner
double-well region (1.3 < Ψ < 3.8) is converged to ~±0.5 kT from ~1.5M steps;
only the steep outer walls are still filling, which does not affect the basin
or barrier numbers.

### Effect of update_freq — CORRECTED

| update_freq | Observation |
|-------------|-------------|
| 100–1000 | F(Ψ) curves agree (rms spread ≈ 0.7 kT over the common range) |

The earlier conclusion that update_freq=100 was "trapped near Ψ~1.6 due to
over-biasing" was wrong: its F(Ψ) lies on top of the other runs. Its *global*
minimum flips to the disordered basin only because the two wells differ by
less than the run-to-run scatter. All four update frequencies are reliable.

### Recommendation

The remaining 3M steps are not the best use of compute: the ODT estimate's
uncertainty is dominated by run-to-run scatter, so a few additional
independent runs of the same length would tighten χN_ODT more efficiently
than extending these runs.

## Output Files

### Data Directory Structure
```
data_wtmd/
├── wtmd_statistics_*.mat    # WTMD statistics (every 100k steps)
├── structure_function_*.mat  # S(k) data (every 100k steps)
└── fields_*.mat             # Field configurations
```

### Key Variables in wtmd_statistics_*.mat

| Variable | Description |
|----------|-------------|
| psi_range | Ψ bin centers (10000 bins) |
| u | Bias potential U(Ψ) |
| up | Derivative U'(Ψ) |
| I0 | Histogram of visited Ψ values |
| I1_A_B | Accumulated ∫dH·P(Ψ)dΨ |
| dH_psi_A_B | dH/dΨ = I1/I0 |

## Plots

- `wtmd_comparison.png`: U(Ψ) and -ln(I₀) for all update_freq values
- `wtmd_free_energy.png`: Integrated free energy F(Ψ) from dH/dΨ
- `wtmd_convergence.png`: U(Ψ) convergence over time

## Conclusions

1. **WTMD Implementation**: Working correctly
   - Bias potential U(Ψ) builds up over time
   - System explores Ψ ∈ [0.82, 4.3] (both basins and the barrier)
   - ⟨∂H/∂χ⟩(Ψ) = I₁/I₀ is accumulated and usable for χN extrapolation

2. **Convergence**: Sufficient for analysis
   - ~2M steps provides adequate sampling
   - U(Ψ) shape has stabilized
   - Ready for comparison with deep-langevin-fts

3. **Analysis**: Use existing MATLAB script
   - `/home/yongdd/polymer/WTMD/plot_AB_wtmd_chib.m`

## Next Steps

- [ ] Compare F(Ψ) with deep-langevin-fts results
- [ ] Verify ODT location from F(Ψ) barrier
- [ ] Test with different χN values near ODT

## References

1. T. M. Beardsley and M. W. Matsen, "Well-tempered metadynamics applied to field-theoretic simulations of diblock copolymer melts", J. Chem. Phys. 157, 114902 (2022).
