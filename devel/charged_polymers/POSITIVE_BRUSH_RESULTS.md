# N=100 brush: actual 1D sampling results — 2026-09-07

The earlier Fig. 2–3 scan used SCFT. These Monte Carlo calculations sample particle configurations and the real field in the attractive channel.
Samples of this finite 1D model were obtained; the PRL's 3D fluctuations and quantitative curves have not been reproduced.

[Distribution and implementation](POSITIVE_BRUSH_SAMPLING.md) · [LaTeX PDF](ONED_PRL_REPRODUCTION.pdf)

## Runs and validation

- Standard: 3 valences × 5 nonzero salt concentrations × 2 initial states + 2 salt-free runs = 32.
- Two particle-only controls for trivalent salt at .01/1 M; 8 grid/area controls. Total: 42 brush trajectories.
- Each brush: 200,000 warmup sweeps + 1,000,000 measurement sweeps. Scalars stored every 100 sweeps; profiles every 1,000 sweeps.
- Exception: z+=2, .01 M initially had split Rhat=1.09546 and relaxation from the compact start, so both seeds received 1,000,000 additional measurement sweeps. Tables use this fixed extension interval. The original interval was not pooled as an independent seed.
- Eighteen separate pure periodic salt baths. Activities were adjusted only during warmup and fixed during measurement.
- Exact 2-cell/2-bead sums were compared with 12 runs: two methods, two seeds, three valences.
- Maximum deviation from exact values: 2.6697 SE. Salt partition truncation bound < 3e-40.
- Maximum absolute detailed-balance/identity error: 1.288e-13; large-grid scaled energy error: 4.310e-15.
- Integer P/S occupancy and total charge are conserved in all trajectories. Maximum accumulated Q update error: 2.981e-12.
- Per-run height ESS: 10.06–79.36; maximum standard paired split Rhat 1.0391.

SE uses the larger of autocorrelation and 10/20/40-block estimates. Combining independent seeds also accounts for between-run scatter.
Height relaxes slowly: a million sweeps are not a million independent samples. Errors exclude finite-grid and model bias.

## Sample averages

h = 2∫z⟨ρP⟩ dz/(σN). Γ is the salt-excess ratio in the core above 0.5 of the mean polymer density peak.
Ψ = (⟨ψwall⟩−⟨ψfar⟩)h/(σNv). Nonlinear observables are computed from mean profiles, with block-jackknife errors.
Bulk density uncertainty and h/ψ covariance are included. Actual net charge (including C and anions) is recorded separately in analysis.json.

| z+ | Target salt [M] | Measured bulk [M] | h [nm] | Γ0.5 | Ψ |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 ± 0 | 58.5607 ± 0.3510 | 1.0000 ± 0.0000 | -30.8028 ± 0.4466 |
| 1 | 0.001 | 0.000995724 ± 9.4e-06 | 57.5166 ± 0.3555 | 0.9583 ± 0.0003 | -20.7814 ± 0.3666 |
| 1 | 0.01 | 0.0100481 ± 2.4e-05 | 54.6272 ± 0.2566 | 0.8576 ± 0.0011 | -13.1627 ± 0.3883 |
| 1 | 0.1 | 0.10016 ± 7.5e-05 | 43.8775 ± 0.1733 | 0.6312 ± 0.0012 | -4.7413 ± 0.2464 |
| 1 | 1 | 0.999995 ± 0.00023 | 28.4894 ± 0.2127 | 0.5417 ± 0.0016 | -0.4192 ± 0.1540 |
| 1 | 10 | 9.99608 ± 0.0011 | 23.5720 ± 0.1412 | 0.5580 ± 0.0042 | 0.1359 ± 0.1261 |
| 2 | 0.001 | 0.00101316 ± 7.4e-06 | 54.3083 ± 0.2232 | 0.8688 ± 0.0013 | -15.1694 ± 0.3467 |
| 2 | 0.01 | 0.00997499 ± 2e-05 | 46.0193 ± 0.2106 | 0.6495 ± 0.0034 | -7.1467 ± 0.2663 |
| 2 | 0.1 | 0.10019 ± 7.9e-05 | 33.0426 ± 0.1861 | 0.4067 ± 0.0038 | -1.8721 ± 0.2038 |
| 2 | 1 | 0.999946 ± 0.00019 | 24.8579 ± 0.1702 | 0.3984 ± 0.0024 | -0.2052 ± 0.1293 |
| 2 | 10 | 10.0017 ± 0.00084 | 23.0538 ± 0.1651 | 0.4186 ± 0.0061 | 0.0452 ± 0.1358 |
| 3 | 0.001 | 0.00100569 ± 9.4e-06 | 50.2135 ± 0.2903 | 0.7774 ± 0.0029 | -11.8280 ± 0.3145 |
| 3 | 0.01 | 0.0100335 ± 2.1e-05 | 39.2816 ± 0.3541 | 0.4856 ± 0.0056 | -4.5453 ± 0.3697 |
| 3 | 0.1 | 0.0998522 ± 5.3e-05 | 28.5975 ± 0.3301 | 0.3098 ± 0.0040 | -0.5994 ± 0.2435 |
| 3 | 1 | 1.00008 ± 0.00019 | 23.7963 ± 0.1887 | 0.3292 ± 0.0033 | 0.0296 ± 0.1224 |
| 3 | 10 | 10.0031 ± 0.00078 | 22.7973 ± 0.2936 | 0.3500 ± 0.0152 | -0.0342 ± 0.1406 |

## Grid/cross-sectional-area checks

| run | h [nm] | height ESS |
|---|---:|---:|
| check_area20_z3_r0.01_s6101 | 38.1583 ± 0.7654 | 20.9 |
| check_area20_z3_r0.01_s6201 | 37.5512 ± 0.6061 | 34.8 |
| check_area80_z3_r0.01_s6101 | 39.7251 ± 0.2355 | 49.7 |
| check_area80_z3_r0.01_s6201 | 39.5296 ± 0.3090 | 42.9 |
| check_grid800_z1_r0_s6101 | 58.7606 ± 0.3331 | 45.6 |
| check_grid800_z1_r0_s6201 | 58.8128 ± 0.9427 | 10.1 |
| check_grid800_z3_r0.01_s6101 | 39.1419 ± 0.3389 | 35.4 |
| check_grid800_z3_r0.01_s6201 | 39.2180 ± 0.2270 | 53.6 |

## Interpretation and remaining scope

Parameters: N=100, b=1 nm, σ=.1 nm⁻², v=1 nm³, χ=.5, ℓB=.7 nm, a=.25 nm. Standard A=40 nm², L=200 nm, M=400.
This explicit finite action has discrete bead positions and integer cell occupancy. It is not claimed identical to the existing CL or 9-bead validation action.
A=20/80 nm² uses 2/8 chains, respectively, to retain σ. Area sets 1D fluctuation strength and is more than a mesh-accuracy parameter.
Height decreases at the sampled concentrations. This restricted 1D measure has not quantitatively reproduced contraction/re-expansion or charge/potential reversal from the paper's 3D correlations.
Remaining checks include dilute-limit tails, mesh/box/area dependence at all concentrations, longer sampling, and comparison with full 3D theory/simulation.
Correct convergence of the original CL-FTS also remains unproven. Production FFT/C++/CUDA and the existing CL drift were unchanged in this work.

## Rerunning

Activate conda `polymerfts`. Numba was loaded from the existing local dependency directory:
`/home/yongdd/polymer/dh_salt_runs/exact_cl_diagnosis_20260907/python_deps`.
Add this directory and `devel/charged_polymers` to PYTHONPATH. Run calculations longer than one minute through SLURM.
Full run commands are preserved in `/home/yongdd/polymer/dh_salt_runs/positive_brush_20260907/jobs/*.sh`; job IDs are in `jobs.jsonl`.
The main runner is `run_positive_brush.py`: first use `--bulk` to create a bath summary, then pass it to the brush via `--bath`.
`--rho 0` needs no separate bath. The runner stores `checkpoint.npz` and RNG state and supports `--resume`.
Particle-only controls use `--mode particles`; exact-sum/detailed-balance checks use `check_positive_brush.py`.
Analysis: `python devel/charged_polymers/analyze_positive_brush.py /home/yongdd/polymer/dh_salt_runs/positive_brush_20260907 --prl-root /home/yongdd/polymer/dh_salt_runs/fig23_prl_20260907`.
This document is generated from analysis.json. Raw data are in samples.npz and summary.json.
