# Direct comparison of 1D sampling and PRL-type SCFT

The samples are closer to mean-field SCFT with correlations disabled than to smoothed-LDA SCFT with correlations.
This is not a comparison of sampling methods for the same action.

## Comparison method

- MC: the completed finite planar sampler with N=100, A=40 nm², L=200 nm, M=400.
- LDA: the existing smoothed-LDA SCFT using PRL parameters. Equivalence to the full nonlocal theory in the paper has not been validated.
- MF: the same SCFT with correlation self-energies disabled (uK=0), retaining the mean Coulomb potential and canonical C.
- SCFT was recomputed at the measured MC bulk concentrations for 15 nonzero salt points. The MF salt-free reference was solved separately.
- Retained parameters: N=100, b=1 nm, σ=.1 nm⁻², v=1 nm³, χ=.5, ℓB=.7 nm, a=.25 nm, and canonical C with Nσ=10 nm⁻².
- Concentration means bulk cation molarity, distinct from ionic strength. 1 M = .6022 nm⁻³.
- Height is h=2∫zρP dz/(σN). Normalized curves use each model's independently computed h0, without matching to a shared h0.
- SCFT M=1600 densities were averaged over four cells onto MC M=400. Γ uses the same .5 peak core definition.
- SCFT potentials were interpolated to z=.25/199.75 nm and compared as differences at the same locations as MC, not as absolute gauges.
- MC error bars are 1 SE and exclude model bias and SCFT approximation errors.

## Height [nm]

| z+ | Target ρb [M] | Matched ρb [M] | MC ± SE | LDA SCFT | MF SCFT | MC−LDA | MC−MF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 58.561 ± 0.351 | 50.911 | 59.602 | +7.650 | -1.041 |
| 1 | 0.001 | 0.000995724 | 57.517 ± 0.355 | 49.698 | 58.383 | +7.819 | -0.867 |
| 1 | 0.01 | 0.01004805 | 54.627 ± 0.257 | 46.342 | 54.827 | +8.285 | -0.199 |
| 1 | 0.1 | 0.1001598 | 43.878 ± 0.173 | 37.104 | 43.745 | +6.774 | +0.133 |
| 1 | 1 | 0.9999946 | 28.489 ± 0.213 | 25.785 | 27.795 | +2.704 | +0.694 |
| 1 | 10 | 9.996082 | 23.572 ± 0.141 | 22.366 | 22.490 | +1.206 | +1.082 |
| 2 | 0.001 | 0.00101316 | 54.308 ± 0.223 | 44.068 | 55.637 | +10.241 | -1.329 |
| 2 | 0.01 | 0.009974988 | 46.019 ± 0.211 | 29.379 | 46.943 | +16.640 | -0.924 |
| 2 | 0.1 | 0.10019 | 33.043 ± 0.186 | 19.511 | 33.037 | +13.532 | +0.006 |
| 2 | 1 | 0.9999455 | 24.858 ± 0.170 | 21.180 | 23.931 | +3.678 | +0.927 |
| 2 | 10 | 10.00168 | 23.054 ± 0.165 | 21.930 | 22.015 | +1.124 | +1.038 |
| 3 | 0.001 | 0.001005687 | 50.213 ± 0.290 | 29.942 | 52.821 | +20.272 | -2.607 |
| 3 | 0.01 | 0.01003352 | 39.282 ± 0.354 | 14.121 | 40.366 | +25.161 | -1.084 |
| 3 | 0.1 | 0.09985221 | 28.597 ± 0.330 | 14.899 | 28.395 | +13.698 | +0.203 |
| 3 | 1 | 1.000078 | 23.796 ± 0.189 | 20.005 | 22.881 | +3.791 | +0.916 |
| 3 | 10 | 10.00312 | 22.797 ± 0.294 | 21.842 | 21.897 | +0.956 | +0.901 |

## Charge and potential: trivalent salt

| Target ρb [M] | MC Γ ± SE | LDA Γ | MF Γ | MC Ψ ± SE | LDA Ψ | MF Ψ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.7774 ± 0.0029 | 0.4179 | 0.8396 | -11.8280 ± 0.3145 | -5.6654 | -14.4799 |
| 0.01 | 0.4856 ± 0.0056 | 0.0480 | 0.5089 | -4.5453 ± 0.3697 | -0.2740 | -5.2131 |
| 0.1 | 0.3098 ± 0.0040 | -0.3966 | 0.3238 | -0.5994 ± 0.2435 | 0.7841 | -0.8601 |
| 1 | 0.3292 ± 0.0033 | -0.3357 | 0.3292 | 0.0296 ± 0.1224 | 0.0964 | -0.0930 |
| 10 | 0.3500 ± 0.0152 | 0.2238 | 0.3349 | -0.0342 ± 0.1406 | -0.0063 | -0.0093 |

## Interpretation

The large multivalent contraction and re-expansion in LDA-SCFT are absent from MC. MC heights and densities are much closer to MF.
MC and MF nevertheless differ: finite A, solvent entropy from integer cell occupancy, grid, charge smearing, and actual 1D fluctuations all matter.
A 1D mean density ρ(z) differs from restricting charge fluctuations to 1D. The PRL correlation theory includes a 3D Green function.
Current MC samples a planar Coulomb action retaining only transverse wave number q_parallel=0. More iterations cannot restore excluded transverse correlation modes.
These results motivate investigating model differences first; this comparison does not establish an MC algorithm error or failure of the original CL.
The paper neglects uP^ex but retains the individual-charge term uP^0. That omission alone is not an established cause of the discrepancy.
[Paper and Eqs. 3–5](https://arxiv.org/html/2404.09103v3), [saved SI](../../references/2025_prl_duan_si.pdf).
PRL squares in the plots are digitized theory curves. The paper uses a Gibbs dividing-surface height; our height is a first moment.

## Numerical checks

- Maximum full SCFT residual 8.762e-08, independent log-chain density error 2.017e-13, contour Q error 1.566e-12.
- Matching nominal concentrations to the actual MC bath changes LDA height by at most 0.053068 nm.
- Maximum profile rebinning integral error 4.547e-13. Polymer/C integrals were each checked against 10 nm⁻².
- The failed initial MF salt-free attempt is preserved. A retry from the existing smooth LDA solution converged without changing residual thresholds or equations.

Raw data: `/home/yongdd/polymer/dh_salt_runs/sampling_scft_20260907`. `cases.json` records every MC–SCFT mapping.
`comparison.json`, `comparison.csv`, and `comparison_profiles.npz` contain numbers and full profiles. `jobs.jsonl` and each `run.sh` preserve execution details.
[Height plot](prl_comparison_figs/sampling_scft_20260907/mc_scft_height.pdf) · [Charge/potential plot](prl_comparison_figs/sampling_scft_20260907/mc_scft_charge.pdf) · [Density profiles](prl_comparison_figs/sampling_scft_20260907/mc_scft_profiles.pdf)
