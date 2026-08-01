# WTMD TODO

## Completed

- [x] Fix CUDA library path in run_wtmd.sh (added OpenHPC CUDA path)
- [x] Fix lfts.py initialization order bug (validation_runtime before compute_concentrations)
- [x] Run WTMD simulations with different update_freq (100, 200, 500, 1000)
- [x] Analyze partial results (~40% completion, 2M steps)
- [x] Document results in RESULTS.md

## Completed (2026-08-01 analysis)

- [x] Full F(Ψ) analysis: double well (Ψ≈1.62 dis / Ψ≈3.26 lam), barrier
      3.3–4.0 kT, ΔF = 0 within scatter → simulation sits at the ODT
- [x] ODT from ⟨∂H/∂χ⟩ extrapolation: **χN_ODT ≈ 17.14 ± 0.02**
      (parameter study over χN no longer needed for the ODT location)
- [x] update_freq robustness: 100–1000 all agree (earlier "trapped at freq=100"
      conclusion retracted)

## Pending Tasks

- [ ] **Compare with deep-langevin-fts**
  - Compare F(Ψ) curves, barrier height, and χN_ODT
- [ ] (optional) Additional independent runs to tighten χN_ODT
      (more efficient than extending the cancelled runs)

## Notes

- Simulations cancelled on 2026-02-07 at ~40% completion
- Results saved in `data_wtmd/` directory
- See `RESULTS.md` for detailed analysis
