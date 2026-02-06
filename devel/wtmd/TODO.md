# WTMD TODO

## Completed

- [x] Fix CUDA library path in run_wtmd.sh (added OpenHPC CUDA path)
- [x] Fix lfts.py initialization order bug (validation_runtime before compute_concentrations)
- [x] Run WTMD simulations with different update_freq (100, 200, 500, 1000)
- [x] Analyze partial results (~40% completion, 2M steps)
- [x] Document results in RESULTS.md

## Pending Tasks

- [ ] **Compare with deep-langevin-fts**
  - Compare F(Ψ) curves
  - Verify order parameter Ψ trajectory matches
  - Check free energy barrier at ODT

- [ ] **Parameter study**
  - Test different χN values near ODT
  - Verify ODT location from F(Ψ) vs Ψ

## Notes

- Simulations cancelled on 2026-02-07 at ~40% completion
- Results saved in `data_wtmd/` directory
- See `RESULTS.md` for detailed analysis
