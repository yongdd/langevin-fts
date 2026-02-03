# Development Scripts

This directory contains development and utility scripts that are not part of the main library.

## Scripts

| Script | Description |
|--------|-------------|
| `check_latex.py` | Validate LaTeX syntax in documentation using KaTeX |
| `benchmark_crysfft_triplet.py` | Benchmark CrysFFT triplet algorithm performance |
| `benchmark_reduced_basis.py` | Benchmark reduced basis space group operations |
| `benchmark_space_group_speed.py` | Benchmark space group symmetry performance |
| `generate_phase_fields.py` | Generate initial field configurations for various phases |
| `slurm_run.sh` | SLURM job submission template (not tracked in git) |

## Usage

### LaTeX Validation

```bash
# Requires: npm install -g katex
python scripts/check_latex.py
```

### Benchmarks

```bash
# Run with SLURM for long benchmarks
sbatch scripts/slurm_run.sh

# Or run directly
python scripts/benchmark_space_group_speed.py
```

### Generate Phase Fields

```bash
python scripts/generate_phase_fields.py
```
