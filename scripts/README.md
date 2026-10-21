# Development Scripts

This directory contains development and utility scripts that are not part of the main library.

## Scripts

| Script | Description |
|--------|-------------|
| `check_latex.py` | Check the math in the Markdown files: GitHub-safe syntax (`` $`...`$ ``, `` ```math ``) and, with the KaTeX CLI, the TeX itself |
| `benchmark_crysfft_triplet.py` | Benchmark CrysFFT triplet algorithm performance |
| `benchmark_reduced_basis.py` | Benchmark reduced basis space group operations |
| `benchmark_space_group_speed.py` | Benchmark space group symmetry performance |
| `generate_phase_fields.py` | Generate initial field configurations for various phases |
| `reconverge_phase_fixtures.py` | Re-converge `tests/data/*.mat` under the current grid/symmetry convention and print the `expected_fe` values for `TestSpaceGroupPhases.py` |
| `install_versioned.sh` | Build a wheel for the current version and install it into its own venv (`~/wheels`, `~/venvs/polymerfts-<version>`), so several versions can be kept side by side; see docs/getting-started/Installation.md |
| `slurm_run.sh` | SLURM job submission template |

## Usage

### Math Check

```bash
npm install -g katex   # optional: without it the TeX syntax is not checked
python scripts/check_latex.py
```

The rules it enforces are in CLAUDE.md (Math in Documentation).

### Benchmarks

```bash
# Run with SLURM for long benchmarks
sbatch scripts/slurm_run.sh

# Or run directly
python scripts/benchmark_space_group_speed.py
```

### Re-converge Test Fixtures

`tests/TestSpaceGroupPhases.py` expects every fixture in `tests/data/` to be already
converged (1-2 iterations). After a change of the discretization or the symmetry
convention, re-converge them (one SLURM job per phase is fine) and paste the printed
values into the test:

```bash
python scripts/reconverge_phase_fixtures.py --platform cuda        # all phases
python scripts/reconverge_phase_fixtures.py --verify               # check: iterations and FE
```

### Generate Phase Fields

```bash
python scripts/generate_phase_fields.py
```
