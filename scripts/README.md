# Development Scripts

This directory contains development and utility scripts that are not part of the main library.

## Scripts

| Script | Description |
|--------|-------------|
| `simulation_gui.py` | Browser editor for SCFT/L-FTS/CL-FTS scripts (beta); see the [GUI guide](../docs/getting-started/SimulationGUI.md) |
| `check_latex.py` | Check the math in the Markdown files: GitHub-safe syntax (`` $`...`$ ``, `` ```math ``) and, with the KaTeX CLI, the TeX itself |
| `benchmark_crysfft_triplet.py` | Benchmark CrysFFT triplet algorithm performance |
| `benchmark_reduced_basis.py` | Benchmark reduced basis space group operations |
| `benchmark_space_group_speed.py` | Benchmark space group symmetry performance |
| `generate_phase_fields.py` | Generate initial field configurations for various phases |
| `reconverge_phase_fixtures.py` | Re-converge `tests/data/*.mat` under the current grid/symmetry convention and print the `expected_fe` values for `TestSpaceGroupPhases.py` |
| `install_versioned.sh` | Build a wheel for the current version and install it into its own venv (`~/wheels`, `~/venvs/polymerfts-<version>`), so several versions can be kept side by side; see docs/getting-started/Installation.md |
| `slurm_run.sh` | SLURM job submission template |

## Usage

### Simulation Script Builder (beta)

```bash
python3 scripts/simulation_gui.py
```

Open `http://127.0.0.1:8765` in a browser. For a remote server, connect with
`ssh -L 8765:127.0.0.1:8765 user@server` from your computer, then start the editor
in that SSH session. VS Code Remote-SSH port forwarding also works.

The editor uses only the Python standard library and creates SCFT/L-FTS/CL-FTS
scripts. **Save on server** writes to the directory where the editor was started
by default; **Download locally** saves to your browser's download location.
Phase-specific initial fields currently support **lamella and gyroid**, with
random and `.mat` inputs also available. More phase templates are planned.

See [Simulation Script Builder (beta)](../docs/getting-started/SimulationGUI.md)
for SSH setup, box and polymer editing, space groups, CrysFFT, save locations,
current limitations and development checks.

### Math Check

```bash
npm install -g katex   # optional: without it the TeX syntax is not checked
python scripts/check_latex.py
```

The rules it enforces are in CLAUDE.md (Math in Documentation).

### Benchmarks

`benchmark_space_group_speed.py` times one phase (`--phase BCC|DG|Sigma|HCP`) on one
platform (`--platform cpu-fftw|cuda`) with the space group on or off; all three
options are required. It prints a JSON summary and also writes it to `--output`:

```bash
python scripts/benchmark_space_group_speed.py --phase BCC --platform cuda --space-group on --output bcc_cuda_on.json
python scripts/benchmark_space_group_speed.py --phase BCC --platform cuda --space-group off --output bcc_cuda_off.json
```

For long benchmarks submit one SLURM job per combination: replace the `python` line
of the template `slurm_run.sh` with one of the commands above and run
`sbatch scripts/slurm_run.sh`.

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
python scripts/generate_phase_fields.py        # all: BCC, FCC, SC, A15, Sigma, HCP, PL
python scripts/generate_phase_fields.py BCC    # one phase
```

The converged fields are written to `scripts/data/<PHASE>.mat` (the directory is
created if missing). Nothing reads them there: the test fixtures are
`tests/data/<PHASE>.mat`, and the examples in `examples/scft/phases/` build their own
initial fields or load the `.mat` files next to them. To replace a test fixture, copy
the file to `tests/data/` and re-converge it with `reconverge_phase_fixtures.py`
(above), which also prints the expected free energies for the test.
