# Tools Reference

This document describes the utility scripts in `scripts/analysis/` and `scripts/pscf/` for visualization, data conversion, and post-processing of simulation results.

## Table of Contents

1. [Visualization Tools](#1-visualization-tools)
2. [PSCF Interoperability](#2-pscf-interoperability)
3. [χN Renormalization](#3-χn-renormalization)
4. [Field Conversion](#4-field-conversion)
5. [File Format Reference](#5-file-format-reference)
6. [CL-FTS Drift Distribution](#6-cl-fts-drift-distribution)

---

## 1. Visualization Tools

### 1.1 Density Plots

| File | Language | Description |
|------|----------|-------------|
| `plot_2d_density.m` | MATLAB | Plot 2D monomer density from `fields.mat` |
| `plot_2d_density.ipynb` | Python | Plot `phi_A` and `phi_B` of a 2D `.mat` file (set the file name in the notebook) |
| `plot_2d_slice.m` | MATLAB | Extract and plot a 2D slice from 3D density data in `fields_002000.mat` |
| `plot_3d_isodensity.m` | MATLAB | Render 3D isodensity surfaces with lighting from `fields_002000.mat` |
| `plot_3d_isodensity.ipynb` | Python | Interactive Plotly isosurfaces of `fields_002000.mat` |

**Usage (MATLAB):**
```matlab
% Run in the directory that holds the input file (edit the file name in the script if needed)
cd /path/to/simulation
matlab -nodisplay -r "run('/path/to/scripts/analysis/plot_3d_isodensity.m'); exit"
```

**Usage (Python):**
```bash
cd /path/to/simulation
jupyter notebook /path/to/scripts/analysis/plot_2d_density.ipynb
```

**Output:** the MATLAB scripts write PNG image files (`2d_density.png`, `2d_slice_image.png`, `isodensity.png`); the notebooks only display the figures.

### 1.2 Structure Function

| File | Language | Description |
|------|----------|-------------|
| `plot_structure_function.m` | MATLAB | Plot spherically-averaged structure function S(k) |
| `plot_structure_function.ipynb` | Python | Same functionality as Jupyter notebook |

Reads structure function data from L-FTS simulations and computes spherically-averaged S(k) by grouping Fourier modes with equal |k|².

**Usage:**
```matlab
% Edit file_path and type_pair ("A_A", "A_B", or "B_B") in script
% Reads: data_simulation/structure_function_*.mat
matlab -nodisplay -r "run('plot_structure_function.m'); exit"
```

**Parameters to edit in script:**
- `file_path`: Path to structure function files
- `type_pair`: Correlation type (`"A_A"`, `"A_B"`, or `"B_B"`)

### 1.3 WTMD Statistics

| File | Language | Description |
|------|----------|-------------|
| `plot_wtmd_statistics.m` | MATLAB | Plot WTMD bias potential and probability distribution |
| `plot_wtmd_statistics.ipynb` | Python | Same functionality as Jupyter notebook |
| `plot_wtmd_chib_AB.ipynb` | Python | χN reweighting analysis for free energy computation |

These tools visualize Well-Tempered Metadynamics (WTMD) simulation output, including:
- **U(Ψ)**: Bias potential as a function of order parameter
- **U'(Ψ)**: Derivative of bias potential
- **P(Ψ)**: Probability distribution (from converged bias potential)
- **∂F/∂χN**: Free energy derivative for histogram reweighting

**Usage (basic statistics):**
```matlab
% Run in the directory that holds the wtmd_statistics_*.mat files
cd /path/to/simulation/data_simulation
matlab -nodisplay -r "run('/path/to/scripts/analysis/plot_wtmd_statistics.m'); exit"
```

**Usage (χN reweighting):**
```bash
cd /path/to/simulation
jupyter notebook /path/to/scripts/analysis/plot_wtmd_chib_AB.ipynb
# Edit chin (simulation χN), dchiN (shift), and data_dir in the notebook
```

**χN Reweighting Analysis:**

The `plot_wtmd_chib_AB.ipynb` notebook computes free energy at different χN values using histogram reweighting:

```math
F(\Psi; \chi N + \delta\chi N) = F(\Psi; \chi N) + \frac{\partial F}{\partial \chi N} \cdot \frac{\delta\chi N}{z_\infty}
```

where $`z_\infty`$ is the renormalization factor from the discrete chain model.

**Parameters in reweighting notebook:**
- `chin`: Simulation χN value
- `dchiN`: χN shift for reweighting (set to 0 for no shift)
- `z_inf`: Renormalization factor (default: 0.7691138 for discrete chain)
- `langevin_step`: Step number for statistics file

**Input files:** `wtmd_statistics_XXXXXX.mat` containing:
- `psi_range`: Order parameter values
- `u`: Bias potential U(Ψ)
- `up`: Bias potential derivative U'(Ψ)
- `I0`: Histogram of visited states
- `dH_psi_A_B`: Free energy derivative ∂F/∂χN
- `dT`: Temperature factor
- `dpsi`: Order parameter bin width

**Reference:**
- T. M. Beardsley and M. W. Matsen, "Well-tempered metadynamics applied to field-theoretic simulations of diblock copolymer melts," *J. Chem. Phys.* **2022**, 157, 114902

---

### 1.4 Polymer Architecture Drawing

The chain architectures are drawn as graphs (vertices = chain ends and junctions, edges = blocks labelled with their segment numbers) into `polymer_0.png`, `polymer_1.png`, ...:

- `LFTS` and `CLFTS` draw them into the recording directory (`recording.dir`) when a run starts;
- `SCFT` draws them at construction only into `polymer_figure_dir`, when the input gives one ([Parameters](Parameters.md)).

Constructing a calculation therefore writes no files into the current directory. To draw them elsewhere:

```python
from polymerfts import draw_polymer_architectures
draw_polymer_architectures(calc.polymers, calc.ds, calc.segment_lengths, directory="figures")
```

The colors of the first six monomer types are fixed; further types take colors from a fixed local random stream, which leaves the global NumPy random state unchanged.

Uses networkx and matplotlib; a graphviz layout is used when pygraphviz is installed.

### 1.5 VTK Export for ParaView

`save_vtk` writes fields as a legacy ASCII VTK file:

```python
from polymerfts import save_vtk
result = calc.run(initial_fields=w, return_result=True)
save_vtk("phi.vtk", result.phi, calc.prop_solver.get_lx(), nx=calc.prop_solver.nx,
         angles=calc.prop_solver.get_angles_degrees())
```

- `fields` is one array (named by `field_name`, default `phi`) or a dictionary `{name: array}`, which puts several fields in one file. Arrays have the grid shape, or are flat when `nx` is given; 1D and 2D grids are padded to 3D.
- Grid points sit at the cell centres, `(i + 1/2) * lx / nx`, as in the solvers. The outline ParaView draws through the points therefore lies half a cell inside the box; the walls of a non-periodic box are at 0 and `lx`.
- With `angles` (degrees) of a non-orthogonal cell, the file is a structured grid with explicit point coordinates along the lattice vectors (`a` along x, `b` in the xy plane, as in `ComputationBox`).
- Values are written as single-precision text; complex fields are rejected (write the real and imaginary parts separately).

## 2. PSCF Interoperability

Tools for converting between PolymerFTS and [PSCF](https://github.com/dmorse/pscfpp) formats.

### 2.1 Export to PSCF

```bash
python scripts/pscf/pscf_converter.py export fields.mat -o pscf-input
```

The converter reads saved SCFT results or a parameter file with separate fields.
It writes `param`, `command`, `in/omega`, and a monomer-name mapping in
`metadata.json`. It supports continuous Gaussian chains, mixtures, linear and
branched polymers, and 1D/2D/3D periodic cells. Python input is read statically;
conversion never runs a simulation.

### 2.2 Import from PSCF

```bash
python scripts/pscf/pscf_converter.py import pscf-input/out/omega \
    --rho pscf-input/out/rho --metadata pscf-input/metadata.json -o fields.mat
```

Use PSCF r-grid fields produced by `WRITE_W_RGRID` and `WRITE_C_RGRID`.
`--rho` is optional; `--monomers A B ...` supplies column names when export
metadata is unavailable. Grid order, cell geometry, field counts and finite
values are checked before writing. Existing outputs are preserved.

### 2.3 Generate a polymerfts execution script

```bash
python scripts/pscf/pscf_converter.py import-script pscf-input/param \
    --command pscf-input/command --monomers A B -o polymerfts-job
```

This creates `simulation.py`, bundled `initial.mat` fields and `conversion.json`
for one `READ_W_RGRID` / `ITERATE` / save / `FINISH` SCFT calculation. Run the
script separately with polymerfts installed. Results are complete polymerfts
`.mat` snapshots. Sweeps, basis input, other iterators, partially flexible cells
and unsupported physical models are rejected; the guide lists the restrictions.

See the [PSCF conversion guide](../getting-started/PscfConversion.md) for dependencies,
cell conventions, supported models, migration from the old tool paths, and
examples. The converter uses PSCF's full-grid iterator; it does not impose the
source space group or convert discrete/complex models by approximation.

---

## 3. χN Renormalization

On a grid, the interaction of a field-theoretic simulation is renormalized by the ultraviolet (UV) fluctuations. To linear order, the effective Flory-Huggins parameter is

```math
\chi N = z_\infty \, \chi_b N
```

where the bare $`\chi_b N`$ is the `chi_n` of the simulation. In CL-FTS, the fluctuations of the pressure field also change the chain size. The bare segment length $`a_b`$ must then satisfy $`R_0 = a N^{1/2}`$ ([theory, Section 5](../theory/LangevinFieldTheoreticSimulation.md#5-renormalization-and-calibration), which gives the formulas the script evaluates). Both corrections are needed to compare simulations with experiments, particle-based simulations or universal theory.

### 3.1 Usage (`renormalization.py`)

```bash
python scripts/analysis/renormalization.py data_simulation/fields_010000.mat
python scripts/analysis/renormalization.py --chain-model discrete --n 90 --nbar 1e5 \
    --nx 32 32 32 --lx 4.5 4.5 4.5 --zeta-n 90 --a-int 0.02 --target 9
```

The script uses NumPy and SciPy and does not need the compiled package.

**Inputs.** A fields file supplies `chain_model`, `nx`, `lx`, `ds`, `nbar`, `zeta_n`, `smearing`, `chi_n` and `segment_lengths`. Options override these values or replace the file:

| Option | Meaning |
|--------|---------|
| `--chain-model`, `--nx`, `--lx`, `--n`, `--nbar` | Chain model, grid, box in $`R_0`$, segments $`N = 1/\Delta s`$ (discrete chains), $`\bar{N}`$ |
| `--zeta-n` | Compressibility $`\zeta N`$; `inf` gives the incompressible model |
| `--a-int`, or `--k-int` and `--dk-int` | Gaussian or sigmoidal smearing as in [Parameters](Parameters.md#smearing); `--a-int 0` removes smearing |
| `--target` | Prints the $`\chi_b N`$ that gives this $`z_\infty \chi_b N`$ |
| `--quadrature` | Gauss-Legendre points per axis (default 64) |

**Output:** the example command above prints

```
discrete chains, N = 90, Nbar = 100000, grid [32, 32, 32], box [4.5, 4.5, 4.5], zeta N = 90, smearing: {'type': 'gaussian', 'a_int': 0.02}
z_inf = 0.9595665
a_b/a = 0.9930857  (CL-FTS segment_lengths; L-FTS and SCFT use a_b = a)
chi_b N for z_inf chi_b N = 9: 9.3792356
```

With a fields file, the script also prints $`z_\infty \chi_b N`$ for each `chi_n` pair. For CL-FTS, multiply all `segment_lengths` by $`a_b/a`$.

**Requirements and accuracy:**
- 3D orthogonal box;
- conformationally symmetric chains (equal `segment_lengths`);
- one smearing function for all monomer types: `a_int` is a number, and the per-type
  lengths of charged CL-FTS are rejected.

Both integrands approach $`c/k^2`$ at $`k = 0`$ ($`c = 12`$ for $`z_\infty`$, $`c = 2`$ for $`a_b/a`$, since $`\Gamma(0) = 1`$). The script integrates this term exactly, by one-dimensional quadrature of $`\int d\mathbf{k}/k^2 = \int_0^\infty ds \prod_\nu (\pi/s)^{1/2} \operatorname{erf}(\pi s^{1/2}/\Delta_\nu)`$, and the Gauss-Legendre quadrature only sees the bounded remainder. With the default 64 points per axis, $`z_\infty`$ agrees with the bond sum to $`10^{-13}`$. With Gaussian smearing and compressibility, the change on refinement is below $`10^{-12}`$. Sigmoidal smearing depends on $`\vert k \vert`$, so its remainder is not smooth at $`k = 0`$; there the error is below $`10^{-9}`$. `tests/scripts/TestRenormalization.py` checks the results against three references:

- the bond sum of Beardsley and Matsen (2019);
- the limit $`N \to \infty`$;
- Fig. 2 of Matsen et al. (2026).

**References:**
- T. M. Beardsley and M. W. Matsen, "Calibration of the Flory-Huggins interaction parameter in field-theoretic simulations," *J. Chem. Phys.* **2019**, 150, 174902
- T. M. Beardsley and M. W. Matsen, "Fluctuation correction for the order–disorder transition of diblock copolymer melts," *J. Chem. Phys.* **2021**, 154, 124902
- B. Vorselaars, P. Stasiak, and M. W. Matsen, "Field-Theoretic Simulation of Block Copolymers at Experimentally Relevant Molecular Weights," *Macromolecules* **2015**, 48, 9071
- M. W. Matsen, J. Ramírez, J. D. Willis, and P. D. Pina, "Accessing the universal phase behavior of block copolymer melts with complex-Langevin field-theoretic simulations," *J. Chem. Phys.* **2026**, 164, 014905

---

## 4. Field Conversion

### 4.1 Monomer to Auxiliary Fields (`convert_into_auxiliary_fields.m`)

Convert monomer potential fields $`(w_A, w_B, \ldots)`$ to auxiliary potential fields $`(w_{\text{aux}})`$ using the inverse transformation matrix from eigenvalue decomposition.

```math
\mathbf{w}_{\text{aux}} = \mathbf{A}^{-1} \mathbf{w}_{\text{monomer}}
```

**Usage:**
```matlab
% Run in directory containing fields_*.mat
matlab -nodisplay -r "run('/path/to/scripts/analysis/convert_into_auxiliary_fields.m'); exit"
```

**Input:** `fields_*.mat` containing:
- `w_A`, `w_B`, ... (monomer potential fields)
- `matrix_a_inverse` (inverse transformation matrix, saved by simulation)

**Output:** `w_aux.mat` containing the auxiliary fields

**Reference:**
- D. Düchs, K. T. Delaney, and G. H. Fredrickson, "A multi-species exchange model for fully fluctuating polymer field theory simulations," *J. Chem. Phys.* **2014**, 141, 174103

---

## 5. File Format Reference

### 5.1 fields.mat Structure

Simulation output files (`fields_XXXXXX.mat`) contain:

| Variable | Type | Description |
|----------|------|-------------|
| `nx` | int array | Grid dimensions [nx, ny, nz] |
| `lx` | float array | Box dimensions in units of $`b\sqrt{N}`$ |
| `ds` | float | Contour step interval (= 1/N) |
| `nbar` | float | Invariant polymerization index $`\bar{N}`$ |
| `chain_model` | string | `"discrete"` or `"continuous"` |
| `monomer_types` | string | List of monomer type names (e.g., `"AB"`) |
| `w_A`, `w_B`, ... | float array | Monomer potential fields (flattened) |
| `phi_A`, `phi_B`, ... | float array | Monomer density fields (flattened) |
| `matrix_a` | float matrix | Transformation matrix $`\mathbf{A}`$ |
| `matrix_a_inverse` | float matrix | Inverse transformation matrix $`\mathbf{A}^{-1}`$ |

L-FTS and CL-FTS checkpoints additionally hold `initial_params`, `langevin_step`, `dt`, `eigenvalues`, `aux_fields_real`/`aux_fields_imag`, the noise array `normal_noise_prev` and the random-generator state (`random_generator`, `random_state_state`, `random_state_inc`), which `continue_run()` restores. CL-FTS stores complex fields as `w_A_real`/`w_A_imag` and `phi_A_real`/`phi_A_imag`.

**SCFT results** (`SCFT.save_results(path)`, `.mat`, `.json` or `.yaml` by extension) contain `initial_params`, `dim`, `nx`, `lx`, `angles`, `monomer_types`, `chi_n`, `chain_model`, `ds`, `eigenvalues`, `matrix_a`, `matrix_a_inverse`, `simulation_metadata_json` (run metadata, including whether the run converged), `free_energy`, `free_energy_absolute`, `free_energy_homogeneous`, `partition_functions`, `polymer_lengths`, `polymer_volume_fractions`, `chemical_potentials` and `grand_potential` (omitted for grafted chains or supplied initial propagators), `w_<type>` and `phi_<type>` on the full grid (expanded from the reduced basis when a space group was used), and `space_group_symbol`/`space_group_hall_number` if one was used. Load `.json` with `json.load` and `.yaml` with `yaml.safe_load`; arrays are stored as lists there. The `.json` file is strict JSON: a non-finite number, such as the entry `-inf` of `chemical_potentials` for a species with volume fraction 0, is stored as `{"nonfinite_float": "-inf"}`.

### 5.2 Reading Files

**Python (SciPy):**
```python
from scipy.io import loadmat, savemat

data = loadmat("fields_000200.mat", squeeze_me=True)
nx = data['nx']
lx = data['lx']
phi_A = data['phi_A'].reshape(nx)
```

**MATLAB:**
```matlab
data = load("fields_000200.mat");
nx = data.nx;
lx = data.lx;
phi_A = permute(reshape(data.phi_A, flip(nx)), numel(nx):-1:1);  % stored with the last index fastest
```

### 5.3 Structure Function Files

L-FTS simulations save structure function data in `structure_function_XXXXXX.mat`:

| Variable | Description |
|----------|-------------|
| `nx`, `lx` | Grid and box dimensions |
| `structure_function_A_A` | $`S_{AA}(\mathbf{k}) = V \sqrt{\bar{N}} \langle \tilde{\mu}_A(\mathbf{k}) \tilde{\phi}_A(-\mathbf{k}) \rangle`$ |
| `structure_function_A_B` | $`S_{AB}(\mathbf{k}) = V \sqrt{\bar{N}} \langle \tilde{\mu}_A(\mathbf{k}) \tilde{\phi}_B(-\mathbf{k}) \rangle`$ |
| `structure_function_B_B` | $`S_{BB}(\mathbf{k}) = V \sqrt{\bar{N}} \langle \tilde{\mu}_B(\mathbf{k}) \tilde{\phi}_B(-\mathbf{k}) \rangle`$ |

$`\tilde{\mu}_i`$ is the Fourier transform of the chemical potential field of monomer $`i`$, $`\mu_i = \sum_k A_{ik}\Omega_k/\lambda_k`$, and the average runs over the steps at which the structure function is computed within one recording period; $`\bar{N}`$ is `nbar`; see the [estimator](../theory/LangevinFieldTheoreticSimulation.md#42-estimator). With smearing in a periodic box the recorded $`S`$ is divided by $`\Gamma(\mathbf{k})`$ (modes with $`\Gamma < 10^{-12}`$ are recorded as 0); with walls, where $`\Gamma`$ is defined in the cosine basis, the estimator is recorded as it is.

Fields are stored in Fourier space with dimensions `[nx, ny, nz/2+1]` (real FFT format).

CL-FTS uses complex fields, so its files differ from those of L-FTS:
- They hold full FFT arrays of shape `[nx, ny, nz]`, with the parts `structure_function_A_B_real` and `structure_function_A_B_imag`.
- With electrostatics they also hold `charge_structure_function_real` and `_imag`: the structure function of the smeared charge density. At each wavevector it combines the mixed and the field-field estimators of the electrostatic potential with the weight of least variance; the weight is stored as `charge_structure_function_weight`. It is less noisy than $`\sum_{ij} z_i z_j S_{ij}`$, by orders of magnitude where screening is strong. The two forms alone are stored as `charge_structure_function_mixed_*` and `charge_structure_function_field_field_*`: in the Gaussian regime only they show how well ψ is sampled.

See [Structure Function Estimators](../theory/StructureFunctionEstimators.md) for all three estimators.

## 6. CL-FTS Drift Distribution

Complex Langevin averages are correct only if the distribution of the drift magnitude falls off exponentially or faster at large magnitude (Nagata, Nishimura and Shimasaki, Phys. Rev. D **94**, 114515 (2016)). CL-FTS records the largest drift over the grid of every Langevin step in the `dH_XXXXXX.mat` files: `drift_max_step_history` with one column per auxiliary field, and with electrostatics `psi_drift_max_step_history`. `drift_tail.py` reads them for one recording directory:

```bash
python scripts/analysis/drift_tail.py data_simulation
python scripts/analysis/drift_tail.py data_simulation --skip 20000 --plot drift_tail.png
```

`--skip` leaves out the equilibration steps. For each field the script prints the number of steps, the median and the maximum, the fraction $`S(u)`$ of steps with a drift above 10, 100 and 1000 times the median, and the log-log slope of $`S(u)`$ per interval of $`u`$. An exponential tail steepens without bound on a log-log plot. A slope that stays constant over the largest decades is a power-law tail: near a simple zero of a partition function $`S(u) \propto u^{-2}`$. The weight of such a tail sets the size of the bias, and a finite run cannot show an exponential tail, so compare with an independent reference where one exists ([ComplexLangevinSimulation.md](../theory/ComplexLangevinSimulation.md#32-test-against-exact-averages), Section 3.2). The end of a CL-FTS run prints, for the steps of that run, $`S(u)`$ at 10 and 100 times the median and the slope between them.

