# Tools Reference

This document describes the utility scripts in the `tools/` directory for visualization, data conversion, and post-processing of simulation results.

## Table of Contents

1. [Visualization Tools](#1-visualization-tools)
2. [PSCF Interoperability](#2-pscf-interoperability)
3. [χN Renormalization](#3-χn-renormalization)
4. [Field Conversion](#4-field-conversion)
5. [File Format Reference](#5-file-format-reference)

---

## 1. Visualization Tools

### 1.1 Density Plots

| File | Language | Description |
|------|----------|-------------|
| `plot_2d_density.m` | MATLAB | Plot 2D monomer density from `fields.mat` |
| `plot_2d_density.ipynb` | Python | Same functionality as Jupyter notebook |
| `plot_2d_slice.m` | MATLAB | Extract and plot a 2D slice from 3D density data |
| `plot_3d_isodensity.m` | MATLAB | Render 3D isodensity surfaces with lighting |
| `plot_3d_isodensity.ipynb` | Python | Same functionality as Jupyter notebook |

**Usage (MATLAB):**
```matlab
% Copy script to directory containing fields.mat
cd /path/to/simulation
matlab -nodisplay -r "run('/path/to/tools/plot_3d_isodensity.m'); exit"
```

**Usage (Python):**
```bash
cd /path/to/simulation
jupyter notebook /path/to/tools/plot_2d_density.ipynb
```

**Output:** PNG image files (`2d_density.png`, `2d_slice_image.png`, `isodensity.png`)

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

---

## 2. PSCF Interoperability

Tools for converting between langevin-fts and [PSCF](https://github.com/dmorse/pscfpp) formats.

### 2.1 Export to PSCF (`conversion_to_pscf/`)

Convert langevin-fts Python scripts to PSCF input files.

| File | Description |
|------|-------------|
| `to_pscf.py` | Main conversion script |
| `scft_example.py` | Example SCFT script for testing |

**Usage:**
```bash
cd tools/conversion_to_pscf
python to_pscf.py --file_name /path/to/your_scft_script.py
```

**Output files:**
- `param`: PSCF parameter file (system definition, mesh, iteration settings)
- `in/omega`: PSCF field file (potential fields in PSCF format)

**Supported features:**
- Linear and branched polymer architectures
- Multi-monomer systems (arbitrary number of species)
- Automatic vertex re-numbering for branched polymers
- Orthorhombic lattice (default)

**Limitations:**
- Non-orthorhombic lattices not yet supported
- Space group symmetry not transferred

### 2.2 Import from PSCF (`read_pscf_output/`)

Convert PSCF output files to langevin-fts `.mat` format.

| File | Description |
|------|-------------|
| `to_matlab.py` | Main conversion script |
| `omega`, `rho`, `param` | Example PSCF files |

**Usage:**
```bash
cd tools/read_pscf_output
# Place omega and rho files from PSCF in this directory
python to_matlab.py
```

**Output:**
- `fields.mat`: MATLAB/Python compatible file with fields and densities

**Supported crystal systems:**
- Orthorhombic, cubic, tetragonal (cell parameters adjusted automatically)

---

## 3. χN Renormalization

Compute the renormalized Flory-Huggins parameter for ultraviolet (UV) divergence correction. Required for quantitative comparison with experiments or particle-based simulations.

The effective interaction parameter is:

$$\chi N_{\text{eff}} = z_\infty \times \chi N$$

where $z_\infty$ is the renormalization factor computed by these scripts.

### 3.1 Discrete Chain Model (`renormalization_discrete.m`)

**Recommended** for L-FTS simulations using the discrete chain model.

**Method:** Computes $z_\infty$ using discrete chain bond functions:

$$z_\infty = 1 - \frac{1 + 2\sum_{i=1}^{N_{\text{bond}}} P_i + P_{\text{cont}}}{\sqrt{\bar{N}} \cdot N \cdot \Delta V}$$

where $P_i$ are discrete bond contributions and $P_{\text{cont}}$ is a continuous chain correction for the tail.

**Usage:**
```matlab
% Run in directory containing fields_*.mat
matlab -nodisplay -r "run('/path/to/tools/renormalization_discrete.m'); exit"
```

**Output:**
```
z_inf: 0.XXXXXXX
```

**Requirements:**
- Conformationally symmetric chains only ($\epsilon = b_A/b_B = 1$)
- Reads `nbar`, `ds`, `lx`, `nx` from fields file

**References:**
- T. M. Beardsley and M. W. Matsen, "Calibration of the Flory-Huggins interaction parameter in field-theoretic simulations," *J. Chem. Phys.* **2019**, 150, 174902
- T. M. Beardsley and M. W. Matsen, "Fluctuation correction for the order–disorder transition of diblock copolymer melts," *J. Chem. Phys.* **2021**, 154, 124902

### 3.2 Continuous Chain Model (`renormalization_rpa.m`)

For continuous chain model (legacy, less commonly used).

**Method:** Uses RPA structure function integral:

$$z_\infty = 1 - \frac{1}{\sqrt{\bar{N}}} \frac{1}{8\pi^3 f(1-f)} \int_{-\pi/\Delta x}^{\pi/\Delta x} S_{\text{RPA}}(k) \, d^3k$$

**Usage:**
```matlab
% Run in directory containing fields_*.mat
matlab -nodisplay -r "run('/path/to/tools/renormalization_rpa.m'); exit"
```

**Requirements:**
- Continuous chain model only
- Conformationally symmetric chains only ($\epsilon = 1$)

**Reference:**
- B. Vorselaars, P. Stasiak, and M. W. Matsen, "Field-Theoretic Simulation of Block Copolymers at Experimentally Relevant Molecular Weights," *Macromolecules* **2015**, 48, 9071

---

## 4. Field Conversion

### 4.1 Monomer to Auxiliary Fields (`convert_into_auxiliary_fields.m`)

Convert monomer potential fields $(w_A, w_B, \ldots)$ to auxiliary potential fields $(w_{\text{aux}})$ using the inverse transformation matrix from eigenvalue decomposition.

$$\mathbf{w}_{\text{aux}} = \mathbf{A}^{-1} \mathbf{w}_{\text{monomer}}$$

**Usage:**
```matlab
% Run in directory containing fields_*.mat
matlab -nodisplay -r "run('/path/to/tools/convert_into_auxiliary_fields.m'); exit"
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
| `lx` | float array | Box dimensions in units of $b\sqrt{N}$ |
| `ds` | float | Contour step interval (= 1/N) |
| `nbar` | float | Invariant polymerization index $\bar{N}$ |
| `chain_model` | string | `"discrete"` or `"continuous"` |
| `monomer_types` | string | List of monomer type names (e.g., `"AB"`) |
| `w_A`, `w_B`, ... | float array | Monomer potential fields (flattened) |
| `phi_A`, `phi_B`, ... | float array | Monomer density fields (flattened) |
| `matrix_a` | float matrix | Transformation matrix $\mathbf{A}$ |
| `matrix_a_inverse` | float matrix | Inverse transformation matrix $\mathbf{A}^{-1}$ |

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
phi_A = reshape(data.phi_A, nx);
```

### 5.3 Structure Function Files

L-FTS simulations save structure function data in `structure_function_XXXXXX.mat`:

| Variable | Description |
|----------|-------------|
| `nx`, `lx` | Grid and box dimensions |
| `structure_function_A_A` | $\langle \phi_A(k) \phi_A(-k) \rangle$ |
| `structure_function_A_B` | $\langle \phi_A(k) \phi_B(-k) \rangle$ |
| `structure_function_B_B` | $\langle \phi_B(k) \phi_B(-k) \rangle$ |

Fields are stored in Fourier space with dimensions `[nx, ny, nz/2+1]` (real FFT format).
