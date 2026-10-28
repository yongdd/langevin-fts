# PSCF Conversion

[`scripts/pscf/pscf_converter.py`](../../scripts/pscf/pscf_converter.py) converts saved
polymerfts SCFT results to periodic PSCF input files, converts PSCF r-grid
fields back to `.mat`, and builds runnable polymerfts SCFT scripts from supported
PSCF `param` and `command` files. Conversion does not construct a solver or run a
simulation.

The converter targets the periodic `pscf_rpc` / `pscf_rpg` input formats described
in the [PSCF manual](https://dmorse.github.io/pscfpp-man/user_field_rgrid_page.html).
It supports continuous Gaussian chains, mixtures, linear and acyclic branched
polymers, multi-character monomer identifiers, and 1D, 2D and 3D periodic cells.

## Dependencies

Run the scripts from a repository checkout with NumPy and SciPy installed. No
compiled polymerfts backend or GPU is needed. YAML input additionally requires
PyYAML. Exporting a space-group input requires Python spglib to identify the
crystal system, unless `--lattice` is specified explicitly.
Running a generated polymerfts script separately requires an installed
polymerfts package with a suitable computational backend.

## Export an SCFT result

Use the file saved by `SCFT.save_results`, in `.mat`, `.json` or `.yaml` format:

```bash
python scripts/pscf/pscf_converter.py export /path/to/fields.mat -o /path/to/pscf-input
```

The output directory must be new. The converter validates inputs before creating
it and refuses to replace an existing directory. It writes:

| File | Purpose |
|------|---------|
| `param` | Mixture, interactions, periodic domain and `AmIteratorGrid` settings |
| `in/omega` | Initial potential fields in PSCF r-grid format |
| `command` | Read fields, iterate, and write final fields, densities and thermodynamics |
| `metadata.json` | Mapping from PSCF column indices to the original monomer names |
| `out/` | Directory for the later PSCF calculation's output |

Grid dimensions and final cell lengths/angles come from the result's top-level
metadata, which may differ from the initial cell in `initial_params` after
relaxation. An inconsistent grid is rejected. Block contour lengths, segment
lengths and interaction parameters retain polymerfts reference-chain units;
PSCF's continuous-chain reference length is taken as 1.

The generated command file can be run separately with a compatible PSCF build,
for example for a 3D problem:

```bash
cd /path/to/pscf-input
pscf_rpc -d 3 -p param -c command
```

Choose `-d 1` or `-d 2` for lower-dimensional periodic problems. This is a periodic
PSCF input, including in 1D; it is not an input for the distinct `pscf_r1d` program.
Use the appropriate job scheduler for calculations on a cluster.

### Separate parameters and initial fields

A configuration dictionary in JSON or YAML can be paired with initial fields:

```bash
python scripts/pscf/pscf_converter.py export params.json --fields initial.mat -o pscf-input
```

The parameter dictionary must include `nx`, `lx`, `ds`, `chain_model`,
`segment_lengths`, `chi_n` and `polymers`. Fields must include a finite real
`w_<name>` array for every monomer, with one value per grid point.
Monomer identifiers must be nonempty and contain no whitespace.
The canonical names `polymers`, `relax_cell`, `stress_scale` and
`space_group.hall_number` are supported alongside the legacy input aliases.
Supplying both an alias and its canonical name is an error.

A Python parameter file can also be used with `--fields`. The converter reads a
module-level `params` dictionary and simple numerical expressions such as
`f = 0.36` and `1-f` statically. It never executes the Python file. Runtime
configuration loading, recognized parameter mutations (including through
container aliases), and constants changed by unsupported assignments or control
flow are rejected. Literal helper containers and indexing are supported;
container method calls are rejected. Other runtime code, including Python
initial-field generation and calls to external functions, is not executed or
extracted. Use a saved result or JSON/YAML configuration for a model assembled
or changed at runtime.

To try a complete conversion without running a simulation:

```bash
python scripts/pscf/pscf_example.py -o initial.mat
python scripts/pscf/pscf_converter.py export initial.mat -o pscf-input
python scripts/pscf/pscf_converter.py import pscf-input/in/omega \
    --metadata pscf-input/metadata.json -o roundtrip.mat
```

The example creates small lamellar initial fields on a non-cubic grid.

## Import PSCF fields

Export the potential and optional density fields from PSCF using
`WRITE_W_RGRID` and `WRITE_C_RGRID`, then run:

```bash
python scripts/pscf/pscf_converter.py import pscf-input/out/omega \
    --rho pscf-input/out/rho --metadata pscf-input/metadata.json -o fields.mat
```

For files produced independently of this converter, supply monomer names in
PSCF column order:

```bash
python scripts/pscf/pscf_converter.py import omega --rho rho \
    --monomers A B C -o fields.mat
```

Without a name mapping the defaults are `A` through `Z`, then `M27`, `M28`, etc.
The importer checks both headers independently and rejects mismatched grids,
cells, groups or monomer counts. It also rejects missing or extra data rows,
wrong column counts, nonfinite values, and basis/Fourier-space files. Both
`mesh` and the legacy `ngrid` header labels are accepted. Existing output files
are never overwritten.

The `.mat` output contains `nx`, `lx`, `angles`, `monomer_types`, `w_<name>` and,
when `--rho` is supplied, `phi_<name>`. Fields are flattened in polymerfts C order.
The source lattice and group label are recorded as `pscf_lattice` and
`pscf_group_name`; no symmetry constraint is enabled automatically. The output is
an initial-field file, not a Langevin checkpoint or a complete polymer model.
The r-grid header does not identify a chain model: `chain_model` is included only
when supplied by the export metadata or `--chain-model`.

## Convert a PSCF calculation to a Python script

Supply the PSCF parameter and command files together. The initial potential
fields are read from the file named by `READ_W_RGRID`:

```bash
python scripts/pscf/pscf_converter.py import-script /path/to/pscf-input/param \
    --command /path/to/pscf-input/command --monomers A B -o polymerfts-job
```

If the input was exported by this converter, use its column mapping instead:

```bash
python scripts/pscf/pscf_converter.py import-script pscf-input/param \
    --command pscf-input/command --metadata pscf-input/metadata.json -o polymerfts-job
```

Without either mapping, monomer names default to `A`, `B`, etc., as for field
import. Runnable scripts require monomer names containing ASCII letters and
underscores only, following the native polymerfts engine. Names containing
digits or punctuation are rejected; use `--monomers` to supply compatible
column names. In particular, supply names explicitly for more than 26 monomers,
because the field-only import defaults above that count contain digits.
Relative `READ_W_RGRID` paths are resolved against the **parameter file's
directory**. If PSCF was run from a different working directory, pass
`--input-dir /path/to/that/directory`. An absolute initial-field path is also
accepted. Paths passed to CLI options are resolved from the converter's current
working directory.

The output directory must be new. All model and command checks happen before
it is created. The bundle contains:

| File | Purpose |
|------|---------|
| `simulation.py` | Editable `params` dictionary, initial-field loading, SCFT iteration and result saving |
| `initial.mat` | Bundled initial potentials with their grid and cell metadata |
| `conversion.json` | Source paths, command sequence, output mapping and conversion notes |

The bundle can be moved and run independently of the PSCF input files:

```bash
python polymerfts-job/simulation.py
```

Use allocated compute resources on a cluster. The generated script chooses a
backend automatically; add `--platform cpu-mkl`, `--platform cpu-fftw` or
`--platform cuda` during conversion to select one explicitly. Conversion does
not check whether that backend is installed. The script sets OpenMP defaults
before importing numerical libraries and preserves existing environment values.

### Command and output mapping

The accepted command sequence is one `READ_W_RGRID`, one `ITERATE`, zero or more
of the writes below, and a final `FINISH`. Comments and quoted paths are accepted.
This follows PSCF's [command-file format](https://dmorse.github.io/pscfpp-man/scft_command_pc_page.html)
while translating the supported calculation to polymerfts operations.

| PSCF command | Generated behavior |
|--------------|--------------------|
| `READ_W_RGRID in/omega` | Load the bundled initial potentials |
| `ITERATE` | Run `SCFT.run(..., return_result=True)` and require convergence |
| `WRITE_W_RGRID out/omega` | Save `results/out/omega.mat` |
| `WRITE_C_RGRID out/rho` | Save `results/out/rho.mat` |
| `WRITE_PARAM out/data` | Save `results/out/data.mat` |
| `WRITE_THERMO out/data` | Save the same snapshot once when sharing a path with `WRITE_PARAM` |
| `FINISH` | End the generated script |

Every mapped write creates a **complete polymerfts `.mat` snapshot** through
`SCFT.save_results`: potentials, densities, parameters, cell and thermodynamic
data. These files are not PSCF r-grid or text thermodynamics files, and
`WRITE_THERMO` does not reproduce PSCF's append-to-text behavior. The original
relative output name is retained with `.mat` appended. `conversion.json` records
each command's mapping. If there are no write commands, `results/fields.mat` is
saved by default. Output paths must be relative filenames without `..`;
conflicting file/directory paths are rejected.

The script creates a fresh `results/` beside `simulation.py` and leaves the working
directory unchanged; it draws no polymer architecture figures. It refuses to reuse an
existing results directory. A failed or unconverged calculation raises an error before
saving snapshots, leaving an empty results directory. Move or remove that directory
before rerunning the script.

### Supported parameters and limitations

Script import supports the periodic `System` / `Mixture` / `Interaction` /
`Domain` structure with `AmIteratorGrid`. It reads continuous `thread` polymers,
segment lengths, sparse `chi` interactions, `ds`, canonical `phi` volume
fractions, linear chains and connected acyclic branched chains. Multiple
polymer species are supported. `vMonomer` must be 1 and `nSolvent` must be 0.
The domain mesh/lattice and field header must agree; the initial cell lengths
and angles come from the r-grid header. No interpolation is performed.

`epsilon`, `maxItr`, `maxHist`, `isFlexible` and `scaleStress` become
`tolerance`, `max_iter`, `optimizer.max_hist`, `relax_cell` and `stress_scale`.
For omitted optional settings, the PSCF v1.4 `AmIteratorGrid` defaults used here
are 200 iterations, 50 history vectors, a flexible cell, and stress scale 10.
`flexibleParams` is accepted only when all cell parameters are free or all are
fixed. All 13 lattice geometries below can be imported with a fixed cell.

The converter rejects the following rather than producing a partial script:

| Unsupported input | Reason or alternative |
|-------------------|-----------------------|
| `SWEEP`, repeated reads/iterations, `COMPUTE`, cell-changing commands, continuation workflows | Only a single SCFT solve is translated; implement the additional Python control flow manually |
| Basis or k-grid reads/writes, `WRITE_STRESS`, other commands | Use full real-space potential fields and the supported writes above |
| `AmIteratorBasis` and other iterators | Use `AmIteratorGrid`; source space-group operations are not translated |
| Partially flexible cells; flexible 2D square/rhombic or 3D rhombohedral cells | Their relaxation constraints cannot be represented by this translation; fixed cells are supported |
| Bead chains, solvents, grand-canonical `mu`, nonunit `vMonomer` | The translator supports the continuous incompressible canonical polymer model only |
| External potentials, masks, environments, FTS settings and other unrecognized blocks or labels | These require model-specific implementation |
| Iterator controls such as `errorType`, `verbose`, `lambda`, `useLambdaRamp`, `r`; other implementation-specific options | No direct setting translation is implemented; remove them only after reviewing how the solver settings should change |

Group labels are preserved in `conversion.json` and the field metadata; no
space-group symmetry is imposed on the generated full-grid calculation.
The generated script explicitly uses polymerfts `rqm4` and Anderson mixing
with `start_error=0.01`, `mix_min=0.1` and `mix_init=0.1`. These are starting
settings, not a reproduction of PSCF's mixing algorithm or error norm.
Contour discretization, gauge choices, stress normalization and thermodynamic
reference conventions can also differ. In particular, a copied `epsilon` does
not establish the same stopping accuracy, and saved energies are polymerfts
thermodynamic values. Review convergence and discretization before numerical
comparisons.

## Cell and grid conventions

| Dimension | Supported PSCF lattices |
|-----------|-------------------------|
| 1D | lamellar |
| 2D | square, rectangular, hexagonal, rhombic, oblique |
| 3D | cubic, tetragonal, orthorhombic, hexagonal, rhombohedral, monoclinic, triclinic |

The converter preserves axis names and mesh sizes. PSCF lists the first index
fastest; polymerfts arrays use the last index fastest. The converter reorders
values rather than reversing the cell lengths or renaming axes. This follows
PSCF's [r-grid ordering](https://dmorse.github.io/pscfpp-man/user_field_rgrid_page.html).

Angles in polymerfts metadata are in degrees. PSCF's variable cell angles are
in radians, and its triclinic parameters are `a, b, c, phi, theta, gamma`, rather
than `a, b, c, alpha, beta, gamma`. The conversion preserves the cell metric.
Hexagonal cells use 120 degrees between the first two vectors, as implemented in
PSCF's [UnitCell2.cpp](https://github.com/dmorse/pscfpp/blob/dfaddb98d6436fadee2ca16b84d82880f1759710/src/prdc/crystal/UnitCell2.cpp)
and [UnitCell3.cpp](https://github.com/dmorse/pscfpp/blob/dfaddb98d6436fadee2ca16b84d82880f1759710/src/prdc/crystal/UnitCell3.cpp).

Lattice selection follows SCFT: a space group's crystal system takes precedence
over `crystal_system`. An explicit `--lattice` override is checked against the
actual geometry. Flexible skewed 3D inputs without an explicit crystal system
require this choice so that angle constraints are not silently changed.
Flexible skewed 2D cells with explicit `crystal_system="Orthorhombic"` are rejected
by default, as polymerfts SCFT rejects them: with `relax_cell`, Orthorhombic needs
90-degree angles. Use a fixed cell or explicitly choose a PSCF lattice and its
corresponding relaxation constraints.

Grid values are transferred without interpolation. Polymerfts normally samples
at cell centers and PSCF at nodes, so assigning the same values changes the
physical origin by a constant half-grid translation. This is compatible with the
periodic full-grid workflow; the converter does not translate symmetry origins
or create a PSCF symmetry-adapted basis.

## Supported model scope

Export uses `AmIteratorGrid`, which does not impose space-group symmetry. A
source space group can determine lattice constraints, but its operations and
Hall setting are not transferred. All exported fields must be on the full grid.
SCFT result files already expand symmetry-reduced fields before saving.

Discrete chains, complex fields, finite compressibility, random-copolymer blocks,
grafting, masks, nonperiodic boundaries and interaction smearing are rejected
instead of being approximated. Use a supported continuous incompressible model
for cross-code comparisons. A PSCF calculation still needs its own convergence
checks; conversion does not establish thermodynamic equivalence of different
models or solver stopping criteria.

## Migration and checks

The old `tools/conversion_to_pscf/to_pscf.py --file_name ...` exporter and
`tools/read_pscf_output/to_matlab.py` importer are replaced by `export` and
`import`; `import-script` additionally translates supported PSCF calculations.
Other MATLAB scripts and analysis notebooks have moved from `tools/` to
`scripts/analysis/`. The historical PSCF sample fields are preserved in
[`tests/data/pscf/`](../../tests/data/pscf/).

Run the format, coordinate, model-validation and command-line checks with:

```bash
python -m pytest tests/scripts/TestPscfConverter.py tests/scripts/TestPscfScript.py -q
```

Optional integration tests run the generated files with real PSCF executables.
Set `PSCF_RPC` for CPU checks and/or `PSCF_RPG` for GPU checks; tests for an unset
executable are skipped. Run calculations on allocated compute resources and
select one simulation configuration per cluster job. For example:

```bash
export PSCF_RPC=/path/to/pscf_rpc
python -m pytest 'tests/scripts/TestPscfIntegration.py::test_generated_command_converges[3d-rpc]' -q
```

The integration cases cover r-grid reading, computation, writing and import for
all supported lattice types, branched polymers and mixtures, plus convergence
with the generated command file in 1D, 2D and 3D. They check the returned fields,
cell geometry, integrated monomer amounts and converged incompressibility.

All 36 integration cases passed with the locally patched PSCF v1.4.0 CPU
(`pscf_rpc`) and GPU (`pscf_rpg`) builds: 13 lattice types, one branched polymer
and one mixture for field interchange, plus three dimensionalities for SCFT
convergence. This validates these interchange and execution paths; it does not
establish numerical equivalence between the two codes for every physical model.

Generated-script execution tests are separately enabled with
`POLYMERFTS_SCRIPT_TEST_PLATFORM=cpu-mkl`, `cpu-fftw` or `cuda` on allocated compute
resources. They are skipped when that variable is unset. For example:

```bash
POLYMERFTS_SCRIPT_TEST_PLATFORM=cpu-fftw \
    python -m pytest 'tests/scripts/TestPscfScriptIntegration.py::test_imported_script_converges[3d]' -q
```

These cases cover 1D/2D/3D, a fixed triclinic cell, a branched polymer, a mixture
and cell relaxation. If `PSCF_RPC` is also set, each case first executes the
source `param` and `command` with PSCF. The checks require convergence, saved
snapshots, incompressibility, integrated composition and the expected cell
behavior; they do not compare absolute free energies across codes.

All seven generated-script cases passed on CPU-MKL, CPU-FFTW and CUDA
(21 executions), with the source commands also run using the locally patched
PSCF v1.4.0 CPU build. Backend-free tests additionally cover all 13 fixed-cell
geometries, representable relaxation constraints, rejected models and command
sequences, output preservation, and the generated script's failure paths.
