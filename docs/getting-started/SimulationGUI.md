# Simulation Script Builder (beta)

The Simulation Script Builder is a browser editor for preparing
SCFT, L-FTS and CL-FTS Python scripts. It works on headless Linux servers through
SSH port forwarding. Configure the system, review the generated code, and save
the script on the server or download it to your computer.

**Release status: beta.** Phase-specific initial fields are currently limited to **lamella and gyroid**.
Random fields and existing `.mat` fields are also supported. Additional phase
templates are [planned](../../TODO.md). The editor generates scripts; it does not
run simulations or submit jobs.

## Start the editor

From the repository root:

```bash
python3 scripts/gui/simulation_gui.py
```

Open `http://127.0.0.1:8765` in a browser on the same computer. The editor uses only
the Python standard library. Keep the `scripts/gui/assets/` asset directory
next to `simulation_gui.py`; all browser assets are served locally. Running an
exported script later requires the installed polymerfts package and its
dependencies; see [Installation.md](Installation.md).

## Connect through SSH

On your own computer, open an SSH connection with port forwarding:

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

In that SSH session, change to the repository and start the editor:

```bash
cd /path/to/langevin-fts
python3 scripts/gui/simulation_gui.py
```

Open `http://127.0.0.1:8765` in your own computer's browser. Keep the SSH connection
and editor running while editing; press **Ctrl+C** to stop the editor. The server
listens on `127.0.0.1`. X11 forwarding is unnecessary.

In VS Code Remote-SSH, run the same command in the remote terminal and forward
port `8765` in the **Ports** tab. Open the forwarded address in your local browser.
If port `8765` is occupied, use another port with `--port 8766` and adjust the SSH
forwarding command and browser address to match.

## Configure a script

1. Choose a **Starting template** and click **Load template**. Available templates
   are SCFT AB lamella, AB gyroid, ABA lamella, L-FTS AB lamella and CL-FTS AB
   lamella. Loading a template replaces the current settings with its defaults.
2. In **System**, select the simulation mode, chain model, contour step and
   computational platform. Fractions such as `1/100` are accepted. Enable
   **Reduce memory usage** only when memory is insufficient: it slows down the
   simulation. **Numerical method** keeps the parameter name `numerical_method`.
   CN-ADI2 supports real fields in orthogonal cells and is unavailable for CL-FTS.
3. In **Box**, set the dimension, cell shape, grid and lengths. Drag the cell
   preview or use the arrow keys to rotate it.
4. In **Molecules**, add monomers, polymers, blocks and interaction rows. Choose
   monomers from dropdowns and set each polymer's volume fraction. Use **Equal
   fractions** or edit the fractions so their total is 1. Unlisted interaction
   pairs have `chi_n = 0`.
5. Set convergence and mixing parameters for SCFT, or Langevin dynamics and
   recording periods for FTS. **CL-FTS** also exposes compressibility `zeta_n`,
   dynamical stabilization and optional Gaussian or sigmoidal interaction smearing.
6. In **Fields / Output**, choose initial fields and a simulation output directory.
   Review the automatically updated **Script preview**, then save or download it.

The editor opens with a **3D Cubic** box, `nx = [64, 64, 64]` and
`lx = [4.0, 4.0, 4.0]`. SCFT also supports 1D and 2D boxes; the ABA lamella
template explicitly uses 1D. The 3D shape choices are Cubic, Orthorhombic,
Hexagonal and Triclinic. Boundaries are periodic.

For SCFT without a space group, the selected 3D shape is exported as
`crystal_system`: Cubic keeps `a = b = c`, Orthorhombic frees the three lengths,
Hexagonal keeps `a = b` and fixed angles, and Triclinic frees all lengths and
angles when `relax_cell` is enabled. With a space group, its crystal system
determines the constraints instead. A 2D Square cell currently requires fixed-cell
SCFT; the relaxation checkbox is disabled. Choose Rectangle to relax a 2D cell.

**L-FTS and CL-FTS are restricted to 3D orthogonal (non-skewed) boxes in this
editor.** Choose Cubic or Orthorhombic; all angles are fixed at 90°. Switching
from SCFT to either FTS mode applies this restriction while retaining the
per-axis grid values and lengths. Generation and server saves also validate it.

### Linear and branched polymers

Each polymer has an **Architecture** selector. **Linear** connects blocks in row
order. **Branched** gives each block a compact row with monomer type, length,
**Start** and **End** nodes, and row actions.

Choose **Attach new block to**, or click a node in the diagram, then click **Add
block** to attach a new terminal block. Reuse a node to create a junction; choose
**+ New node** in an endpoint dropdown to create a node. All blocks must form one
connected tree. Rings, duplicate connections and self-connections are rejected.

Remove terminal blocks first. Reordering branched rows preserves their node
connections; switching to **Linear** reconnects blocks in row order. Block
lengths use reference-chain units, so a chain can have total length other than 1.
The diagrams show architecture; chain positions in the cell are illustrative.

## Initial fields and beta limitations

| Initial fields | Current support |
|----------------|-----------------|
| `lamella` | Two monomer types; cosine seed along the first box axis in 1D, 2D or 3D |
| `gyroid` | Two monomer types; 3D analytic seed from the Gyroid example |
| `random` | Gaussian fields; two types use opposite fields, larger systems draw each field independently and subtract their mean over types |
| `mat` | Existing fields for every monomer, with one finite value per grid point |

For more than two monomer types, choose `random` or `mat`. **Seed scale / random
std** sets the analytic seed amplitude or Gaussian draw scale. Subtracting the
mean for larger systems changes the resulting standard deviation.

For `.mat` input, enter a path on the machine where the exported script will run.
The editor does not upload or read that file. The generated script loads
`w_<name>` for every monomer, for example `w_A` and `w_B`. SCFT and L-FTS require
real fields. CL-FTS accepts real or complex fields, including paired
`w_<name>_real` and `w_<name>_imag` arrays. This initializes fields rather than
resuming the step counter and random state of a Langevin checkpoint. For FTS,
start from suitable SCFT fields or allow sufficient equilibration.

Phase templates such as cylinders, BCC, FCC, SC, A15, Sigma, HCP and perforated
lamellae are not yet available in the editor. Use compatible `.mat` fields or
edit the exported initial-field code for these phases. Selecting a space group
does not create the corresponding phase or ensure convergence to it.

The editor covers a subset of the Python API. Review the exported script against
the [parameter reference](../references/Parameters.md) when adding settings.
Generated scripts use `polymers`, `relax_cell` and `space_group.hall_number`.
Advanced random-block composition dictionaries use `monomer_fractions` in the
Python API; this editor currently builds blocks of individual monomer types.

### FTS recording and step numbers

All recording intervals are measured in **Langevin steps** and apply to both
L-FTS and CL-FTS.

| Parameter in `recording` | Meaning |
|--------------------------|---------|
| `checkpoint_interval` | Save the complete state needed to restart a simulation |
| `sample_interval` | Collect observable samples, including energy, interaction derivatives, structure functions and, for CL-FTS, drift |
| `statistics_save_interval` | Save accumulated observables and reset the accumulators; must be a multiple of `sample_interval` |

`langevin.max_step` is the **final step number**, including steps already
completed before a restart. Loading `.mat` fields in this editor initializes a
new run; resuming the full state uses the simulation's checkpoint continuation API.
The simulation classes continue to accept legacy input names and checkpoint
recording keys; new exports use the names above.

## Space groups and CrysFFT

**Space group (SCFT only)** in **System** defaults to **None (full grid)**. The
selector offers 423 supported Hall settings across the orthorhombic, tetragonal,
trigonal, hexagonal and cubic crystal systems. Common polymer-phase groups appear
first. A **Hall number** distinguishes axis and origin settings; it differs from
the International space group number.

Space group symmetry requires 3D. The editor checks cell lengths, angles,
equal-grid constraints and translation divisors before exporting both the
`symbol` and `hall_number`. For tetragonal groups, use Orthorhombic with `a = b`.
For rhombohedral-axis settings, use Triclinic with equal lengths and equal angles,
and disable box relaxation. The selected group's requirements appear below the
selector. Initial fields are projected onto the selected symmetry by SCFT.

The selector is disabled for L-FTS and CL-FTS, and space group settings are omitted
from their scripts. Switching back to SCFT restores the selection.

**CrysFFT is selected automatically when applicable**, with no separate GUI
toggle. With no space group, the default periodic pseudo-spectral calculation
uses regular FFTs. For pseudo-spectral calculations with a compatible space
group, grid and cell, SCFT selects a CrysFFT physical basis. Other supported
groups use regular FFTs with symmetry reduction of the fields. See
[SpaceGroup.md](../theory/SpaceGroup.md) and
[FFTImplementation.md](../internals/FFTImplementation.md) for the conditions.

## Save on the server or download locally

| Action | Where the Python script is written |
|--------|------------------------------------|
| **Save on server** | The server base directory shown in the editor, plus the optional subfolder |
| **Download locally** | Your browser's download location on your own computer |

Both actions generate the script from the current settings. By default, the
server base directory is the **working directory where you started the editor**;
no `--save-dir` option is needed. To save scripts in a job directory, start there:

```bash
mkdir -p ~/polymer-jobs
cd ~/polymer-jobs
python3 /path/to/langevin-fts/scripts/gui/simulation_gui.py
```

With **Filename** set to `simulation.py` and **Server subfolder** empty, a server
save creates `~/polymer-jobs/simulation.py`. A subfolder of `gyroid/run01` instead
creates `~/polymer-jobs/gyroid/run01/simulation.py`. New subfolders are created as
needed. Existing files are never overwritten; choose another filename or folder
for a new version. Subfolders must stay within the base directory, and symbolic
links are not followed. Folder and file names may not contain `:`. On Windows,
which has no directory descriptors, the editor refuses symbolic links and junctions
when it checks each folder, but does not catch a folder that another process
replaces by a link during a save. The optional `--save-dir` flag overrides the
base directory.

The **Simulation output directory** is separate: the exported script creates it
when you run the simulation later. Relative input and output paths resolve next
to the saved Python script. Architecture drawings and SCFT `fields.mat` go into
the output directory; L-FTS and CL-FTS recordings go into its `data/` subdirectory.
Use a separate output directory for each run. Review the editable script and
submit long calculations through SLURM on clusters.

## Development checks

The script-generation and HTTP checks require pytest but no display or
computational backend:

```bash
python -m pytest tests/scripts -q
```

The space group catalog is bundled in `scripts/gui/assets/space-groups.json`.
The editor reads it offline; neither Python `spglib` nor a catalog-generation
tool is needed to use the editor.
