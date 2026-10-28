# User Scripts

These tools help library users prepare simulations, convert files and analyze
results. Run the commands below from the repository root unless a section says
otherwise.

| Folder | Contents |
|--------|----------|
| [gui/](gui/) | Browser script builder and [web assets](gui/assets/) |
| [pscf/](pscf/) | PSCF converter and a small initial-field example |
| [analysis/](analysis/) | MATLAB scripts, Jupyter notebooks and a Python script for density, structure functions, WTMD and interaction calibration |

## Simulation Script Builder (beta)

```bash
python3 scripts/gui/simulation_gui.py
```

Open the printed URL (`http://127.0.0.1:8765/?token=...`) in a browser. For a remote server, connect with
`ssh -L 8765:127.0.0.1:8765 user@server` from your computer, then start the editor
in that SSH session. VS Code Remote-SSH port forwarding also works.

The editor uses only the Python standard library and creates SCFT/L-FTS/CL-FTS
scripts. **Save on server** writes to the directory where the editor was started
by default; **Download locally** saves to your browser's download location.
Phase-specific initial fields currently support **lamella and gyroid**, with
random and `.mat` inputs also available. More phase templates are planned.

See the [GUI guide](../docs/getting-started/SimulationGUI.md) for SSH setup,
box and polymer editing, space groups and CrysFFT.

## PSCF Conversion

```bash
python scripts/pscf/pscf_converter.py export fields.mat -o pscf-input
python scripts/pscf/pscf_converter.py import pscf-input/out/omega --rho pscf-input/out/rho --metadata pscf-input/metadata.json -o imported.mat
python scripts/pscf/pscf_converter.py import-script pscf-input/param --command pscf-input/command --metadata pscf-input/metadata.json -o polymerfts-job
```

`import-script` bundles a runnable `simulation.py` and initial fields for one
supported PSCF SCFT calculation. Conversion uses NumPy and SciPy and does not
run a simulation. See the
[conversion guide](../docs/getting-started/PscfConversion.md) for initial-field
examples, cell conventions, input validation and supported model scope.

## Analysis

MATLAB scripts and notebooks formerly under `tools/` are in [analysis/](analysis/).
Run them from the directory containing the simulation output, for example:

```bash
jupyter notebook /path/to/polymerfts/scripts/analysis/plot_2d_density.ipynb
```

`renormalization.py` computes the UV renormalization factor $`z_\infty`$ of $`\chi N`$ and the
bare segment length $`a_b/a`$ of CL-FTS. It reads a fields file or takes
options, and needs only NumPy and SciPy:

```bash
python scripts/analysis/renormalization.py data_simulation/fields_010000.mat
```

See the [tools reference](../docs/references/Tools.md) for usage and
[post-processing references](analysis/POSTPROCESSING_REFERENCES.md) for citations.
