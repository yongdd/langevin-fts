# Charged-polymer research

The two maintained English reports contain results, equations, code mappings,
parameters, initialization, warmup, sampling schedules, acceptance criteria,
error estimation, controls, and replay commands:

| Report | Scope | Editable source |
|---|---|---|
| [PRL_REPRODUCTION.pdf](reports/PRL_REPRODUCTION.pdf) | Duan brush SCFT/planar MC; PRL-2022 electrolyte/JCTC wall benchmarks; McCarty bulk Gaussian phase separation | [PRL_REPRODUCTION.md](reports/PRL_REPRODUCTION.md) |
| [CLFTS_VALIDATION.pdf](reports/CLFTS_VALIDATION.pdf) | Charged theory, native CL checks, exact finite-action benchmarks, convergence/pole diagnosis, positive marginal and small-system Gibbs controls, earlier electrolyte/brush tests | [CLFTS_VALIDATION.md](reports/CLFTS_VALIDATION.md) |

PRL reproduction remains partial. The planar MC action omits transverse charge
fluctuations. Native CL fails the investigated exact finite-model equilibrium
benchmark despite passing deterministic implementation checks. Neither report
claims complete PRL reproduction or general CL certification.

The 2026-09-14 electrolyte campaign is documented in the electrolyte section of the
PRL report: bulk coexistence and Galvani curves agree closely, dilute wall
profiles agree, and the symmetric nonlocal interface is only partially
reproduced. The [campaign record](reports/metadata/literature_scft_20260914.json)
links all 48 SLURM attempts, including three unconverged attempts.

The 2026-09-15 McCarty implementation adds sequence-resolved, salt-free bulk
Gaussian thermodynamics: free energy, pressure, chain chemical potential,
critical points and stable two-phase coexistence. The final PRL-report section
records 83 completed CPU jobs and ten independent numerical tests. This is an
ideal-chain approximation with an explicit paper-RPA comparison mode; it is
not the full Duan conformational closure or a CL/MD reproduction. The default
excluded volume comes from the McCarty SI structure-factor benchmark; its
exact correspondence to the main phase diagram remains to be confirmed.
See the [computed binodals](reports/figures/mccarty_gaussian_20260915/mccarty_binodals.pdf),
[critical-point data](reports/figures/mccarty_gaussian_20260915/critical.csv), and
[coexistence data](reports/figures/mccarty_gaussian_20260915/coexistence.csv).

## Open issue: charged CL convergence

Resolve the charged CL discrepancies against exact finite-lattice particles
(opened 2026-09-07; moved here from the repository TODO, item 30, on 2026-10-26).

The 24-run benchmark in [CLFTS_VALIDATION.md](reports/CLFTS_VALIDATION.md) passes
mass/partition and final-state conditional-density checks, but charged CL
averages differ from exhaustive positive particle sums. At E=100, dt=0.0025,
the cell-0 counterion density is 0.21308439 +/- 0.00112541 versus exact
0.20718911. E=0 controls agree within estimated uncertainty.
Distinguish integration bias, rare-event sampling, and possible incorrect CL
convergence before treating target PRL simulations as validated. Preserve the
finite action and native FFT; changing the model cannot validate this benchmark.

Follow-up: 21 trajectories and direct original-contour field quadrature are in
[CLFTS_VALIDATION.md](reports/CLFTS_VALIDATION.md). Quadrature agrees with the
particle sum to 2.8e-15. At dt=0.00015625, longer LM/ETD, Heun and adaptive Euler
still give C density near 0.2132 versus exact 0.20719. Entire field moments also
disagree. Large drifts occur near analytically located Q_P or Q_SP zeros;
native and independent densities agree at the drift extremes. Longer E=0
controls also reveal a smaller bias. These results strongly support incorrect
CL convergence in this fixture. A validated sampling formulation or validated
parameter regime is still needed; no production FFT or drift patch was made.

Further diagnostic: [CLFTS_VALIDATION.md](reports/CLFTS_VALIDATION.md) exactly
integrates the repulsive fields, leaving a positive one-real-field marginal
of the SAME finite action. Twelve real-Langevin runs recover all 19 reference
observables within 1.36 estimated SE. Original CL remains biased even though
its tested low-order stationarity identities pass; near-zero measurements
do not account directly for the density bias. This supplies a finite-model
control, not a scalable production repair or a complete pole-boundary proof.
Original CL validation and PRL reproduction remain open; FFT is unchanged.

1D follow-up: [CLFTS_VALIDATION.md](reports/CLFTS_VALIDATION.md) and
[PRL_REPRODUCTION.md](reports/PRL_REPRODUCTION.md) implements
particle/real-field Gibbs updates without exhaustive enumeration, tested on
4/16/32 cells of the same finite action. It also recalculates PRL Fig. 1 at
two grid resolutions. Current LDA curves still differ from digitized paper
values beyond grid/raster errors; ensemble/box and nonlocal-theory differences
remain. A working positive sampler for this small system does not validate
the existing CL dynamics or reproduce the complete PRL theory.

Drift-tail follow-up (2026-10-26): the "Drift-tail criterion and convergent
regimes" chapter of [CLFTS_VALIDATION.md](reports/CLFTS_VALIDATION.md) applies the
criterion of Nagata, Nishimura and Shimasaki (PRD 94, 114515) to this fixture with
$`\zeta N`$ and $`E`$ as parameters. Every setting shows the $`u^{-2}`$ tail of a
partition-function zero in the distribution of the largest drift, so the strict
criterion is violated everywhere. The bias grows with the weight of the tail. The
CL averages agree with the exact sums within two standard errors where at most
$`2.9\times10^{-5}`$ of the steps have a drift above ten times the median: without
charges at $`\zeta N\le1`$, and up to $`E=10`$ at $`\zeta N=0.25`$–0.5. They differ from
a tail weight of $`3.6\times10^{-5}`$ on: at $`\zeta N\ge2`$ without charges, at every
$`E`$ with $`\zeta N=4`$, and at $`E\ge25`$. The thresholds belong to this fixture. The
cause of the failure, a sampling formulation that removes it, and the behavior at
production $`\bar N`$ remain open.

Mainline: charged CL-FTS in `src/python/clfts.py` (`src/python/electrostatics.py`)
uses the same CL dynamics, so this issue applies to it as well. It records the
largest drift of every step for each auxiliary field and for $`\psi`$
(`drift_max_step_history`, `psi_drift_max_step_history`), so the same tail can be
inspected in production runs. When the issue is resolved,
update the "Checked range" paragraph of
[ComplexLangevinSimulation.md](../../docs/theory/ComplexLangevinSimulation.md)
(Section 6) and its Section 3 on correct convergence.

## Directory layout

```text
charged_polymers/
├── README.md                 # Start here
├── workflow.py               # Run, analyze, test, and build reports
├── models/                   # CL/SCFT models, electrostatics, chain references
├── studies/                  # Research runners, samplers and analysis
├── tests/                    # CL neutral reduction and sequence checks
├── examples/                 # Lamella example
├── reports/
│   ├── *.md, *.pdf           # The two maintained reports
│   ├── build_reports.py      # Pandoc/LaTeX builder
│   ├── figures/              # Figures used by the maintained reports
│   ├── metadata/             # Campaign, relocation and verification records
│   └── build/                # Optional build diagnostics (ignored)
└── archive/
    ├── RESEARCH_LOG.md        # Original notes, preserved verbatim
    ├── figures/              # Earlier plots, outside the current reports
    └── prl_comparison_figs/   # Unique paper panels for the archived comparison
```

Paths in this README are relative to `devel/charged_polymers`. Reports use
links relative to `reports/`. Historical notes and source-hash records retain
the paths recorded at the time; [relocations.json](reports/metadata/relocations.json)
maps files moved on 2026-09-14 to their then-current locations; retired
files in that frozen record may no longer exist.
It also records subsequent helper extraction, with the relocation hashes
preserved separately and the shared helper included in source audits.

## Code organization

The original twenty-three research scripts were consolidated into twelve modules
under [studies/](studies). Electrolyte and McCarty studies each add one module;
the 2026-10-26 drift-tail study adds `studies/nagata.py`. All studies use the
[workflow.py](workflow.py) command entry point.
Model implementations are in `models/`; run and analysis code remains in `studies/`.

| Module | Responsibility / workflow commands |
|---|---|
| `studies/prl_runs.py` | Shared checked SCFT/chain/PB methods; `prl fig1`, `prl salt`, `prl meanfield` |
| `studies/literature_scft.py` | PRL electrolyte limits, bulk coexistence and wall/interface attempts; `prl electrolyte`, `analyze electrolyte` |
| `studies/mccarty.py` | Sequence-resolved bulk Gaussian phases and saved-data plots; `prl mccarty`, `analyze mccarty` |
| `studies/prl_analysis.py` | Paper digitization, SCFT scans and matched comparison; `analyze fig1-finite`, `analyze prl-scan`, `analyze matched` |
| `studies/validation.py` | Deterministic implementation checks and same-model MF/CL; `check implementation`, `cl brush` |
| `studies/exact.py` | Independent finite-particle sum and native CL benchmark; `cl exact` |
| `studies/reduced.py` | Reduced CL dynamics and integrator controls; `cl reduced` |
| `studies/cl_diagnostics.py` | Checkpoints, field quadrature, poles, identities; `check checkpoints`, `field-integral`, `poles`, `identities` |
| `studies/cl_analysis.py` | Native/reduced/MF/marginal statistics; `analyze cl-exact`, `cl-reduced`, `cl-brush`, `marginal` |
| `studies/nagata.py` | Drift-tail criterion with exact sums for varied $`\zeta N`$ and $`E`$; `cl nagata`, `analyze nagata` |
| `studies/finite_sampling.py` | Nine-bead Gibbs sampler; `sample finite` |
| `studies/marginal_sampling.py` | Two-cell real-field marginal; `sample marginal` |
| `studies/brush_sampling.py` | N=100 brush, bath calibration and independent weight checks; `sample brush`, `check brush` |
| `studies/brush_analysis.py` | Planar MC statistics and controls; `analyze brush` |
| `studies/statistics.py` | Shared autocorrelation and block-error estimators |

The underlying models are grouped under `models/`:

- `models/prl_scft.py`: approximate PRL-type SCFT; `models/prl_scft2d.py`: earlier experimental 2D/scaling extension, not a validated Fig. 4 reproduction.
- `models/prl_electrolyte.py`: electrolyte extension of the PRL self-energy, with finite ion volumes, bulk phase equilibrium and charged-wall calculations. The nonlocal interface routine is experimental; its convergence status is recorded in the report.
- `models/mccarty.py`: salt-free E/K sequences, discrete-chain form factors, three explicitly named Gaussian approximations and bulk phase equilibrium. `sequence_blocks()` maps a sequence to the existing native propagator's block format. This bulk calculation uses analytical form factors and changes no native solver.
- `models/positive_brush_1d.py`: explicit finite planar MC action and updates.
- `models/clfts_charged.py`, `models/brush_clfts.py`, `models/electrostatics.py`: native charged CL and graft/wall/electrostatic implementations.
- `models/stable_grafted_chain.py`: independent scaled direct-chain reference.
- `models/brush_scft_charged.py`: charged-brush mean-field solver with Anderson mixing and a Poisson saddle; `solve()` is its execution path.
- `models/common.py`: parameter serialization and species-smearing adapters shared by the brush SCFT and CL models.
- `tests/test_neutral_reduction_clfts.py`: CL neutral-reduction check; `examples/Lamella.py`: example.

Exact particle sums, direct field quadrature and independent chain references
remain separate algorithms. Their independence is needed for validation.

## Running and rebuilding

From the repository root, activate `polymerfts`, set the stack/OpenMP environment
as described in the reports, and use:

```bash
python devel/charged_polymers/workflow.py --help
python devel/charged_polymers/workflow.py prl salt --help
python devel/charged_polymers/workflow.py prl electrolyte --help
python devel/charged_polymers/workflow.py prl mccarty --help
python devel/charged_polymers/workflow.py test mccarty
python devel/charged_polymers/workflow.py sample brush --help
python devel/charged_polymers/workflow.py test neutral-cl
python devel/charged_polymers/workflow.py example lamella --help
python devel/charged_polymers/workflow.py reports
```

The test commands execute the original calibrated scripts. Select an idle GPU
before running them. `example lamella --help` displays the entry; omitting
`--help` runs the example and belongs in SLURM. For direct Python imports, add
`devel/charged_polymers` to `PYTHONPATH` and use, for example,
`from models.brush_clfts import BrushChargedCLFTS`.

Sampling/reduced modules use Numba and llvmlite; the reports record the historical
local dependency path. Native checks require the installed `polymerfts` extension.
Choose an idle GPU for CUDA. Submit calculations longer than one minute through
SLURM, one parameter configuration per job. Preserve old run directories and use
fresh output paths for replays.

[build_reports.py](reports/build_reports.py) uses Pandoc → LaTeX/Tectonic and pypdf.
It typesets the curated Markdown and packaged figures, with no simulation or raw
sample dependency. Both PDFs must compile and pass glyph/overflow/text checks
before either published PDF is replaced. Successful builds remove their
intermediates; use `workflow.py reports --keep-build` to retain generated TeX,
logs, extracted text and metadata in ignored `reports/build/`. Failed builds
retain diagnostics automatically. Edit the Markdown under `reports/`.

Analysis commands write numerical summaries and plots beside their output data.
The PRL scan, brush and matched-comparison notes link to the maintained report
for model definitions, protocols and interpretation; those explanations are
edited only in the report. A new analysis note does not refresh a report edition.
Updating a report edition requires reviewing the new analysis, updating its
Markdown/results and copying selected figures into `reports/figures/`, then
rebuilding. Typesetting alone does not refresh numbers from raw campaigns.

## Historical commands and provenance

[reports/metadata/campaigns.json](reports/metadata/campaigns.json) freezes the campaign paths,
summary hashes, stored parameters, job-script hashes, concentration grid,
MC/SCFT case mapping, and packaged figure hashes. It includes failed and
superseded attempts; the reports and original comparison/analysis files specify
which runs enter final figures. Large samples/checkpoints remain under
`/home/yongdd/polymer/dh_salt_runs/`.

[studies/legacy_commands.json](studies/legacy_commands.json) maps all 23 old
script names to their new implementations. For example:

```bash
python devel/charged_polymers/workflow.py legacy run_prl_fig23_checked.py --help
```

For current-code replays, replace the old Python script invocation in a saved
job with `workflow.py legacy OLD_SCRIPT.py`, keeping its arguments and using a
fresh output directory. Direct old script paths have been removed. For an
exact historical source replay, use the saved source snapshot or pre-cleanup
revision `d395ed1a`. Source audits distinguish moved runners from unchanged
native kernels; a relocated file is not reported as byte-identical.

[archive/RESEARCH_LOG.md](archive/RESEARCH_LOG.md) preserves all original notes
verbatim with per-source hashes, including the pre-existing edited Fig. 1 note.
It contains superseded conclusions and historical paths, and is not another
maintained report. The originally archived comparison PDF/TeX remain in
`archive/`. Redundant report fragments, old generated PDFs and the old builder
were removed after the replacement reports compiled successfully.

## Maintenance records

| Edition | Evidence |
|---|---|
| 2026-09-08 script/report consolidation | [refactor_validation.json](reports/metadata/refactor_validation.json) |
| 2026-09-14 folder reorganization | [layout_validation.json](reports/metadata/layout_validation.json) |
| 2026-09-14 duplicate removal and shared helpers | [cleanup_validation.json](reports/metadata/cleanup_validation.json) |

These checks cover code maintenance, not a new equilibrium campaign. Historical
validation records describe their respective editions. Unique archived figures
and the original notes are preserved; the edited `archive/figures/fig2_reproduction.png`
is unchanged. Sixteen identical figure copies and generated build/cache files
were removed, and the archived TeX now uses the single retained copy of each plot.
