# Phase Examples

Initial configurations of network phases are obtained from the Kevin Dorfman's publicly available data.

## Available Phases

| Phase | File | Space Group | Hall # | Free Energy (βF/n_0) |
|-------|------|-------------|--------|---------------------|
| Lamellar | Lamella1D.py | - | - | -0.1429224 |
| Cylinder | Cylinder2D.py | - | - | -0.5058274 |
| A15 | A15.py | Pm-3n | 520 | -0.8688101 |
| BCC | BCC.py | Im-3m | 529 | -0.0899198 |
| FCC | FCC.py | Fm-3m | 523 | -0.0889438 |
| HCP | HCP_Hexagonal.py | P6_3/mmc | 488 | -0.1335563 |
| Sigma | Sigma.py | P4_2/mnm | 419 | -0.4703563 |
| SC | SC.py | Pm-3m | 517 | -0.1218295 |
| SD | SD.py | Fd-3m | 526 | -0.1927233 |
| SG | SG.py | I4_132 | 510 | -0.1973792 |
| SP | SP.py | Pm-3m | 517 | -0.1735711 |
| DD | DD.py | Pn-3m | 522 | -0.1997226 |
| DG | DG.py | Ia-3d | 530 | -0.2129858 |
| DP | DP.py | Im-3m | 529 | -0.1445790 |
| Fddd | Fddd.py | Fddd | 336 | -0.1606772 |
| PL | PL_Hexagonal.py | P6_3/mmc | 488 | -0.2100188 |

---

## Initial Field Generation

Phases with non-cubic crystal systems (HCP, Sigma) generate initial fields from **Wyckoff positions** instead of loading from .mat files. PL loads from a converged PSCF solution (PL.mat).

Example (HCP with P6_3/mmc):
```python
# Wyckoff 2c positions for P6_3/mmc
sphere_positions = [
    [1/3, 2/3, 1/4],
    [2/3, 1/3, 3/4],
]
```

---

## Standard Axis Orderings

| Crystal System | lx Convention | Constraint |
|----------------|---------------|------------|
| Cubic | [a, a, a] | lx[0] = lx[1] = lx[2] |
| Tetragonal | [a, a, c] | lx[0] = lx[1] |
| Hexagonal | [a, a, c] | lx[0] = lx[1] |
| Orthorhombic | [a, b, c] | all independent |

---

## Mesh Reduction

Space group symmetry reduces the computational mesh significantly:

| Space Group | Reduction Factor |
|-------------|------------------|
| Fm-3m (FCC) | ~192x |
| Fd-3m (SD) | ~192x |
| Ia-3d (DG) | ~96x |
| Im-3m (BCC) | ~48x |
| Pm-3m (SC) | ~48x |
| P6_3/mmc (HCP) | ~22x |
| P6_3/mmc (PL) | ~22x |
| P4_2/mnm (Sigma) | ~15x |

---

Last updated: 2026-03-22
