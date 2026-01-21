# Phase Examples

Initial configurations of network phases are obtained from the Kevin Dorfman's publicly available data.

## Available Phases

* Lamellar1D.py
* Cylinder2D.py
* PL.py (perforated lamellar)
* A15.py
* BCC.py (body-centered cubic)
* FCC.py (face-centered cubic)
* HCP.py (hexagonal close-packed)
* Sigma.py
* SC.py (simple cubic)
* SD.py (single diamond)
* SG.py (single gyroid)
* SP.py (single primitive)
* DD.py (double diamond)
* DG.py (double gyroid)
* DP.py (double primitive)
* Fddd.py

---

# Space Group Validation Status

Validation of SCFT phase examples with and without space_group symmetry constraints.

## Validation Criteria

A phase **PASSES** if:
1. Both WITH and WITHOUT space_group converge to tolerance (1e-8)
2. Free energies match within numerical precision

## Validated Phases (PASS)

| Phase | Space Group | Hall # | Free Energy | Notes |
|-------|-------------|--------|-------------|-------|
| A15 | Pm-3n | 520 | -0.8685845 | Cubic |
| BCC | Im-3m | 529 | -0.0899204 | Cubic |
| DD | Pn-3m | 528 | -0.1665725 | Cubic, Double Diamond |
| DG | Ia-3d | 530 | -0.2131824 | Cubic, Double Gyroid. DG.mat regenerated with Ia-3d symmetry |
| DP | Fd-3m | 525 | -0.1445778 | Cubic |
| FCC | Fm-3m | 523 | -0.0889892 | Cubic |
| Fddd | Fddd | 544 | -0.1606698 | Orthorhombic |
| SC | Pm-3m | 517 | -0.1218046 | Cubic |
| SD | Im-3m | 529 | -0.1927212 | Cubic |
| SG | I4_132 | 510 | -0.1973548 | Cubic, Single Gyroid (chiral) |
| SP | Pm-3n | 520 | -0.1735695 | Cubic |

## In Progress

| Phase | Space Group | Hall # | Status |
|-------|-------------|--------|--------|
| Sigma | P4_2/mnm | 419 | Testing. Fixed scft.py bug. Updated to standard [a,a,c] ordering. Needs more iterations. |

## Not Yet Tested

| Phase | Space Group | Hall # |
|-------|-------------|--------|
| HCP_Hexagonal | P6_3/mmc | 488 |
| HCP_Orthorhombic | - | - |
| PL | P6_3/mmc | 488 |

## Bug Fixes Applied

### 1. DG.mat Regeneration
- **Issue**: Original DG.mat had Single Gyroid (I4_132), not Double Gyroid (Ia-3d)
- **Fix**: Regenerated DG.mat with proper Ia-3d symmetry

### 2. Tetragonal lx_full_indices Bug (scft.py:812-815)
- **Issue**: `lx_full_indices = [0, 1, 1]` enforced b=c instead of a=b
- **Fix**: Changed to `[0, 0, 1]` for standard tetragonal (a=b)

### 3. Sigma.py Axis Ordering
- **Issue**: Used `lx=[8.0,7.0,7.0]` ([c,a,a])
- **Fix**: Changed to `lx=[7.0,7.0,8.0]` ([a,a,c]) with rotated sphere positions

## Standard Axis Orderings

| Crystal System | lx Convention | Constraint |
|----------------|---------------|------------|
| Cubic | [a, a, a] | lx[0] = lx[1] = lx[2] |
| Tetragonal | [a, a, c] | lx[0] = lx[1] |
| Hexagonal | [a, a, c] | lx[0] = lx[1] |
| Orthorhombic | [a, b, c] | all independent |

---

Last updated: 2026-01-21
