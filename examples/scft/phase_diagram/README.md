# Phase Diagrams

`phase_diagram.py` computes the SCFT phase diagram of a polymer system in the plane
of two parameters (x, y). It needs a function that puts (x, y) into the simulation
parameters and initial fields (a seed) of every candidate phase; it makes no other
assumption about the system.

| File | Content |
|------|---------|
| `phase_diagram.py` | The algorithm: `run()` computes and saves a diagram, `connect()` joins its boundary points into curves, `plot()` draws a saved one |
| `seeds.py` | Initial fields of lamellae, cylinders, double gyroid and BCC spheres of AB block copolymers |
| `AB_Diblock.py` | AB diblock melt in the (f, χN) plane |
| `ABA_Triblock.py` | ABA triblock melt in the (f, χN) plane; only the system definition differs |

Run a system script from this folder on a GPU:

```bash
python AB_Diblock.py        # writes AB_Diblock.json and AB_Diblock.png
```

## Algorithm

Every SCFT solve starts from the nearest converged solution of the same phase
(continuation), and the boundaries come from root finding, so a coarse grid is enough.

1. **Seeds.** Each seed is converged at its own point (x, y) and moved to the nearest
   grid point.
2. **Grid.** Each phase spreads over the grid from that point, every new grid point
   continued from a neighbour, until the phase melts or no longer converges. A
   continuation step that fails is retried with half the step, down to
   `smallest_step` (default 1/4) of the way to the grid point, and a run whose error
   stalls is stopped early.
3. **Boundaries.** Wherever the most stable phase changes between neighbouring grid
   points, F<sub>I</sub> = F<sub>II</sub> is solved along that grid line. If a third
   phase is lower where I and II cross, it is stable in a window narrower than the grid,
   and the two boundaries of that window are solved instead.
4. **Refinement.** A boundary that passes through a grid cell crosses its edges twice;
   one that crosses them an odd number of times ends inside, at a triple point or a
   critical point. Such a cell is split into four, the most stable phase is found at
   the new points and the boundaries on the two new lines. This is repeated `refine`
   times (default 2) on the new cells.
5. **Connection** (`connect`). Inside each cell, two points of a pair on its edges
   belong to one curve. Curve ends in touching cells belong together: two ends of the
   same pair are one curve whose crossing fell into the neighbouring cell within the
   root tolerance, and three or more ends meet at a junction, a triple or critical
   point, if every phase appears in an even number of their pairs, since each phase
   there lies between two curves. Other ends mark a break; the script prints each break,
   and the plot marks it with a red cross.

### Root finding

Along χN the boundaries are found with Newton steps, and along f with Brent's method.

Newton steps need the derivative of F along the line, and for an interaction parameter
it follows from the converged solution without another solve. The SCFT free energy is
the Hamiltonian at its saddle point $`w^*`$,

```math
F(\chi) = H[w^*(\chi), \chi] .
```

Since $`\partial H/\partial w = 0`$ there, the change of $`w^*`$ with χ does not
contribute (envelope theorem), and $`dF/d\chi`$ is the explicit derivative
$`\partial H/\partial \chi`$ at $`w^*`$. In H, $`\chi_{ij} N`$ appears only in the
interaction term $`\chi_{ij} N \langle \phi_i \phi_j \rangle`$, and F is measured from the
homogeneous state, whose interaction term is $`\chi_{ij} N \langle \phi_i \rangle \langle \phi_j \rangle`$.
Hence

```math
\frac{\partial F}{\partial (\chi_{ij} N)} = \langle \phi_i \phi_j \rangle - \langle \phi_i \rangle \langle \phi_j \rangle
```

(`chi_derivative`). This holds for Flory–Huggins interactions without smearing; it agrees
with a central difference of F to about 1e-5 relative.

Along f there is no such derivative, so the boundaries use Brent's method, which needs
only values of F and keeps the root bracketed.

A phase that melts or cannot be continued to an evaluation point does not exist there.
It counts as just above the other phase, so the root lies where both phases exist. A
root at which one of two ordered phases still cannot be continued is the end of that
phase, not a crossing, and is skipped.

## A New System

Copy a system script and change:

- `base_params`: the parameters shared by all phases.
- `set_point(params, x, y)`: puts the two parameters into `params`. Any parameter can be
  an axis: block lengths, χN, segment lengths, volume fractions.
- `phases`: for every candidate phase a seed, the point (x, y) where it is converged,
  and the parameters of the phase (`nx`, `space_group`, ...). A seed is initial fields
  `{"w": {monomer: array}, "lx": lx}`, as built by `seeds.py`, or a file saved by
  `SCFT.save_results` (`.json` or `.mat`) in `seed_dir`. An optional `region` limits a
  phase to a rectangle.
- The arguments of `run()`: grids, boundary tolerances, derivatives, axis labels and
  `mirror_x` for a diagram symmetric about a vertical line. Pass `homogeneous=False` for
  a system without a homogeneous state, for example with walls or grafted chains. A
  smaller `smallest_step` (for example 1/8) follows a phase closer to the end of its
  existence at the cost of more retries.

A seed needs only the symmetry, the rough shape and roughly the period of its phase,
since its first solve converges it, and the same seed can serve several systems
(`seeds.py` serves the diblock and, in cells of 0.63 the size, the triblock). Its
point must lie where the phase exists in the system; a phase whose seed melts there is
left out. A seed in a much larger cell than the period can melt or converge to a
supercell.

## Results

| System | Check | Result |
|--------|-------|--------|
| AB diblock | Order-disorder transition at f = 0.5 | χN = 10.502 (mean-field critical point 10.495; `y_tol` = 0.01) |
| AB diblock | Cost | about 2000 SCFT solves, about 1 h on an A10 |
| ABA triblock | Order-disorder transition at f = 0.5 | χN = 17.992 |
| ABA triblock | Cost | about 1500 SCFT solves, 45 min on an A10 |

For the AB diblock without refinement, halving failed steps down to 1/8 of the way,
also in the root finding, found the same 58 boundaries with 50% more SCFT solves.

## Limitations

- Phases are compared by their free energy. That decides the diagram of a melt of one
  copolymer, or of any system that cannot separate into coexisting phases of different
  composition. A blend can form two-phase regions, which need coexistence conditions
  instead ([Phase Equilibrium](../../../docs/theory/PhaseEquilibrium.md)).
- Only the candidate phases are compared; a phase without a seed does not appear.
- Close to a critical point all ordered phases have F ≈ 0 and converge slowly, so the
  phase that orders first there is not resolved, although the transition itself is.
- A boundary whose root lies where one of two ordered phases ends is skipped. The script
  prints each skipped boundary, and the gap it leaves appears as a break.
