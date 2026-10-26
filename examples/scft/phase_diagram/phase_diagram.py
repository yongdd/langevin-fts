"""Phase diagram of a polymer system in the plane of two parameters (x, y).

run() needs a function that puts (x, y) into the simulation parameters and initial
fields (a seed) of every candidate phase; it makes no other assumption about the
system. AB_Diblock.py and ABA_Triblock.py in this folder are examples, and seeds.py
builds the initial fields of common AB phases.

Every SCFT solve starts from the nearest converged solution of the same phase
(continuation), and the boundaries come from root finding, so a coarse grid is
enough:

  1. Seeds: each seed is converged at its own point (x, y) and moved to the
     nearest grid point.
  2. Grid: each phase spreads over the grid from that point, every new grid point
     continued from a neighbour, until the phase melts or no longer converges.
  3. Boundaries: wherever the most stable phase changes between neighbouring grid
     points, F_I = F_II is solved along that grid line, with Newton steps when the
     derivative of F along the line is known (for chi N it follows from the
     concentrations, see chi_derivative) and with Brent's method otherwise. If a
     third phase is lower where I and II cross, it is stable in a window narrower
     than the grid, and the two boundaries of that window are solved instead.
  4. Refinement: a cell in which a boundary ends (it crosses the cell edges an odd
     number of times: a triple point or a critical point) is split into four, and
     stage 3 is repeated on the new lines.
  5. Connection: the boundary points are joined into curves, and their junctions
     and breaks are checked (see connect).

Phases are compared by their free energy relative to the homogeneous state. That
decides the phase diagram of a melt of one copolymer, or of any system that
cannot separate into coexisting phases of different composition. A blend can form
two-phase regions, which need coexistence conditions instead (see
docs/theory/PhaseEquilibrium.md).

A seed needs only the symmetry and rough shape of its phase, and the same seed can
serve several systems (seeds.py serves the AB diblock and the ABA triblock), but its
point must lie where the phase exists in the system; a phase whose seed melts there
is left out. Close to a critical point all ordered phases
have F ~ 0 and converge slowly, so the phase that orders first there is not
resolved, although the transition itself is.

Set the OpenMP environment variables before importing this module, as for any
script that imports polymerfts.
"""

import os
import sys
import copy
import json
import time
import contextlib
from collections import deque

import numpy as np
import scipy.io
import scipy.ndimage
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from polymerfts import scft

DIS = "DIS"             # name of the homogeneous state


def chi_derivative(i, j):
    """dF/d(chi_ij N) = <phi_i phi_j> - <phi_i><phi_j> at a saddle point (envelope
    theorem), for the free energy relative to the homogeneous state and
    Flory-Huggins interactions without smearing."""
    def derivative(result):
        phi_i, phi_j = np.asarray(result.phi[i]), np.asarray(result.phi[j])
        return float(np.mean(phi_i*phi_j) - np.mean(phi_i)*np.mean(phi_j))
    return derivative


@contextlib.contextmanager
def quiet():
    """Silence the per-iteration output of SCFT, including the native library's."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


def deviation(w):
    """Deviations of all fields from their means as one vector; zero in the
    homogeneous state."""
    return np.concatenate([np.ravel(w[m]) - np.mean(w[m]) for m in sorted(w)])


def on_edges(boundaries, cell, x_end, y_end):
    """Boundary points on the edges of a cell (x0, x1, y0, y1). A point lies only on
    the line along which it was found, and every edge is half open (closed at the
    end x_end, y_end of the grid), so that a point at a grid corner belongs to one
    edge of each line through that corner."""
    x0, x1, y0, y1 = cell
    points = []
    for b in boundaries:
        if b["along"] == "y":
            on_edge = ((np.isclose(b["x"], x0) or np.isclose(b["x"], x1)) and y0 <= b["y"]
                       and (b["y"] < y1 or np.isclose(y1, y_end)))
        else:
            on_edge = ((np.isclose(b["y"], y0) or np.isclose(b["y"], y1)) and x0 <= b["x"]
                       and (b["x"] < x1 or np.isclose(x1, x_end)))
        if on_edge:
            points.append(b)
    return points


def connect(boundaries, cells, x_step, y_step, x_end, y_end):
    """Connect the boundary points into curves; returns the segments, the junctions
    and the breaks.

    Two points of a pair on the edges of a cell belong to one curve through it; a
    point left over is a curve end inside the cell. Ends in touching cells belong
    together. Two ends of the same pair are one curve whose crossing between the
    two cells fell into the neighbouring cell within the root tolerance. Three or
    more ends meet at a junction, a triple or critical point: every phase there lies
    between two of the curves, so it must appear in an even number of their pairs.
    Ends that fit neither mark a break in the diagram."""
    segments, ends = [], []
    for cell in cells:
        by_pair = {}
        for b in on_edges(boundaries, cell, x_end, y_end):
            by_pair.setdefault(tuple(b["phases"]), []).append((b["x"], b["y"]))
        for pair, points in by_pair.items():
            while len(points) >= 2:
                p = points.pop(0)
                q = min(points, key=lambda q: np.hypot((q[0] - p[0])/x_step, (q[1] - p[1])/y_step))
                points.remove(q)
                segments.append({"phases": list(pair), "points": [p, q]})
            if points:
                ends.append({"phases": list(pair), "point": points[0], "cell": cell})

    def touching(c, d):
        return c[0] <= d[1] + 1e-9 and d[0] <= c[1] + 1e-9 and c[2] <= d[3] + 1e-9 and d[2] <= c[3] + 1e-9

    clusters = []
    for end in ends:
        joined = [cluster for cluster in clusters if any(touching(end["cell"], e["cell"]) for e in cluster)]
        clusters = [cluster for cluster in clusters if cluster not in joined]
        clusters.append([end] + [e for cluster in joined for e in cluster])
    junctions, breaks = [], []
    for cluster in clusters:
        appearances = {}
        for e in cluster:
            for p in e["phases"]:
                appearances[p] = appearances.get(p, 0) + 1
        centre = [float(np.mean([e["point"][k] for e in cluster])) for k in (0, 1)]
        found = [{"phases": e["phases"], "point": e["point"]} for e in cluster]
        if len(cluster) == 2 and cluster[0]["phases"] == cluster[1]["phases"]:
            segments.append({"phases": cluster[0]["phases"], "points": [e["point"] for e in cluster]})
        elif len(cluster) >= 3 and all(n % 2 == 0 for n in appearances.values()):
            junctions.append({"point": centre, "ends": found})
        else:
            breaks.append({"point": centre, "ends": found})
    return segments, junctions, breaks


def run(base_params, set_point, phases, x_grid, y_grid, x_tol, y_tol,
        derivatives=None, homogeneous=True, refine=2, smallest_step=1/4, seed_dir=".",
        labels=("x", "y"), mirror_x=None, output="phase_diagram"):
    """Compute the phase diagram, save <output>.json and <output>.png and return
    the saved data.

    base_params    parameters shared by all phases
    set_point      function(params, x, y) that puts (x, y) into params
    phases         {name: {"seed": initial fields {"w": {monomer: array}, "lx": lx}
                                   (see seeds.py), or a file in seed_dir saved by
                                   SCFT.save_results (.json or .mat),
                           "point": (x, y) where the seed is converged,
                           "params": parameters of this phase (nx, space_group, ...),
                           "region": optional (x_min, x_max, y_min, y_max)}}
    x_grid, y_grid grid values; x_tol, y_tol: accuracy of the boundary points
    derivatives    {"x": function(result) or None, "y": ...}: dF along each axis
    homogeneous    include the homogeneous state with F = 0 (False for systems
                   without one, e.g. with walls or grafted chains)
    refine         number of times a cell in which a boundary ends (a triple point,
                   a critical point) is split into four
    smallest_step  a failed continuation step is halved until it is shorter than
                   this fraction of the way to the next grid point; root finding,
                   which starts from a close state, stops one halving earlier
    labels, mirror_x  axis labels and an optional mirror line x = mirror_x of the plot
    """
    derivatives = {"x": None, "y": None, **(derivatives or {})}
    x_grid, y_grid = np.asarray(x_grid, dtype=float), np.asarray(y_grid, dtype=float)
    x_step, y_step = x_grid[1] - x_grid[0], y_grid[1] - y_grid[0]
    count = {"solves": 0, "iterations": 0}

    def make_params(phase, x, y, lx):
        params = copy.deepcopy(base_params)
        params.update(copy.deepcopy(phases[phase].get("params", {})))
        set_point(params, x, y)
        params["lx"] = list(lx)
        return params

    def solve(phase, x, y, guess, reference):
        """One SCFT solve from the guess. Returns the converged state, or None when it
        does not converge, melts, or its structure differs from the reference state.
        Without a reference the guess is a seed, which may need many iterations."""
        errors = []

        def stop_when_stalled(info):
            # Give up on a continuation step whose lowest error has not dropped tenfold
            # during the last 300 iterations; continue_to then retries with a smaller
            # step. A seed can rise and fall for longer before it converges.
            errors.append(info.error_level)
            if (reference is not None and len(errors) > 300
                    and min(errors[-300:]) > 0.1*min(errors[:-300])):
                return False
            return None

        try:
            with quiet():
                calculation = scft.SCFT(params=make_params(phase, x, y, guess["lx"]))
                result = calculation.run(initial_fields=guess["w"], on_iteration=stop_when_stalled,
                                         return_result=True)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            print(f"    {phase} at ({x:.4f}, {y:.4f}): {error}")
            return None
        count["solves"] += 1
        count["iterations"] += result.iterations
        if not result.converged:
            return None
        w = {m: np.array(v) for m, v in result.w.items()}
        dw = deviation(w)
        if homogeneous and np.std(dw) < 1e-3:
            return None         # melted into the homogeneous state
        if reference is not None:
            # Continuation must stay on the branch of this phase: reject a step whose
            # fields are no longer similar to the previous ones
            dw_ref = deviation(reference["w"])
            overlap = np.sum(dw*dw_ref)/np.sqrt(np.sum(dw**2)*np.sum(dw_ref**2))
            if overlap < 0.9:
                return None
        dF = {axis: (None if d is None else d(result)) for axis, d in derivatives.items()}
        return {"x": x, "y": y, "F": result.free_energy, "dF": dF, "w": w,
                "lx": [float(v) for v in result.lx]}

    def distance(state, x, y):
        """Distance between parameter points in units of the grid spacings."""
        return np.hypot((state["x"] - x)/x_step, (state["y"] - y)/y_step)

    def extrapolate(previous, state, x, y):
        """Linear extrapolation of fields and cell from the last two states on a path."""
        if previous is None:
            return state
        d_prev = distance(previous, state["x"], state["y"])
        d_next = distance(state, x, y)
        # Use the extrapolation only when the new point continues the same direction
        direction = ((state["x"] - previous["x"])*(x - state["x"])/x_step**2 +
                     (state["y"] - previous["y"])*(y - state["y"])/y_step**2)
        if d_prev == 0.0 or direction <= 0.0:
            return state
        r = min(d_next/d_prev, 2.0)
        w = {m: state["w"][m] + r*(state["w"][m] - previous["w"][m]) for m in state["w"]}
        lx = [v + r*(v - v_prev) for v, v_prev in zip(state["lx"], previous["lx"])]
        return {"w": w, "lx": lx}

    def continue_to(phase, start, x, y, previous=None, smallest=smallest_step):
        """Move a converged state to (x, y) along a straight path. The step doubles
        after each success and halves after a failure; returns None when a step of
        smallest times the path fails."""
        state, t, dt = start, 0.0, 1.0
        while t < 1.0:
            s = 1.0 if t + dt > 1.0 - 1e-9 else t + dt
            target_x = start["x"] + s*(x - start["x"]) if s < 1.0 else x
            target_y = start["y"] + s*(y - start["y"]) if s < 1.0 else y
            guess = extrapolate(previous, state, target_x, target_y)
            new = solve(phase, target_x, target_y, guess, reference=state)
            if new is None:
                dt /= 2
                if dt < smallest:
                    return None
                continue
            previous, state, t, dt = state, new, s, min(2*dt, 1.0)
        return state

    # Every converged state of each phase; later solves start from the nearest one
    states = {phase: [] for phase in phases}
    homogeneous_state = {"F": 0.0, "dF": {"x": 0.0, "y": 0.0}}

    def nearest(phase, x, y):
        return min(states[phase], key=lambda s: distance(s, x, y))

    def state_at(phase, x, y, tries=1):
        """State of a phase at any point, continued from its nearest state; None if
        the phase melts or cannot be continued there. With tries=2 a failure is
        retried from the second nearest state (the nearest can be almost homogeneous
        next to a critical point). The nearest states are close during root finding,
        so the step is halved once less than on the grid."""
        if phase == DIS:
            return homogeneous_state
        starts = sorted(states[phase], key=lambda s: distance(s, x, y))[:tries]
        if distance(starts[0], x, y) < 1e-9:
            return starts[0]
        for start in starts:
            state = continue_to(phase, start, x, y, smallest=min(2*smallest_step, 1.0))
            if state is not None:
                states[phase].append(state)
                return state
        return None

    def load_seed(phase):
        """Fields and cell of a seed, interpolated to the grid of the phase if needed."""
        x, y = phases[phase]["point"]
        seed = phases[phase]["seed"]
        if isinstance(seed, dict):
            return {"x": x, "y": y, "w": seed["w"], "lx": list(seed["lx"])}
        path = os.path.join(seed_dir, seed)
        data = scipy.io.loadmat(path, squeeze_me=True) if path.endswith(".mat") else json.load(open(path))
        nx_seed = [int(n) for n in np.atleast_1d(data["nx"])]
        nx = phases[phase].get("params", {}).get("nx", base_params.get("nx", nx_seed))
        w = {}
        for key in data:
            if key.startswith("w_"):
                field = np.reshape(np.array(data[key], dtype=np.float64), nx_seed)
                if list(nx) != nx_seed:
                    field = scipy.ndimage.zoom(field, np.array(nx)/np.array(nx_seed), mode="grid-wrap")
                w[key[2:]] = field
        return {"x": x, "y": y, "w": w, "lx": [float(v) for v in np.atleast_1d(data["lx"])]}

    time_start = time.perf_counter()
    grid = {}       # (phase, i, j) -> converged state at (x_grid[i], y_grid[j])

    # -------------------------------------------------------------------------
    # Stages 1 and 2: seeds, then each phase spreads over the grid
    # -------------------------------------------------------------------------
    print("Grid")
    for phase in phases:
        seed = load_seed(phase)
        i0 = int(np.argmin(np.abs(x_grid - seed["x"])))
        j0 = int(np.argmin(np.abs(y_grid - seed["y"])))
        state = solve(phase, seed["x"], seed["y"], seed, reference=None)
        if state is not None:
            state = continue_to(phase, state, x_grid[i0], y_grid[j0])
        if state is None:
            print(f"  {phase}: left out; its seed does not converge at {phases[phase]['point']} "
                  f"or cannot be moved to the nearest grid point")
            continue
        states[phase].append(state)
        grid[(phase, i0, j0)] = state

        # Breadth-first spreading: every grid point is continued from the neighbour
        # it is reached from, with extrapolation when the point behind that neighbour
        # is known too. A point where the phase fails is not tried again.
        x_min, x_max, y_min, y_max = phases[phase].get("region", (-np.inf, np.inf, -np.inf, np.inf))
        visited = {(i0, j0)}
        queue = deque([(i0, j0)])
        while queue:
            i, j = queue.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n_i, n_j = i + di, j + dj
                if (n_i, n_j) in visited or not (0 <= n_i < len(x_grid) and 0 <= n_j < len(y_grid)):
                    continue
                visited.add((n_i, n_j))
                if not (x_min <= x_grid[n_i] <= x_max and y_min <= y_grid[n_j] <= y_max):
                    continue
                new = continue_to(phase, grid[(phase, i, j)], x_grid[n_i], y_grid[n_j],
                                  previous=grid.get((phase, i - di, j - dj)))
                if new is not None:
                    states[phase].append(new)
                    grid[(phase, n_i, n_j)] = new
                    queue.append((n_i, n_j))
        n_points = sum(1 for key in grid if key[0] == phase)
        print(f"  {phase}: {n_points} grid points  ({count['solves']} solves, "
              f"{time.perf_counter() - time_start:.0f} s)")

    def stable(i, j):
        """Most stable phase on a grid point."""
        values = {p: s["F"] for (p, a, b), s in grid.items() if (a, b) == (i, j)}
        if homogeneous:
            values[DIS] = 0.0
        return min(values, key=values.get) if values else None

    def nearby(x, y):
        """The homogeneous state and the phases with a converged state within one
        grid spacing."""
        return ([DIS] if homogeneous else []) + [
            p for p in phases if states[p] and distance(nearest(p, x, y), x, y) <= 1.01]

    # -------------------------------------------------------------------------
    # Stage 3: boundaries between neighbouring grid points
    # -------------------------------------------------------------------------
    def point(line, t):
        """(x, y) at position t on a grid line ("y", x) or ("x", y): the first entry
        is the axis that varies."""
        axis, value = line
        return (value, t) if axis == "y" else (t, value)

    def difference(a, b, x, y, axis):
        """F_a - F_b at one point and its derivative along the axis (None if unknown).
        A phase that melts or cannot be continued to the point does not exist there
        and counts as just above the other one, so that the root lies where both
        exist (against the homogeneous state: the order-disorder transition)."""
        state_a, state_b = state_at(a, x, y), state_at(b, x, y)
        if state_a is None and state_b is None:
            raise ValueError(f"neither {a} nor {b} converges at ({x:.4f}, {y:.4f})")
        if state_a is None or state_b is None:
            # The derivative is unknown here: the root finder bisects
            return (1e-12 if state_a is None else -1e-12), None
        d_a, d_b = state_a["dF"][axis], state_b["dF"][axis]
        dg = None if d_a is None or d_b is None else d_a - d_b
        return state_a["F"] - state_b["F"], dg

    def newton_root(func, lo, hi, xtol):
        """Root of func, which returns a value and its derivative, in [lo, hi] where
        it changes sign. The first step starts from the end closer to the root, whose
        value and derivative are already known; a step that leaves the bracket
        bisects it."""
        g_lo, dg_lo = func(lo)
        g_hi, dg_hi = func(hi)
        t, g, dg = (lo, g_lo, dg_lo) if abs(g_lo) < abs(g_hi) else (hi, g_hi, dg_hi)
        for _ in range(50):
            t_new = t - g/dg if dg else 0.5*(lo + hi)
            if not lo < t_new < hi:
                t_new = 0.5*(lo + hi)
            if abs(t_new - t) < xtol:
                return t_new
            t = t_new
            g, dg = func(t)
            if (g < 0.0) == (g_lo < 0.0):
                lo, g_lo = t, g
            else:
                hi = t
        raise ValueError("Newton iteration did not converge")

    def find_boundaries(a, b, line, lo, hi):
        """Boundaries on a grid line between lo, where a is stable, and hi, where b is."""
        axis = line[0]
        tol = x_tol if axis == "x" else y_tol
        func = lambda t: difference(a, b, *point(line, t), axis)
        try:
            if derivatives[axis] is not None:
                t = newton_root(func, lo, hi, tol)
            else:
                t = brentq(lambda t: func(t)[0], lo, hi, xtol=tol)
        except ValueError as error:
            (x_lo, y_lo), (x_hi, y_hi) = point(line, lo), point(line, hi)
            print(f"  skipped {a}/{b} between ({x_lo:.4f}, {y_lo:.4f}) and "
                  f"({x_hi:.4f}, {y_hi:.4f}): {error}")
            return []
        x, y = point(line, t)
        if DIS in (a, b):
            state_ab = homogeneous_state
        else:
            # Between two ordered phases the root must be a crossing; if one phase
            # ends there, it is the limit where that phase stops existing
            state_ab, state_other = state_at(a, x, y), state_at(b, x, y)
            if state_ab is None or state_other is None:
                ended = a if state_ab is None else b
                print(f"  skipped {a}/{b} at ({x:.4f}, {y:.4f}): {ended} cannot be continued there")
                return []
        # A third phase lower than a and b where they cross is stable in between.
        # The candidates include phases computed only on neighbouring lines.
        # Differences below 1e-6 are not resolved: near a critical point all phases
        # have F ~ 0.
        for c in nearby(x, y):
            if c in (a, b):
                continue
            state_c = state_at(c, x, y)
            if state_c is not None and state_c["F"] < state_ab["F"] - 1e-6:
                return find_boundaries(a, c, line, lo, t) + find_boundaries(c, b, line, t, hi)
        print(f"  {a}/{b}: ({x:.4f}, {y:.4f})")
        return [{"phases": sorted([a, b]), "x": float(x), "y": float(y), "along": axis}]

    print("Boundaries")
    boundaries = []
    for i, x in enumerate(x_grid):                         # along y at fixed x
        for j in range(len(y_grid) - 1):
            a, b = stable(i, j), stable(i, j + 1)
            if a is not None and b is not None and a != b:
                boundaries += find_boundaries(a, b, ("y", x), y_grid[j], y_grid[j + 1])
    for j, y in enumerate(y_grid):                         # along x at fixed y
        for i in range(len(x_grid) - 1):
            a, b = stable(i, j), stable(i + 1, j)
            if a is not None and b is not None and a != b:
                boundaries += find_boundaries(a, b, ("x", y), x_grid[i], x_grid[i + 1])

    # -------------------------------------------------------------------------
    # Stage 4: refine the cells in which a boundary ends
    # -------------------------------------------------------------------------
    # A boundary that only passes through a cell crosses its edges twice; one that
    # crosses them an odd number of times ends inside (a triple point, a critical
    # point or a skipped root). Such a cell is split into four: the most stable
    # phase is found at the five new points and the boundaries on the two new lines.
    point_phase = {(round(x, 10), round(y, 10)): stable(i, j)
                   for i, x in enumerate(x_grid) for j, y in enumerate(y_grid)}

    def stable_at(x, y):
        """Most stable phase at any point among the phases computed nearby. A phase
        missed here would make a wrong label, so a failure is retried once."""
        key = (round(x, 10), round(y, 10))
        if key not in point_phase:
            values = {}
            for p in nearby(x, y):
                state = state_at(p, x, y, tries=2)
                if state is not None:
                    values[p] = state["F"]
            point_phase[key] = min(values, key=values.get) if values else None
        return point_phase[key]

    def boundary_ends_inside(cell):
        crossings = {}
        for b in on_edges(boundaries, cell, x_grid[-1], y_grid[-1]):
            crossings[tuple(b["phases"])] = crossings.get(tuple(b["phases"]), 0) + 1
        return any(n % 2 for n in crossings.values())

    cells = [(x_grid[i], x_grid[i + 1], y_grid[j], y_grid[j + 1])
             for i in range(len(x_grid) - 1) for j in range(len(y_grid) - 1)]
    candidates = list(cells)    # cells is the current set of cells, refined or not
    for level in range(refine):
        chosen = [cell for cell in candidates if boundary_ends_inside(cell)]
        print(f"Refinement {level + 1}: {len(chosen)} cells")
        candidates = []
        for x0, x1, y0, y1 in chosen:
            xm, ym = 0.5*(x0 + x1), 0.5*(y0 + y1)
            for line, t0, t1 in ((("y", xm), y0, ym), (("y", xm), ym, y1),
                                 (("x", ym), x0, xm), (("x", ym), xm, x1)):
                a, b = stable_at(*point(line, t0)), stable_at(*point(line, t1))
                if a is not None and b is not None and a != b:
                    boundaries += find_boundaries(a, b, line, t0, t1)
            children = [(x0, xm, y0, ym), (xm, x1, y0, ym), (x0, xm, ym, y1), (xm, x1, ym, y1)]
            cells.remove((x0, x1, y0, y1))
            cells += children
            candidates += children
        print(f"  ({count['solves']} solves, {time.perf_counter() - time_start:.0f} s)")

    # -------------------------------------------------------------------------
    # Stage 5: connect the boundary points into curves (see connect)
    # -------------------------------------------------------------------------
    segments, junctions, breaks = connect(boundaries, cells, x_step, y_step, x_grid[-1], y_grid[-1])
    print(f"Curves: {len(segments)} segments, {len(junctions)} junctions, {len(breaks)} breaks")
    for brk in breaks:
        print(f"  break near ({brk['point'][0]:.4f}, {brk['point'][1]:.4f}): " + ", ".join(
            f"{'/'.join(e['phases'])} ends at ({e['point'][0]:.4f}, {e['point'][1]:.4f})" for e in brk["ends"]))

    elapsed = time.perf_counter() - time_start
    print(f"{count['solves']} SCFT solves, {count['iterations']} iterations, {elapsed:.0f} s")

    data = {"x_grid": x_grid.tolist(), "y_grid": y_grid.tolist(),
            "free_energy": [{"phase": p, "x": float(x_grid[i]), "y": float(y_grid[j]),
                             "F": s["F"], "dF": s["dF"]}
                            for (p, i, j), s in sorted(grid.items(), key=lambda item: item[0])],
            "stable": [[stable(i, j) for j in range(len(y_grid))] for i in range(len(x_grid))],
            "refined_points": [{"x": x, "y": y, "phase": p} for (x, y), p in point_phase.items()
                               if not (np.any(np.isclose(x, x_grid)) and np.any(np.isclose(y, y_grid)))],
            "boundaries": boundaries,
            "cells": [[float(v) for v in cell] for cell in cells],
            "segments": segments, "junctions": junctions, "breaks": breaks,
            "solves": count["solves"], "iterations": count["iterations"], "seconds": elapsed}
    with open(output + ".json", "w") as fp:
        json.dump(data, fp, indent=1)
    plot(data, output + ".png", labels=labels, mirror_x=mirror_x)
    print(f"Saved {output}.json and {output}.png")
    return data


def plot(data, filename, labels=("x", "y"), mirror_x=None):
    """Draw the boundary curves of the data saved by run(), with the curves of a
    junction meeting at its centre and a red cross at each break, label each region
    with its phase and, with mirror_x, add the mirror image at x = mirror_x."""
    x_grid, y_grid = np.array(data["x_grid"]), np.array(data["y_grid"])
    x_step = x_grid[1] - x_grid[0]
    copies = [lambda x: x] if mirror_x is None else [lambda x: x, lambda x: 2*mirror_x - x]

    fig, ax = plt.subplots(figsize=(6, 5))
    lines = [s["points"] for s in data["segments"]]
    lines += [[e["point"], j["point"]] for j in data["junctions"] for e in j["ends"]]
    xs = np.array([b["x"] for b in data["boundaries"]])
    ys = np.array([b["y"] for b in data["boundaries"]])
    for reflect in copies:
        for p, q in lines:
            ax.plot(reflect(np.array([p[0], q[0]])), [p[1], q[1]], "k-", lw=1.2)
        ax.plot(reflect(xs), ys, "k.", ms=3)
        for brk in data["breaks"]:
            ax.plot(reflect(np.array([brk["point"][0]])), [brk["point"][1]], "rx", ms=9, mew=2)

    # Phase names at the centre of the grid points where each phase is stable; a
    # region centred close to the mirror line gets one name on that line
    for name in sorted({p for column in data["stable"] for p in column if p is not None}):
        cells = np.array([(x_grid[i], y_grid[j]) for i, column in enumerate(data["stable"])
                          for j, p in enumerate(column) if p == name])
        x_c, y_c = cells.mean(axis=0)
        if mirror_x is not None and abs(x_c - mirror_x) < 1.5*x_step:
            ax.text(mirror_x, y_c, name, ha="center", va="center", fontsize=11)
            continue
        for reflect in copies:
            ax.text(reflect(x_c), y_c, name, ha="center", va="center", fontsize=11)

    x_lo, x_hi = x_grid[0], x_grid[-1]
    if mirror_x is not None:
        x_lo, x_hi = min(x_lo, 2*mirror_x - x_hi), max(x_hi, 2*mirror_x - x_lo)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_grid[0], y_grid[-1])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
