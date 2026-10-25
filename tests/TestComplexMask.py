#!/usr/bin/env python3
"""Array handling of the Python bindings.

1. A mask given to the complex-field factory is read as a real array (the
   same integral as with the real-field factory).
2. Strided and Fortran-ordered NumPy arrays are copied before the C++ side
   reads them, so they give the same result as contiguous copies.
3. Objects created by the factory keep the objects they point to alive
   (dropping the Python references to the molecules, optimizer and box
   before computing must not crash). Weak references check every keep_alive
   of the bindings, including those of the space group: each object lives as
   long as the objects that point to it and is released after them.
"""

import gc
import sys
import weakref

import numpy as np
from polymerfts import _core


def cpu_platform():
    avail = _core.PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return cpu[0] if cpu else None


def main():
    platform = cpu_platform()
    if platform is None:
        print("SKIP: no CPU platform in this build")
        return 77  # ctest SKIP_RETURN_CODE
    ok = True
    nx, lx = [8, 8, 8], [2.0, 2.0, 2.0]
    n = int(np.prod(nx))
    rng = np.random.default_rng(7)

    # ---- mask with the real and the complex factory
    mask = np.ones(n)
    mask[:64] = 0.0
    integrals = {}
    for field_type in ("real", "complex"):
        factory = _core.PlatformSelector.create_factory(platform, False, field_type)
        cb = factory.create_computation_box(nx, lx, mask=mask)
        ones = np.ones(n, dtype=np.complex128 if field_type == "complex" else np.float64)
        integrals[field_type] = cb.integral(ones)
    expected = np.prod(lx) * (n - 64) / n
    if not (np.isclose(integrals["real"], expected) and np.isclose(integrals["complex"], expected)):
        print(f"FAILED: masked integral real {integrals['real']} complex {integrals['complex']} expected {expected}")
        ok = False

    # ---- strided view and Fortran order give the same answer as contiguous copies
    factory = _core.PlatformSelector.create_factory(platform, False, "real")
    cb = factory.create_computation_box(nx, lx)
    x = rng.normal(size=2 * n)
    if not np.isclose(cb.mean(x[::2]), cb.mean(np.ascontiguousarray(x[::2]))):
        print("FAILED: strided view read as contiguous")
        ok = False
    y = np.asfortranarray(rng.normal(size=nx))
    if not np.isclose(cb.integral(y), cb.integral(np.ascontiguousarray(y))):
        print("FAILED: Fortran-ordered array read as C-ordered")
        ok = False
    z = rng.normal(size=n).astype(np.float32)
    if not np.isclose(cb.mean(z), np.mean(z.astype(np.float64))):
        print("FAILED: float32 array not cast")
        ok = False
    # zero_mean works in place, so a view that would have to be copied is rejected
    try:
        cb.zero_mean(x[::2])
        print("FAILED: zero_mean accepted a strided view")
        ok = False
    except Exception:
        pass
    y_c = np.ascontiguousarray(y)
    cb.zero_mean(y_c)
    if not np.isclose(cb.mean(y_c), 0.0):
        print("FAILED: zero_mean did not change the array in place")
        ok = False

    # ---- the propagator computation keeps its inputs alive
    def make_solver():
        factory = _core.PlatformSelector.create_factory(platform, False, "real")
        cb = factory.create_computation_box(nx, lx)
        molecules = factory.create_molecules_information("continuous", 0.1, {"A": 1.0, "B": 1.0})
        molecules.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
        optimizer = factory.create_propagator_computation_optimizer(molecules, True)
        return factory.create_propagator_computation(cb, molecules, optimizer, "rqm4")

    solver = make_solver()
    gc.collect()
    w = {"A": rng.normal(0.0, 1.0, n), "B": rng.normal(0.0, 1.0, n)}
    solver.compute_propagators(w, None)
    solver.compute_concentrations()
    q = solver.get_total_partition(0)
    phi = solver.get_total_concentration("A") + solver.get_total_concentration("B")
    if not (np.isfinite(q) and q > 0 and np.isclose(np.mean(phi), 1.0)):
        print(f"FAILED: computation after dropping references: Q = {q}, mean phi = {np.mean(phi)}")
        ok = False
    cb_from_solver = solver.get_cb()
    del solver
    gc.collect()
    if cb_from_solver.get_total_grid() != n:
        print("FAILED: computation box returned by the solver outlived it incorrectly")
        ok = False

    # ---- each keep_alive, through weak references: an object that a C++ object
    # points to stays alive with it and is released after it
    def alive(refs):
        gc.collect()
        return sorted(name for name, ref in refs.items() if ref() is not None)

    def make_molecules():
        molecules = factory.create_molecules_information("continuous", 0.1, {"A": 1.0, "B": 1.0})
        molecules.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
        return molecules

    molecules = make_molecules()
    optimizer = factory.create_propagator_computation_optimizer(molecules, True)
    refs = {"molecules": weakref.ref(molecules)}
    del molecules
    if alive(refs) != ["molecules"]:
        print("FAILED: the optimizer does not keep its molecules alive")
        ok = False
    del optimizer
    if alive(refs):
        print("FAILED: the molecules outlive their optimizer")
        ok = False

    def make_solver_with_space_group():
        cb = factory.create_computation_box(nx, lx)
        molecules = make_molecules()
        optimizer = factory.create_propagator_computation_optimizer(molecules, True)
        space_group = _core.SpaceGroup(nx, "Im-3m", 529)
        solver = factory.create_propagator_computation(cb, molecules, optimizer, "rqm4", space_group)
        refs = {"box": weakref.ref(cb), "molecules": weakref.ref(molecules),
                "optimizer": weakref.ref(optimizer), "space group": weakref.ref(space_group)}
        return solver, cb, refs

    solver, cb, refs = make_solver_with_space_group()
    if alive(refs) != sorted(refs):
        print(f"FAILED: the solver keeps only {alive(refs)} alive")
        ok = False
    del solver
    if alive(refs) != ["box", "space group"]:
        print(f"FAILED: without the solver, alive objects are {alive(refs)}; "
              "the box must keep only its space group")
        ok = False
    del cb
    if alive(refs):
        print(f"FAILED: {alive(refs)} outlive the box and the solver")
        ok = False

    cb = factory.create_computation_box(nx, lx)
    space_group = _core.SpaceGroup(nx, "Im-3m", 529)
    cb.set_space_group(space_group)
    refs = {"space group": weakref.ref(space_group)}
    del space_group
    if alive(refs) != ["space group"]:
        print("FAILED: set_space_group does not keep the space group alive")
        ok = False
    del cb
    if alive(refs):
        print("FAILED: the space group outlives its box")
        ok = False

    if ok:
        print("Test passed, array handling of the bindings")
        return 0
    print("Test failed, array handling of the bindings")
    return 1


if __name__ == "__main__":
    sys.exit(main())
