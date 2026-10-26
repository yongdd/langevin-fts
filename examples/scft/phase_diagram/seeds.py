"""Initial fields of common phases of AB block copolymers with a minority A block,
for the seeds of phase_diagram.run().

Each function returns {"w": {"A": w_A, "B": w_B}, "lx": lx}, the same initial
fields as the scripts of examples/scft/phases. The first SCFT solve of run()
converges a seed at its point, so these fields need only the symmetry and the
rough shape of the phase.
"""

import os

import numpy as np
import scipy.io
import scipy.ndimage

PHASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases")


def lamellae(nx, lx):
    """One lamellar period along the single axis of a 1D grid."""
    w = np.cos(2*np.pi*np.arange(nx[0])/nx[0])
    return {"w": {"A": w, "B": -w}, "lx": list(lx)}


def wells(nx, lx, positions):
    """Smoothed wells of w_A at the given fractional positions (A domains)."""
    w_A = np.zeros(nx)
    for position in positions:
        index = tuple(np.round(np.array(position)*np.array(nx)).astype(int) % np.array(nx))
        w_A[index] = -1/(np.prod(lx)/np.prod(nx))
    w_A = scipy.ndimage.gaussian_filter(w_A, sigma=np.min(nx)/15, mode="wrap")
    return {"w": {"A": w_A, "B": np.zeros(nx)}, "lx": list(lx)}


def cylinders(nx, lx):
    """Hexagonally packed A cylinders in a 2D rectangular cell with sides in the
    ratio sqrt(3), which holds two of them."""
    return wells(nx, lx, [(0.0, 0.0), (0.5, 0.5)])


def bcc_spheres(nx, lx):
    """A spheres on a body-centred cubic lattice."""
    return wells(nx, lx, [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)])


def double_gyroid(nx, lx=None):
    """Double gyroid fields of examples/scft/phases/DG.mat, interpolated to nx, in
    its cell or in a cubic cell of side lx."""
    data = scipy.io.loadmat(os.path.join(PHASES_DIR, "DG.mat"), squeeze_me=True)
    nx_data = [int(n) for n in data["nx"]]
    w = {}
    for m in ("A", "B"):
        field = np.reshape(data[f"w_{m}"], nx_data)
        if list(nx) != nx_data:
            field = scipy.ndimage.zoom(field, np.array(nx)/np.array(nx_data), mode="grid-wrap")
        w[m] = field
    return {"w": w, "lx": [float(x) for x in data["lx"]] if lx is None else list(lx)}
