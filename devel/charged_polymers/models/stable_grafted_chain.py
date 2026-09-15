"""Direct, scaled discrete Gaussian chains for brush validation.

This deliberately slow reference backend uses separable bond matrices (NumPy
or CuPy), preserving complex linearity. It avoids the global roundoff leakage
of FFT/DST propagation in strongly depleted regions. Scaling is bookkeeping;
densities use ONE partition normalization, never a separate fit per bead.

``absorbing`` means the Dirichlet image kernel, as in the spectral solver.
``hard-reflecting`` means a truncated Gaussian at z=0 and a reflected Gaussian
at z=L, as in prl_scft._bond_conv. These are different discrete chain models.
Quenched grafting fixes every lateral anchor; a nonzero graft width still
averages the vertical position of each anchor over its prescribed source.
"""
import numpy as np


def gaussian_bond_matrix(n, length, bond_length, boundary):
    """Cell-centred quadrature of a symmetric Gaussian bond kernel."""
    if n < 1 or length <= 0 or bond_length <= 0:
        raise ValueError("Positive grid, box length and bond length required.")
    dx = length / n
    z = (np.arange(n, dtype=np.longdouble) + .5) * dx
    x, y = z[:, None], z[None, :]
    variance = np.longdouble(bond_length)**2 / 3
    norm = dx / np.sqrt(2 * np.longdouble(np.pi) * variance)

    def g(d):
        return norm * np.exp(-d*d/(2*variance))

    if boundary == "hard-reflecting":
        kernel = g(x-y) + g(x+y-2*length)
    elif boundary in ("periodic", "reflecting", "absorbing"):
        period = length if boundary == "periodic" else 2*length
        # Include image tails beyond twelve Gaussian standard deviations.
        images = int(np.ceil(12*np.sqrt(variance)/period)) + 2
        kernel = np.zeros((n, n), dtype=np.longdouble)
        for m in range(-images, images+1):
            kernel += g(x-y+m*period)
            if boundary != "periodic":
                kernel += (1 if boundary == "reflecting" else -1)*g(x+y+m*period)
    else:
        raise ValueError(f"Unknown bond boundary: {boundary}")
    # The exact heat kernel is nonnegative. Remove only cancellation
    # roundoff in constructing the REAL kernel, never clip propagators.
    if np.min(kernel) < -100*np.finfo(np.longdouble).eps*np.max(kernel):
        raise FloatingPointError("Negative Gaussian image kernel.")
    return np.asarray(np.maximum((kernel+kernel.T)/2, 0), dtype=np.float64)


class ScaledGraftedChain:
    """One homopolymer; potential is per bead, density has spatial mean 1."""

    def __init__(self, nx, lx, n_beads, bond_length, boundaries,
                 grafting="annealed", platform="cpu-mkl"):
        if grafting not in ("annealed", "quenched"):
            raise ValueError("grafting must be 'annealed' or 'quenched'.")
        if len(nx) != len(lx) or len(nx) != len(boundaries) or n_beads < 1:
            raise ValueError("Invalid chain geometry.")
        self.nx, self.n_beads, self.grafting = tuple(nx), n_beads, grafting
        if platform == "cuda":
            import cupy as cp
            self.xp = cp
        elif platform in ("cpu-mkl", "cpu-fftw"):
            self.xp = np
        else:
            raise ValueError(f"Unsupported platform: {platform}")
        self.kernels = [self.xp.asarray(gaussian_bond_matrix(n, L, bond_length, b))
                        for n, L, b in zip(nx, lx, boundaries)]

    def _host(self, value):
        return np.asarray(value) if self.xp is np else self.xp.asnumpy(value)

    def bond(self, q):
        xp = self.xp
        for axis, k in enumerate(self.kernels):
            q = xp.moveaxis(xp.tensordot(k, q, axes=(1, axis)), 0, axis)
        return q

    def _normalize(self, q, log_scale):
        scale = float(self.xp.max(self.xp.abs(q)))
        if not np.isfinite(scale) or scale == 0:
            raise FloatingPointError("Zero or nonfinite propagator; reduce field range/grid error.")
        return q/scale, log_scale+np.log(scale)

    def compute(self, potential, source=None):
        xp = self.xp
        v = xp.asarray(potential, dtype=xp.complex128).reshape(self.nx)
        if not bool(xp.all(xp.isfinite(v))):
            raise FloatingPointError("Nonfinite chain potential.")
        shift = float(xp.min(v.real))
        ew = xp.exp(-(v-shift))
        src = xp.ones(self.nx) if source is None else xp.asarray(source).reshape(self.nx)
        if self.grafting == "quenched":
            if source is None or not bool(xp.all(xp.isreal(src) & (src.real >= 0))):
                raise ValueError("Quenched grafting needs a nonnegative real source.")
            if not bool(xp.all(xp.sum(src, axis=-1) > 0)):
                raise ValueError("Every lateral anchor must have nonzero source weight.")

        # b[n] includes bead n, whereas the forward incoming a[n] does not.
        # This avoids dividing a density by a potentially underflowed ew.
        b = [None]*self.n_beads
        b[-1] = self._normalize(ew, 0.)
        for n in range(self.n_beads-2, -1, -1):
            b[n] = self._normalize(ew*self.bond(b[n+1][0]), b[n+1][1])
        b0, lb0 = b[0]
        if self.grafting == "quenched":
            col = xp.mean(src*b0, axis=-1, keepdims=True)
            if bool(xp.any(xp.abs(col) == 0)):
                raise FloatingPointError("A quenched anchor partition vanished (CL drift pole).")
            log_q = xp.mean(xp.log(col)) + lb0 - self.n_beads*shift
            a, la = self._normalize(src/col, 0.)
            denominator, log_denominator = 1., lb0
        else:
            denominator = xp.mean(src*b0)
            if complex(denominator) == 0:
                raise FloatingPointError("Chain partition vanished (CL drift pole).")
            log_q = xp.log(denominator) + lb0 - self.n_beads*shift
            log_denominator = lb0
            a, la = self._normalize(src, 0.)
        density = xp.zeros(self.nx, dtype=xp.complex128)
        slices = []
        for n in range(self.n_beads):
            bn, lbn = b[n]
            sl = a*bn * xp.exp(la+lbn-log_denominator)/denominator
            if n == 0:
                self.anchor_density = self._host(sl)
            slices.append(complex(xp.mean(sl)))
            density += sl/self.n_beads
            if n+1 < self.n_beads:
                a, la = self._normalize(self.bond(ew*a), la)
        self.log_partition = complex(log_q)
        self.partition_error = float(np.max(np.abs(np.asarray(slices)-1)))
        self.density = self._host(density)
        if not np.isfinite(self.partition_error) or self.partition_error > 1e-7:
            raise FloatingPointError(f"Relative contour partition error {self.partition_error:.3e}.")
        return self.density


class StableBrushSolver:
    """Small adapter for BrushChargedCLFTS: linear, single-block polymers only.

    Both density and Hamiltonian use the quenched log partition when selected.
    Access log Q directly for extreme fields; exp(log Q) is diagnostic only.
    """

    def __init__(self, params, polymers, segment_lengths):
        if params["chain_model"] != "discrete":
            raise ValueError("The stable brush reference backend is discrete only.")
        self.polymers, self.ds = polymers, params["ds"]
        self.chains, self.types, self.labels = [], [], []
        boundaries = params.get("bond_boundaries", params.get(
            "bc", ["reflecting"]*(2*len(params["nx"])))[::2])
        for p in polymers:
            if len(p["blocks"]) != 1:
                raise ValueError("Stable brush backend supports single-block linear chains only.")
            block = p["blocks"][0]
            t = block["type"]
            n = round(block["length"]/self.ds)
            if n < 1 or abs(n*self.ds-block["length"]) > 1e-10:
                raise ValueError("Chain length must be an integer number of beads.")
            gp = p.get("grafting_points", {})
            if len(gp) > 1:
                raise ValueError("At most one end may be grafted.")
            if gp and int(next(iter(gp))) not in (int(block.get("v", 0)), int(block.get("u", 1))):
                raise ValueError("Graft vertex must be an end of the block.")
            self.labels.append(str(next(iter(gp.values()))) if gp else None)
            self.types.append(t)
            self.chains.append(ScaledGraftedChain(
                params["nx"], params["lx"], n, segment_lengths[t]*np.sqrt(self.ds),
                boundaries, params.get("grafting", "annealed") if gp else "annealed",
                params.get("platform", "cpu-mkl")))

    def compute_propagators(self, fields, q_init=None):
        self.phi = {t: np.zeros(np.prod(self.chains[0].nx), dtype=np.complex128)
                    for t in fields}
        for p, t, label, chain in zip(self.polymers, self.types, self.labels, self.chains):
            source = None if label is None else q_init[label]
            self.phi[t] += p["volume_fraction"]*chain.compute(
                self.ds*fields[t], source).reshape(-1)

    def compute_concentrations(self):
        pass  # computed together with the scaled propagators

    def get_total_concentration(self, t):
        return self.phi[t].copy()

    def get_log_partition(self, p):
        return self.chains[p].log_partition

    def get_total_partition(self, p):
        with np.errstate(over="ignore", under="ignore"):
            return np.exp(self.get_log_partition(p))

    def check_total_partition(self):
        return all(c.partition_error <= 1e-7 for c in self.chains)
