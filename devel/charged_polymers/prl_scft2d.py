"""2D (x periodic, z hard-wall) pilot of the PRL-2025 correlation SCFT:
lateral microphase separation of PE brushes (their Fig. 4(a,b),
sigma = 0.03 nm^-2, z+ = 3, rho_b = 1 mM).

Same physics as prl_scft.py (real-space discrete-chain backend, exact
guarded-Newton PB, Born-smoothed LDA self-energies, exact xi
elimination), promoted to 2D. In 2D the micelles of their 3D
calculation appear as stripes/cylinders; the deliverable here is the
lateral symmetry breaking itself and its length scale, not the
hexagonal lattice.

NOTE on provenance: the recorded Fig-4(a,b) states were produced by the
drivers in ~/polymer/dh_salt_runs (fig4ab_C.py and variants), which add
finite-amplitude stripe seeding — the uniform film is LINEARLY stable,
so the small-noise __main__ below alone converges back to uniform. The
GC-only / lag-corrected-salt model choices are documented in
RESULTS_PRL_SCFT.md 3k. CAVEAT: the default Lz=30 nm ceiling clips the
low-salt SWOLLEN mean-field states (osmotic h ~ 50-60 nm) during the
alpha ladder — only the collapsed states (h < ~10 nm) are box-safe.

Run: python prl_scft2d.py --out fig4ab.npz  (see __main__ for options)
"""
import numpy as np
from scipy.special import erfc
from scipy.ndimage import correlate1d, gaussian_filter
from scipy import sparse
from scipy.sparse.linalg import splu


class PRLBrush2D:
    def __init__(self, zplus=3, sigma=0.03, Ib_molar=6e-3, N=100, b=1.0,
                 v=1.0, aP=0.25, aion=0.25, aplus=0.25, lB=0.7, chi=0.5,
                 Lx=80.0, Nx=80, Lz=30.0, Nz=60, correlations=True,
                 counterions=True):
        self.N, self.b, self.v, self.chi, self.lB = N, b, v, chi, lB
        self.zP, self.zp, self.sigma = -1.0, float(zplus), sigma
        self.aP, self.aion, self.aplus = aP, aion, aplus
        self.Lx, self.Nx, self.Lz, self.Nz = Lx, Nx, Lz, Nz
        self.dx, self.dz = Lx / Nx, Lz / Nz
        self.x = (np.arange(Nx) + 0.5) * self.dx
        self.z = (np.arange(Nz) + 0.5) * self.dz
        self.corr = correlations
        self.with_counterions = counterions
        Ib = Ib_molar * 0.6022
        self.rho_b = 2.0 * Ib / (self.zp * (self.zp + 1.0))
        self.kappa_b = np.sqrt(8.0 * np.pi * lB * Ib)

        def kern(d, dcell):
            half = int(np.ceil(6.0 * b / dcell))
            u = np.arange(-half, half + 1) * dcell
            k = np.exp(-1.5 * (u / b) ** 2)
            return k / k.sum()
        self.kx = kern(6.0, self.dx)
        self.kz = kern(6.0, self.dz)
        # graft source: half-Gaussian on the wall, uniform in x
        srcz = np.exp(-0.5 * (self.z / (0.5 * b)) ** 2)
        self.src = np.broadcast_to(srcz, (Nx, Nz)).copy()

        # fields (Nx, Nz)
        self.W = np.zeros((Nx, Nz))
        self.psi = np.zeros((Nx, Nz))
        self.uP = np.zeros((Nx, Nz))
        self.up = np.zeros((Nx, Nz))
        self.um = np.zeros((Nx, Nz))
        self._plap = self._build_laplacian()

    # ---------- chain sector ----------
    def _bond_conv(self, q):
        # z: hard wall via zero ghosts (mode='constant' zeroes BOTH ends;
        # the far top is unreachable, so absorbing there is harmless)
        out = correlate1d(q, self.kz, axis=1, mode="constant", cval=0.0)
        # x: periodic
        return correlate1d(out, self.kx, axis=0, mode="wrap")

    def chain_density(self, VP):
        ew = np.exp(-np.clip(VP, -300.0, 300.0))
        N = self.N
        # backward propagator first (free end -> bead 1)
        qb = np.empty((N,) + VP.shape)
        q = ew.copy()
        q /= max(q.max(), 1e-300)
        qb[0] = q
        for n in range(1, N):
            q = ew * self._bond_conv(q)
            q /= max(q.max(), 1e-300)
            qb[n] = q
        # QUENCHED grafting (their SI Eq. S30): q~(r;1) = src / q_dagger(r;1),
        # which pins the bead-1 (graft) LATERAL distribution to uniform
        # sigma. Without this division the grafts are effectively annealed
        # and the brush dewets laterally (chains migrate along x),
        # violating the fixed grafting-density constraint.
        qdag1 = qb[N - 1]
        floor = 1e-10 * float(qdag1.max())
        qf = np.empty_like(qb)
        q = ew * self.src / np.maximum(qdag1, floor)
        q /= max(q.max(), 1e-300)
        qf[0] = q
        for n in range(1, N):
            q = ew * self._bond_conv(q)
            q /= max(q.max(), 1e-300)
            qf[n] = q
        iew = 1.0 / np.maximum(ew, 1e-300)
        phi = np.zeros_like(VP)
        for n in range(N):
            sl = qf[n] * qb[N - 1 - n] * iew
            m = sl.sum() * self.dx * self.dz
            phi += sl / max(m, 1e-300)
        # each bead carries sigma*Lx monomers total
        return phi * (self.sigma * self.Lx)

    # ---------- electrostatics ----------
    def _build_laplacian(self):
        Nx, Nz = self.Nx, self.Nz
        ix, iz = 1.0 / self.dx ** 2, 1.0 / self.dz ** 2
        n = Nx * Nz

        def idx(i, j):
            return i * Nz + j
        rows, cols, vals = [], [], []
        for i in range(Nx):
            for j in range(Nz):
                p = idx(i, j)
                diag = 0.0
                for ii in [(i - 1) % Nx, (i + 1) % Nx]:
                    rows.append(p); cols.append(idx(ii, j)); vals.append(ix)
                    diag -= ix
                for jj in (j - 1, j + 1):
                    if 0 <= jj < Nz:
                        rows.append(p); cols.append(idx(i, jj)); vals.append(iz)
                        diag -= iz
                    # Neumann: missing neighbor -> no flux (mirror), drop term
                rows.append(p); cols.append(p); vals.append(diag)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def solve_pb(self, rhoP, n_newton=60):
        psi = self.psi.ravel().copy()
        L = self._plap
        rP = rhoP.ravel()
        up, um = self.up.ravel(), self.um.ravel()

        nC = (abs(self.zP) * self.sigma * self.N * self.Lx) \
            if getattr(self, "with_counterions", True) else 0.0

        def dens(p):
            bp = np.exp(-np.clip(self.zp * p + up, -300, 300))
            bm = np.exp(-np.clip(-p + um, -300, 300))
            if nC > 0.0:
                bC = np.exp(-np.clip(p + um, -300, 300))
                rC = nC * bC / max(bC.sum() * self.dx * self.dz, 1e-280)
            else:
                rC = 0.0
            return self.rho_b * bp, self.zp * self.rho_b * bm, rC

        def resid(p):
            rp_, rm_, rC_ = dens(p)
            return (L @ p + 4 * np.pi * self.lB *
                    (self.zP * rP + rC_ + self.zp * rp_ - rm_)), rp_, rm_, rC_

        F, rp_, rm_, rC_ = resid(psi)
        fn = np.linalg.norm(F)
        tol = 1e-8 * max(1.0, 4 * np.pi * self.lB
                         * (np.abs(rP).max() + self.zp * self.rho_b))
        for _ in range(n_newton):
            if np.abs(F).max() < tol:
                break
            diagm = 4 * np.pi * self.lB * (self.zp ** 2 * rp_ + rm_
                                           + (rC_ if nC > 0 else 0.0))
            A = (L - sparse.diags(diagm)).tocsc()
            dpsi = splu(A).solve(-F)
            t = 1.0
            while True:
                F_t, rp_t, rm_t, rC_t = resid(psi + t * dpsi)
                fn_t = np.linalg.norm(F_t)
                if fn_t < fn or t < 1e-8:
                    psi += t * dpsi
                    F, rp_, rm_, rC_, fn = F_t, rp_t, rm_t, rC_t, fn_t
                    break
                t *= 0.5
        if nC == 0.0:
            psi -= psi.reshape(self.Nx, self.Nz)[:, -1].mean()
        self.psi = psi.reshape(self.Nx, self.Nz)
        _F, rp_, rm_, rC_ = resid(psi)
        self.rhoC = (rC_ if nC > 0 else np.zeros_like(rp_)
                     ).reshape(self.Nx, self.Nz)
        return (rp_.reshape(self.Nx, self.Nz),
                rm_.reshape(self.Nx, self.Nz))

    # ---------- correlations ----------
    @staticmethod
    def _u_fn(x):
        return 1.0 - x * np.exp(x * x / np.pi) * erfc(x / np.sqrt(np.pi))

    def self_energies(self, rhoP, rhop, rhom):
        if not self.corr:
            zz = np.zeros_like(rhoP)
            return zz, zz, zz
        rhoC = getattr(self, "rhoC", 0.0)
        I0 = 0.5 * (rhoP + self.zp ** 2 * rhop + rhoC + rhom)
        a_min = min(self.aion, self.aplus)
        I0 = gaussian_filter(I0, sigma=(a_min / self.dx, a_min / self.dz),
                             mode=("wrap", "nearest"))
        kap = np.sqrt(np.maximum(8 * np.pi * self.lB * I0, 0.0))
        du = lambda a, zq: (zq ** 2) * self.lB / (2 * a) * \
            (self._u_fn(a * kap) - self._u_fn(a * self.kappa_b))
        return du(self.aP, 1.0), du(self.aplus, self.zp), du(self.aion, 1.0)

    # ---------- SCF loop ----------
    def solve(self, max_iter=6000, tol=1e-3, lam=0.01, cap=0.05,
              verbose_every=200):
        rhoP = self.chain_density(self.W)
        err = 1e9
        for it in range(max_iter):
            rhop, rhom = self.solve_pb(rhoP)
            uP, up, um = self.self_energies(rhoP, rhop, rhom)
            lam_u = min(5 * lam, 0.2)
            self.uP = (1 - lam_u) * self.uP + lam_u * uP
            self.up = (1 - lam_u) * self.up + lam_u * up
            self.um = (1 - lam_u) * self.um + lam_u * um
            phi = np.clip(self.v * rhoP, 0.0, 0.999)
            xi = -np.log(1.0 - phi) - self.chi * phi
            W_new = self.chi * (1 - phi) + xi + self.zP * self.psi + self.uP
            err = float(np.abs(W_new - self.W).max())
            if err < tol and it > 10:
                break
            dW = lam * (W_new - self.W)
            mx = np.abs(dW).max()
            if mx > cap:
                dW *= cap / mx
            self.W = self.W + dW
            rhoP = self.chain_density(self.W)
            if verbose_every and it % verbose_every == 0:
                col = rhoP.sum(axis=1) * self.dz
                print(f"    it {it}: err={err:.3e} lat_contrast="
                      f"{(col.max()-col.min())/max(col.mean(),1e-9):.3f}",
                      flush=True)
        self.rhoP = rhoP
        self.n_iter, self.err = it + 1, err
        return rhoP


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    s = PRLBrush2D(sigma=args.sigma, correlations=False)
    # mean-field alpha ladder (stays x-uniform)
    for f in [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]:
        s.zP = -f
        s.solve(max_iter=4000, tol=1e-3, lam=0.02, cap=0.1, verbose_every=0)
        print(f"[MF a={f:.2f}] iters={s.n_iter} err={s.err:.2e}", flush=True)
    # correlation ramp with lateral noise seeding
    rng = np.random.default_rng(args.seed)
    s.corr = True
    for cscale in [0.25, 0.5, 0.75, 1.0]:
        base = s.self_energies
        s.self_energies = (lambda rP, rp, rm, _b=base, _c=cscale:
                           tuple(_c * a for a in _b(rP, rp, rm)))
        s.W += 0.02 * rng.standard_normal(s.W.shape)
        s.solve(max_iter=6000, tol=1e-3, lam=0.01, cap=0.05)
        s.self_energies = base
        col = s.rhoP.sum(axis=1) * s.dz
        print(f"[CORR c={cscale:.2f}] iters={s.n_iter} err={s.err:.2e} "
              f"lat_contrast={(col.max()-col.min())/col.mean():.3f}",
              flush=True)
    np.savez(args.out, x=s.x, z=s.z, rhoP=s.rhoP, psi=s.psi, W=s.W,
             err=s.err, sigma=args.sigma)
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)
