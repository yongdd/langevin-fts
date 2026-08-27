"""Reproduction of Duan, Agrawal & Wang, PRL 134, 048101 (2025):
electrostatic-correlation-augmented SCFT for polyelectrolyte brushes.

Direct implementation of THEIR deterministic theory (paper + SI), in
physical units, 1D in z — independent of this codebase's FTS machinery.

Model (SI Secs. 1.2, II, III):
- Discrete Gaussian chain, N beads, bond length b; z-resolved propagators
  with the 1D bond kernel (variance b^2/3). Grafted at the substrate.
- Theta solvent (chi = 0.5 per monomer pair, v_P = v_S = v), ions with
  zero excluded volume; incompressibility rho_P v + rho_S v = 1 enforced
  by the pressure field xi.
- Ions analytic (Boltzmann). Two ensembles:
  * counterions=False (default): ALL ions grand-canonical against the
    bulk z:1 salt reservoir (multivalent cations double as counterions;
    == the fully-ion-exchanged limit, their S34 regime). I_b > 0
    required (pure-Neumann PB is singular without a reservoir).
  * counterions=True: their SEMICANONICAL ensemble — an additional
    CANONICAL monovalent counterion cloud (n_C = |zP| sigma N per
    area), needed for the Fig.-2 ion-exchange physics. CAVEAT: the
    canonical cloud has a uniform far-field tail, so its brush
    retention depends on the box length L (their finite-box numerics
    share this); keep L fixed when comparing onsets.
- Electrostatics: nonlinear Poisson for psi (in kT/e units),
  psi'' = -4 pi l_B sum_K z_K rho_K, Neumann at the wall, psi -> 0 in
  the bulk. Uniform dielectric (l_B = 0.7 nm for water/80).
- CORRELATION: Gaussian-fluctuation self-energy in the local-density
  approximation (their analytic short-range Green function G_s, which
  they identify as the dominant piece; the long-range correction G_l is
  a second-stage refinement):
      u_K(z) = z_K^2 l_B/(2 a_K) * [u(a_K kappa(z)) - u(a_K kappa_b)],
      u(x)   = 1 - x exp(x^2/pi) erfc(x/sqrt(pi)),
      kappa(z)^2 = 8 pi l_B I0(z),  I0 = (1/2) sum_K z_K^2 rho_K
  (polymer segments included in I0, per their local ionic strength).
  Mean-field mode: u_K = 0 (their dashed curves).
- The polymer's per-monomer field: V_P = chi(1-phi_P) + xi + z_P psi + u_P
  with xi eliminated EXACTLY (volumeless ions): xi = -ln(1-phi_P) - chi phi_P.
- Production algorithm (v6+): real-space hard-wall chain backend
  (chain_backend="realspace", the default — the spectral repo solver is
  kept for weak fields only, see RESULTS_PRL_SCFT.md 3c), exact-PB psi
  each iteration (psi_exact=True), plain slow mixing on W with a trust
  region, alpha- and correlation-strength continuation.

Observables: rho_P(z) profiles (Fig 1), Gibbs-dividing-surface height h
(Fig 2), effective charge Gamma and surface potential psi_S (Fig 3).
"""
import numpy as np
from scipy.fft import dct, idct
from scipy.special import erfc


class PRLBrushSCFT:
    """PRL-2025 correlation-augmented SCFT for a PE brush (1D).

    Chain sector (default): real-space banded bond convolution with a
    hard wall (`_chain_density_rs`; `chain_backend="spectral"` switches
    to the repo PropagatorSolver, valid only for per-chain field
    contrasts < ~37 kT, see RESULTS_PRL_SCFT.md 3c).
    Field update (production): plain slow mixing with trust region and
    exact-PB psi (`solve(..., psi_exact=True)`); `solve_lm` et al. are
    Newton-type alternatives kept for the record.
    Electrostatics: guarded-Newton nonlinear PB (analytic Boltzmann
    ions; GC salt + optional canonical monovalent counterions, see
    module docstring). Production field update: AM-first with
    simple-mixing fallback (see dh_salt_runs drivers).
    Correlations: LDA self-energies u_K = z^2 lB/(2a) [u(a kappa)-u(a kappa_b)]
    with kappa built from the Born-radius-smoothed local ionic strength.
    """

    def __init__(self, N=100, b=1.0, v=1.0, zP=-1.0, zplus=3, sigma=0.1,
                 Ib_molar=0.3, aP=0.25, aion=0.25, lB=0.7, chi=0.5,
                 L=100.0, Nz=800, correlations=True, aplus=None,
                 counterions=False):
        from polymerfts.propagator_solver import PropagatorSolver
        self.N, self.b, self.v = N, b, v
        self.zP, self.zp = zP, float(zplus)
        self.sigma = sigma
        self.aP, self.aion, self.lB, self.chi = aP, aion, lB, chi
        # their Fig. 4(c,d) reduces only the CATION Born radius
        self.aplus = aplus if aplus is not None else aion
        # canonical monovalent counterions (their semicanonical ensemble)
        self.with_counterions = counterions
        self.L, self.Nz = L, Nz
        self.dz = L / Nz
        self.z = (np.arange(Nz) + 0.5) * self.dz
        self.corr = correlations

        Ib = Ib_molar * 0.6022
        self.rho_b = 2.0 * Ib / (self.zp * (self.zp + 1.0)) if Ib > 0 else 0.0
        self.kappa_b = np.sqrt(8.0 * np.pi * lB * Ib) if Ib > 0 else 0.0

        # repo propagator solver (R0 units)
        self.R0 = b * np.sqrt(N)
        self.ps = PropagatorSolver(
            nx=[Nz], lx=[L / self.R0], ds=1.0 / N,
            bond_lengths={"P": 1.0}, bc=["reflecting", "reflecting"],
            chain_model="discrete", numerical_method="rqm4",
            platform="cpu-mkl", reduce_memory=False)
        self.ps.add_polymer(1.0, [["P", 1.0, 0, 1]], grafting_points={0: "G"})
        self.ps._initialize_solver()
        # graft source: Gaussian of width 0.5 b centered AT the hard wall
        # (z=0) -- the wall clips it to a half-Gaussian hugging the plate,
        # i.e. the smooth stable analog of the paper's delta(z) source.
        # A point source is numerically unstable (keep it broadened), but
        # any center displaced INTO the domain puts the tethered bead-1
        # density peak inside the depletion zone and shows up as a small
        # interior bump on the dilute MF plateau (0.5b: slight, b: +0.07).
        src = np.exp(-0.5 * (self.z / (0.5 * b)) ** 2)
        src /= np.trapz(src, dx=self.dz / self.R0)
        self.q_init = {"G": np.ascontiguousarray(src)}

        # repo Anderson mixing on the per-chain field W (Nz variables)
        self.am = self.ps.create_anderson_mixing(
            Nz, 20, 1e-1, 0.02, 0.02)

        self.psi = np.zeros(Nz)
        self.uP = np.zeros(Nz)
        self.up = np.zeros(Nz)
        self.um = np.zeros(Nz)
        self.W = np.zeros(Nz)          # per-chain field N * V_P

    def set_bulk(self, rho_b):
        """Re-point the bulk reservoir (salt continuation for their Fig. 2/3
        sweeps). rho_b = bulk MULTIVALENT-cation number density in nm^-3
        (their x-axis variable); anion bulk density is z+ * rho_b."""
        self.rho_b = float(rho_b)
        Ib = 0.5 * (self.zp * (self.zp + 1.0)) * self.rho_b
        self.kappa_b = np.sqrt(8.0 * np.pi * self.lB * Ib) if Ib > 0 else 0.0

    # ---------------- self-energy (LDA, their Eq. S33) ----------------
    @staticmethod
    def _u_fn(x):
        return 1.0 - x * np.exp(x * x / np.pi) * erfc(x / np.sqrt(np.pi))

    def _self_energies(self, rhoP, rhoC, rhop, rhom):
        if not self.corr:
            zz = np.zeros(self.Nz)
            return zz, zz, zz
        # include_polymer_I0=False drops the polymer charge from the LDA
        # screening (crude bound on their nonlocal chain-connectivity
        # correction Iex, SI Eq. S25: connected charges do not screen
        # like free ions; LDA-with-P over-screens, excluding-P
        # under-screens -- the two bracket the full theory)
        wP = 1.0 if getattr(self, "include_polymer_I0", True) else 0.0
        # rhoC = MONOVALENT canonical counterions (z_C = +1)
        I0 = 0.5 * (wP * self.zP ** 2 * rhoP
                    + self.zp ** 2 * rhop + rhoC + rhom)
        # UV regularization of the LDA: an ion samples the ionic strength
        # within its Gaussian charge spread (Born radius a), so smooth I0
        # on that scale. Without this the pointwise LDA feedback loop
        # phi -> I0 -> u -> phi has no wavelength cutoff and, past the
        # z+=3 full-coupling threshold, blows up at the grid scale
        # (2*dz checkerboard). The paper's nonlocal G_s carries this
        # regularization intrinsically; smooth profiles are unaffected.
        from scipy.ndimage import gaussian_filter1d
        a_min = min(self.aion, self.aplus)
        I0 = gaussian_filter1d(I0, sigma=a_min / self.dz, mode="reflect")
        kap = np.sqrt(np.maximum(8.0 * np.pi * self.lB * I0, 0.0))
        du = lambda a, zq: (zq ** 2) * self.lB / (2.0 * a) * \
            (self._u_fn(a * kap) - self._u_fn(a * self.kappa_b))
        return du(self.aP, self.zP), du(self.aplus, self.zp), du(self.aion, 1.0)

    # ---------------- chain density ----------------
    # Two backends:
    #  - repo PropagatorSolver (spectral/DCT): exact but poisoned by
    #    float leakage amplification when the per-chain field CONTRAST
    #    exceeds ~ln(1e16) ~ 37 kT: the ~1e-16 DCT ringing floor seeds the
    #    low-field exterior and grows by e^{+dw} per bead, ending as O(1)
    #    garbage (isolated repro: step field 100->50 at z=60 gives
    #    phi(far wall)=23 and phi<0). Our brush crosses that threshold at
    #    alpha ~ 0.5 -- the entire high-alpha "instability" was this.
    #  - real-space banded bond convolution (truncated Gaussian kernel,
    #    'reflect' boundary = the same cell-centered mirror as DCT-II):
    #    transport is local, no spectral floor, values beyond reach decay
    #    to true exponential smallness. Used by default.
    def _chain_density(self, VP_per_monomer):
        if getattr(self, "chain_backend", "realspace") == "spectral":
            W = np.ascontiguousarray(self.N * VP_per_monomer)
            self.ps.compute_propagators({"P": W}, q_init=self.q_init)
            self.ps.compute_concentrations()
            phi = np.array(self.ps.get_concentration("P"), dtype=float)
        else:
            phi = self._chain_density_rs(VP_per_monomer)
        phi = np.maximum(phi, 0.0)
        norm = np.trapz(phi, dx=self.dz)          # in nm
        rho = phi * (self.sigma * self.N / max(norm, 1e-280))
        return rho

    def _bond_kernel(self):
        if not hasattr(self, "_bk"):
            half = int(np.ceil(8.0 * self.b / self.dz))   # 8b: tail e^-96
            d = np.arange(-half, half + 1) * self.dz
            k = np.exp(-1.5 * (d / self.b) ** 2)
            self._bk = k / k.sum()
        return self._bk

    def _bond_conv(self, q):
        """Bond convolution with a HARD WALL at z=0: the integral over
        bond vectors is truncated at the wall (zero ghost cells), which
        is the exact discrete-chain impenetrable-wall rule and produces
        the ~1-2 b entropic depletion layer of their Fig. 1 (a 'reflect'
        mirror has no such entropy loss -- wall density was wrongly
        maximal). Far wall: even mirror (nothing reaches it)."""
        k = self._bond_kernel()
        H = (len(k) - 1) // 2
        qp = np.concatenate([np.zeros(H), q, q[-1:-H - 1:-1]])
        return np.convolve(qp, k, mode="valid")

    def _chain_density_rs(self, VP):
        """Discrete-chain (N beads, N-1 Gaussian bonds) density by
        real-space banded convolution. Only the profile SHAPE matters
        (global normalization to sigma*N happens in _chain_density), so
        per-step max renormalization is safe."""
        ew = np.exp(-np.clip(VP, -300.0, 300.0))
        src = self.q_init["G"]
        N = self.N
        qf = np.empty((N, self.Nz))
        q = ew * src
        q /= max(q.max(), 1e-300)
        qf[0] = q
        for n in range(1, N):
            q = ew * self._bond_conv(q)
            q /= max(q.max(), 1e-300)
            qf[n] = q
        qb = np.empty((N, self.Nz))
        q = ew.copy()
        q /= max(q.max(), 1e-300)
        qb[0] = q
        for n in range(1, N):
            q = ew * self._bond_conv(q)
            q /= max(q.max(), 1e-300)
            qb[n] = q
        # phi_n = qf[n] * qb[N-1-n] / ew  (one e^-w per bead); rescale each
        # bead slice to equal mass: every bead contributes the same number
        # of monomers for a grafted brush.
        phi = np.zeros(self.Nz)
        iew = 1.0 / np.maximum(ew, 1e-300)
        for n in range(N):
            sl = qf[n] * qb[N - 1 - n] * iew
            m = np.trapz(sl, dx=self.dz)
            phi += sl / max(m, 1e-300)
        return phi

    # ---------------- nonlinear Poisson-Boltzmann (guarded Newton) ----------------
    def _solve_pb(self, rhoP, nC_total=None, n_newton=200):
        """Given rho_P and the self-energy fields, solve
        psi'' = -4 pi lB [zP rhoP + rhoC(psi) + zp rhop(psi) - rhom(psi)]
        with Neumann walls, GUARDED Newton: backtracking line search on the
        residual 2-norm + a residual-based convergence test. (An
        iteration-capped unconverged Newton output is chaotically
        input-sensitive and poisoned every outer solver; PB is convex, so
        guarded Newton converges globally.)

        rhoC: CANONICAL MONOVALENT counterions (their semicanonical
        ensemble, SI: n_C fixed, z_C=+1) with total per-area
        |zP| sigma N, enabled by self.with_counterions. Their self-energy
        equals um (same valence and radius as the anion). At rho_b >>
        alpha rho_P they are displaced by the multivalent cations (their
        S34 regime, where the GC-only model used before is equivalent);
        at low salt they neutralize the brush with only z^2=1
        correlations -- this ion-exchange threshold IS the Fig.-2
        collapse onset rho*_b."""
        from scipy.linalg import solve_banded
        with_C = getattr(self, "with_counterions", False)
        nC_total = (abs(self.zP) * self.sigma * self.N) if with_C else 0.0
        if self.rho_b <= 0.0 and not with_C:
            raise ValueError("PRLBrushSCFT requires I_b > 0 (GC salt "
                             "reservoir); salt-free PB is singular here.")
        dz2 = self.dz ** 2
        psi = self.psi.copy()

        def residual(p):
            bp = np.exp(-np.clip(self.zp * p + self.up, -300, 300))
            bm = np.exp(-np.clip(-p + self.um, -300, 300))
            rhop = self.rho_b * bp
            rhom = self.zp * self.rho_b * bm
            if nC_total > 0.0:
                bC = np.exp(-np.clip(p + self.um, -300, 300))
                rhoC = nC_total * bC / max(np.trapz(bC, dx=self.dz), 1e-280)
            else:
                rhoC = 0.0
            rho_e = self.zP * rhoP + rhoC + self.zp * rhop - rhom
            lap = np.empty_like(p)
            lap[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / dz2
            lap[0] = (p[1] - p[0]) / dz2
            lap[-1] = (p[-2] - p[-1]) / dz2
            return lap + 4.0 * np.pi * self.lB * rho_e, rhoC, rhop, rhom

        F, rhoC, rhop, rhom = residual(psi)
        fn = np.linalg.norm(F)
        f_tol = 1e-9 * max(1.0, 4.0 * np.pi * self.lB
                           * float(np.abs(rhoP).max() + self.zp * self.rho_b
                                   + (nC_total / self.L)))
        for _ in range(n_newton):
            if np.abs(F).max() < f_tol:
                break
            # local part of the Jacobian (the canonical-normalization
            # rank-1 term is omitted; the line search absorbs it)
            diagm = 4.0 * np.pi * self.lB * (self.zp ** 2 * rhop + rhom
                                             + rhoC)
            ab = np.zeros((3, self.Nz))
            ab[0, 1:] = 1.0 / dz2
            ab[2, :-1] = 1.0 / dz2
            ab[1, :] = -2.0 / dz2 - diagm
            ab[1, 0] = -1.0 / dz2 - diagm[0]
            ab[1, -1] = -1.0 / dz2 - diagm[-1]
            dpsi = solve_banded((1, 1), ab, -F)
            t = 1.0
            while True:
                F_try, rhoC_t, rhop_t, rhom_t = residual(psi + t * dpsi)
                fn_try = np.linalg.norm(F_try)
                if fn_try < fn or t < 1e-8:
                    psi = psi + t * dpsi
                    F, rhoC, rhop, rhom, fn = F_try, rhoC_t, rhop_t, rhom_t, fn_try
                    break
                t *= 0.5
        if not with_C:
            psi -= psi[-1]        # bulk gauge (with canonical C the far
                                  # field fixes the gauge through the salt)
        self.psi = psi
        _F, rhoC, rhop, rhom = residual(psi)
        if np.isscalar(rhoC):
            rhoC = np.zeros(self.Nz)
        return rhoC, rhop, rhom

    # ---------------- main SCF loop ----------------
    def solve(self, max_iter=3000, tol=1e-7, lam_u=0.02, lam_psi=0.02,
              lam_w=0.02, use_am=False, verbose=False, W_init=None,
              dW_cap=0.5, psi_exact=False):
        if W_init is not None:
            self.W = W_init.copy()
        elif not np.any(self.W):
            # physical initialization: a decaying brush-like field (the
            # wild zero-field transient seeds far-field wells that the
            # phi-clip kick then sustains as a limit cycle)
            self.W = self.N * (0.3 * np.exp(-self.z / 20.0))
        rhoP = self._chain_density(self.W / self.N)
        self.am.reset_count()
        err = old_err = 1e9
        anneal_next = 0.15
        # annealing floor must never RAISE a deliberately small lambda
        lam_floor = min(2e-3, lam_w, lam_psi, lam_u)
        for it in range(max_iter):
            # PB solved exactly, then UNDER-RELAXED into the state (the
            # paper mixes psi with the same simple-mixing rule as omega;
            # jumping to the exact PB solution each iteration couples the
            # soft Theta chain modes to instant electrostatic response and
            # limit-cycles)
            psi_prev = self.psi.copy()
            rhoC, rhop, rhom = self._solve_pb(rhoP, 0.0)
            if not psi_exact:
                # under-relax psi into the state (paper's mixing scheme);
                # psi_exact=True keeps the exact PB solution -- the right
                # mode when W is Anderson-mixed at mean field (psi is then
                # a pure function of rhoP; a slowly drifting psi corrupts
                # the AM secant history)
                dpsi = self.psi - psi_prev
                cap = 0.5
                mx = np.abs(dpsi).max()
                if mx * lam_psi > cap:
                    dpsi *= cap / (mx * lam_psi)
                self.psi = psi_prev + lam_psi * dpsi
            if not psi_exact:
                # re-evaluate densities at the relaxed psi
                bp = np.exp(-np.clip(self.zp * self.psi + self.up, -300, 300))
                bm = np.exp(-np.clip(-self.psi + self.um, -300, 300))
                rhop = self.rho_b * bp
                rhom = self.zp * self.rho_b * bm
                if getattr(self, "with_counterions", False):
                    bC = np.exp(-np.clip(self.psi + self.um, -300, 300))
                    nC = abs(self.zP) * self.sigma * self.N
                    rhoC = nC * bC / max(np.trapz(bC, dx=self.dz), 1e-280)
                else:
                    rhoC = np.zeros(self.Nz)

            phiP = np.clip(self.v * rhoP, 0.0, 0.999)
            xi = -np.log(1.0 - phiP) - self.chi * phiP
            VP_new = self.chi * (1.0 - phiP) + xi \
                + self.zP * self.psi + self.uP
            W_new = self.N * VP_new

            old_err = err
            err = float(np.abs(W_new - self.W).max()) / self.N
            # lambda annealing: shrink all mixing rates as the orbit tightens
            if err < anneal_next:
                lam_w = max(lam_w * 0.5, lam_floor)
                lam_psi = max(lam_psi * 0.5, lam_floor)
                lam_u = max(lam_u * 0.5, lam_floor)
                anneal_next *= 0.5
            if verbose and it % 100 == 0:
                print(f"  it {it:5d}: err={err:.3e} h={self.height(rhoP):.2f}")
            if err < tol and it > 10:
                break
            if use_am:
                self.W = np.reshape(self.am.calculate_new_fields(
                    self.W, W_new - self.W, old_err, err), self.Nz)
            else:
                dW = lam_w * (W_new - self.W)
                mx = np.abs(dW).max()
                if mx > dW_cap:
                    dW *= dW_cap / mx
                self.W = self.W + dW

            rhoP = self._chain_density(self.W / self.N)

            uP_new, up_new, um_new = self._self_energies(rhoP, rhoC, rhop, rhom)
            self.uP = (1 - lam_u) * self.uP + lam_u * uP_new
            self.up = (1 - lam_u) * self.up + lam_u * up_new
            self.um = (1 - lam_u) * self.um + lam_u * um_new

        self.rhoP, self.rhoC, self.rhop, self.rhom = rhoP, rhoC, rhop, rhom
        self.n_iter, self.err = it + 1, err
        return rhoP

    # ---------------- Newton-Krylov outer solver ----------------
    def _solve_ion_sector(self, rhoP, nmax=500, tol=1e-10, m_hist=5):
        """Fully equilibrate (psi, rho_ion, u) at fixed rhoP, by Anderson
        acceleration on the stacked u = (uP, up, um) fixed point (psi is
        solved exactly by guarded-Newton PB inside each evaluation).
        Plain lam-mixing is marginal/slow near ion-condensation saturation
        (z+=3 collapsed states) and an UNCONVERGED inner loop makes the
        outer W-map effectively discontinuous -- the same disease as the
        spectral leakage, one level down. AA with a plain-mixing safeguard
        converges in O(10) evaluations."""
        def gmap(u):
            self.uP, self.up, self.um = u[0], u[1], u[2]
            rhoC, rhop, rhom = self._solve_pb(rhoP, 0.0)
            uP, up, um = self._self_energies(rhoP, rhoC, rhop, rhom)
            return np.stack([uP, up, um])

        u = np.stack([self.uP, self.up, self.um])
        g = gmap(u) - u
        gn = np.abs(g).max()
        hist_u, hist_g = [], []
        for it in range(nmax):
            if gn < tol:
                break
            hist_u.append(u.copy())
            hist_g.append(g.copy())
            if len(hist_u) > m_hist + 1:
                hist_u.pop(0)
                hist_g.pop(0)
            if len(hist_u) >= 2:
                dG = np.stack([(hist_g[i + 1] - hist_g[i]).ravel()
                               for i in range(len(hist_g) - 1)], axis=1)
                dU = np.stack([(hist_u[i + 1] - hist_u[i]).ravel()
                               for i in range(len(hist_u) - 1)], axis=1)
                try:
                    gamma, *_ = np.linalg.lstsq(dG, g.ravel(), rcond=None)
                    u_new = u - ((dU + dG) @ gamma).reshape(u.shape) + g
                except np.linalg.LinAlgError:
                    u_new = u + 0.5 * g
            else:
                u_new = u + 0.5 * g
            g_new = gmap(u_new) - u_new
            if np.abs(g_new).max() < 2.0 * gn or len(hist_u) < 2:
                u, g = u_new, g_new
            else:                      # AA step went wild: damped fallback
                u = u + 0.05 * g
                g = gmap(u) - u
                hist_u, hist_g = [], []
            gn = np.abs(g).max()
        self.uP, self.up, self.um = u[0], u[1], u[2]
        self._ion_converged = bool(gn < 1e-6)
        return self._solve_pb(rhoP, 0.0)

    def _w_map(self, W):
        """Full self-consistent map W -> W_new (exact ion sector inside).
        The fixed-point Jacobian has a real eigenvalue ~ +N^2 f^2
        (4 pi lB / kappa^2) rhoP = O(10^2-10^3) (power-iteration measured
        mu ~ +300-600 at alpha=0.5): NO fixed-point mixing can converge.
        Use solve_newton()."""
        rhoP = self._chain_density(W / self.N)
        self._solve_ion_sector(rhoP)
        phiP = np.clip(self.v * rhoP, 0.0, 0.999)
        xi = -np.log(1.0 - phiP) - self.chi * phiP
        VP = self.chi * (1.0 - phiP) + xi + self.zP * self.psi + self.uP
        self._last_rhoP = rhoP
        return self.N * VP

    def solve_newton(self, W_init=None, f_tol=5e-4, maxiter=60,
                     inner_maxiter=40, verbose=False):
        """Newton-Krylov on R(W) = w_map(W) - W. Converges regardless of
        the huge real fixed-point eigenvalue. f_tol is on max|R|/N to match
        the mixing loop's err convention."""
        from scipy.optimize import newton_krylov, NoConvergence
        if W_init is not None:
            self.W = W_init.copy()
        it = [0]

        def R(W):
            r = self._w_map(W) - W
            it[0] += 1
            if verbose and it[0] % 10 == 0:
                print(f"    nk eval {it[0]}: |R|max/N = "
                      f"{np.abs(r).max()/self.N:.3e}", flush=True)
            return r

        try:
            Wsol = newton_krylov(R, self.W, method="lgmres",
                                 f_tol=f_tol * self.N, maxiter=maxiter,
                                 inner_maxiter=inner_maxiter)
            self.W = np.asarray(Wsol, dtype=float)
        except NoConvergence as e:
            self.W = np.asarray(e.args[0], dtype=float)
        r = self._w_map(self.W) - self.W
        self.rhoP = self._last_rhoP
        bp = np.exp(-np.clip(self.zp * self.psi + self.up, -300, 300))
        bm = np.exp(-np.clip(-self.psi + self.um, -300, 300))
        if getattr(self, "with_counterions", False):
            bC = np.exp(-np.clip(self.psi + self.um, -300, 300))
            nC = abs(self.zP) * self.sigma * self.N
            self.rhoC = nC * bC / max(np.trapz(bC, dx=self.dz), 1e-280)
        else:
            self.rhoC = np.zeros(self.Nz)
        self.rhop = self.rho_b * bp
        self.rhom = self.zp * self.rho_b * bm
        self.n_iter, self.err = it[0], float(np.abs(r).max() / self.N)
        return self.rhoP

    def solve_newton_dense(self, W_init=None, f_tol=5e-4, max_newton=40,
                           fd_eps=1e-4, jac_refresh=3, verbose=False):
        """Damped Newton with a full finite-difference Jacobian.
        Map evaluations cost ~3-10 ms in 1D, so a dense FD Jacobian
        (Nz evals) is affordable and far more robust than Krylov line
        searches against the mu ~ +500 Donnan eigenvalue. The Jacobian is
        reused for up to jac_refresh Newton steps (Shamanskii scheme)."""
        if W_init is not None:
            self.W = W_init.copy()
        n = self.Nz
        evals = [0]

        def R(W):
            evals[0] += 1
            return self._w_map(W) - W

        r = R(self.W)
        J = None
        age = jac_refresh  # force build on first step
        for k in range(max_newton):
            rn = np.abs(r).max() / self.N
            if verbose:
                print(f"    newton {k}: |R|max/N={rn:.3e}", flush=True)
            if rn < f_tol:
                break
            if age >= jac_refresh:
                J = np.empty((n, n))
                eps = fd_eps * self.N
                for j in range(n):
                    Wp = self.W.copy()
                    Wp[j] += eps
                    J[:, j] = (R(Wp) - r) / eps
                age = 0
            dW = np.linalg.solve(J, -r)
            t = 1.0
            accepted = False
            for _ in range(12):
                r_try = R(self.W + t * dW)
                if np.abs(r_try).max() < np.abs(r).max():
                    self.W = self.W + t * dW
                    r = r_try
                    accepted = True
                    break
                t *= 0.5
            age += 1
            if not accepted:
                if age <= 1:
                    # fresh Jacobian and still no descent: bail out
                    break
                age = jac_refresh  # stale Jacobian: rebuild and retry
        self.rhoP = self._last_rhoP
        bp = np.exp(-np.clip(self.zp * self.psi + self.up, -300, 300))
        bm = np.exp(-np.clip(-self.psi + self.um, -300, 300))
        if getattr(self, "with_counterions", False):
            bC = np.exp(-np.clip(self.psi + self.um, -300, 300))
            nC = abs(self.zP) * self.sigma * self.N
            self.rhoC = nC * bC / max(np.trapz(bC, dx=self.dz), 1e-280)
        else:
            self.rhoC = np.zeros(self.Nz)
        self.rhop = self.rho_b * bp
        self.rhom = self.zp * self.rho_b * bm
        self.n_iter = evals[0]
        self.err = float(np.abs(r).max() / self.N)
        return self.rhoP

    def solve_lm(self, W_init=None, f_tol=5e-4, max_outer=60, fd_eps=1e-3,
                 jac_refresh=4, lam0=1e-2, verbose=False):
        """Levenberg-Marquardt on R(W) = w_map(W) - W with a dense FD
        Jacobian. Plain damped Newton fails here: the map's gain spectrum
        crosses +1 continuously, so J = R' has near-singular directions
        along which the Newton step is enormous and never enters the
        linearization region even at t = 2^-12. LM damping
        (J^T J + lam^2 diag) guarantees a descent direction for |R|^2 at
        large lam and approaches Newton as lam -> 0."""
        if W_init is not None:
            self.W = W_init.copy()
        n = self.Nz
        evals = [0]

        def R(W):
            evals[0] += 1
            return self._w_map(W) - W

        r = R(self.W)
        lam = lam0
        J = None
        age = jac_refresh
        for k in range(max_outer):
            rn = np.abs(r).max() / self.N
            if rn < f_tol:
                break
            if age >= jac_refresh:
                J = np.empty((n, n))
                eps = fd_eps * self.N
                for j in range(n):
                    Wp = self.W.copy()
                    Wp[j] += eps
                    J[:, j] = (R(Wp) - r) / eps
                JTJ = J.T @ J
                dscale = np.sqrt(np.maximum(np.diag(JTJ), 1e-12))
                age = 0
            g = J.T @ r
            accepted = False
            for _ in range(25):
                A = JTJ + (lam ** 2) * np.diag(dscale ** 2)
                try:
                    dW = np.linalg.solve(A, -g)
                except np.linalg.LinAlgError:
                    lam *= 3.0
                    continue
                r_try = R(self.W + dW)
                if np.dot(r_try, r_try) < np.dot(r, r):
                    self.W = self.W + dW
                    r = r_try
                    lam = max(lam / 3.0, 1e-8)
                    accepted = True
                    break
                lam *= 3.0
            age += 1
            if verbose:
                print(f"    lm {k}: |R|max/N={np.abs(r).max()/self.N:.3e} "
                      f"lam={lam:.2e} acc={accepted}", flush=True)
            if not accepted:
                if age <= 1:
                    break       # fresh Jacobian, no descent at any lam
                age = jac_refresh
        # re-evaluate at the ACCEPTED W: the last _w_map call may have
        # been a rejected trial or an FD probe, leaving psi/u/_last_rhoP
        # inconsistent with self.W (observables would silently mix states)
        r = R(self.W)
        self.rhoP = self._last_rhoP
        bp = np.exp(-np.clip(self.zp * self.psi + self.up, -300, 300))
        bm = np.exp(-np.clip(-self.psi + self.um, -300, 300))
        if getattr(self, "with_counterions", False):
            bC = np.exp(-np.clip(self.psi + self.um, -300, 300))
            nC = abs(self.zP) * self.sigma * self.N
            self.rhoC = nC * bC / max(np.trapz(bC, dx=self.dz), 1e-280)
        else:
            self.rhoC = np.zeros(self.Nz)
        self.rhop = self.rho_b * bp
        self.rhom = self.zp * self.rho_b * bm
        self.n_iter = evals[0]
        self.err = float(np.abs(r).max() / self.N)
        return self.rhoP

    # ---------------- observables ----------------
    def height(self, rhoP=None):
        """Gibbs dividing surface: h = int z rho / int rho * 2? — the GDS of
        a step profile equals first-moment*2 for a box; use the standard
        h = 2 * int z rho dz / int rho dz? No: GDS h satisfies
        int_0^inf rho dz = rho(0+) * h for a step; for general profiles the
        common choice (their Ref [52]) is h = 2 <z>."""
        if rhoP is None:
            rhoP = self.rhoP
        m0 = np.trapz(rhoP, dx=self.dz)
        m1 = np.trapz(self.z * rhoP, dx=self.dz)
        return 2.0 * m1 / max(m0, 1e-280)

    def gamma_eff(self, core_frac=0.5):
        """THEIR exact observable (paper text below Eq. 7):
        Gamma = 1 - z+ (rho+ - rho_b) / rho_P, the effective relative charge
        on the brush after cation adsorption -- a LOCAL quantity evaluated in
        the brush interior. We report the density-weighted average over the
        core region phi_P > core_frac * max(phi_P). (The old integral
        'excess/bare' estimator diverged for collapsed layers; the net-charge
        (Gauss) version is ~0 by screening -- neither is their Gamma.)"""
        bp = np.exp(-np.clip(self.zp * self.psi + self.up, -300, 300))
        rhop = self.rho_b * bp
        phi = self.v * self.rhoP
        mask = phi > core_frac * phi.max()
        g = 1.0 - self.zp * (rhop[mask] - self.rho_b) \
            / np.maximum(self.rhoP[mask], 1e-280)
        wgt = self.rhoP[mask]
        return float(np.sum(g * wgt) / np.sum(wgt))

    def gamma_algebraic(self):
        """Their Eq. (7)/(S38): the asymptotic (|drho| << rho_b) algebraic
        Gamma as a function of bulk salt only -- this is what their Fig. 3
        curves plot. ((z+1)G - 1)/(G - 1) = -z(z+1) (kb lB/4) u'(a kb)."""
        x = self.aion * self.kappa_b
        g = np.exp(x * x / np.pi) * erfc(x / np.sqrt(np.pi))
        uprime = -g * (1.0 + 2.0 * x * x / np.pi) + 2.0 * x / np.pi
        R = -self.zp * (self.zp + 1.0) * (self.kappa_b * self.lB / 4.0) * uprime
        return (1.0 - R) / (self.zp + 1.0 - R)

    def phiP(self):
        return self.v * self.rhoP
