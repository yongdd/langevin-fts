"""Anderson-mixing SCFT (mean-field) solver for charged brushes in a wall box.

The deterministic saddle-point solver for the grafted-polyelectrolyte
system of the PRL-2025 reproduction plan (RESULTS_BRUSH_STATUS.md,
obstacle 2.4): naive drift relaxation converges the k>0 structure but
leaves a slowly-relaxing, residual-std-invisible mean (k=0) channel and
takes O(10^4) iterations; Anderson mixing solves the joint root problem
directly. Built on ChargedLFTS's validated find_saddle_point (AM on the
imaginary SPT fields + screening-preconditioned Newton for psi) with
three brush-specific changes:

1. Wall geometry: params["bc"] = ["reflecting"]*2dim flows through
   LFTS -> PropagatorSolver natively; the psi sector is swapped to
   ElectrostaticsReflecting (DCT-II basis, Neumann walls).
2. Grafted chains: the propagator solver is rebuilt so polymers with a
   per-polymer "grafting_points" entry ({vertex: label}) register them;
   a uniform delta sheet at the z=0 wall cell (amplitude 1/dz) is passed
   as q_init on every propagator solve (laterally annealed grafting).
3. psi Newton preconditioner: S_scr uses the CHAIN-COHERENT response
   z_i^2 phibar_i N_i hhat^2 per species (N_i = number of segments:
   the low-k density response of an N-segment chain is ~N x the
   single-segment one; the base class's segment-count-blind S_scr
   underdamps the polymer channel by ~N and limit-cycles — obstacle
   2.5). Global electroneutrality makes kappa^2 here approximate, which
   only affects the convergence RATE, not the fixed point.

The solver is used through `solve(E_targets)`: a continuation ladder in
the Coulomb coupling E (the double layer deepens adiabatically; jumping
straight to large E throws the grafted chain into e^{-large} partition
territory before ions screen — obstacle ladder 2.2/2.3).

Mean-field only (this is SCFT: psi at its Poisson saddle, no
fluctuations) — the CL warm-start / PRL-dashed-line reference. The
converged (w_aux, psi) pair is returned for handoff to BrushChargedCLFTS.
"""
import numpy as np

from polymerfts.propagator_solver import PropagatorSolver
from polymerfts.validation import ValidationError

from lfts_charged import ChargedLFTS, _PerSpeciesSmearing
from electrostatics import ElectrostaticsReflecting


class BrushChargedSCFT(ChargedLFTS):
    """Deterministic charged-brush SCFT: AM saddle in an all-reflecting box."""

    def __init__(self, params, random_seed=None):
        dim = len(params["nx"])
        bc = params.get("bc", ["reflecting"] * (2 * dim))
        if any(b != "reflecting" for b in bc):
            raise ValidationError(
                "BrushChargedSCFT supports all-'reflecting' boxes only "
                f"(got {bc}).")
        params = dict(params)
        params["bc"] = bc
        platform = params.get("platform", "cpu-mkl")
        if platform not in ("cpu-mkl", "cpu-fftw"):
            raise ValidationError(
                "BrushChargedSCFT requires a CPU platform (CUDA "
                "non-periodic real-transform path is untested here).")

        super().__init__(params=params, random_seed=random_seed)

        # ---- rebuild the propagator solver with grafting points ----
        ps = PropagatorSolver(
            nx=params["nx"], lx=params["lx"], ds=params["ds"],
            bond_lengths=self.segment_lengths, bc=bc,
            chain_model=params["chain_model"],
            numerical_method=params.get("numerical_method", "rqm4"),
            platform=platform,
            reduce_memory=params.get("reduce_memory", False))
        graft_labels = []
        for polymer in self.distinct_polymers:
            gp = polymer.get("grafting_points", None)
            if gp:
                gp = {int(k): str(v) for k, v in gp.items()}
                ps.add_polymer(polymer["volume_fraction"],
                               polymer["blocks_input"], grafting_points=gp)
                graft_labels.extend(gp.values())
            else:
                ps.add_polymer(polymer["volume_fraction"],
                               polymer["blocks_input"])
        ps._initialize_solver()
        self.prop_solver = ps
        self.cb = ps._computation_box

        # ---- graft source: wall-attached Gaussian sheet ----
        # A strict delta at the first cell forces sub-cell structure that
        # rings on the pseudo-spectral propagator (negative Q_P at strong
        # coupling). Smear the source over graft_width (default: the
        # smearing radius scale) — physically a finite anchor size;
        # profiles beyond the first cells are insensitive.
        nx = params["nx"]
        dz = params["lx"][-1] / nx[-1]
        width = float(params.get("graft_width", 0.2))
        zc = (np.arange(nx[-1]) + 0.5) * dz
        prof = np.exp(-0.5 * (zc / width) ** 2)
        prof /= prof.sum() * dz          # normalized: int q_init dz = 1
        sheet = np.zeros(nx, dtype=np.float64)
        sheet[...] = prof[(None,) * (len(nx) - 1)]
        sheet = np.ascontiguousarray(sheet.reshape(-1))
        self.q_init = {label: sheet for label in set(graft_labels)}

        # ---- wall-basis electrostatics with chain-coherent S_scr ----
        species_fractions = {t: 0.0 for t in self.monomer_types}
        seg_counts = {t: 0.0 for t in self.monomer_types}
        n_ref = int(round(1.0 / params["ds"]))
        for polymer in self.distinct_polymers:
            alpha = sum(blk["length"] for blk in polymer["blocks"])
            for blk in polymer["blocks"]:
                t = blk["type"]
                species_fractions[t] += \
                    polymer["volume_fraction"] * blk["length"] / alpha
                seg_counts[t] = max(seg_counts[t],
                                    blk["length"] * n_ref)
        self._species_fractions = species_fractions
        self._seg_counts = seg_counts
        e_target = self.electro.e_coupling
        self.electro = ElectrostaticsReflecting(
            self.cb.get_nx(), self.cb.get_lx(), self.monomer_types,
            params["charges"], params["radiuses"], e_target,
            species_fractions=species_fractions)
        self.smearing = _PerSpeciesSmearing(self.electro)
        self._rebuild_psi_jacobian()

        # gauge: pin the pressure-field mean after each saddle solve.
        # A zero target is the saddle value only when all chi_n vanish
        # (pure-pressure active sector); with chi != 0 the mean-mode
        # saddle is nonzero and pinning to 0 would bias H.
        if all(abs(v) < 1e-12 for v in self.chi_n.values()):
            self._target_mean_wp = 0.0

        # Adaptive per-species propagator-input shifts (obstacle 2.3).
        # Canonical densities are exactly invariant under W_t -> W_t + c_t
        # (per species), but ln Q_p shifts by -c_t alpha_p. The frozen
        # zero-eigenvalue modes pin the per-species chemical-potential
        # (k=0) freedom, so the converged gauge can park ln Q_P at ~ -50
        # (graft + wall compressibility structure), leaving no
        # double-precision headroom for the propagators — the E>~400
        # NaN wall. The shifts below are updated each iteration to keep
        # every ln Q_solver ~ 0; they are bookkeeping only (densities
        # untouched; the Hamiltonian is corrected in
        # _compute_hamiltonian_charged).
        for p in self.distinct_polymers:
            btypes = {b["type"] for b in p["blocks"]}
            if len(btypes) > 1:
                raise ValidationError(
                    "The per-species re-gauge bookkeeping (shift/Hamiltonian "
                    "correction) is only correct for polymers whose blocks "
                    f"share one monomer type; got {sorted(btypes)}. "
                    "Extend the shift algebra before using multi-type chains.")
        self._w_shift = {t: 0.0 for t in self.monomer_types}
        self._polymer_block_type = [
            p["blocks"][0]["type"] for p in self.distinct_polymers]
        self._polymer_alpha = [
            sum(b["length"] for b in p["blocks"])
            for p in self.distinct_polymers]

        print("Brush charged SCFT: bc = all-reflecting, "
              f"graft labels = {sorted(set(graft_labels))}, E = {e_target:g}")

    # ------------------------------------------------------------------ #
    def _rebuild_psi_jacobian(self):
        """Newton damping with chain-coherent response (per-chain units).

        Linear density response of an N_i-segment species to psi is
        ~ z_i^2 phibar_i * (N_i * ds) * (chain coherence ~ N_i at low k)
        = z_i^2 phibar_i N_i ds^2 ... in code units the empirically
        correct scale is z_i^2 phibar_i N_i ds = z_i^2 phibar_i
        (chain, N_i = 1/ds) and z_i^2 phibar_i ds (1-segment ion); both
        are covered by z_i^2 phibar_i N_i ds.
        """
        ds = 1.0 / max(self._seg_counts.values())
        scr = np.zeros_like(self.electro.k_sq)
        for t in self.monomer_types:
            z = self.electro.charges[t]
            if z == 0.0:
                continue
            h = self.electro.h_hat[t]
            h2 = 1.0 if h is None else h * h
            scr = scr + z * z * self._species_fractions[t] \
                * self._seg_counts[t] * ds * h2
        self.electro.jacobian = self.electro.k_sq \
            + self.electro.e_coupling * scr
        self.electro.jacobian[tuple([0] * self.electro.dim)] = 1.0

    # ------------------------------------------------------------------ #
    def compute_concentrations(self, w_aux):
        """ChargedLFTS propagator inputs + grafted q_init sources."""
        import time as _time
        M = len(self.monomer_types)
        elapsed_time = {}

        w = self.mpt.to_monomer_fields(w_aux)
        w_input = {self.monomer_types[i]: w[i] for i in range(M)}
        # snapshot the shifts THIS solve uses; the Hamiltonian correction
        # must use the same values (the adaptive update below applies only
        # from the next iteration).
        self._w_shift_used = dict(self._w_shift)
        w_input_for_propagator = {
            t: self.electro.smear(t, w_input[t] + self.electro.charge_potential(t))
            - self._w_shift_used[t]
            for t in w_input
        }

        t0 = _time.time()
        self.prop_solver.compute_propagators(
            w_input_for_propagator,
            q_init=self.q_init if self.q_init else None)
        elapsed_time["solver"] = _time.time() - t0

        # Re-gauge: nudge each species' shift so its ln Q_solver -> 0,
        # keeping the propagators centered in double-precision range.
        # (Subtracting s from W multiplies Q_solver by e^{+s alpha}, so
        # ln Q_solver = ln Q_true + s alpha; drive it to zero.)
        for p, t in enumerate(self._polymer_block_type):
            Q = self.prop_solver.get_partition_function(p)
            lnq = float(np.log(np.abs(Q)))
            if np.isfinite(lnq):
                self._w_shift[t] -= lnq / self._polymer_alpha[p]

        t0 = _time.time()
        phi = {}
        self.prop_solver.compute_concentrations()
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.prop_solver.get_concentration(monomer_type)
        elapsed_time["phi"] = _time.time() - t0
        return phi, elapsed_time

    # ------------------------------------------------------------------ #
    def _compute_hamiltonian_charged(self, w_aux, charge_c):
        """Parent Hamiltonian + undo the per-species re-gauging shifts.

        ln Q_true = ln Q_solver - s_t alpha_p, and H contains
        -sum_p (phibar_p/alpha_p) ln Q_p, so the correction is
        +sum_p vf_p * s_{t(p)}.
        """
        h = super()._compute_hamiltonian_charged(w_aux, charge_c)
        shifts = getattr(self, "_w_shift_used", self._w_shift)
        for p, t in enumerate(self._polymer_block_type):
            h += self.distinct_polymers[p]["volume_fraction"] * shifts[t]
        return h

    # ------------------------------------------------------------------ #
    def solve(self, e_targets, w_aux=None, verbose=True):
        """Continuation ladder in E: converge the saddle at each coupling.

        Parameters
        ----------
        e_targets : sequence of float
            Increasing Coulomb couplings; the saddle converged at one is
            the initial guess for the next (adiabatic double layer).
        w_aux : ndarray, optional
            Initial auxiliary fields (M x n_grid); zeros if omitted.

        Returns
        -------
        dict with converged 'w_aux', 'psi', 'phi', per-E convergence info.
        """
        M = len(self.monomer_types)
        n_grid = self.cb.get_total_grid()
        if w_aux is None:
            w_aux = np.zeros((M, n_grid), dtype=np.float64)

        history = []
        for E in e_targets:
            self.electro.e_coupling = float(E)
            self._rebuild_psi_jacobian()
            phi, hamiltonian, n_iter, err = self.find_saddle_point(w_aux)
            history.append({"E": float(E), "iters": int(n_iter),
                            "error": float(err),
                            "hamiltonian": float(np.real(hamiltonian))})
            if verbose:
                Qs = [self.prop_solver.get_partition_function(p)
                      for p in range(self.prop_solver.get_n_polymer_types())]
                print(f"E={E:10.1f}: {n_iter:5d} iters, err={err:.3e}, "
                      f"H={np.real(hamiltonian):+.6f}, "
                      f"lnQ0={np.log(np.abs(Qs[0])):+.3f}, "
                      f"max|psi|={np.abs(self.electro.psi).max():.2f}",
                      flush=True)
            if not np.isfinite(err) or err > self.saddle["tolerance"] * 100:
                print(f"WARNING: continuation stalled at E={E} "
                      f"(err={err:.3e}); returning last state.")
                break
        return {"w_aux": w_aux, "psi": self.electro.psi.copy(),
                "phi": phi, "history": history}
