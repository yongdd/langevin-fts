"""Grafted-brush charged CL-FTS in a wall box.

Stage 2 of the PRL-2025 reproduction plan (Duan/Agrawal/Wang, PRL 134,
048101 — references/2025_prl_duan.pdf): inhomogeneous polyelectrolyte
brush with a fluctuating electrostatic potential, built on the validated
ChargedCLFTS (see RESULTS_ONELOOP_DH.md, RESULTS_LPF2008.md,
RESULTS_MULTIVALENT.md).

Geometry
--------
All six faces 'reflecting' (the pseudo-spectral solvers reject mixing
periodic with non-periodic axes). The wall carrying the brush is the
z=0 face (last axis); the opposite face at z=Lz acts as a mirror, so the
far region doubles as a bulk salt reservoir when Lz is large enough.
Lateral reflecting faces are mirrors too; lateral structure and finite-size
effects must be checked before comparison with a periodic or infinite brush.

What changes vs ChargedCLFTS
----------------------------
1. Propagators: ComputationBox built with bc=['reflecting']*2dim;
   polymers may carry per-polymer "grafting_points" ({vertex: label});
   for every graft label a wall-attached GAUSSIAN sheet (width
   "graft_width", default 0.2; normalized to unit line integral) is
   passed as q_init — a one-cell delta rings on the pseudo-spectral
   propagator at strong coupling. The legacy spectral source is laterally
   annealed. It is NOT the quenched ensemble of SI Eq. S31 in fluctuating
   lateral fields. Select chain_backend='direct', grafting='quenched' for
   the scaled direct Gaussian reference backend (CPU or CuPy CUDA).
   A graft_width of zero pins each anchor to the nearest cell.
2. psi sector: ElectrostaticsReflecting (DCT-II basis, k_d = pi n_d/L_d,
   Neumann walls = neutral non-polarizable mirrors). The ETD
   (exact-OU-per-mode) integrator carries over verbatim — the linear
   part is diagonal in the DCT basis, and transforming real-space white
   noise into that basis reproduces the exact stationary variance of the
   real-space Langevin SDE for any dt, exactly as in the periodic case.
3. Structure-function (FFT) diagnostics are disabled (meaningless in the
   wall basis). The run loop is ChargedCLFTS._run; this class supplies the
   wall solver, the log-partition Hamiltonian and the frozen re-gauge
   correction through _total_partitions and _hamiltonian.

Platform: cpu-mkl / cpu-fftw / cuda (discrete chain model only on CUDA;
the complex + non-periodic transforms were fixed in mainline 2026-08-18:
real/imaginary parts transform independently through the DCT/DST with
interleaved complex coefficients, ~12x faster than CPU on brush-sized
grids).

Usage: params as ChargedCLFTS plus
    "bc": ["reflecting"]*6            (optional; this is the default here)
    polymers[i]["grafting_points"] = {0: "G"}   (grafted chains)
"""
import time

import numpy as np

from polymerfts import _core
from polymerfts.validation import ValidationError

from .clfts_charged import ChargedCLFTS
from .common import _PerSpeciesSmearing
from .electrostatics import ElectrostaticsReflecting


class BrushChargedCLFTS(ChargedCLFTS):
    """Charged CL-FTS with reflecting/Dirichlet chain walls and grafted chains.

    The direct backend also supports explicitly truncated Gaussian bonds.
    """

    def __init__(self, params, random_seed=None, *, integrator=None, callbacks=None):
        backend = params.get("chain_backend", "spectral")
        grafting = params.get("grafting", "annealed")
        if backend not in ("spectral", "direct"):
            raise ValidationError("chain_backend must be 'spectral' or 'direct'.")
        if grafting not in ("annealed", "quenched"):
            raise ValidationError("grafting must be 'annealed' or 'quenched'.")
        if backend != "direct" and (grafting != "annealed" or "bond_boundaries" in params):
            raise ValidationError("Quenched grafting/custom bond boundaries require chain_backend='direct'.")
        dim = len(params["nx"])
        bc = params.get("bc", ["reflecting"] * (2 * dim))
        # reflecting/absorbing may be mixed ACROSS axes (repo-supported;
        # cf. TestStressNonPeriodic a/a+r/r); periodic may not be mixed
        # with non-periodic, and both faces of one axis must match.
        # ElectrostaticsReflecting is independent of the chain box BC.
        # Absorbing DST uses a Dirichlet image kernel. The discrete truncated
        # Gaussian wall in prl_scft requires direct + hard-reflecting instead.
        if len(bc) != 2 * dim:
            raise ValidationError(f"'bc' needs {2*dim} entries, got {len(bc)}.")
        if any(b not in ("reflecting", "absorbing") for b in bc):
            raise ValidationError(
                f"BrushChargedCLFTS: bc must be reflecting/absorbing (got {bc}).")
        for ax in range(dim):
            if bc[2 * ax] != bc[2 * ax + 1]:
                raise ValidationError(
                    f"BrushChargedCLFTS: both faces of axis {ax} must share "
                    f"one BC (got {bc[2*ax]}/{bc[2*ax+1]}).")
        platform = params.get("platform", "cpu-mkl")
        if platform not in ("cpu-mkl", "cpu-fftw", "cuda"):
            raise ValidationError(f"Unsupported platform '{platform}'.")
        if platform == "cuda" and params.get("chain_model") != "discrete":
            raise ValidationError(
                "CUDA complex + non-periodic is currently ported for the "
                "discrete chain model only (RQM4/RK2 CUDA solvers raise); "
                "use 'discrete' or a CPU platform.")

        # Parent builds the (periodic) machinery, validates the charged
        # model, and sets up noise streams / frozen-mode bookkeeping. The
        # periodic solver and Electrostatics are replaced below.
        super().__init__(params=params, random_seed=random_seed,
                         integrator=integrator, callbacks=callbacks)

        # ---- rebuild the computational core with wall BCs ----
        reduce_memory = params.get("reduce_memory", False)
        factory = _core.PlatformSelector.create_factory(
            platform, reduce_memory, "complex")
        self.cb = factory.create_computation_box(
            params["nx"], params["lx"], bc=list(bc))
        molecules = factory.create_molecules_information(
            params["chain_model"], params["ds"], self.segment_lengths)
        graft_labels = []
        for polymer in self.polymers:
            gp = polymer.get("grafting_points", None)
            if gp:
                gp = {int(k): str(v) for k, v in gp.items()}
                molecules.add_polymer(
                    polymer["volume_fraction"], polymer["blocks_input"], gp)
                graft_labels.extend(gp.values())
            else:
                molecules.add_polymer(
                    polymer["volume_fraction"], polymer["blocks_input"])
        optimizer = factory.create_propagator_computation_optimizer(
            molecules, params.get("aggregate_propagator_computation", True))
        self.solver = factory.create_propagator_computation(
            self.cb, molecules, optimizer, "rqm4")
        self.molecules = molecules
        # solver holds a raw pointer to the optimizer; keep it alive
        self._propagator_optimizer = optimizer

        # ---- graft sources: wall-attached Gaussian sheet ----
        # Same as BrushChargedSCFT: a one-cell delta rings on the
        # pseudo-spectral propagator (negative Q_P at strong coupling).
        nx = params["nx"]
        dz = params["lx"][-1] / nx[-1]
        width = float(params.get("graft_width", 0.2))
        center = float(params.get("graft_center", 0.0))
        # with absorbing z (hard wall) the sheet must sit OFF the wall
        # (a wall-centered sheet is annihilated by the q=0 face)
        zc = (np.arange(nx[-1]) + 0.5) * dz
        if not np.isfinite(width) or width < 0 or not 0 <= center < params["lx"][-1]:
            raise ValidationError("Require graft_width >= 0 and graft_center inside the box.")
        if width == 0:
            prof = np.zeros(nx[-1])
            prof[np.argmin(abs(zc-center))] = 1.
        else:
            exponent = -0.5 * ((zc - center) / width) ** 2
            prof = np.exp(exponent-exponent.max())
        prof /= prof.sum() * dz
        q_init_sheet = np.zeros(nx, dtype=np.complex128)
        q_init_sheet[...] = prof[(None,) * (len(nx) - 1)]
        q_init_sheet = np.ascontiguousarray(q_init_sheet.reshape(-1))
        self.q_init = {label: q_init_sheet for label in set(graft_labels)}
        if backend == "direct":
            from .stable_grafted_chain import StableBrushSolver
            self.solver = StableBrushSolver(params, self.polymers, self.segment_lengths)

        # ---- frozen per-species re-gauge shifts (from the SCFT saddle) ----
        # Canonical densities are invariant under W_t -> W_t - s_t; the
        # shifts keep the grafted-chain propagators centered in double
        # precision (obstacle 2.3 of RESULTS_BRUSH_STATUS.md). FROZEN
        # during CL (pure bookkeeping; the reported H gains the constant
        # +sum_p vf_p s_t, applied in _hamiltonian()).
        for p in self.polymers:
            btypes = {b["type"] for b in p["blocks"]}
            if len(btypes) > 1 and any(
                    params.get("w_shifts", {}).get(t) for t in btypes):
                raise ValidationError(
                    "w_shifts bookkeeping is only correct for polymers whose "
                    "blocks share one monomer type (see brush_scft_charged).")
        unknown = set(params.get("w_shifts", {})) - set(self.monomer_types)
        if unknown:
            raise ValidationError(
                f"'w_shifts' contains unknown monomer types: {sorted(unknown)}")
        self.w_shifts = {t: float(params.get("w_shifts", {}).get(t, 0.0))
                         for t in self.monomer_types}
        self._h_shift_correction = 0.0
        for p in self.polymers:
            t = p["blocks"][0]["type"]
            self._h_shift_correction += p["volume_fraction"] * self.w_shifts[t]

        # ---- wall-basis electrostatics (replaces the periodic one) ----
        self.electro = ElectrostaticsReflecting(
            self.cb.get_nx(), self.cb.get_lx(), self.monomer_types,
            params["charges"], params["radiuses"], self.electro.e_coupling,
            species_fractions=self._species_fractions())
        self.smearing = _PerSpeciesSmearing(self.electro, self.psi_dt_scaling)

        # ---- ETD (exact-OU) coefficients on the DCT k grid ----
        self._build_psi_etd()

        print(f"Brush charged CL-FTS: bc = {bc}, "
              f"graft labels = {sorted(set(graft_labels))}, platform = {platform}, "
              f"chain_backend = {backend}, grafting = {grafting}")

    # ------------------------------------------------------------------ #
    def compute_concentrations(self, w_aux):
        """Parent's charged propagator inputs + grafted q_init sources."""
        if self.electro.e_coupling == 0:
            self.psi.fill(0.)
        M = len(self.monomer_types)
        elapsed_time = {}

        w = self.mpt.to_monomer_fields(w_aux)
        w_input = {self.monomer_types[i]: w[i] for i in range(M)}

        w_input_for_propagator = {
            t: self.electro.smear(t, w_input[t] + self.electro.charges[t] * self.psi)
            - self.w_shifts[t]
            for t in w_input
        }

        time_solver_start = time.time()
        self.solver.compute_propagators(
            w_input_for_propagator,
            q_init=self.q_init if self.q_init else None)
        if not self.solver.check_total_partition():
            raise FloatingPointError("Brush propagators failed relative complex partition consistency.")
        elapsed_time["solver"] = time.time() - time_solver_start

        time_phi_start = time.time()
        phi = {}
        self.solver.compute_concentrations()
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.solver.get_total_concentration(monomer_type)
        elapsed_time["phi"] = time.time() - time_phi_start

        return phi, elapsed_time

    def _chain_hamiltonian(self, w_aux, total_partitions, phi=None):
        """Use log partitions directly when the reference solver scales Q.

        phi (smeared concentrations) gives the mean-field term of zero-eigenvalue
        modes, which are not uniform in a wall box.
        """
        if not hasattr(self.solver, "get_log_partition"):
            return self.mpt.compute_hamiltonian(
                self.molecules, w_aux, total_partitions, self.cb, include_const_term=True, phi=phi)
        h = self.mpt.compute_hamiltonian(
            self.molecules, w_aux, [1.]*len(self.polymers), self.cb,
            include_const_term=True, phi=phi)
        for p, polymer in enumerate(self.polymers):
            alpha = sum(b["length"] for b in polymer["blocks"])
            h -= polymer["volume_fraction"]/alpha * self.solver.get_log_partition(p)
        return h

    def _total_partitions(self):
        return [self.solver.get_total_partition(p)
                for p in range(self.molecules.get_n_polymer_types())]

    def _hamiltonian(self, w_aux, phi_for_force, total_partitions):
        """Chain part from the wall solver, electrostatic term and the constant
        +sum_p vf_p s_t of the frozen re-gauge shifts."""
        return (self._chain_hamiltonian(w_aux, total_partitions, phi_for_force)
                + self._electrostatic_hamiltonian() + self._h_shift_correction)

    # FFT structure functions are meaningless in the wall basis
    _record_structure_function = False
