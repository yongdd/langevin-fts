"""Charged-polymer CL-FTS (devel only — mainline is untouched).

Subclass of polymerfts.clfts.CLFTS adding the smeared-charge compressible
model of reports/CLFTS_VALIDATION.md with a FULLY FLUCTUATING electrostatic potential psi:
psi is an imaginary-type field evolving with complex Langevin dynamics like
the pressure field W+:

    dpsi/dtau = lap(psi)/E + c + i eta,   c = sum_i z_i (h_i * phi_i)

(rotated-contour stored field: the saddle of psi is real, noise is applied
in the imaginary direction, exactly the W+ convention of CLFTS). The
drift's fixed point is the Poisson equation -lap(psi) = E c. The stiff
linear part (rate gamma_k = k^2/E) is integrated by ETD (exact
Ornstein-Uhlenbeck step per k-mode with variance-exact noise scaling) —
explicit Euler is unstable at high k and semi-implicit Euler suppresses
the stationary variance by 1/(1+2 kappa dt) per mode, which wrecks
fluctuation thermodynamics. The explicit Hamiltonian term is the analytic
+(1/2E V) int psi lap(psi) (= -(1/2) mean(c psi) only at the Poisson
saddle); see reports/CLFTS_VALIDATION.md, charged-theory section.

Propagator inputs (complex): W_i = h_i * (W_i^SPT + z_i psi).

The run loop follows CLFTS._run of the mainline (integrator interface,
callbacks, checkpoints, measurements on the post-update state); the
charged additions are the methods _advance_psi, _electrostatic_hamiltonian
and _hamiltonian, which BrushChargedCLFTS reuses with its wall basis.
After the auxiliary fields advance with the selected integrator, psi takes
one ETD step with the charge density of the same (pre-update) state.

The k=0 mode of psi is projected out every step (electroneutrality makes
its drift vanish; without projection the noise random-walks a gauge-like
mode).

psi noise comes from a SEPARATE PCG64 stream (seed+1) so the main field
noise sequence is identical to plain CLFTS — the z=0 neutral reduction is
then exact against CLFTS with global smearing.

The structure functions are those of the mainline estimator,
<mu_i(k) phi_j(-k)>. The auxiliary fields couple to the smeared densities
h_i * phi_i, so the estimator gives h_i(k) S_ij(k); it is divided by
h_i(k) of the first species of each pair (modes with h_i < 1e-12 are
recorded as 0).

Activated params: "charges", "zeta_n",
"radiuses", "bjerrum_e" (= E) or "bjerrum_length" (= l_B/R0). Optional:
"psi_dt_scaling" (mobility scaling of the psi update, default 1.0).

Known limitations: no random copolymers; no Maxwell stress. Checkpoints
store both noise streams; old checkpoints lacking the psi RNG cannot
resume the same stochastic trajectory.
"""
import os
import time
import pathlib
import itertools
import logging
import json

import numpy as np
from scipy.io import savemat, loadmat

from polymerfts import clfts
from polymerfts.integrators import checked_fields
from polymerfts.utils import atomic_savemat, notify_callbacks, restore_recording_state
from polymerfts.validation import ValidationError

from .common import _drop_none_values, _PerSpeciesSmearing
from .electrostatics import Electrostatics

logger = logging.getLogger(__name__)


def _json_numpy(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to checkpoint JSON.")


class ChargedCLFTS(clfts.CLFTS):
    """CL-FTS with smeared charges and a fluctuating Coulomb psi."""

    def __init__(self, params, random_seed=None, *, integrator=None, callbacks=None):
        if params.get("charges", None) is None:
            raise ValidationError("ChargedCLFTS requires the 'charges' parameter.")
        if params.get("zeta_n", None) is None:
            raise ValidationError(
                "Charged-polymer mode requires the COMPRESSIBLE model: "
                "set 'zeta_n' explicitly (see reports/CLFTS_VALIDATION.md).")
        if params.get("smearing", None) is not None:
            raise ValidationError(
                "Do not combine 'smearing' with 'charges': charged mode "
                "uses per-species smearing via 'radiuses'.")
        if "radiuses" not in params:
            raise ValidationError(
                "Charged-polymer mode requires 'radiuses' (per-species "
                "smearing lengths / Born radii, in R0 units).")

        super().__init__(params=params, random_seed=random_seed,
                         integrator=integrator, callbacks=callbacks)

        if len(self.random_fraction) > 0:
            raise NotImplementedError(
                "Charged-polymer mode does not support random copolymers yet.")

        # Species-name and radius validation (typos would silently produce
        # neutral/unsmeared species; unsmeared charges are UV-singular).
        unknown = set(params["charges"]) - set(self.monomer_types)
        if unknown:
            raise ValidationError(
                f"'charges' contains unknown monomer types: {sorted(unknown)}")
        unknown = set(params["radiuses"]) - set(self.monomer_types)
        if unknown:
            raise ValidationError(
                f"'radiuses' contains unknown monomer types: {sorted(unknown)}")
        dx_max = max(self.cb.get_lx()[d] / self.cb.get_nx()[d]
                     for d in range(len(self.cb.get_nx())))
        for t in self.monomer_types:
            a = params["radiuses"].get(t)
            if a is None or float(a) <= 0.0:
                raise ValidationError(
                    f"'radiuses' must give a positive smearing radius for "
                    f"every monomer type; missing or non-positive for '{t}'.")
            if float(a) < 0.5 * dx_max:
                print(f"Warning: smearing radius a_{t} = {float(a)} < dx/2 "
                      f"= {0.5*dx_max:.4f}; the smeared charge is "
                      "under-resolved on this grid.")

        # Coulomb coupling E = 4 pi l_B rho_0 N R0^2 = sqrt(nbar)*4pi(l_B/R0)N^2
        if "bjerrum_e" in params:
            e_coupling = float(params["bjerrum_e"])
        elif "bjerrum_length" in params:
            if abs(1.0 / params["ds"] - round(1.0 / params["ds"])) > 1e-9:
                raise ValidationError(
                    f"'bjerrum_length' requires 1/ds to be an integer "
                    f"(N_Ref); got 1/ds = {1.0/params['ds']}. Provide "
                    "'bjerrum_e' directly instead.")
            n_ref = int(round(1.0 / params["ds"]))
            e_coupling = 4.0 * np.pi * float(params["bjerrum_length"]) \
                * n_ref**2 * np.sqrt(params["langevin"]["nbar"])
        else:
            raise ValidationError(
                "Charged-polymer mode requires 'bjerrum_e' (= E) or "
                "'bjerrum_length' (= l_B/R0).")
        if not np.isfinite(e_coupling) or e_coupling < 0:
            raise ValidationError("Coulomb coupling E must be finite and nonnegative.")

        # Electroneutrality (per-segment fractions); also gives the psi
        # preconditioner data (unused in CL but kept for diagnostics).
        Electrostatics.check_electroneutrality(
            self._species_fractions(), params["charges"])

        self.electro = Electrostatics(
            self.cb.get_nx(), self.cb.get_lx(), self.monomer_types,
            params["charges"], params["radiuses"], e_coupling,
            species_fractions=self._species_fractions())

        # psi state (complex; rotated-contour storage like W+)
        n_grid = self.cb.get_total_grid()
        self.psi = np.zeros(n_grid, dtype=np.complex128)
        self.psi_dt_scaling = float(params.get("psi_dt_scaling", 1.0))
        if not np.isfinite(self.psi_dt_scaling) or self.psi_dt_scaling <= 0:
            raise ValidationError("psi_dt_scaling must be finite and positive.")

        # Mainline smearing call sites (forces, checkpoint settings) go through
        # the per-species kernels; the charged model enters the settings hash.
        self.smearing = _PerSpeciesSmearing(self.electro, self.psi_dt_scaling)
        self._build_psi_etd()

        # Separate noise stream for psi keeps the main field-noise sequence
        # identical to plain CLFTS (exact neutral reduction).
        if random_seed is None:
            self._random_psi = np.random.Generator(np.random.PCG64())
        else:
            self._random_psi = np.random.Generator(np.random.PCG64(random_seed + 1))

        # Zero-eigenvalue (non-interacting) aux modes, e.g. the exchange mode of a
        # chi=0 polycation/polyanion mixture whose composition fluctuations enter
        # only through psi, carry no auxiliary field. Mainline pins them to zero
        # and gives them zero mobility (SymmetricPolymerTheory.aux_fields_dropped_idx).
        M = len(self.monomer_types)
        self._frozen_aux = list(self.mpt.aux_fields_dropped_idx)
        self._active_aux = [i for i in range(M) if i not in self._frozen_aux]

        # savemat cannot store None values
        self.params = _drop_none_values(self.params)

        print("Charged CL-FTS: E = %g, psi_dt_scaling = %g" %
              (e_coupling, self.psi_dt_scaling))

    # ------------------------------------------------------------------ #
    def _species_fractions(self):
        """Overall segment volume fraction of each monomer type."""
        species_fractions = {t: 0.0 for t in self.monomer_types}
        for polymer in self.polymers:
            alpha = sum(blk["length"] for blk in polymer["blocks"])
            for blk in polymer["blocks"]:
                species_fractions[blk["type"]] += \
                    polymer["volume_fraction"] * blk["length"] / alpha
        return species_fractions

    def _build_psi_etd(self):
        """ETD (exact-OU) coefficients of the linear psi update on the electro k grid:
        decay a, drift (1-a)E/k^2, and the per-mode noise amplitude that reproduces
        the exact stationary variance."""
        e_coupling = self.electro.e_coupling
        if e_coupling > 0.0:
            dt = float(self.langevin["dt"])
            s = self.psi_dt_scaling
            gamma = self.electro.k_sq / e_coupling
            a_dec = np.exp(-gamma * dt * s)
            with np.errstate(divide="ignore", invalid="ignore"):
                drift = -np.expm1(-gamma * dt * s) / gamma
                namp = np.sqrt(-np.expm1(-2.0 * gamma * dt * s) / (2.0 * gamma * dt))
            zero = tuple([0] * self.electro.dim)
            drift[zero] = dt * s
            namp[zero] = np.sqrt(s)
            self._psi_decay = a_dec
            self._psi_drift = drift
            self._psi_namp = namp

    # ------------------------------------------------------------------ #
    def _psi_force(self, phi):
        """lambda_psi = dH/dpsi = lap(psi)/E + c (complex), and c itself."""
        c = self.electro.charge_density(phi)
        if self.electro.e_coupling == 0:
            return np.zeros_like(c), c
        psi_hat = self.electro._t(np.reshape(self.psi, self.electro.nx))
        lap = self.electro._it(-self.electro.k_sq * psi_hat).reshape(-1)
        return lap / self.electro.e_coupling + c, c

    def _advance_psi(self, charge_c):
        """One psi step (imaginary-type: +lambda drift, i-noise).

        The linear lap(psi)/E part (rate gamma_k = k^2/E) is stiff at high k, and
        both explicit Euler (unstable) and semi-implicit Euler (stationary variance
        suppressed by 1/(1+2 kappa dt) per mode — ruins fluctuation thermodynamics)
        fail. ETD (exact OU integration) per k-mode:
            psi_hat' = a psi_hat + (1-a) E c_hat/k^2 + i amp_k xi_hat,
            a = exp(-gamma_k dt s),
            amp_k = sigma sqrt((1-a^2)/(2 gamma_k dt))  (-> sigma sqrt(s) as gamma -> 0),
        which reproduces the exact per-mode stationary variance of the linear part
        for ANY dt. The c[psi] screening part of the drift stays explicit (per-step
        rate <= sum z^2 phibar dt). E = 0: no Coulomb interaction — psi has infinite
        stiffness (k^2/E -> inf) and stays pinned at zero.
        """
        if self.electro.e_coupling <= 0.0:
            return
        xi = self._random_psi.normal(
            0.0, self.langevin["sigma"], self.cb.get_total_grid())
        psi_hat = self.electro._t(np.reshape(self.psi, self.electro.nx))
        c_hat = self.electro._t(np.reshape(charge_c, self.electro.nx))
        xi_hat = self.electro._t(np.reshape(xi, self.electro.nx))
        psi_hat = (self._psi_decay * psi_hat
                   + self._psi_drift * c_hat
                   + 1j * self._psi_namp * xi_hat)
        psi_hat[tuple([0] * self.electro.dim)] = 0.0   # gauge: k=0
        self.psi = np.ascontiguousarray(self.electro._it(psi_hat).reshape(-1))

    def _electrostatic_hamiltonian(self):
        """Explicit electrostatic term (1/2E V) int psi lap(psi) of the current psi.

        Equals -(1/2) mean(c psi) only at the Poisson saddle.
        """
        if self.electro.e_coupling <= 0.0:
            return 0.0
        psi_hat = self.electro._t(np.reshape(self.psi, self.electro.nx))
        lap = self.electro._it(-self.electro.k_sq * psi_hat).reshape(-1)
        return 0.5 * np.mean(self.psi * lap) / self.electro.e_coupling

    def _total_partitions(self):
        return [self.prop_solver.get_partition_function(p)
                for p in range(self.prop_solver.get_n_polymer_types())]

    def _hamiltonian(self, w_aux, phi_for_force, total_partitions):
        """Chain and field part of the mainline Hamiltonian plus the electrostatic term."""
        hamiltonian = self.mpt.compute_hamiltonian(
            self.molecules, w_aux, total_partitions, self.cb,
            include_const_term=True, phi=phi_for_force)
        return hamiltonian + self._electrostatic_hamiltonian()

    # FFT structure functions (the wall basis of BrushChargedCLFTS disables them)
    _record_structure_function = True

    # ------------------------------------------------------------------ #
    def compute_concentrations(self, w_aux):
        """Charged version: complex propagator inputs W_i = h_i*(W_SPT + z_i psi)."""
        if self.electro.e_coupling == 0:
            self.psi.fill(0.)
        M = len(self.monomer_types)
        elapsed_time = {}

        w = self.mpt.to_monomer_fields(w_aux)
        w_input = {self.monomer_types[i]: w[i] for i in range(M)}

        w_input_for_propagator = {
            t: self.electro.smear(t, w_input[t] + self.electro.charges[t] * self.psi)
            for t in w_input
        }

        time_solver_start = time.time()
        self.prop_solver.compute_propagators(w_input_for_propagator)
        elapsed_time["solver"] = time.time() - time_solver_start

        time_phi_start = time.time()
        phi = {}
        self.prop_solver.compute_concentrations()
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.prop_solver.get_concentration(monomer_type)
        elapsed_time["phi"] = time.time() - time_phi_start

        # Material conservation: mean(sum phi) must stay 1 (a numerical blow-up shows up here first)
        mass_error = self.prop_solver.mean(sum(phi[m] for m in self.monomer_types) - 1.0)
        if not abs(mass_error) <= 5e-8:
            raise ValidationError(f"Material conservation failed: mean(sum(phi) - 1) = {mass_error:.3e}")

        # Partition consistency check (forward vs backward propagators), once per run
        if not getattr(self, "_partition_checked", False):
            self._partition_checked = True
            if not self.prop_solver.check_total_partition():
                raise ValidationError("Partition consistency check failed (forward/backward mismatch).")

        return phi, elapsed_time

    # ------------------------------------------------------------------ #
    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev, w_aux=None):
        """Mainline checkpoint plus psi and the independent electrostatic RNG."""
        super().save_simulation_data(path, w, phi, langevin_step, normal_noise_prev, w_aux=w_aux)
        data = loadmat(path, squeeze_me=False)
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        # Empty MATLAB structs (e.g. integer-key graft maps) do not roundtrip
        # through scipy's structured arrays. Reuse the original clean params.
        data["initial_params"] = self.params
        # MATLAB field names cannot represent integer graft vertices.
        data["initial_params_json"] = json.dumps(self.params, default=_json_numpy)
        data["psi_real"] = np.real(self.psi)
        data["psi_imag"] = np.imag(self.psi)
        data["psi_rng_state_json"] = json.dumps(self._random_psi.bit_generator.state)
        if hasattr(self.solver, "get_log_partition"):
            logq = np.array([self.solver.get_log_partition(p)
                             for p in range(len(self.polymers))])
            data["chain_log_partition_real"] = logq.real
            data["chain_log_partition_imag"] = logq.imag
        if hasattr(self, "w_shifts"):
            data["w_shifts_json"] = json.dumps(self.w_shifts, sort_keys=True)
            data["brush_chain_model_json"] = json.dumps({k: self.params.get(k) for k in
                ("chain_backend", "grafting", "bond_boundaries", "graft_width", "graft_center", "bc")},
                sort_keys=True)
        atomic_savemat(path, data, long_field_names=True, do_compression=True)

    def continue_run(self, file_name, *, allow_legacy=False):
        """Continue from a checkpoint, restoring psi if present."""
        data = loadmat(file_name, squeeze_me=True)
        if "w_shifts_json" in data and json.loads(str(data["w_shifts_json"])) != getattr(self, "w_shifts", None):
            raise ValidationError("Checkpoint w_shifts differ from the requested brush model.")
        if "brush_chain_model_json" in data:
            saved = json.loads(str(data["brush_chain_model_json"]))
            if saved != {k: self.params.get(k) for k in saved}:
                raise ValidationError("Checkpoint grafting/bond model differs from requested model.")
        if "psi_rng_state_json" in data:
            self._random_psi.bit_generator.state = json.loads(str(data["psi_rng_state_json"]))
        else:
            logger.warning("Checkpoint lacks psi RNG state; continuation is not trajectory-exact.")
        if "psi_real" in data:
            self.psi = np.asarray(data["psi_real"]).reshape(-1) \
                + 1j * np.asarray(data["psi_imag"]).reshape(-1)
        else:
            self.psi.fill(0.)
            print("Note: checkpoint has no psi field; psi restarts from zero.")
        return super().continue_run(file_name, allow_legacy=allow_legacy)

    # ------------------------------------------------------------------ #
    def _run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None, *, _restart_data=None):
        """CL loop of CLFTS._run with the psi field co-evolved.

        After the auxiliary-field step, psi takes one ETD step with the charge
        density of the same pre-update state; the Hamiltonian gains the explicit
        (1/2EV) int psi lap(psi) term; the printed error block gains
        std(lambda_psi) as the Poisson-residual monitor.
        """
        print("---------- Run (charged CL-FTS) ----------")

        M = len(self.monomer_types)
        n_grid = self.cb.get_total_grid()
        nx = list(self.cb.get_nx())

        self._partition_checked = False
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        w = np.array([np.reshape(initial_fields[m], n_grid).astype(np.complex128)
                      for m in self.monomer_types])
        w_aux = self.mpt.to_aux_fields(w)

        # Zero-eigenvalue modes carry no auxiliary field and stay zero.
        for i in self._frozen_aux:
            proj = float(np.max(np.abs(w_aux[i])))
            if proj > 1e-10:
                print(f"Warning: initial fields have a nonzero projection "
                      f"({proj:.3e}) onto the zero-eigenvalue aux mode {i}; "
                      f"projecting it out.")
        self.mpt.pin_dropped_fields(w_aux)
        if _restart_data is not None and "w_aux" in _restart_data:
            w_aux = np.asarray(_restart_data["w_aux"], dtype=np.complex128).reshape(M, n_grid).copy()
        if not np.all(np.isfinite(w_aux)):
            raise ValidationError("Initial fields must be finite")

        H_history = []
        drift_history = []
        dH_history = {key: [] for key in self.chi_n}

        sf_average = {}  # <mu(k) phi(-k)>
        for monomer_id_pair in itertools.combinations_with_replacement(list(range(M)), 2):
            sorted_pair = sorted(monomer_id_pair)
            type_pair = self.monomer_types[sorted_pair[0]] + "," + self.monomer_types[sorted_pair[1]]
            sf_average[type_pair] = np.zeros(nx, dtype=np.complex128)
        sf_sample_count = restore_recording_state(_restart_data, H_history, dH_history, sf_average, drift_history)

        if normal_noise_prev is None:
            normal_noise_prev = np.zeros([M, n_grid], dtype=np.float64)
        else:
            normal_noise_prev = np.reshape(np.asarray(normal_noise_prev, dtype=np.float64), [M, n_grid])
        if start_langevin_step is None:
            start_langevin_step = 1

        time_start = time.time()
        langevin_step = start_langevin_step - 1

        # Concentrations of the initial fields; afterwards they are computed right
        # after each update (measurements and the next forces use the same state).
        if _restart_data is not None and "w_aux" in _restart_data:
            phi = {m: (np.asarray(_restart_data["phi_" + m + "_real"]) +
                       1j * np.asarray(_restart_data["phi_" + m + "_imag"])).reshape(n_grid)
                   for m in self.monomer_types}
        else:
            phi, _ = self.compute_concentrations(w_aux=w_aux)
        phi_for_force = self.smearing.apply_to_dict(phi)

        for langevin_step in range(start_langevin_step, self.langevin["max_step"] + 1):
            step_start = time.perf_counter()

            # Forces of the current state (zero-mobility modes are not advanced)
            w_lambda = self.mpt.compute_func_deriv(w_aux, phi_for_force, range(M))
            if self.alpha_ds > 0:
                for i in self.mpt.aux_fields_real_idx:
                    w_lambda[i] += 1j * self.alpha_ds * np.imag(w_aux[i])
            drift_max = float(np.max(np.abs(w_lambda * self.dt_scaling[:, None])))

            # Charge density of the same (pre-update) state for the psi step
            _, charge_c = self._psi_force(phi)

            # Auxiliary fields: the selected integrator (default Leimkuhler-Matthews)
            normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], [M, n_grid])
            noise_direction = np.array([1.0 if i in self.mpt.aux_fields_real_idx else 1j for i in range(M)])
            proposed = self.integrator.step(
                w_aux.copy(), w_lambda.copy(), dt=self.langevin["dt"], mobility=self.dt_scaling.copy(),
                noise=normal_noise_current.copy(), previous_noise=normal_noise_prev.copy(),
                noise_direction=noise_direction,
                evaluate_force=lambda fields: self.evaluate_force(fields))
            proposed = checked_fields(proposed, w_aux)
            if not np.array_equal(proposed[self.dt_scaling == 0], w_aux[self.dt_scaling == 0]):
                raise ValidationError("An integrator must preserve fields with zero mobility")
            w_aux = proposed.copy()
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            # Electrostatic potential
            self._advance_psi(charge_c)

            # Concentrations and partition functions of the updated state
            phi, _ = self.compute_concentrations(w_aux=w_aux)
            phi_for_force = self.smearing.apply_to_dict(phi)
            if not np.all(np.isfinite(w_aux)) or not np.all(np.isfinite(self.psi)) \
                    or not all(np.all(np.isfinite(v)) for v in phi.values()):
                raise ValidationError(f"Non-finite complex fields or concentrations at step {langevin_step}")

            # ---- monitoring ----
            h_deriv = self.mpt.compute_func_deriv(w_aux, phi_for_force, range(M))
            error_level_array = np.std(h_deriv, axis=1)
            psi_lambda, _ = self._psi_force(phi)
            psi_error = float(np.std(psi_lambda))

            total_partitions = self._total_partitions()
            hamiltonian = self._hamiltonian(w_aux, phi_for_force, total_partitions)
            if not np.isfinite(hamiltonian):
                raise ValidationError(f"Non-finite Hamiltonian at step {langevin_step}")
            self.run_status["last_completed_step"] = langevin_step

            if self.verbose_level >= 1:
                mass_error = self.cb.mean(h_deriv[M - 1])
                print(f"{langevin_step:8d} {mass_error.real:+.3E}{mass_error.imag:+.3E}j", end="  [ ")
                for Q in total_partitions:
                    print(f"{Q.real:.6E}{Q.imag:+.6E}j", end=" ")
                print(f"] {hamiltonian.real:+.9E}{hamiltonian.imag:+.9E}j  [", end="")
                for i in range(M):
                    print(f" {error_level_array[i]:.3E}", end="")
                print(f" {psi_error:.3E}", end="")
                print(" ]")

            # H and dH/dchi_n
            if langevin_step % self.recording["sample_interval"] == 0:
                H_history.append(hamiltonian)
                drift_history.append(drift_max)
                sf_sample_count += 1
                dH = self.mpt.compute_h_deriv_chin(w_aux=w_aux, phi=phi_for_force)
                for key in self.chi_n:
                    dH_history[key].append(dH[key])

            if langevin_step % self.recording["statistics_save_interval"] == 0:
                H_history_array = np.array(H_history)
                mdic = {"H_history_real": np.real(H_history_array), "H_history_imag": np.imag(H_history_array),
                        "sample_count": sf_sample_count, "drift_max_history": np.asarray(drift_history),
                        "simulation_metadata_json": json.dumps(self.metadata)}
                for key in self.chi_n:
                    dH_array = np.array(dH_history[key])
                    monomer_pair = sorted(key.split(","))
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1] + "_real"] = np.real(dH_array)
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1] + "_imag"] = np.imag(dH_array)
                savemat(os.path.join(self.recording["dir"], "dH_%06d.mat" % langevin_step),
                        mdic, long_field_names=True, do_compression=True)
                H_history = []
                drift_history = []
                for key in self.chi_n:
                    dH_history[key] = []

            # Structure function <mu_i(k) phi_j(-k)> of the mainline estimator
            if self._record_structure_function and langevin_step % self.recording["sample_interval"] == 0:
                mu_fourier = {}
                phi_fourier = {}   # phi(-k) = conj(FFT[conj phi](k))
                for i in range(M):
                    key = self.monomer_types[i]
                    phi_fourier[key] = np.conj(np.fft.fftn(
                        np.conj(np.reshape(phi[key], nx)))) / n_grid
                    mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
                    for k in range(M):
                        if np.isclose(self.mpt.eigenvalues[k], 0.0):
                            continue
                        mu_fourier[key] += (
                            np.fft.fftn(np.reshape(w_aux[k], nx)) *
                            self.mpt.matrix_a[i, k] / self.mpt.eigenvalues[k] / n_grid)
                for key in sf_average:
                    monomer_pair = sorted(key.split(","))
                    sf_average[key] += mu_fourier[monomer_pair[0]] * phi_fourier[monomer_pair[1]]

            if self._record_structure_function and langevin_step % self.recording["statistics_save_interval"] == 0:
                chi_n_mat = {}
                for key in self.chi_n:
                    monomer_pair = sorted(key.split(","))
                    chi_n_mat[monomer_pair[0] + "," + monomer_pair[1]] = self.chi_n[key]
                mdic = {"dim": self.cb.get_dim(), "nx": nx, "lx": self.cb.get_lx(),
                        "chi_n": chi_n_mat, "chain_model": self.chain_model, "ds": self.ds,
                        "dt": self.langevin["dt"], "nbar": self.langevin["nbar"],
                        "initial_params": self.params}
                for key in sf_average:
                    monomer_pair = sorted(key.split(","))
                    sf_scaled = (sf_average[key] / max(sf_sample_count, 1) *
                                 self.cb.get_volume() * np.sqrt(self.langevin["nbar"]))
                    # The fields couple to h_i * phi_i: <mu_i phi_j> = h_i(k) S_ij(k)
                    h = self.electro.h_hat[monomer_pair[0]]
                    if h is not None:
                        sf_scaled = np.divide(sf_scaled, h, out=np.zeros_like(sf_scaled), where=h > 1e-12)
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1] + "_real"] = np.real(sf_scaled)
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1] + "_imag"] = np.imag(sf_scaled)
                mdic["sample_count"] = sf_sample_count
                mdic["simulation_metadata_json"] = json.dumps(self.metadata)
                savemat(os.path.join(self.recording["dir"], "structure_function_%06d.mat" % langevin_step),
                        mdic, long_field_names=True, do_compression=True)
                for key in sf_average:
                    sf_average[key][:] = 0.0
                sf_sample_count = 0
            elif not self._record_structure_function and langevin_step % self.recording["statistics_save_interval"] == 0:
                sf_sample_count = 0

            notify_callbacks(self, langevin_step, w_aux, phi, hamiltonian, drift_max,
                             error_level_array, time.perf_counter() - step_start)

            if langevin_step % self.recording["checkpoint_interval"] == 0:
                self._recording_state = {"H_history": H_history, "dH_history": dH_history,
                                         "sf_average": sf_average, "sample_count": sf_sample_count,
                                         "drift_history": drift_history}
                w = self.mpt.to_monomer_fields(w_aux)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % langevin_step),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev, w_aux=w_aux)

        time_duration = time.time() - time_start
        n_steps = max(langevin_step + 1 - start_langevin_step, 1)
        print("total time: %f, time per step: %f" % (time_duration, time_duration / n_steps))
