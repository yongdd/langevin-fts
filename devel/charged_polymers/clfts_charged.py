"""Charged-polymer CL-FTS (devel only — mainline is untouched).

Subclass of polymerfts.clfts.CLFTS adding the smeared-charge compressible
model of THEORY.md with a FULLY FLUCTUATING electrostatic potential psi:
unlike the L-FTS prototype (lfts_charged.py), psi is NOT partial-saddled —
it is an imaginary-type field evolving with complex Langevin dynamics like
the pressure field W+:

    dpsi/dtau = lap(psi)/E + c + i eta,   c = sum_i z_i (h_i * phi_i)

(rotated-contour stored field: the saddle of psi is real, noise is applied
in the imaginary direction, exactly the W+ convention of CLFTS.run). The
drift's fixed point is the Poisson equation -lap(psi) = E c. The stiff
linear part (rate gamma_k = k^2/E) is integrated by ETD (exact
Ornstein-Uhlenbeck step per k-mode with variance-exact noise scaling) —
explicit Euler is unstable at high k and semi-implicit Euler suppresses
the stationary variance by 1/(1+2 kappa dt) per mode, which wrecks
fluctuation thermodynamics. The explicit Hamiltonian term is the analytic
+(1/2E V) int psi lap(psi) (= -(1/2) mean(c psi) only at the Poisson
saddle); see THEORY.md sections 4-5.

Propagator inputs (complex): W_i = h_i * (W_i^SPT + z_i psi).

The k=0 mode of psi is projected out every step (electroneutrality makes
its drift vanish; without projection the noise random-walks a gauge-like
mode).

psi noise comes from a SEPARATE PCG64 stream (seed+1) so the main field
noise sequence is identical to plain CLFTS — the z=0 neutral reduction is
then bit-exact against CLFTS with global smearing.

Activated params (same as the L-FTS prototype): "charges", "zeta_n",
"radiuses", "bjerrum_e" (= E) or "bjerrum_length" (= l_B/R0). Optional:
"psi_dt_scaling" (mobility scaling of the psi update, default 1.0).

Known limitations: no random copolymers; no Maxwell stress; restarting
from a checkpoint restores psi if present (older checkpoints start it at
zero and let it re-equilibrate).
"""
import os
import time
import pathlib
import itertools
import logging

import numpy as np
from scipy.io import savemat, loadmat

from polymerfts import clfts
from polymerfts.validation import ValidationError

from electrostatics import Electrostatics

logger = logging.getLogger(__name__)


def _drop_none_values(obj):
    """Recursively remove None-valued entries (savemat cannot store None)."""
    if isinstance(obj, dict):
        return {k: _drop_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_drop_none_values(v) for v in obj if v is not None]
    return obj


class _PerSpeciesSmearing:
    """Adapter exposing Electrostatics.smear_dict through the Smearing API."""

    def __init__(self, electro):
        self._electro = electro
        self.enabled = True

    def apply_to_dict(self, fields):
        return self._electro.smear_dict(fields)

    def apply(self, field):
        raise NotImplementedError(
            "Per-species smearing has no species-agnostic apply(); "
            "use apply_to_dict.")


class ChargedCLFTS(clfts.CLFTS):
    """CL-FTS with smeared charges and a fluctuating Coulomb psi."""

    def __init__(self, params, random_seed=None):
        if params.get("charges", None) is None:
            raise ValidationError("ChargedCLFTS requires the 'charges' parameter.")
        if params.get("zeta_n", None) is None:
            raise ValidationError(
                "Charged-polymer mode requires the COMPRESSIBLE model: "
                "set 'zeta_n' explicitly (see THEORY.md).")
        if params.get("smearing", None) is not None:
            raise ValidationError(
                "Do not combine 'smearing' with 'charges': charged mode "
                "uses per-species smearing via 'radiuses'.")
        if "radiuses" not in params:
            raise ValidationError(
                "Charged-polymer mode requires 'radiuses' (per-species "
                "smearing lengths / Born radii, in R0 units).")

        super().__init__(params=params, random_seed=random_seed)

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

        # Electroneutrality (per-segment fractions); also gives the psi
        # preconditioner data (unused in CL but kept for diagnostics).
        species_fractions = {t: 0.0 for t in self.monomer_types}
        for polymer in self.distinct_polymers:
            alpha = sum(blk["length"] for blk in polymer["blocks"])
            for blk in polymer["blocks"]:
                species_fractions[blk["type"]] += \
                    polymer["volume_fraction"] * blk["length"] / alpha
        Electrostatics.check_electroneutrality(
            species_fractions, params["charges"])

        self.electro = Electrostatics(
            self.cb.get_nx(), self.cb.get_lx(), self.monomer_types,
            params["charges"], params["radiuses"], e_coupling,
            species_fractions=species_fractions)

        # Route mainline phi-smearing call sites through per-species kernels.
        self.smearing = _PerSpeciesSmearing(self.electro)

        # psi state (complex; rotated-contour storage like W+)
        n_grid = self.cb.get_total_grid()
        self.psi = np.zeros(n_grid, dtype=np.complex128)
        self.psi_dt_scaling = float(params.get("psi_dt_scaling", 1.0))

        # ETD (exact-OU) coefficients for the linear part of the psi
        # update (see run()): decay a, drift (1-a)E/k^2, and the per-mode
        # noise amplitude that reproduces the exact stationary variance.
        if e_coupling > 0.0:
            dt = float(params["langevin"]["dt"])
            s = self.psi_dt_scaling
            gamma = self.electro.k_sq / e_coupling
            a_dec = np.exp(-gamma * dt * s)
            with np.errstate(divide="ignore", invalid="ignore"):
                drift = (1.0 - a_dec) * e_coupling / self.electro.k_sq
                namp = np.sqrt((1.0 - a_dec**2) / (2.0 * gamma * dt))
            zero = tuple([0] * self.electro.dim)
            drift[zero] = dt * s
            namp[zero] = np.sqrt(s)
            self._psi_decay = a_dec
            self._psi_drift = drift
            self._psi_namp = namp

        # Separate noise stream for psi keeps the main field-noise sequence
        # identical to plain CLFTS (bit-exact neutral reduction).
        if random_seed is None:
            self._random_psi = np.random.Generator(np.random.PCG64())
        else:
            self._random_psi = np.random.Generator(np.random.PCG64(random_seed + 1))

        # Zero-eigenvalue (non-interacting) aux modes appear in neither
        # aux_fields_real_idx nor aux_fields_imag_idx (e.g. the exchange
        # mode of a chi=0 polycation/polyanion mixture, whose composition
        # fluctuations enter only through psi). The HS transform introduces
        # no auxiliary field for them, so they must stay identically zero.
        M = len(self.monomer_types)
        self._active_aux = [i for i in range(M)
                            if i in self.mpt.aux_fields_real_idx
                            or i in self.mpt.aux_fields_imag_idx]
        self._frozen_aux = [i for i in range(M) if i not in self._active_aux]
        # The parent __init__ computes dt_scaling = |eig|/max|eig[:M-1]|,
        # which is 0/0 = NaN when a zero mode dominates the slice; frozen
        # modes are never stepped, so give them a defined mobility of 0.
        for i in self._frozen_aux:
            self.dt_scaling[i] = 0.0

        # savemat cannot store None values
        self.params = _drop_none_values(self.params)

        print("Charged CL-FTS: E = %g, psi_dt_scaling = %g" %
              (e_coupling, self.psi_dt_scaling))

    # ------------------------------------------------------------------ #
    def _psi_force(self, phi):
        """lambda_psi = dH/dpsi = lap(psi)/E + c (complex), and c itself."""
        c = self.electro.charge_density(phi)
        psi_hat = np.fft.fftn(np.reshape(self.psi, self.electro.nx))
        lap = np.fft.ifftn(-self.electro.k_sq * psi_hat).reshape(-1)
        return lap / self.electro.e_coupling + c, c

    # ------------------------------------------------------------------ #
    def compute_concentrations(self, w_aux):
        """Charged version: complex propagator inputs W_i = h_i*(W_SPT + z_i psi)."""
        M = len(self.monomer_types)
        elapsed_time = {}

        w = self.mpt.to_monomer_fields(w_aux)
        w_input = {self.monomer_types[i]: w[i] for i in range(M)}

        w_input_for_propagator = {
            t: self.electro.smear(t, w_input[t] + self.electro.charges[t] * self.psi)
            for t in w_input
        }

        time_solver_start = time.time()
        self.solver.compute_propagators(w_input_for_propagator)
        elapsed_time["solver"] = time.time() - time_solver_start

        time_phi_start = time.time()
        phi = {}
        self.solver.compute_concentrations()
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.solver.get_total_concentration(monomer_type)
        elapsed_time["phi"] = time.time() - time_phi_start

        return phi, elapsed_time

    # ------------------------------------------------------------------ #
    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev):
        """Mainline checkpoint plus the psi field and its LM noise."""
        super().save_simulation_data(path, w, phi, langevin_step, normal_noise_prev)
        data = loadmat(path, squeeze_me=False)
        data["psi_real"] = np.real(self.psi)
        data["psi_imag"] = np.imag(self.psi)
        savemat(path, data, long_field_names=True, do_compression=True)

    def continue_run(self, file_name):
        """Continue from a checkpoint, restoring psi if present."""
        data = loadmat(file_name, squeeze_me=True)
        if "psi_real" in data:
            self.psi = np.asarray(data["psi_real"]).reshape(-1) \
                + 1j * np.asarray(data["psi_imag"]).reshape(-1)
        else:
            print("Note: checkpoint has no psi field; psi restarts from zero.")
        super().continue_run(file_name)

    # ------------------------------------------------------------------ #
    def run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None):
        """CL loop with the psi field co-evolved (copy of CLFTS.run + psi).

        The psi update is inserted right after the w_aux update; the
        Hamiltonian gains the explicit -(1/2V) int c psi term; the printed
        error block gains std(lambda_psi) as the Poisson-residual monitor.
        """
        print("---------- Run (charged CL-FTS) ----------")

        M = len(self.monomer_types)

        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        w = np.array([np.reshape(initial_fields[m], self.cb.get_total_grid()).astype(np.complex128)
                      for m in self.monomer_types])
        w_aux = self.mpt.to_aux_fields(w)

        # Project out zero-eigenvalue modes: they carry no auxiliary field
        # (see __init__) and must be identically zero throughout the run.
        active = self._active_aux
        for i in self._frozen_aux:
            proj = float(np.max(np.abs(w_aux[i])))
            if proj > 1e-10:
                print(f"Warning: initial fields have a nonzero projection "
                      f"({proj:.3e}) onto the zero-eigenvalue aux mode {i}; "
                      f"projecting it out.")
            w_aux[i] = 0.0

        H_history = []
        dH_history = {key: [] for key in self.chi_n}

        sf_average = {}
        for monomer_id_pair in itertools.combinations_with_replacement(list(range(M)), 2):
            sorted_pair = sorted(monomer_id_pair)
            type_pair = self.monomer_types[sorted_pair[0]] + "," + self.monomer_types[sorted_pair[1]]
            sf_average[type_pair] = np.zeros_like(
                np.fft.fftn(np.reshape(w[0], self.cb.get_nx())), np.complex128)

        if normal_noise_prev is None:
            normal_noise_prev = np.zeros([M, self.cb.get_total_grid()], dtype=np.float64)
        if start_langevin_step is None:
            start_langevin_step = 1

        time_start = time.time()

        for langevin_step in range(start_langevin_step, self.langevin["max_step"] + 1):

            phi, _ = self.compute_concentrations(w_aux=w_aux)
            phi_for_force = self.smearing.apply_to_dict(phi)

            # Forces only for the active modes; frozen (zero-eigenvalue)
            # modes stay at zero. (Pre-patch failure: the parent __init__'s
            # dt_scaling is 0/0 = NaN for a zero mode, poisoning the first
            # w_aux update; compute_func_deriv itself is finite for them.)
            w_lambda = np.zeros((M, self.cb.get_total_grid()), dtype=np.complex128)
            w_lambda[active] = self.mpt.compute_func_deriv(
                w_aux, phi_for_force, active)

            if self.alpha_ds > 0:
                for i in active:
                    if i in self.mpt.aux_fields_real_idx:
                        w_lambda[i] += 1j * self.alpha_ds * np.imag(w_aux[i])

            # psi force with the CURRENT phi (same time slice as w_lambda)
            psi_lambda, charge_c = self._psi_force(phi)
            # Explicit electrostatic Hamiltonian term at THIS time slice
            # (pre-update psi, consistent with phi/Q): (1/2E V) int psi
            # lap(psi) = (1/2) mean(psi (lambda_psi - c)); reduces to
            # -(1/2) mean(c psi) at the Poisson saddle.
            h_elec = 0.5 * np.mean(self.psi * (psi_lambda - charge_c))

            # ---- update w_aux (identical to mainline CLFTS) ----
            normal_noise_current = self.random.normal(
                0.0, self.langevin["sigma"], [M, self.cb.get_total_grid()])
            for i in active:
                scaling = self.dt_scaling[i]
                if i in self.mpt.aux_fields_real_idx:
                    noise_factor = 1.0
                else:
                    noise_factor = 1j
                w_aux[i] += (-w_lambda[i] * self.langevin["dt"] * scaling +
                             0.5 * noise_factor * (normal_noise_prev[i] + normal_noise_current[i]) * np.sqrt(scaling))
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            # ---- update psi (imaginary-type: +lambda drift, i-noise) ----
            # The linear lap(psi)/E part (rate gamma_k = k^2/E) is stiff at
            # high k, and both explicit Euler (unstable) and semi-implicit
            # Euler (stationary variance suppressed by 1/(1+2 kappa dt) per
            # mode — ruins fluctuation thermodynamics) fail. Use ETD
            # (exact OU integration) per k-mode:
            #   psi_hat' = a psi_hat + (1-a) E c_hat/k^2 + i amp_k xi_hat,
            #   a = exp(-gamma_k dt s),
            #   amp_k = sigma sqrt((1-a^2)/(2 gamma_k dt))  (-> sigma
            #   sqrt(s) as gamma -> 0),
            # which reproduces the exact per-mode stationary variance of
            # the linear part for ANY dt. The c[psi] screening part of the
            # drift stays explicit (per-step rate <= sum z^2 phibar dt).
            if self.electro.e_coupling > 0.0:
                xi = self._random_psi.normal(
                    0.0, self.langevin["sigma"], self.cb.get_total_grid())
                psi_hat = np.fft.fftn(np.reshape(self.psi, self.electro.nx))
                c_hat = np.fft.fftn(np.reshape(charge_c, self.electro.nx))
                xi_hat = np.fft.fftn(np.reshape(xi, self.electro.nx))
                psi_hat = (self._psi_decay * psi_hat
                           + self._psi_drift * c_hat
                           + 1j * self._psi_namp * xi_hat)
                psi_hat[tuple([0] * self.electro.dim)] = 0.0   # gauge: k=0
                self.psi = np.ascontiguousarray(np.fft.ifftn(psi_hat).reshape(-1))
            # E = 0: no Coulomb interaction — psi has infinite stiffness
            # (k^2/E -> inf) and stays pinned at zero.

            # ---- monitoring ----
            h_deriv = np.zeros((M, self.cb.get_total_grid()), dtype=np.complex128)
            h_deriv[active] = self.mpt.compute_func_deriv(
                w_aux, phi_for_force, active)
            error_level_array = np.std(h_deriv, axis=1)
            psi_error = float(np.std(psi_lambda))

            total_partitions = [self.solver.get_total_partition(p)
                                for p in range(self.molecules.get_n_polymer_types())]
            hamiltonian = self.mpt.compute_hamiltonian(
                self.molecules, w_aux, total_partitions, self.cb, include_const_term=True)
            hamiltonian += h_elec

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

            if langevin_step % self.recording["sf_computing_period"] == 0:
                H_history.append(hamiltonian)
                dH = self.mpt.compute_h_deriv_chin(self.chi_n, w_aux)
                for key in self.chi_n:
                    dH_history[key].append(dH[key])

            if langevin_step % self.recording["sf_recording_period"] == 0:
                H_history_array = np.array(H_history)
                mdic = {"H_history_real": np.real(H_history_array),
                        "H_history_imag": np.imag(H_history_array)}
                for key in self.chi_n:
                    dH_array = np.array(dH_history[key])
                    monomer_pair = sorted(key.split(","))
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1] + "_real"] = np.real(dH_array)
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1] + "_imag"] = np.imag(dH_array)
                savemat(os.path.join(self.recording["dir"], "dH_%06d.mat" % langevin_step),
                        mdic, long_field_names=True, do_compression=True)
                H_history = []
                for key in self.chi_n:
                    dH_history[key] = []

            if langevin_step % self.recording["sf_computing_period"] == 0:
                mu_fourier = {}
                phi_fourier = {}
                for i in range(M):
                    key = self.monomer_types[i]
                    phi_fourier[key] = np.fft.fftn(
                        np.reshape(phi[self.monomer_types[i]], self.cb.get_nx())
                    ) / self.cb.get_total_grid()
                    mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
                    for k in range(M - 1):
                        # w/lambda is undefined for a zero-eigenvalue mode
                        # (w_aux[k] is identically zero anyway) — skip it
                        # instead of writing 0/0 = NaN into every S(k).
                        if k in self._frozen_aux:
                            continue
                        mu_fourier[key] += (
                            np.fft.fftn(np.reshape(w_aux[k], self.cb.get_nx())) *
                            self.mpt.matrix_a_inv[k, i] / self.mpt.eigenvalues[k] / self.cb.get_total_grid())
                for key in sf_average:
                    monomer_pair = sorted(key.split(","))
                    sf_average[key] += mu_fourier[monomer_pair[0]] * np.conj(phi_fourier[monomer_pair[1]])

            if langevin_step % self.recording["sf_recording_period"] == 0:
                chi_n_mat = {}
                for key in self.chi_n:
                    monomer_pair = sorted(key.split(","))
                    chi_n_mat[monomer_pair[0] + "," + monomer_pair[1]] = self.chi_n[key]
                mdic = {"dim": self.cb.get_dim(), "nx": self.cb.get_nx(), "lx": self.cb.get_lx(),
                        "chi_n": chi_n_mat, "chain_model": self.chain_model, "ds": self.ds,
                        "dt": self.langevin["dt"], "nbar": self.langevin["nbar"],
                        "initial_params": self.params}
                for key in sf_average:
                    sf_scaled = (sf_average[key] *
                                 self.recording["sf_computing_period"] / self.recording["sf_recording_period"] *
                                 self.cb.get_volume() * np.sqrt(self.langevin["nbar"]))
                    monomer_pair = sorted(key.split(","))
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1] + "_real"] = np.real(sf_scaled)
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1] + "_imag"] = np.imag(sf_scaled)
                savemat(os.path.join(self.recording["dir"], "structure_function_%06d.mat" % langevin_step),
                        mdic, long_field_names=True, do_compression=True)
                for key in sf_average:
                    sf_average[key][:] = 0.0

            if langevin_step % self.recording["recording_period"] == 0:
                w = self.mpt.to_monomer_fields(w_aux)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % langevin_step),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev)

        time_duration = time.time() - time_start
        print("total time: %f, time per step: %f" %
              (time_duration, time_duration / (langevin_step + 1 - start_langevin_step)))
