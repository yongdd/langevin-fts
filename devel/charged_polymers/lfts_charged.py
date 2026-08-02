"""Charged-polymer L-FTS prototype (devel only — mainline is untouched).

Subclass of polymerfts.lfts.LFTS adding the smeared-charge compressible
model of THEORY.md: per-species Gaussian smearing h_i (radius a_i), a
Coulomb potential psi solved at its PARTIAL SADDLE inside the saddle loop
(screening-preconditioned Newton update), and propagator inputs

    W_i = h_i * (W_i^SPT + z_i psi).

IMPORTANT SCOPE NOTE: keeping psi at its partial saddle is a mean-field
(Poisson-Boltzmann-level) treatment of the electrostatics — L-FTS only
samples the real exchange fields, so psi FLUCTUATIONS (true charge
correlation / self-energy effects beyond DH screening) are NOT captured
here. The eventual home of the fluctuating-psi theory is CL-FTS
(src/python/clfts.py), where psi acquires complex Langevin dynamics. This
prototype exists to develop and validate the smearing/Poisson/coupling
machinery (see test_neutral_reduction.py, test_debye_huckel.py) before
that integration.

Activated by the "charges" parameter; requires the compressible model
("zeta_n"), per-species "radiuses", and "bjerrum_e" (= E) or
"bjerrum_length" (= l_B/R0). Use the AM compressor (the LR/LRAM Hessian
is built during mainline __init__, before the charged smearing adapter is
installed).

Known limitations (deliberate, documented in THEORY.md):
- psi is not checkpointed (restarts re-converge it to tolerance).
- psi is not rolled back when a saddle solve is rejected in run()
  (harmless: its fixed point depends only on phi).
- No random copolymers; no Maxwell stress / box_is_altering.
"""
import time
import logging

import numpy as np

from polymerfts import lfts
from polymerfts.validation import ValidationError

from electrostatics import Electrostatics

logger = logging.getLogger(__name__)


def _drop_none_values(obj):
    """Recursively remove None-valued entries from dicts/lists.

    scipy.io.savemat cannot represent None (e.g. "charges": {"S": None});
    an absent key carries the same meaning, so drop them from the saved
    copy of the parameter dictionary.
    """
    if isinstance(obj, dict):
        return {k: _drop_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_drop_none_values(v) for v in obj if v is not None]
    return obj


class _PerSpeciesSmearing:
    """Adapter exposing Electrostatics.smear_dict through the Smearing API.

    Installed as self.smearing so every mainline call site
    (phi_for_force in run(), phi_for_saddle, ...) applies the per-species
    h_i instead of a global kernel. Only the phi -> smeared-phi direction
    goes through here; the propagator-input assembly (which also needs
    z_i psi) is handled in ChargedLFTS.compute_concentrations.
    """

    def __init__(self, electro):
        self._electro = electro
        self.enabled = True

    def apply_to_dict(self, fields):
        return self._electro.smear_dict(fields)

    def apply(self, field):
        raise NotImplementedError(
            "Per-species smearing has no species-agnostic apply(); "
            "use apply_to_dict.")


class ChargedLFTS(lfts.LFTS):
    """L-FTS with smeared charges and a partial-saddled Coulomb psi."""

    def __init__(self, params, random_seed=None):
        if params.get("charges", None) is None:
            raise ValidationError("ChargedLFTS requires the 'charges' parameter.")
        if params.get("zeta_n", None) is None:
            raise ValidationError(
                "Charged-polymer mode requires the COMPRESSIBLE model: "
                "set 'zeta_n'. (The incompressible charged theory is not "
                "implemented; see THEORY.md.)")
        if params.get("smearing", None) is not None:
            raise ValidationError(
                "Do not combine 'smearing' with 'charges': charged mode "
                "uses per-species smearing via 'radiuses'. Remove the "
                "global 'smearing' parameter.")
        if "radiuses" not in params:
            raise ValidationError(
                "Charged-polymer mode requires 'radiuses' (per-species "
                "smearing lengths / Born radii, in R0 units).")

        super().__init__(params=params, random_seed=random_seed)

        if len(self.random_fraction) > 0:
            raise NotImplementedError(
                "Charged-polymer mode does not support random copolymers yet.")

        # Reject typo'd species names (they would silently make a species
        # neutral/unsmeared) and require a positive smearing radius for
        # every species: the fluctuating compressible model is UV-singular
        # without smearing.
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

        # Coulomb coupling E = 4 pi l_B rho_0 N R0^2 = sqrt(nbar) * 4 pi l_B N^2 / R0
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

        # Overall segment volume fraction per monomer type, for the
        # electroneutrality check and the psi-update preconditioner:
        # sum_p vf_p * len_type / alpha_p.
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

        # Route every mainline phi-smearing call site through the
        # per-species kernels.
        self.smearing = _PerSpeciesSmearing(self.electro)

        # scipy savemat crashes on None values; save a sanitized copy.
        self.params = _drop_none_values(self.params)

    # ------------------------------------------------------------------ #
    def compute_concentrations(self, w_aux):
        """Charged version: propagator inputs W_i = h_i*(W_SPT + z_i psi).

        Same contract as LFTS.compute_concentrations (takes AUXILIARY
        fields, returns (phi, elapsed_time)).
        """
        M = len(self.monomer_types)
        elapsed_time = {}

        # Convert auxiliary fields to monomer fields
        w = self.mpt.to_monomer_fields(w_aux)
        w_input = {self.monomer_types[i]: w[i] for i in range(M)}

        # Assemble propagator inputs: add the electrostatic one-body term
        # BEFORE the outer smearing, then smear per species.
        w_input_for_propagator = {
            t: self.electro.smear(t, w_input[t] + self.electro.charge_potential(t))
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

        # Runtime guardrails (same as mainline)
        time_validation_start = time.time()
        validation_config = self.validation_runtime.copy() if self.validation_runtime else {}
        if self._partition_checked:
            validation_config["partition_check"] = False
        from polymerfts.validation import validate_runtime_state
        runtime_metrics = validate_runtime_state(
            prop_solver=self.prop_solver,
            phi=phi,
            monomer_types=self.monomer_types,
            numerical_method=self.prop_solver.numerical_method,
            user_config=validation_config,
        )
        if not self._partition_checked:
            self._partition_checked = True
        elapsed_time["validation"] = time.time() - time_validation_start
        elapsed_time.update(runtime_metrics)

        return phi, elapsed_time

    # ------------------------------------------------------------------ #
    def find_saddle_point(self, w_aux):
        """Charged version: joint saddle of the SPT imaginary fields (AM)
        and psi (screening-preconditioned Newton, exact Poisson fixed
        point). The Poisson residual enters the stopping criterion; psi is
        NOT part of the compressor state. Returns (phi, hamiltonian,
        saddle_iter, error_level) like the mainline method (and, unlike
        current mainline, also assigns a valid hamiltonian when
        verbose_level == 0).
        """
        M = len(self.monomer_types)
        I = len(self.mpt.aux_fields_imag_idx)

        error_level = 1e20
        hamiltonian = None

        self.compressor.reset_count()

        for saddle_iter in range(1, self.saddle["max_iter"]+1):

            phi, runtime_info = self.compute_concentrations(w_aux)
            if runtime_info.get("partition_ok", 1.0) < 0.5:
                logger.warning(
                    "Partition consistency check reported a mismatch during "
                    "saddle iteration %d (method=%s).",
                    saddle_iter, self.prop_solver.numerical_method)

            # Functional derivatives w.r.t. imaginary fields use the
            # smeared densities (chain rule of the smeared propagator inputs).
            phi_for_saddle = self.electro.smear_dict(phi)
            h_deriv = self.mpt.compute_func_deriv(
                w_aux, phi_for_saddle, self.mpt.aux_fields_imag_idx)

            # psi partial saddle: measure the Poisson residual with the
            # CURRENT psi (convergence indicator), then take one
            # preconditioned Newton step toward -lap(psi) = E c.
            charge_c = self.electro.charge_density(phi)
            psi_error = float(np.std(self.electro.residual(charge_c)))
            self.electro.solve_psi(charge_c)

            old_error_level = error_level
            error_level_array = np.std(h_deriv, axis=1)
            error_level = max(np.max(error_level_array), psi_error)

            # Hamiltonian BEFORE any compressor update so the returned
            # value is consistent with the current propagators.
            if (self.verbose_level == 2
                    or error_level < self.saddle["tolerance"]
                    or saddle_iter == self.saddle["max_iter"]):
                hamiltonian = self._compute_hamiltonian_charged(w_aux, charge_c)

            if(self.verbose_level == 2 or self.verbose_level == 1 and
            (error_level < self.saddle["tolerance"] or saddle_iter == self.saddle["max_iter"])):
                total_partitions = [self.prop_solver.get_partition_function(p)
                                    for p in range(self.prop_solver.get_n_polymer_types())]
                mass_error = self.cb.mean(h_deriv[I-1])
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for Q in total_partitions:
                    print("%13.7E " % (Q), end=" ")
                print("] %15.9f   [" % (hamiltonian), end="")
                for i in range(I):
                    print("%13.7E" % (error_level_array[i]), end=" ")
                print("%13.7E" % (psi_error), end=" ")
                print("]")

            if error_level < self.saddle["tolerance"]:
                break

            for count, i in enumerate(self.mpt.aux_fields_imag_idx):
                h_deriv[count] *= self.dt_scaling[i]

            w_aux[self.mpt.aux_fields_imag_idx] = np.reshape(
                self.compressor.calculate_new_fields(
                    w_aux[self.mpt.aux_fields_imag_idx], -h_deriv,
                    old_error_level, error_level),
                [I, self.prop_solver.n_grid])

        # Pressure-field mean fixing (same as mainline; charged mode is
        # always compressible).
        if hasattr(self, '_target_mean_wp'):
            w_aux[M-1] -= (self.cb.mean(w_aux[M-1]) - self._target_mean_wp)

        return phi, hamiltonian, saddle_iter, error_level

    # ------------------------------------------------------------------ #
    def _compute_hamiltonian_charged(self, w_aux, charge_c):
        """SPT Hamiltonian + explicit electrostatic term -(1/2V) int c psi."""
        total_partitions = [self.prop_solver.get_partition_function(p)
                            for p in range(self.prop_solver.get_n_polymer_types())]
        hamiltonian = self.mpt.compute_hamiltonian(
            self.prop_solver._molecules, w_aux, total_partitions, self.cb,
            include_const_term=True)
        hamiltonian += self.electro.hamiltonian_per_chain(charge_c)
        return hamiltonian
