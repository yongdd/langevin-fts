"""Complex Langevin field-theoretic simulation (CL-FTS).

All auxiliary fields are complex and fluctuate: real noise on the exchange
fields, imaginary noise on the pressure field, with optional dynamical
stabilization (Willis & Matsen, J. Chem. Phys. 161, 244903 (2024)),
smearing (Matsen et al., J. Chem. Phys. 164, 014905 (2026)) and smeared
Coulomb interactions through a fluctuating electrostatic potential
(electrostatics.py). Parameters are documented in
docs/references/Parameters.md; complete scripts are examples/block_copolymer/clfts/Lamella.py,
examples/charged/clfts/ and examples/blend/clfts/GibbsEnsemble.py.
"""

import os
import time
import pathlib
import warnings
import contextlib
import copy
import json
import numpy as np
from .integrators import checked_fields
import itertools

from scipy.io import loadmat

from .polymer_field_theory import SymmetricPolymerTheory
from .smearing import Smearing
from .electrostatics import Electrostatics
from .propagator_solver import PropagatorSolver
from .validation import (normalize_params, validate_lfts_params, ValidationError, parse_chi_n, point_ion_concentrations,
                         open_charges)
from .utils import (
    process_polymer_blocks,
    process_random_copolymer,
    draw_polymer_architectures,
    report_partition_consistency,
    calculate_sigma,
    load_config,
    atomic_savemat,
    mat_ready,
    simulation_metadata,
    checkpoint_data,
    validate_checkpoint,
    restore_recording_state,
    write_run_status,
    fields_hash,
    initialize_extensions,
    extension_state,
    restore_extensions,
    notify_callbacks,
    DriftTailSummary,
    PartitionRatioSummary,
)

# OpenMP environment variables
os.environ.setdefault("OMP_NUM_THREADS", "1")  # respect user override
os.environ.setdefault("OMP_STACKSIZE", "1G")


class CLFTS:
    """Complex Langevin field-theoretic simulation.

    Parameters
    ----------
    params : dict
        Simulation parameters; see docs/references/Parameters.md (CL-FTS
        section). Required: the SCFT system keys plus langevin, recording,
        verbose_level. zeta_n defaults to 100.
    random_seed : int, optional
        Seed of the PCG64 noise generator (None: non-reproducible).
    integrator : object, optional
        Langevin step implementation; otherwise langevin.integrator (default "lm").
        See docs/internals/Extensions.md for the step and checkpoint contracts.
    callbacks : dict, optional
        Stable names to callable observers of accepted, read-only step snapshots.

    Attributes
    ----------
    prop_solver : PropagatorSolver
        Complex-field propagator computation (C++ objects inside).
    cb : ComputationBox
        Grid, box and integrals (C++).
    mpt : SymmetricPolymerTheory
        Monomer <-> auxiliary field transformations.
    alpha_ds : float
        Dynamical stabilization constant.

    Examples
    --------
    >>> sim = CLFTS(params)
    >>> sim.run(initial_fields={"A": w_A, "B": w_B})
    """

    @classmethod
    def from_file(cls, path, random_seed=None, *, integrator=None, callbacks=None, **overrides):
        """Create a simulation from a YAML or JSON parameter file.

        Reads the file with utils.load_config, applies the keyword overrides
        and calls the constructor, as SCFT.from_file does.
        """
        params = normalize_params(load_config(path))
        params.update(normalize_params(overrides))
        return cls(params, random_seed=random_seed, integrator=integrator, callbacks=callbacks)

    def __init__(self, params, random_seed=None, *, integrator=None, callbacks=None):
        """Initialize the Complex Langevin FTS simulation."""

        # Validate input parameters (same checks as SCFT and LFTS)
        params = normalize_params(params)
        validate_lfts_params(params, complex_fields=True)
        for key in ("langevin", "recording", "verbose_level"):
            if key not in params:
                raise ValidationError(f"Missing required parameter for CLFTS: '{key}'")
        if params["verbose_level"] not in (0, 1, 2):
            raise ValidationError(f"'verbose_level' must be 0, 1 or 2, got {params['verbose_level']}")

        # Monomer types and statistical segment lengths
        self.monomer_types = sorted(list(params["segment_lengths"].keys()))
        self.segment_lengths = copy.deepcopy(params["segment_lengths"])
        self.polymers = copy.deepcopy(params["polymers"])

        # Flory-Huggins parameters chi N; pairs the user did not give do not interact
        self.chi_n = parse_chi_n(params["chi_n"], self.segment_lengths)
        for pair in itertools.combinations(self.monomer_types, 2):
            key = ",".join(sorted(pair))
            if key not in self.chi_n:
                self.chi_n[key] = 0.0

        # Compressibility parameter; complex Langevin needs a finite one (default: high compressibility)
        self.zeta_n = 100.0 if params.get("zeta_n", None) is None else params["zeta_n"]

        # Polymer chains: block connectivity, then random copolymers (one effective monomer type each)
        for polymer in self.polymers:
            polymer["blocks_input"] = process_polymer_blocks(polymer)
        self.random_fraction = {}
        for polymer in self.polymers:
            result = process_random_copolymer(polymer, self.segment_lengths)
            if result is not None:
                random_type, effective_segment_length, fraction = result
                polymer["block_monomer_types"] = [random_type]
                self.segment_lengths[random_type] = effective_segment_length
                self.random_fraction[random_type] = fraction

        # Boundary conditions (default: periodic)
        bc = params.get("bc", None)
        if bc is None:
            bc = ["periodic"] * (2 * len(params["nx"]))

        # Multimonomer polymer field theory
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n, self.zeta_n)

        # The number of monomer types
        M = len(self.monomer_types)

        # Complex-valued propagator solver (complex Langevin fields)
        self.prop_solver = PropagatorSolver(
            nx=params["nx"], lx=params["lx"], ds=params["ds"],
            bond_lengths=self.segment_lengths, bc=bc,
            chain_model=params["chain_model"],
            numerical_method=params.get("numerical_method", None),   # None: "rqm4"
            platform=params.get("platform", "auto"),   # auto: CPU for 1D, CUDA for 2D/3D when available
            reduce_memory=params.get("reduce_memory", False),
            angles=params.get("angles", None),
            aggregate_propagator_computation=params.get("aggregate_propagator_computation", True),
            field_type="complex",
        )
        # Grand-canonical polymers (chemical_potential βμ_p) get the volume fraction 0: their chains,
        # e^(βμ_p) Q_p per reference chain, come from get_concentration_gce (compute_concentrations)
        for polymer in self.polymers:
            self.prop_solver.add_polymer(polymer.get("volume_fraction", 0.0), polymer["blocks_input"],
                                         polymer.get("initial_conditions", None))
        self.polymer_chemical_potentials = np.array([polymer.get("chemical_potential", np.nan) for polymer in self.polymers])
        self.fugacities = np.where(np.isnan(self.polymer_chemical_potentials), 0.0, np.exp(self.polymer_chemical_potentials))
        self.grand_canonical = [p for p, polymer in enumerate(self.polymers) if "chemical_potential" in polymer]
        self.canonical_fraction = sum(polymer.get("volume_fraction", 0.0) for polymer in self.polymers)
        self.prop_solver.display_platform_info()

        # (C++ class) Computation box; molecules, optimizer and propagator computation
        # are used through prop_solver (aliases at the end of __init__ for scripts
        # that reach the C++ objects directly)
        self.cb = self.prop_solver.computation_box

        # Standard deviation of normal noise of Langevin dynamics
        langevin_sigma = calculate_sigma(
            params["langevin"]["nbar"],
            params["langevin"]["dt"],
            np.prod(params["nx"]),
            self.prop_solver.get_volume()
        )

        # Dynamical stabilization constant for complex Langevin simulations
        # This prevents runaway solutions ("hot spots") by adding iαds·Im[W-] to exchange field forces.
        # "dynamic_stabilization" is the combined coefficient αds (as in Willis & Matsen 2024);
        # normalize_params has renamed its alias "alpha_ds", kept for compatibility with reference codes.
        # Typical values: 0.01-0.1 for N̄ ≥ 10^5 (Willis & Matsen 2024)
        self.alpha_ds = params.get("dynamic_stabilization", 0.0)

        # dH/dw_aux[i] is scaled by dt_scaling[i]
        self.dt_scaling = np.ones(M)
        for i in range(M - 1):
            if i not in self.mpt.aux_fields_dropped_idx:
                self.dt_scaling[i] = np.abs(self.mpt.eigenvalues[i]) / np.max(np.abs(self.mpt.eigenvalues[:M-1]))
        self.dt_scaling[self.mpt.aux_fields_dropped_idx] = 0.0

        # Set random generator
        if random_seed is None:
            self.random_bg = np.random.PCG64()
        else:
            self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)

        print("---------- Simulation Parameters ----------")
        print("Platform:", self.prop_solver.platform)
        print("Box Dimension: %d" % (self.prop_solver.dim))
        print("Nx:", self.prop_solver.nx)
        print("Lx [a, b, c]:", self.prop_solver.get_lx())
        print("Angles [alpha, beta, gamma]:", list(self.prop_solver.get_angles_degrees()))
        print("dx:", self.prop_solver.get_dx())
        print("Volume: %f" % (self.prop_solver.get_volume()))

        print("Chain model: %s" % (params["chain_model"]))
        print("Segment lengths:\n\t", list(self.segment_lengths.items()))
        print("Conformational asymmetry (epsilon):")
        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1],
                                   self.segment_lengths[monomer_pair[0]] / self.segment_lengths[monomer_pair[1]]))

        print("χN:")
        for key in self.chi_n:
            print("\t%s: %f" % (key, self.chi_n[key]))

        print("zeta_n: %f" % (self.zeta_n))

        self.prop_solver.display_architectures()

        print("Invariant Polymerization Index (N_Ref): %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Scaling factor of delta tau N for each field:", self.dt_scaling)
        print("Dynamical stabilization constant: %f" % (self.alpha_ds))
        print("Random Number Generator:", self.random_bg.state)

        self.prop_solver.display_statistics()

        # Save internal variables
        self.params = copy.deepcopy(params)
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma": langevin_sigma})

        self.verbose_level = params["verbose_level"]
        self.recording = params["recording"].copy()

        # Aliases of the C++ objects inside prop_solver (for scripts that use them directly)
        self.solver = self.prop_solver.propagator_computation
        self.molecules = self.prop_solver.molecules
        self.propagator_computation_optimizer = self.prop_solver.propagator_optimizer

        # Initialize smearing (must be after self.cb is set)
        # Γ(k) is applied with the platform FFT (PropagatorSolver.fourier), except in a box
        # that mixes periodic and wall axes, which smearing transforms with NumPy/SciPy
        periodic_sides = [b.lower() == "periodic" for b in self.prop_solver.bc]
        mixed_box = any(periodic_sides) and not all(periodic_sides)
        self._periodic_box = all(periodic_sides)   # the structure functions transform with fftn
        self.smearing = Smearing(
            self.prop_solver.nx,
            self.prop_solver.get_lx(),
            params.get("smearing", None),
            angles=self.prop_solver.get_angles_degrees(),
            bc=self.prop_solver.bc,
            transform=self.prop_solver.fourier if params.get("smearing") is not None and not mixed_box else None
        )

        # Smeared Coulomb interactions (optional): the electrostatic potential ψ, felt by
        # charged types as z_i ψ, fluctuates next to the auxiliary fields (electrostatics.py)
        self.electrostatics = None
        if params.get("electrostatics", None) is not None:
            # Segments N of the reference chain: 1/ds of discrete chains, given for continuous chains
            n_segments = (round(1.0 / params["ds"]) if params["chain_model"].lower() == "discrete"
                          else params["electrostatics"].get("n_segments"))
            self.electrostatics = Electrostatics(params["electrostatics"], self.monomer_types, self.smearing,
                                                 n_segments, self.langevin, self.random_bg.jumped(),
                                                 fourier=self.prop_solver.fourier,
                                                 ion_concentrations=point_ion_concentrations(params),
                                                 open_charges=open_charges(params))
            print("Electrostatics: charges", self.electrostatics.charges,
                  "coupling E = %g, psi_dt_scaling = %g" % (self.electrostatics.coupling, self.electrostatics.psi_dt_scaling))
            for ion in self.electrostatics.ions:
                amount = ("chemical potential %g (grand canonical)" % ion["chemical_potential"] if ion["concentration"] is None
                          else "concentration %g (with counterions)" % ion["concentration"])
                print("Point ion %s: valence %g, %s, smearing length %g"
                      % (ion["name"], ion["valence"], amount, ion["smearing_length"]))
        for p in self.grand_canonical:
            print("Polymer %d: grand canonical, chemical potential %g" % (p, self.polymer_chemical_potentials[p]))

        # Structure functions: composition modes with a zero eigenvalue carry no auxiliary field,
        # but ψ measures the charge direction z, <((k² + κ²)/E) ψ(k) φ_j(-k)> = Σ_i z_i <Γ_i φ_i φ_j>.
        # Its part outside the eigenvectors of the auxiliary fields, u = P_0 z (P_0: projector onto
        # the zero-eigenvalue modes), completes the estimator (docs/theory/StructureFunctionEstimators.md,
        # Section 5).
        # The estimators with ψ need a periodic box: with walls ψ lives in the cosine basis, while the
        # structure functions are Fourier transforms
        self._psi_probe = None
        if self.electrostatics is not None and self.electrostatics.coupling > 0 and self._periodic_box:
            kept = [k for k in range(M) if not np.isclose(self.mpt.eigenvalues[k], 0.0)]
            projector = np.eye(M) - sum(np.outer(self.mpt.matrix_a[:, k], self.mpt.matrix_a[:, k]) for k in kept) / M
            u = projector @ np.array([self.electrostatics.charges[name] for name in self.monomer_types])
            if u @ u > 1e-12:
                self._psi_probe = u / (u @ u)

        # Thermodynamic operators, recorded with H: the chemical potential of each species,
        # βμ_p = ln(φ_p/α_p) - ln Q_p per chain (validation rejects grafted chains), and with
        # recording.pressure the osmotic pressure (pressure_virial())
        chains = [self.prop_solver.get_polymer(p) for p in range(self.prop_solver.get_n_polymer_types())]
        self.chain_densities = np.array([chain.get_volume_fraction() / chain.get_alpha() for chain in chains])
        # One chemical potential per polymer, then one per point ion; with grand-canonical species the
        # number of chains or ions per reference chain of every species is recorded too
        self.n_chemical_potentials = len(self.chain_densities) + (len(self.electrostatics.ions) if self.electrostatics else 0)
        self.record_number_densities = bool(self.grand_canonical) or (
            self.electrostatics is not None and any(ion["concentration"] is None for ion in self.electrostatics.ions))
        self.record_chemical_potentials = not any(polymer.get("initial_conditions") for polymer in self.polymers)
        self.record_pressure = bool(self.recording.get("pressure", False))
        if self.record_pressure:
            if not self.record_chemical_potentials:
                raise ValidationError("'recording.pressure' needs free chains (no initial_conditions)")
            if any(b != "periodic" for b in self.prop_solver.bc):
                raise ValidationError("'recording.pressure' needs a periodic box")
            # The O(ds²) contour error of rk2 in the stress, and so in the pressure, is large where one
            # contour step b √ds spans more than the smearing length, the shortest length on which the
            # smeared fields vary (docs/theory/ComplexLangevinSimulation.md, Section 7); without smearing
            # they vary down to the grid. The types are those of the blocks, so a random block counts
            # with its effective segment length
            if self.chain_model.lower() == "continuous" and self.prop_solver.numerical_method == "rk2":
                coarse = []
                for name in sorted({block["type"] for polymer in self.polymers for block in polymer["blocks"]}):
                    if not self.smearing.enabled:
                        length = 0.0
                    elif self.smearing.type == "gaussian":
                        a_int = self.smearing.params["a_int"]
                        length = a_int.get(name, 0.0) if isinstance(a_int, dict) else a_int
                    else:
                        length = 1.0 / self.smearing.params["k_int"]
                    step = self.segment_lengths[name] * np.sqrt(self.ds)
                    if step > length:
                        coarse.append(f"{name}: step {step:.3g} > smearing length {length:.3g}")
                if coarse:
                    warnings.warn("With rk2 the recorded pressure carries an O(ds^2) contour error that is large "
                                  "when one contour step b sqrt(ds) exceeds the smearing length ("
                                  + ", ".join(coarse) + ", in R0); use numerical_method 'rqm4' or a smaller ds")
        # Canonical single-site species, discrete chains of one segment and point ions, whose ratio
        # |Q_s|/<|e^(-u_s)|> is recorded every step (partition_ratios())
        self._single_site_polymers = [
            p for p, polymer in enumerate(self.polymers)
            if self.chain_model.lower() == "discrete" and p not in self.grand_canonical
            and not polymer.get("initial_conditions") and len(polymer["blocks"]) == 1
            and polymer["blocks"][0]["type"] in self.monomer_types
            and np.isclose(polymer["blocks"][0]["length"], self.ds)]
        self.partition_ratio_species = [f"polymer {p} ({self.polymers[p]['blocks'][0]['type']})"
                                        for p in self._single_site_polymers]
        if self.electrostatics is not None:
            self.partition_ratio_species += [f"ion {ion['name']}" for ion in self.electrostatics.ions
                                             if ion["concentration"] is not None]
        self.metadata = simulation_metadata(self, random_seed)
        initialize_extensions(self, integrator if integrator is not None else self.langevin.get("integrator", "lm"), callbacks,
                              capture_initial_state=False)
        # ψ takes the noise scheme the integrator declares (electrostatics.py): the average of
        # two noises like Leimkuhler-Matthews, or fresh noise like Euler
        if self.electrostatics is not None:
            averaged_noise = getattr(self.integrator, "averaged_noise", None)
            if not isinstance(averaged_noise, (bool, np.bool_)):
                raise ValidationError("With 'electrostatics' the integrator must declare averaged_noise: True if "
                                      "it averages the current and the previous noise (Leimkuhler-Matthews), "
                                      "False if it takes fresh noise")
            self.electrostatics.averaged_noise = bool(averaged_noise)

        # An integrator that declares linear_response (the built-in "etd") integrates the linear
        # response of the homogeneous state exactly; it gets the Jacobian of the forces and, with
        # Coulomb coupling, also advances ψ as the last field. It holds the response of this
        # simulation, so each simulation keeps its own copy of the integrator
        self._joint_psi = False
        if getattr(self.integrator, "linear_response", False):
            if any(b != "periodic" for b in self.prop_solver.bc):
                raise ValidationError(f"The integrator '{self.integrator.name}' (linear_response) needs a periodic box")
            self.integrator = copy.deepcopy(self.integrator)
            self._joint_psi = self.electrostatics is not None and self.electrostatics.coupling > 0
            self.integrator.set_linear_response(self._symmetric_jacobian(self._force_jacobian()), dt=self.langevin["dt"],
                                                mobility=self._integrated_mobility(),
                                                noise_direction=self._integrated_noise_direction(),
                                                fourier=self.prop_solver.fourier)
        # A stateful method may save its prepared response. A fresh run must
        # restore the completed setup, not the state before preparation.
        self._initial_extension_state = copy.deepcopy(extension_state(self))
        self.run_status = {"status": "initialized", "last_completed_step": 0}
        # run() draws the chain architectures into the recording directory; a driver that runs the
        # same chains again (gibbs_ensemble.py) sets this to False after the first run
        self.draw_architectures = True

    def _fft_fields(self, fields):
        """fftn of each field (n_fields, n_grid) on the grid: with the platform FFT
        (PropagatorSolver.fourier) in a periodic box, with numpy.fft otherwise."""
        fields = np.reshape(fields, (-1, self.prop_solver.n_grid))
        if self._periodic_box:
            return self.prop_solver.fourier.forward(fields)
        nx = tuple(self.prop_solver.nx)
        return np.fft.fftn(np.reshape(fields, (-1,) + nx), axes=tuple(range(1, len(nx) + 1)))

    def _integrated_mobility(self):
        """Mobilities of the fields the integrator advances: dt_scaling, and that of ψ when joint."""
        if self._joint_psi:
            return np.append(self.dt_scaling, self.electrostatics.psi_dt_scaling)
        return self.dt_scaling.copy()

    def _integrated_noise_direction(self):
        """Real noise for the real-type auxiliary fields, imaginary for the others (and ψ)."""
        direction = np.array([1.0 if i in self.mpt.aux_fields_real_idx else 1j for i in range(len(self.monomer_types))])
        return np.append(direction, 1j) if self._joint_psi else direction

    def _force_jacobian(self):
        """Jacobian J(k) = dF/dw of the forces of the integrated fields at the homogeneous state.

        The fields are the auxiliary fields and, when the integrator advances it, ψ with the
        force -δH/δψ, so that every field steps as w - δt s F. A kick iε at one grid point of
        field j has the same Fourier amplitude iε at every wavevector; the concentrations
        respond linearly and in a homogeneous periodic box do not mix wavevectors, so the
        imaginary part of the change of the forces divided by ε gives column j of J(k) at all k
        at once (complex-step derivative, error O(ε²)). The forces of the unkicked state are
        subtracted: transforms of real data, for example FFTs of sizes with large prime factors,
        leave imaginary rounding errors of order 1e-15 that a division by ε alone would amplify.
        With ε = 1e-6 the error O(ε²) is about 1e-12 and the rounding error of J below 1e-7 of its
        size, even on large grids. The
        homogeneous state is w = 0 with ψ = 0: a uniform shift of the fields does not change
        the concentrations of canonical chains. Grand-canonical chains have the density
        e^(βμ_p) α_p there, not that of the homogeneous saddle point; J only sets the linear
        part that ETD integrates exactly. Dynamical stabilization is not part of J.
        """
        M, n_grid, nx = len(self.monomer_types), self.prop_solver.n_grid, self.prop_solver.nx
        es = self.electrostatics
        m = M + int(self._joint_psi)
        epsilon = 1e-6

        def forces(w_aux, psi):
            with es.trial(psi) if es is not None else contextlib.nullcontext():
                phi, _ = self.compute_concentrations(w_aux)
                phi_smeared = self.smearing.apply_to_dict(phi)
                force = self.mpt.compute_func_deriv(w_aux, phi_smeared, range(M))
                if self._joint_psi:
                    force = np.vstack([force, -es.charge_density(phi_smeared)])
            return force

        base = forces(np.zeros((M, n_grid), dtype=np.complex128), np.zeros(n_grid, dtype=np.complex128))
        jacobian = np.zeros((m, m) + tuple(nx), dtype=np.complex128)
        for j in range(m):
            if j < M and self.dt_scaling[j] == 0:
                continue
            w_aux = np.zeros((M, n_grid), dtype=np.complex128)
            psi = np.zeros(n_grid, dtype=np.complex128)
            if j < M:
                w_aux[j, 0] = 1j * epsilon
            else:
                psi[0] = 1j * epsilon
            change = forces(w_aux, psi) - base
            for i in range(m):
                jacobian[i, j] = np.fft.fftn(np.reshape(np.imag(change[i]) / epsilon, nx))
        if self._joint_psi:
            jacobian[M, M] += es.screened_k_sq / es.coupling
        return jacobian

    def _symmetric_jacobian(self, jacobian):
        """Symmetric part of the measured Jacobian in the variables w/n of the noise directions n.

        K = N⁻¹ J N holds second derivatives of H and is complex symmetric. The forces of
        continuous chains are the gradient of the discretized H only up to the contour error, so
        the measured K is symmetric only to O(ds²) (rk2 and rqm4 alike), notably where the
        segment lengths differ; the linear response takes its symmetric part. A relative
        asymmetry above 1e-2 among the fields that move means a far too coarse contour step or
        a wrong measurement. The diagonal (k² + κ²)/E of a jointly advanced ψ can exceed the
        auxiliary entries by orders of magnitude, so the auxiliary block is measured on its own
        scale and the coupling of ψ to it on the geometric mean of the auxiliary scale and the
        measured charge response of ψ (its diagonal without the analytic (k² + κ²)/E).
        """
        m, M = jacobian.shape[0], len(self.monomer_types)
        direction = self._integrated_noise_direction()
        ratio = (direction[None, :] / direction[:, None]).reshape((m, m) + (1,) * (jacobian.ndim - 2))
        K = jacobian * ratio
        aux = np.flatnonzero(self._integrated_mobility()[:M] > 0)
        K_aux = K[np.ix_(aux, aux)]
        aux_scale = max(np.max(np.abs(K_aux)), 1e-300) if aux.size else 1e-300
        asymmetry = np.max(np.abs(K_aux - np.swapaxes(K_aux, 0, 1))) / aux_scale if aux.size else 0.0
        if self._joint_psi and aux.size:
            es = self.electrostatics
            psi_scale = max(np.max(np.abs(K[M, M] - es.screened_k_sq / es.coupling)), 1e-300)
            asymmetry = max(asymmetry, np.max(np.abs(K[aux, M] - K[M, aux])) / np.sqrt(aux_scale * psi_scale))
        if asymmetry > 1e-2:
            raise ValidationError(f"The Jacobian of the forces is not symmetric (relative asymmetry {asymmetry:.1e}); "
                                  "with continuous chains use a smaller ds")
        if asymmetry > 1e-5:
            print(f"Jacobian of the forces: relative asymmetry {asymmetry:.1e} from the contour step, symmetrized")
        return 0.5 * (K + np.swapaxes(K, 0, 1)) / ratio

    def compute_concentrations(self, w_aux):
        """Compute monomer concentration fields from auxiliary fields.

        NOTE: this takes AUXILIARY fields (exchange-mapped), unlike
        SCFT.compute_concentrations which takes monomer potential fields.

        Parameters
        ----------
        w_aux : numpy.ndarray
            Complex auxiliary fields array of shape (M, n_grid).

        Returns
        -------
        phi : dict
            Dictionary of concentration fields for each monomer type.
        elapsed_time : dict
            Dictionary of timing information.
        """
        M = len(self.monomer_types)
        elapsed_time = {}

        # Convert auxiliary fields to monomer fields
        w = self.mpt.to_monomer_fields(w_aux)

        # Make a dictionary for input fields
        w_input = {self.monomer_types[i]: w[i] for i in range(M)}
        for random_polymer_name, random_fraction in self.random_fraction.items():
            w_input[random_polymer_name] = np.zeros(self.prop_solver.n_grid, dtype=np.complex128)
            for monomer_type, fraction in random_fraction.items():
                w_input[random_polymer_name] += w_input[monomer_type] * fraction

        # Charged types feel the electrostatic potential: w_i + z_i ψ
        if self.electrostatics is not None:
            self.electrostatics.add_potential(w_input)

        # Apply smearing to fields before computing propagators
        # Following Matsen et al., J. Chem. Phys. 164, 014905 (2026), the fields
        # are smeared before computing the Boltzmann weights in the propagator.
        w_input_for_propagator = self.smearing.apply_to_dict(w_input)

        # For the given fields, compute propagators
        time_solver_start = time.time()
        self.prop_solver.compute_propagators(w_input_for_propagator)
        elapsed_time["solver"] = time.time() - time_solver_start

        # Compute concentrations for each monomer type
        time_phi_start = time.time()
        phi = {}
        self.prop_solver.compute_concentrations()
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.prop_solver.get_concentration(monomer_type)

        # Add random copolymer concentration to each monomer type
        for random_polymer_name, random_fraction in self.random_fraction.items():
            phi[random_polymer_name] = self.prop_solver.get_concentration(random_polymer_name)
            for monomer_type, fraction in random_fraction.items():
                phi[monomer_type] += phi[random_polymer_name] * fraction
        elapsed_time["phi"] = time.time() - time_phi_start

        # Material conservation of the canonical chains: mean(sum phi) stays at their total volume
        # fraction, 1 without grand-canonical chains (a numerical blow-up shows up here first)
        mass_error = self.prop_solver.mean(sum(phi[m] for m in self.monomer_types) - self.canonical_fraction)
        if not abs(mass_error) <= 5e-8:
            raise ValidationError("Material conservation failed: mean(sum(phi)) differs from the total volume "
                                  f"fraction of the canonical chains by {mass_error:.3e}")

        # Grand-canonical chains: e^(βμ_p) times their unnormalized segment densities
        for p in self.grand_canonical:
            for monomer_type in self.monomer_types:
                phi[monomer_type] = phi[monomer_type] + self.prop_solver.get_concentration_gce(
                    self.fugacities[p], p, monomer_type)

        # Partition consistency check (forward vs backward propagators), once per run
        if not getattr(self, "_partition_checked", False):
            self._partition_checked = True
            if not report_partition_consistency(self.prop_solver):
                raise ValidationError("Partition consistency check failed (forward/backward mismatch).")

        return phi, elapsed_time

    def evaluate_force(self, fields):
        """Intermediate complex dH/dw, including smearing and stabilization."""
        fields = np.asarray(fields, dtype=np.complex128)
        if fields.shape != (len(self.monomer_types), self.prop_solver.n_grid) or not np.all(np.isfinite(fields)):
            raise ValidationError("Intermediate CLFTS fields have an invalid shape or non-finite values")
        phi, _ = self.compute_concentrations(fields)
        force = self.mpt.compute_func_deriv(fields, self.smearing.apply_to_dict(phi), range(len(self.monomer_types)))
        if self.alpha_ds > 0:
            for i in self.mpt.aux_fields_real_idx:
                force[i] += 1j * self.alpha_ds * np.imag(fields[i])
        if not np.all(np.isfinite(force)):
            raise ValidationError("Non-finite intermediate CLFTS force")
        return force

    def _evaluate_force_with_psi(self, fields):
        """Intermediate forces when the integrator advances ψ as the last field (-δH/δψ)."""
        fields = np.asarray(fields, dtype=np.complex128)
        M, es = len(self.monomer_types), self.electrostatics
        if fields.shape != (M + 1, self.prop_solver.n_grid) or not np.all(np.isfinite(fields)):
            raise ValidationError("Intermediate CLFTS fields have an invalid shape or non-finite values")
        with es.trial(fields[M]):
            phi, _ = self.compute_concentrations(fields[:M])
            phi_smeared = self.smearing.apply_to_dict(phi)
            psi_force = -es.force(es.charge_density(phi_smeared))
        force = self.mpt.compute_func_deriv(fields[:M], phi_smeared, range(M))
        if self.alpha_ds > 0:
            for i in self.mpt.aux_fields_real_idx:
                force[i] += 1j * self.alpha_ds * np.imag(fields[i])
        force = np.vstack([force, psi_force])
        if not np.all(np.isfinite(force)):
            raise ValidationError("Non-finite intermediate CLFTS force")
        return force

    def chemical_potentials(self, total_partitions):
        """Chemical potential operator βμ_p = ln(φ_p/α_p) - ln Q_p of each species (complex).

        H depends on the number of chains n_p only through -n_p ln Q_p, and the ideal term of
        the chains gives ln(n_p/V), so ∂ln Z/∂n_p yields this operator; its average is the
        chemical potential per chain in the convention of SCFT.compute_chemical_potentials,
        which it equals at the saddle point. Its real part does not depend on the branch of
        the logarithm. An absent canonical species has -inf; grand-canonical species have
        their imposed chemical potentials, including at Q_p = 0. The point ions follow the polymers,
        βμ_s = ln(N c_s) - ln Q_s (Electrostatics.ion_chemical_potentials).
        """
        values = np.full(self.chain_densities.shape, -np.inf, dtype=np.complex128)
        occupied_canonical = self.chain_densities > 0
        occupied_canonical[self.grand_canonical] = False
        partitions = np.asarray(total_partitions, dtype=np.complex128)
        values[occupied_canonical] = (np.log(self.chain_densities[occupied_canonical])
                                      - np.log(partitions[occupied_canonical]))
        values[self.grand_canonical] = self.polymer_chemical_potentials[self.grand_canonical]
        if self.electrostatics is not None and self.electrostatics.ions:
            values = np.append(values, self.electrostatics.ion_chemical_potentials())
        return values

    def number_densities(self, total_partitions):
        """Chains (or ions) per reference chain of each species, n_p/n_ref: φ_p/α_p of canonical
        polymers, e^(βμ_p) Q_p of grand-canonical ones, then the point ions (complex operators)."""
        values = np.asarray(self.chain_densities, dtype=np.complex128).copy()
        values[self.grand_canonical] = self.fugacities[self.grand_canonical] * np.asarray(total_partitions)[self.grand_canonical]
        if self.electrostatics is not None and self.electrostatics.ions:
            values = np.append(values, self.electrostatics.ion_number_densities())
        return values

    def grand_canonical_hamiltonian(self, total_partitions):
        """-Σ_p e^(βμ_p) Q_p of the grand-canonical chains, per reference chain (complex)."""
        return -complex(np.sum(self.fugacities * np.asarray(total_partitions, dtype=np.complex128)))

    def partition_ratios(self, w_aux):
        """|Q_s|/<|e^(-u_s)|> of each canonical single-site species (partition_ratio_species).

        Such a species has Q_s = <e^(-u_s)>, the grid mean of the Boltzmann weight of its one site:
        e^(-w_t ds) with the smeared field w_t (+ z_t ψ) of its type for a discrete chain of one
        segment, e^(-z_s (Γ_s * ψ)/N) for a point ion. The ratio is 1 for real fields and falls
        towards 0 as the complex weights cancel in the mean. -n_s ln Q_s has a pole at Q_s = 0,
        where the force n_s e^(-u_s)/Q_s of the species grows without bound. Grand-canonical
        species enter H as -e^(βμ_s) Q_s, without a pole, and are left out.
        """
        ratios = []
        if self._single_site_polymers:
            w = self.mpt.to_monomer_fields_dict(w_aux)
            if self.electrostatics is not None:
                self.electrostatics.add_potential(w)
            types = {self.polymers[p]["blocks"][0]["type"] for p in self._single_site_polymers}
            w = self.smearing.apply_to_dict({t: w[t] for t in types})
            for p in self._single_site_polymers:
                u = self.ds * w[self.polymers[p]["blocks"][0]["type"]]
                weight = np.exp(-(u - np.min(np.real(u))))   # the ratio does not change with a real shift of u
                ratios.append(min(1.0, float(np.abs(np.mean(weight)) / np.mean(np.abs(weight)))))   # 1 up to rounding
        if self.electrostatics is not None:
            ratios += self.electrostatics.ion_partition_ratios()
        return np.array(ratios)

    def pressure(self, w_aux, phi):
        """Osmotic pressure operator βΠ R₀³/C of the auxiliary fields w_aux (complex).

        The propagators of the solver must be those of w_aux, as after compute_concentrations,
        and phi are its concentrations. βΠ = ∂ln Z/∂V at fixed numbers of canonical chains and
        ions and fixed chemical potentials of grand-canonical species, with the grid values of
        the fields held fixed while the box is dilated in its shape (k² ∝ V^(-2/d)). The terms,
        per C/R₀³ = ρ₀/N:
        - the ideal gas of the chains, Σ_p n_p/n_ref: φ_p/α_p of canonical polymers and
          e^(βμ_p) Q_p of grand-canonical ones (from their term -e^(βμ_p) Q_p of H);
        - the conformations, -(1/d) Σ_a L_a ∂h/∂L_a with h = -Σ_p (φ_p/α_p) ln Q_p - Σ_p e^(βμ_p) Q_p
          over the canonical and the grand-canonical polymers (the solver stress, at fixed
          smeared fields);
        - the smearing functions, which follow the box: -Σ_i <φ_i ((∂Γ_i/∂ln V) * (w_i + z_i ψ))>;
        - the field terms of H, which scale with V: -(Σ_i <h2_i Ω_i² + h1_i Ω_i> + h_const);
        - the Gaussian normalization of each auxiliary field, n_grid/2 per C V/R₀³ chains, which
          removes the grid-dependent part of the field term;
        - ψ and the point ions: their terms of H and the normalization of ψ (Electrostatics.pressure).
        At the homogeneous saddle point this is the Flory-Huggins osmotic pressure
        Σ_p φ_p/α_p + φᵀXφ/2 + (ζN/2)((Σφ)² - 1) as C → ∞, plus E q²/(2κ²) for a screened system
        with the net charge q = Σ_i z_i φ_i per segment (ψ₀ = E q/κ² there).
        """
        M, dim, es = len(self.monomer_types), self.prop_solver.dim, self.electrostatics
        n_chains = np.sqrt(self.langevin["nbar"]) * self.prop_solver.get_volume()   # C V/R₀³
        total_partitions = [self.prop_solver.get_partition_function(p) for p in range(len(self.polymers))]
        value = complex(np.sum(self.number_densities(total_partitions)[:len(self.polymers)]))

        self.prop_solver.compute_stress()
        stress, lx = np.asarray(self.prop_solver.get_stress()), self.prop_solver.get_lx()
        if self.grand_canonical:
            # h = -Σ_p e^(βμ_p) Q_p of the grand-canonical chains: dh/dθ = -Σ_p e^(βμ_p) dQ_p/dθ
            stress = stress + np.asarray(self.prop_solver.get_stress_gce(self.fugacities))
        value -= sum(lx[a] * stress[a] for a in range(dim)) / dim

        if self.smearing.enabled:
            w = self.mpt.to_monomer_fields_dict(w_aux)
            if es is not None:
                es.add_potential(w)
            for name in self.monomer_types:
                value -= np.mean(phi[name] * self.smearing.dilation(w[name], name))

        value -= sum(self.mpt.h_coef_mu2[i] * np.mean(w_aux[i] ** 2) + self.mpt.h_coef_mu1[i] * np.mean(w_aux[i])
                     for i in range(M)) + self.mpt.h_const
        value += 0.5 * (self.mpt.R + self.mpt.I) * self.prop_solver.n_grid / n_chains
        if es is not None:
            value += es.pressure(dim, n_chains)
        return value

    def pressure_virial(self, w_aux, phi):
        """Virial form of the osmotic pressure operator: pressure() plus terms of zero average.

        Every auxiliary field Ω_i whose Gaussian integral pressure() normalizes obeys the
        Schwinger-Dyson identity <C V/R₀³ mean(Ω_i ∂H/∂Ω_i)> = n_grid (integration by parts over
        its n_grid grid values; ∂H/∂Ω_i is compute_func_deriv without its sign flip). Adding
        (1/2)[mean(Ω_i ∂H/∂Ω_i) - n_grid/(C V/R₀³)] for each field replaces the field term
        -(h2_i Ω_i² + h1_i Ω_i) and the normalization n_grid/(2 C V/R₀³) by
        -h1_i Ω_i/2 + (Ω_i/2) Σ_j A_ji φ_j: the same average, without the Gaussian fluctuation
        of order sqrt(n_grid/2)/(C V/R₀³) per field and sample. ψ gets the same treatment mode by
        mode (Electrostatics.pressure_virial_correction). CLFTS records this operator.

        The identities hold for exact CL sampling of H; an O(δτ) bias of the integrator or the
        stabilization term of dynamical stabilization, which is not part of H, can make the added
        terms average to a small non-zero value.
        """
        M, es = len(self.monomer_types), self.electrostatics
        n_chains = np.sqrt(self.langevin["nbar"]) * self.prop_solver.get_volume()   # C V/R₀³
        value = self.pressure(w_aux, phi)
        phi_smeared = self.smearing.apply_to_dict(phi)
        deriv = self.mpt.compute_func_deriv(w_aux, phi_smeared, range(M))
        for i in self.mpt.aux_fields_real_idx:
            value += 0.5 * (np.mean(w_aux[i] * deriv[i]) - self.prop_solver.n_grid / n_chains)
        for i in self.mpt.aux_fields_imag_idx:
            value += 0.5 * (-np.mean(w_aux[i] * deriv[i]) - self.prop_solver.n_grid / n_chains)
        if es is not None:
            value += es.pressure_virial_correction(es.charge_density(phi_smeared), self.prop_solver.dim, n_chains)
        return value

    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev, w_aux=None):
        """Save CL-FTS simulation checkpoint data to MATLAB file.

        Parameters
        ----------
        path : str
            Output file path.
        w : numpy.ndarray
            Monomer potential fields.
        phi : dict
            Concentration fields.
        langevin_step : int
            Current Langevin step number.
        normal_noise_prev : numpy.ndarray
            Previous noise array for Leimkuhler-Matthews integration.
        """
        # Make dictionary for data
        mdic = {
            "initial_params": mat_ready(self.params),
            "dim": self.prop_solver.dim,
            "nx": self.prop_solver.nx,
            "lx": self.prop_solver.get_lx(),
            "monomer_types": self.monomer_types,
            "chi_n": self.chi_n,
            "chain_model": self.chain_model,
            "ds": self.ds,
            "dt": self.langevin["dt"],
            "nbar": self.langevin["nbar"],
            "zeta_n": self.zeta_n,
            "eigenvalues": self.mpt.eigenvalues,
            "aux_fields_real": self.mpt.aux_fields_real_idx,
            "aux_fields_imag": self.mpt.aux_fields_imag_idx,
            "matrix_a": self.mpt.matrix_a,
            "matrix_a_inverse": self.mpt.matrix_a_inv,
            "langevin_step": langevin_step,
            "random_generator": self.random_bg.state["bit_generator"],
            "random_state_state": str(self.random_bg.state["state"]["state"]),
            "random_state_inc": str(self.random_bg.state["state"]["inc"]),
            "normal_noise_prev": normal_noise_prev,
        }

        mdic.update(checkpoint_data(self))
        mdic["w_aux"] = self.mpt.to_aux_fields(w) if w_aux is None else w_aux

        # Add w fields to the dictionary (complex values)
        for i, name in enumerate(self.monomer_types):
            mdic["w_" + name + "_real"] = np.real(w[i])
            mdic["w_" + name + "_imag"] = np.imag(w[i])

        # Add concentrations to the dictionary (complex values)
        for name in self.monomer_types:
            mdic["phi_" + name + "_real"] = np.real(phi[name])
            mdic["phi_" + name + "_imag"] = np.imag(phi[name])

        # Electrostatic potential and its noise stream
        if self.electrostatics is not None:
            mdic.update(self.electrostatics.checkpoint_data())

        # Save data with matlab format
        atomic_savemat(path, mdic, long_field_names=True, do_compression=True)

    def continue_run(self, file_name, *, allow_legacy=False):
        """Continue simulation from a saved checkpoint.

        Parameters
        ----------
        file_name : str
            Path to the checkpoint file.
        """
        # Load data
        load_data = loadmat(file_name, squeeze_me=True)
        validate_checkpoint(self, load_data, allow_legacy=allow_legacy)
        restore_extensions(self, load_data)
        if "simulation_metadata_json" in load_data:
            previous_metadata = json.loads(str(load_data["simulation_metadata_json"]))
            self.metadata["random_seed"] = previous_metadata.get("random_seed")
            self.metadata["initial_fields_hash"] = previous_metadata.get("initial_fields_hash", "unknown")
            self.metadata["parent_checkpoint"] = str(file_name)

        # Check if load_data["langevin_step"] is a multiple of statistics_save_interval
        if "record_sample_count" not in load_data and load_data["langevin_step"] % self.recording["statistics_save_interval"] != 0:
            print(f"(Warning!) 'langevin_step' of {file_name} is not a multiple of 'statistics_save_interval'.")
            next_sf_langevin_step = (load_data["langevin_step"] // self.recording["statistics_save_interval"] + 1) * self.recording["statistics_save_interval"]
            print(f"The structure function will be correctly recorded after {next_sf_langevin_step}th langevin_step.")

        # Versioned checkpoints carry the complete PCG64 state; legacy scalar
        # fields are only needed for explicitly accepted unversioned files.
        if "rng_state_json" in load_data:
            self.random_bg.state = json.loads(str(load_data["rng_state_json"]))
        else:
            self.random_bg.state = {
                "bit_generator": "PCG64",
                "state": {"state": int(load_data["random_state_state"]),
                          "inc": int(load_data["random_state_inc"])},
                "has_uint32": 0, "uinteger": 0}
        print("Restored Random Number Generator:", self.random_bg.state)
        if self.electrostatics is not None:
            self.electrostatics.restore(load_data)

        # Make initial_fields (complex)
        initial_fields = {}
        for name in self.monomer_types:
            w_real = np.array(load_data["w_" + name + "_real"])
            w_imag = np.array(load_data["w_" + name + "_imag"])
            initial_fields[name] = w_real + 1j * w_imag

        # Run
        return self.run(
            initial_fields=initial_fields,
            normal_noise_prev=load_data["normal_noise_prev"],
            start_langevin_step=load_data["langevin_step"] + 1, _restart_data=load_data
        )

    def run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None, *, initial_psi=None,
            psi_noise_prev=None, _restart_data=None):
        """Run complex Langevin dynamics and return a status dict; errors are persisted and raised."""
        first_step = 1 if start_langevin_step is None else start_langevin_step
        if isinstance(first_step, (bool, np.bool_)) or not isinstance(first_step, (int, np.integer)) or first_step < 1:
            raise ValidationError("start_langevin_step must be a positive integer")
        if initial_psi is not None:
            if self.electrostatics is None:
                raise ValidationError("initial_psi needs 'electrostatics'")
            if np.size(initial_psi) != self.prop_solver.n_grid or not np.all(np.isfinite(initial_psi)):
                raise ValidationError("initial_psi must hold one finite value per grid point")
            # Without Coulomb coupling ψ does not move, so a nonzero ψ would be a fixed potential
            if self.electrostatics.coupling <= 0 and np.any(np.asarray(initial_psi) != 0):
                raise ValidationError("initial_psi must be zero without Coulomb coupling (coupling E = 0)")
        if psi_noise_prev is not None:
            if initial_psi is None:
                raise ValidationError("psi_noise_prev needs initial_psi")
            if np.size(psi_noise_prev) != self.prop_solver.n_grid or not np.all(np.isfinite(psi_noise_prev)) \
                    or np.iscomplexobj(psi_noise_prev):
                raise ValidationError("psi_noise_prev must hold one real, finite value per grid point")
        self.run_status = {"status": "running", "last_completed_step": first_step - 1,
                           "requested_last_step": self.langevin["max_step"]}
        self.metadata["initial_fields_hash" if _restart_data is None else "restart_fields_hash"] = fields_hash(initial_fields)
        if _restart_data is not None and "run_status_json" in _restart_data:
            previous_status = json.loads(str(_restart_data["run_status_json"]))
            for key in ("saddle_failures", "noise_resamples"):
                if key in self.run_status:
                    self.run_status[key] = previous_status.get(key, 0)
        self.metadata["initial_rng_state"] = self.random_bg.state
        if self.electrostatics is not None:
            self.metadata["initial_psi_rng_state"] = self.electrostatics.random_bg.state
        write_run_status(self)
        try:
            if _restart_data is None:
                restore_extensions(self)
            self._run(initial_fields, normal_noise_prev, start_langevin_step, initial_psi=initial_psi,
                      psi_noise_prev=psi_noise_prev, _restart_data=_restart_data)
            self.run_status["status"] = "completed"
        except (KeyboardInterrupt, SystemExit):
            self.run_status.update(status="interrupted", reason="Run interrupted")
            raise
        except Exception as exc:
            self.run_status.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_run_status(self)
        return dict(self.run_status)

    def _run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None, *, initial_psi=None,
             psi_noise_prev=None, _restart_data=None):
        """Run the Complex Langevin FTS simulation.

        Parameters
        ----------
        initial_fields : dict
            Initial potential fields for each monomer type.
            Values can be real or complex numpy arrays.
        normal_noise_prev : numpy.ndarray, optional
            Previous noise array for continuing simulation.
        start_langevin_step : int, optional
            Starting step number for continuing simulation.
        initial_psi : numpy.ndarray, optional
            Initial electrostatic potential (with `electrostatics`), e.g. psi_real + 1j psi_imag
            of a fields file; zero if omitted. A continuation by hand does not restore the
            random-number streams; continue_run() continues a checkpoint exactly.
        psi_noise_prev : numpy.ndarray, optional
            Previous noise of ψ (psi_noise_prev of a fields file) for the Leimkuhler-Matthews
            step, with initial_psi; zero if omitted.
        """
        print("---------- Run ----------")

        # The number of components
        M = len(self.monomer_types)

        # The numbers of real and imaginary fields
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        # Check the partition consistency on the first field of this run
        self._partition_checked = False

        # Simulation data directory, with the drawings of the chain architectures (polymer_<id>.png)
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)
        if self.draw_architectures:
            draw_polymer_architectures(self.polymers, self.ds, self.segment_lengths, directory=self.recording["dir"])

        # Reshape initial fields (convert to complex if needed)
        w = np.array([np.reshape(initial_fields[m], self.prop_solver.n_grid).astype(np.complex128)
                      for m in self.monomer_types])

        # Convert monomer chemical potential fields into auxiliary fields
        w_aux = self.mpt.to_aux_fields(w)
        self.mpt.pin_dropped_fields(w_aux)
        if _restart_data is not None and "w_aux" in _restart_data:
            w_aux = np.asarray(_restart_data["w_aux"], dtype=np.complex128).reshape(M, self.prop_solver.n_grid).copy()
        if not np.all(np.isfinite(w_aux)):
            raise ValidationError("Initial fields must be finite")

        # The electrostatic potential starts from initial_psi or at zero; continue_run() has
        # restored it with its noise stream
        if self.electrostatics is not None and _restart_data is None:
            if initial_psi is not None:
                self.electrostatics.psi = initial_psi
            else:
                if start_langevin_step is not None and start_langevin_step > 1 and self.electrostatics.coupling > 0:
                    warnings.warn("The run continues without initial_psi: the electrostatic potential restarts "
                                  "at zero and relaxes again", RuntimeWarning)
                self.electrostatics.psi = np.zeros_like(self.electrostatics.psi)

            # Without a checkpoint ψ starts with the given previous noise or none, like the auxiliary fields
            self.electrostatics.previous_noise = (np.zeros(self.prop_solver.n_grid) if psi_noise_prev is None
                                                  else np.asarray(psi_noise_prev, dtype=np.float64).reshape(-1).copy())

        # Dictionary to record history of H and dH/dchi_n
        H_history = []
        drift_history = []
        psi_drift_history = []
        # Largest drift of every step for each auxiliary field, and of ψ, for the distribution of
        # the drift magnitude; a summary of this run's distribution is printed at the end
        drift_step_history = []
        psi_drift_step_history = []
        # |Q_s|/<|e^(-u_s)|> of every step for each canonical single-site species (partition_ratios())
        partition_ratio_step_history = []
        n_ratios = len(self.partition_ratio_species)
        ratio_summary = PartitionRatioSummary(self.partition_ratio_species)
        kinds = {i: "real" for i in self.mpt.aux_fields_real_idx} | {i: "imaginary" for i in self.mpt.aux_fields_imag_idx}
        tail_labels = [f"auxiliary field {i} ({kinds.get(i, 'dropped')})" for i in range(M)]
        drift_tail = DriftTailSummary(tail_labels + (["psi"] if self.electrostatics is not None else []))
        dH_history = {}
        for key in self.chi_n:
            dH_history[key] = []
        # Chemical potentials (one row of all species per sample), osmotic pressure and, with
        # grand-canonical species, the numbers of chains and ions
        chemical_potential_history = []
        pressure_history = []
        number_density_history = []

        # Arrays for structure function (use full FFT for complex fields)
        sf_average = {}  # <u(k) phi(-k)>
        for monomer_id_pair in itertools.combinations_with_replacement(list(range(M)), 2):
            sorted_pair = sorted(monomer_id_pair)
            type_pair = self.monomer_types[sorted_pair[0]] + "," + self.monomer_types[sorted_pair[1]]
            sf_average[type_pair] = np.zeros_like(
                np.fft.fftn(np.reshape(w[0], self.prop_solver.nx)), np.complex128
            )
        # Charge structure function (with Coulomb coupling): sums of the mixed form M = ((k² + κ²)/E) ψ c, of the
        # zero-mean difference D = (field-field form) - M and of their products, for a combination
        # M + w D of least variance (docs/theory/StructureFunctionEstimators.md, Section 6)
        recorded = dict(sf_average)
        charge_sums = None
        if self.electrostatics is not None and self.electrostatics.coupling > 0 and self._periodic_box:
            charge_sums = {key: np.zeros(self.prop_solver.nx, dtype=np.complex128)
                           for key in self.electrostatics.CHARGE_SUMS}
            recorded.update(charge_sums)
        sf_sample_count = restore_recording_state(
            _restart_data, H_history, dH_history, recorded, drift_history,
            psi_drift_history=psi_drift_history if self.electrostatics is not None else None,
            drift_step_history=drift_step_history, n_fields=M,
            psi_drift_step_history=psi_drift_step_history if self.electrostatics is not None else None,
            partition_ratio_step_history=partition_ratio_step_history if n_ratios else None, n_ratios=n_ratios,
            chemical_potential_history=chemical_potential_history if self.record_chemical_potentials else None,
            pressure_history=pressure_history if self.record_pressure else None, n_polymers=self.n_chemical_potentials,
            number_density_history=number_density_history if self.record_number_densities else None)

        # Create an empty array for field update algorithm
        # (a checkpoint loaded with squeeze_me drops the leading axis when M == 1)
        if normal_noise_prev is None:
            normal_noise_prev = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
        else:
            normal_noise_prev = np.reshape(np.asarray(normal_noise_prev, dtype=np.float64), [M, self.prop_solver.n_grid])

        if start_langevin_step is None:
            start_langevin_step = 1
        # First step of the per-step drift records: a restored record ends at the step before
        # this run's first step; a run continued by hand or from a checkpoint without the
        # record starts it anew
        drift_step_start = start_langevin_step - len(drift_step_history)

        # Init timers
        time_start = time.time()

        # Langevin iteration begins here
        langevin_step = start_langevin_step - 1   # stays when max_step < start_langevin_step

        # Concentrations of the initial fields. Afterwards they are computed right after
        # each update, so that the measurements of a step (H, dH/dχN, S(k), saved φ) and
        # the forces of the next step use the concentrations of the same fields.
        # Smearing implements finite-range interactions: φ_Γ = FFT⁻¹[FFT[φ]·Γ(k)]
        # Reference: Matsen et al., J. Chem. Phys. 164, 014905 (2026)
        if _restart_data is not None and "w_aux" in _restart_data:
            phi = {m: (np.asarray(_restart_data["phi_" + m + "_real"]) +
                       1j * np.asarray(_restart_data["phi_" + m + "_imag"])).reshape(self.prop_solver.n_grid)
                   for m in self.monomer_types}
        else:
            phi, _ = self.compute_concentrations(w_aux=w_aux)
        phi_for_force = self.smearing.apply_to_dict(phi)

        # Charge density of the current state, and the force of ψ computed at most once for it
        # (for the drift record, the Poisson residual of the progress line and the next step)
        es = self.electrostatics
        charge_density = None if es is None else es.charge_density(phi_for_force)
        psi_force = None

        for langevin_step in range(start_langevin_step, self.langevin["max_step"] + 1):
            step_start = time.perf_counter()

            # Compute functional derivatives of Hamiltonian w.r.t. all fields
            # For an AB diblock, compute_func_deriv transforms per-monomer concentrations as:
            #   - Exchange field (w-): uses phi_A - phi_B (composition difference)
            #   - Pressure field (w+): uses phi_A + phi_B (total concentration)
            # This is handled automatically through the transformation matrix A.
            # Reference: Matsen et al., J. Chem. Phys. 164, 014905 (2026)
            w_lambda = self.mpt.compute_func_deriv(w_aux, phi_for_force, range(M))

            # Add dynamical stabilization (Willis & Matsen, J. Chem. Phys. 161, 244903 (2024))
            # Adds iαds·N·Im[W] to the exchange field forces to prevent runaway solutions
            # in the imaginary direction. Applied to exchange fields (real-type auxiliary fields).
            # Note: self.alpha_ds should be the combined coefficient αds·N.
            if self.alpha_ds > 0:
                for i in range(M):
                    if i in self.mpt.aux_fields_real_idx:
                        w_lambda[i] += 1j * self.alpha_ds * np.imag(w_aux[i])

            field_drift_max = np.max(np.abs(w_lambda * self.dt_scaling[:, None]), axis=1)
            drift_max = float(np.max(field_drift_max))
            drift_step_history.append(field_drift_max)

            # Largest drift of ψ, s |δH/δψ|, recorded apart from drift_max: its linear part ∇²ψ/E
            # is integrated exactly and would dominate the maximum at large k
            if es is not None:
                if psi_force is None:
                    psi_force = es.force(charge_density)
                psi_drift_max = float(np.max(np.abs(es.psi_dt_scaling * psi_force)))
                psi_drift_step_history.append(psi_drift_max)
            drift_tail.add(np.r_[field_drift_max, [psi_drift_max] if es is not None else []])
            if n_ratios:
                ratios = self.partition_ratios(w_aux)
                partition_ratio_step_history.append(ratios)
                ratio_summary.add(ratios)

            # Integrate active fields; multistage methods can evaluate new forces.
            normal_noise_current = self.random.normal(
                0.0, self.langevin["sigma"], [M, self.prop_solver.n_grid]
            )
            if self._joint_psi:
                # The integrator advances ψ with the auxiliary fields (linear_response): ψ is the last
                # field, with the force -δH/δψ, its mobility and imaginary noise from its own stream
                psi_noise = es.draw_noise()
                fields = np.vstack([w_aux, es.psi])
                proposed = self.integrator.step(
                    fields.copy(), np.vstack([w_lambda, -psi_force]), dt=self.langevin["dt"],
                    mobility=self._integrated_mobility(),
                    noise=np.vstack([normal_noise_current, psi_noise]),
                    previous_noise=np.vstack([normal_noise_prev, es.previous_noise]),
                    noise_direction=self._integrated_noise_direction(),
                    evaluate_force=lambda fields: self._evaluate_force_with_psi(fields))
                proposed = checked_fields(proposed, fields)
            else:
                proposed = self.integrator.step(
                    w_aux.copy(), w_lambda.copy(), dt=self.langevin["dt"], mobility=self.dt_scaling.copy(),
                    noise=normal_noise_current.copy(), previous_noise=normal_noise_prev.copy(),
                    noise_direction=self._integrated_noise_direction(),
                    evaluate_force=lambda fields: self.evaluate_force(fields))
                proposed = checked_fields(proposed, w_aux)
            if not np.array_equal(proposed[:M][self.dt_scaling == 0], w_aux[self.dt_scaling == 0]):
                raise ValidationError("An integrator must preserve fields with zero mobility")
            w_aux = proposed[:M].copy()

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            # Electrostatic potential: from the step of the integrator, or one step of its own with
            # its own noise stream, from the charge density of the state before the step
            if es is not None:
                if self._joint_psi:
                    es.accept(proposed[M], psi_noise)
                else:
                    es.step(charge_density)
                if not np.all(np.isfinite(es.psi)):
                    raise ValidationError(f"Non-finite electrostatic potential at step {langevin_step}")

            # Concentrations and partition functions of the updated fields
            phi, _ = self.compute_concentrations(w_aux=w_aux)
            phi_for_force = self.smearing.apply_to_dict(phi)
            if not np.all(np.isfinite(w_aux)) or not all(np.all(np.isfinite(v)) for v in phi.values()):
                raise ValidationError(f"Non-finite complex fields or concentrations at step {langevin_step}")
            if es is not None:
                charge_density, psi_force = es.charge_density(phi_for_force), None

            # Compute functional derivatives for error monitoring (use smeared phi for consistency)
            h_deriv = self.mpt.compute_func_deriv(w_aux, phi_for_force, range(M))

            # Compute error levels
            error_level_array = np.std(h_deriv, axis=1)

            # Calculate Hamiltonian
            total_partitions = [self.prop_solver.get_partition_function(p) for p in range(self.prop_solver.get_n_polymer_types())]
            hamiltonian = self.mpt.compute_hamiltonian(self.prop_solver.molecules, w_aux, total_partitions, self.cb,
                                                      include_const_term=True, phi=phi_for_force)
            if self.electrostatics is not None:
                hamiltonian += self.electrostatics.hamiltonian()
            if self.grand_canonical:
                hamiltonian += self.grand_canonical_hamiltonian(total_partitions)
            if not np.isfinite(hamiltonian):
                raise ValidationError(f"Non-finite Hamiltonian at step {langevin_step}")
            self.run_status["last_completed_step"] = langevin_step

            # Print progress
            if self.verbose_level >= 1:
                mass_error = self.cb.mean(h_deriv[M - 1])
                print(f"{langevin_step:8d} {mass_error.real:+.3E}{mass_error.imag:+.3E}j", end="  [ ")
                for Q in total_partitions:
                    print(f"{Q.real:.6E}{Q.imag:+.6E}j", end=" ")
                print(f"] {hamiltonian.real:+.9E}{hamiltonian.imag:+.9E}j  [", end="")
                for i in range(M):
                    print(f" {error_level_array[i]:.3E}", end="")
                # Poisson residual of the electrostatic potential
                if es is not None:
                    if psi_force is None:
                        psi_force = es.force(charge_density)
                    print(f" {np.std(psi_force):.3E}", end="")
                print(" ]")

            # Compute H and dH/dchi_n
            if langevin_step % self.recording["sample_interval"] == 0:
                H_history.append(hamiltonian)
                drift_history.append(drift_max)
                if self.electrostatics is not None:
                    psi_drift_history.append(psi_drift_max)
                sf_sample_count += 1
                dH = self.mpt.compute_h_deriv_chin(w_aux=w_aux, phi=phi_for_force)
                for key in self.chi_n:
                    dH_history[key].append(dH[key])
                if self.record_chemical_potentials:
                    chemical_potential_history.append(self.chemical_potentials(total_partitions))
                if self.record_number_densities:
                    number_density_history.append(self.number_densities(total_partitions))
                if self.record_pressure:
                    pressure_history.append(self.pressure_virial(w_aux, phi))

            # Save H and dH/dchi_n
            if langevin_step % self.recording["statistics_save_interval"] == 0:
                H_history_array = np.array(H_history)
                mdic = {"H_history_real": np.real(H_history_array), "H_history_imag": np.imag(H_history_array),
                        "sample_count": sf_sample_count, "drift_max_history": np.asarray(drift_history),
                        "simulation_metadata_json": json.dumps(self.metadata)}
                mdic["drift_max_step_history"] = np.reshape(drift_step_history, (-1, M))
                if self.electrostatics is not None:
                    mdic["psi_drift_max_history"] = np.asarray(psi_drift_history)
                    mdic["psi_drift_max_step_history"] = np.asarray(psi_drift_step_history)
                if n_ratios:
                    mdic["partition_ratio_step_history"] = np.reshape(partition_ratio_step_history, (-1, n_ratios))
                    mdic["partition_ratio_species_json"] = json.dumps(self.partition_ratio_species)
                for key in self.chi_n:
                    dH_array = np.array(dH_history[key])
                    monomer_pair = sorted(key.split(","))
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1] + "_real"] = np.real(dH_array)
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1] + "_imag"] = np.imag(dH_array)
                if self.record_chemical_potentials:
                    mu_array = np.reshape(np.array(chemical_potential_history, dtype=np.complex128),
                                          (-1, self.n_chemical_potentials))
                    mdic["chemical_potential_history_real"] = np.real(mu_array)
                    mdic["chemical_potential_history_imag"] = np.imag(mu_array)
                if self.record_number_densities:
                    density_array = np.reshape(np.array(number_density_history, dtype=np.complex128),
                                               (-1, self.n_chemical_potentials))
                    mdic["number_density_history_real"] = np.real(density_array)
                    mdic["number_density_history_imag"] = np.imag(density_array)
                if self.record_pressure:
                    pressure_array = np.array(pressure_history, dtype=np.complex128)
                    mdic["pressure_history_real"] = np.real(pressure_array)
                    mdic["pressure_history_imag"] = np.imag(pressure_array)
                atomic_savemat(
                    os.path.join(self.recording["dir"], "dH_%06d.mat" % langevin_step),
                    mdic, long_field_names=True, do_compression=True
                )
                # Reset dictionary
                H_history = []
                drift_history = []
                psi_drift_history = []
                drift_step_history = []
                psi_drift_step_history = []
                partition_ratio_step_history = []
                drift_step_start = langevin_step + 1
                chemical_potential_history = []
                pressure_history = []
                number_density_history = []
                for key in self.chi_n:
                    dH_history[key] = []

            # Calculate structure function
            if langevin_step % self.recording["sample_interval"] == 0:
                # Perform Fourier transforms (full FFT for complex fields).
                # Shifting the Gaussian integral over Ω_k gives <Ω_k φ_j> = (λ_k/M) Σ_l A_lk <φ_l φ_j>,
                # and A Aᵀ = M I, so μ_i = Σ_k A_ik Ω_k/λ_k gives <μ_i φ_j> = <δφ_i δφ_j>. All fields
                # fluctuate in CL-FTS; only the incompressible pressure field (λ = 0) is left out.
                # The fields are complex, so φ(-k) is not the conjugate of φ(k):
                # φ(-k) = conj(FFT[conj φ](k)).
                # All densities are transformed in one call, all fields in another
                kept = [k for k in range(M) if not np.isclose(self.mpt.eigenvalues[k], 0.0)]
                phi_mk = np.conj(self._fft_fields(np.conj([phi[name] for name in self.monomer_types]))) / self.prop_solver.n_grid
                w_k = self._fft_fields(w_aux[kept]) / self.prop_solver.n_grid if kept else None
                mu_fourier = {}
                phi_fourier = {}   # φ(-k)
                for i in range(M):
                    key = self.monomer_types[i]
                    phi_fourier[key] = phi_mk[i]
                    mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
                    for j, k in enumerate(kept):
                        mu_fourier[key] += w_k[j] * self.mpt.matrix_a[i, k] / self.mpt.eigenvalues[k]
                # Zero-eigenvalue modes along u: μ_i += u_i [((k² + κ²)/E) ψ(k) - Σ_l z_l μ_l(k)] / |u|²
                if self._psi_probe is not None:
                    es = self.electrostatics
                    probe = es.psi_k * es.screened_k_sq / es.coupling / self.prop_solver.n_grid
                    # ψ measures the charge of the point ions and the fixed charge as well: remove them
                    if es.ions:
                        probe -= np.reshape(self._fft_fields(es.ion_charge())[0], probe.shape) / self.prop_solver.n_grid
                    if es.fixed_charge is not None:
                        probe -= np.reshape(self._fft_fields(es.fixed_charge)[0], probe.shape) / self.prop_solver.n_grid
                    # k = 0 is not a structure function in the canonical ensemble; with screening
                    # κ²ψ(0)/E would carry a net charge there
                    probe[(0,) * len(self.prop_solver.nx)] = 0.0
                    for name in self.monomer_types:
                        probe -= es.charges[name] * mu_fourier[name]
                    for i, name in enumerate(self.monomer_types):
                        mu_fourier[name] = mu_fourier[name] + self._psi_probe[i] * probe
                # Accumulate S_ij(K) = <μ_i(k) φ_j(-k)>
                for key in sf_average:
                    monomer_pair = sorted(key.split(","))
                    sf_average[key] += mu_fourier[monomer_pair[0]] * phi_fourier[monomer_pair[1]]
                if charge_sums is not None:
                    es = self.electrostatics
                    n_grid, nx = self.prop_solver.n_grid, self.prop_solver.nx
                    norm = self.prop_solver.get_volume() * np.sqrt(self.langevin["nbar"])
                    psi_k = es.psi_k / n_grid
                    psi_mk = es.psi_minus_k() / n_grid
                    # c(-k) = Σ_i z_i Γ_i(k) φ_i(-k) from the φ_i(-k) above (Γ_i is even in k), and the
                    # smeared charge of the point ions
                    charge_mk = sum(z * self.smearing.kernel(name) * phi_fourier[name]
                                    for name, z in es.charges.items() if z != 0.0)
                    if es.ions:
                        ion_mk = np.conj(self._fft_fields(np.conj(es.ion_charge()))[0]) / n_grid
                        charge_mk = charge_mk + np.reshape(ion_mk, np.shape(psi_k))
                    if es.fixed_charge is not None:
                        fixed_mk = np.conj(self._fft_fields(es.fixed_charge)[0]) / n_grid
                        charge_mk = charge_mk + np.reshape(fixed_mk, np.shape(psi_k))
                    rate = es.screened_k_sq / es.coupling
                    with np.errstate(divide="ignore", invalid="ignore"):
                        mixed = rate * norm * psi_k * charge_mk
                        difference = rate ** 2 * (norm * psi_k * psi_mk + 1.0 / rate) - mixed
                    mixed[(0,) * len(nx)] = 0.0
                    difference[(0,) * len(nx)] = 0.0
                    charge_sums["charge_mixed"] += mixed
                    charge_sums["charge_difference"] += difference
                    charge_sums["charge_mixed_difference"] += mixed.real * difference.real
                    charge_sums["charge_difference_squared"] += difference.real ** 2

            # Save structure function
            if langevin_step % self.recording["statistics_save_interval"] == 0:
                # Make a dictionary for chi_n
                chi_n_mat = {}
                for key in self.chi_n:
                    monomer_pair = sorted(key.split(","))
                    chi_n_mat[monomer_pair[0] + "," + monomer_pair[1]] = self.chi_n[key]
                mdic = {
                    "dim": self.prop_solver.dim,
                    "nx": self.prop_solver.nx,
                    "lx": self.prop_solver.get_lx(),
                    "chi_n": chi_n_mat,
                    "chain_model": self.chain_model,
                    "ds": self.ds,
                    "dt": self.langevin["dt"],
                    "nbar": self.langevin["nbar"],
                    "initial_params": mat_ready(self.params)
                }
                # Add structure functions to the dictionary
                for key in sf_average:
                    sf_scaled = (
                        sf_average[key] *
                        1.0 / max(sf_sample_count, 1) *
                        self.prop_solver.get_volume() * np.sqrt(self.langevin["nbar"])
                    )
                    # With smearing the fields couple to the smeared densities, so <μ_i φ_j> = Γ_i(k) S_ij(k):
                    # divide by Γ_i to record S(k) of the densities (modes the smeared interaction
                    # does not see, Γ < 1e-12, are recorded as 0). With walls Γ lives in the cosine
                    # basis while S(k) is a Fourier transform, so it is recorded as it is.
                    monomer_pair = sorted(key.split(","))
                    if self.smearing.enabled and self.smearing.periodic:
                        gamma = self.smearing.kernel(monomer_pair[0])
                        sf_scaled = np.divide(sf_scaled, gamma, out=np.zeros_like(sf_scaled), where=gamma > 1e-12)
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1] + "_real"] = np.real(sf_scaled)
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1] + "_imag"] = np.imag(sf_scaled)
                # Structure function of the smeared charge density c = Σ_i z_i Γ_i φ_i: <M> + w <D>
                # with the weight w = -Cov(M, D)/Var(D) of least variance at each wavevector (w = 0:
                # the mixed form, w = 1: the field-field form). Both forms are recorded too: in the
                # Gaussian regime the combination follows from the linear response of the chains
                # and does not show how well ψ is sampled (docs/theory/StructureFunctionEstimators.md)
                if charge_sums is not None:
                    count = max(sf_sample_count, 1)
                    mixed = charge_sums["charge_mixed"] / count
                    difference = charge_sums["charge_difference"] / count
                    covariance = charge_sums["charge_mixed_difference"].real / count - mixed.real * difference.real
                    variance = charge_sums["charge_difference_squared"].real / count - difference.real ** 2
                    weight = np.divide(-covariance, variance, out=np.zeros_like(variance), where=variance > 0)
                    s_charge = mixed + weight * difference
                    mdic["charge_structure_function_real"] = np.real(s_charge)
                    mdic["charge_structure_function_imag"] = np.imag(s_charge)
                    mdic["charge_structure_function_weight"] = weight
                    for form, s_form in (("mixed", mixed), ("field_field", mixed + difference)):
                        mdic["charge_structure_function_" + form + "_real"] = np.real(s_form)
                        mdic["charge_structure_function_" + form + "_imag"] = np.imag(s_form)
                mdic["sample_count"] = sf_sample_count
                mdic["simulation_metadata_json"] = json.dumps(self.metadata)
                atomic_savemat(
                    os.path.join(self.recording["dir"], "structure_function_%06d.mat" % langevin_step),
                    mdic, long_field_names=True, do_compression=True
                )
                # Reset arrays
                for key in recorded:
                    recorded[key][:] = 0.0
                sf_sample_count = 0

            # Observe an accepted state before saving its callback/integrator state.
            notify_callbacks(self, langevin_step, w_aux, phi, hamiltonian, drift_max,
                             error_level_array, time.perf_counter() - step_start)

            # Save simulation data
            if langevin_step % self.recording["checkpoint_interval"] == 0:
                self._recording_state = {"H_history": H_history, "dH_history": dH_history,
                                         "sf_average": recorded, "sample_count": sf_sample_count, "drift_history": drift_history,
                                         "drift_step_history": drift_step_history, "drift_step_start": drift_step_start}
                if self.electrostatics is not None:
                    self._recording_state["psi_drift_history"] = psi_drift_history
                    self._recording_state["psi_drift_step_history"] = psi_drift_step_history
                if n_ratios:
                    self._recording_state["partition_ratio_step_history"] = partition_ratio_step_history
                if self.record_chemical_potentials:
                    self._recording_state["chemical_potential_history"] = chemical_potential_history
                if self.record_pressure:
                    self._recording_state["pressure_history"] = pressure_history
                if self.record_number_densities:
                    self._recording_state["number_density_history"] = number_density_history
                w = self.mpt.to_monomer_fields(w_aux)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % langevin_step),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev, w_aux=w_aux
                )

        # Estimate execution time
        time_duration = time.time() - time_start

        n_steps = max(langevin_step + 1 - start_langevin_step, 1)
        print("total time: %f, time per step: %f" % (time_duration, time_duration / n_steps))
        drift_tail.report()
        ratio_summary.report()
