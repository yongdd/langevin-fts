"""Complex Langevin field-theoretic simulation (CL-FTS).

All auxiliary fields are complex and fluctuate: real noise on the exchange
fields, imaginary noise on the pressure field, with optional dynamical
stabilization (Willis & Matsen, J. Chem. Phys. 161, 244903 (2024)) and
smearing (Matsen et al., J. Chem. Phys. 164, 014905 (2026)). Parameters are
documented in docs/references/Parameters.md; a complete script is
examples/clfts/Lamella.py.
"""

import os
import time
import pathlib
import copy
import numpy as np
import itertools

from scipy.io import savemat, loadmat

from .polymer_field_theory import SymmetricPolymerTheory
from .smearing import Smearing
from .propagator_solver import PropagatorSolver
from .validation import validate_system_params, ValidationError
from .utils import (
    parse_chi_n,
    process_polymer_blocks,
    process_random_copolymer,
    draw_polymer_architectures,
    calculate_sigma,
    load_config,
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
    def from_file(cls, path, random_seed=None, **overrides):
        """Create a simulation from a YAML or JSON parameter file.

        Reads the file with utils.load_config, applies the keyword overrides
        and calls the constructor, as SCFT.from_file does.
        """
        params = load_config(path)
        params.update(overrides)
        return cls(params, random_seed=random_seed)

    def __init__(self, params, random_seed=None):
        """Initialize the Complex Langevin FTS simulation."""

        # Validate input parameters (same checks as SCFT and LFTS)
        validate_system_params(params)
        for key in ("langevin", "recording", "verbose_level"):
            if key not in params:
                raise ValidationError(f"Missing required parameter for CLFTS: '{key}'")
        if params["verbose_level"] not in (0, 1, 2):
            raise ValidationError(f"'verbose_level' must be 0, 1 or 2, got {params['verbose_level']}")

        # Monomer types and statistical segment lengths
        self.monomer_types = sorted(list(params["segment_lengths"].keys()))
        self.segment_lengths = copy.deepcopy(params["segment_lengths"])
        self.distinct_polymers = copy.deepcopy(params["distinct_polymers"])

        # Flory-Huggins parameters chi N; pairs the user did not give do not interact
        self.chi_n = parse_chi_n(params["chi_n"], self.segment_lengths)
        for pair in itertools.combinations(self.monomer_types, 2):
            key = ",".join(sorted(pair))
            if key not in self.chi_n:
                self.chi_n[key] = 0.0

        # Compressibility parameter; complex Langevin needs a finite one (default: high compressibility)
        self.zeta_n = 100.0 if params.get("zeta_n", None) is None else params["zeta_n"]

        # Polymer chains: block connectivity, then random copolymers (one effective monomer type each)
        for polymer in self.distinct_polymers:
            polymer["blocks_input"] = process_polymer_blocks(polymer)
        self.random_fraction = {}
        for polymer in self.distinct_polymers:
            result = process_random_copolymer(polymer, self.segment_lengths)
            if result is not None:
                random_type, effective_segment_length, fraction = result
                polymer["block_monomer_types"] = [random_type]
                self.segment_lengths[random_type] = effective_segment_length
                self.random_fraction[random_type] = fraction

        # Draw the chain architectures (polymer_<id>.png in the current directory)
        draw_polymer_architectures(self.distinct_polymers, params["ds"], self.segment_lengths)

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
        for polymer in self.distinct_polymers:
            self.prop_solver.add_polymer(polymer["volume_fraction"], polymer["blocks_input"],
                                         polymer.get("initial_conditions", None))
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
        # The parameter can be specified as either:
        #   - "dynamic_stabilization": the combined coefficient αds (as in Willis & Matsen 2024)
        #   - "alpha_ds": alias for compatibility with reference codes
        # Typical values: 0.01-0.1 for N̄ ≥ 10^5 (Willis & Matsen 2024)
        if "dynamic_stabilization" in params:
            self.alpha_ds = params["dynamic_stabilization"]
        elif "alpha_ds" in params:
            self.alpha_ds = params["alpha_ds"]
        else:
            self.alpha_ds = 0.0

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
        self.params = params
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
        self.smearing = Smearing(
            self.prop_solver.nx,
            self.prop_solver.get_lx(),
            params.get("smearing", None),
            angles=self.prop_solver.get_angles_degrees(),
            bc=self.prop_solver.bc
        )

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

    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev):
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
            "initial_params": self.params,
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

        # Add w fields to the dictionary (complex values)
        for i, name in enumerate(self.monomer_types):
            mdic["w_" + name + "_real"] = np.real(w[i])
            mdic["w_" + name + "_imag"] = np.imag(w[i])

        # Add concentrations to the dictionary (complex values)
        for name in self.monomer_types:
            mdic["phi_" + name + "_real"] = np.real(phi[name])
            mdic["phi_" + name + "_imag"] = np.imag(phi[name])

        # Save data with matlab format
        savemat(path, mdic, long_field_names=True, do_compression=True)

    def continue_run(self, file_name):
        """Continue simulation from a saved checkpoint.

        Parameters
        ----------
        file_name : str
            Path to the checkpoint file.
        """
        # Load data
        load_data = loadmat(file_name, squeeze_me=True)

        # Check if load_data["langevin_step"] is a multiple of sf_recording_period
        if load_data["langevin_step"] % self.recording["sf_recording_period"] != 0:
            print(f"(Warning!) 'langevin_step' of {file_name} is not a multiple of 'sf_recording_period'.")
            next_sf_langevin_step = (load_data["langevin_step"] // self.recording["sf_recording_period"] + 1) * self.recording["sf_recording_period"]
            print(f"The structure function will be correctly recorded after {next_sf_langevin_step}th langevin_step.")

        # Restore random state
        self.random_bg.state = {
            'bit_generator': 'PCG64',
            'state': {
                'state': int(load_data["random_state_state"]),
                'inc': int(load_data["random_state_inc"])
            },
            'has_uint32': 0,
            'uinteger': 0
        }
        print("Restored Random Number Generator:", self.random_bg.state)

        # Make initial_fields (complex)
        initial_fields = {}
        for name in self.monomer_types:
            w_real = np.array(load_data["w_" + name + "_real"])
            w_imag = np.array(load_data["w_" + name + "_imag"])
            initial_fields[name] = w_real + 1j * w_imag

        # Run
        self.run(
            initial_fields=initial_fields,
            normal_noise_prev=load_data["normal_noise_prev"],
            start_langevin_step=load_data["langevin_step"] + 1
        )

    def run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None):
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
        """
        print("---------- Run ----------")

        # The number of components
        M = len(self.monomer_types)

        # The numbers of real and imaginary fields
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        # Check the partition consistency on the first field of this run
        self._partition_checked = False

        # Simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # Reshape initial fields (convert to complex if needed)
        w = np.array([np.reshape(initial_fields[m], self.prop_solver.n_grid).astype(np.complex128)
                      for m in self.monomer_types])

        # Convert monomer chemical potential fields into auxiliary fields
        w_aux = self.mpt.to_aux_fields(w)
        self.mpt.pin_dropped_fields(w_aux)

        # Dictionary to record history of H and dH/dchi_n
        H_history = []
        dH_history = {}
        for key in self.chi_n:
            dH_history[key] = []

        # Arrays for structure function (use full FFT for complex fields)
        sf_average = {}  # <u(k) phi(-k)>
        for monomer_id_pair in itertools.combinations_with_replacement(list(range(M)), 2):
            sorted_pair = sorted(monomer_id_pair)
            type_pair = self.monomer_types[sorted_pair[0]] + "," + self.monomer_types[sorted_pair[1]]
            sf_average[type_pair] = np.zeros_like(
                np.fft.fftn(np.reshape(w[0], self.prop_solver.nx)), np.complex128
            )

        # Create an empty array for field update algorithm
        # (a checkpoint loaded with squeeze_me drops the leading axis when M == 1)
        if normal_noise_prev is None:
            normal_noise_prev = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
        else:
            normal_noise_prev = np.reshape(np.asarray(normal_noise_prev, dtype=np.float64), [M, self.prop_solver.n_grid])

        if start_langevin_step is None:
            start_langevin_step = 1

        # Init timers
        time_start = time.time()

        # Langevin iteration begins here
        langevin_step = start_langevin_step - 1   # stays when max_step < start_langevin_step

        # Concentrations of the initial fields. Afterwards they are computed right after
        # each update, so that the measurements of a step (H, dH/dχN, S(k), saved φ) and
        # the forces of the next step use the concentrations of the same fields.
        # Smearing implements finite-range interactions: φ_Γ = FFT⁻¹[FFT[φ]·Γ(k)]
        # Reference: Matsen et al., J. Chem. Phys. 164, 014905 (2026)
        phi, _ = self.compute_concentrations(w_aux=w_aux)
        phi_for_force = self.smearing.apply_to_dict(phi)

        for langevin_step in range(start_langevin_step, self.langevin["max_step"] + 1):

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

            # Update w_aux using Leimkuhler-Matthews method
            normal_noise_current = self.random.normal(
                0.0, self.langevin["sigma"], [M, self.prop_solver.n_grid]
            )
            for i in range(M):
                if i in self.mpt.aux_fields_dropped_idx:
                    continue
                scaling = self.dt_scaling[i]
                # Determine noise direction based on field type
                if i in self.mpt.aux_fields_real_idx:
                    noise_factor = 1.0  # Real noise for exchange fields
                else:
                    noise_factor = 1j  # Imaginary noise for pressure field
                w_aux[i] += (-w_lambda[i] * self.langevin["dt"] * scaling +
                            0.5 * noise_factor * (normal_noise_prev[i] + normal_noise_current[i]) * np.sqrt(scaling))

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            # Concentrations and partition functions of the updated fields
            phi, _ = self.compute_concentrations(w_aux=w_aux)
            phi_for_force = self.smearing.apply_to_dict(phi)

            # Compute functional derivatives for error monitoring (use smeared phi for consistency)
            h_deriv = self.mpt.compute_func_deriv(w_aux, phi_for_force, range(M))

            # Compute error levels
            error_level_array = np.std(h_deriv, axis=1)

            # Calculate Hamiltonian
            total_partitions = [self.prop_solver.get_partition_function(p) for p in range(self.prop_solver.get_n_polymer_types())]
            hamiltonian = self.mpt.compute_hamiltonian(self.prop_solver.molecules, w_aux, total_partitions, self.cb,
                                                      include_const_term=True, phi=phi_for_force)

            # Print progress
            if self.verbose_level >= 1:
                mass_error = self.cb.mean(h_deriv[M - 1])
                print(f"{langevin_step:8d} {mass_error.real:+.3E}{mass_error.imag:+.3E}j", end="  [ ")
                for Q in total_partitions:
                    print(f"{Q.real:.6E}{Q.imag:+.6E}j", end=" ")
                print(f"] {hamiltonian.real:+.9E}{hamiltonian.imag:+.9E}j  [", end="")
                for i in range(M):
                    print(f" {error_level_array[i]:.3E}", end="")
                print(" ]")

            # Compute H and dH/dchi_n
            if langevin_step % self.recording["sf_computing_period"] == 0:
                H_history.append(hamiltonian)
                dH = self.mpt.compute_h_deriv_chin(self.chi_n, w_aux, phi_for_force)
                for key in self.chi_n:
                    dH_history[key].append(dH[key])

            # Save H and dH/dchi_n
            if langevin_step % self.recording["sf_recording_period"] == 0:
                H_history_array = np.array(H_history)
                mdic = {"H_history_real": np.real(H_history_array), "H_history_imag": np.imag(H_history_array)}
                for key in self.chi_n:
                    dH_array = np.array(dH_history[key])
                    monomer_pair = sorted(key.split(","))
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1] + "_real"] = np.real(dH_array)
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1] + "_imag"] = np.imag(dH_array)
                savemat(
                    os.path.join(self.recording["dir"], "dH_%06d.mat" % langevin_step),
                    mdic, long_field_names=True, do_compression=True
                )
                # Reset dictionary
                H_history = []
                for key in self.chi_n:
                    dH_history[key] = []

            # Calculate structure function
            if langevin_step % self.recording["sf_computing_period"] == 0:
                # Perform Fourier transforms (full FFT for complex fields).
                # Shifting the Gaussian integral over Ω_k gives <Ω_k φ_j> = (λ_k/M) Σ_l A_lk <φ_l φ_j>,
                # and A Aᵀ = M I, so μ_i = Σ_k A_ik Ω_k/λ_k gives <μ_i φ_j> = <δφ_i δφ_j>. All fields
                # fluctuate in CL-FTS; only the incompressible pressure field (λ = 0) is left out.
                # The fields are complex, so φ(-k) is not the conjugate of φ(k):
                # φ(-k) = conj(FFT[conj φ](k)).
                mu_fourier = {}
                phi_fourier = {}   # φ(-k)
                for i in range(M):
                    key = self.monomer_types[i]
                    phi_fourier[key] = np.conj(np.fft.fftn(
                        np.conj(np.reshape(phi[self.monomer_types[i]], self.prop_solver.nx))
                    )) / self.prop_solver.n_grid
                    mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
                    for k in range(M):
                        if np.isclose(self.mpt.eigenvalues[k], 0.0):
                            continue
                        mu_fourier[key] += (
                            np.fft.fftn(np.reshape(w_aux[k], self.prop_solver.nx)) *
                            self.mpt.matrix_a[i, k] / self.mpt.eigenvalues[k] / self.prop_solver.n_grid
                        )
                # Accumulate S_ij(K) = <μ_i(k) φ_j(-k)>
                for key in sf_average:
                    monomer_pair = sorted(key.split(","))
                    sf_average[key] += mu_fourier[monomer_pair[0]] * phi_fourier[monomer_pair[1]]

            # Save structure function
            if langevin_step % self.recording["sf_recording_period"] == 0:
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
                    "initial_params": self.params
                }
                # Add structure functions to the dictionary
                for key in sf_average:
                    sf_scaled = (
                        sf_average[key] *
                        self.recording["sf_computing_period"] / self.recording["sf_recording_period"] *
                        self.prop_solver.get_volume() * np.sqrt(self.langevin["nbar"])
                    )
                    # With smearing the fields couple to the smeared densities, so <μ φ> = Γ(k) S(k):
                    # divide by Γ to record S(k) of the densities (modes the smeared interaction
                    # does not see, Γ < 1e-12, are recorded as 0). With walls Γ lives in the cosine
                    # basis while S(k) is a Fourier transform, so it is recorded as it is.
                    if self.smearing.enabled and self.smearing.periodic:
                        gamma = self.smearing.fourier
                        sf_scaled = np.divide(sf_scaled, gamma, out=np.zeros_like(sf_scaled), where=gamma > 1e-12)
                    monomer_pair = sorted(key.split(","))
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1] + "_real"] = np.real(sf_scaled)
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1] + "_imag"] = np.imag(sf_scaled)
                savemat(
                    os.path.join(self.recording["dir"], "structure_function_%06d.mat" % langevin_step),
                    mdic, long_field_names=True, do_compression=True
                )
                # Reset arrays
                for key in sf_average:
                    sf_average[key][:] = 0.0

            # Save simulation data
            if langevin_step % self.recording["recording_period"] == 0:
                w = self.mpt.to_monomer_fields(w_aux)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % langevin_step),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev
                )

        # Estimate execution time
        time_duration = time.time() - time_start

        n_steps = max(langevin_step + 1 - start_langevin_step, 1)
        print("total time: %f, time per step: %f" % (time_duration, time_duration / n_steps))
