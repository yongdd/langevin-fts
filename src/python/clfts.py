"""
Complex Langevin Field-Theoretic Simulation (CL-FTS) module.

This module implements Complex Langevin dynamics for polymer field theory,
which extends field values to the complex plane to handle the sign problem
in field-theoretic simulations.

The Complex Langevin method evolves fields according to:
    dw = -dH/dw * dt + sqrt(2*dt) * eta

where w are complex-valued fields, H is the Hamiltonian, and eta is
Gaussian white noise applied only to the real part of the field evolution.

Key features:
- Complex-valued auxiliary fields
- Dynamical stabilization for numerical stability
- Smearing interactions for improved stability (finite-range interactions)
- Support for multi-component polymer systems
- Structure function calculations

Example usage:

    from polymerfts import clfts
    import numpy as np

    params = {
        "nx": [32, 32, 32],
        "lx": [4.0, 4.0, 4.0],
        "chain_model": "discrete",
        "ds": 1/90,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 12.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": 0.5},
                {"type": "B", "length": 0.5},
            ],
        }],
        "langevin": {
            "max_step": 10000,
            "dt": 1.0,
            "nbar": 10000,
        },
        "recording": {
            "dir": "data",
            "recording_period": 1000,
            "sf_computing_period": 10,
            "sf_recording_period": 1000,
        },
        "verbose_level": 1,
        # Optional: Smearing for finite-range interactions
        # "smearing": {"type": "gaussian", "a_int": 0.02},
    }

    simulation = clfts.CLFTS(params)
    simulation.run(initial_fields={"A": w_A, "B": w_B})

References:
- Ganesan & Fredrickson, Europhys. Lett. 55, 814 (2001)
- Lennon et al., Phys. Rev. Lett. 101, 138302 (2008)
- Willis & Matsen, J. Chem. Phys. 161, 244903 (2024) - Dynamical stabilization
- Matsen et al., J. Chem. Phys. 164, 014905 (2026) - Universal phase behavior with CL-FTS
- Delaney & Fredrickson, J. Phys. Chem. B 120, 7615 (2016) - Smearing interactions
"""

import os
import time
import re
import pathlib
import copy
import numpy as np
import itertools

from scipy.io import savemat, loadmat

from . import _core
from .polymer_field_theory import SymmetricPolymerTheory
from .smearing import Smearing

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"


def calculate_sigma(langevin_nbar, langevin_dt, n_grids, volume):
    """Calculate the standard deviation of Langevin noise.

    Parameters
    ----------
    langevin_nbar : float
        Invariant polymerization index.
    langevin_dt : float
        Langevin time step.
    n_grids : int
        Total number of grid points.
    volume : float
        Box volume.

    Returns
    -------
    float
        Standard deviation of the Gaussian noise.
    """
    return np.sqrt(2 * langevin_dt * n_grids / (volume * np.sqrt(langevin_nbar)))


class CLFTS:
    """Complex Langevin Field-Theoretic Simulation class.

    This class implements Complex Langevin dynamics for polymer field theory
    simulations. Unlike standard L-FTS which uses real-valued fields, CL-FTS
    extends fields to the complex plane to handle systems where the action
    has a non-zero imaginary part (sign problem).

    Parameters
    ----------
    params : dict
        Simulation parameters dictionary containing:
        - nx : list of int - Grid points in each dimension
        - lx : list of float - Box size in each dimension
        - chain_model : str - "discrete" or "continuous"
        - ds : float - Contour step size
        - segment_lengths : dict - Statistical segment lengths
        - chi_n : dict - Flory-Huggins parameters
        - distinct_polymers : list - Polymer specifications
        - langevin : dict - Langevin dynamics parameters
        - recording : dict - Data recording parameters
        - verbose_level : int - Output verbosity (1 or 2)
        - dynamic_stabilization : float, optional - Stabilization coefficient αds
          to prevent hot spots by adding iαds·Im[W] to exchange field forces.
          Typical values: 0.01-0.1 for N̄ ≥ 10^5 (Willis & Matsen 2024)
        - alpha_ds : float, optional - Alias for dynamic_stabilization parameter
        - smearing : dict, optional - Finite-range interaction parameters.
          Smearing helps stabilize CL-FTS by using finite-range interactions
          instead of contact forces. Supported types:

          * Gaussian: {"type": "gaussian", "a_int": float}
            Γ(k) = exp(-a_int² · k² / 2)
            a_int is the smearing length in units of R₀ (unperturbed chain size).
            Universality requires a_int ≲ 0.02·R₀.

          * Sigmoidal: {"type": "sigmoidal", "k_int": float, "dk_int": float}
            Γ(k) = C₀[1 - tanh((k - k_int) / Δk_int)]
            k_int is the cutoff wavenumber, dk_int controls transition width.
            Universality requires k_int ≳ 20/R₀. Default dk_int = 5.0.

          Reference: Matsen et al., J. Chem. Phys. 164, 014905 (2026)
        - platform : str, optional - "cuda" or "cpu-mkl"
        - reduce_memory_usage : bool, optional - Memory optimization flag
    random_seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    monomer_types : list
        Sorted list of monomer type names.
    chi_n : dict
        Flory-Huggins interaction parameters.
    mpt : SymmetricPolymerTheory
        Multi-component polymer field theory handler.

    Examples
    --------
    >>> params = {
    ...     "nx": [32, 32],
    ...     "lx": [4.0, 4.0],
    ...     "chain_model": "discrete",
    ...     "ds": 0.01,
    ...     "segment_lengths": {"A": 1.0, "B": 1.0},
    ...     "chi_n": {"A,B": 15.0},
    ...     "distinct_polymers": [{
    ...         "volume_fraction": 1.0,
    ...         "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}],
    ...     }],
    ...     "langevin": {"max_step": 1000, "dt": 1.0, "nbar": 10000},
    ...     "recording": {"dir": "data", "recording_period": 100,
    ...                   "sf_computing_period": 10, "sf_recording_period": 100},
    ...     "verbose_level": 1,
    ... }
    >>> sim = CLFTS(params)
    >>> sim.run(initial_fields={"A": np.zeros(1024), "B": np.zeros(1024)})
    """

    def __init__(self, params, random_seed=None):
        """Initialize the Complex Langevin FTS simulation."""

        # Segment length
        self.monomer_types = sorted(list(params["segment_lengths"].keys()))
        self.segment_lengths = copy.deepcopy(params["segment_lengths"])
        self.distinct_polymers = copy.deepcopy(params["distinct_polymers"])

        assert len(self.monomer_types) == len(set(self.monomer_types)), \
            "There are duplicated monomer_types"

        # Choose platform among [cuda, cpu-mkl]
        avail_platforms = _core.PlatformSelector.avail_platforms()
        if "platform" in params:
            platform = params["platform"]
        elif "cpu-mkl" in avail_platforms and len(params["nx"]) == 1:
            platform = "cpu-mkl"
        elif "cuda" in avail_platforms:
            platform = "cuda"
        else:
            platform = avail_platforms[0]

        # Get reduce_memory_usage option
        reduce_memory_usage = params.get("reduce_memory_usage", False)

        # (C++ class) Create a factory for given platform and chain_model
        # Use "complex" type for complex Langevin
        factory = _core.PlatformSelector.create_factory(platform, reduce_memory_usage, "complex")
        factory.display_info()

        # (C++ class) Computation box
        if "angles" in params:
            self.cb = factory.create_computation_box(params["nx"], params["lx"], angles=params["angles"])
        else:
            self.cb = factory.create_computation_box(params["nx"], params["lx"])

        # Flory-Huggins parameters, chi_n
        self.chi_n = {}
        for monomer_pair_str, chin_value in params["chi_n"].items():
            monomer_pair = re.split(',| |_|/', monomer_pair_str)
            assert monomer_pair[0] in self.segment_lengths, \
                f"Monomer type '{monomer_pair[0]}' is not in 'segment_lengths'."
            assert monomer_pair[1] in self.segment_lengths, \
                f"Monomer type '{monomer_pair[1]}' is not in 'segment_lengths'."
            assert monomer_pair[0] != monomer_pair[1], \
                "Do not add self interaction parameter, " + monomer_pair_str + "."
            monomer_pair.sort()
            sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1]
            assert sorted_monomer_pair not in self.chi_n, \
                f"There are duplicated chi_n ({sorted_monomer_pair}) parameters."
            self.chi_n[sorted_monomer_pair] = chin_value

        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            monomer_pair = list(monomer_pair)
            monomer_pair.sort()
            sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1]
            if sorted_monomer_pair not in self.chi_n:
                self.chi_n[sorted_monomer_pair] = 0.0

        # Compressibility parameter (zeta_n)
        if "zeta_n" in params:
            self.zeta_n = params["zeta_n"]
        else:
            self.zeta_n = 100.0  # Default high compressibility

        # Multimonomer polymer field theory
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n, self.zeta_n)

        # The numbers of real and imaginary fields
        M = len(self.monomer_types)
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        # Total volume fraction
        assert len(self.distinct_polymers) >= 1, \
            "There is no polymer chain."

        total_volume_fraction = 0.0
        for polymer in self.distinct_polymers:
            total_volume_fraction += polymer["volume_fraction"]
        assert np.isclose(total_volume_fraction, 1.0), \
            "The sum of volume fractions must be equal to 1."

        # Polymer chains
        for polymer_counter, polymer in enumerate(self.distinct_polymers):
            blocks_input = []
            alpha = 0.0  # total_relative_contour_length
            has_node_number = "v" not in polymer["blocks"][0]
            for block in polymer["blocks"]:
                alpha += block["length"]
                if has_node_number:
                    assert "v" not in block, \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer."
                    assert "u" not in block, \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer."
                    blocks_input.append([block["type"], block["length"], len(blocks_input), len(blocks_input) + 1])
                else:
                    assert "v" in block, \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer."
                    assert "u" in block, \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer."
                    blocks_input.append([block["type"], block["length"], block["v"], block["u"]])
            polymer.update({"blocks_input": blocks_input})

        # Random copolymer chains
        self.random_fraction = {}
        for polymer in self.distinct_polymers:
            is_random = False
            for block in polymer["blocks"]:
                if "fraction" in block:
                    is_random = True
            if not is_random:
                continue

            assert len(polymer["blocks"]) == 1, \
                "Only single block random copolymer is allowed."

            statistical_segment_length = 0
            total_random_fraction = 0
            for monomer_type in polymer["blocks"][0]["fraction"]:
                statistical_segment_length += self.segment_lengths[monomer_type] ** 2 * polymer["blocks"][0]["fraction"][monomer_type]
                total_random_fraction += polymer["blocks"][0]["fraction"][monomer_type]
            statistical_segment_length = np.sqrt(statistical_segment_length)

            assert np.isclose(total_random_fraction, 1.0), \
                "The sum of volume fractions of random copolymer must be equal to 1."

            random_type_string = polymer["blocks"][0]["type"]
            assert random_type_string not in self.segment_lengths, \
                f"The name of random copolymer '{random_type_string}' is already used as a type in 'segment_lengths' or other random copolymer"

            # Add random copolymers
            polymer["block_monomer_types"] = [random_type_string]
            self.segment_lengths.update({random_type_string: statistical_segment_length})
            self.random_fraction[random_type_string] = polymer["blocks"][0]["fraction"]

        # (C++ class) Molecules list
        molecules = factory.create_molecules_information(params["chain_model"], params["ds"], self.segment_lengths)

        # Add polymer chains
        for polymer in self.distinct_polymers:
            molecules.add_polymer(polymer["volume_fraction"], polymer["blocks_input"])

        # (C++ class) Propagator Computation Optimizer
        if "aggregate_propagator_computation" in params:
            propagator_computation_optimizer = factory.create_propagator_computation_optimizer(
                molecules, params["aggregate_propagator_computation"]
            )
        else:
            propagator_computation_optimizer = factory.create_propagator_computation_optimizer(molecules, True)

        # (C++ class) Solver using Pseudo-spectral method
        self.solver = factory.create_pseudospectral_solver(self.cb, molecules, propagator_computation_optimizer)

        # Standard deviation of normal noise of Langevin dynamics
        langevin_sigma = calculate_sigma(
            params["langevin"]["nbar"],
            params["langevin"]["dt"],
            np.prod(params["nx"]),
            np.prod(params["lx"])
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
            self.dt_scaling[i] = np.abs(self.mpt.eigenvalues[i]) / np.max(np.abs(self.mpt.eigenvalues))

        # Set random generator
        if random_seed is None:
            self.random_bg = np.random.PCG64()
        else:
            self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)

        print("---------- Simulation Parameters ----------")
        print("Platform:", platform)
        print("Box Dimension: %d" % (self.cb.get_dim()))
        print("Nx:", self.cb.get_nx())
        print("Lx:", self.cb.get_lx())
        print("dx:", self.cb.get_dx())
        print("Volume: %f" % (self.cb.get_volume()))

        print("Chain model: %s" % (params["chain_model"]))
        print("Segment lengths:\n\t", list(self.segment_lengths.items()))
        print("Conformational asymmetry (epsilon):")
        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1],
                                   self.segment_lengths[monomer_pair[0]] / self.segment_lengths[monomer_pair[1]]))

        print("chi_n:")
        for key in self.chi_n:
            print("\t%s: %f" % (key, self.chi_n[key]))

        print("zeta_n: %f" % (self.zeta_n))

        molecules.display_architectures()

        print("Invariant Polymerization Index (N_Ref): %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Scaling factor of delta tau N for each field:", self.dt_scaling)
        print("Dynamical stabilization constant: %f" % (self.alpha_ds))
        print("Random Number Generator:", self.random_bg.state)

        propagator_computation_optimizer.display_statistics()

        # Save internal variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma": langevin_sigma})

        self.verbose_level = params["verbose_level"]
        self.recording = params["recording"].copy()

        self.molecules = molecules
        self.propagator_computation_optimizer = propagator_computation_optimizer

        # Initialize smearing (must be after self.cb is set)
        self.smearing = Smearing(
            self.cb.get_nx(),
            self.cb.get_lx(),
            params.get("smearing", None)
        )

    def compute_concentrations(self, w_aux):
        """Compute monomer concentration fields from auxiliary fields.

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
        w_input = {}
        for i in range(M):
            w_input[self.monomer_types[i]] = w[i]
        for random_polymer_name, random_fraction in self.random_fraction.items():
            w_input[random_polymer_name] = np.zeros(self.cb.get_total_grid(), dtype=np.complex128)
            for monomer_type, fraction in random_fraction.items():
                w_input[random_polymer_name] += w_input[monomer_type] * fraction

        # Apply smearing to fields before computing propagators
        # Following Matsen et al., J. Chem. Phys. 164, 014905 (2026), the fields
        # are smeared before computing the Boltzmann weights in the propagator.
        w_input_for_propagator = self.smearing.apply_to_dict(w_input)

        # For the given fields, compute propagators
        time_solver_start = time.time()
        self.solver.compute_propagators(w_input_for_propagator)
        elapsed_time["solver"] = time.time() - time_solver_start

        # Compute concentrations for each monomer type
        time_phi_start = time.time()
        phi = {}
        self.solver.compute_concentrations()
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.solver.get_total_concentration(monomer_type)

        # Add random copolymer concentration to each monomer type
        for random_polymer_name, random_fraction in self.random_fraction.items():
            phi[random_polymer_name] = self.solver.get_total_concentration(random_polymer_name)
            for monomer_type, fraction in random_fraction.items():
                phi[monomer_type] += phi[random_polymer_name] * fraction
        elapsed_time["phi"] = time.time() - time_phi_start

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
        # Make a dictionary for chi_n
        chi_n_mat = {}
        for key in self.chi_n:
            chi_n_mat[key] = self.chi_n[key]

        # Make dictionary for data
        mdic = {
            "initial_params": self.params,
            "dim": self.cb.get_dim(),
            "nx": self.cb.get_nx(),
            "lx": self.cb.get_lx(),
            "monomer_types": self.monomer_types,
            "chi_n": chi_n_mat,
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

        Notes
        -----
        The Complex Langevin dynamics evolves fields according to
        (Willis & Matsen, J. Chem. Phys. 161, 244903 (2024), Eqs. 8-10):

            W-(t+dt) = W-(t) - Λ-*dt + N(0,σ)      [real noise]
            W+(t+dt) = W+(t) + Λ+*dt + i*N(0,σ)    [imaginary noise]

        where:
        - W- is the exchange/composition field (real-type auxiliary field)
        - W+ is the pressure field (imaginary-type auxiliary field)
        - Λ± are the deterministic forces including functional derivatives

        Dynamical stabilization adds iαds·N·Im[W-] to Λ- to prevent
        runaway solutions in the imaginary direction (hot spots).
        """
        print("---------- Run ----------")

        # The number of components
        M = len(self.monomer_types)

        # The numbers of real and imaginary fields
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        # Simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # Reshape initial fields (convert to complex if needed)
        w = np.zeros([M, self.cb.get_total_grid()], dtype=np.complex128)
        for i in range(M):
            field = initial_fields[self.monomer_types[i]]
            w[i] = np.reshape(field, self.cb.get_total_grid()).astype(np.complex128)

        # Convert monomer chemical potential fields into auxiliary fields
        w_aux = self.mpt.to_aux_fields(w)

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
                np.fft.fftn(np.reshape(w[0], self.cb.get_nx())), np.complex128
            )

        # Create an empty array for field update algorithm
        if normal_noise_prev is None:
            normal_noise_prev = np.zeros([M, self.cb.get_total_grid()], dtype=np.float64)

        if start_langevin_step is None:
            start_langevin_step = 1

        # Init timers
        time_start = time.time()

        # Langevin iteration begins here
        for langevin_step in range(start_langevin_step, self.langevin["max_step"] + 1):

            # Compute total concentrations
            phi, _ = self.compute_concentrations(w_aux=w_aux)

            # Apply smearing to concentrations for force calculation
            # Smearing implements finite-range interactions: φ_Γ = FFT⁻¹[FFT[φ]·Γ(k)]
            # Reference: Matsen et al., J. Chem. Phys. 164, 014905 (2026)
            phi_for_force = self.smearing.apply_to_dict(phi)

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
                0.0, self.langevin["sigma"], [M, self.cb.get_total_grid()]
            )
            for i in range(M):
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

            # Compute functional derivatives for error monitoring (use smeared phi for consistency)
            h_deriv = self.mpt.compute_func_deriv(w_aux, phi_for_force, range(M))

            # Compute error levels
            error_level_array = np.std(h_deriv, axis=1)

            # Calculate Hamiltonian
            total_partitions = [self.solver.get_total_partition(p) for p in range(self.molecules.get_n_polymer_types())]
            hamiltonian = self.mpt.compute_hamiltonian(self.molecules, w_aux, total_partitions, include_const_term=True)

            # Print progress
            if self.verbose_level >= 1:
                mass_error = np.mean(h_deriv[M - 1])
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
                dH = self.mpt.compute_h_deriv_chin(self.chi_n, w_aux)
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
                # Perform Fourier transforms (full FFT for complex fields)
                mu_fourier = {}
                phi_fourier = {}
                for i in range(M):
                    key = self.monomer_types[i]
                    phi_fourier[key] = np.fft.fftn(
                        np.reshape(phi[self.monomer_types[i]], self.cb.get_nx())
                    ) / self.cb.get_total_grid()
                    mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
                    for k in range(M - 1):
                        mu_fourier[key] += (
                            np.fft.fftn(np.reshape(w_aux[k], self.cb.get_nx())) *
                            self.mpt.matrix_a_inv[k, i] / self.mpt.eigenvalues[k] / self.cb.get_total_grid()
                        )
                # Accumulate S_ij(K)
                for key in sf_average:
                    monomer_pair = sorted(key.split(","))
                    sf_average[key] += mu_fourier[monomer_pair[0]] * np.conj(phi_fourier[monomer_pair[1]])

            # Save structure function
            if langevin_step % self.recording["sf_recording_period"] == 0:
                # Make a dictionary for chi_n
                chi_n_mat = {}
                for key in self.chi_n:
                    monomer_pair = sorted(key.split(","))
                    chi_n_mat[monomer_pair[0] + "," + monomer_pair[1]] = self.chi_n[key]
                mdic = {
                    "dim": self.cb.get_dim(),
                    "nx": self.cb.get_nx(),
                    "lx": self.cb.get_lx(),
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
                        self.cb.get_volume() * np.sqrt(self.langevin["nbar"])
                    )
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

        print("total time: %f, time per step: %f" %
              (time_duration, time_duration / (langevin_step + 1 - start_langevin_step)))
