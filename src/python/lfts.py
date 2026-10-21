import os
import time
import pathlib
import copy
import numpy as np
import itertools
from scipy.io import savemat, loadmat

from .polymer_field_theory import SymmetricPolymerTheory
from .propagator_solver import PropagatorSolver
from . import _core
from .validation import validate_lfts_params, ValidationError
from .utils import (
    parse_chi_n,
    process_polymer_blocks,
    process_random_copolymer,
    draw_polymer_architectures,
    calculate_sigma,
    load_config,
)
from .smearing import Smearing
from .wtmd import WTMD

# OpenMP environment variables
os.environ.setdefault("OMP_NUM_THREADS", "1")  # respect user override
os.environ.setdefault("OMP_STACKSIZE", "1G")


class LR:
    """Linear Response (LR) field compressor for L-FTS.

    Uses the local Hessian (second derivative matrix) of the Hamiltonian
    to find the saddle point of imaginary auxiliary fields via linear response
    theory. This is faster than iterative methods but requires computing and
    inverting the Hessian matrix.

    The update is: w_new = w_old - J^(-1) · δH/δw, where J is the Jacobian
    (Hessian) in Fourier space.

    Parameters
    ----------
    nx : list of int
        Grid dimensions [nx, ny, nz].
    lx : list of float
        Box dimensions in units of a_Ref * N_Ref^(1/2).
    jk : ndarray
        Jacobian (Hessian) matrix in Fourier space. Precomputed from
        finite difference of δH/δw. Shape matches rfft output of nx.

    Attributes
    ----------
    nx : list of int
        Grid dimensions.
    lx : list of float
        Box dimensions.
    jk : ndarray
        Jacobian in Fourier space.
    """
    def __init__(self, nx, lx, jk):
        self.nx = nx
        self.lx = lx
        self.jk = jk.copy()

    def reset_count(self,):
        """Reset compressor state (no-op for LR).

        LR is stateless, so this method does nothing. Included for
        compatibility with Anderson Mixing interface.
        """
        pass

    def calculate_new_fields(self, w_current, negative_h_deriv, old_error_level, error_level):
        """Compute new fields using linear response method.

        Performs one linear response update to move toward the saddle point.

        Parameters
        ----------
        w_current : ndarray
            Current field values, shape (1, total_grid) or (total_grid,).
        negative_h_deriv : ndarray
            Negative functional derivative -δH/δw, same shape as w_current.
        old_error_level : float
            Previous error level (not used by LR, kept for interface compatibility).
        error_level : float
            Current error level (not used by LR, kept for interface compatibility).

        Returns
        -------
        w_new : ndarray
            Updated field values, same shape as w_current.
        """
        nx = self.nx
        negative_h_deriv_k = np.fft.rfftn(np.reshape(negative_h_deriv, nx))/np.prod(nx)
        w_diff_k = negative_h_deriv_k/self.jk
        w_diff = np.reshape(np.fft.irfftn(w_diff_k, nx, axes=tuple(range(len(nx))))*np.prod(nx), np.shape(w_current))

        return w_current + w_diff

class LRAM:
    """Linear Response + Anderson Mixing (LRAM) field compressor.

    Combines Linear Response for coarse optimization with Anderson Mixing for
    final refinement. This hybrid approach is more robust than LR alone while
    being faster than pure Anderson Mixing.

    The algorithm:
    1. Use LR to get approximate update
    2. Use AM to refine the LR result

    This combines the speed of LR with the robustness of AM.

    Parameters
    ----------
    lr : LR
        Linear Response compressor instance.
    am : AndersonMixing
        Anderson Mixing instance (from C++ backend).

    Attributes
    ----------
    lr : LR
        Linear Response compressor.
    am : AndersonMixing
        Anderson Mixing optimizer.
    """
    def __init__(self, lr, am):
        self.lr = lr
        self.am = am

    def reset_count(self,):
        """Reset Anderson Mixing history.

        Clears the AM history buffer. Should be called when starting a new
        saddle point search (e.g., at beginning of each Langevin step).
        """
        self.am.reset_count()

    def calculate_new_fields(self, w_current, negative_h_deriv, old_error_level, error_level):
        """Compute new fields using LRAM hybrid method.

        Performs Linear Response followed by Anderson Mixing refinement.

        Parameters
        ----------
        w_current : ndarray
            Current field values, shape (1, total_grid) or (total_grid,).
        negative_h_deriv : ndarray
            Negative functional derivative -δH/δw, same shape as w_current.
        old_error_level : float
            Previous error level (used by AM for mixing parameter adjustment).
        error_level : float
            Current error level (used by AM for mixing parameter adjustment).

        Returns
        -------
        w_new : ndarray
            Updated field values after LR + AM, same shape as w_current.
        """
        nx = self.lr.nx

        w_lr_old = np.reshape(w_current, [1, np.prod(nx)])
        w_lr_new = np.reshape(self.lr.calculate_new_fields(w_lr_old, negative_h_deriv, old_error_level, error_level), [1, np.prod(nx)])
        w_diff = w_lr_new - w_lr_old
        w_new = self.am.calculate_new_fields(w_lr_new, w_diff, old_error_level, error_level)

        return np.reshape(w_new, np.shape(w_current))


class LFTS:
    """Langevin field-theoretic simulation (L-FTS) of polymer melts and solutions.

    Samples the field-theoretic partition function with Langevin dynamics
    (Leimkuhler-Matthews scheme) of the real auxiliary fields, while the
    imaginary fields are compressed to their saddle point every step with
    Anderson mixing, linear response, or both (LRAM). Records fields,
    structure functions and d H / d chi N at the requested periods.

    Parameters
    ----------
    params : dict
        Simulation parameters; every key is documented in
        docs/references/Parameters.md. Required: the SCFT system keys plus
        langevin, compressor, saddle, recording, verbose_level.
    random_seed : int, optional
        Seed of the PCG64 noise generator (None: non-reproducible).

    Attributes
    ----------
    prop_solver : PropagatorSolver
        Propagator computation (C++ objects inside).
    cb : ComputationBox
        Grid, box and integrals (C++).
    mpt : SymmetricPolymerTheory
        Monomer <-> auxiliary field transformations.
    langevin, saddle, recording : dict
        Copies of the corresponding parameter dictionaries (langevin also holds 'sigma').

    Examples
    --------
    >>> sim = LFTS(params, random_seed=12345)
    >>> sim.run(initial_fields={"A": w_A, "B": w_B})   # writes data/fields_*.mat, structure_function_*.mat
    >>> sim.continue_run("data/fields_001000.mat")

    Complete scripts: examples/lfts/ (Lamella.py, Gyroid.py, MixtureBlockRandom.py).
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

        # Validate input parameters
        validate_lfts_params(params)
        for key in ("compressor", "saddle", "recording", "verbose_level"):
            if key not in params:
                raise ValidationError(f"Missing required parameter for LFTS: '{key}'")
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

        # Compressibility parameter (None: incompressible)
        self.zeta_n = params.get("zeta_n", None)

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
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n, zeta_n=self.zeta_n)

        # The number of monomer types
        M = len(self.monomer_types)

        # Propagator solver: the C++ computation box, molecules and solver live inside it
        self.prop_solver = PropagatorSolver(
            nx=params["nx"], lx=params["lx"], ds=params["ds"],
            bond_lengths=self.segment_lengths, bc=bc,
            chain_model=params["chain_model"],
            numerical_method=params.get("numerical_method", None),   # None: "rqm4"
            platform=params.get("platform", "auto"),   # auto: CPU for 1D, CUDA for 2D/3D when available
            reduce_memory=params.get("reduce_memory", False),
            angles=params.get("angles", None),
            aggregate_propagator_computation=params.get("aggregate_propagator_computation", True),
        )
        for polymer in self.distinct_polymers:
            self.prop_solver.add_polymer(polymer["volume_fraction"], polymer["blocks_input"],
                                         polymer.get("initial_conditions", None))

        # Computation box as a public attribute (this creates the C++ objects)
        self.cb = self.prop_solver.computation_box

        # Display factory info
        self.prop_solver.display_platform_info()

        # Initialize smearing (must be after self.cb is set)
        self.smearing = Smearing(
            self.cb.get_nx(),
            self.cb.get_lx(),
            params.get("smearing", None),
            angles=self.prop_solver.get_angles_degrees(),
            bc=self.prop_solver.bc
        )

        # Initialize Well-Tempered Metadynamics (WTMD) if parameters provided
        # Uses same interface as deep-langevin-fts
        wtmd_params = params.get("wtmd", None)
        if wtmd_params is not None:
            self.wtmd = WTMD(
                nx=list(self.cb.get_nx()),
                lx=list(self.cb.get_lx()),
                angles=self.prop_solver.get_angles_degrees(),
                nbar=params["langevin"]["nbar"],
                eigenvalues=self.mpt.eigenvalues,
                real_fields_idx=self.mpt.aux_fields_real_idx,
                l=wtmd_params.get("ell", 4),
                kc=wtmd_params.get("kc", 6.02),
                dT=wtmd_params.get("delta_t", 5.0),
                sigma_psi=wtmd_params.get("sigma_psi", 0.16),
                psi_min=wtmd_params.get("psi_min", 0.0),
                psi_max=wtmd_params.get("psi_max", 10.0),
                dpsi=wtmd_params.get("dpsi", 1e-3),
                update_freq=wtmd_params.get("update_freq", 1000),
                recording_period=wtmd_params.get("recording_period", 100000),
            )
            print("WTMD enabled: l=%d, sigma_psi=%.4f, dT=%.4f, kc=%.4f, exchange_idx=%d, langevin_idx=%d" % (
                self.wtmd.l, self.wtmd.sigma_psi, self.wtmd.dT, self.wtmd.kc,
                self.wtmd.exchange_idx, self.wtmd.langevin_idx))
        else:
            self.wtmd = None

        # (C++ class) Fields Relaxation using Anderson Mixing
        if params["compressor"]["name"] == "am" or params["compressor"]["name"] == "lram":
            n_am = int(len(self.mpt.aux_fields_imag_idx)*np.prod(params["nx"]))
            if n_am > _core.MAX_GRID_ARRAY:
                raise ValidationError(
                    f"Anderson mixing would adjust {n_am} variables ({len(self.mpt.aux_fields_imag_idx)} imaginary fields "
                    f"of {int(np.prod(params['nx']))} points); at most {_core.MAX_GRID_ARRAY} are supported (32-bit indices).")
            am = self.prop_solver.create_anderson_mixing(
                n_am,                                                      # the number of variables
                params["compressor"]["max_hist"],                          # maximum number of history
                params["compressor"]["start_error"],                       # when switch to AM from simple mixing
                params["compressor"]["mix_min"],                           # minimum mixing rate of simple mixing
                params["compressor"]["mix_init"])                          # initial mixing rate of simple mixing

        # Fields Relaxation using Linear Response Method
        if params["compressor"]["name"] == "lr" or params["compressor"]["name"] == "lram":
            if len(self.mpt.aux_fields_imag_idx) != 1:
                raise ValidationError("Currently, LR methods are not working for imaginary-valued auxiliary fields.")

            # Baseline: h_deriv at w=0
            w_aux_zero = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
            phi_zero, _ = self.compute_concentrations(w_aux_zero)
            h_deriv_zero = self.mpt.compute_func_deriv(w_aux_zero, self.smearing.apply_to_dict(phi_zero), self.mpt.aux_fields_imag_idx)

            # Perturbed: h_deriv at w with small perturbation
            w_aux_perturbed = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
            w_aux_perturbed[M-1,0] = 1e-3 # add a small perturbation at the pressure field
            w_aux_perturbed_k = np.fft.rfftn(np.reshape(w_aux_perturbed[M-1], self.prop_solver.nx))/np.prod(self.prop_solver.nx)

            phi_perturbed, _ = self.compute_concentrations(w_aux_perturbed)
            h_deriv_perturbed = self.mpt.compute_func_deriv(w_aux_perturbed, self.smearing.apply_to_dict(phi_perturbed), self.mpt.aux_fields_imag_idx)

            # jk = (h(w_pert) - h(0)) / w_pert  (correct finite difference)
            h_deriv_diff = h_deriv_perturbed - h_deriv_zero
            h_deriv_diff_k = np.fft.rfftn(np.reshape(h_deriv_diff, self.prop_solver.nx))/np.prod(self.prop_solver.nx)
            jk_numeric = np.real(h_deriv_diff_k/w_aux_perturbed_k)

            lr = LR(self.prop_solver.nx, self.prop_solver.get_lx(), jk_numeric)

        if params["compressor"]["name"] == "am":
            self.compressor = am
        elif params["compressor"]["name"] == "lr":
            self.compressor = lr
        elif params["compressor"]["name"] == "lram":
            self.compressor = LRAM(lr, am)
        else:
            raise ValidationError(f"compressor['name'] must be 'am', 'lr' or 'lram', got '{params['compressor']['name']}'")

        # Standard deviation of normal noise of Langevin dynamics
        langevin_sigma = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"], np.prod(params["nx"]), self.prop_solver.get_volume())

        # dH/dw_aux[i] is scaled by dt_scaling[i]
        self.dt_scaling = np.ones(M)
        for i in range(M-1):
            self.dt_scaling[i] = np.abs(self.mpt.eigenvalues[i])/np.max(np.abs(self.mpt.eigenvalues[:M-1]))

        # Set random generator
        if random_seed is None:         
            self.random_bg = np.random.PCG64()  # Set random bit generator
        else:
            self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)
        
        print("---------- Simulation Parameters ----------")
        print("Platform:", self.prop_solver.platform)
        print("Box Dimension: %d" % (self.prop_solver.dim))
        print("Nx:", self.prop_solver.nx)
        print("Lx [a, b, c]:", self.prop_solver.get_lx())
        angles_deg = list(self.prop_solver.get_angles_degrees())
        print("Angles [alpha, beta, gamma]:", angles_deg)
        print("dx:", self.prop_solver.get_dx())
        print("Volume: %f" % (self.prop_solver.get_volume()))

        print("Chain model: %s" % (params["chain_model"]))
        print("Segment lengths:\n\t", list(self.segment_lengths.items()))
        print("Conformational asymmetry (epsilon): ")
        for monomer_pair in itertools.combinations(self.monomer_types,2):
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1], self.segment_lengths[monomer_pair[0]]/self.segment_lengths[monomer_pair[1]]))

        print("χN: ")
        for key in self.chi_n:
            print("\t%s: %f" % (key, self.chi_n[key]))
        if self.zeta_n is not None:
            print("zeta_n: %f" % (self.zeta_n))

        self.prop_solver.display_architectures()

        print("Invariant Polymerization Index (N_Ref): %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Scaling factor of delta tau N for each field: ", self.dt_scaling)
        print("Random Number Generator: ", self.random_bg.state)

        self.prop_solver.display_statistics()

        #  Save internal variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma":langevin_sigma})

        self.verbose_level = params["verbose_level"]
        self.saddle = params["saddle"].copy()
        self.recording = params["recording"].copy()

    def compute_concentrations(self, w_aux):
        """Compute monomer concentration fields from auxiliary fields.

        NOTE: this takes AUXILIARY fields (exchange-mapped), unlike
        SCFT.compute_concentrations which takes monomer potential fields.

        Converts auxiliary fields to monomer potential fields via inverse
        field transformation, then computes chain propagators and monomer
        concentrations.

        Parameters
        ----------
        w_aux : ndarray
            Auxiliary potential fields, shape (M, total_grid) where M is
            the number of auxiliary fields (= number of monomer types).

        Returns
        -------
        phi : dict
            Monomer concentration fields, {monomer_type: concentration_array}.
        elapsed_time : dict
            Timing information:

            - 'solver' : float
                Time (seconds) for propagator computation.
            - 'phi' : float
                Time (seconds) for concentration integration.
        """
        M = len(self.monomer_types)
        elapsed_time = {}

        # Convert auxiliary fields to monomer fields
        w = self.mpt.to_monomer_fields(w_aux)

        # Make a dictionary for input fields
        w_input = {self.monomer_types[i]: w[i] for i in range(M)}
        for random_polymer_name, random_fraction in self.random_fraction.items():
            w_input[random_polymer_name] = np.zeros(self.prop_solver.n_grid, dtype=np.float64)
            for monomer_type, fraction in random_fraction.items():
                w_input[random_polymer_name] += w_input[monomer_type]*fraction

        # Apply smearing to fields before computing propagators
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
                phi[monomer_type] += phi[random_polymer_name]*fraction
        elapsed_time["phi"] = time.time() - time_phi_start

        # Material conservation: mean(sum phi) must stay 1 (a numerical blow-up shows up here first)
        mass_error = self.prop_solver.mean(sum(phi[m] for m in self.monomer_types) - 1.0)
        if abs(mass_error) > 5e-8:
            raise ValidationError(f"Material conservation failed: mean(sum(phi) - 1) = {mass_error:.3e}")

        # Partition consistency check (forward vs backward propagators), once per run
        if not getattr(self, "_partition_checked", False):
            self._partition_checked = True
            if not self.prop_solver.check_total_partition():
                raise ValidationError("Partition consistency check failed (forward/backward mismatch).")

        return phi, elapsed_time

    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev):
        """Save L-FTS simulation checkpoint data to MATLAB file.

        Saves the current state of the simulation including fields, concentrations,
        random generator state, and all parameters needed to continue the simulation.

        Parameters
        ----------
        path : str
            Output file path (should end with '.mat').
        w : ndarray
            Current monomer potential fields, shape (M, total_grid).
        phi : dict
            Current monomer concentration fields, {monomer_type: array}.
        langevin_step : int
            Current Langevin iteration number.
        normal_noise_prev : ndarray
            Previous Langevin noise (needed for Leimkuhler-Matthews method),
            shape (R, total_grid) where R is number of real auxiliary fields.
        """

        # Make dictionary for data
        mdic = {
            "initial_params": self.params,
            "dim":self.prop_solver.dim, "nx":self.prop_solver.nx, "lx":self.prop_solver.get_lx(),
            "monomer_types":self.monomer_types, "chi_n":self.chi_n, "chain_model":self.chain_model, "ds":self.ds,
            "dt":self.langevin["dt"], "nbar":self.langevin["nbar"],
            "eigenvalues": self.mpt.eigenvalues,
            "aux_fields_real": self.mpt.aux_fields_real_idx,
            "aux_fields_imag": self.mpt.aux_fields_imag_idx,
            "matrix_a": self.mpt.matrix_a, "matrix_a_inverse": self.mpt.matrix_a_inv,
            "langevin_step":langevin_step,
            "random_generator":self.random_bg.state["bit_generator"],
            "random_state_state":str(self.random_bg.state["state"]["state"]),
            "random_state_inc":str(self.random_bg.state["state"]["inc"]),
            "normal_noise_prev":normal_noise_prev}

        # Add w fields to the dictionary
        for i, name in enumerate(self.monomer_types):
            mdic["w_" + name] = w[i]
        
        # Add concentrations to the dictionary
        for name in self.monomer_types:
            mdic["phi_" + name] = phi[name]

        # Add WTMD state if enabled (same format as deep-langevin-fts)
        if self.wtmd is not None:
            mdic["wtmd_u"] = self.wtmd.u * self.wtmd.CV
            mdic["wtmd_up"] = self.wtmd.up * self.wtmd.CV
            mdic["wtmd_I0"] = self.wtmd.I0
            # I1 and the samples stored since the last update, so that a continued run
            # carries the same statistics as an uninterrupted one
            for key, value in self.wtmd.I1.items():
                monomer_pair = sorted(key.split(","))
                mdic["wtmd_I1_" + monomer_pair[0] + "_" + monomer_pair[1]] = value * self.wtmd.CV
            mdic["wtmd_order_parameter_history"] = np.array(self.wtmd.order_parameter_history, dtype=np.float64)
            for key, values in self.wtmd.dH_history.items():
                monomer_pair = sorted(key.split(","))
                mdic["wtmd_dH_history_" + monomer_pair[0] + "_" + monomer_pair[1]] = np.array(values, dtype=np.float64)
            mdic["wtmd_params"] = {
                "l": self.wtmd.l,
                "sigma_psi": self.wtmd.sigma_psi,
                "dT": self.wtmd.dT,
                "kc": self.wtmd.kc,
                "psi_min": self.wtmd.psi_min,
                "psi_max": self.wtmd.psi_max,
                "dpsi": self.wtmd.dpsi,
                "bins": self.wtmd.bins,
                "update_freq": self.wtmd.update_freq,
                "recording_period": self.wtmd.recording_period,
            }

        # Save data with matlab format
        savemat(path, mdic, long_field_names=True, do_compression=True)

    def continue_run(self, file_name):
        """Continue L-FTS simulation from a saved checkpoint file.

        Loads simulation state from a MATLAB .mat file (created by save_simulation_data)
        and continues the Langevin dynamics from where it left off, preserving the
        random number generator state for reproducibility.

        Parameters
        ----------
        file_name : str
            Path to checkpoint file (MATLAB .mat format created by save_simulation_data).
        """

        # Load_data
        load_data = loadmat(file_name, squeeze_me=True)
        
        # Check if load_data["langevin_step"] is a multiple of self.recording["sf_recording_period"]
        if load_data["langevin_step"] % self.recording["sf_recording_period"] != 0:
            print(f"(Warning!) 'langevin_step' of {file_name} is not a multiple of 'sf_recording_period'.")
            next_sf_langevin_step = (load_data["langevin_step"]//self.recording["sf_recording_period"] + 1)*self.recording["sf_recording_period"]
            print(f"The structure function will be correctly recorded after {next_sf_langevin_step}th langevin_step." )

        # Restore random state
        self.random_bg.state ={'bit_generator': 'PCG64',
            'state': {'state': int(load_data["random_state_state"]),
                      'inc':   int(load_data["random_state_inc"])},
                      'has_uint32': 0, 'uinteger': 0}
        print("Restored Random Number Generator: ", self.random_bg.state)

        # Restore WTMD state if available (normalize by CV as in deep-langevin-fts)
        if self.wtmd is not None and "wtmd_u" in load_data:
            self.wtmd.u = np.array(load_data["wtmd_u"]) / self.wtmd.CV
            self.wtmd.up = np.array(load_data["wtmd_up"]) / self.wtmd.CV
            if "wtmd_I0" in load_data:
                self.wtmd.I0 = np.array(load_data["wtmd_I0"])
            n_pending = np.atleast_1d(load_data.get("wtmd_order_parameter_history", np.zeros(0))).size
            self.wtmd.order_parameter_history = list(np.atleast_1d(load_data["wtmd_order_parameter_history"])) if n_pending > 0 else []
            self.wtmd.dH_history = {}
            for key in self.chi_n:
                monomer_pair = sorted(key.split(","))
                name = monomer_pair[0] + "_" + monomer_pair[1]
                if "wtmd_I1_" + name in load_data:
                    self.wtmd.I1[key] = np.array(load_data["wtmd_I1_" + name]) / self.wtmd.CV
                if n_pending > 0 and "wtmd_dH_history_" + name in load_data:
                    self.wtmd.dH_history[key] = list(np.atleast_1d(load_data["wtmd_dH_history_" + name]))
            print("Restored WTMD state")

        # Make initial_fields
        initial_fields = {}
        for name in self.monomer_types:
            initial_fields[name] = np.array(load_data["w_" + name])

        # Run
        self.run(initial_fields=initial_fields,
            normal_noise_prev=load_data["normal_noise_prev"],
            start_langevin_step=load_data["langevin_step"]+1)

    def run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None):
        """Run Langevin Field-Theoretic Simulation (L-FTS).

        Performs Langevin dynamics to sample compositional fluctuations in the
        field-theoretic representation of the polymer system. At each Langevin
        step:

        1. Computes functional derivative δH/δw for real auxiliary fields
        2. Updates real fields via Leimkuhler-Matthews method with Gaussian noise
        3. Finds saddle point of imaginary auxiliary fields via compression
        4. Records fields, Hamiltonian, and structure function data
        5. Saves checkpoint periodically

        The simulation continues for langevin.max_step iterations.

        Parameters
        ----------
        initial_fields : dict
            Initial monomer potential fields, {monomer_type: field_array}.
            Typically initialized from SCFT saddle point for faster equilibration.
            Each array has length total_grid.
        normal_noise_prev : ndarray, optional
            Previous Langevin noise array for Leimkuhler-Matthews method,
            shape (R, total_grid) where R = number of real auxiliary fields.
            Default: None (initialized as zeros). Used when continuing from checkpoint.
        start_langevin_step : int, optional
            Starting Langevin iteration number. Default: 1.
            Used when continuing from checkpoint; a value above 1 assumes the
            fields are already at the saddle point, as a checkpoint stores them.
        """

        print("---------- Run  ----------")

        # The number of components
        M = len(self.monomer_types)

        # The numbers of real and imaginary fields, respectively
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        # Check the partition consistency on the first field of this run
        # (the LR/LRAM setup already called compute_concentrations on w = 0)
        self._partition_checked = False

        # Simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # Reshape initial fields
        w = np.array([np.reshape(initial_fields[m], self.prop_solver.n_grid)
                      for m in self.monomer_types])
            
        # Convert monomer potential fields into auxiliary potential fields
        w_aux = self.mpt.to_aux_fields(w)

        if start_langevin_step is None:
            start_langevin_step = 1

        # Find the saddle point of the initial fields. A continued run loads fields
        # that are already at the saddle point; solving again would change the
        # trajectory slightly, so the loop's own solve is the first one then.
        print("iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)")
        if start_langevin_step == 1:
            phi, _, _, _, = self.find_saddle_point(w_aux=w_aux)
        else:
            phi, _ = self.compute_concentrations(w_aux)

        # Dictionary to record history of H and dH/dχN
        H_history = []
        dH_history = {}
        for key in self.chi_n:
            dH_history[key] = []

        # Arrays for structure function
        sf_average = {} # <u(k) phi(-k)>
        for monomer_id_pair in itertools.combinations_with_replacement(list(range(M)),2):
            sorted_pair = sorted(monomer_id_pair)
            type_pair = self.monomer_types[sorted_pair[0]] + "," + self.monomer_types[sorted_pair[1]]
            sf_average[type_pair] = np.zeros_like(np.fft.rfftn(np.reshape(w[0], self.prop_solver.nx)), np.complex128)

        # Create an empty array for field update algorithm
        # (a checkpoint loaded with squeeze_me drops the leading axis when R == 1)
        if normal_noise_prev is None:
            normal_noise_prev = np.zeros([R, self.prop_solver.n_grid], dtype=np.float64)
        else:
            normal_noise_prev = np.reshape(np.asarray(normal_noise_prev, dtype=np.float64), [R, self.prop_solver.n_grid])


        # The number of times that 'find_saddle_point' has failed to find a saddle point
        saddle_fail_count = 0

        # Init timers
        total_saddle_iter = 0
        total_error_level = 0
        time_start = time.time()

        # Langevin iteration begins here
        langevin_step = start_langevin_step - 1   # stays when max_step < start_langevin_step
        for langevin_step in range(start_langevin_step, self.langevin["max_step"]+1):
            print("Langevin step: ", langevin_step)

            # Copy data for restoring
            w_aux_copy = w_aux.copy()
            phi_copy = phi.copy()

            # Compute functional derivatives of Hamiltonian w.r.t. real-valued fields
            phi_for_force = self.smearing.apply_to_dict(phi)
            w_lambda = self.mpt.compute_func_deriv(w_aux, phi_for_force, self.mpt.aux_fields_real_idx)

            # Add WTMD bias force if enabled (same as deep-langevin-fts)
            if self.wtmd is not None:
                # Compute order parameter
                psi = self.wtmd.compute_order_parameter(langevin_step, w_aux)

                # Add bias to w_lambda
                self.wtmd.add_bias_to_langevin(psi, w_lambda)

            # Update w_aux using Leimkuhler-Matthews method. If the saddle point of the
            # imaginary fields is not found, the step is repeated from the same fields with
            # new noise (the previous noise of the Leimkuhler-Matthews pair is kept), up to
            # five times, so that every Langevin step yields one sample
            for attempt in range(6):
                w_aux = w_aux_copy.copy()
                normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], [R, self.prop_solver.n_grid])
                for count, i in enumerate(self.mpt.aux_fields_real_idx):
                    scaling = self.dt_scaling[i]
                    w_aux[i] += -w_lambda[count]*self.langevin["dt"]*scaling + 0.5*(normal_noise_prev[count] + normal_noise_current[count])*np.sqrt(scaling)

                # Find saddle point of the pressure field
                phi, hamiltonian, saddle_iter, error_level = self.find_saddle_point(w_aux=w_aux)
                total_saddle_iter += saddle_iter
                total_error_level += error_level
                saddle_failed = np.isnan(error_level) or error_level >= self.saddle["tolerance"]
                if not saddle_failed:
                    break
                saddle_fail_count += 1
                if attempt < 5:
                    print("The tolerance of the saddle point was not met. Langevin random noise is regenerated.")
            if saddle_failed:
                print("The tolerance of the saddle point was not met %d times in a row. Simulation is aborted." % (attempt + 1))
                w_aux = w_aux_copy
                phi = phi_copy
                break

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            # WTMD: store order parameter and update statistics (same as deep-langevin-fts)
            if self.wtmd is not None:
                # Store the order parameter with dH/dχN of the same fields (the fields
                # before the update, at which psi and the bias were computed)
                dH = self.mpt.compute_h_deriv_chin(self.chi_n, w_aux_copy, phi_for_force, wtmd=True)
                self.wtmd.store_order_parameter(psi, dH)

                # Update WTMD bias
                if langevin_step % self.wtmd.update_freq == 0:
                    self.wtmd.update_statistics()

                # Write WTMD data
                if langevin_step % self.wtmd.recording_period == 0:
                    self.wtmd.write_data(os.path.join(self.recording["dir"], "wtmd_statistics_%06d.mat" % (langevin_step)))

            # Compute H and dH/dχN
            if langevin_step % self.recording["sf_computing_period"] == 0:
                H_history.append(hamiltonian)
                dH = self.mpt.compute_h_deriv_chin(self.chi_n, w_aux, self.smearing.apply_to_dict(phi))
                for key in self.chi_n:
                    dH_history[key].append(dH[key])

            # Save H and dH/dχN
            if langevin_step % self.recording["sf_recording_period"] == 0:
                H_history = np.array(H_history)
                mdic = {"H_history": H_history}
                for key in self.chi_n:
                    dH_history[key] = np.array(dH_history[key])
                    monomer_pair = sorted(key.split(","))
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1]] = dH_history[key]
                savemat(os.path.join(self.recording["dir"], "dH_%06d.mat" % (langevin_step)), mdic, long_field_names=True, do_compression=True)
                # Reset dictionary
                H_history = []
                for key in self.chi_n:
                    dH_history[key] = []
                    
            # Calculate structure function
            if langevin_step % self.recording["sf_computing_period"] == 0:
                # Perform Fourier transforms.
                # Shifting the Gaussian integral over Ω_k gives <Ω_k φ_j> = (λ_k/M) Σ_l A_lk <φ_l φ_j>,
                # and A Aᵀ = M I, so μ_i = Σ_k A_ik Ω_k/λ_k gives <μ_i φ_j> = <δφ_i δφ_j>
                mu_fourier = {}
                phi_fourier = {}
                for i in range(M):
                    key = self.monomer_types[i]
                    phi_fourier[key] = np.fft.rfftn(np.reshape(phi[self.monomer_types[i]], self.prop_solver.nx))/self.prop_solver.n_grid
                    mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
                    for k in range(M-1) :
                        if np.isclose(self.mpt.eigenvalues[k], 0.0):
                            continue
                        mu_fourier[key] += np.fft.rfftn(np.reshape(w_aux[k], self.prop_solver.nx))*self.mpt.matrix_a[i,k]/self.mpt.eigenvalues[k]/self.prop_solver.n_grid
                # Accumulate S_ij(K), assuming that <mu(k)>*<phi(-k)> is zero
                for key in sf_average:
                    monomer_pair = sorted(key.split(","))
                    sf_average[key] += mu_fourier[monomer_pair[0]]* np.conj( phi_fourier[monomer_pair[1]])

            # Save structure function
            if langevin_step % self.recording["sf_recording_period"] == 0:
                # Make a dictionary for chi_n
                chi_n_mat = {}
                for key in self.chi_n:
                    monomer_pair = sorted(key.split(","))
                    chi_n_mat[monomer_pair[0] + "," + monomer_pair[1]] = self.chi_n[key]
                mdic = {"dim":self.prop_solver.dim, "nx":self.prop_solver.nx, "lx":self.prop_solver.get_lx(),
                        "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
                        "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "initial_params":self.params}
                # Add structure functions to the dictionary
                for key in sf_average:
                    sf_average[key] *= self.recording["sf_computing_period"]/self.recording["sf_recording_period"]* \
                            self.prop_solver.get_volume()*np.sqrt(self.langevin["nbar"])
                    # With smearing the fields couple to the smeared densities, so <μ φ> = Γ(k) S(k):
                    # divide by Γ to record S(k) of the densities (modes the smeared interaction
                    # does not see, Γ < 1e-12, are recorded as 0). With walls Γ lives in the cosine
                    # basis while S(k) is a Fourier transform, so it is recorded as it is.
                    if self.smearing.enabled and self.smearing.periodic:
                        gamma = self.smearing.fourier[..., :self.prop_solver.nx[-1]//2+1]
                        sf_average[key] = np.divide(sf_average[key], gamma, out=np.zeros_like(sf_average[key]), where=gamma > 1e-12)
                    monomer_pair = sorted(key.split(","))
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1]] = sf_average[key]
                savemat(os.path.join(self.recording["dir"], "structure_function_%06d.mat" % (langevin_step)), mdic, long_field_names=True, do_compression=True)
                # Reset arrays
                for key in sf_average:
                    sf_average[key][:] = 0.0

            # Save simulation data
            if langevin_step % self.recording["recording_period"] == 0:
                w = self.mpt.to_monomer_fields(w_aux)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev)

        print( "The number of times that tolerance of saddle point was not met and Langevin random noise was regenerated: %d times" % 
            (saddle_fail_count))

        # Estimate execution time
        time_duration = time.time() - time_start

        n_steps = max(langevin_step + 1 - start_langevin_step, 1)
        print("total time: %f, time per step: %f" % (time_duration, time_duration/n_steps))
        print("Total iterations for saddle points: %d, Iterations per Langevin step: %f" %
            (total_saddle_iter, total_saddle_iter/n_steps))

    def find_saddle_point(self, w_aux):
        """Find saddle point of imaginary auxiliary fields.

        Iteratively compresses imaginary-valued auxiliary fields to their saddle
        point values while keeping real-valued fields fixed. This is performed at
        each Langevin step to ensure imaginary fields remain at their saddle point.

        The saddle point satisfies δH/δw_imag = 0, enforcing incompressibility and
        other constraints.

        Parameters
        ----------
        w_aux : ndarray
            Auxiliary potential fields, shape (M, total_grid).
            Real auxiliary fields (indices in aux_fields_real_idx) remain fixed.
            Imaginary fields (indices in aux_fields_imag_idx) are optimized.

        Returns
        -------
        phi : dict
            Final monomer concentration fields {monomer_type: array}.
        hamiltonian : float
            Hamiltonian value at converged saddle point.
        saddle_iter : int
            Number of iterations performed to find saddle point.
        error_level : float
            Final convergence error (max standard deviation of δH/δw_imag).
        """

        # The number of components
        M = len(self.monomer_types)

        # The numbers of real and imaginary fields, respectively
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        # Assign large initial value for error
        error_level = 1e20

        # Reset compressor
        self.compressor.reset_count()

        # The uniform mode of each imaginary field (except the incompressible pressure
        # field, a gauge freedom) has the analytic saddle value
        # mean(Ω_k) = -(c1_k + Σ_j A_jk mean(φ_j))/(2 c2_k), mean(φ) being fixed by the
        # chain compositions. It is pinned in every iteration, so the concentrations and
        # partition functions of the last iteration belong to the returned fields; the
        # convergence test (standard deviation) does not see this mode.
        phi_mean = self.mean_concentrations()
        mean_targets = {}
        for k in self.mpt.aux_fields_imag_idx:
            if not np.isclose(self.mpt.h_coef_mu2[k], 0.0):
                mean_targets[k] = -(self.mpt.h_coef_mu1[k] + np.dot(self.mpt.matrix_a[:, k], phi_mean)) / (2*self.mpt.h_coef_mu2[k])

        # Saddle point iteration begins here
        for saddle_iter in range(1,self.saddle["max_iter"]+1):

            for k, target in mean_targets.items():
                w_aux[k] += target - self.cb.mean(w_aux[k])

            # Compute total concentrations
            phi, _ = self.compute_concentrations(w_aux)

            # Compute functional derivatives of Hamiltonian w.r.t. imaginary fields
            phi_for_saddle = self.smearing.apply_to_dict(phi)
            h_deriv = self.mpt.compute_func_deriv(w_aux, phi_for_saddle, self.mpt.aux_fields_imag_idx)

            # Compute total error
            old_error_level = error_level
            error_level_array = np.std(h_deriv, axis=1)
            error_level = np.max(error_level_array)

            # Hamiltonian on the last iteration (every iteration when verbose_level == 2)
            is_last = error_level < self.saddle["tolerance"] or saddle_iter == self.saddle["max_iter"]
            if self.verbose_level == 2 or is_last:
                total_partitions = [self.prop_solver.get_partition_function(p) for p in range(self.prop_solver.get_n_polymer_types())]
                hamiltonian = self.mpt.compute_hamiltonian(self.prop_solver.molecules, w_aux, total_partitions, self.cb, include_const_term=True)

            # Print iteration # and error levels
            if self.verbose_level == 2 or (self.verbose_level == 1 and is_last):
                mass_error = self.cb.mean(h_deriv[I-1])
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for Q in total_partitions:
                    print("%13.7E " % (Q), end=" ")
                print("] %15.9f   [" % (hamiltonian), end="")
                for i in range(I):
                    print("%13.7E" % (error_level_array[i]), end=" ")
                print("]")

            # Conditions to end the iteration
            if error_level < self.saddle["tolerance"]:
                break

            # Scaling h_deriv
            for count, i in enumerate(self.mpt.aux_fields_imag_idx):
                h_deriv[count] *= self.dt_scaling[i]

            # Calculate new fields using compressor (AM, LR, LRAM)
            w_aux[self.mpt.aux_fields_imag_idx] = np.reshape(self.compressor.calculate_new_fields(w_aux[self.mpt.aux_fields_imag_idx], -h_deriv, old_error_level, error_level), [I, self.prop_solver.n_grid])

        # Incompressible: gauge fix of the pressure field to mean 0 (H and φ do not change)
        if self.zeta_n is None:
            w_aux[M-1] -= self.cb.mean(w_aux[M-1])

        return phi, hamiltonian, saddle_iter, error_level

    def mean_concentrations(self):
        """Volume-averaged concentration of each monomer type from the chain compositions.

        mean(phi_i) = sum_p (f_p / alpha_p) sum_{blocks b of p} length_b * (fraction of i in b),
        which is what the C++ normalization of the concentrations gives.
        """
        phi_mean = np.zeros(len(self.monomer_types))
        for p, polymer in enumerate(self.distinct_polymers):
            alpha = self.prop_solver.get_polymer(p).get_alpha()
            for block in polymer["blocks"]:
                weight = polymer["volume_fraction"] * block["length"] / alpha
                fractions = self.random_fraction.get(block["type"], {block["type"]: 1.0})
                for monomer_type, fraction in fractions.items():
                    phi_mean[self.monomer_types.index(monomer_type)] += weight * fraction
        return phi_mean
