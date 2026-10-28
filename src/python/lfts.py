import os
import time
import pathlib
import copy
import json
import warnings
import numpy as np
import itertools
import scipy.fft
from scipy.io import loadmat

from .polymer_field_theory import SymmetricPolymerTheory
from .propagator_solver import PropagatorSolver
from .integrators import checked_fields
from .validation import normalize_params, validate_lfts_params, ValidationError, SaddlePointError, parse_chi_n
from .utils import (
    process_polymer_blocks,
    process_random_copolymer,
    draw_polymer_architectures,
    report_partition_consistency,
    check_partition_functions,
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
    restore_extensions,
    notify_callbacks,
    extension_object_state,
)
from .smearing import Smearing
from .wtmd import WTMD

# OpenMP environment variables
os.environ.setdefault("OMP_NUM_THREADS", "1")  # respect user override
os.environ.setdefault("OMP_STACKSIZE", "1G")


def _axis_transform(field, axis_bc, inverse=False):
    """Orthonormal transform of each axis: FFT ("periodic") or cosine series (DCT-II, "wall"), or its inverse."""
    for axis, kind in enumerate(axis_bc):
        if kind == "periodic":
            field = (scipy.fft.ifft if inverse else scipy.fft.fft)(field, axis=axis, norm="ortho")
        else:
            field = (scipy.fft.idct if inverse else scipy.fft.dct)(field, type=2, axis=axis, norm="ortho")
    return field


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
        finite difference of δH/δw. Shape: the coefficients of fourier when it
        is given, the rfftn half spectrum of nx in a periodic box, or nx with axis_bc.
    fourier : FourierTransform, optional
        PropagatorSolver.fourier of real fields (FFT in a periodic box, cosine series in
        a box with reflecting walls), which applies 1/J(k) (on the GPU with CUDA);
        without it the update uses numpy.fft or, with axis_bc, scipy.fft.
    axis_bc : list of str, optional
        "periodic" or "wall" for each axis of a box that mixes periodic axes and
        reflecting walls (no fourier there): FFT and cosine series (orthonormal).

    Attributes
    ----------
    nx : list of int
        Grid dimensions.
    lx : list of float
        Box dimensions.
    jk : ndarray
        Jacobian in Fourier space.
    """
    def __init__(self, nx, lx, jk, fourier=None, axis_bc=None):
        self.nx = nx
        self.lx = lx
        self.jk = jk.copy()
        self._fourier = fourier
        self._inverse = None if fourier is None else fourier.set_multiplier(1.0 / self.jk)
        self.axis_bc = axis_bc

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
        if self._fourier is not None:
            w_diff = self._fourier.apply([(self._inverse, np.reshape(negative_h_deriv, -1))])
            return w_current + np.reshape(w_diff, np.shape(w_current))
        nx = self.nx
        if self.axis_bc is not None:
            w_diff_k = _axis_transform(np.reshape(negative_h_deriv, nx), self.axis_bc)/self.jk
            w_diff = np.real(_axis_transform(w_diff_k, self.axis_bc, inverse=True))
            return w_current + np.reshape(w_diff, np.shape(w_current))
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
    integrator : object, optional
        Langevin step implementation; otherwise langevin.integrator (default "lm").
        See docs/internals/Extensions.md for the step and checkpoint contracts.
    callbacks : dict, optional
        Stable names to callable observers of accepted, read-only step snapshots.

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
    def from_file(cls, path, random_seed=None, *, integrator=None, callbacks=None, **overrides):
        """Create a simulation from a YAML or JSON parameter file.

        Reads the file with utils.load_config, applies the keyword overrides
        and calls the constructor, as SCFT.from_file does.
        """
        params = normalize_params(load_config(path))
        params.update(normalize_params(overrides))
        return cls(params, random_seed=random_seed, integrator=integrator, callbacks=callbacks)

    def __init__(self, params, random_seed=None, *, integrator=None, callbacks=None):

        # Validate input parameters
        params = normalize_params(params)
        validate_lfts_params(params)
        for key in ("compressor", "saddle", "recording", "verbose_level"):
            if key not in params:
                raise ValidationError(f"Missing required parameter for LFTS: '{key}'")
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

        # Compressibility parameter (None: incompressible)
        self.zeta_n = params.get("zeta_n", None)

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
        for polymer in self.polymers:
            self.prop_solver.add_polymer(polymer["volume_fraction"], polymer["blocks_input"],
                                         polymer.get("initial_conditions", None))

        # Computation box as a public attribute (this creates the C++ objects)
        self.cb = self.prop_solver.computation_box

        # Display factory info
        self.prop_solver.display_platform_info()

        # Initialize smearing (must be after self.cb is set)
        # Γ(k) is applied with the platform FFT (PropagatorSolver.fourier), except in a box
        # that mixes periodic and wall axes, which smearing transforms with NumPy/SciPy
        periodic_sides = [b.lower() == "periodic" for b in self.prop_solver.bc]
        mixed_box = any(periodic_sides) and not all(periodic_sides)
        periodic_box = all(periodic_sides)   # WTMD and the structure functions transform with rfftn
        self._periodic_box = periodic_box
        self.smearing = Smearing(
            self.cb.get_nx(),
            self.cb.get_lx(),
            params.get("smearing", None),
            angles=self.prop_solver.get_angles_degrees(),
            bc=self.prop_solver.bc,
            transform=self.prop_solver.fourier if params.get("smearing") is not None and not mixed_box else None
        )

        # Initialize Well-Tempered Metadynamics (WTMD) if parameters provided. The WTMD class
        # follows deep-langevin-fts; the input names differ: "ell" and "delta_t" here for its
        # "l" and "dT"
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
                fourier=self.prop_solver.fourier if periodic_box else None,
            )
            print("WTMD enabled: l=%d, sigma_psi=%.4f, dT=%.4f, kc=%.4f, exchange_idx=%d, langevin_idx=%d" % (
                self.wtmd.l, self.wtmd.sigma_psi, self.wtmd.dT, self.wtmd.kc,
                self.wtmd.exchange_idx, self.wtmd.langevin_idx))
        else:
            self.wtmd = None

        # (C++ class) Fields Relaxation using Anderson Mixing
        if params["compressor"]["name"] == "am" or params["compressor"]["name"] == "lram":
            n_am = int(len(self.mpt.aux_fields_imag_idx)*np.prod(params["nx"]))
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
            # The response J to a kick at w = 0 is diagonal in the Fourier basis of the box:
            # the FFT in a periodic box, the cosine series along reflecting walls (the box is
            # half of a periodic box of twice its size, with the mirror image of the kick).
            # Next to an absorbing wall the concentrations at w = 0 are depleted, and no
            # Fourier basis makes J diagonal
            if any(b.lower() == "absorbing" for b in self.prop_solver.bc):
                raise ValidationError("compressor 'lr' and 'lram' need periodic axes or reflecting walls; "
                                      "use 'am' in a box with absorbing walls")

            # Baseline: h_deriv at w=0
            w_aux_zero = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
            phi_zero, _ = self.compute_concentrations(w_aux_zero)
            h_deriv_zero = self.mpt.compute_func_deriv(w_aux_zero, self.smearing.apply_to_dict(phi_zero), self.mpt.aux_fields_imag_idx)

            # Perturbed: h_deriv at w with small perturbation
            w_aux_perturbed = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
            w_aux_perturbed[M-1,0] = 1e-3 # add a small perturbation at the pressure field

            phi_perturbed, _ = self.compute_concentrations(w_aux_perturbed)
            h_deriv_perturbed = self.mpt.compute_func_deriv(w_aux_perturbed, self.smearing.apply_to_dict(phi_perturbed), self.mpt.aux_fields_imag_idx)

            # jk = (h(w_pert) - h(0)) / w_pert  (correct finite difference)
            h_deriv_diff = h_deriv_perturbed - h_deriv_zero
            nx = self.prop_solver.nx
            if periodic_box:
                w_aux_perturbed_k = np.fft.rfftn(np.reshape(w_aux_perturbed[M-1], nx))/np.prod(nx)
                h_deriv_diff_k = np.fft.rfftn(np.reshape(h_deriv_diff, nx))/np.prod(nx)
                jk_numeric = np.real(h_deriv_diff_k/w_aux_perturbed_k)
                lr = LR(nx, self.prop_solver.get_lx(), jk_numeric, fourier=self.prop_solver.fourier)
            elif not mixed_box:
                # Reflecting walls only: the cosine series of PropagatorSolver.fourier
                fourier = self.prop_solver.fourier
                jk_numeric = np.real(fourier.forward(np.reshape(h_deriv_diff, -1))/fourier.forward(w_aux_perturbed[M-1]))
                lr = LR(nx, self.prop_solver.get_lx(), jk_numeric, fourier=fourier)
            else:
                # Periodic axes and reflecting walls: FFT and cosine series with scipy.fft
                axis_bc = ["periodic" if periodic else "wall" for periodic in periodic_sides[::2]]
                jk_numeric = np.real(_axis_transform(np.reshape(h_deriv_diff, nx), axis_bc)
                                     /_axis_transform(np.reshape(w_aux_perturbed[M-1], nx), axis_bc))
                lr = LR(nx, self.prop_solver.get_lx(), jk_numeric, axis_bc=axis_bc)

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
            if i not in self.mpt.aux_fields_dropped_idx:
                self.dt_scaling[i] = np.abs(self.mpt.eigenvalues[i])/np.max(np.abs(self.mpt.eigenvalues[:M-1]))
        self.dt_scaling[self.mpt.aux_fields_dropped_idx] = 0.0

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
        self.params = copy.deepcopy(params)
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma":langevin_sigma})

        self.verbose_level = params["verbose_level"]
        self.saddle = params["saddle"].copy()
        self.recording = params["recording"].copy()
        self.metadata = simulation_metadata(self, random_seed)
        initialize_extensions(self, integrator if integrator is not None else self.langevin.get("integrator", "lm"), callbacks)
        # The linear response of the homogeneous state (the "etd" integrator) is measured by CLFTS only
        if getattr(self.integrator, "linear_response", False):
            raise ValidationError(f"The integrator '{self.integrator.name}' (linear_response) is supported in CL-FTS only")
        self.run_status = {"status": "initialized", "last_completed_step": 0}

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
        if not abs(mass_error) <= 5e-8:
            raise ValidationError(f"Material conservation failed: mean(sum(phi) - 1) = {mass_error:.3e}")

        # Partition consistency check (forward vs backward propagators), once per run
        if not getattr(self, "_partition_checked", False):
            self._partition_checked = True
            if not report_partition_consistency(self.prop_solver):
                raise ValidationError("Partition consistency check failed (forward/backward mismatch).")

        return phi, elapsed_time

    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev, w_aux=None):
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
            "initial_params": mat_ready(self.params),
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

        mdic.update(checkpoint_data(self))
        mdic["w_aux"] = self.mpt.to_aux_fields(w) if w_aux is None else w_aux

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
            # The same arrays without the factor CV, which a continued run restores exactly:
            # (x*CV)/CV can differ from x in the last bit
            mdic["wtmd_unscaled_u"] = self.wtmd.u
            mdic["wtmd_unscaled_up"] = self.wtmd.up
            # I1 and the samples stored since the last update, so that a continued run
            # carries the same statistics as an uninterrupted one
            for key, value in self.wtmd.I1.items():
                monomer_pair = sorted(key.split(","))
                mdic["wtmd_I1_" + monomer_pair[0] + "_" + monomer_pair[1]] = value * self.wtmd.CV
                mdic["wtmd_unscaled_I1_" + monomer_pair[0] + "_" + monomer_pair[1]] = value
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
        atomic_savemat(path, mdic, long_field_names=True, do_compression=True)

    def continue_run(self, file_name, *, allow_legacy=False):
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
        validate_checkpoint(self, load_data, allow_legacy=allow_legacy)
        restore_extensions(self, load_data)
        if "simulation_metadata_json" in load_data:
            previous_metadata = json.loads(str(load_data["simulation_metadata_json"]))
            self.metadata["random_seed"] = previous_metadata.get("random_seed")
            self.metadata["initial_fields_hash"] = previous_metadata.get("initial_fields_hash", "unknown")
            self.metadata["parent_checkpoint"] = str(file_name)
        
        # Check if load_data["langevin_step"] is a multiple of self.recording["statistics_save_interval"]
        if "record_sample_count" not in load_data and load_data["langevin_step"] % self.recording["statistics_save_interval"] != 0:
            print(f"(Warning!) 'langevin_step' of {file_name} is not a multiple of 'statistics_save_interval'.")
            next_sf_langevin_step = (load_data["langevin_step"]//self.recording["statistics_save_interval"] + 1)*self.recording["statistics_save_interval"]
            print(f"The structure function will be correctly recorded after {next_sf_langevin_step}th langevin_step." )

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
        print("Restored Random Number Generator: ", self.random_bg.state)

        # Restore WTMD state if available: the arrays without the factor CV, or for an older
        # checkpoint the scaled ones divided by CV (as in deep-langevin-fts)
        if self.wtmd is not None and "wtmd_u" in load_data:
            def bias(name):
                if "wtmd_unscaled_u" in load_data:
                    return np.array(load_data["wtmd_unscaled_" + name])
                return np.array(load_data["wtmd_" + name]) / self.wtmd.CV
            self.wtmd.u = bias("u")
            self.wtmd.up = bias("up")
            if "wtmd_I0" in load_data:
                self.wtmd.I0 = np.array(load_data["wtmd_I0"])
            n_pending = np.atleast_1d(load_data.get("wtmd_order_parameter_history", np.zeros(0))).size
            self.wtmd.order_parameter_history = list(np.atleast_1d(load_data["wtmd_order_parameter_history"])) if n_pending > 0 else []
            self.wtmd.I1 = {}
            self.wtmd.dH_history = {}
            for key in self.chi_n:
                monomer_pair = sorted(key.split(","))
                name = monomer_pair[0] + "_" + monomer_pair[1]
                if "wtmd_I1_" + name in load_data:
                    self.wtmd.I1[key] = bias("I1_" + name)
                if n_pending > 0 and "wtmd_dH_history_" + name in load_data:
                    self.wtmd.dH_history[key] = list(np.atleast_1d(load_data["wtmd_dH_history_" + name]))
            print("Restored WTMD state")

        # Make initial_fields
        initial_fields = {}
        for name in self.monomer_types:
            initial_fields[name] = np.array(load_data["w_" + name])

        # Run
        return self.run(initial_fields=initial_fields,
            normal_noise_prev=load_data["normal_noise_prev"],
            start_langevin_step=load_data["langevin_step"]+1, _restart_data=load_data)

    def run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None, *, _restart_data=None):
        """Run Langevin dynamics, returning a status dict; failures raise and persist run_status.json.

        ``saddle.failure_policy='retry'`` (default) retains the proposed real fields
        and noise while continuing the saddle solve. ``'abort'`` stops immediately;
        ``'resample'`` explicitly opts into the former noise-selection policy.
        ``saddle.max_retries`` defaults to five. Failed steps are never recorded.
        """
        first_step = 1 if start_langevin_step is None else start_langevin_step
        if isinstance(first_step, (bool, np.bool_)) or not isinstance(first_step, (int, np.integer)) or first_step < 1:
            raise ValidationError("start_langevin_step must be a positive integer")
        # The histogram, the Gaussian kernels and the integrals all use the bin width dpsi.
        # Checked unless WTMD statistics are restored, so that a run started with another
        # dpsi still continues
        if self.wtmd is not None and (_restart_data is None or "wtmd_u" not in _restart_data):
            bins = (self.wtmd.psi_max - self.wtmd.psi_min) / self.wtmd.dpsi
            if abs(bins - round(bins)) > 1e-9 * bins:
                raise ValidationError("wtmd.dpsi must divide psi_max - psi_min into a whole number of bins")
        self.run_status = {"status": "running", "last_completed_step": first_step - 1,
                           "requested_last_step": self.langevin["max_step"],
                           "saddle_failures": 0, "noise_resamples": 0}
        self.metadata["initial_fields_hash" if _restart_data is None else "restart_fields_hash"] = fields_hash(initial_fields)
        if _restart_data is not None and "run_status_json" in _restart_data:
            previous_status = json.loads(str(_restart_data["run_status_json"]))
            for key in ("saddle_failures", "noise_resamples"):
                if key in self.run_status:
                    self.run_status[key] = previous_status.get(key, 0)
        self.metadata["initial_rng_state"] = self.random_bg.state
        write_run_status(self)
        try:
            if _restart_data is None:
                restore_extensions(self)
            self._run(initial_fields, normal_noise_prev, start_langevin_step, _restart_data=_restart_data)
            self.run_status["status"] = "completed"
        except SaddlePointError as exc:
            self.run_status.update(status="aborted_saddle_failure", reason=str(exc))
            raise
        except (KeyboardInterrupt, SystemExit):
            self.run_status.update(status="interrupted", reason="Run interrupted")
            raise
        except Exception as exc:
            self.run_status.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_run_status(self)
        return dict(self.run_status)

    def _run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None, *, _restart_data=None):
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

        # Simulation data directory, with the drawings of the chain architectures (polymer_<id>.png)
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)
        draw_polymer_architectures(self.polymers, self.ds, self.segment_lengths, directory=self.recording["dir"])

        # Reshape initial fields
        w = np.array([np.reshape(initial_fields[m], self.prop_solver.n_grid)
                      for m in self.monomer_types])
            
        # Convert monomer potential fields into auxiliary potential fields
        w_aux = self.mpt.to_aux_fields(w)
        self.mpt.pin_dropped_fields(w_aux)
        if _restart_data is not None and "w_aux" in _restart_data:
            w_aux = np.asarray(_restart_data["w_aux"], dtype=np.float64).reshape(M, self.prop_solver.n_grid).copy()
        if not np.all(np.isfinite(w_aux)):
            raise ValidationError("Initial fields must be finite")

        if start_langevin_step is None:
            start_langevin_step = 1

        # Find the saddle point of the initial fields. A continued run loads fields
        # that are already at the saddle point; solving again would change the
        # trajectory slightly, so the loop's own solve is the first one then.
        print("iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)")
        if start_langevin_step == 1:
            for attempt in range(self.saddle.get("max_retries", 5) + 1):
                phi, initial_hamiltonian, _, error_level = self.find_saddle_point(w_aux=w_aux)
                if not np.isfinite(initial_hamiltonian):
                    # H contains log Q: name a partition function that is not positive as the cause
                    check_partition_functions(self.prop_solver, "the initial saddle point")
                    raise ValidationError("Non-finite initial Hamiltonian")
                if np.isfinite(error_level) and error_level < self.saddle["tolerance"]:
                    break
                self.run_status["saddle_failures"] += 1
                if self.saddle.get("failure_policy", "retry") == "abort":
                    break
            if not np.isfinite(error_level) or error_level >= self.saddle["tolerance"]:
                raise SaddlePointError(f"Initial saddle point failed after {attempt + 1} attempts (error={error_level})")
        elif _restart_data is not None and "w_aux" in _restart_data:
            phi = {m: np.asarray(_restart_data["phi_" + m]).reshape(self.prop_solver.n_grid).copy()
                   for m in self.monomer_types}
        else:
            phi, _ = self.compute_concentrations(w_aux)

        # Dictionary to record history of H and dH/dχN
        H_history = []
        drift_history = []
        dH_history = {}
        for key in self.chi_n:
            dH_history[key] = []

        # Arrays for structure function
        sf_average = {} # <u(k) phi(-k)>
        for monomer_id_pair in itertools.combinations_with_replacement(list(range(M)),2):
            sorted_pair = sorted(monomer_id_pair)
            type_pair = self.monomer_types[sorted_pair[0]] + "," + self.monomer_types[sorted_pair[1]]
            sf_average[type_pair] = np.zeros_like(np.fft.rfftn(np.reshape(w[0], self.prop_solver.nx)), np.complex128)
        sf_sample_count = restore_recording_state(_restart_data, H_history, dH_history, sf_average, drift_history)

        # Create an empty array for field update algorithm
        # (a checkpoint loaded with squeeze_me drops the leading axis when R == 1)
        if normal_noise_prev is None:
            normal_noise_prev = np.zeros([R, self.prop_solver.n_grid], dtype=np.float64)
        else:
            normal_noise_prev = np.reshape(np.asarray(normal_noise_prev, dtype=np.float64), [R, self.prop_solver.n_grid])


        # Init timers
        total_saddle_iter = 0
        self._intermediate_saddle_iterations = 0
        total_error_level = 0
        time_start = time.time()

        # Langevin iteration begins here
        langevin_step = start_langevin_step - 1   # stays when max_step < start_langevin_step
        for langevin_step in range(start_langevin_step, self.langevin["max_step"]+1):
            step_start = time.perf_counter()
            if self.verbose_level >= 1:
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

            drift_max = float(np.max(np.abs(w_lambda * self.dt_scaling[self.mpt.aux_fields_real_idx, None]))) if R else 0.0

            # Retrying the solver must not select the Langevin noise by convergence.
            policy = self.saddle.get("failure_policy", "retry")
            integrator_state = extension_object_state(self.integrator)
            for attempt in range(self.saddle.get("max_retries", 5) + 1):
                if attempt == 0 or policy == "resample":
                    w_aux = w_aux_copy.copy()
                    normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], [R, self.prop_solver.n_grid])
                    if attempt:
                        self.run_status["noise_resamples"] += 1
                        warnings.warn("Saddle failure: resampling noise can change the sampled distribution", RuntimeWarning)
                    if hasattr(self.integrator, "load_state_dict"):
                        self.integrator.load_state_dict(copy.deepcopy(integrator_state))
                    active = w_aux[self.mpt.aux_fields_real_idx]
                    try:
                        proposed = self.integrator.step(
                            active.copy(), w_lambda.copy(), dt=self.langevin["dt"],
                            mobility=self.dt_scaling[self.mpt.aux_fields_real_idx].copy(),
                            noise=normal_noise_current.copy(), previous_noise=normal_noise_prev.copy(),
                            noise_direction=np.ones(R),
                            evaluate_force=lambda fields: self.evaluate_force(fields, w_aux=w_aux_copy, step=langevin_step))
                    except SaddlePointError:
                        if hasattr(self.integrator, "load_state_dict"):
                            self.integrator.load_state_dict(copy.deepcopy(integrator_state))
                        if policy == "resample" and attempt < self.saddle.get("max_retries", 5):
                            continue
                        raise
                    w_aux[self.mpt.aux_fields_real_idx] = checked_fields(proposed, active)

                # Find saddle point of the pressure field
                phi, hamiltonian, saddle_iter, error_level = self.find_saddle_point(w_aux=w_aux)
                if not np.isfinite(hamiltonian):
                    # H contains log Q: name a partition function that is not positive as the cause
                    check_partition_functions(self.prop_solver, f"the saddle point of step {langevin_step}")
                    raise ValidationError(f"Non-finite Hamiltonian at step {langevin_step}")
                total_saddle_iter += saddle_iter
                total_error_level += error_level
                saddle_failed = not np.isfinite(error_level) or error_level >= self.saddle["tolerance"]
                if not saddle_failed:
                    break
                self.run_status["saddle_failures"] += 1
                if policy == "abort" or not np.all(np.isfinite(w_aux)):
                    break
            if saddle_failed:
                if hasattr(self.integrator, "load_state_dict"):
                    self.integrator.load_state_dict(copy.deepcopy(integrator_state))
                w_aux = w_aux_copy
                phi = phi_copy
                raise SaddlePointError(f"Saddle point failed at step {langevin_step} after {attempt + 1} attempts (error={error_level})")

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev
            self.run_status["last_completed_step"] = langevin_step

            # WTMD: store order parameter and update statistics (wtmd.py)
            if self.wtmd is not None:
                # Store the order parameter with dH/dχN of the same fields (the fields
                # before the update, at which psi and the bias were computed)
                dH = self.mpt.compute_h_deriv_chin(w_aux=w_aux_copy, phi=phi_for_force, wtmd=True)
                self.wtmd.store_order_parameter(psi, dH)

                # Update WTMD bias
                if langevin_step % self.wtmd.update_freq == 0:
                    self.wtmd.update_statistics()

                # Write WTMD data
                if langevin_step % self.wtmd.recording_period == 0:
                    self.wtmd.write_data(os.path.join(self.recording["dir"], "wtmd_statistics_%06d.mat" % (langevin_step)))

            # Compute H and dH/dχN
            if langevin_step % self.recording["sample_interval"] == 0:
                H_history.append(hamiltonian)
                drift_history.append(drift_max)
                sf_sample_count += 1
                dH = self.mpt.compute_h_deriv_chin(w_aux=w_aux, phi=self.smearing.apply_to_dict(phi))
                for key in self.chi_n:
                    dH_history[key].append(dH[key])

            # Save H and dH/dχN
            if langevin_step % self.recording["statistics_save_interval"] == 0:
                H_history = np.array(H_history)
                mdic = {"H_history": H_history, "sample_count": sf_sample_count, "drift_max_history": np.asarray(drift_history),
                        "simulation_metadata_json": json.dumps(self.metadata)}
                for key in self.chi_n:
                    dH_history[key] = np.array(dH_history[key])
                    monomer_pair = sorted(key.split(","))
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1]] = dH_history[key]
                atomic_savemat(os.path.join(self.recording["dir"], "dH_%06d.mat" % (langevin_step)), mdic, long_field_names=True, do_compression=True)
                # Reset dictionary
                H_history = []
                drift_history = []
                for key in self.chi_n:
                    dH_history[key] = []
                    
            # Calculate structure function
            if langevin_step % self.recording["sample_interval"] == 0:
                # Perform Fourier transforms (all densities in one call, all fields in another).
                # Shifting the Gaussian integral over Ω_k gives <Ω_k φ_j> = (λ_k/M) Σ_l A_lk <φ_l φ_j>,
                # and A Aᵀ = M I, so μ_i = Σ_k A_ik Ω_k/λ_k gives <μ_i φ_j> = <δφ_i δφ_j>
                kept = [k for k in range(M-1) if not np.isclose(self.mpt.eigenvalues[k], 0.0)]
                phi_k = self._rfft_fields([phi[name] for name in self.monomer_types])/self.prop_solver.n_grid
                w_k = self._rfft_fields(w_aux[kept])/self.prop_solver.n_grid if kept else None
                mu_fourier = {}
                phi_fourier = {}
                for i in range(M):
                    key = self.monomer_types[i]
                    phi_fourier[key] = phi_k[i]
                    mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
                    for j, k in enumerate(kept):
                        mu_fourier[key] += w_k[j]*self.mpt.matrix_a[i,k]/self.mpt.eigenvalues[k]
                # Accumulate S_ij(K), assuming that <mu(k)>*<phi(-k)> is zero
                for key in sf_average:
                    monomer_pair = sorted(key.split(","))
                    sf_average[key] += mu_fourier[monomer_pair[0]]* np.conj( phi_fourier[monomer_pair[1]])

            # Save structure function
            if langevin_step % self.recording["statistics_save_interval"] == 0:
                # Make a dictionary for chi_n
                chi_n_mat = {}
                for key in self.chi_n:
                    monomer_pair = sorted(key.split(","))
                    chi_n_mat[monomer_pair[0] + "," + monomer_pair[1]] = self.chi_n[key]
                mdic = {"dim":self.prop_solver.dim, "nx":self.prop_solver.nx, "lx":self.prop_solver.get_lx(),
                        "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
                        "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "initial_params":mat_ready(self.params)}
                # Add structure functions to the dictionary
                for key in sf_average:
                    sf_average[key] *= 1.0/max(sf_sample_count, 1)* \
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
                mdic["sample_count"] = sf_sample_count
                mdic["simulation_metadata_json"] = json.dumps(self.metadata)
                atomic_savemat(os.path.join(self.recording["dir"], "structure_function_%06d.mat" % (langevin_step)), mdic, long_field_names=True, do_compression=True)
                # Reset arrays
                for key in sf_average:
                    sf_average[key][:] = 0.0
                sf_sample_count = 0

            # Observe an accepted state before saving its callback/integrator state.
            notify_callbacks(self, langevin_step, w_aux, phi, hamiltonian, drift_max,
                             error_level, time.perf_counter() - step_start)

            # Save simulation data
            if langevin_step % self.recording["checkpoint_interval"] == 0:
                self._recording_state = {"H_history": H_history, "dH_history": dH_history,
                                         "sf_average": sf_average, "sample_count": sf_sample_count, "drift_history": drift_history}
                w = self.mpt.to_monomer_fields(w_aux)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev, w_aux=w_aux)

        print("Cumulative failed saddle-point solves: %d times" % self.run_status["saddle_failures"])

        # Estimate execution time
        time_duration = time.time() - time_start

        n_steps = max(langevin_step + 1 - start_langevin_step, 1)
        print("total time: %f, time per step: %f" % (time_duration, time_duration/n_steps))
        total_saddle_iter += self._intermediate_saddle_iterations
        print("Total iterations for saddle points: %d, Iterations per Langevin step: %f" %
            (total_saddle_iter, total_saddle_iter/n_steps))

    def evaluate_force(self, fields, *, w_aux, step):
        """dH/dw at intermediate active fields, with a saddle solve and fixed bias.

        No noise is drawn and no observations or WTMD statistics are recorded.
        ``w_aux`` supplies the imaginary-field warm start of this proposal.
        """
        trial = w_aux.copy()
        indices = self.mpt.aux_fields_real_idx
        trial[indices] = checked_fields(fields, trial[indices])
        for attempt in range(self.saddle.get("max_retries", 5) + 1):
            phi, hamiltonian, iterations, error = self.find_saddle_point(trial)
            self._intermediate_saddle_iterations += iterations
            if not np.isfinite(hamiltonian) or not np.all(np.isfinite(trial)):
                raise ValidationError("Non-finite intermediate LFTS state")
            if np.isfinite(error) and error < self.saddle["tolerance"]:
                break
            self.run_status["saddle_failures"] += 1
            if self.saddle.get("failure_policy", "retry") == "abort":
                break
        else:
            raise SaddlePointError(f"Intermediate saddle failed at step {step} (error={error})")
        if not np.isfinite(error) or error >= self.saddle["tolerance"]:
            raise SaddlePointError(f"Intermediate saddle failed at step {step} (error={error})")
        force = self.mpt.compute_func_deriv(trial, self.smearing.apply_to_dict(phi), indices)
        if self.wtmd is not None:
            psi = self.wtmd.compute_order_parameter(step, trial)
            self.wtmd.add_bias_to_langevin(psi, force)
        if not np.all(np.isfinite(force)):
            raise ValidationError("Non-finite intermediate LFTS force")
        return force

    def _rfft_fields(self, fields):
        """rfftn of each field (n_fields, n_grid) on the grid: with the platform FFT
        (PropagatorSolver.fourier) in a periodic box, with numpy.fft otherwise."""
        fields = np.reshape(fields, (-1, self.prop_solver.n_grid))
        if self._periodic_box:
            return self.prop_solver.fourier.forward(fields)
        nx = tuple(self.prop_solver.nx)
        return np.fft.rfftn(np.reshape(fields, (-1,) + nx), axes=tuple(range(1, len(nx) + 1)))

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
                hamiltonian = self.mpt.compute_hamiltonian(self.prop_solver.molecules, w_aux, total_partitions, self.cb,
                                                          include_const_term=True, phi=self.smearing.apply_to_dict(phi))

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
        for p, polymer in enumerate(self.polymers):
            alpha = self.prop_solver.get_polymer(p).get_alpha()
            for block in polymer["blocks"]:
                weight = polymer["volume_fraction"] * block["length"] / alpha
                fractions = self.random_fraction.get(block["type"], {block["type"]: 1.0})
                for monomer_type, fraction in fractions.items():
                    phi_mean[self.monomer_types.index(monomer_type)] += weight * fraction
        return phi_mean
