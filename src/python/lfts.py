import os
import time
import re
import pathlib
import copy
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

from . import _core
from .polymer_field_theory import *
from .compressor import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"

def calculate_sigma(langevin_nbar, langevin_dt, n_grids, volume):
    """Calculate Langevin noise strength for L-FTS simulations.

    Computes the standard deviation of Gaussian noise to be added at each
    Langevin step, ensuring correct statistical weighting of field configurations
    in the field-theoretic ensemble.

    Parameters
    ----------
    langevin_nbar : float
        Average polymerization index N̄ for the Langevin scheme. Related to
        the prefactor of the Hamiltonian. Typical values: 1-100.
    langevin_dt : float
        Langevin time step size Δt. Controls the magnitude of field updates.
        Typical values: 0.001-0.1.
    n_grids : int
        Total number of grid points M (product of nx dimensions).
    volume : float
        System volume V in units of (a_Ref * N_Ref^0.5)³.

    Returns
    -------
    sigma : float
        Standard deviation of Gaussian noise for each field component.

    Notes
    -----
    The noise strength is derived from the fluctuation-dissipation theorem:

    .. math::
        \\sigma = \\sqrt{\\frac{2 \\Delta t M}{V \\sqrt{\\bar{N}}}}

    This ensures that the Langevin dynamics samples the field-theoretic
    partition function correctly, with Boltzmann weighting ∝ exp(-H/√N̄).

    **Physical Interpretation:**

    - Larger N̄: Reduces thermal fluctuations (mean-field limit as N̄→∞)
    - Larger Δt: Larger noise for faster equilibration (but less accurate)
    - Larger M or V: Reduces noise per grid point (larger system)

    See Also
    --------
    LFTS.run : Main Langevin dynamics loop that uses this noise.

    Examples
    --------
    >>> # Calculate noise for typical L-FTS parameters
    >>> sigma = calculate_sigma(langevin_nbar=10.0, langevin_dt=0.01,
    ...                         n_grids=32**3, volume=4.0**3)
    >>> print(f"Noise standard deviation: {sigma:.6f}")
    """
    return np.sqrt(2*langevin_dt*n_grids/(volume*np.sqrt(langevin_nbar)))

class LFTS:
    """Langevin Field-Theoretic Simulation solver for polymer systems.

    This class implements L-FTS calculations for polymer melts and solutions,
    simulating thermal fluctuations in the field-theoretic representation.
    Unlike SCFT which finds the saddle point, L-FTS samples the full partition
    function including fluctuation effects via Langevin dynamics.

    The implementation uses auxiliary fields obtained from eigenvalue decomposition
    of the interaction matrix, enabling efficient simulation of multi-monomer systems.
    Real-valued auxiliary fields evolve via Langevin dynamics, while imaginary-valued
    fields are compressed to their saddle point values using Anderson Mixing or
    Linear Response methods.

    Parameters
    ----------
    params : dict
        Simulation parameters dictionary containing:

        **Grid and Box Parameters:**

        - nx : list of int
            Number of grid points in each dimension, e.g., [32, 32, 32] for 3D.
        - lx : list of float
            Box dimensions in each direction, units: a_Ref * N_Ref^(1/2).

        **Chain Model Parameters:**

        - chain_model : {'discrete', 'continuous'}
            Chain propagation model. 'continuous' is typical for L-FTS.
        - ds : float
            Contour step size, typically 1/N_Ref.

        **Monomer and Interaction Parameters:**

        - segment_lengths : dict
            Relative statistical segment lengths, {monomer_type: length}.
        - chi_n : dict
            Flory-Huggins interaction parameters × N_Ref, {pair: value}.

        **Polymer Architecture:**

        - distinct_polymers : list of dict
            Polymer chain specifications (same format as SCFT).

        **Langevin Dynamics Parameters:**

        - langevin_dt : float
            Langevin time step Δt. Typical values: 0.001-0.1.
            Smaller values → more accurate but slower.
        - langevin_nbar : float
            Average polymerization index N̄. Typical values: 1-100.
            Controls thermal fluctuation strength (larger → weaker fluctuations).

        **Field Compression Parameters:**

        - compressor : dict
            Configuration for compressing imaginary auxiliary fields:

            For Linear Response (name='lr'):

            - name : str
                'lr' for Linear Response method.
            - lr_saddle_n_iter : int
                Iterations for finding saddle point.
            - lr_tolerance : float
                Convergence tolerance for saddle point.
            - lr_sep_delta : float
                Finite difference step for Hessian calculation.

            For Anderson Mixing (name='am'):

            - name : str
                'am' for Anderson Mixing.
            - max_hist, start_error, mix_min, mix_init : parameters
                Same as SCFT optimizer parameters.

            For LRAM (name='lram'):

            - Uses both Linear Response and Anderson Mixing.
            - Requires both sets of parameters above.

        **Simulation Control:**

        - recording_period : int
            Number of Langevin steps between data recordings.
        - total_recorded_iterations : int
            Total number of recorded snapshots.

        **Platform Selection:**

        - platform : {'cuda', 'cpu-mkl'}, optional
            Computational backend (auto-selected if not specified).
        - reduce_memory_usage : bool, optional
            Enable memory-saving mode.

        **Advanced Options:**

        - aggregate_propagator_computation : bool, optional
            Enable propagator optimization (default: True).

    random_seed : int, optional
        Seed for random number generator (for reproducible noise).
        If None, uses system time.

    Attributes
    ----------
    monomer_types : list of str
        Sorted list of monomer type labels.
    segment_lengths : dict
        Statistical segment lengths for each monomer type.
    chi_n : dict
        Flory-Huggins interaction parameters.
    cb : ComputationBox
        Computation box managing grid and FFT.
    solver : PropagatorComputation
        Propagator solver object.
    mpt : SymmetricPolymerTheory
        Multi-monomer field theory transformations.
    random_g : PCG64
        Random number generator for Langevin noise.

    Raises
    ------
    AssertionError
        If parameter validation fails (volume fractions, monomer types, etc.).

    See Also
    --------
    SCFT : Self-consistent field theory (saddle point approximation).
    SymmetricPolymerTheory : Field theory transformations.
    calculate_sigma : Noise strength calculation.
    LR : Linear Response compressor.
    LRAM : Linear Response + Anderson Mixing compressor.

    Notes
    -----
    **Theoretical Background:**

    L-FTS samples configurations from the field-theoretic partition function:

    .. math::
        Z = \\int \\mathcal{D}w \\, \\exp\\left(-\\frac{H[w]}{\\sqrt{\\bar{N}}}\\right)

    where H is the field-theoretic Hamiltonian and N̄ controls fluctuation strength.

    **Auxiliary Field Representation:**

    The multi-monomer Hamiltonian is transformed via eigenvalue decomposition:

    .. math::
        H = \\sum_i \\lambda_i w_i^2 + \\text{(other terms)}

    This yields auxiliary fields, some real-valued (thermally fluctuating) and
    some imaginary-valued (compressed to saddle point).

    **Langevin Update Scheme:**

    Real auxiliary fields evolve via the Leimkuhler-Matthews method:

    .. math::
        w_{t+1} = w_t - \\Delta t \\frac{\\delta H}{\\delta w} + \\sqrt{2\\Delta t} \\eta

    where η is Gaussian noise with variance 1/√N̄.

    **Field Compression:**

    Imaginary fields are compressed each Langevin step:

    - **LR**: Uses linear response (local Hessian) to find saddle point
    - **AM**: Uses Anderson Mixing iteration
    - **LRAM**: Combines LR for coarse optimization, AM for final convergence

    **Performance:**

    - L-FTS is more expensive than SCFT (~10-100x depending on recording frequency)
    - CUDA provides 10-20x speedup over CPU for 2D/3D
    - Memory usage: ~1GB for 64³ grid with compression history

    **Structure Function Calculation:**

    L-FTS enables computation of structure functions S(k) to characterize
    fluctuations and compare with scattering experiments.

    References
    ----------
    .. [1] Delaney, K. T., & Fredrickson, G. H. "Recent Developments in Fully
           Fluctuating Field-Theoretic Simulations of Polymer Melts and
           Solutions." J. Phys. Chem. B 2016, 120, 7615.
    .. [2] Lee, W.-B., et al. "Fluctuation Effects in Ternary AB+A+B Polymeric
           Emulsions." Macromolecules 2013, 46, 8037.
    .. [3] Kang, H., et al. "Leimkuhler-Matthews Method for Langevin Dynamics."
           Polymers 2021, 13, 2437.

    Examples
    --------
    **Basic L-FTS run for AB diblock:**

    >>> import numpy as np
    >>> from polymerfts import lfts
    >>>
    >>> # Define parameters
    >>> params = {
    ...     "nx": [32, 32, 32],
    ...     "lx": [4.36, 4.36, 4.36],
    ...     "chain_model": "continuous",
    ...     "ds": 1/100,
    ...     "segment_lengths": {"A": 1.0, "B": 1.0},
    ...     "chi_n": {"A,B": 15.0},
    ...     "distinct_polymers": [{
    ...         "volume_fraction": 1.0,
    ...         "blocks": [
    ...             {"type": "A", "length": 0.5},
    ...             {"type": "B", "length": 0.5}
    ...         ]
    ...     }],
    ...     "langevin_dt": 0.01,
    ...     "langevin_nbar": 10.0,
    ...     "compressor": {
    ...         "name": "am",
    ...         "max_hist": 20,
    ...         "start_error": 1e-1,
    ...         "mix_min": 0.1,
    ...         "mix_init": 0.1
    ...     },
    ...     "recording_period": 100,
    ...     "total_recorded_iterations": 1000
    ... }
    >>>
    >>> # Initialize L-FTS with random seed for reproducibility
    >>> sim = lfts.LFTS(params, random_seed=12345)
    >>>
    >>> # Start from SCFT saddle point (recommended)
    >>> # (assuming w_scft obtained from prior SCFT calculation)
    >>> sim.run(initial_fields=w_scft)
    >>>
    >>> # Access recorded data
    >>> # Data is saved automatically during run to "fields_*.mat" files

    **Continue from checkpoint:**

    >>> # Continue from previous L-FTS run
    >>> sim.continue_run("simulation_data.mat")

    **Compute structure function:**

    >>> # After L-FTS run, compute S(k) from recorded configurations
    >>> # (requires loading saved field data and post-processing)

    For complete examples, see:

    - examples/fts/Lamella.py - L-FTS for lamellar phase
    - examples/fts/Gyroid.py - Gyroid phase with fluctuations
    - examples/fts/MixtureBlockRandom.py - Block/random copolymer mixture
    """
    def __init__(self, params, random_seed=None):

        # Segment length
        self.monomer_types = sorted(list(params["segment_lengths"].keys()))
        self.segment_lengths = copy.deepcopy(params["segment_lengths"])
        self.distinct_polymers = copy.deepcopy(params["distinct_polymers"])

        assert(len(self.monomer_types) == len(set(self.monomer_types))), \
            "There are duplicated monomer_types"

        # Choose platform among [cuda, cpu-mkl]
        avail_platforms = _core.PlatformSelector.avail_platforms()
        if "platform" in params:
            platform = params["platform"]
        elif "cpu-mkl" in avail_platforms and len(params["nx"]) == 1: # for 1D simulation, use CPU
            platform = "cpu-mkl"
        elif "cuda" in avail_platforms: # If cuda is available, use GPU
            platform = "cuda"
        else:
            platform = avail_platforms[0]

        # (C++ class) Create a factory for given platform and chain_model
        if "reduce_memory_usage" in params and platform == "cuda":
            factory = _core.PlatformSelector.create_factory(platform, params["reduce_memory_usage"], "real")
        else:
            factory = _core.PlatformSelector.create_factory(platform, False, "real")
        factory.display_info()

        # (C++ class) Computation box
        if "angles" in params:
            self.cb = factory.create_computation_box(params["nx"], params["lx"], angles=params["angles"])
        else:
            self.cb = factory.create_computation_box(params["nx"], params["lx"])

        # Flory-Huggins parameters, χN
        self.chi_n = {}
        for monomer_pair_str, chin_value in params["chi_n"].items():
            monomer_pair = re.split(',| |_|/', monomer_pair_str)
            assert(monomer_pair[0] in self.segment_lengths), \
                f"Monomer type '{monomer_pair[0]}' is not in 'segment_lengths'."
            assert(monomer_pair[1] in self.segment_lengths), \
                f"Monomer type '{monomer_pair[1]}' is not in 'segment_lengths'."
            assert(monomer_pair[0] != monomer_pair[1]), \
                "Do not add self interaction parameter, " + monomer_pair_str + "."
            monomer_pair.sort()
            sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1]
            assert(not sorted_monomer_pair in self.chi_n), \
                f"There are duplicated χN ({sorted_monomer_pair}) parameters."
            self.chi_n[sorted_monomer_pair] = chin_value

        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            monomer_pair = list(monomer_pair)
            monomer_pair.sort()
            sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1] 
            if not sorted_monomer_pair in self.chi_n:
                self.chi_n[sorted_monomer_pair] = 0.0

        # Multimonomer polymer field theory
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n, zeta_n=None)

        # The numbers of real and imaginary fields, respectively
        M = len(self.monomer_types)
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        # Total volume fraction
        assert(len(self.distinct_polymers) >= 1), \
            "There is no polymer chain."

        total_volume_fraction = 0.0
        for polymer in self.distinct_polymers:
            total_volume_fraction += polymer["volume_fraction"]
        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fractions must be equal to 1."

        # Polymer chains
        for polymer_counter, polymer in enumerate(self.distinct_polymers):
            blocks_input = []
            alpha = 0.0             # total_relative_contour_length
            has_node_number = not "v" in polymer["blocks"][0]
            for block in polymer["blocks"]:
                alpha += block["length"]
                if has_node_number:
                    assert(not "v" in block), \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    assert(not "u" in block), \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    blocks_input.append([block["type"], block["length"], len(blocks_input), len(blocks_input)+1])
                else:
                    assert("v" in block), \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    assert("u" in block), \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    blocks_input.append([block["type"], block["length"], block["v"], block["u"]])
            polymer.update({"blocks_input":blocks_input})

        # Random copolymer chains
        self.random_fraction = {}
        for polymer in self.distinct_polymers:

            is_random = False
            for block in polymer["blocks"]:
                if "fraction" in block:
                    is_random = True
            if not is_random:
                continue

            assert(len(polymer["blocks"]) == 1), \
                "Only single block random copolymer is allowed."

            statistical_segment_length = 0
            total_random_fraction = 0
            for monomer_type in polymer["blocks"][0]["fraction"]:
                statistical_segment_length += self.segment_lengths[monomer_type]**2 * polymer["blocks"][0]["fraction"][monomer_type]
                total_random_fraction += polymer["blocks"][0]["fraction"][monomer_type]
            statistical_segment_length = np.sqrt(statistical_segment_length)

            assert(np.isclose(total_random_fraction, 1.0)), \
                "The sum of volume fractions of random copolymer must be equal to 1."

            random_type_string = polymer["blocks"][0]["type"]
            assert(not random_type_string in self.segment_lengths), \
                f"The name of random copolymer '{random_type_string}' is already used as a type in 'segment_lengths' or other random copolymer"

            # Add random copolymers
            polymer["block_monomer_types"] = [random_type_string]
            self.segment_lengths.update({random_type_string:statistical_segment_length})
            self.random_fraction[random_type_string] = polymer["blocks"][0]["fraction"]

        # Make a monomer color dictionary
        dict_color= {}
        colors = ["red", "blue", "green", "cyan", "magenta", "yellow"]
        for count, type in enumerate(self.segment_lengths.keys()):
            if count < len(colors):
                dict_color[type] = colors[count]
            else:
                dict_color[type] = np.random.rand(3,)
        print("Monomer color: ", dict_color)
            
        # Draw polymer chain architectures
        for idx, polymer in enumerate(self.distinct_polymers):
        
            # Make a graph
            G = nx.Graph()
            for block in polymer["blocks_input"]:
                type = block[0]
                length = round(block[1]/params["ds"])
                v = block[2]
                u = block[3]
                G.add_edge(v, u, weight=length, monomer_type=type)

            # Set node colors
            color_map = []
            for node in G:
                if len(G.edges(node)) == 1:
                    color_map.append('yellow')
                else: 
                    color_map.append('gray')

            labels = nx.get_edge_attributes(G, 'weight')
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='twopi')
            colors = [dict_color[G[u][v]['monomer_type']] for u,v in G.edges()]

            plt.figure(figsize=(20,20))
            title = "Polymer ID: %2d," % (idx)
            title += "\nColors of monomers: " + str(dict_color) + ","
            title += "\nColor of chain ends: 'yellow',"
            title += "\nColor of junctions: 'gray',"
            title += "\nPlease note that the length of each edge is not proportional to the number of monomers in this image."
            plt.title(title)
            nx.draw(G, pos, node_color=color_map, edge_color=colors, width=4, with_labels=True) #, node_size=100, font_size=15)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, rotate=False, bbox=dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0), alpha=0.5)) #, font_size=12)
            plt.savefig("polymer_%01d.png" % (idx))

        # (C++ class) Molecules list
        molecules = factory.create_molecules_information(params["chain_model"], params["ds"], self.segment_lengths)

        # Add polymer chains
        for polymer in self.distinct_polymers:
            molecules.add_polymer(polymer["volume_fraction"], polymer["blocks_input"])

        # (C++ class) Propagator Computation Optimizer
        if "aggregate_propagator_computation" in params:
            propagator_computation_optimizer = factory.create_propagator_computation_optimizer(molecules, params["aggregate_propagator_computation"])
        else:
            propagator_computation_optimizer = factory.create_propagator_computation_optimizer(molecules, True)

        # (C++ class) Solver using Pseudo-spectral method
        self.solver = factory.create_pseudospectral_solver(self.cb, molecules, propagator_computation_optimizer)

        # (C++ class) Fields Relaxation using Anderson Mixing
        if params["compressor"]["name"] == "am" or params["compressor"]["name"] == "lram":
            am = factory.create_anderson_mixing(
                len(self.mpt.aux_fields_imag_idx)*np.prod(params["nx"]),   # the number of variables
                params["compressor"]["max_hist"],                          # maximum number of history
                params["compressor"]["start_error"],                       # when switch to AM from simple mixing
                params["compressor"]["mix_min"],                           # minimum mixing rate of simple mixing
                params["compressor"]["mix_init"])                          # initial mixing rate of simple mixing

        # Fields Relaxation using Linear Reponse Method
        if params["compressor"]["name"] == "lr" or params["compressor"]["name"] == "lram":
            assert(I == 1), \
                f"Currently, LR methods are not working for imaginary-valued auxiliary fields."

            w_aux_perturbed = np.zeros([M, self.cb.get_total_grid()], dtype=np.float64)
            w_aux_perturbed[M-1,0] = 1e-3 # add a small perturbation at the pressure field
            w_aux_perturbed_k = np.fft.rfftn(np.reshape(w_aux_perturbed[M-1], self.cb.get_nx()))/np.prod(self.cb.get_nx())

            phi_perturbed, _ = self.compute_concentrations(w_aux_perturbed)
            h_deriv_perturbed = self.mpt.compute_func_deriv(w_aux_perturbed, phi_perturbed, self.mpt.aux_fields_imag_idx)
            h_deriv_perturbed_k = np.fft.rfftn(np.reshape(h_deriv_perturbed, self.cb.get_nx()))/np.prod(self.cb.get_nx())
            jk_numeric = np.real(h_deriv_perturbed_k/w_aux_perturbed_k)
            # print(np.mean(jk_numeric), np.std(jk_numeric))
            # print(np.mean(self.lr.jk), np.std(self.lr.jk))

            lr = LR(self.cb.get_nx(), self.cb.get_lx(), jk_numeric)

        if params["compressor"]["name"] == "am":
            self.compressor = am
        elif params["compressor"]["name"] == "lr":
            self.compressor = lr
        elif params["compressor"]["name"] == "lram":
            self.compressor = LRAM(lr, am)            

        # Standard deviation of normal noise of Langevin dynamics
        langevin_sigma = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"], np.prod(params["nx"]), np.prod(params["lx"]))

        # dH/dw_aux[i] is scaled by dt_scaling[i]
        self.dt_scaling = np.ones(M)
        for i in range(M-1):
            self.dt_scaling[i] = np.abs(self.mpt.eigenvalues[i])/np.max(np.abs(self.mpt.eigenvalues))

        # Set random generator
        if random_seed is None:         
            self.random_bg = np.random.PCG64()  # Set random bit generator
        else:
            self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)
        
        print("---------- Simulation Parameters ----------")
        print("Platform :", platform)
        print("Box Dimension: %d" % (self.cb.get_dim()))
        print("Nx:", self.cb.get_nx())
        print("Lx:", self.cb.get_lx())
        print("dx:", self.cb.get_dx())
        print("Volume: %f" % (self.cb.get_volume()))

        print("Chain model: %s" % (params["chain_model"]))
        print("Segment lengths:\n\t", list(self.segment_lengths.items()))
        print("Conformational asymmetry (epsilon): ")
        for monomer_pair in itertools.combinations(self.monomer_types,2):
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1], self.segment_lengths[monomer_pair[0]]/self.segment_lengths[monomer_pair[1]]))

        print("χN: ")
        for key in self.chi_n:
            print("\t%s: %f" % (key, self.chi_n[key]))

        for p in range(molecules.get_n_polymer_types()):
            print("distinct_polymers[%d]:" % (p) )
            print("\tvolume fraction: %f, alpha: %f, N: %d" %
                (molecules.get_polymer(p).get_volume_fraction(),
                 molecules.get_polymer(p).get_alpha(),
                 molecules.get_polymer(p).get_n_segment_total()))

        print("Invariant Polymerization Index (N_Ref): %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Scaling factor of delta tau N for each field: ", self.dt_scaling)
        print("Random Number Generator: ", self.random_bg.state)

        propagator_computation_optimizer.display_blocks()
        propagator_computation_optimizer.display_propagators()

        #  Save internal variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma":langevin_sigma})

        self.verbose_level = params["verbose_level"]
        self.saddle = params["saddle"].copy()
        self.recording = params["recording"].copy()

        self.molecules = molecules
        self.propagator_computation_optimizer = propagator_computation_optimizer

    def compute_concentrations(self, w_aux):
        """Compute monomer concentration fields from auxiliary fields.

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

        Notes
        -----
        **Field Transformation:**

        Auxiliary fields w_aux are related to monomer fields w via:

        .. math::
            \\mathbf{w} = \\mathbf{A} \\mathbf{w}_{\\text{aux}}

        where A is the eigenvector matrix from field theory eigendecomposition.

        This method:

        1. Transforms w_aux → w using matrix_a_inv
        2. Calls solver.compute_propagators(w) to solve diffusion equations
        3. Calls solver.compute_concentrations() to integrate propagators
        4. Returns concentration dict and timing info

        See Also
        --------
        run : Main L-FTS loop that calls this method.
        find_saddle_point : Finds saddle point of imaginary auxiliary fields.
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
            w_input[random_polymer_name] = np.zeros(self.cb.get_total_grid(), dtype=np.float64)
            for monomer_type, fraction in random_fraction.items():
                w_input[random_polymer_name] += w_input[monomer_type]*fraction

        # For the given fields, compute propagators
        time_solver_start = time.time()
        self.solver.compute_propagators(w_input)
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
                phi[monomer_type] += phi[random_polymer_name]*fraction
        elapsed_time["phi"] = time.time() - time_phi_start

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

        Notes
        -----
        **Saved Data Structure:**

        The MATLAB file contains:

        - initial_params : dict
            Original params dict from __init__.
        - dim, nx, lx : int, list, list
            Grid dimension, points, box sizes.
        - monomer_types, chi_n, chain_model, ds : simulation parameters
        - dt, nbar : Langevin parameters
        - eigenvalues, aux_fields_real, aux_fields_imag : field theory data
        - matrix_a, matrix_a_inverse : field transformation matrices
        - langevin_step : current iteration number
        - random_generator, random_state_state, random_state_inc : RNG state
        - normal_noise_prev : previous Langevin noise
        - w_{monomer_type} : potential fields for each monomer
        - phi_{monomer_type} : concentration fields for each monomer

        **Random Generator State:**

        The PCG64 random number generator state is saved to enable exact
        continuation of the stochastic dynamics. This ensures reproducibility
        when restarting simulations.

        **File Format:**

        - MATLAB .mat format (scipy.io.savemat)
        - Uses compression (do_compression=True)
        - Supports long field names (long_field_names=True)

        See Also
        --------
        continue_run : Load checkpoint and continue simulation.
        run : Main L-FTS loop that calls this method.

        Examples
        --------
        >>> # During L-FTS run, save checkpoint
        >>> sim.save_simulation_data("checkpoint_step_1000.mat", w, phi,
        ...                          langevin_step=1000, normal_noise_prev=noise)
        >>>
        >>> # Later, continue from this checkpoint
        >>> sim.continue_run("checkpoint_step_1000.mat")
        """

        # Make a dictionary for chi_n
        chi_n_mat = {}
        for key in self.chi_n:
            chi_n_mat[key] = self.chi_n[key]

        # Make dictionary for data
        mdic = {
            "initial_params": self.params,
            "dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "monomer_types":self.monomer_types, "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
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

        Notes
        -----
        **Restored State:**

        This method restores:

        - Random number generator (PCG64) state for exact continuation
        - Field configurations (w fields)
        - Langevin step counter
        - Previous noise array (normal_noise_prev) for Leimkuhler-Matthews scheme

        **Structure Function Recording:**

        If the checkpoint's langevin_step is not a multiple of sf_recording_period,
        a warning is printed. The structure function will only be correctly recorded
        after the next multiple of sf_recording_period.

        **Workflow:**

        1. Load checkpoint data from file
        2. Restore random generator state
        3. Extract field configurations
        4. Call run() with restored state and start_langevin_step = loaded_step + 1

        See Also
        --------
        save_simulation_data : Save checkpoint file.
        run : Main L-FTS loop called by this method.

        Examples
        --------
        >>> # Start new L-FTS simulation
        >>> sim = lfts.LFTS(params, random_seed=12345)
        >>> sim.run(initial_fields=w_init)
        >>>
        >>> # Simulation saves checkpoints automatically during run
        >>> # Later, continue from last checkpoint
        >>> sim_continued = lfts.LFTS(params)  # Create new instance
        >>> sim_continued.continue_run("simulation_data.mat")
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

        Performs Langevin dynamics to sample thermal fluctuations in the
        field-theoretic representation of the polymer system. At each Langevin
        step:

        1. Computes functional derivative δH/δw for real auxiliary fields
        2. Updates real fields via Leimkuhler-Matthews method with Gaussian noise
        3. Finds saddle point of imaginary auxiliary fields via compression
        4. Records fields, Hamiltonian, and structure function data
        5. Saves checkpoint periodically

        The simulation continues for langevin.max_step iterations or until
        total_recorded_iterations snapshots are recorded.

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
            Used when continuing from checkpoint.

        Notes
        -----
        **Langevin Dynamics Update:**

        Real-valued auxiliary fields evolve via the Leimkuhler-Matthews method:

        .. math::
            w_{t+1} = w_t - \\Delta t \\frac{\\delta H}{\\delta w} + \\frac{1}{2}(\\eta_t + \\eta_{t+1})

        where η_t is Gaussian noise with standard deviation σ (from calculate_sigma).
        This 2nd-order scheme provides better accuracy than Euler-Maruyama.

        **Field Compression:**

        Imaginary-valued auxiliary fields are compressed to their saddle point
        at each Langevin step using the configured compressor:

        - **AM**: Anderson Mixing iteration
        - **LR**: Linear Response (local Hessian approximation)
        - **LRAM**: Linear Response with Anderson Mixing refinement

        If saddle point finding fails (error > tolerance), the Langevin step
        is rejected and noise is regenerated. After 5 successive failures,
        the simulation continues anyway (prints warning).

        **Data Recording:**

        Fields are periodically saved according to recording parameters:

        - Fields saved to "dir/fields_*.mat" every field_recording_period steps
        - Hamiltonian H and derivatives dH/dχN recorded every recording_period steps
        - Structure functions <φ(k)φ(-k)> accumulated every sf_recording_period steps

        **Structure Function:**

        The structure function quantifies density fluctuations in Fourier space:

        .. math::
            S(\\mathbf{k}) = \\langle \\phi(\\mathbf{k}) \\phi^*(-\\mathbf{k}) \\rangle

        Averaged over recorded configurations to compare with scattering experiments.

        **Performance:**

        L-FTS is computationally expensive:

        - Each Langevin step requires 1+ saddle point iterations
        - Typical runs: 10,000-100,000 Langevin steps
        - Runtime: hours to days depending on system size and recording frequency
        - CUDA provides 10-20x speedup over CPU

        **Checkpointing:**

        Simulation data is automatically saved to enable restart:

        - Checkpoint: "dir/simulation_data.mat" (latest state)
        - Field snapshots: "dir/fields_<step>.mat" (periodic)
        - Contains all state needed for continue_run()

        **Error Handling:**

        - If saddle point fails repeatedly, step is accepted with warning
        - NaN in concentrations/Hamiltonian triggers step rejection
        - Random noise is regenerated on failed steps

        See Also
        --------
        compute_concentrations : Compute φ from auxiliary fields.
        find_saddle_point : Find saddle point of imaginary fields.
        save_simulation_data : Save checkpoint data.
        continue_run : Resume from checkpoint.
        calculate_sigma : Noise strength calculation.

        Examples
        --------
        **Basic L-FTS run starting from SCFT:**

        >>> import numpy as np
        >>> from polymerfts import scft, lfts
        >>>
        >>> # First, run SCFT to find saddle point
        >>> scft_calc = scft.SCFT(scft_params)
        >>> scft_calc.run(initial_fields=w_random)
        >>>
        >>> # Use SCFT result as initial condition for L-FTS
        >>> w_scft = {m: scft_calc.w[i] for i, m in enumerate(scft_calc.monomer_types)}
        >>>
        >>> # Run L-FTS
        >>> lfts_params = {
        ...     # ... (grid, polymers, chi_n same as SCFT)
        ...     "langevin": {
        ...         "max_step": 100000,
        ...         "dt": 0.01,
        ...         "nbar": 10.0
        ...     },
        ...     "recording": {
        ...         "dir": "recording",
        ...         "recording_period": 100,
        ...         "sf_recording_period": 10,
        ...         "field_recording_period": 1000
        ...     },
        ...     "compressor": {
        ...         "name": "am",
        ...         "max_hist": 20,
        ...         "start_error": 1e-1,
        ...         "mix_min": 0.1,
        ...         "mix_init": 0.1
        ...     },
        ...     # ... other params
        ... }
        >>> sim = lfts.LFTS(lfts_params, random_seed=12345)
        >>> sim.run(initial_fields=w_scft)

        **Continue from checkpoint:**

        >>> # If simulation was interrupted, continue from last checkpoint
        >>> sim_new = lfts.LFTS(lfts_params)
        >>> sim_new.continue_run("recording/simulation_data.mat")

        **Analyze structure function:**

        >>> # After L-FTS run, load and analyze structure function data
        >>> import scipy.io
        >>> data = scipy.io.loadmat("recording/fields_statistics.mat")
        >>> S_k = data["structure_function"]
        >>> # ... analyze S(k) for comparison with experiments

        For complete examples, see:

        - examples/fts/Lamella.py - Basic L-FTS workflow
        - examples/fts/Gyroid.py - Gyroid phase with fluctuations
        """

        print("---------- Run  ----------")

        # The number of components
        M = len(self.monomer_types)

        # The numbers of real and imaginary fields, respectively
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        # Simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # Reshape initial fields
        w = np.zeros([M, self.cb.get_total_grid()], dtype=np.float64)
        for i in range(M):
            w[i] = np.reshape(initial_fields[self.monomer_types[i]],  self.cb.get_total_grid())
            
        # Convert monomer potential fields into auxiliary potential fields
        w_aux = self.mpt.to_aux_fields(w)

        # Find saddle point
        print("iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)")
        phi, _, _, _, = self.find_saddle_point(w_aux=w_aux)

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
            sf_average[type_pair] = np.zeros_like(np.fft.rfftn(np.reshape(w[0], self.cb.get_nx())), np.complex128)

        # Create an empty array for field update algorithm
        if normal_noise_prev is None :
            normal_noise_prev = np.zeros([R, self.cb.get_total_grid()], dtype=np.float64)
        else:
            normal_noise_prev = normal_noise_prev

        if start_langevin_step is None :
            start_langevin_step = 1

        # The number of times that 'find_saddle_point' has failed to find a saddle point
        saddle_fail_count = 0
        successive_fail_count = 0

        # Init timers
        total_saddle_iter = 0
        total_error_level = 0
        time_start = time.time()

        # Langevin iteration begins here
        for langevin_step in range(start_langevin_step, self.langevin["max_step"]+1):
            print("Langevin step: ", langevin_step)

            # Copy data for restoring
            w_aux_copy = w_aux.copy()
            phi_copy = phi.copy()

            # Compute functional derivatives of Hamiltonian w.r.t. real-valued fields 
            w_lambda = self.mpt.compute_func_deriv(w_aux, phi, self.mpt.aux_fields_real_idx)

            # Update w_aux using Leimkuhler-Matthews method
            normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], [R, self.cb.get_total_grid()])
            for count, i in enumerate(self.mpt.aux_fields_real_idx):
                scaling = self.dt_scaling[i]
                w_aux[i] += -w_lambda[count]*self.langevin["dt"]*scaling + 0.5*(normal_noise_prev[count] + normal_noise_current[count])*np.sqrt(scaling)

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            # Find saddle point of the pressure field
            phi, hamiltonian, saddle_iter, error_level = self.find_saddle_point(w_aux=w_aux)
            total_saddle_iter += saddle_iter
            total_error_level += error_level

            # If the tolerance of the saddle point was not met, regenerate Langevin random noise and continue
            if np.isnan(error_level) or error_level >= self.saddle["tolerance"]:
                if successive_fail_count < 5:                
                    print("The tolerance of the saddle point was not met. Langevin random noise is regenerated.")

                    # Restore w_aux and phi
                    w_aux = w_aux_copy
                    phi = phi_copy
                    
                    # Increment counts and continue
                    successive_fail_count += 1
                    saddle_fail_count += 1
                    continue
                else:
                    print("The tolerance of the saddle point was not met %d times in a row. Simulation is aborted." % (successive_fail_count))
                    break
            else:
                successive_fail_count = 0

            # Compute H and dH/dχN
            if langevin_step % self.recording["sf_computing_period"] == 0:
                H_history.append(hamiltonian)
                dH = self.mpt.compute_h_deriv_chin(self.chi_n, w_aux)
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
                # Perform Fourier transforms
                mu_fourier = {}
                phi_fourier = {}
                for i in range(M):
                    key = self.monomer_types[i]
                    phi_fourier[key] = np.fft.rfftn(np.reshape(phi[self.monomer_types[i]], self.cb.get_nx()))/self.cb.get_total_grid()
                    mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
                    for k in range(M-1) :
                        mu_fourier[key] += np.fft.rfftn(np.reshape(w_aux[k], self.cb.get_nx()))*self.mpt.matrix_a_inv[k,i]/self.mpt.eigenvalues[k]/self.cb.get_total_grid()
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
                mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
                        "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
                        "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "initial_params":self.params}
                # Add structure functions to the dictionary
                for key in sf_average:
                    sf_average[key] *= self.recording["sf_computing_period"]/self.recording["sf_recording_period"]* \
                            self.cb.get_volume()*np.sqrt(self.langevin["nbar"])
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

        print("total time: %f, time per step: %f" %
            (time_duration, time_duration/(langevin_step+1-start_langevin_step)) )
        
        print("Total iterations for saddle points: %d, Iterations per Langevin step: %f" %
            (total_saddle_iter, total_saddle_iter/(langevin_step+1-start_langevin_step)))

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

        Notes
        -----
        **Saddle Point Condition:**

        For imaginary auxiliary fields, the Hamiltonian must be stationary:

        .. math::
            \\frac{\\delta H}{\\delta w_{\\text{imag},i}} = 0

        This is solved iteratively using the configured compressor:

        - **AM**: Anderson Mixing iteration
        - **LR**: Linear Response using local Hessian
        - **LRAM**: LR for coarse optimization, AM for refinement

        **Incompressibility:**

        The last imaginary field (w_aux[M-1]) is the pressure field enforcing
        incompressibility constraint Σ_i φ_i = 1. Its saddle point condition
        ensures ⟨Σφ_i - 1⟩ = 0.

        **Convergence Criterion:**

        Iteration stops when:

        .. math::
            \\text{error\\_level} = \\max_i \\text{std}(\\delta H / \\delta w_{\\text{imag},i}) < \\text{tolerance}

        **Error Handling:**

        - If error_level > tolerance after max_iter, returns with NaN
        - If concentrations become negative, returns with high error
        - Calling code (run method) may regenerate Langevin noise on failure

        **Computational Cost:**

        - Each iteration calls compute_concentrations (propagator solve)
        - Typical: 5-20 iterations per Langevin step
        - Dominates L-FTS runtime

        **Pressure Field Normalization:**

        After convergence, the pressure field mean is set to zero:
        w_aux[M-1] -= mean(w_aux[M-1]). This gauge choice doesn't affect
        physics but improves numerical stability.

        See Also
        --------
        run : Main L-FTS loop that calls this method.
        compute_concentrations : Computes φ for given w_aux.
        LR : Linear Response compressor.
        LRAM : Combined Linear Response + Anderson Mixing.

        Examples
        --------
        >>> # Called internally by run() at each Langevin step
        >>> phi, H, iterations, error = sim.find_saddle_point(w_aux)
        >>>
        >>> # Check convergence
        >>> if error < sim.saddle["tolerance"]:
        ...     print(f"Converged in {iterations} iterations")
        ...     print(f"Hamiltonian: {H:.6f}")
        ... else:
        ...     print(f"Failed to converge, error: {error:.3e}")
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

        # Saddle point iteration begins here
        for saddle_iter in range(1,self.saddle["max_iter"]+1):
            
            # Compute total concentrations
            phi, _ = self.compute_concentrations(w_aux)

            # Compute functional derivatives of Hamiltonian w.r.t. imaginary fields 
            h_deriv = self.mpt.compute_func_deriv(w_aux, phi, self.mpt.aux_fields_imag_idx)

            # Compute total error
            old_error_level = error_level
            error_level_array = np.std(h_deriv, axis=1)
            error_level = np.max(error_level_array)

            # Print iteration # and error levels
            if(self.verbose_level == 2 or self.verbose_level == 1 and
            (error_level < self.saddle["tolerance"] or saddle_iter == self.saddle["max_iter"])):
            
                # Calculate Hamiltonian
                total_partitions = [self.solver.get_total_partition(p) for p in range(self.molecules.get_n_polymer_types())]
                hamiltonian = self.mpt.compute_hamiltonian(self.molecules, w_aux, total_partitions, include_const_term=True)

                # Check the mass conservation
                mass_error = np.mean(h_deriv[I-1])
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
            w_aux[self.mpt.aux_fields_imag_idx] = np.reshape(self.compressor.calculate_new_fields(w_aux[self.mpt.aux_fields_imag_idx], -h_deriv, old_error_level, error_level), [I, self.cb.get_total_grid()])

        # Set mean of pressure field to zero
        w_aux[M-1] -= np.mean(w_aux[M-1])

        return phi, hamiltonian, saddle_iter, error_level