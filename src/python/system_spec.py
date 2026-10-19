"""Shared configuration layer for the SCFT, L-FTS and CL-FTS simulation classes.

The user describes a simulation with a plain parameter dictionary. The parts of
that dictionary that all three simulation classes share (grid, chain model,
monomer types, interaction parameters, polymer definitions, platform) are
interpreted and validated in exactly one place, :func:`parse_system`, which
returns an immutable :class:`SystemSpec`. The simulation classes then build
their C++ objects from the spec instead of re-reading the dictionary.

Method-specific options (``optimizer``, ``langevin``, ``compressor``, ...)
stay with the simulation class that owns them.

Example
-------
>>> spec = parse_system(params)
>>> spec.grid.nx, spec.chain.model, spec.monomer_types
((32, 32, 32), 'continuous', ('A', 'B'))
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import _core
from .validation import validate_system_params, ValidationError
from .utils import (
    parse_chi_n,
    validate_polymers,
    process_polymer_blocks,
    process_random_copolymer,
)


@dataclass(frozen=True)
class GridSpec:
    """Simulation grid: grid points, box lengths, boundary conditions, angles.

    Attributes
    ----------
    nx : tuple of int
        Number of grid points per dimension.
    lx : tuple of float
        Box lengths per dimension (units of a_Ref * N_Ref^(1/2)).
    bc : tuple of str
        Boundary conditions, two entries per dimension (lower, upper).
    angles : tuple of float or None
        Lattice angles [alpha, beta, gamma] in degrees, or None for an
        orthogonal box.
    """
    nx: Tuple[int, ...]
    lx: Tuple[float, ...]
    bc: Tuple[str, ...]
    angles: Optional[Tuple[float, ...]] = None

    @property
    def dim(self) -> int:
        return len(self.nx)

    @property
    def total_grid(self) -> int:
        return int(np.prod(self.nx))


@dataclass(frozen=True)
class ChainSpec:
    """Chain model and contour discretization.

    Attributes
    ----------
    model : str
        ``"continuous"`` or ``"discrete"``.
    ds : float
        Contour step (1/N_Ref).
    segment_lengths : dict
        Relative statistical segment length per monomer type. Random
        copolymer pseudo-types are included with their effective length.
    numerical_method : str or None
        Propagator solver requested by the user: ``"rqm4"``, ``"rk2"`` or
        ``"cn-adi2"``; None when not given (the solver then uses RQM4 for
        continuous chains and the bond-convolution scheme for discrete chains).
    """
    model: str
    ds: float
    segment_lengths: Dict[str, float]
    numerical_method: Optional[str] = None


@dataclass(frozen=True)
class SystemSpec:
    """Validated description of the polymer system shared by all simulations.

    Attributes
    ----------
    grid : GridSpec
    chain : ChainSpec
    monomer_types : tuple of str
        Sorted monomer type labels (random copolymer pseudo-types excluded).
    chi_n : dict
        Flory-Huggins parameters times N_Ref with sorted ``"A,B"`` keys. Every
        monomer pair is present; pairs the user did not give are 0.0.
    zeta_n : float or None
        Compressibility parameter, or None when the user gave none.
    polymers : tuple of dict
        Deep copies of ``distinct_polymers`` with the processed
        ``"blocks_input"`` list (``[type, length, v, u]`` per block) and, for
        random copolymers, ``"block_monomer_types"``.
    random_fraction : dict
        Monomer fractions of each random copolymer pseudo-type.
    platform : str
        Computational backend: ``"cuda"``, ``"cpu-mkl"`` or ``"cpu-fftw"``.
    reduce_memory : bool
        Store only propagator checkpoints.
    aggregate_propagator_computation : bool
        Let the propagator optimizer merge equivalent computations.
    """
    grid: GridSpec
    chain: ChainSpec
    monomer_types: Tuple[str, ...]
    chi_n: Dict[str, float]
    zeta_n: Optional[float]
    polymers: Tuple[Dict[str, Any], ...]
    random_fraction: Dict[str, Dict[str, float]]
    platform: str
    reduce_memory: bool = False
    aggregate_propagator_computation: bool = True


def select_platform(requested: Optional[str], dim: int) -> str:
    """Pick the computational platform.

    An explicit request wins. Otherwise 1D runs prefer the CPU (GPU overhead
    outweighs the gain), 2D/3D runs prefer CUDA when it is available, and the
    first available platform is the last resort.
    """
    avail_platforms = _core.PlatformSelector.avail_platforms()
    if requested is not None:
        return requested
    if "cpu-mkl" in avail_platforms and dim == 1:
        return "cpu-mkl"
    if "cuda" in avail_platforms:
        return "cuda"
    if not avail_platforms:
        raise RuntimeError("No computational platform available in this build.")
    return avail_platforms[0]


def parse_system(params: Dict[str, Any], validate: bool = True) -> SystemSpec:
    """Interpret and validate the shared part of a parameter dictionary.

    Parameters
    ----------
    params : dict
        Simulation parameters. Required keys: ``nx``, ``lx``, ``ds``,
        ``chain_model``, ``segment_lengths``, ``chi_n``, ``distinct_polymers``.
        Optional keys read here: ``bc``, ``angles``, ``zeta_n``, ``platform``,
        ``reduce_memory``, ``numerical_method``,
        ``aggregate_propagator_computation``.
    validate : bool
        Run :func:`validate_system_params` first (default). Callers that have
        already validated ``params`` can pass False.

    Returns
    -------
    SystemSpec

    Raises
    ------
    ValidationError
        If any shared parameter is missing or invalid.
    """
    if validate:
        validate_system_params(params)
    if "chain_model" not in params:
        raise ValidationError("Missing required parameter: 'chain_model'")

    nx = tuple(int(n) for n in params["nx"])
    lx = tuple(params["lx"])
    dim = len(nx)

    bc = params.get("bc", None)
    if bc is None:
        bc = ["periodic"] * (2 * dim)   # default to periodic boundaries
    bc = tuple(bc)

    angles = params.get("angles", None)
    angles = tuple(angles) if angles is not None else None

    grid = GridSpec(nx=nx, lx=lx, bc=bc, angles=angles)

    # Monomer types and statistical segment lengths
    segment_lengths = copy.deepcopy(params["segment_lengths"])
    monomer_types = tuple(sorted(segment_lengths.keys()))

    # Flory-Huggins parameters; fill missing pairs with zero interaction
    chi_n = parse_chi_n(params["chi_n"], segment_lengths)
    for pair in itertools.combinations(monomer_types, 2):
        key = ",".join(sorted(pair))
        if key not in chi_n:
            chi_n[key] = 0.0

    # Polymer chains: block connectivity and random copolymers
    polymers: List[Dict[str, Any]] = copy.deepcopy(params["distinct_polymers"])
    validate_polymers(polymers)
    random_fraction: Dict[str, Dict[str, float]] = {}
    for polymer in polymers:
        polymer["blocks_input"] = process_polymer_blocks(polymer)
    for polymer in polymers:
        result = process_random_copolymer(polymer, segment_lengths)
        if result is not None:
            random_type, effective_length, fraction = result
            polymer["block_monomer_types"] = [random_type]
            segment_lengths[random_type] = effective_length
            random_fraction[random_type] = fraction

    chain = ChainSpec(
        model=params["chain_model"],
        ds=params["ds"],
        segment_lengths=segment_lengths,
        numerical_method=params.get("numerical_method", None),
    )

    return SystemSpec(
        grid=grid,
        chain=chain,
        monomer_types=monomer_types,
        chi_n=chi_n,
        zeta_n=params.get("zeta_n", None),
        polymers=tuple(polymers),
        random_fraction=random_fraction,
        platform=select_platform(params.get("platform", None), dim),
        reduce_memory=bool(params.get("reduce_memory", False)),
        aggregate_propagator_computation=bool(params.get("aggregate_propagator_computation", True)),
    )
