"""Structured result objects for polymer field theory simulations.

This module provides dataclass-based result objects that encapsulate
simulation outputs in a clean, documented interface.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class SCFTResult:
    """Result object for SCFT simulations.

    Encapsulates all outputs from an SCFT simulation run, providing
    a clean interface for accessing results.

    Attributes
    ----------
    converged : bool
        Whether the simulation converged within tolerance.
    free_energy : float
        Final free energy per unit volume (dimensionless).
    error_level : float
        Final convergence error (L2 norm of field residuals).
    iterations : int
        Number of iterations performed.
    phi : dict
        Monomer concentration fields {monomer_type: array}.
    w : np.ndarray
        Potential fields, shape (M, total_grid).
    partition_functions : list
        Single-chain partition functions for each polymer type.
    stress : np.ndarray or None
        Stress tensor components (if box_is_altering was True).
    lx : list
        Final box dimensions.
    nx : list
        Grid dimensions.
    monomer_types : list
        List of monomer type labels.
    params : dict
        Original simulation parameters.

    Examples
    --------
    >>> calc = SCFT(params)
    >>> result = calc.run(initial_fields)
    >>> if result.converged:
    ...     print(f"Free energy: {result.free_energy}")
    ...     phi_A = result.phi["A"]
    """
    converged: bool
    free_energy: float
    error_level: float
    iterations: int
    phi: Dict[str, np.ndarray]
    w: np.ndarray
    partition_functions: List[float]
    lx: List[float]
    nx: List[int]
    monomer_types: List[str]
    params: Dict[str, Any]
    stress: Optional[np.ndarray] = None
    angles: Optional[List[float]] = None

    def get_field(self, monomer_type: str) -> np.ndarray:
        """Get potential field for a monomer type.

        Parameters
        ----------
        monomer_type : str
            Monomer type label (e.g., "A", "B").

        Returns
        -------
        np.ndarray
            Potential field for the specified monomer.

        Raises
        ------
        KeyError
            If monomer type not found.
        """
        idx = self.monomer_types.index(monomer_type)
        return self.w[idx]

    def get_concentration(self, monomer_type: str) -> np.ndarray:
        """Get concentration field for a monomer type.

        Parameters
        ----------
        monomer_type : str
            Monomer type label (e.g., "A", "B").

        Returns
        -------
        np.ndarray
            Concentration field for the specified monomer.

        Raises
        ------
        KeyError
            If monomer type not found.
        """
        return self.phi[monomer_type]

    def reshape_field(self, field: np.ndarray) -> np.ndarray:
        """Reshape a flattened field to grid dimensions.

        Parameters
        ----------
        field : np.ndarray
            Flattened field array.

        Returns
        -------
        np.ndarray
            Field reshaped to nx dimensions.
        """
        return field.reshape(self.nx)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns
        -------
        dict
            Dictionary containing all result data.
        """
        result = {
            "converged": self.converged,
            "free_energy": self.free_energy,
            "error_level": self.error_level,
            "iterations": self.iterations,
            "lx": self.lx,
            "nx": self.nx,
            "monomer_types": self.monomer_types,
            "partition_functions": self.partition_functions,
        }

        # Add fields
        for i, mt in enumerate(self.monomer_types):
            result[f"w_{mt}"] = self.w[i]
            result[f"phi_{mt}"] = self.phi[mt]

        if self.stress is not None:
            result["stress"] = self.stress

        if self.angles is not None:
            result["angles"] = self.angles

        return result


@dataclass
class LFTSResult:
    """Result object for L-FTS simulations.

    Encapsulates outputs from a Langevin FTS simulation step or run.

    Attributes
    ----------
    step : int
        Current Langevin step number.
    hamiltonian : float
        Current Hamiltonian value.
    phi : dict
        Monomer concentration fields {monomer_type: array}.
    w : np.ndarray
        Current potential fields.
    structure_function : np.ndarray or None
        Structure function S(k) if computed.
    lx : list
        Current box dimensions.
    nx : list
        Grid dimensions.
    monomer_types : list
        List of monomer type labels.
    """
    step: int
    hamiltonian: float
    phi: Dict[str, np.ndarray]
    w: np.ndarray
    lx: List[float]
    nx: List[int]
    monomer_types: List[str]
    structure_function: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            "step": self.step,
            "hamiltonian": self.hamiltonian,
            "lx": self.lx,
            "nx": self.nx,
            "monomer_types": self.monomer_types,
        }

        for i, mt in enumerate(self.monomer_types):
            result[f"w_{mt}"] = self.w[i]
            result[f"phi_{mt}"] = self.phi[mt]

        if self.structure_function is not None:
            result["structure_function"] = self.structure_function

        return result


@dataclass
class IterationInfo:
    """Information about a single SCFT iteration.

    Used for progress callbacks to provide iteration details.

    Attributes
    ----------
    iteration : int
        Current iteration number.
    error_level : float
        Current convergence error.
    energy : float
        Current free energy.
    mass_error : float
        Mass conservation error.
    partition_functions : list
        Partition functions for each polymer.
    lx : list or None
        Current box dimensions (if box_is_altering).
    angles : list or None
        Current lattice angles (if non-orthogonal).
    """
    iteration: int
    error_level: float
    energy: float
    mass_error: float
    partition_functions: List[float]
    lx: Optional[List[float]] = None
    angles: Optional[List[float]] = None
