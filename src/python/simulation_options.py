"""Method-specific options of the simulation classes.

:func:`polymerfts.system_spec.parse_system` covers the part of a parameter
dictionary that SCFT, L-FTS and CL-FTS share (grid, chain model, monomers,
interactions, polymers, platform). The dataclasses here cover the rest. Each
simulation class reads its own options exactly once through ``from_params``
and keeps the result as ``self.options``; the defaults live here and nowhere
else.

The dictionaries inside an options object are deep copies of the user's
input, so a simulation never mutates the parameter dictionary it was given.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .validation import ValidationError


def _require(params: Dict[str, Any], key: str, owner: str) -> Any:
    if key not in params:
        raise ValidationError(f"Missing required parameter for {owner}: '{key}'")
    return params[key]


@dataclass(frozen=True)
class ScftOptions:
    """Options of :class:`polymerfts.scft.SCFT`.

    Attributes
    ----------
    optimizer : dict
        Field optimizer: ``{"name": "am", "max_hist", "start_error", "mix_min",
        "mix_init"}`` or ``{"name": "adam", "lr", "gamma"}``.
    box_is_altering : bool
        Relax the box lengths (and, for monoclinic/triclinic/oblique crystal
        systems, the angles) together with the fields.
    max_iter : int
        Maximum number of SCFT iterations.
    tolerance : float
        Stop when the self-consistency error falls below this value.
    stress_interval : int or "adaptive"
        Compute the stress every N iterations, or adaptively (more often
        while the error is large).
    scale_stress : float
        Scaling of the stress entries in the optimizer vector.
    fts_consistent_energy : bool
        Pin the field means to the saddle point so that the SCFT free energy
        matches the field-theoretic Hamiltonian of L-FTS/CL-FTS.
    space_group : dict or None
        ``{"symbol": Hermann-Mauguin symbol, "number": Hall number (optional)}``.
    crystal_system : str or None
        Which lattice parameters may vary; None selects it automatically.
    smearing : any or None
        Smearing parameters (not available together with a space group).

    ``axis_ordering`` is not an option of the solver: SCFT applies it to
    nx/lx/angles before the system is parsed (see SCFT._apply_axis_ordering).
    """
    optimizer: Dict[str, Any]
    box_is_altering: bool
    max_iter: int = 2000
    tolerance: float = 1e-8
    stress_interval: Union[int, str] = 1
    scale_stress: float = 1
    fts_consistent_energy: bool = False
    space_group: Optional[Dict[str, Any]] = None
    crystal_system: Optional[str] = None
    smearing: Optional[Any] = None

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "ScftOptions":
        return cls(
            optimizer=copy.deepcopy(_require(params, "optimizer", "SCFT")),
            box_is_altering=bool(_require(params, "box_is_altering", "SCFT")),
            max_iter=params.get("max_iter", cls.max_iter),
            tolerance=params.get("tolerance", cls.tolerance),
            stress_interval=params.get("stress_interval", cls.stress_interval),
            scale_stress=params.get("scale_stress", cls.scale_stress),
            fts_consistent_energy=params.get("fts_consistent_energy", cls.fts_consistent_energy),
            space_group=copy.deepcopy(params.get("space_group", None)),
            crystal_system=params.get("crystal_system", None),
            smearing=copy.deepcopy(params.get("smearing", None)),
        )


@dataclass(frozen=True)
class LftsOptions:
    """Options of :class:`polymerfts.lfts.LFTS`.

    Attributes
    ----------
    langevin : dict
        ``{"nbar": invariant polymerization index, "dt": Langevin step,
        "max_step": number of steps}``.
    compressor : dict
        Saddle-point compressor: ``{"name": "am" | "lr" | "lram", ...}``.
    saddle : dict
        Saddle-point iteration control (``max_iter``, ``tolerance``).
    recording : dict
        Output control (``dir``, ``recording_period``, ``sf_computing_period``,
        ``sf_recording_period``).
    verbose_level : int
        Amount of console output.
    wtmd : dict or None
        Well-tempered metadynamics parameters.
    validation_runtime : any or None
        Runtime partition-function checks.
    smearing : any or None
        Smearing parameters.
    """
    langevin: Dict[str, Any]
    compressor: Dict[str, Any]
    saddle: Dict[str, Any]
    recording: Dict[str, Any]
    verbose_level: int
    wtmd: Optional[Dict[str, Any]] = None
    validation_runtime: Optional[Any] = None
    smearing: Optional[Any] = None

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "LftsOptions":
        return cls(
            langevin=copy.deepcopy(_require(params, "langevin", "LFTS")),
            compressor=copy.deepcopy(_require(params, "compressor", "LFTS")),
            saddle=copy.deepcopy(_require(params, "saddle", "LFTS")),
            recording=copy.deepcopy(_require(params, "recording", "LFTS")),
            verbose_level=_require(params, "verbose_level", "LFTS"),
            wtmd=copy.deepcopy(params.get("wtmd", None)),
            validation_runtime=copy.deepcopy(params.get("validation_runtime", None)),
            smearing=copy.deepcopy(params.get("smearing", None)),
        )


@dataclass(frozen=True)
class ClftsOptions:
    """Options of :class:`polymerfts.clfts.CLFTS`.

    Attributes
    ----------
    langevin : dict
        ``{"nbar", "dt", "max_step"}`` as for L-FTS.
    recording : dict
        Output control.
    verbose_level : int
        Amount of console output.
    alpha_ds : float
        Dynamical stabilization constant (Willis & Matsen 2024); given as
        ``"dynamic_stabilization"`` or its alias ``"alpha_ds"``.
    smearing : any or None
        Smearing parameters.
    """
    langevin: Dict[str, Any]
    recording: Dict[str, Any]
    verbose_level: int
    alpha_ds: float = 0.0
    smearing: Optional[Any] = None

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "ClftsOptions":
        if "dynamic_stabilization" in params:
            alpha_ds = params["dynamic_stabilization"]
        elif "alpha_ds" in params:
            alpha_ds = params["alpha_ds"]
        else:
            alpha_ds = cls.alpha_ds
        return cls(
            langevin=copy.deepcopy(_require(params, "langevin", "CLFTS")),
            recording=copy.deepcopy(_require(params, "recording", "CLFTS")),
            verbose_level=_require(params, "verbose_level", "CLFTS"),
            alpha_ds=alpha_ds,
            smearing=copy.deepcopy(params.get("smearing", None)),
        )
