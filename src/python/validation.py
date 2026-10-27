"""Parameter checks for SCFT, L-FTS and CL-FTS.

The checks raise ValidationError (never assert, which disappears under
``python -O``) before any expensive C++ object is created.
"""

import copy
from typing import Any, Dict
from os import PathLike

import numpy as np


class ValidationError(Exception):
    """Raised when a simulation parameter is missing or invalid."""
    pass


class SaddlePointError(RuntimeError):
    """A Langevin run stopped before accepting a step; see ``sim.run_status``."""


def _rename_parameter(mapping, old, new, path=""):
    if old in mapping:
        if new in mapping:
            raise ValidationError(f"Specify only one of '{path}{old}' and '{path}{new}'")
        mapping[new] = mapping.pop(old)


def normalize_params(params):
    """Copy input parameters and translate supported legacy names.

    Normalization stays at the input boundary so the simulation loops use only
    the documented names. Supplying both names is an error, even if equal.
    """
    if not isinstance(params, dict):
        raise ValidationError("Simulation parameters must be a dict")
    params = copy.deepcopy(params)
    for old, new in (("distinct_polymers", "polymers"),
                     ("box_is_altering", "relax_cell"),
                     ("scale_stress", "stress_scale"),
                     ("alpha_ds", "dynamic_stabilization")):
        _rename_parameter(params, old, new)
    polymers = params.get("polymers")
    if isinstance(polymers, list):
        for i, polymer in enumerate(polymers):
            if isinstance(polymer, dict) and isinstance(polymer.get("blocks"), list):
                for j, block in enumerate(polymer["blocks"]):
                    if isinstance(block, dict):
                        _rename_parameter(block, "fraction", "monomer_fractions", f"polymers[{i}].blocks[{j}].")
    group = params.get("space_group")
    if isinstance(group, dict):
        _rename_parameter(group, "number", "hall_number", "space_group.")
    recording = params.get("recording")
    if isinstance(recording, dict):
        for old, new in (("recording_period", "checkpoint_interval"),
                         ("sf_computing_period", "sample_interval"),
                         ("sf_recording_period", "statistics_save_interval")):
            _rename_parameter(recording, old, new, "recording.")
    return params


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating)) or
            not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0)):
        qualifier = "positive " if positive else "non-negative " if nonnegative else ""
        raise ValidationError(f"'{name}' must be a finite {qualifier}number, got {value!r}")


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValidationError(f"'{name}' must be an integer >= {minimum}, got {value!r}")


def validate_system_params(params: Dict[str, Any]) -> None:
    """Check the parameters shared by SCFT, L-FTS and CL-FTS.

    Required: nx, lx, ds, chain_model, segment_lengths, chi_n, polymers.
    The first problem found raises ValidationError with a message naming the key.
    """
    params = normalize_params(params)
    for key in ("nx", "lx", "ds", "chain_model", "segment_lengths", "chi_n", "polymers"):
        if key not in params:
            raise ValidationError(f"Missing required parameter: '{key}'")

    nx, lx = params["nx"], params["lx"]
    if not isinstance(nx, (list, tuple, np.ndarray)) or not isinstance(lx, (list, tuple, np.ndarray)):
        raise ValidationError("'nx' and 'lx' must be lists")
    if len(nx) not in (1, 2, 3):
        raise ValidationError("'nx' must have 1, 2, or 3 dimensions")
    for n in nx:
        _integer(n, "nx")
    if len(lx) != len(nx):
        raise ValidationError(f"'lx' must have {len(nx)} elements, got {len(lx)}")
    for length in lx:
        _finite_number(length, "lx", positive=True)

    ds = params["ds"]
    _finite_number(ds, "ds", positive=True)
    if ds > 1:
        raise ValidationError(f"'ds' must be a number in (0, 1], got {ds}")
    if params["chain_model"] not in ("continuous", "discrete"):
        raise ValidationError(f"'chain_model' must be 'continuous' or 'discrete', got '{params['chain_model']}'")

    segment_lengths = params["segment_lengths"]
    if not isinstance(segment_lengths, dict) or len(segment_lengths) == 0:
        raise ValidationError("'segment_lengths' must be a non-empty dict {monomer_type: length}")
    for monomer, length in segment_lengths.items():
        if not isinstance(monomer, str) or not monomer:
            raise ValidationError("Monomer names must be non-empty strings")
        _finite_number(length, f"segment_lengths.{monomer}", positive=True)

    if not isinstance(params["chi_n"], dict):
        raise ValidationError("'chi_n' must be a dict, e.g. {'A,B': 15.0}")
    for pair, value in params["chi_n"].items():
        _finite_number(value, f"chi_n.{pair}")

    polymers = params["polymers"]
    if not isinstance(polymers, list) or len(polymers) == 0:
        raise ValidationError("'polymers' must be a non-empty list")
    total_volume_fraction = 0.0
    for i, polymer in enumerate(polymers):
        if not isinstance(polymer, dict) or "volume_fraction" not in polymer or "blocks" not in polymer:
            raise ValidationError(f"polymers[{i}] must be a dict with 'volume_fraction' and 'blocks'")
        vf = polymer["volume_fraction"]
        _finite_number(vf, f"polymers[{i}].volume_fraction", nonnegative=True)
        if vf > 1:
            raise ValidationError(f"polymers[{i}]['volume_fraction'] must be in [0, 1], got {vf}")
        total_volume_fraction += vf
        if not isinstance(polymer["blocks"], list) or len(polymer["blocks"]) == 0:
            raise ValidationError(f"polymers[{i}]['blocks'] must be a non-empty list")
        for j, block in enumerate(polymer["blocks"]):
            name = f"polymers[{i}].blocks[{j}]"
            if not isinstance(block, dict) or not isinstance(block.get("type"), str) or "length" not in block:
                raise ValidationError(f"'{name}' needs 'type' and 'length'")
            _finite_number(block["length"], name + ".length", positive=True)
            for vertex in ("v", "u"):
                if vertex in block:
                    _integer(block[vertex], name + "." + vertex, minimum=0)
            if "monomer_fractions" in block:
                fractions = block["monomer_fractions"]
                if not isinstance(fractions, dict) or not fractions:
                    raise ValidationError(f"'{name}.monomer_fractions' must be a non-empty dict")
                for monomer, fraction in fractions.items():
                    if monomer not in segment_lengths:
                        raise ValidationError(f"Unknown monomer '{monomer}' in '{name}.monomer_fractions'")
                    _finite_number(fraction, name + ".monomer_fractions." + monomer, nonnegative=True)
    if abs(total_volume_fraction - 1.0) > 1e-10:
        raise ValidationError(f"Sum of volume fractions must be 1.0, got {total_volume_fraction}")

    if "platform" in params and params["platform"] not in ("auto", "cuda", "cpu-fftw", "cpu-mkl"):
        raise ValidationError(f"'platform' must be 'auto', 'cuda', 'cpu-fftw' or 'cpu-mkl', got '{params['platform']}'")
    if params.get("zeta_n") is not None:
        _finite_number(params["zeta_n"], "zeta_n", nonnegative=True)


def parse_chi_n(
    chi_n_params: Dict[str, float],
    segment_lengths: Dict[str, float]
) -> Dict[str, float]:
    """Parse and validate Flory-Huggins interaction parameters.

    Parameters
    ----------
    chi_n_params : dict
        Dictionary of chi_n values with monomer pair keys (e.g., "A,B": 15.0).
    segment_lengths : dict
        Dictionary of segment lengths for validation.

    Returns
    -------
    dict
        Validated and normalized chi_n dictionary with sorted monomer pair keys.

    Raises
    ------
    ValidationError
        If monomer types are invalid, self-interactions are specified,
        or duplicate parameters are found.
    """
    chi_n = {}
    for monomer_pair_str, chin_value in chi_n_params.items():
        if not isinstance(monomer_pair_str, str):
            raise ValidationError("A chi_n key must name two monomer types, for example 'A,B'.")
        if ',' in monomer_pair_str:
            monomer_pair = [name.strip() for name in monomer_pair_str.split(',')]
            if len(monomer_pair) != 2:
                raise ValidationError(f"chi_n key '{monomer_pair_str}' must name exactly two monomer types.")
        else:
            # Keep legacy pair separators without splitting underscores inside names.
            pairs = [(a, b) for a in segment_lengths for b in segment_lengths
                     if any(monomer_pair_str == a + sep + b for sep in (' ', '_', '/'))]
            if len(pairs) != 1:
                raise ValidationError(f"chi_n key '{monomer_pair_str}' is invalid or ambiguous; use 'type1,type2'.")
            monomer_pair = list(pairs[0])
        if monomer_pair[0] not in segment_lengths:
            raise ValidationError(f"Monomer type '{monomer_pair[0]}' is not in 'segment_lengths'.")
        if monomer_pair[1] not in segment_lengths:
            raise ValidationError(f"Monomer type '{monomer_pair[1]}' is not in 'segment_lengths'.")
        if monomer_pair[0] == monomer_pair[1]:
            raise ValidationError(f"Do not add self interaction parameter, {monomer_pair_str}.")
        monomer_pair.sort()
        sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1]
        if sorted_monomer_pair in chi_n:
            raise ValidationError(f"There are duplicated χN ({sorted_monomer_pair}) parameters.")
        chi_n[sorted_monomer_pair] = chin_value
    return chi_n


def validate_scft_params(params: Dict[str, Any]) -> None:
    """Validate the SCFT options before constructing numerical objects."""
    params = normalize_params(params)
    validate_system_params(params)
    for key in ("optimizer", "relax_cell"):
        if key not in params:
            raise ValidationError(f"Missing required parameter for SCFT: '{key}'")
    figure_dir = params.get("polymer_figure_dir")
    if figure_dir is not None and not isinstance(figure_dir, (str, PathLike)):
        raise ValidationError(f"'polymer_figure_dir' must be a directory path or None, got {figure_dir!r}")
    if params.get("electrostatics") is not None:
        raise ValidationError("'electrostatics' is supported in CL-FTS only")
    if params.get("stress_interval", 1) != "adaptive":
        _integer(params.get("stress_interval", 1), "stress_interval")
    smearing = params.get("smearing")
    if isinstance(smearing, dict) and isinstance(smearing.get("a_int"), dict):
        raise ValidationError("Per-type 'smearing.a_int' is supported in CL-FTS only")


def validate_lfts_params(params: Dict[str, Any], *, complex_fields=False) -> None:
    """Validate the shared Langevin options before constructing numerical objects."""
    params = normalize_params(params)
    validate_system_params(params)
    langevin = params.get("langevin", None)
    if not isinstance(langevin, dict):
        raise ValidationError("Missing required parameter: 'langevin' (dict with 'nbar', 'dt', 'max_step')")
    for key in ("nbar", "dt"):
        _finite_number(langevin.get(key), "langevin." + key, positive=True)
    _integer(langevin.get("max_step"), "langevin.max_step", minimum=0)
    recording = params.get("recording")
    if not isinstance(recording, dict) or not isinstance(recording.get("dir"), (str, PathLike)):
        raise ValidationError("'recording' needs a 'dir' path")
    for key in ("checkpoint_interval", "sample_interval", "statistics_save_interval"):
        _integer(recording.get(key), "recording." + key)
    if "pressure" in recording:
        if not isinstance(recording["pressure"], (bool, np.bool_)):
            raise ValidationError("'recording.pressure' must be True or False")
        if recording["pressure"] and not complex_fields:
            raise ValidationError("'recording.pressure' is supported in CL-FTS only")
    _integer(params.get("verbose_level"), "verbose_level", minimum=0)
    if params["verbose_level"] > 2:
        raise ValidationError("'verbose_level' must be 0, 1 or 2")
    if not complex_fields:
        saddle = params.get("saddle")
        if not isinstance(saddle, dict):
            raise ValidationError("Missing required dict: 'saddle'")
        _integer(saddle.get("max_iter"), "saddle.max_iter")
        _finite_number(saddle.get("tolerance"), "saddle.tolerance", positive=True)
        _integer(saddle.get("max_retries", 5), "saddle.max_retries", minimum=0)
        if saddle.get("failure_policy", "retry") not in ("retry", "abort", "resample"):
            raise ValidationError("saddle.failure_policy must be 'retry', 'abort' or 'resample'")
        compressor = params.get("compressor")
        if not isinstance(compressor, dict) or compressor.get("name") not in ("am", "lr", "lram"):
            raise ValidationError("compressor.name must be 'am', 'lr' or 'lram'")
        if compressor["name"] in ("am", "lram"):
            _integer(compressor.get("max_hist"), "compressor.max_hist", minimum=0)
            for key in ("start_error", "mix_min", "mix_init"):
                _finite_number(compressor.get(key), "compressor." + key, nonnegative=True)
    if "dynamic_stabilization" in params:
        _finite_number(params["dynamic_stabilization"], "dynamic_stabilization", nonnegative=True)
    smearing = params.get("smearing")
    if smearing is not None:
        if not isinstance(smearing, dict):
            raise ValidationError("'smearing' must be a dict or None")
        kind = smearing.get("type", "gaussian")
        if kind not in ("gaussian", "sigmoidal"):
            raise ValidationError("Unknown smearing type")
        if kind == "gaussian" and isinstance(smearing.get("a_int"), dict):
            # One smearing length per monomer type (CL-FTS)
            if not complex_fields:
                raise ValidationError("Per-type 'smearing.a_int' is supported in CL-FTS only")
            if set(smearing["a_int"]) != set(params["segment_lengths"]):
                raise ValidationError("Per-type 'smearing.a_int' needs one length for each monomer type")
            if _has_random_copolymer(params):
                raise ValidationError("Per-type 'smearing.a_int' does not support random copolymers")
            for name, a_int in smearing["a_int"].items():
                _finite_number(a_int, f"smearing.a_int.{name}", nonnegative=True)
        else:
            keys = ("a_int",) if kind == "gaussian" else ("k_int", "dk_int")
            for key in keys:
                _finite_number(smearing.get(key, 5.0 if key == "dk_int" else None), "smearing." + key,
                               positive=key != "a_int", nonnegative=key == "a_int")
    if params.get("electrostatics") is not None:
        if not complex_fields:
            raise ValidationError("'electrostatics' is supported in CL-FTS only")
        validate_electrostatics(params)
    wtmd = params.get("wtmd")
    if wtmd is not None:
        if complex_fields:
            raise ValidationError("'wtmd' is supported in L-FTS only")
        if not isinstance(wtmd, dict):
            raise ValidationError("'wtmd' must be a dict or None")
        for key in ("update_freq", "recording_period"):
            _integer(wtmd.get(key, 1000 if key == "update_freq" else 100000), "wtmd." + key)
        for key, default in (("ell", 4), ("kc", 6.02), ("delta_t", 5.0), ("sigma_psi", 0.16), ("dpsi", 1e-3)):
            _finite_number(wtmd.get(key, default), "wtmd." + key, positive=True)
        lo, hi = wtmd.get("psi_min", 0.0), wtmd.get("psi_max", 10.0)
        _finite_number(lo, "wtmd.psi_min")
        _finite_number(hi, "wtmd.psi_max")
        if hi <= lo or (hi - lo) / wtmd.get("dpsi", 1e-3) < 2:
            raise ValidationError("WTMD range must contain at least two bins")


def _has_random_copolymer(params):
    return any(isinstance(block, dict) and "monomer_fractions" in block
               for polymer in params.get("polymers", []) for block in polymer.get("blocks", []))


def validate_electrostatics(params: Dict[str, Any]) -> None:
    """Charges, coupling and electroneutrality of the CL-FTS 'electrostatics' input."""
    electrostatics = params["electrostatics"]
    if not isinstance(electrostatics, dict):
        raise ValidationError("'electrostatics' must be a dict or None")
    unknown = set(electrostatics) - {"charges", "bjerrum_length", "coupling", "psi_dt_scaling", "screening_length"}
    if unknown:
        raise ValidationError(f"'electrostatics' has unknown keys: {sorted(unknown)}")
    charges = electrostatics.get("charges")
    if not isinstance(charges, dict) or not charges:
        raise ValidationError("'electrostatics.charges' must be a dict {monomer type: valence}")
    unknown = set(charges) - set(params["segment_lengths"])
    if unknown:
        raise ValidationError(f"'electrostatics.charges' has unknown monomer types: {sorted(unknown)}")
    for name, valence in charges.items():
        _finite_number(valence, f"electrostatics.charges.{name}")
    if ("bjerrum_length" in electrostatics) == ("coupling" in electrostatics):
        raise ValidationError("Give exactly one of 'electrostatics.bjerrum_length' (l_B/R0) and "
                              "'electrostatics.coupling' (E)")
    for key in ("bjerrum_length", "coupling"):
        if key in electrostatics:
            _finite_number(electrostatics[key], "electrostatics." + key, nonnegative=True)
    # E = 4π (l_B/R₀) N² N̄^(1/2) needs the number of segments N. Discrete chains have N = 1/ds;
    # ds of continuous chains is a contour step, so E would change under contour refinement
    if "bjerrum_length" in electrostatics:
        if str(params["chain_model"]).lower() != "discrete":
            raise ValidationError("'electrostatics.bjerrum_length' needs discrete chains (N = 1/ds); "
                                  "give 'coupling' (E = 4 pi l_B rho0 N R0^2) for continuous chains")
        n_segments = 1.0 / params["ds"]
        if abs(n_segments - round(n_segments)) > 1e-9:
            raise ValidationError("'electrostatics.bjerrum_length' needs an integer 1/ds; give 'coupling' instead")
    _finite_number(electrostatics.get("psi_dt_scaling", 1.0), "electrostatics.psi_dt_scaling", positive=True)
    if electrostatics.get("screening_length") is not None:
        _finite_number(electrostatics["screening_length"], "electrostatics.screening_length", positive=True)
    # Point charges have a grid-dependent self-energy; the charges are smeared
    smearing = params.get("smearing")
    if smearing is None:
        raise ValidationError("'electrostatics' needs 'smearing' (the smeared charge distribution)")
    if smearing.get("type", "gaussian") == "gaussian":
        a_int = smearing.get("a_int")
        lengths = a_int if isinstance(a_int, dict) else dict.fromkeys(charges, a_int)
        point_charges = sorted(name for name, valence in charges.items() if valence != 0 and not lengths[name] > 0)
        if point_charges:
            raise ValidationError(f"'electrostatics' needs a positive smearing length 'a_int' for the charged "
                                  f"types {point_charges}; a_int = 0 leaves point charges")
    if any(str(b).lower() != "periodic" for b in params.get("bc") or []):
        raise ValidationError("'electrostatics' supports periodic boxes only")
    if _has_random_copolymer(params):
        raise ValidationError("'electrostatics' does not support random copolymers")
    # Electroneutrality: Σ_i z_i φ_i = 0 with the overall segment fraction φ_i of each type. An implicit
    # electrolyte (screening_length) screens a net charge, so a screened system need not be neutral
    if electrostatics.get("screening_length") is not None:
        return
    total = 0.0
    for polymer in params["polymers"]:
        length = sum(block["length"] for block in polymer["blocks"])
        for block in polymer["blocks"]:
            total += charges.get(block["type"], 0.0) * polymer["volume_fraction"] * block["length"] / length
    if abs(total) > 1e-10:
        raise ValidationError(f"The system is not electroneutral: sum of z_i phi_i = {total:.3e}")
