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


def polymer_block_lengths(params: Dict[str, Any], polymer: Dict[str, Any]) -> tuple:
    """Effective contour lengths in input block order, without changing input metadata.

    Continuous chains retain their lengths. Discrete chains use the same native
    acceptance and snapping to integer multiples of ds as Molecules.add_polymer.
    """
    if str(params["chain_model"]).lower() != "discrete":
        return tuple(block["length"] for block in polymer["blocks"])
    from . import _core

    lengths = []
    for j, block in enumerate(polymer["blocks"]):
        try:
            lengths.append(_core._discrete_contour_length(block["length"], params["ds"]))
        except (RuntimeError, TypeError, ValueError) as exc:
            index = next((i for i, candidate in enumerate(params.get("polymers", ()))
                          if candidate is polymer), None)
            path = f"polymers[{index}]." if index is not None else ""
            raise ValidationError(f"'{path}blocks[{j}].length': {exc}") from exc
    return tuple(lengths)


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
        if not isinstance(polymer, dict) or "blocks" not in polymer or \
                ("volume_fraction" in polymer) == ("chemical_potential" in polymer):
            raise ValidationError(f"polymers[{i}] must be a dict with 'volume_fraction' and 'blocks' (or, in "
                                  "CL-FTS, 'chemical_potential' in place of 'volume_fraction')")
        if "chemical_potential" in polymer:
            # Grand canonical: the chemical potential fixes the number of chains (CL-FTS checks the rest)
            _finite_number(polymer["chemical_potential"], f"polymers[{i}].chemical_potential")
        else:
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
    if any("chemical_potential" in polymer for polymer in polymers):
        if total_volume_fraction > 1.0 + 1e-10:
            raise ValidationError(f"Sum of the volume fractions of the canonical polymers must not exceed 1.0, "
                                  f"got {total_volume_fraction}")
    elif abs(total_volume_fraction - 1.0) > 1e-10:
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


def _validate_smearing(params: Dict[str, Any], complex_fields: bool) -> None:
    """The `smearing` option; per-type Gaussian lengths need complex fields (CL-FTS)."""
    smearing = params.get("smearing")
    if smearing is None:
        return
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


def validate_scft_params(params: Dict[str, Any]) -> None:
    """Validate the SCFT options before constructing numerical objects."""
    params = normalize_params(params)
    validate_system_params(params)
    group = params.get("space_group")
    if group is not None:
        if not isinstance(group, dict) or not isinstance(group.get("symbol"), str) or not group["symbol"]:
            raise ValidationError("'space_group' needs a dict with the Hermann-Mauguin 'symbol' (and optionally "
                                  "'hall_number'), e.g. {'symbol': 'Ia-3d'}")
        hall = group.get("hall_number")
        if hall is not None and (isinstance(hall, (bool, np.bool_)) or not isinstance(hall, (int, np.integer))
                                 or not 1 <= hall <= 530):
            raise ValidationError(f"'space_group.hall_number' must be an integer from 1 to 530, got {hall!r}")
    for key in ("optimizer", "relax_cell"):
        if key not in params:
            raise ValidationError(f"Missing required parameter for SCFT: '{key}'")
    figure_dir = params.get("polymer_figure_dir")
    if figure_dir is not None and not isinstance(figure_dir, (str, PathLike)):
        raise ValidationError(f"'polymer_figure_dir' must be a directory path or None, got {figure_dir!r}")
    electrostatics = params.get("electrostatics")
    if electrostatics is not None:
        # The mean field of the Coulomb interaction (ψ from the Poisson equation); SCFT has no N̄
        if isinstance(electrostatics, dict) and "bjerrum_length" in electrostatics:
            raise ValidationError("SCFT has no N̄ for 'electrostatics.bjerrum_length': give 'coupling' "
                                  "(E = 4 pi l_B rho0 N R0^2)")
        if isinstance(electrostatics, dict) and "psi_dt_scaling" in electrostatics:
            raise ValidationError("'electrostatics.psi_dt_scaling' is a CL-FTS option")
    # Before validate_electrostatics, which reads the smearing length of the charges
    _validate_smearing(params, complex_fields=False)
    if electrostatics is not None:
        validate_electrostatics(params)
        if not electrostatics["coupling"] > 0:
            raise ValidationError("SCFT needs a positive 'electrostatics.coupling'; leave out 'electrostatics' "
                                  "for uncharged chains")
    _reject_grand_canonical(params, "SCFT")
    _reject_mask(params, "SCFT")
    if params.get("stress_interval", 1) != "adaptive":
        _integer(params.get("stress_interval", 1), "stress_interval")
    optimizer = params["optimizer"]
    if not isinstance(optimizer, dict) or optimizer.get("name") != "am":
        raise ValidationError("optimizer.name must be 'am' (Anderson mixing)")
    _integer(optimizer.get("max_hist"), "optimizer.max_hist", minimum=0)
    for key in ("start_error", "mix_min", "mix_init"):
        _finite_number(optimizer.get(key), "optimizer." + key, nonnegative=True)
    _finite_number(params.get("tolerance", 1e-8), "tolerance", positive=True)
    _finite_number(params.get("stress_scale", 1), "stress_scale", positive=True)
    # Preserve the ordering of option checks, while rejecting invalid discrete
    # lengths before any solver is constructed (including uncharged systems).
    for polymer in params["polymers"]:
        polymer_block_lengths(params, polymer)


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
    _reject_mask(params, "CL-FTS" if complex_fields else "L-FTS")
    if not complex_fields:
        _reject_grand_canonical(params, "L-FTS")
    else:
        for i, polymer in enumerate(params["polymers"]):
            if "chemical_potential" in polymer and (polymer.get("initial_conditions") or _has_random_copolymer(
                    {"polymers": [polymer]})):
                raise ValidationError(f"polymers[{i}]: grand-canonical polymers cannot be grafted or random copolymers")
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
    # The propagators of grafted chains start from q_init, which only SCFT.run takes
    for i, polymer in enumerate(params["polymers"]):
        if polymer.get("initial_conditions"):
            raise ValidationError(f"polymers[{i}]: L-FTS and CL-FTS need free chains; grafted chains "
                                  "(initial_conditions) need q_init, which only SCFT takes")
    if "dynamic_stabilization" in params:
        _finite_number(params["dynamic_stabilization"], "dynamic_stabilization", nonnegative=True)
    _validate_smearing(params, complex_fields)
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
        if wtmd.get("ell", 4) < 1:
            raise ValidationError("wtmd.ell must be at least 1")
        lo, hi = wtmd.get("psi_min", 0.0), wtmd.get("psi_max", 10.0)
        _finite_number(lo, "wtmd.psi_min")
        _finite_number(hi, "wtmd.psi_max")
        if hi <= lo or (hi - lo) / wtmd.get("dpsi", 1e-3) < 2:
            raise ValidationError("WTMD range must contain at least two bins")
    for polymer in params["polymers"]:
        polymer_block_lengths(params, polymer)


def _has_random_copolymer(params):
    return any(isinstance(block, dict) and "monomer_fractions" in block
               for polymer in params.get("polymers", []) for block in polymer.get("blocks", []))


def _reject_grand_canonical(params: Dict[str, Any], method: str) -> None:
    if any("chemical_potential" in polymer for polymer in params["polymers"]):
        raise ValidationError(f"Grand-canonical polymers ('chemical_potential') are supported in CL-FTS only, "
                              f"not in {method}")


def _reject_mask(params: Dict[str, Any], method: str) -> None:
    # Impenetrable regions exist in the propagator solver only; the simulation loops do not mask their fields
    if params.get("mask") is not None:
        raise ValidationError(f"{method} does not take a 'mask': impenetrable regions are available through "
                              "PropagatorSolver(mask=...) only")


def validate_electrostatics(params: Dict[str, Any]) -> None:
    """Charges, coupling and electroneutrality of the CL-FTS 'electrostatics' input."""
    electrostatics = params["electrostatics"]
    if not isinstance(electrostatics, dict):
        raise ValidationError("'electrostatics' must be a dict or None")
    unknown = set(electrostatics) - {"charges", "bjerrum_length", "coupling", "psi_dt_scaling", "screening_length",
                                     "ions", "counterion", "n_segments", "fixed_charge"}
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
    # E = 4π (l_B/R₀) N² N̄^(1/2) and the point ions need the number of segments N of the reference
    # chain. Discrete chains have N = 1/ds; ds of continuous chains is a contour step, so N is given
    ions = electrostatics.get("ions") or {}
    if "n_segments" in electrostatics:
        _finite_number(electrostatics["n_segments"], "electrostatics.n_segments", positive=True)
    if str(params["chain_model"]).lower() == "discrete":
        n_segments = 1.0 / params["ds"]
        if "n_segments" in electrostatics and abs(electrostatics["n_segments"] - n_segments) > 1e-9 * n_segments:
            raise ValidationError(f"'electrostatics.n_segments' of discrete chains is 1/ds = {n_segments:g}")
        if ("bjerrum_length" in electrostatics or ions) and abs(n_segments - round(n_segments)) > 1e-9:
            raise ValidationError("'electrostatics.bjerrum_length' and 'ions' need an integer 1/ds; give 'coupling' "
                                  "and no point ions instead")
    elif "n_segments" not in electrostatics:
        if "bjerrum_length" in electrostatics:
            raise ValidationError("'electrostatics.bjerrum_length' needs discrete chains (N = 1/ds) or the number of "
                                  "segments N of the reference chain, 'electrostatics.n_segments'; or give 'coupling' "
                                  "(E = 4 pi l_B rho0 N R0^2)")
        if ions:
            raise ValidationError("'electrostatics.ions' need the number of segments N of the reference chain: give "
                                  "'electrostatics.n_segments' for continuous chains")
    # Point ions: no volume, no chi, only their smeared charge
    if not isinstance(ions, dict):
        raise ValidationError("'electrostatics.ions' must be a dict "
                              "{ion name: {valence, concentration, smearing_length}}")
    for name, ion in ions.items():
        if not isinstance(name, str) or name in params["segment_lengths"]:
            raise ValidationError(f"Ion name {name!r} must be a string that is not a monomer type")
        keys = set(ion) if isinstance(ion, dict) else set()
        allowed = {"valence", "concentration", "chemical_potential", "smearing_length"}
        if not isinstance(ion, dict) or keys - allowed or not {"valence", "smearing_length"} <= keys or \
                ("concentration" in ion) == ("chemical_potential" in ion):
            raise ValidationError(f"'electrostatics.ions.{name}' needs 'valence', 'smearing_length' and one of "
                                  "'concentration' and 'chemical_potential' (grand canonical)")
        _finite_number(ion["valence"], f"electrostatics.ions.{name}.valence")
        if ion["valence"] == 0:
            raise ValidationError(f"'electrostatics.ions.{name}.valence' must not be zero")
        if "concentration" in ion:
            _finite_number(ion["concentration"], f"electrostatics.ions.{name}.concentration", nonnegative=True)
        else:
            _finite_number(ion["chemical_potential"], f"electrostatics.ions.{name}.chemical_potential")
        _finite_number(ion["smearing_length"], f"electrostatics.ions.{name}.smearing_length", positive=True)
    counterion = electrostatics.get("counterion", "")
    if counterion not in ("", None) and (counterion not in ions or "concentration" not in ions[counterion]):
        raise ValidationError(f"'electrostatics.counterion' {counterion!r} is not one of the canonical 'ions' "
                              "(with a concentration)")
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
    # Periodic boxes, or closed boxes with walls on every side (ψ then has no normal derivative there)
    sides = [str(b).lower() == "periodic" for b in params.get("bc") or []]
    if any(sides) and not all(sides):
        raise ValidationError("'electrostatics' needs a periodic box or walls on every side, not both")
    if electrostatics.get("fixed_charge") is not None:
        fixed = np.asarray(electrostatics["fixed_charge"])
        if fixed.size != int(np.prod(params["nx"])) or not np.all(np.isfinite(fixed)) or np.iscomplexobj(fixed):
            raise ValidationError("'electrostatics.fixed_charge' needs one real, finite charge density per grid point")
    if _has_random_copolymer(params):
        raise ValidationError("'electrostatics' does not support random copolymers")
    # Electroneutrality: Σ_i z_i φ_i + Σ_s z_s c_s = 0 with the overall segment fraction φ_i of each type
    # and the point ions, counterions included. An implicit electrolyte (screening_length) screens a net
    # charge, so a screened system need not be neutral; with charged grand-canonical species the uniform
    # mode of ψ neutralizes the system
    concentrations = point_ion_concentrations(params)
    if electrostatics.get("screening_length") is not None:
        return
    if open_charges(params):
        # The Donnan potential neutralizes the canonical charge only with grand-canonical
        # species of the opposite sign (of both signs for a neutral canonical part)
        canonical = net_polymer_charge(params) + sum(ions[name]["valence"] * c for name, c in concentrations.items())
        signs = {np.sign(ion["valence"]) for ion in ions.values() if "chemical_potential" in ion}
        for polymer in params["polymers"]:
            if "chemical_potential" in polymer:
                charge = _chain_charge(params, polymer, charges)
                if charge != 0.0:
                    signs.add(np.sign(charge))
        needed = {1.0, -1.0} if abs(canonical) <= 1e-14 else {-np.sign(canonical)}
        if not needed <= signs:
            raise ValidationError(f"The grand-canonical species cannot neutralize the system (canonical net charge "
                                  f"{canonical:.3e}): without screening they need charges of "
                                  + ("both signs" if len(needed) == 2 else "the opposite sign"))
        return
    total = net_polymer_charge(params) + sum(ions[name]["valence"] * c for name, c in concentrations.items())
    if abs(total) > 1e-10:
        raise ValidationError(f"The system is not electroneutral: sum of z_i phi_i = {total:.3e}"
                              + ("; add point ions ('electrostatics.ions') as counterions" if not ions else ""))


def _chain_charge(params: Dict[str, Any], polymer: Dict[str, Any], charges: Dict[str, float]) -> float:
    """Net charge of a chain per segment of the reference chain, Σ_blocks z_i α_i (zero within round-off)."""
    lengths = polymer_block_lengths(params, polymer)
    charge = sum(charges.get(block["type"], 0.0) * length for block, length in zip(polymer["blocks"], lengths))
    scale = sum(abs(charges.get(block["type"], 0.0)) * length for block, length in zip(polymer["blocks"], lengths))
    return 0.0 if abs(charge) <= 1e-12 * scale else charge


def open_charges(params: Dict[str, Any]) -> bool:
    """True if a grand-canonical species (polymer or point ion) carries a net charge."""
    electrostatics = params["electrostatics"]
    charges = electrostatics["charges"]
    if any("chemical_potential" in ion for ion in (electrostatics.get("ions") or {}).values()):
        return True
    # The uniform mode of ψ does not change the weight of a chain without net charge
    return any("chemical_potential" in polymer and _chain_charge(params, polymer, charges) != 0.0
               for polymer in params["polymers"])


def net_polymer_charge(params: Dict[str, Any]) -> float:
    """Σ_i z_i φ_i of the canonical polymers and the mean fixed charge density, the net charge per
    segment of the reference density without the point ions."""
    charges = params["electrostatics"]["charges"]
    fixed = params["electrostatics"].get("fixed_charge")
    total = 0.0 if fixed is None else float(np.mean(fixed))
    for polymer in params["polymers"]:
        if "chemical_potential" in polymer:
            continue
        lengths = polymer_block_lengths(params, polymer)
        alpha = sum(lengths)
        for block, length in zip(polymer["blocks"], lengths):
            total += charges.get(block["type"], 0.0) * polymer["volume_fraction"] * length / alpha
    return total


def point_ion_concentrations(params: Dict[str, Any]) -> Dict[str, float]:
    """Concentration of each canonical point ion (per segment of the reference density), counterions included.

    The given concentrations are added salt; grand-canonical ions (chemical_potential) have none.
    The counterion species, 'electrostatics.counterion' or the only canonical ion species whose
    valence has the opposite sign of the net charge, receives the concentration that neutralizes
    the canonical polymers and the given ions; 'counterion': None, or no such species, adds none
    (the electroneutrality check of validate_electrostatics then applies).
    """
    electrostatics = params["electrostatics"]
    ions = {name: ion for name, ion in (electrostatics.get("ions") or {}).items() if "concentration" in ion}
    concentrations = {name: float(ion["concentration"]) for name, ion in ions.items()}
    total = net_polymer_charge(params) + sum(ions[name]["valence"] * c for name, c in concentrations.items())
    counterion = electrostatics.get("counterion", "")
    if not ions or counterion is None or abs(total) <= 1e-14:
        return concentrations
    if counterion == "":
        candidates = [name for name, ion in ions.items() if ion["valence"] * total < 0]
        if not candidates:
            return concentrations
        if len(candidates) > 1:
            raise ValidationError(f"The net charge {total:.3e} has {len(candidates)} ion species of opposite sign; "
                                  "name the counterion with 'electrostatics.counterion' (None: no counterions)")
        counterion = candidates[0]
    added = -total / ions[counterion]["valence"]
    if added < 0:
        raise ValidationError(f"The counterion {counterion!r} has the sign of the net charge {total:.3e}")
    concentrations[counterion] += added
    return concentrations


def validate_gibbs_params(params: Dict[str, Any]) -> None:
    """The `gibbs_ensemble` section of a Gibbs-ensemble CL-FTS input (gibbs_ensemble.py)."""
    gibbs = params.get("gibbs_ensemble")
    if not isinstance(gibbs, dict):
        raise ValidationError("Missing required dict: 'gibbs_ensemble'")
    unknown = set(gibbs) - {"boxes", "cycles", "cycle_steps", "exchange_rate", "skip_steps"}
    if unknown:
        raise ValidationError(f"'gibbs_ensemble' has unknown keys: {sorted(unknown)}")
    _integer(gibbs.get("cycles"), "gibbs_ensemble.cycles", minimum=1)
    _integer(gibbs.get("cycle_steps"), "gibbs_ensemble.cycle_steps", minimum=1)
    _finite_number(gibbs.get("exchange_rate", 0.2), "gibbs_ensemble.exchange_rate", positive=True)
    _integer(gibbs.get("skip_steps", 0), "gibbs_ensemble.skip_steps", minimum=0)
    if gibbs.get("skip_steps", 0) >= gibbs["cycle_steps"]:
        raise ValidationError("'gibbs_ensemble.skip_steps' must be smaller than 'cycle_steps'")
    recording = params.get("recording")
    if not isinstance(recording, dict) or "dir" not in recording:
        raise ValidationError("The Gibbs ensemble needs 'recording.dir' for its boxes and history")
    _integer(recording.get("sample_interval", 1), "recording.sample_interval")
    if gibbs["cycle_steps"] % recording.get("sample_interval", 1) != 0:
        raise ValidationError("'gibbs_ensemble.cycle_steps' must be a multiple of 'recording.sample_interval'")
    if any(str(b).lower() != "periodic" for b in params.get("bc") or []):
        raise ValidationError("The Gibbs ensemble needs periodic boxes (the osmotic pressure)")
    polymers = params.get("polymers") or []
    for i, polymer in enumerate(polymers):
        if "chemical_potential" in polymer or polymer.get("initial_conditions"):
            raise ValidationError(f"polymers[{i}]: the Gibbs ensemble exchanges free canonical chains; grand-canonical "
                                  "and grafted polymers are not supported")
    electrostatics = params.get("electrostatics") or {}
    if electrostatics.get("fixed_charge") is not None:
        raise ValidationError("The Gibbs ensemble does not support 'electrostatics.fixed_charge' "
                              "(its boxes change size)")
    canonical_ions = [name for name, ion in (electrostatics.get("ions") or {}).items()
                      if isinstance(ion, dict) and "concentration" in ion]
    if canonical_ions and str(params.get("chain_model")).lower() != "discrete" and "n_segments" not in electrostatics:
        raise ValidationError("'electrostatics.ions' need the number of segments N of the reference chain: give "
                              "'electrostatics.n_segments' (continuous chains)")
    boxes = gibbs.get("boxes")
    if not isinstance(boxes, list) or len(boxes) != 2:
        raise ValidationError("'gibbs_ensemble.boxes' must be a list of two dicts")
    for b, box in enumerate(boxes):
        name = f"gibbs_ensemble.boxes[{b}]"
        if not isinstance(box, dict) or set(box) - {"lx", "volume_fractions", "ion_concentrations"} or \
                not {"lx", "volume_fractions"} <= set(box):
            raise ValidationError(f"'{name}' must be a dict with 'lx', 'volume_fractions' and optionally "
                                  "'ion_concentrations'")
        if not isinstance(box["lx"], (list, tuple)) or len(box["lx"]) != len(params["nx"]):
            raise ValidationError(f"'{name}.lx' needs one length per dimension of 'nx'")
        for length in box["lx"]:
            _finite_number(length, name + ".lx", positive=True)
        fractions = box["volume_fractions"]
        if not isinstance(fractions, (list, tuple)) or len(fractions) != len(polymers):
            raise ValidationError(f"'{name}.volume_fractions' needs one value per polymer")
        for value in fractions:
            _finite_number(value, name + ".volume_fractions", nonnegative=True)
        if abs(sum(fractions) - 1.0) > 1e-10:
            raise ValidationError(f"'{name}.volume_fractions' must sum to 1, got {sum(fractions)}")
        concentrations = box.get("ion_concentrations")
        if concentrations is not None:
            if not isinstance(concentrations, dict) or set(concentrations) != set(canonical_ions):
                raise ValidationError(f"'{name}.ion_concentrations' needs the concentration of every canonical point "
                                      f"ion {canonical_ions}")
            for ion, value in concentrations.items():
                _finite_number(value, f"{name}.ion_concentrations.{ion}", nonnegative=True)
