"""Input aliases and exact continuation across the parameter-name migration."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import loadmat

from polymerfts import SCFT, LFTS, CLFTS, PlatformSelector, ValidationError
from polymerfts.utils import atomic_savemat, input_hash, settings_json
from polymerfts.validation import normalize_params, validate_lfts_params
from TestFtsReliability import parameters, fields, expect_error
from TestParams import base_params


def legacy_params(params):
    """Construct the earlier public input format independently of normalization."""
    result = copy.deepcopy(params)
    for current, old in (("polymers", "distinct_polymers"),
                         ("relax_cell", "box_is_altering"),
                         ("stress_scale", "scale_stress")):
        if current in result:
            result[old] = result.pop(current)
    for polymer in result.get("distinct_polymers", []):
        for block in polymer["blocks"]:
            if "monomer_fractions" in block:
                block["fraction"] = block.pop("monomer_fractions")
    if "space_group" in result:
        result["space_group"]["number"] = result["space_group"].pop("hall_number")
    recording = result.get("recording", {})
    for current, old in (("checkpoint_interval", "recording_period"),
                         ("sample_interval", "sf_computing_period"),
                         ("statistics_save_interval", "sf_recording_period")):
        if current in recording:
            recording[old] = recording.pop(current)
    return result


def read(path):
    return {k: v for k, v in loadmat(path, squeeze_me=True).items() if not k.startswith("__")}


def compare_output(first, second):
    for filename in ("fields_000004.mat", "dH_000002.mat", "structure_function_000002.mat"):
        a, b = read(first / filename), read(second / filename)
        for key in a:
            if key.startswith(("w_", "phi_", "record_", "normal_noise_prev", "H_history",
                               "dH_history", "structure_function")) or key == "sample_count":
                np.testing.assert_array_equal(a[key], b[key], err_msg=f"{filename}: {key}")
        if "rng_state_json" in a:
            assert str(a["rng_state_json"]) == str(b["rng_state_json"])


def check_aliases(root):
    params = parameters(root / "aliases")
    params.update(relax_cell=False, stress_scale=0.5,
                  space_group={"symbol": "Im-3m", "hall_number": 529},
                  wtmd={"recording_period": 100})
    params["polymers"][0]["blocks"] = [
        {"type": "R", "length": 1.0, "monomer_fractions": {"A": 0.3, "B": 0.7}}]
    old = legacy_params(params)
    before = copy.deepcopy(old)
    assert normalize_params(old) == params
    assert old == before
    assert normalize_params(params) == params
    validate_lfts_params(old)
    assert normalize_params(old)["wtmd"]["recording_period"] == 100
    # Every alias is rejected when both spellings are supplied; values need
    # not be valid because ambiguity is caught before numerical construction.
    pairs = [({old: 1, new: 1}, "") for old, new in (
        ("distinct_polymers", "polymers"), ("box_is_altering", "relax_cell"),
        ("scale_stress", "stress_scale"), ("alpha_ds", "dynamic_stabilization"))]
    pairs.extend([({"recording": {old: 1, new: 1}}, "recording.") for old, new in (
        ("recording_period", "checkpoint_interval"), ("sf_computing_period", "sample_interval"),
        ("sf_recording_period", "statistics_save_interval"))])
    pairs.extend([
        ({"space_group": {"number": 529, "hall_number": 529}}, "space_group."),
        ({"polymers": [{"blocks": [{"fraction": {}, "monomer_fractions": {}}]}]}, "polymers[0].blocks[0]."),
    ])
    for ambiguous, path in pairs:
        try:
            normalize_params(ambiguous)
            raise AssertionError("Duplicate alias was accepted")
        except ValidationError as error:
            assert path in str(error)


def check_scft(root):
    params = base_params()
    params.update(relax_cell=True, stress_scale=0.5, max_iter=2,
                  space_group={"symbol": "Im-3m", "hall_number": 529})
    old = legacy_params(params)
    before = copy.deepcopy(old)
    canonical, legacy = SCFT(params), SCFT(old)
    assert legacy.params == canonical.params and old == before
    assert legacy.relax_cell and legacy.stress_scale == 0.5
    assert legacy.sg.get_hall_number() == 529
    rng = np.random.default_rng(51)
    initial = {m: rng.normal(0, 5, params["nx"]) for m in ("A", "B")}
    for calc in (canonical, legacy):
        calc.run(initial)
        calc.prop_solver.check_total_partition()
    np.testing.assert_array_equal(legacy.w, canonical.w)
    np.testing.assert_array_equal(legacy.prop_solver.get_lx(), canonical.prop_solver.get_lx())
    for monomer in canonical.phi:
        np.testing.assert_array_equal(legacy.phi[monomer], canonical.phi[monomer])

    # A keyword override must win even when its spelling differs from the file.
    for cls in (SCFT, LFTS, CLFTS):
        config = parameters(root / (cls.__name__ + "config"))
        config.update(relax_cell=True, stress_scale=0.5, optimizer=params["optimizer"])
        path = root / (cls.__name__ + ".json")
        path.write_text(json.dumps(legacy_params(config)))
        assert cls.from_file(path, relax_cell=False).params["relax_cell"] is False
        path.write_text(json.dumps(config))
        assert cls.from_file(path, box_is_altering=False).params["relax_cell"] is False


def check_checkpoint(root, cls):
    output = root / (cls.__name__ + "full")
    params = parameters(output)
    params["polymers"][0]["volume_fraction"] = 0.6
    params["polymers"].append({"volume_fraction": 0.4, "blocks": [
        {"type": "R", "length": 1.0, "monomer_fractions": {"A": 0.3, "B": 0.7}}]})
    full = cls(params, random_seed=11)
    assert full.run(fields())["status"] == "completed"
    full.prop_solver.check_total_partition()
    old_params = legacy_params(params)
    old_params["recording"]["dir"] = str(root / (cls.__name__ + "old-input"))
    legacy = cls(old_params, random_seed=11)
    assert legacy.run(fields())["status"] == "completed"
    assert legacy.random_bg.state == full.random_bg.state
    compare_output(output, Path(legacy.recording["dir"]))

    original = read(output / "fields_000001.mat")
    settings = json.loads(str(original["checkpoint_settings_json"]))
    # Earlier checkpoint settings already used 'polymers' at the top level,
    # but their random-block keys and flat recording keys used earlier names.
    for polymer in settings["polymers"]:
        for block in polymer["blocks"]:
            if "monomer_fractions" in block:
                block["fraction"] = block.pop("monomer_fractions")
    settings["sf_computing_period"] = settings.pop("sample_interval")
    settings["sf_recording_period"] = settings.pop("statistics_save_interval")
    for version in (1, 2):
        data = copy.deepcopy(original)
        stored = copy.deepcopy(settings)
        if version == 1:
            stored.pop("extensions")
            data.pop("extension_state_json")
        data.update(checkpoint_format_version=version, checkpoint_settings_json=settings_json(stored),
                    checkpoint_settings_hash=input_hash(stored))
        checkpoint = root / (cls.__name__ + f"-version{version}.mat")
        atomic_savemat(checkpoint, data)
        resumed_params = copy.deepcopy(params)
        resumed_params["recording"]["dir"] = str(root / (cls.__name__ + f"resumed{version}"))
        resumed = cls(resumed_params, random_seed=99)
        assert resumed.continue_run(checkpoint)["status"] == "completed"
        assert resumed.random_bg.state == full.random_bg.state
        compare_output(output, Path(resumed.recording["dir"]))

    candidate = cls(params, random_seed=99)
    before_rng = copy.deepcopy(candidate.random_bg.state)
    for rehash in (False, True):
        changed = copy.deepcopy(settings)
        changed["sf_computing_period"] = 2
        bad = copy.deepcopy(original)
        bad["checkpoint_settings_json"] = settings_json(changed)
        if rehash:
            bad["checkpoint_settings_hash"] = input_hash(changed)
        path = root / (cls.__name__ + f"-invalid-{rehash}.mat")
        atomic_savemat(path, bad)
        expect_error(lambda: candidate.continue_run(path), ValidationError)
        assert candidate.random_bg.state == before_rng


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as directory:
        root, previous = Path(directory), Path.cwd()
        os.chdir(root)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                check_aliases(root)
                check_scft(root)
                for cls in (LFTS, CLFTS):
                    check_checkpoint(root, cls)
        finally:
            os.chdir(previous)
    print("Test passed, canonical parameter names and legacy input/checkpoint compatibility")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
