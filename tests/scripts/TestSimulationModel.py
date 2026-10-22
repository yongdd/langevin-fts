"""Structured editor exports: mixtures, monomer pairs and cell geometry, without a solver."""

import json
from pathlib import Path
import runpy
import threading
from urllib.request import Request, urlopen

import numpy as np
import pytest
from scipy.io import savemat


CHECKS = runpy.run_path(str(Path(__file__).with_name("TestSimulationScript.py")))
BUILDER = CHECKS["BUILDER"]


def values():
    result = BUILDER["preset_settings"](BUILDER["PRESETS"][0])
    for key in ("nx", "lx", "blocks", "b_a", "b_b", "chi_n"):
        result.pop(key)
    result["model"] = {
        "box": {"dimension": 3, "shape": "orthorhombic", "nx": ["8", "10", "12"],
                "lx": ["3", "4", "5"], "angles": ["90", "90", "90"]},
        "monomers": [{"name": "A", "segment_length": "1"}, {"name": "B", "segment_length": "1.2"},
                     {"name": "C", "segment_length": "0.8"}],
        "polymers": [{"volume_fraction": "0.6", "blocks": [{"type": "A", "length": "0.3"}, {"type": "B", "length": "0.7"}]},
                     {"volume_fraction": "0.4", "blocks": [{"type": "C", "length": "1"}]}],
        "interactions": [{"a": "A", "b": "B", "chi_n": "20"}, {"a": "A", "b": "C", "chi_n": "15"}],
    }
    result.update(initial="random", amplitude="5")
    return result


@pytest.mark.parametrize("mode", ("SCFT", "L-FTS"))
def test_mixture_export(mode, tmp_path, monkeypatch):
    form = values()
    form["mode"] = mode
    calls = CHECKS["execute_export"](form, tmp_path, monkeypatch)
    params = calls["params"]
    CHECKS["VALIDATION"]["validate_system_params" if mode == "SCFT" else "validate_lfts_params"](params)
    assert params["nx"] == [8, 10, 12]
    assert params["segment_lengths"] == {"A": 1, "B": 1.2, "C": .8}
    assert [p["volume_fraction"] for p in params["polymers"]] == [.6, .4]
    assert params["polymers"][1]["blocks"] == [{"type": "C", "length": 1}]
    assert params["chi_n"] == {"A,B": 20, "A,C": 15, "B,C": 0}
    assert set(calls["fields"]) == {"A", "B", "C"}
    assert all(field.shape == (8, 10, 12) and np.isfinite(field).all() for field in calls["fields"].values())
    np.testing.assert_allclose(sum(calls["fields"].values()), 0, rtol=0, atol=1e-13)
    assert np.std(calls["fields"]["A"] - calls["fields"]["C"]) > 0


@pytest.mark.parametrize("dim, shape, lengths, angles", (
    (1, "line", [3], [90, 90, 90]), (2, "square", [3, 3], [90, 90, 90]),
    (2, "rectangle", [3, 4], [90, 90, 90]), (3, "cubic", [3, 3, 3], [90, 90, 90]),
    (3, "hexagonal", [3, 3, 4], [90, 90, 120]),
    (3, "triclinic", [3, 4, 5], [85, 80, 70]),
))
def test_box_exports(dim, shape, lengths, angles, tmp_path, monkeypatch):
    form = values()
    form["model"]["box"] = {"dimension": dim, "shape": shape, "nx": ["8"] * dim,
                             "lx": [str(n) for n in lengths], "angles": [str(a) for a in angles]}
    if shape == "square":
        # The API has no square-preserving 2D relaxation mode. Export a fixed
        # square only after checking that the unsupported request is rejected.
        with pytest.raises(ValueError, match="2D Square cells require fixed-cell"):
            BUILDER["build_script"](form)
        form["relax_cell"] = False
    calls = CHECKS["execute_export"](form, tmp_path, monkeypatch)
    assert calls["params"]["lx"] == lengths
    assert calls["params"].get("angles", [90, 90, 90]) == angles
    assert calls["params"]["crystal_system"] == (shape.title() if dim == 3 else "Orthorhombic")
    assert calls["fields"]["A"].shape == (8,) * dim


def test_three_monomer_mat_fields(tmp_path, monkeypatch):
    form = values()
    form.update(initial="mat", mat_file="three.mat")
    data = {"nx": [8, 10, 12], **{"w_" + name: np.full((8, 10, 12), i) for i, name in enumerate("ABC")}}
    (tmp_path / "export").mkdir()
    savemat(tmp_path / "export/three.mat", data)
    calls = CHECKS["execute_export"](form, tmp_path, monkeypatch)
    for i, name in enumerate("ABC"):
        np.testing.assert_array_equal(calls["fields"][name], np.full((8, 10, 12), i))


@pytest.mark.parametrize("problem", ("duplicate_name", "invalid_name", "unknown_block", "negative_block",
    "volume_total", "negative_fraction", "duplicate_pair", "self_pair", "unused_monomer", "empty_polymer",
    "angles", "cubic_lengths", "hexagonal_angles", "grid_length", "dimension", "ordered_seed", "cn_oblique"))
def test_invalid_structured_models(problem):
    form = values()
    model = form["model"]
    if problem == "duplicate_name": model["monomers"][1]["name"] = "A"
    elif problem == "invalid_name": model["monomers"][0]["name"] = "A,B"
    elif problem == "unknown_block": model["polymers"][0]["blocks"][0]["type"] = "unknown"
    elif problem == "negative_block": model["polymers"][0]["blocks"][0]["length"] = "-1"
    elif problem == "volume_total": model["polymers"][0]["volume_fraction"] = "0.8"
    elif problem == "negative_fraction": model["polymers"][0]["volume_fraction"] = "-0.6"
    elif problem == "duplicate_pair": model["interactions"].append({"a": "B", "b": "A", "chi_n": "3"})
    elif problem == "self_pair": model["interactions"][0]["b"] = "A"
    elif problem == "unused_monomer": model["monomers"].append({"name": "D", "segment_length": "1"})
    elif problem == "empty_polymer": model["polymers"][0]["blocks"] = []
    elif problem == "angles": model["box"].update(shape="triclinic", angles=["10", "10", "170"])
    elif problem == "cubic_lengths": model["box"]["shape"] = "cubic"
    elif problem == "hexagonal_angles": model["box"].update(shape="hexagonal", lx=["3", "3", "5"])
    elif problem == "grid_length": model["box"]["nx"] = ["8"]
    elif problem == "dimension": model["box"]["dimension"] = 4
    elif problem == "ordered_seed": form["initial"] = "lamella"
    elif problem == "cn_oblique":
        model["box"].update(shape="triclinic", angles=["85", "80", "70"])
        form["numerical_method"] = "cn-adi2"
    with pytest.raises(ValueError):
        BUILDER["build_script"](form)


def test_structured_model_http_save(tmp_path):
    server = BUILDER["create_server"](0, tmp_path)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        payload = {"values": values(), "filename": "mixture.py", "directory": "mixture"}
        request = Request(f"http://127.0.0.1:{server.server_address[1]}/api/save",
                          data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            assert response.status == 201
            result = json.load(response)
        assert result["path"] == str(tmp_path / "mixture/mixture.py")
        assert Path(result["path"]).read_text() == result["script"]
        compile(result["script"], result["path"], "exec")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
