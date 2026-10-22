"""Regression checks for cell relaxation, complex solvers and public option names."""

import ast
import json
from pathlib import Path
import runpy
import threading

import pytest


MODEL = runpy.run_path(str(Path(__file__).with_name("TestSimulationModel.py")))
BUILDER = MODEL["BUILDER"]
WEB = runpy.run_path(str(Path(__file__).with_name("TestSimulationWeb.py")))


def parameters(form):
    script = BUILDER["build_script"](form)
    return next(ast.literal_eval(n.value) for n in ast.parse(script).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "params" for t in n.targets))


@pytest.mark.parametrize("mode", ("SCFT", "L-FTS", "CL-FTS"))
def test_exports_use_canonical_parameter_names(mode):
    form = BUILDER["preset_settings"](f"{mode}: AB lamella")
    form["numerical_method"] = "auto"
    if mode == "SCFT":
        form["space_group"] = "530"
    params = parameters(form)
    assert "polymers" in params
    assert not {"distinct_polymers", "box_is_altering", "scale_stress"}.intersection(params)
    if mode == "SCFT":
        assert params["relax_cell"] is True
        assert params["space_group"] == {"symbol": "Ia-3d", "hall_number": 530}
    else:
        assert set(params["recording"]) == {
            "dir", "checkpoint_interval", "sample_interval", "statistics_save_interval"}


@pytest.mark.parametrize("shape, lengths, system", (
    ("cubic", ["3", "3", "3"], "Cubic"),
    ("orthorhombic", ["3", "4", "5"], "Orthorhombic"),
    ("hexagonal", ["3", "3", "5"], "Hexagonal"),
    ("triclinic", ["3", "4", "5"], "Triclinic"),
))
def test_scft_preserves_selected_relaxation_constraints(shape, lengths, system):
    form = MODEL["values"]()
    angles = ["90", "90", "120"] if shape == "hexagonal" else ["90"] * 3
    form["model"]["box"].update(shape=shape, lx=lengths, angles=angles)
    params = parameters(form)
    assert params["relax_cell"] is True
    assert params["crystal_system"] == system
    # A triclinic cell must retain free angles even when it starts orthogonal.
    assert params.get("angles", [90, 90, 90]) == [int(a) for a in angles]


@pytest.mark.parametrize("mode", ("L-FTS", "CL-FTS"))
def test_fts_does_not_export_scft_cell_relaxation(mode):
    form = MODEL["values"]()
    form["mode"] = mode
    assert not {"crystal_system", "relax_cell"}.intersection(parameters(form))


def test_space_group_determines_crystal_system():
    form = MODEL["values"]()
    form["model"]["box"].update(lx=["3", "3", "5"], nx=["8"] * 3)
    form["space_group"] = "419"  # Tetragonal symmetry in the Orthorhombic editor shape.
    params = parameters(form)
    assert params["space_group"]["hall_number"] == 419
    assert "crystal_system" not in params


def test_cn_adi_cannot_relax_initially_orthogonal_triclinic_angles():
    form = MODEL["values"]()
    form["model"]["box"]["shape"] = "triclinic"
    form["numerical_method"] = "cn-adi2"
    with pytest.raises(ValueError, match="relax triclinic angles"):
        BUILDER["build_script"](form)


@pytest.mark.parametrize("mode", ("SCFT", "L-FTS"))
def test_cn_adi_remains_available_for_real_orthogonal_fields(mode):
    form = MODEL["values"]()
    form.update(mode=mode, numerical_method="cn-adi2")
    assert parameters(form)["numerical_method"] == "cn-adi2"


@pytest.mark.parametrize("method", ("auto", "rqm4", "rk2"))
def test_clfts_continuous_pseudo_spectral_methods(method):
    form = BUILDER["preset_settings"]("CL-FTS: AB lamella")
    form.update(chain_model="continuous", numerical_method=method)
    params = parameters(form)
    assert params.get("numerical_method", "auto") == method


def test_clfts_cn_adi_is_rejected_before_generation_or_save(tmp_path):
    form = BUILDER["preset_settings"]("CL-FTS: AB lamella")
    form.update(chain_model="continuous", numerical_method="cn-adi2")
    with pytest.raises(ValueError, match="does not support complex fields"):
        BUILDER["build_script"](form)
    server = BUILDER["create_server"](0, tmp_path / "exports")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        for route, payload in (("/api/generate", form),
                ("/api/save", {"values": form, "filename": "invalid.py", "directory": ""})):
            status, _, body = WEB["request"](server, "POST", route, payload)
            assert status == 400
            assert "does not support complex fields" in json.loads(body)["error"]
        assert not server.save_dir.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
