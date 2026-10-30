"""The browser editor permits Langevin FTS exports only in 3D orthogonal cells."""

import json
from pathlib import Path
import runpy
import threading

import pytest


MODEL = runpy.run_path(str(Path(__file__).with_name("TestSimulationModel.py")))
BUILDER = MODEL["BUILDER"]
WEB = runpy.run_path(str(Path(__file__).with_name("TestSimulationWeb.py")))


@pytest.mark.parametrize("mode", ("L-FTS", "CL-FTS"))
@pytest.mark.parametrize("dimension", (1, 2))
def test_fts_rejects_lower_dimensions(mode, dimension):
    form = BUILDER["preset_settings"](f"{mode}: AB lamella")
    form.update(nx=", ".join(["8"] * dimension), lx=", ".join(["3"] * dimension))
    with pytest.raises(ValueError, match="3D orthogonal"):
        BUILDER["build_script"](form)
    form = MODEL["values"]()
    form["mode"] = mode
    form["model"]["box"] = {"dimension": dimension, "shape": "line" if dimension == 1 else "square",
                            "nx": ["8"] * dimension, "lx": ["3"] * dimension, "angles": ["90"] * 3}
    with pytest.raises(ValueError, match="3D orthogonal"):
        BUILDER["build_script"](form)


@pytest.mark.parametrize("mode", ("L-FTS", "CL-FTS"))
@pytest.mark.parametrize("shape, lengths, angles", (
    ("hexagonal", ["3", "3", "4"], ["90", "90", "120"]),
    ("monoclinic", ["3", "4", "5"], ["90", "105", "90"]),
    ("triclinic", ["3", "4", "5"], ["85", "80", "70"]),
    ("triclinic", ["3", "4", "5"], ["90", "90", "90.000001"]),
))
def test_fts_rejects_skewed_cells(mode, shape, lengths, angles):
    form = MODEL["values"]()
    form["mode"] = mode
    form["model"]["box"].update(shape=shape, lx=lengths, angles=angles)
    with pytest.raises(ValueError, match="3D orthogonal"):
        BUILDER["build_script"](form)


@pytest.mark.parametrize("mode", ("L-FTS", "CL-FTS"))
@pytest.mark.parametrize("shape, lengths", (
    ("cubic", ["3", "3", "3"]), ("orthorhombic", ["3", "4", "5"]),
    ("triclinic", ["3", "4", "5"]),  # Physical angles, rather than the shape label, define skew.
))
def test_fts_accepts_only_orthogonal_geometry(mode, shape, lengths, tmp_path, monkeypatch):
    form = MODEL["values"]()
    form["mode"] = mode
    form["model"]["box"].update(shape=shape, lx=lengths)
    calls = MODEL["CHECKS"]["execute_export"](form, tmp_path, monkeypatch)
    assert calls["params"]["nx"] == [8, 10, 12]
    assert calls["params"]["lx"] == [float(x) for x in lengths]
    assert "angles" not in calls["params"]
    assert all(field.shape == (8, 10, 12) for field in calls["fields"].values())


@pytest.mark.parametrize("mode", ("L-FTS", "CL-FTS"))
@pytest.mark.parametrize("dimension, shape, lengths, angles", (
    (1, "line", ["3"], ["90", "90", "90"]),
    (2, "rectangle", ["3", "4"], ["90", "90", "90"]),
    (3, "triclinic", ["3", "4", "5"], ["85", "80", "70"]),
))
def test_http_rejects_invalid_fts_geometry_before_saving(mode, dimension, shape, lengths, angles, tmp_path):
    form = MODEL["values"]()
    form["mode"] = mode
    form["model"]["box"] = {"dimension": dimension, "shape": shape,
                             "nx": ["8"] * dimension, "lx": lengths, "angles": angles}
    server = BUILDER["create_server"](0, tmp_path / "scripts")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        for route, payload in (
            ("/api/generate", form),
            ("/api/save", {"values": form, "filename": "fts.py", "directory": "invalid"}),
        ):
            status, _, body = WEB["request"](server, "POST", route, payload)
            assert status == 400
            assert "3D orthogonal" in json.loads(body)["error"]
        assert not server.save_dir.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
