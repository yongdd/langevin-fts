"""SCFT-only symmetry selection, with offline cell/grid validation and export."""

import ast
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys
import threading

import pytest


MODEL = runpy.run_path(str(Path(__file__).with_name("TestSimulationModel.py")))
BUILDER = MODEL["BUILDER"]
WEB = runpy.run_path(str(Path(__file__).with_name("TestSimulationWeb.py")))


def values(hall=530):
    form = MODEL["values"]()
    group = BUILDER["space_groups"]()[str(hall)]
    form["space_group"] = str(hall)
    system = group["crystal_system"]
    shape, lengths, angles = "cubic", ["3", "3", "3"], ["90", "90", "90"]
    if system in ("Orthorhombic", "Tetragonal"):
        shape, lengths = "orthorhombic", ["3", "3" if system == "Tetragonal" else "4", "5"]
    elif group["hexagonal_axes"]:
        shape, lengths, angles = "hexagonal", ["3", "3", "5"], ["90", "90", "120"]
    elif system == "Trigonal":
        shape, angles = "triclinic", ["80", "80", "80"]
        form["relax_cell"] = False
    elif system == "Monoclinic":
        # The angle between the two axes perpendicular to the unique axis (alpha for a, ...) is free
        shape, lengths = "triclinic", ["3", "4", "5"]
        angles = ["100" if i == group["unique_axis"] else "90" for i in range(3)]
    elif system == "Triclinic":
        shape, lengths, angles = "triclinic", ["3", "4", "5"], ["80", "100", "110"]
    n = 8 * math.lcm(*group["grid_divisors"])
    form["model"]["box"] = {"dimension": 3, "shape": shape, "nx": [str(n)] * 3,
                             "lx": lengths, "angles": angles}
    return form


def parameters(script):
    return next(ast.literal_eval(n.value) for n in ast.parse(script).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "params" for t in n.targets))


def test_every_catalog_setting_can_be_exported():
    groups = BUILDER["space_groups"]()
    assert len(groups) == 530
    for key, group in groups.items():
        script = BUILDER["build_script"](values(int(key)))
        compile(script, key, "exec")
        assert parameters(script)["space_group"] == {"symbol": group["symbol"], "hall_number": group["hall"]}


@pytest.mark.parametrize("hall, symbol", ((529, "Im-3m"), (530, "Ia-3d"), (526, "Fd-3m"),
    (419, "P4_2/mnm"), (488, "P6_3/mmc"), (336, "Fddd"), (434, "R3"), (81, "P2_1/c"), (58, "P2/m"), (2, "P-1")))
def test_selected_symmetry_reaches_scft(hall, symbol, tmp_path, monkeypatch):
    form = values(hall)
    calls = MODEL["CHECKS"]["execute_export"](form, tmp_path, monkeypatch)
    assert calls["params"]["space_group"] == {"symbol": symbol, "hall_number": hall}
    assert calls["saved"] == "fields.mat"


@pytest.mark.parametrize("mode", ("L-FTS", "CL-FTS"))
@pytest.mark.parametrize("selection", ("530", "488", "invalid"))
def test_space_group_is_ignored_by_fts(mode, selection, tmp_path, monkeypatch):
    form = MODEL["values"]()
    form.update(mode=mode, space_group=selection)
    calls = MODEL["CHECKS"]["execute_export"](form, tmp_path, monkeypatch)
    assert "space_group" not in calls["params"]


@pytest.mark.parametrize("hall, shape", ((419, "tetragonal"), (81, "monoclinic")))
def test_dedicated_cell_shapes_with_space_groups(hall, shape):
    form = values(hall)
    form["model"]["box"]["shape"] = shape
    params = parameters(BUILDER["build_script"](form))
    assert params["space_group"]["hall_number"] == hall
    assert "crystal_system" not in params


@pytest.mark.parametrize("problem, message", (
    ("2d", "requires a 3D"), ("lengths", "a = b = c"),
    ("angles", "angles"), ("grid_axes", "nx = ny"),
    ("gyroid_divisor", "multiple of 4"), ("bcc_divisor", "multiple of 2"),
    ("tetragonal_lengths", "a = b"), ("hexagonal_angles", "120"),
    ("hexagonal_divisor", "nz.*multiple of 2"),
    ("rhombohedral_relaxation", "require fixed box size"),
    ("unknown_hall", "supported space group"), ("monoclinic_angles", "unique axis b perpendicular"),
))
def test_invalid_symmetry_is_rejected(problem, message):
    form = values()
    box = form["model"]["box"]
    if problem == "2d": box.update(dimension=2, shape="square", nx=["8", "8"], lx=["3", "3"])
    elif problem == "lengths": box.update(shape="orthorhombic", lx=["3", "4", "5"])
    elif problem == "angles": box.update(shape="triclinic", angles=["85", "80", "70"])
    elif problem == "grid_axes": box["nx"][1] = "12"
    elif problem == "gyroid_divisor": box["nx"] = ["10"] * 3
    elif problem == "bcc_divisor": form = values(529); form["model"]["box"]["nx"] = ["9"] * 3
    elif problem == "tetragonal_lengths": form = values(419); form["model"]["box"]["lx"][1] = "4"
    elif problem == "hexagonal_angles": form["space_group"] = "488"
    elif problem == "hexagonal_divisor": form = values(488); form["model"]["box"]["nx"] = ["7"] * 3
    elif problem == "rhombohedral_relaxation": form = values(434); form["relax_cell"] = True
    elif problem == "unknown_hall": form["space_group"] = "531"
    elif problem == "monoclinic_angles": form = values(57); form["model"]["box"]["angles"] = ["90", "90", "100"]
    with pytest.raises(ValueError, match=message):
        BUILDER["build_script"](form)


def test_hexagonal_translation_only_constrains_z():
    form = values(488)
    form["model"]["box"]["nx"] = ["7", "7", "8"]
    assert parameters(BUILDER["build_script"](form))["nx"] == [7, 7, 8]


def test_none_keeps_scft_lower_dimensions_available():
    form = BUILDER["preset_settings"]("SCFT: ABA lamella")
    assert form["space_group"] == "none"
    assert "space_group" not in parameters(BUILDER["build_script"](form))


def test_symmetry_generation_without_site_packages():
    path = Path(__file__).resolve().parents[2] / "scripts/gui/simulation_gui.py"
    code = (f"import runpy, sys\nb = runpy.run_path({str(path)!r})\n"
            "v = b['preset_settings'](b['PRESETS'][0]); v['space_group'] = '530'\n"
            "compile(b['build_script'](v), 'sg.py', 'exec')\n"
            "assert not {'numpy', 'spglib', 'polymerfts'}.intersection(sys.modules)\n")
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True, timeout=10, capture_output=True)


def test_symmetry_catalog_and_server_save(tmp_path):
    server = BUILDER["create_server"](0, tmp_path / "scripts")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        status, _, body = WEB["request"](server, path="/api/config")
        assert status == 200
        options = json.loads(body)["space_groups"]
        assert len(options) == 530 and len({o["hall"] for o in options}) == 530
        assert any(o["hall"] == 530 and "Gyroid" in o["label"] for o in options[:11])
        form = values()
        form["model"]["box"]["nx"] = ["10"] * 3
        payload = {"values": form, "filename": "symmetry.py", "directory": "gyroid"}
        status, _, body = WEB["request"](server, "POST", "/api/save", payload)
        assert status == 400 and "multiple of 4" in json.loads(body)["error"]
        assert not server.save_dir.exists()
        payload["values"] = values()
        status, _, body = WEB["request"](server, "POST", "/api/save", payload)
        assert status == 201
        result = json.loads(body)
        assert Path(result["path"]).read_text() == result["script"]
        assert parameters(result["script"])["space_group"] == {"symbol": "Ia-3d", "hall_number": 530}
        payload["values"]["mode"] = "CL-FTS"
        payload["filename"] = "clfts.py"
        status, _, body = WEB["request"](server, "POST", "/api/save", payload)
        assert status == 201
        assert "space_group" not in parameters(json.loads(body)["script"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
