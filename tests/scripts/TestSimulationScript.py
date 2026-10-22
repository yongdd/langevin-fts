"""Exercise exported scripts with a recording simulation stand-in, without a backend or GUI."""

import copy
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[2]
BUILDER = runpy.run_path(str(ROOT / "scripts" / "simulation_gui.py"))
VALIDATION = runpy.run_path(str(ROOT / "src" / "python" / "validation.py"))


def execute_export(values, tmp_path, monkeypatch):
    """Check the exported script's real control flow, intercepting only the solver."""
    calls = {}

    class Simulation:
        def __init__(self, params, **kwargs):
            calls["params"] = copy.deepcopy(params)
            calls["kwargs"] = kwargs
            calls["cwd"] = Path.cwd()

        def run(self, initial_fields):
            calls["fields"] = initial_fields

        def save_results(self, path):
            calls["saved"] = path

    fake = ModuleType("polymerfts")
    fake.scft = SimpleNamespace(SCFT=Simulation)
    fake.lfts = SimpleNamespace(LFTS=Simulation)
    fake.clfts = SimpleNamespace(CLFTS=Simulation)
    monkeypatch.setitem(sys.modules, "polymerfts", fake)
    for name in ("OMP_NUM_THREADS", "OMP_MAX_ACTIVE_LEVELS", "OMP_STACKSIZE"):
        # Track environment changes made by the generated code for pytest cleanup.
        monkeypatch.setenv(name, os.environ.get(name, "1"))
    monkeypatch.chdir(tmp_path)
    script = BUILDER["build_script"](values)
    script_path = tmp_path / "export" / "simulation.py"
    script_path.parent.mkdir(exist_ok=True)
    exec(compile(script, str(script_path), "exec"), {"__file__": str(script_path), "__name__": "__main__"})
    return calls


@pytest.mark.parametrize("preset", BUILDER["PRESETS"])
def test_exported_presets(preset, tmp_path, monkeypatch):
    values = BUILDER["preset_settings"](preset)
    values["output_dir"] = "results with 'quotes'"
    calls = execute_export(values, tmp_path, monkeypatch)
    params = calls["params"]
    if values["mode"] == "SCFT":
        VALIDATION["validate_system_params"](params)
    else:
        VALIDATION["validate_lfts_params"](params, complex_fields=values["mode"] == "CL-FTS")
    assert calls["cwd"] == tmp_path / "export" / values["output_dir"]
    assert calls["fields"]["A"].shape == tuple(params["nx"])
    assert np.isfinite(calls["fields"]["A"]).all()
    assert np.std(calls["fields"]["A"]) > 0
    np.testing.assert_array_equal(calls["fields"]["A"], -calls["fields"]["B"])
    if values["mode"] == "SCFT":
        assert calls["saved"] == "fields.mat"
    else:
        assert "saved" not in calls
        assert calls["kwargs"]["random_seed"] == 12345
    if "ABA" in preset:
        assert sum(b["length"] for b in params["polymers"][0]["blocks"]) == 2


@pytest.mark.parametrize("dimension", (1, 2, 3))
@pytest.mark.parametrize("initial", ("lamella", "random"))
def test_initial_field_dimensions(dimension, initial, tmp_path, monkeypatch):
    values = BUILDER["preset_settings"](BUILDER["PRESETS"][0])
    values.update(nx=", ".join(["8"] * dimension), lx=", ".join(["3"] * dimension), initial=initial,
                  platform="cpu-fftw", numerical_method="rk2")
    first = execute_export(values, tmp_path, monkeypatch)
    second = execute_export(values, tmp_path, monkeypatch)
    assert first["params"]["platform"] == "cpu-fftw"
    assert first["params"]["numerical_method"] == "rk2"
    assert first["fields"]["A"].shape == (8,) * dimension
    np.testing.assert_array_equal(first["fields"]["A"], second["fields"]["A"])


@pytest.mark.parametrize("updates", (
    {"nx": ""}, {"nx": "8, 8", "lx": "3"}, {"nx": "1.5"}, {"lx": "-1"},
    {"ds": "0"}, {"ds": "2"}, {"ds": "1/0"}, {"ds": "__import__('os').getcwd()"},
    {"threads": "5"}, {"seed": "-1"}, {"chi_n": "nan"}, {"b_a": "inf"},
    {"blocks": "A:0.5, C:0.5"}, {"blocks": "A:0.5"}, {"blocks": "A:0, B:1"},
    {"initial": "gyroid", "nx": "64", "lx": "4.0"}, {"initial": "mat"}, {"output_dir": ""},
    {"chain_model": "discrete", "numerical_method": "cn-adi2"},
    {"mix_init": "1.1"}, {"mode": "L-FTS", "chi_n": "0"},
    {"mode": "L-FTS", "statistics_save_interval": "11", "sample_interval": "10"},
))
def test_invalid_settings(updates):
    values = BUILDER["preset_settings"](BUILDER["PRESETS"][0])
    values.update(updates)
    with pytest.raises(ValueError):
        BUILDER["build_script"](values)


@pytest.mark.parametrize("problem", (None, "missing", "grid", "nonfinite", "complex"))
def test_mat_fields(problem, tmp_path, monkeypatch):
    values = BUILDER["preset_settings"](BUILDER["PRESETS"][0])
    values.update(nx="8, 4", lx="3, 2", initial="mat", mat_file="initial 'fields'.mat")
    field = np.arange(32, dtype=float)
    data = {"w_A": field.copy(), "w_B": -field, "nx": [8, 4]}
    if problem == "missing":
        del data["w_B"]
    elif problem == "grid":
        data["nx"] = [4, 8]  # Same number of points, different grid.
    elif problem == "nonfinite":
        data["w_A"][0] = np.nan
    elif problem == "complex":
        data["w_A"] = field + 1j
    (tmp_path / "export").mkdir()
    savemat(tmp_path / "export" / values["mat_file"], data)
    if problem:
        with pytest.raises(ValueError):
            execute_export(values, tmp_path, monkeypatch)
        assert not (tmp_path / "export" / "simulation").exists()
    else:
        calls = execute_export(values, tmp_path, monkeypatch)
        np.testing.assert_array_equal(calls["fields"]["A"].ravel(), field)


def test_large_seed_and_blank_seed():
    values = BUILDER["preset_settings"](BUILDER["PRESETS"][-1])
    # Preserve integer seeds exactly rather than round-tripping through float.
    values.update(initial="random", seed="9007199254740993")
    assert "random_seed=9007199254740993" in BUILDER["build_script"](values)
    values["seed"] = ""
    assert "random_seed=None" in BUILDER["build_script"](values)


def test_generation_without_site_packages():
    code = (
        "import runpy, sys\n"
        f"builder = runpy.run_path({str(ROOT / 'scripts' / 'simulation_gui.py')!r})\n"
        "for preset in builder['PRESETS']:\n"
        "    compile(builder['build_script'](builder['preset_settings'](preset)), preset, 'exec')\n"
        "assert 'polymerfts' not in sys.modules\n"
        "assert 'tkinter' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True, capture_output=True, timeout=10)
