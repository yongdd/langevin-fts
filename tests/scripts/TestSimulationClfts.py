"""CL-FTS script exports, including complex input fields, without a simulation backend."""

from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.io import savemat


CHECKS = runpy.run_path(str(Path(__file__).with_name("TestSimulationScript.py")))
BUILDER = CHECKS["BUILDER"]


def values():
    return BUILDER["preset_settings"]("CL-FTS: AB lamella")


@pytest.mark.parametrize("smearing, expected", (
    ("none", None), ("gaussian", {"type": "gaussian", "a_int": .02}),
    ("sigmoidal", {"type": "sigmoidal", "k_int": 20, "dk_int": 5}),
))
def test_clfts_parameters_and_dispatch(smearing, expected, tmp_path, monkeypatch):
    form = values()
    form["smearing"] = smearing
    script = BUILDER["build_script"](form)
    assert "from polymerfts import clfts" in script
    assert "simulation = clfts.CLFTS(params=params, random_seed=12345)" in script
    calls = CHECKS["execute_export"](form, tmp_path, monkeypatch)
    params = calls["params"]
    CHECKS["VALIDATION"]["validate_lfts_params"](params, complex_fields=True)
    assert params["zeta_n"] == 100
    assert params["dynamic_stabilization"] == .01
    assert params["langevin"] == {"max_step": 100000, "dt": 1, "nbar": 10000}
    assert params.get("smearing") == expected
    assert not {"optimizer", "saddle", "compressor", "relax_cell"}.intersection(params)
    assert "saved" not in calls


def test_clfts_ignores_inactive_solver_controls(tmp_path, monkeypatch):
    form = values()
    form.update(max_hist="", mix_init="", max_iter="", saddle_max_iter="",
                compressor="", a_int="", k_int="", dk_int="",
                zeta_n="0", dynamic_stabilization="0")
    calls = CHECKS["execute_export"](form, tmp_path, monkeypatch)
    CHECKS["VALIDATION"]["validate_lfts_params"](calls["params"], complex_fields=True)
    assert calls["params"]["dynamic_stabilization"] == 0
    assert calls["params"]["zeta_n"] == 0
    assert "smearing" not in calls["params"]


@pytest.mark.parametrize("update", (
    {"zeta_n": "-1"}, {"zeta_n": "inf"}, {"dynamic_stabilization": "-1"},
    {"dynamic_stabilization": "nan"}, {"smearing": "unknown"},
    {"smearing": "gaussian", "a_int": "-1"},
    {"smearing": "sigmoidal", "k_int": "0"},
    {"smearing": "sigmoidal", "dk_int": "0"},
    {"statistics_save_interval": "11", "sample_interval": "10"},
))
def test_invalid_clfts_controls(update):
    form = values()
    form.update(update)
    with pytest.raises(ValueError):
        BUILDER["build_script"](form)


@pytest.mark.parametrize("imaginary", (0, 2))
def test_clfts_preserves_mat_fields(imaginary, tmp_path, monkeypatch):
    form = values()
    form.update(nx="8, 4, 1", lx="3, 2, 1", initial="mat", mat_file="complex.mat")
    real = np.arange(32, dtype=float)
    field = real if imaginary == 0 else real + imaginary * 1j
    (tmp_path / "export").mkdir()
    savemat(tmp_path / "export/complex.mat", {"nx": [8, 4, 1], "w_A": field, "w_B": -field})
    calls = CHECKS["execute_export"](form, tmp_path, monkeypatch)
    np.testing.assert_array_equal(calls["fields"]["A"].ravel(), field)
    np.testing.assert_array_equal(calls["fields"]["B"].ravel(), -field)
    assert calls["fields"]["A"].dtype == np.complex128


def test_clfts_rejects_nonfinite_complex_fields(tmp_path, monkeypatch):
    form = values()
    form.update(nx="8, 1, 1", lx="3, 1, 1", initial="mat", mat_file="complex.mat")
    field = np.full(8, 1 + np.inf * 1j)
    (tmp_path / "export").mkdir()
    savemat(tmp_path / "export/complex.mat", {"w_A": field, "w_B": field})
    with pytest.raises(ValueError, match="finite value"):
        CHECKS["execute_export"](form, tmp_path, monkeypatch)
    assert not (tmp_path / "export/simulation").exists()


@pytest.mark.parametrize("problem", (None, "missing_imag", "shape", "nonfinite", "complex_part"))
def test_clfts_split_mat_fields(problem, tmp_path, monkeypatch):
    form = values()
    form.update(nx="8, 1, 1", lx="3, 1, 1", initial="mat", mat_file="split.mat")
    real = np.arange(8, dtype=float)
    imag = np.full(8, .5)
    data = {"nx": [8, 1, 1], "w_A_real": real, "w_A_imag": imag,
            "w_B_real": -real, "w_B_imag": -imag}
    if problem == "missing_imag": del data["w_A_imag"]
    elif problem == "shape": data["w_A_imag"] = np.full(1, .5)
    elif problem == "nonfinite": data["w_A_imag"] = np.full(8, np.inf)
    elif problem == "complex_part": data["w_A_real"] = real + 1j
    (tmp_path / "export").mkdir()
    savemat(tmp_path / "export/split.mat", data)
    if problem:
        with pytest.raises(ValueError):
            CHECKS["execute_export"](form, tmp_path, monkeypatch)
        assert not (tmp_path / "export/simulation").exists()
    else:
        calls = CHECKS["execute_export"](form, tmp_path, monkeypatch)
        np.testing.assert_array_equal(calls["fields"]["A"].ravel(), real + 1j * imag)
        np.testing.assert_array_equal(calls["fields"]["B"].ravel(), -real - 1j * imag)


def test_clfts_structured_mixture(tmp_path, monkeypatch):
    model_checks = runpy.run_path(str(Path(__file__).with_name("TestSimulationModel.py")))
    form = model_checks["values"]()
    form.update(mode="CL-FTS", dynamic_stabilization="0.005", smearing="gaussian")
    calls = CHECKS["execute_export"](form, tmp_path, monkeypatch)
    CHECKS["VALIDATION"]["validate_lfts_params"](calls["params"], complex_fields=True)
    assert set(calls["fields"]) == {"A", "B", "C"}
    assert calls["params"]["dynamic_stabilization"] == .005
    assert calls["params"]["smearing"] == {"type": "gaussian", "a_int": .02}
    np.testing.assert_allclose(sum(calls["fields"].values()), 0, rtol=0, atol=1e-13)
