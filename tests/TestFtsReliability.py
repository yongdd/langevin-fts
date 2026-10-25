"""Failure semantics, restart compatibility, pending samples and atomic writes."""
import contextlib
import copy
import io
import json
import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import LFTS, CLFTS, PlatformSelector, ValidationError, SaddlePointError
from polymerfts.utils import atomic_savemat
from polymerfts.validation import validate_lfts_params


def parameters(directory, steps=4):
    return {
        "nx": [16], "lx": [3.0], "ds": 0.05, "chain_model": "continuous",
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0},
        "polymers": [{"volume_fraction": 1.0, "blocks": [
            {"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "platform": next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu")),
        "langevin": {"nbar": 10000, "dt": 1.0, "max_step": steps},
        "compressor": {"name": "am", "max_hist": 20, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01},
        "saddle": {"max_iter": 200, "tolerance": 1e-6},
        "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1, "statistics_save_interval": 2},
        "verbose_level": 0,
    }


def fields():
    rng = np.random.default_rng(3)
    return {m: rng.normal(0.0, 5.0, 16) for m in ("A", "B")}


class FailingLFTS(LFTS):
    fail_calls = ()

    def find_saddle_point(self, w_aux):
        self.calls = getattr(self, "calls", 0) + 1
        phi, h, iterations, error = super().find_saddle_point(w_aux)
        return phi, h, iterations, np.inf if self.calls in self.fail_calls else error


def expect_error(call, kind):
    try:
        call()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        previous = Path.cwd()
        os.chdir(root)  # any stray output stays in the temporary directory
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for cls in (LFTS, CLFTS):
                    full = cls(parameters(root / (cls.__name__ + "full")), random_seed=11)
                    assert full.run(fields())["status"] == "completed"
                    # The run draws the chain architectures into its recording directory, and
                    # neither the constructor nor the run writes into the current directory
                    assert (root / (cls.__name__ + "full") / "polymer_0.png").is_file(), cls.__name__
                    assert not (root / "polymer_0.png").exists(), cls.__name__
                    first = cls(parameters(root / (cls.__name__ + "first"), 1), random_seed=11)
                    first.run(fields())
                    continued = cls(parameters(root / (cls.__name__ + "continued")), random_seed=99)
                    result = continued.continue_run(root / (cls.__name__ + "first") / "fields_000001.mat")
                    assert result["last_completed_step"] == 4
                    assert continued.random_bg.state == full.random_bg.state
                    for name in ("fields_000004.mat", "dH_000002.mat", "structure_function_000002.mat"):
                        a = loadmat(root / (cls.__name__ + "full") / name)
                        b = loadmat(root / (cls.__name__ + "continued") / name)
                        for key in a:
                            if key.startswith(("w_", "phi_", "H_history", "dH_history", "structure_function", "drift_max_history")):
                                assert np.array_equal(a[key], b[key]), (cls.__name__, name, key)
                    checkpoint = root / (cls.__name__ + "first") / "fields_000001.mat"
                    equivalent_params = parameters(root / (cls.__name__ + "equivalent"))
                    equivalent_params["langevin"]["nbar"] = 10000.0
                    equivalent_params["chi_n"] = {"A,B": 15.0}
                    equivalent = cls(equivalent_params, random_seed=99)
                    assert equivalent.continue_run(checkpoint)["status"] == "completed"
                    for key in ("dt", "nbar"):
                        params = parameters(root / "incompatible")
                        params["langevin"][key] *= 2
                        bad = cls(params, random_seed=13)
                        rng_before = copy.deepcopy(bad.random_bg.state)
                        expect_error(lambda: bad.continue_run(checkpoint), ValidationError)
                        assert bad.random_bg.state == rng_before

                normal = LFTS(parameters(root / "normal"), random_seed=5)
                normal.run(fields())
                FailingLFTS.fail_calls = (3,)
                recovered = FailingLFTS(parameters(root / "recovered"), random_seed=5)
                result = recovered.run(fields())
                assert result["saddle_failures"] == 1 and result["noise_resamples"] == 0
                assert recovered.random_bg.state == normal.random_bg.state
                params = parameters(root / "resampling")
                params["saddle"]["failure_policy"] = "resample"
                resampling = FailingLFTS(params, random_seed=5)
                with warnings.catch_warnings(record=True) as messages:
                    warnings.simplefilter("always")
                    result = resampling.run(fields())
                assert result["noise_resamples"] == 1
                assert any("sampled distribution" in str(message.message) for message in messages)
                assert resampling.random_bg.state != normal.random_bg.state
                params = parameters(root / "abort")
                params["saddle"]["failure_policy"] = "abort"
                aborting = FailingLFTS(params, random_seed=5)
                expect_error(lambda: aborting.run(fields()), SaddlePointError)
                assert aborting.calls == 3 and aborting.run_status["last_completed_step"] == 1
                FailingLFTS.fail_calls = tuple(range(1, 20))
                initial = FailingLFTS(parameters(root / "initial_failure"), random_seed=5)
                rng_before = copy.deepcopy(initial.random_bg.state)
                expect_error(lambda: initial.run(fields()), SaddlePointError)
                assert initial.run_status["last_completed_step"] == 0
                assert initial.random_bg.state == rng_before
                FailingLFTS.fail_calls = tuple(range(3, 20))
                failing = FailingLFTS(parameters(root / "failure"), random_seed=5)
                expect_error(lambda: failing.run(fields()), SaddlePointError)
                status = json.loads((root / "failure" / "run_status.json").read_text())
                assert status["status"] == "aborted_saddle_failure" and status["last_completed_step"] == 1
                assert not (root / "failure" / "fields_000002.mat").exists()

                # Explicit legacy opt-in and corrupt pending state are checked before RNG mutation.
                checkpoint = root / "LFTSfirst" / "fields_000001.mat"
                data = loadmat(checkpoint, squeeze_me=True)
                data["record_sample_count"] = 2
                atomic_savemat(root / "corrupt.mat", {k: v for k, v in data.items() if not k.startswith("__")})
                candidate = LFTS(parameters(root / "corrupt"), random_seed=21)
                rng_before = copy.deepcopy(candidate.random_bg.state)
                expect_error(lambda: candidate.continue_run(root / "corrupt.mat"), ValidationError)
                assert candidate.random_bg.state == rng_before
                data = loadmat(checkpoint, squeeze_me=True)
                data.pop("checkpoint_format_version")
                atomic_savemat(root / "legacy.mat", {k: v for k, v in data.items() if not k.startswith("__")})
                expect_error(lambda: candidate.continue_run(root / "legacy.mat"), ValidationError)

                # MATLAB squeezes a one-species field basis to a scalar.
                for cls in (LFTS, CLFTS):
                    params = parameters(root / (cls.__name__ + "_homopolymer"), 1)
                    params["segment_lengths"] = {"A": 1.0}
                    params["chi_n"] = {}
                    params["polymers"] = [{"volume_fraction": 1.0,
                                                       "blocks": [{"type": "A", "length": 1.0}]}]
                    homogeneous = cls(params, random_seed=5)
                    homogeneous.run({"A": fields()["A"]})
                    params["langevin"]["max_step"] = 2
                    continued = cls(params, random_seed=99)
                    result = continued.continue_run(Path(params["recording"]["dir"]) / "fields_000001.mat")
                    assert result["last_completed_step"] == 2

            for complex_fields in (False, True):
                for section, key, value in ((None, "ds", np.nan), ("langevin", "dt", np.inf),
                                            ("langevin", "nbar", -1), ("langevin", "max_step", True),
                                            ("recording", "sample_interval", 0)):
                    params = parameters(root)
                    (params if section is None else params[section])[key] = value
                    expect_error(lambda: validate_lfts_params(params, complex_fields=complex_fields), ValidationError)

            target = root / "atomic.mat"
            target.write_bytes(b"previous checkpoint")
            with patch("scipy.io.savemat", side_effect=OSError("injected write failure")):
                expect_error(lambda: atomic_savemat(target, {"x": 1}), OSError)
            assert target.read_bytes() == b"previous checkpoint"
            assert not list(root.glob(".atomic.mat.*"))
        finally:
            os.chdir(previous)
    print("Test passed, FTS reliability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
