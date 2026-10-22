"""Validate reported Monte Carlo errors against an AR(1) process with known variance."""
import importlib.util
import contextlib
import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np


def main():
    spec = importlib.util.spec_from_file_location("sampling_validation", Path(__file__).resolve().parents[1] /
                                                 "scripts" / "validate_fts_sampling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Forty post-burn-in steps can span only three complete recorded blocks
    # when the burn-in boundary cuts through the first block.
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "invalid.json"
        argv = ["validate_fts_sampling.py", "--model", "lfts", "--dt", "1",
                "--steps", "41", "--burn-in", "1", "--block", "10", "--output", str(output)]
        with patch.object(sys, "argv", argv), patch.object(module, "LFTS", side_effect=AssertionError("Simulation started")), contextlib.redirect_stderr(io.StringIO()):
            try:
                module.main()
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError("Incomplete statistical blocks were accepted")
        assert not output.exists() and not (Path(directory) / "invalid_data").exists()
    rng = np.random.default_rng(314)
    samples = rng.normal(size=100000)
    independent = module.mean_error(samples)
    assert independent["ess"] > 70000
    rho = 0.8
    correlated = samples.copy()
    for i in range(1, correlated.size):
        correlated[i] = rho * correlated[i - 1] + np.sqrt(1 - rho**2) * samples[i]
    estimate = module.mean_error(correlated)
    expected_mcse = np.sqrt((1 + rho) / (1 - rho) / correlated.size)
    assert abs(estimate["mcse"] / expected_mcse - 1) < 0.2
    assert module.mean_error(np.ones(20))["mcse"] == 0.0
    print("Test passed, autocorrelation-aware sampling errors")


if __name__ == "__main__":
    main()
