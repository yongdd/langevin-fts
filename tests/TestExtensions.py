"""Extension contracts: stochastic moments, isolation, multistage forces and restart."""
import contextlib
import copy
import importlib.util
import io
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import (LFTS, CLFTS, PropagatorSolver, PlatformSelector,
                        LeimkuhlerMatthews, EulerMaruyama, ValidationError, SaddlePointError)
from polymerfts.utils import atomic_savemat, settings_json, input_hash
from polymerfts.integrators import checked_fields
from TestFtsReliability import parameters, fields, expect_error, FailingLFTS
from cuda_part import cuda_part_missing

example_path = Path(__file__).resolve().parents[1] / "examples/tools/CustomExtensions.py"
spec = importlib.util.spec_from_file_location("custom_fts_extensions", example_path)
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)


def check_stochastic_moments():
    # A real work array returned to CLFTS must still support the next complex
    # force update at the simulation's declared double precision.
    reference = np.zeros((1, 2), dtype=np.complex128)
    normalized = checked_fields(np.zeros((1, 2), dtype=np.float32), reference)
    result = LeimkuhlerMatthews().step(
        normalized, np.ones_like(reference) * 1j, dt=0.1, mobility=np.ones(1),
        noise=np.zeros((1, 2)), previous_noise=np.zeros((1, 2)),
        noise_direction=np.ones(1), evaluate_force=None)
    assert result.dtype == np.complex128 and np.all(result.imag < 0)
    expect_error(lambda: checked_fields(np.array([["bad", "fields"]]), reference), ValidationError)

    # Exact quadrature over independent +/- Gaussian-strength kicks. First two
    # moments agree with Gaussian noise for these linear harmonic dynamics.
    n, dt, k, temperature = 6, 0.2, 1.7, 2.3
    sigma = np.sqrt(2 * temperature * dt)
    kicks = sigma * np.array(list(itertools.product((-1.0, 1.0), repeat=n))).T
    for method in (LeimkuhlerMatthews(), EulerMaruyama(), example.PredictorCorrector()):
        x = np.zeros((1, kicks.shape[1]))
        previous = np.zeros_like(x)
        for noise in kicks:
            noise = noise[None, :]
            x = method.step(x, k * x, dt=dt, mobility=np.ones(1), noise=noise,
                            previous_noise=previous, noise_direction=np.ones(1),
                            evaluate_force=lambda candidate: k * candidate)
            previous = noise
        a = 1 - k * dt
        if method.name == "lm":
            b2 = sigma ** 2 / 4
            expected = a ** (2 * (n - 1)) * b2 + 2 * b2 * (1 + a) * (1 - a ** (2 * (n - 1))) / (1 - a * a)
        else:
            amplitude = sigma
            if method.name == "example-heun":
                a += 0.5 * (k * dt) ** 2
                amplitude *= 1 - 0.5 * k * dt
            expected = amplitude ** 2 * (1 - a ** (2 * n)) / (1 - a * a)
        np.testing.assert_allclose(x.mean(), 0, atol=1e-14)
        np.testing.assert_allclose(np.mean(x * x), expected, rtol=1e-13)


def compare_fields(a, b):
    for key in a:
        if key.startswith(("w_", "phi_", "normal_noise", "record_", "rng_state")):
            assert np.array_equal(a[key], b[key]), key


def check_observers_and_restart(root):
    for cls in (LFTS, CLFTS):
        prefix = cls.__name__
        seen = []
        sparse_steps = []

        def sparse(sample):
            sparse_steps.append(sample["step"])

        sparse.interval = 3

        def observer(sample):
            seen.append(sample["step"])
            assert sample["status"]["last_completed_step"] == sample["step"]
            assert sample["step_seconds"] >= 0
            np.testing.assert_allclose(sum(sample["phi"].values()).mean(), 1, atol=1e-12)
            expect_error(lambda: sample["w_aux"].setflags(write=True), ValueError)
            expect_error(lambda: sample["phi"]["A"].__setitem__(0, 0), ValueError)
            expect_error(lambda: sample.__setitem__("step", -1), AttributeError)

        baseline = cls(parameters(root / (prefix + "baseline")), random_seed=11)
        baseline.run(fields())
        monitored = cls(parameters(root / (prefix + "monitored")), random_seed=11,
                        callbacks={"monitor": observer, "sparse": sparse})
        monitored.run(fields())
        assert seen == [1, 2, 3, 4]
        assert sparse_steps == [3]
        assert baseline.random_bg.state == monitored.random_bg.state
        compare_fields(loadmat(root / (prefix + "baseline") / "fields_000004.mat"),
                       loadmat(root / (prefix + "monitored") / "fields_000004.mat"))

        for method in ("euler", "heun"):
            def create(directory, steps, seed):
                integral = example.PredictorCorrector() if method == "heun" else "euler"
                observable = example.CompositionVariance(every=2)
                sim = cls(parameters(directory, steps), random_seed=seed, integrator=integral,
                           callbacks={"variance": observable})
                return sim, observable

            full_dir = root / (prefix + method + "full")
            first_dir = root / (prefix + method + "first")
            next_dir = root / (prefix + method + "next")
            full, full_obs = create(full_dir, 4, 11)
            full.run(fields())
            first, _ = create(first_dir, 1, 11)
            first.run(fields())
            continued, continued_obs = create(next_dir, 4, 99)
            checkpoint = first_dir / "fields_000001.mat"
            continued.continue_run(checkpoint)
            assert full_obs.state_dict() == continued_obs.state_dict()
            assert continued_obs.count == 2
            assert full.random_bg.state == continued.random_bg.state
            compare_fields(loadmat(full_dir / "fields_000004.mat"), loadmat(next_dir / "fields_000004.mat"))
            if method == "heun":
                assert continued.integrator.force_evaluations == full.integrator.force_evaluations == 4
                bad, obs = create(root / "bad-state", 4, 99)
                data = loadmat(checkpoint, squeeze_me=True)
                state = json.loads(str(data["extension_state_json"]))
                state["integrator"]["force_evaluations"] = "invalid"
                data["extension_state_json"] = json.dumps(state)
                atomic_savemat(root / "bad-state.mat", {k: v for k, v in data.items() if not k.startswith("__")})
                before_rng = copy.deepcopy(bad.random_bg.state)
                expect_error(lambda: bad.continue_run(root / "bad-state.mat"), ValidationError)
                assert bad.random_bg.state == before_rng and obs.count == 0
                assert bad.integrator.force_evaluations == 0
            incompatible = cls(parameters(root / "incompatible"), random_seed=99)
            rng = copy.deepcopy(incompatible.random_bg.state)
            expect_error(lambda: incompatible.continue_run(checkpoint), ValidationError)
            assert incompatible.random_bg.state == rng

        # Version-1 LM checkpoints, written before extension support, still load.
        old = loadmat(root / (prefix + "baseline") / "fields_000001.mat", squeeze_me=True)
        settings = json.loads(str(old["checkpoint_settings_json"]))
        settings.pop("extensions")
        old.update(checkpoint_format_version=1, checkpoint_settings_json=settings_json(settings),
                   checkpoint_settings_hash=input_hash(settings))
        old.pop("extension_state_json")
        atomic_savemat(root / "version1.mat", {k: v for k, v in old.items() if not k.startswith("__")})
        legacy = cls(parameters(root / (prefix + "version1")), random_seed=99)
        legacy.continue_run(root / "version1.mat")
        compare_fields(loadmat(root / (prefix + "baseline") / "fields_000004.mat"),
                       loadmat(root / (prefix + "version1") / "fields_000004.mat"))

    # Failed proposals never reach observers; resampling rolls method state back.
    seen = []
    FailingLFTS.fail_calls = tuple(range(3, 30))
    failing = FailingLFTS(parameters(root / "failed"), random_seed=11,
                         callbacks={"seen": lambda sample: seen.append(sample["step"])})
    expect_error(lambda: failing.run(fields()), SaddlePointError)
    assert seen == [1]
    FailingLFTS.fail_calls = (3,)
    params = parameters(root / "resampled")
    params["saddle"]["failure_policy"] = "resample"
    method = example.PredictorCorrector()
    resampled = FailingLFTS(params, random_seed=11, integrator=method)
    resampled.run(fields())
    assert method.force_evaluations == 4 and resampled.run_status["noise_resamples"] == 1


def check_method_registry():
    registry = PropagatorSolver.numerical_methods()
    for platform in PlatformSelector.avail_platforms():
        for name, info in registry.items():
            assert platform in info["compiled_backends"]
            solver = PropagatorSolver(nx=[16], lx=[3.0], ds=0.05, bond_lengths={"A": 1.0},
                                      bc=["periodic"] * 2, chain_model=info["chain_model"],
                                      numerical_method=None if name == "discrete" else name, platform=platform)
            solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
            w = np.random.default_rng(3).normal(0, 5, 16)
            solver.compute_propagators({"A": w})
            assert solver.check_total_partition()
            solver.compute_concentrations()
            np.testing.assert_allclose(solver.get_concentration("A").mean(), 1, atol=1e-12)
            assert solver.method == info["solver_type"]
            if not info["supports_complex"]:
                expect_error(lambda: PropagatorSolver(nx=[16], lx=[3.0], ds=0.05, bond_lengths={"A": 1.0},
                    bc=["periodic"] * 2, chain_model="continuous", numerical_method=name,
                    platform=platform, field_type="complex"), ValueError)


class ResponseScaledEuler:
    """A stateful custom method whose persistent scale comes from linear response."""
    name = "test-response-scaled-euler"
    settings = {"version": 1}
    linear_response = True
    averaged_noise = False

    def __init__(self):
        self.response_scale = None
        self.steps = 0

    def set_linear_response(self, jacobian, *, dt, mobility, noise_direction, fourier):
        self.response_scale = 1.0 / (1.0 + dt * float(np.max(np.abs(jacobian))))

    def state_dict(self):
        assert self.response_scale is not None, "State was captured before preparing the response"
        return {"response_scale": self.response_scale, "steps": self.steps}

    def load_state_dict(self, state):
        scale, steps = state["response_scale"], state["steps"]
        if scale is not None and (not np.isfinite(scale) or scale <= 0):
            raise ValueError("Invalid response scale")
        if type(steps) is not int or steps < 0:
            raise ValueError("Invalid step count")
        self.response_scale, self.steps = scale, steps

    def step(self, fields, force, *, dt, mobility, noise, previous_noise,
             noise_direction, evaluate_force):
        assert self.response_scale is not None, "Prepared response was lost before the first step"
        effective = self.response_scale * mobility[:, None]
        self.steps += 1
        return fields - dt * effective * force + np.sqrt(effective) * noise_direction[:, None] * noise


def check_prepared_method_state(root):
    def create(name, steps, seed):
        params = parameters(root / name, steps)
        params["langevin"]["dt"] = 0.001
        original = ResponseScaledEuler()
        sim = CLFTS(params, random_seed=seed, integrator=original)
        sim.draw_architectures = False
        assert original.response_scale is None   # preparation belongs to the simulation's copy
        assert sim.integrator.response_scale is not None
        return sim

    full = create("prepared-full", 4, 11)
    assert full.run(fields())["status"] == "completed"
    first = create("prepared-first", 2, 11)
    first.run(fields())
    continued = create("prepared-next", 4, 99)
    continued.continue_run(root / "prepared-first" / "fields_000002.mat")
    assert full.integrator.state_dict() == continued.integrator.state_dict()
    assert full.integrator.steps == 4
    assert full.random_bg.state == continued.random_bg.state
    compare_fields(loadmat(root / "prepared-full" / "fields_000004.mat"),
                   loadmat(root / "prepared-next" / "fields_000004.mat"))
    # A new run resets evolving state but retains the prepared response.
    full.run(fields())
    assert full.integrator.steps == 4


def main():
    check_stochastic_moments()
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as directory:
        previous = Path.cwd()
        os.chdir(directory)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                check_observers_and_restart(Path(directory))
                check_method_registry()
                check_prepared_method_state(Path(directory))
        finally:
            os.chdir(previous)
    if cuda_part_missing():
        return 1
    print("Test passed, extensions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
