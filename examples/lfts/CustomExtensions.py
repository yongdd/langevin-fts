"""A stateful observable and a multistage integrator, without editing LFTS.

Run from a scratch directory with the current polymerfts package on PYTHONPATH.
Predictor-corrector is an example extension; validate time-step bias for your model.
"""
import numpy as np
from polymerfts import LFTS


class PredictorCorrector:
    """Additive-noise predictor-corrector with one explicit Heun correction."""

    # Preserve the identifier in existing example checkpoints.
    name = "example-heun"
    settings = {"version": 1}

    def __init__(self):
        self.force_evaluations = 0

    def step(self, fields, force, *, dt, mobility, noise, previous_noise,
             noise_direction, evaluate_force):
        mobility = mobility[:, None]
        kick = noise_direction[:, None] * noise * np.sqrt(mobility)
        predictor = fields - dt * mobility * force + kick
        predicted_force = evaluate_force(predictor)
        self.force_evaluations += 1
        return fields - 0.5 * dt * mobility * (force + predicted_force) + kick

    def state_dict(self):
        return {"force_evaluations": self.force_evaluations}

    def load_state_dict(self, state):
        count = state["force_evaluations"]
        if type(count) is not int or count < 0:
            raise ValueError("force_evaluations must be a nonnegative integer")
        self.force_evaluations = count


# Compatibility with scripts that imported the example under its earlier name.
StochasticHeun = PredictorCorrector


class CompositionVariance:
    def __init__(self, every=1):
        self.interval = every
        self.settings = {"version": 1}
        self.count = 0
        self.total = 0.0

    def __call__(self, sample):
        difference = sample["phi"]["A"] - sample["phi"]["B"]
        self.total += float(np.mean(np.abs(difference - difference.mean()) ** 2))
        self.count += 1

    def state_dict(self):
        return {"count": self.count, "total": self.total}

    def load_state_dict(self, state):
        if type(state["count"]) is not int or state["count"] < 0 or not np.isfinite(state["total"]):
            raise ValueError("Invalid composition-variance accumulator")
        self.count, self.total = state["count"], state["total"]


def main():
    params = {
        "nx": [16], "lx": [3.0], "ds": 0.05, "chain_model": "continuous",
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0},
        "polymers": [{"volume_fraction": 1.0, "blocks": [
            {"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "langevin": {"nbar": 10000, "dt": 0.1, "max_step": 10},
        "compressor": {"name": "am", "max_hist": 20, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01},
        "saddle": {"max_iter": 200, "tolerance": 1e-6},
        "recording": {"dir": "custom_extensions", "checkpoint_interval": 5,
                      "sample_interval": 1, "statistics_save_interval": 5},
        "verbose_level": 0,
    }
    observable = CompositionVariance(every=2)
    sim = LFTS(params, random_seed=11, integrator=PredictorCorrector(),
               callbacks={"composition_variance": observable})
    rng = np.random.default_rng(3)
    sim.run({m: rng.normal(0.0, 5.0, 16) for m in ("A", "B")})
    print("Mean composition variance:", observable.total / observable.count)


if __name__ == "__main__":
    main()
