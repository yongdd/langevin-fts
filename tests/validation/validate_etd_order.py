#!/usr/bin/env python3
"""Time-step error of stationary averages for the Langevin integrators (one configuration per invocation).

Overdamped Langevin dynamics dx = -F(x) dt + sqrt(2) dW, with the force of
- U = a x^2/2 + b ln cosh x (--model tanh, b = 2): F = a x + b tanh x is bounded beyond its
  linear part, so no step size makes single trajectories run away, and a sets the stiffness;
- U = x^2/2 + x^4/4 (--model quartic): a stronger nonlinearity, whose larger second-order
  errors resolve the order of LM and etd (F grows as x^3, so large steps let rare trajectories
  run away).
The stationary density exp(-U) gives <x^2> by quadrature. Many
independent chains are the grid points of one field, so the integrators of the package advance
them as they advance an auxiliary field:
- euler: EulerMaruyama;
- lm: LeimkuhlerMatthews;
- etd: ExponentialTimeDifferencing with the Jacobian of the force at x = 0 (a + b, or 1 for the
  quartic model), as CLFTS measures it at the homogeneous state;
- etd1: exponential time differencing with fresh noise of the exact Ornstein-Uhlenbeck variance
  (first-order ETD), written out here for comparison.
The error of <x^2> and its standard error over 20 groups of chains are printed as one JSON line.
Submit each configuration as a separate job, e.g.

    python tests/validation/validate_etd_order.py --scheme etd --dt 0.1 --a 1
    python tests/validation/validate_etd_order.py --model quartic --scheme lm --dt 0.05
    python tests/validation/validate_etd_order.py --collect results.jsonl
"""
import argparse
import contextlib
import io
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.integrate import quad
from polymerfts.integrators import EulerMaruyama, ExponentialTimeDifferencing, LeimkuhlerMatthews

B = 2.0          # strength of the bounded nonlinear part b tanh(x)
GROUPS = 20      # groups of chains for the standard error


def model(name, a):
    """Potential U, force F and its derivative at x = 0."""
    if name == "quartic":
        return (lambda x: x ** 2 / 2 + x ** 4 / 4), (lambda x: x + x ** 3), 1.0
    return ((lambda x: a * x ** 2 / 2 + B * (np.logaddexp(x, -x) - np.log(2.0))),   # ln cosh x
            (lambda x: a * x + B * np.tanh(x)), a + B)


def exact_second_moment(name, a):
    weight = lambda x: np.exp(-model(name, a)[0](x))
    return quad(lambda x: x ** 2 * weight(x), -np.inf, np.inf)[0] / quad(weight, -np.inf, np.inf)[0]


def run(name, scheme, dt, a, chains, relaxation_time, sampling_time, seed):
    rng = np.random.default_rng(seed)
    _, force, jacobian = model(name, a)
    x = np.zeros((1, chains))
    previous = np.zeros((1, chains))
    mobility, direction = np.array([1.0]), np.array([1.0])
    integrator = {"euler": EulerMaruyama, "lm": LeimkuhlerMatthews, "etd": ExponentialTimeDifferencing}.get(scheme)
    integrator = integrator() if integrator is not None else None
    if scheme == "etd":
        with contextlib.redirect_stdout(io.StringIO()):
            integrator.set_linear_response(np.full((1, 1, chains), jacobian), dt=dt, mobility=mobility,
                                           noise_direction=direction)
    decay = np.exp(-jacobian * dt)
    sigma = np.sqrt(2 * dt)    # unit temperature: the noise of one step has the variance 2 dt
    relax, steps = int(np.ceil(relaxation_time / dt)), int(np.ceil(sampling_time / dt))
    total = np.zeros(chains)
    for n in range(relax + steps):
        noise = rng.normal(0.0, sigma, (1, chains))
        if scheme == "etd1":
            x = (decay * x - (1 - decay) / jacobian * (force(x) - jacobian * x)
                 + np.sqrt((1 - decay ** 2) / jacobian) * noise / sigma)
        else:
            x = integrator.step(x, force(x), dt=dt, mobility=mobility, noise=noise, previous_noise=previous,
                                noise_direction=direction, evaluate_force=None)
            x = np.real(x)
        previous = noise
        if not np.all(np.isfinite(x)):
            return {"diverged": True, "step": n}
        if n >= relax:
            total += x[0] ** 2
    group_means = (total / steps).reshape(GROUPS, -1).mean(axis=1)
    return {"diverged": False, "mean": float(group_means.mean()),
            "error": float(group_means.std(ddof=1) / np.sqrt(GROUPS))}


def collect(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    for name, a in sorted({(r.get("model", "tanh"), r["a"]) for r in rows}):
        group = [r for r in rows if r.get("model", "tanh") == name and r["a"] == a]
        print(f"\n{name}, a = {a}: exact <x^2> = {exact_second_moment(name, a):.6f}")
        dts = sorted({r["dt"] for r in group})
        print("scheme  " + "  ".join(f"dt = {dt:<16g}" for dt in dts))
        for scheme in ("euler", "etd1", "lm", "etd"):
            cells = []
            for dt in dts:
                r = next((r for r in group if r["dt"] == dt and r["scheme"] == scheme), None)
                if r is None:
                    cells.append(f"{'-':<21}")
                elif r["diverged"]:
                    cells.append(f"{'diverges':<21}")
                else:
                    cells.append(f"{r['bias']:+.2e} ± {r['error']:.1e}".ljust(21))
            print(f"{scheme:7s} " + " ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("tanh", "quartic"), default="tanh")
    parser.add_argument("--scheme", choices=("euler", "lm", "etd", "etd1"))
    parser.add_argument("--dt", type=float)
    parser.add_argument("--a", type=float, default=1.0, help="linear force constant (stiffness)")
    parser.add_argument("--chains", type=int, default=200000)
    parser.add_argument("--relaxation-time", type=float, default=20.0)
    parser.add_argument("--sampling-time", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collect", help="print the table of a file of JSON lines")
    args = parser.parse_args()
    if args.collect:
        collect(args.collect)
        return
    if args.scheme is None or args.dt is None or args.chains % GROUPS:
        parser.error("--scheme and --dt are required, and --chains must be a multiple of 20")
    result = run(args.model, args.scheme, args.dt, args.a, args.chains, args.relaxation_time, args.sampling_time,
                 args.seed)
    result.update(model=args.model, scheme=args.scheme, dt=args.dt, a=args.a, b=B, chains=args.chains,
                  sampling_time=args.sampling_time, exact=exact_second_moment(args.model, args.a))
    if not result["diverged"]:
        result["bias"] = result["mean"] - result["exact"]
    print(json.dumps(result))


if __name__ == "__main__":
    main()
