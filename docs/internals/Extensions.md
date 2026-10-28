# Adding observables, Langevin integrators and propagator methods

The Python run loops keep the physical sequence visible: evaluate the force,
integrate, resolve the saddle or concentrations, accept the step, update WTMD,
record observables, call observers, and checkpoint. The default integrator is
Leimkuhler–Matthews (LM). Existing simulation scripts keep working.

Start with [CustomExtensions.py](../../examples/tools/CustomExtensions.py), which
implements a composition observable and an additive-noise predictor–corrector
integrator (one explicit Heun correction).
Both work with `LFTS` and `CLFTS`. No subclass of either simulation is needed.

## Observables and progress reporting

Pass named callables at construction:

```python
def report(sample):
    print(sample["step"], sample["hamiltonian"], sample["step_seconds"])

sim = LFTS(params, random_seed=123, callbacks={"progress": report})
sim.run(initial_fields)
```

Each callback runs at its interval on accepted steps, after built-in recording
and WTMD updates, before the field checkpoint. Failed attempts, intermediate integrator
stages and the initial saddle solve do not call observers. There is no callback
invocation for the already-recorded step loaded by `continue_run`.

| Sample key | Meaning |
| --- | --- |
| `step`, `dt`, `time` | Global completed step, time step and `step * dt` |
| `w_aux`, `w` | Auxiliary and monomer fields, shape `(M, n_grid)` |
| `phi` | Dictionary of species concentrations |
| `hamiltonian` | Hamiltonian of the accepted fields, complex in CLFTS |
| `drift_max` | Maximum absolute scaled force at the start of the proposal |
| `error_level` | LFTS saddle error (scalar); CLFTS derivative standard deviations |
| `step_seconds` | Wall time through built-in recording, before observers/checkpoint |
| `status` | Copy of run status, including completed step and failure counters |

Mappings and arrays are read-only. Field arrays are detached copies; keeping a
sample for later analysis does not retain live solver buffers. Set an observer's
`interval` attribute to a positive integer to sample only at matching global steps
(default: 1). Snapshots are constructed only when at least one observer is due.
Callback return values are ignored. Exceptions propagate and mark the run failed; the next checkpoint is not written.

A callable object can accumulate statistics. Give it a JSON dictionary `settings`
for its scientific options. Its sampling `interval` is recorded separately.
Implement both `state_dict()` and `load_state_dict(state)` to save and restore its accumulator. The example's
`CompositionVariance` demonstrates this contract. Write custom result files from
the callback, or explicitly export its accumulated results after `run` returns.
External files are the observer's responsibility; include enough state to append
or reconstruct them on restart. A closure with hidden mutable state cannot have
that state restored automatically. Observers needing randomness should own and
checkpoint a separate RNG.

## Langevin integrators

Select a built-in method in the ordinary parameter dictionary:

```python
params["langevin"]["integrator"] = "euler"  # "lm" is the default
```

Or supply an object, which takes precedence over the parameter:

```python
sim = LFTS(params, integrator=PredictorCorrector(), callbacks={"variance": observable})
```

The actual LM and Euler equations are together in
[integrators.py](../../src/python/integrators.py). A custom method needs a stable
nonempty `name` and this method:

```python
def step(self, fields, force, *, dt, mobility, noise, previous_noise,
         noise_direction, evaluate_force):
    # Return new active fields of the same shape; see the predictor-corrector example.
    ...
```

`fields` and `force` have shape `(n_active, n_grid)`. LFTS passes real fluctuating
auxiliary fields; CLFTS passes all auxiliary fields as complex arrays. `force`
means the functional derivative dH/dw of the real-type auxiliary fields and -dH/dw
of the imaginary-type (pressure-like) ones of CLFTS, the sign convention of
`compute_func_deriv`, including current WTMD bias or dynamical stabilization. For
every field the deterministic increment is `-dt * mobility[:, None] * force`.
`mobility` is the per-field `dt_scaling` array. Fields with zero mobility must
remain fixed.

Returned fields are converted to the simulation's storage dtype (`float64` for
LFTS, `complex128` for CLFTS) and must remain finite at that precision.

`noise` is a real Gaussian array with the simulation's sigma, already including
the time-step dependence. Multiply by `sqrt(mobility)` and `noise_direction`
when constructing the stochastic increment. Noise direction is real in LFTS;
CLFTS uses real exchange-field noise and imaginary pressure-field noise. LM
averages `noise` and `previous_noise`; Euler uses only `noise`. At a fresh run,
previous noise is zero, preserving the existing LM initialization.

With `electrostatics`, the integrator does not receive the electrostatic potential ψ
(unless it declares `linear_response`, below).
CLFTS advances ψ after the auxiliary fields with its own exponential step, which takes
the noise scheme the integrator declares in `averaged_noise`: `True` averages two noises
from the ψ stream like `lm`, `False` gives fresh noise like `euler`. The built-in
integrators declare it; a custom integrator used with `electrostatics` must declare it,
or the construction raises `ValidationError`
([CL-FTS theory, Section 6](../theory/ComplexLangevinSimulation.md#6-charged-systems)).

An integrator that declares `linear_response = True` (CLFTS only, periodic boxes) receives
the Jacobian of the forces at the homogeneous state before the run:

```python
def set_linear_response(self, jacobian, *, dt, mobility, noise_direction, fourier):
    # jacobian[i, j] = dF_i(k)/dw_j(k), shape (m, m, *nx); mobility and noise_direction
    # are those that step() will receive; fourier is the simulation's PropagatorSolver.fourier
    ...
```

`fourier` transforms and multiplies whole fields with the FFT of the platform: matrices of
functions of k stored once with `set_multiplier` (on the GPU with CUDA) and applied with
`apply` ([Fourier transform example](../../examples/tools/FourierTransform.py)).

With Coulomb coupling, CLFTS then passes ψ to `step` as the last of the m fields: its force
is -δH/δψ, its mobility `psi_dt_scaling`, its noise imaginary and drawn from the ψ stream,
and `evaluate_force` takes and returns all m fields. Without screening, CLFTS sets the
uniform mode of the returned ψ to zero. Each simulation keeps its own copy (`copy.deepcopy`) of such
an integrator, which holds the response of its system; multipliers of `fourier` cannot be copied, so
an integrator that stores them defines `__deepcopy__` (the built-in one copies its settings without the response). The built-in `etd` integrator is such a method
([CL-FTS theory, Section 2.1](../theory/ComplexLangevinSimulation.md#21-exponential-time-differencing)).
L-FTS rejects these integrators.

The loop draws noise once per proposal and maintains previous noise after
acceptance. Intermediate stages reuse that noise. `evaluate_force(candidate)`
evaluates the force at new active fields: LFTS solves the corresponding imaginary
fields at their saddle, while CLFTS computes complex concentrations. Both apply
smearing and the current bias/stabilization. It does not sample observables,
advance time, draw noise, or update WTMD statistics. WTMD's bias potential remains
fixed throughout the proposal. Final fields are evaluated again by the run loop.

LFTS retries a failed final saddle at the same proposed fields. A failed
intermediate saddle is retried without drawing noise; exhaustion aborts the step.
The explicitly selected `resample` policy retries the whole proposal with new
noise, restoring integrator state first. This retains the existing warning about
noise selection. Failed proposals never reach observers. Custom integrators must
not hide extra randomness or catch saddle errors and accept an unchecked state.

Stateful methods use the same `settings`, `state_dict`, `load_state_dict` contract
as observers. Include all method parameters and a version in `settings`, and
every persistent method variable in the state dictionary. Settings and state
contain finite JSON values; convert NumPy arrays to lists and reconstruct them
explicitly. Loaders must validate values before accepting them. A fresh `run`
restores the extension states captured at construction; use `continue_run` for
continuation. `from_file` accepts the same extension keyword arguments.

## Restart and validation

Version-2 checkpoints contain extension identities, settings and accumulator
states as well as fields, RNG, previous noise and pending built-in measurements.
Recreate the same named callbacks and integrator when calling `continue_run`.
Changing a callback's name, callable identity, interval or settings, or changing an
integrator's name/settings, rejects the continuation before RNG restoration.
Code/source hashes in simulation metadata help identify changed implementations;
they do not replace the extension author's versioned compatibility contract.
Version-1 checkpoints continue with default LM and no callbacks. Unversioned
checkpoints require the existing explicit legacy opt-in and cannot restore custom
extensions. v0.9.1 checkpoints are unversioned; format version 1 was introduced
during development after that release. v0.9.1 does not restore the complete state
in version-2 checkpoints; resume them with v0.10.0.

`TestExtensions` checks exact linear stochastic moments, observer isolation,
multistage force calls, failed-proposal handling, and uninterrupted/split-run
equivalence including extension state. New scientific methods also need their
own time-step refinement and statistical validation; the runnable predictor–corrector example
does not certify sampling correctness for every physical model. See
[NumericalValidation.md](NumericalValidation.md).

## Propagator methods (MDE and discrete chains)

[NumericalMethods.h](../../src/common/NumericalMethods.h) is the common table of
names, chain models, solver families, contour order, complex-field support,
orthogonal-cell requirements and mixed-boundary support. Python reads it through
`PropagatorSolver.numerical_methods()`, which also reports compiled backends.
The discrete entry describes its separate bond-convolution solver; it has no
continuum contour convergence order. `numerical_method` remains ignored for
discrete chains.

To add a new native method:

1. Implement the algorithm in the CPU and CUDA solver families, including stress
   operations needed by its supported workflows.
2. Add its enum value and descriptor in `NumericalMethods.h`.
3. Add its constructor case in `CpuComputationBackend.cpp` and
   `CudaComputationBackend.cu`, alongside the existing constructors.
4. Add conservation, partition consistency, convergence-order and backend
   agreement checks appropriate to the algorithm, then run continuous-chain
   Gyroid as the integration check.

The three platform factories and Python method selection use the common table;
they need no new lists of method names. The table describes coarse capabilities;
geometry, grid-size and boundary restrictions specific to the numerical algorithm
must also be enforced in its solver. Keep descriptors accurate for both backend
implementations. Native methods require rebuilding the package; Python
observers and Langevin methods can be supplied directly from a research script.
