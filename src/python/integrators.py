"""Langevin updates, kept separate from sampling and checkpointing.

An integrator implements ``step(fields, force, *, dt, mobility, noise,
previous_noise, noise_direction, evaluate_force)`` and returns new active fields.
``force`` is dH/dw for real-type fields and -dH/dw for the imaginary-type fields of
CLFTS (the convention of compute_func_deriv), including bias/stabilization; the
deterministic increment is -dt * mobility * force for every field.
``evaluate_force(candidate)`` evaluates intermediate fields, solving the saddle
in LFTS. Noise is drawn by the run loop once per proposal; do not draw it here.
``averaged_noise`` declares the noise scheme: True when a step averages ``noise`` and
``previous_noise`` (Leimkuhler-Matthews), False for fresh noise. CLFTS with
electrostatics gives the electrostatic potential the same scheme and requires it.
An integrator that declares ``linear_response`` receives the Jacobian of the forces at the
homogeneous state through ``set_linear_response(jacobian, *, dt, mobility,
noise_direction, fourier)`` before the run (CLFTS only), with the simulation's
PropagatorSolver.fourier to apply it; CLFTS then also passes the electrostatic potential,
as the last field.
See docs/internals/Extensions.md for stateful methods and a two-stage example.
"""

import copy

import numpy as np

from .validation import ValidationError


class LeimkuhlerMatthews:
    name = "lm"
    settings = {}
    averaged_noise = True

    def step(self, fields, force, *, dt, mobility, noise, previous_noise,
             noise_direction, evaluate_force):
        updated = fields.copy()
        for i, scaling in enumerate(mobility):
            if scaling == 0:
                continue
            updated[i] += (-force[i] * dt * scaling +
                           0.5 * noise_direction[i] * (previous_noise[i] + noise[i]) * np.sqrt(scaling))
        return updated


class EulerMaruyama:
    name = "euler"
    settings = {}
    averaged_noise = False

    def step(self, fields, force, *, dt, mobility, noise, previous_noise,
             noise_direction, evaluate_force):
        updated = fields.copy()
        for i, scaling in enumerate(mobility):
            if scaling == 0:
                continue
            updated[i] += -force[i] * dt * scaling + noise_direction[i] * noise[i] * np.sqrt(scaling)
        return updated


class ExponentialTimeDifferencing:
    """Leimkuhler-Matthews step whose linear part is integrated exactly for each wavevector.

    Before the run the simulation measures the Jacobian J(k) = dF/dw of the forces at the
    homogeneous state and passes it to set_linear_response (CLFTS._force_jacobian). In the
    variables x = w/n, n the noise direction, the noise is real and the linear force is K x,
    K = N⁻¹ J N. With the mobilities S, K' = S^½ K S^½ and φ₁(z) = (1 - e^(-z))/z the step is

        x' = x - δt S^½ φ₁(K' δt) S^½ F(x) + S^½ φ₁(K' δt)^½ (ξ + ξ')/2,

    the Leimkuhler-Matthews step with the effective mobility δt S^½ φ₁(K' δt) S^½ in place of
    δt S. For a linear force its deterministic part is the exact flow, e^(-SK δt) x, and the
    stationary covariance is that of exp(-H) for any time step, as for the LM step of ψ
    (electrostatics.py), which this step contains for a field with K = (k² + κ²)/E. K couples
    all fields at each wavevector; it is complex symmetric where real-type and imaginary-type
    fields couple. The factor φ₁^½ is any B with B Bᵀ = φ₁ (Cholesky without conjugation),
    since only the covariance of the noise matters; where a pivot is small, the symmetric
    square root keeps B of the size of φ₁^½. Where K' has eigenvalues with a negative real
    part (the homogeneous state is unstable at that wavevector, as near an ordered phase),
    these eigenvalues are set to zero: their directions take the plain LM step, while the
    other directions keep the exact linear step. Periodic boxes only. The drift and noise matrices
    are stored as Fourier multipliers of the simulation's PropagatorSolver.fourier (on the GPU with
    CUDA), and a step is one apply() of both.
    """
    name = "etd"
    settings = {"version": 1}
    averaged_noise = True
    linear_response = True

    def __init__(self):
        self._prepared = None

    def __deepcopy__(self, memo):
        # A copy keeps its settings but not the linear response: the multipliers belong to the
        # transform of one simulation, and every simulation sets its own (CLFTS copies the integrator)
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            setattr(copied, key, None if key == "_prepared" else copy.deepcopy(value, memo))
        return copied

    def set_linear_response(self, jacobian, *, dt, mobility, noise_direction, fourier):
        """Store the drift and noise matrices of each wavevector as multipliers of fourier.

        jacobian has the shape (m, m, *nx): J_ij(k) = dF_i(k)/dw_j(k) for the m integrated
        fields; mobility and noise_direction are those the step will receive. fourier is the
        PropagatorSolver.fourier of complex fields in the periodic box with the grid nx.
        """
        jacobian = np.asarray(jacobian, dtype=np.complex128)
        drift, noise, summary = self._matrices(jacobian, dt=dt, mobility=mobility, noise_direction=noise_direction)
        m, nx = jacobian.shape[0], tuple(jacobian.shape[2:])
        if fourier.field_type != "complex" or not fourier.periodic or tuple(fourier.coefficient_shape) != nx:
            raise ValidationError(f"The linear response needs the Fourier transform of complex fields in a periodic box "
                                  f"on the grid of the Jacobian {nx}, got one of {fourier.field_type} fields with "
                                  f"coefficients of shape {tuple(fourier.coefficient_shape)}"
                                  + ("" if fourier.periodic else " and walls"))
        # step adds backward(noise · forward(ξ̄) - drift · forward(F)); the multipliers hold the
        # only copy of the matrices
        self._prepared = {"nx": nx, "dt": float(dt), "mobility": np.asarray(mobility, dtype=np.float64).copy(),
                          "direction": np.asarray(noise_direction, dtype=np.complex128).copy(), "fourier": fourier,
                          "drift_multiplier": fourier.set_multiplier(-drift.reshape((m, m) + nx)),
                          "noise_multiplier": fourier.set_multiplier(noise.reshape((m, m) + nx))}
        print(summary)

    def _matrices(self, jacobian, *, dt, mobility, noise_direction):
        """Drift δt N S^½ φ₁ S^½ N⁻¹ and noise N S^½ B of each wavevector, both (m, m, n_k), and a summary."""
        import scipy.linalg
        jacobian = np.asarray(jacobian, dtype=np.complex128)
        m = jacobian.shape[0]
        mobility = np.asarray(mobility, dtype=np.float64)
        direction = np.asarray(noise_direction, dtype=np.complex128)
        if jacobian.ndim < 3 or jacobian.shape[1] != m or mobility.shape != (m,) or direction.shape != (m,) or np.any(mobility < 0):
            raise ValidationError("The linear response needs an (m, m, *nx) Jacobian and m mobilities and noise directions")
        J = np.moveaxis(jacobian.reshape(m, m, -1), 2, 0)              # (n_k, m, m)
        root = np.sqrt(mobility)
        # K' = S^½ N⁻¹ J N S^½
        K = root[:, None] * (J * direction[None, :] / direction[:, None]) * root[None, :]

        # K' is complex symmetric (the second derivatives of H); an asymmetric Jacobian means an
        # inconsistent measurement, and the factorizations below would not match the drift
        transpose = np.swapaxes(K, 1, 2)
        asymmetry = np.max(np.abs(K - transpose)) / max(np.max(np.abs(K)), 1e-300)
        if asymmetry > 1e-5:
            raise ValidationError(f"The Jacobian of the forces is not symmetric (relative asymmetry {asymmetry:.1e})")
        K = 0.5 * (K + transpose)

        # Eigenvalues with a negative real part (unstable homogeneous state) are set to zero:
        # those directions take the plain LM step (φ₁ = 1), the others keep the exact step. A
        # wavevector whose eigenvectors are ill-conditioned takes the plain LM step altogether
        eigenvalues = np.linalg.eigvals(K)
        scale = np.max(np.abs(eigenvalues), axis=1, keepdims=True)
        unstable = np.flatnonzero(np.any(eigenvalues.real < -1e-12 * scale, axis=1))
        n_directions, n_plain = 0, 0
        if unstable.size:
            values, vectors = np.linalg.eig(K[unstable])
            negative = values.real < -1e-12 * np.max(np.abs(values), axis=1, keepdims=True)
            n_directions = int(np.sum(negative))
            values[negative] = 0.0
            clamped = np.einsum("kij,kj,kjl->kil", vectors, values, np.linalg.inv(vectors))
            well = np.linalg.cond(vectors) < 1e8
            clamped[~well] = 0.0
            n_plain = int(np.sum(~well))
            K[unstable] = 0.5 * (clamped + np.swapaxes(clamped, 1, 2))

        # φ₁(K' δt): upper right block of exp([[-K' δt, I], [0, 0]]); chunks bound the memory
        phi1 = np.empty_like(K)
        identity = np.eye(m)
        for start in range(0, K.shape[0], 8192):
            block = K[start:start + 8192]
            augmented = np.zeros((block.shape[0], 2 * m, 2 * m), dtype=np.complex128)
            augmented[:, :m, :m] = -dt * block
            augmented[:, :m, m:] = identity
            phi1[start:start + 8192] = scipy.linalg.expm(augmented)[:, :m, m:]

        # B with B Bᵀ = φ₁: Cholesky without conjugation (φ₁ is complex symmetric). A small pivot
        # makes the Cholesky factor much larger than φ₁^½: the noise would keep its covariance
        # but get huge real and imaginary parts, so such a wavevector takes the symmetric square
        # root of φ₁
        factor = np.zeros_like(phi1)
        pivots = np.zeros(phi1.shape[:2], dtype=np.complex128)
        for j in range(m):
            pivots[:, j] = phi1[:, j, j] - np.sum(factor[:, j, :j] ** 2, axis=1)
            factor[:, j, j] = np.sqrt(pivots[:, j])
            for i in range(j + 1, m):
                with np.errstate(divide="ignore", invalid="ignore"):
                    factor[:, i, j] = (phi1[:, i, j] - np.sum(factor[:, i, :j] * factor[:, j, :j], axis=1)) / factor[:, j, j]
        size = np.max(np.abs(phi1), axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            bad = (~np.all(np.isfinite(factor), axis=(1, 2))
                   | np.any(np.abs(pivots) < 1e-6 * np.abs(np.diagonal(phi1, axis1=1, axis2=2)), axis=1)
                   | (np.max(np.abs(factor), axis=(1, 2)) > 10 * np.sqrt(size)))
        for k in np.flatnonzero(bad):
            factor[k] = scipy.linalg.sqrtm(phi1[k])

        # Back to the fields w = N x: drift δt N S^½ φ₁ S^½ N⁻¹ and noise N S^½ B
        drift = dt * direction[:, None] * root[:, None] * phi1 * root[None, :] / direction[None, :]
        noise = direction[:, None] * root[:, None] * factor
        if not (np.all(np.isfinite(drift)) and np.all(np.isfinite(noise))):
            raise ValidationError("The linear response of the homogeneous state gives non-finite ETD matrices")
        summary = (f"ETD integrator: linear response of {m} fields at {K.shape[0]} wavevectors; "
                   f"{unstable.size} with an unstable homogeneous state, whose {n_directions} unstable directions "
                   f"take plain LM steps ({n_plain} wavevectors altogether)")
        return np.ascontiguousarray(np.moveaxis(drift, 0, 2)), np.ascontiguousarray(np.moveaxis(noise, 0, 2)), summary

    def step(self, fields, force, *, dt, mobility, noise, previous_noise,
             noise_direction, evaluate_force):
        prepared = self._prepared
        if prepared is None:
            raise ValidationError("The 'etd' integrator needs the linear response of the simulation "
                                  "(set_linear_response); CLFTS provides it")
        if (dt != prepared["dt"] or not np.array_equal(mobility, prepared["mobility"])
                or not np.array_equal(np.asarray(noise_direction, dtype=np.complex128), prepared["direction"])):
            raise ValidationError("The 'etd' step received another time step, mobility or noise direction "
                                  "than its linear response")
        mean_noise = 0.5 * (np.asarray(noise, dtype=np.float64) + previous_noise)
        increment = prepared["fourier"].apply([(prepared["noise_multiplier"], mean_noise),
                                               (prepared["drift_multiplier"], force)])
        return fields + increment


def make_integrator(method):
    if isinstance(method, str):
        methods = {"lm": LeimkuhlerMatthews, "euler": EulerMaruyama, "etd": ExponentialTimeDifferencing}
        if method.lower() not in methods:
            raise ValidationError(f"Unknown Langevin integrator '{method}'; use {tuple(methods)} or pass an integrator object")
        method = methods[method.lower()]()
    if not isinstance(getattr(method, "name", None), str) or not method.name or not callable(getattr(method, "step", None)):
        raise ValidationError("An integrator needs a nonempty 'name' and a 'step' method")
    return method


def checked_fields(candidate, reference):
    """Catch a broken extension before it reaches native array operations."""
    candidate = np.asarray(candidate)
    if candidate.shape != reference.shape or candidate.dtype.kind not in "fciu":
        raise ValidationError("Integrator fields must have the original shape and a numeric dtype")
    if not np.iscomplexobj(reference) and np.iscomplexobj(candidate):
        raise ValidationError("An LFTS integrator must return real active fields")
    # CLFTS replaces the entire field array. Keep its complex128 storage even
    # when a method returns a real list or a lower-precision work array.
    with np.errstate(over="ignore", invalid="ignore"):
        candidate = np.asarray(candidate, dtype=reference.dtype)
    if not np.all(np.isfinite(candidate)):
        raise ValidationError("Integrator fields must be finite at the simulation precision")
    return candidate
