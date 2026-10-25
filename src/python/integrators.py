"""Langevin updates, kept separate from sampling and checkpointing.

An integrator implements ``step(fields, force, *, dt, mobility, noise,
previous_noise, noise_direction, evaluate_force)`` and returns new active fields.
``force`` is dH/dw (including bias/stabilization), not the signed drift.
``evaluate_force(candidate)`` evaluates intermediate fields, solving the saddle
in LFTS. Noise is drawn by the run loop once per proposal; do not draw it here.
``averaged_noise`` declares the noise scheme: True when a step averages ``noise`` and
``previous_noise`` (Leimkuhler-Matthews), False for fresh noise. CLFTS with
electrostatics gives the electrostatic potential the same scheme and requires it.
See docs/internals/Extensions.md for stateful methods and a two-stage example.
"""
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


def make_integrator(method):
    if isinstance(method, str):
        methods = {"lm": LeimkuhlerMatthews, "euler": EulerMaruyama}
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
