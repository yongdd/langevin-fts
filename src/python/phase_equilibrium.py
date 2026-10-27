"""Bulk phase coexistence from canonical SCFT solutions.

free_energy is relative to the homogeneous state; free_energy_absolute includes
the full CE terms. Both use units of rho_0 V / N; chemical potentials use beta mu
per molecule. All SCFT evaluations remain canonical. See docs/theory/PhaseEquilibrium.md.
"""

import numpy as np
from scipy.optimize import least_squares


def _value(state, name, default=None):
    return state.get(name, default) if isinstance(state, dict) else getattr(state, name, default)


def coexistence_quantities(state, reference_polymer=0):
    """Convert a converged CE state to pressure-independent coexistence quantities.

    Accept an SCFTResult or a dictionary from homogeneous_thermodynamics().
    Return g_p = beta mu_p / alpha_p - beta mu_r / alpha_r (including g_r = 0)
    and Psi = free_energy_absolute - sum_p g_p phi_p. Psi is the grand potential
    in the pressure convention beta mu_r = 0. Pure endpoints have infinite exchange potentials
    and cannot be used here. This conversion is for incompressible periodic
    bulk systems with free chains.
    """
    if not _value(state, "converged", False):
        raise ValueError("Phase coexistence requires a converged CE solution.")
    params = _value(state, "params", {})
    if params.get("zeta_n") is not None:
        raise ValueError("Bulk coexistence currently requires the incompressible model.")
    if params.get("bc") is not None and any(str(bc).lower() != "periodic" for bc in params["bc"]):
        raise ValueError("Bulk coexistence requires periodic boundary conditions.")
    mu = _value(state, "chemical_potentials")
    if mu is None:
        raise ValueError("Chemical potentials are unavailable for this CE state (e.g. grafted chains).")
    alpha = np.asarray(_value(state, "polymer_lengths"), dtype=float)
    phi = np.asarray(_value(state, "polymer_volume_fractions"), dtype=float)
    mu = np.asarray(mu, dtype=float)
    if alpha.ndim != 1 or alpha.size == 0 or mu.shape != alpha.shape or phi.shape != alpha.shape:
        raise ValueError("Chemical potentials, polymer lengths and volume fractions must have equal lengths.")
    if not (np.all(np.isfinite(alpha)) and np.all(alpha > 0)
            and np.all(np.isfinite(phi)) and np.all(phi > 0)
            and np.isclose(phi.sum(), 1.0, rtol=0, atol=1e-10)
            and np.all(np.isfinite(mu))):
        raise ValueError("Coexistence needs positive finite chain lengths and compositions, sum(phi) = 1, and finite mu.")
    if (not isinstance(reference_polymer, (int, np.integer))
            or not 0 <= reference_polymer < alpha.size):
        raise ValueError("reference_polymer must be an index of a polymer species.")
    absolute = _value(state, "free_energy_absolute")
    if absolute is None or not np.isfinite(absolute):
        raise ValueError("A finite full CE free energy is required: supply free_energy_absolute.")
    absolute = float(absolute)
    exchange = mu / alpha - mu[reference_polymer] / alpha[reference_polymer]
    return {
        "reference_polymer": int(reference_polymer),
        "exchange_chemical_potentials": exchange.tolist(),
        "reduced_grand_potential": float(absolute - exchange @ phi),
        "free_energy": float(_value(state, "free_energy")),
        "free_energy_absolute": absolute,
        "polymer_lengths": alpha.tolist(),
        "polymer_volume_fractions": phi.tolist(),
    }


def homogeneous_thermodynamics(volume_fractions, polymer_lengths, monomer_fractions, interaction_matrix):
    """Return an analytic homogeneous CE state, without solving propagators.

    monomer_fractions[p, i] is the fraction of monomer i in polymer p (also for
    random copolymers); interaction_matrix[i, j] is chi_ij N. Rows of
    monomer_fractions and volume_fractions each sum to one. Zero polymer volume
    fractions contribute zero to F and have beta mu = -inf. All molecules are
    free chains and the system is incompressible. free_energy is zero and
    free_energy_absolute equals free_energy_homogeneous.
    """
    phi = np.asarray(volume_fractions, dtype=float)
    alpha = np.asarray(polymer_lengths, dtype=float)
    fractions = np.asarray(monomer_fractions, dtype=float)
    chi = np.asarray(interaction_matrix, dtype=float)
    if (phi.ndim != 1 or phi.size == 0 or alpha.shape != phi.shape or fractions.ndim != 2
            or fractions.shape[0] != phi.size or fractions.shape[1] == 0
            or chi.shape != (fractions.shape[1], fractions.shape[1])):
        raise ValueError("Expected phi[p], alpha[p], monomer_fractions[p,i] and interaction_matrix[i,j].")
    if (not all(np.all(np.isfinite(a)) for a in (phi, alpha, fractions, chi))
            or np.any(phi < 0) or np.any(alpha <= 0) or np.any(fractions < 0)
            or not np.isclose(phi.sum(), 1.0, rtol=0, atol=1e-10)
            or not np.allclose(fractions.sum(axis=1), 1.0, rtol=0, atol=1e-10)
            or not np.allclose(chi, chi.T, rtol=0, atol=1e-12)
            or not np.allclose(np.diag(chi), 0.0, rtol=0, atol=1e-12)):
        raise ValueError("Invalid homogeneous composition, chain lengths or symmetric chi N matrix.")
    density = phi / alpha
    present = density > 0
    monomers = phi @ fractions
    ideal = float(np.sum(density[present] * (np.log(density[present]) - 1.0)))
    interaction = 0.5 * float(monomers @ chi @ monomers)
    mu = np.full(phi.shape, -np.inf)
    mu[present] = np.log(density[present]) + alpha[present] * (fractions[present] @ chi @ monomers)
    return {
        "converged": True,
        "free_energy": 0.0,
        "free_energy_absolute": ideal + interaction,
        "free_energy_homogeneous": ideal + interaction,
        "chemical_potentials": mu.tolist(),
        "grand_potential": float(-density.sum() - interaction),
        "polymer_lengths": alpha.tolist(),
        "polymer_volume_fractions": phi.tolist(),
    }


def find_binary_coexistence(phase_a, phase_b, initial_compositions, *,
                            composition_bounds=((1e-8, 1-1e-8), (1e-8, 1-1e-8)),
                            reference_polymer=0, tolerance=1e-8, max_evaluations=100,
                            min_separation=1e-6):
    """Find a common tangent using two canonical phase evaluators.

    phase_a(x), phase_b(x) must return a converged SCFTResult or a homogeneous
    state for polymer fractions [1-x, x], with the same species order, chain
    lengths and physical model. Each callback must follow its intended phase
    branch; it may reuse previous fields and box parameters as initial guesses.
    Scalar thermodynamic evaluations are cached; field arrays are not retained.

    initial_compositions gives (x_a, x_b). Narrow composition_bounds to the two
    intended branches using a preliminary CE scan. A bounded least-squares solve
    matches g and Psi. Failed SCFT evaluations propagate as errors; small optimizer
    steps alone do not count as coexistence. Coincident compositions (critical or
    congruent transitions) are outside this solver's scope. Stability against
    other phases is a separate comparison with their CE free energies.

    Return a dictionary containing compositions, the common exchange chemical
    potential (of the non-reference species), reduced_grand_potential, the two
    residuals, per-phase evaluation counts, and scalar phase summaries.
    """
    initial = np.asarray(initial_compositions, dtype=float)
    bounds = np.asarray(composition_bounds, dtype=float)
    if (initial.shape != (2,) or bounds.shape != (2, 2)
            or not np.all(np.isfinite(initial)) or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 0] <= 0) or np.any(bounds[:, 1] >= 1)
            or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.any(initial <= bounds[:, 0]) or np.any(initial >= bounds[:, 1])):
        raise ValueError("Give two interior initial compositions and two bounds with 0 < lower < x < upper < 1.")
    if not isinstance(reference_polymer, (int, np.integer)) or reference_polymer not in (0, 1):
        raise ValueError("A binary reference_polymer must be 0 or 1.")
    if (not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(min_separation)
            or min_separation <= 0 or not isinstance(max_evaluations, (int, np.integer))
            or max_evaluations <= 0):
        raise ValueError("tolerance, min_separation and max_evaluations must be positive.")
    cache = [{}, {}]
    lengths = None

    def evaluate(phase, x):
        nonlocal lengths
        key = float(x)
        if key not in cache[phase]:
            state = (phase_a, phase_b)[phase](key)
            quantities = coexistence_quantities(state, reference_polymer)
            alpha = np.asarray(quantities["polymer_lengths"])
            if alpha.shape != (2,) or not np.allclose(quantities["polymer_volume_fractions"],
                                                     [1-key, key], rtol=0, atol=1e-10):
                raise ValueError("Each phase callback must evaluate polymer fractions [1-x, x].")
            if lengths is None:
                lengths = alpha.copy()
            elif not np.allclose(lengths, alpha, rtol=1e-12, atol=0):
                raise ValueError("Phase callbacks must use the same polymer lengths and species order.")
            cache[phase][key] = quantities
        return cache[phase][key]

    def residual(x):
        a, b = evaluate(0, x[0]), evaluate(1, x[1])
        return np.array([a["exchange_chemical_potentials"][1-reference_polymer]
                         - b["exchange_chemical_potentials"][1-reference_polymer],
                         a["reduced_grand_potential"] - b["reduced_grand_potential"]])

    optimizer_tolerance = max(tolerance * 0.01, 10*np.finfo(float).eps)
    solution = least_squares(residual, initial, bounds=(bounds[:, 0], bounds[:, 1]),
                             ftol=optimizer_tolerance, xtol=optimizer_tolerance,
                             gtol=optimizer_tolerance, max_nfev=max_evaluations)
    errors = residual(solution.x)
    if not solution.success or np.max(np.abs(errors)) > tolerance:
        raise RuntimeError("CE coexistence did not converge: " + solution.message
                           + "; residual = " + str(errors.tolist()))
    if abs(solution.x[0] - solution.x[1]) < min_separation:
        raise RuntimeError("Coincident compositions: bracket distinct phase branches; critical/congruent transitions need a separate treatment.")
    a, b = evaluate(0, solution.x[0]), evaluate(1, solution.x[1])
    return {
        "compositions": solution.x.tolist(),
        "reference_polymer": int(reference_polymer),
        "exchange_chemical_potential": 0.5*(a["exchange_chemical_potentials"][1-reference_polymer]
                                             + b["exchange_chemical_potentials"][1-reference_polymer]),
        "reduced_grand_potential": 0.5*(a["reduced_grand_potential"] + b["reduced_grand_potential"]),
        "residuals": errors.tolist(),
        "evaluations": [len(c) for c in cache],
        "phases": [a, b],
    }
