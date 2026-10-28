"""Smeared Coulomb interactions for complex Langevin field-theoretic simulations.

Monomer type i carries the valence z_i per segment, and the charges interact through
the smeared charge density c(r) = Σ_i z_i (Γ_i * φ_i)(r) with the Coulomb kernel E/k²,
where Γ_i is the smearing function of the type (smearing.py). A background electrolyte
that is not simulated explicitly screens the kernel to E/(k² + κ²) (a Yukawa interaction),
with the screening length 1/κ (screening_length); it is the Debye length of that background
alone, and explicit ions screen in addition through ψ.
The interaction is decoupled with an electrostatic potential ψ on the rotated contour:
like the pressure field it is an imaginary-type field, and type i feels z_i ψ, so the
propagator input is Γ_i * (w_i + z_i ψ). Per reference chain (as in polymer_field_theory.py),

    βH_el / (C V/R₀³) = (1/(2E V)) ∫ ψ (∇² - κ²) ψ dr,   δH/δψ = (∇² - κ²)ψ/E + c,

with the coupling E = 4π l_B ρ₀ N R₀² = 4π (l_B/R₀) N² N̄^(1/2). At the saddle point
-(∇² - κ²)ψ = E c (Poisson or screened Poisson equation). ψ moves like an imaginary field
(drift +δH/δψ, imaginary noise). Its linear part (∇² - κ²)ψ/E is stiff at large k and is
integrated exactly for each wavevector (exponential time differencing); the charge density
stays explicit. ψ takes the mean of the current and the previous noise, as the
Leimkuhler-Matthews integrator of the auxiliary fields does: the step is a
Leimkuhler-Matthews step of ψ with the time step (1 - exp(-γ δτN))/γ of each mode,
γ = s (k² + κ²)/E, and the Gaussian statistics of ψ and the auxiliary fields are exact for
any time step. With an integrator that declares fresh noise (averaged_noise = False, as
Euler does), ψ takes fresh noise; the explicit charge density then biases its variance at
first order in γ δτN. An integrator that declares linear_response (integrators.py) advances
ψ together with the auxiliary fields instead (draw_noise, accept).
ψ is kept in Fourier space; its grid values follow from one inverse transform per step,
and H is evaluated by Parseval's theorem. The transforms are those of PropagatorSolver.fourier
(on the GPU with CUDA).
Without screening the uniform mode of ψ is zero (electroneutrality); with screening it
fluctuates, and the system may carry a net charge. A fixed charge density (fixed_charge) adds
to c. Periodic boxes, or closed boxes with walls on every side: the transforms are then the
cosine series, so ψ has no normal derivative at the walls (no field through them).

Point ions (`ions`) have no volume and no χ: ion s with valence z_s and Gaussian smearing
length a_s feels its smeared potential z_s (Γ_s * ψ)/N, where a segment of the reference chain
feels z_i ψ/N (per-reference-chain fields, N segments). With the concentration c_s (ions per
segment of the reference density), Q_s = <exp(-z_s (Γ_s * ψ)/N)>, the ion density is
c_s exp(-z_s (Γ_s * ψ)/N)/Q_s, H gains -N c_s ln Q_s and the charge density z_s Γ_s * c_s(r).
A grand-canonical ion (chemical_potential βμ_s, fugacity e^(βμ_s)) has the density
e^(βμ_s) exp(-z_s (Γ_s * ψ)/N)/N and adds -e^(βμ_s) Q_s to H. With charged grand-canonical species
and no screening, the uniform mode of ψ is integrated as well: it is the Lagrange multiplier
(Donnan potential) of electroneutrality.

SCFT takes the saddle point of the same Hamiltonian (MeanFieldElectrostatics): ψ is real there
and solves the Poisson equation with the charges of the chains and the point ions.

References: Riggleman, Kumar & Fredrickson, J. Chem. Phys. 136, 024903 (2012);
Delaney & Fredrickson, J. Phys. Chem. B 120, 7615 (2016).
"""

import contextlib
import hashlib
import json

import numpy as np

from .validation import ValidationError


def point_ions(params, ion_concentrations):
    """The point ions of the `electrostatics` input: name, valence, concentration (with the
    counterions; None if grand canonical), chemical potential (None if canonical) and smearing length."""
    ions = []
    for name, ion in (params.get("ions") or {}).items():
        grand = "chemical_potential" in ion
        ions.append({"name": name, "valence": float(ion["valence"]),
                     "concentration": None if grand else float(ion_concentrations[name]),
                     "chemical_potential": float(ion["chemical_potential"]) if grand else None,
                     "smearing_length": float(ion["smearing_length"])})
    return ions


def homogeneous_potential(polymer_charge, ions, coupling, kappa_sq, n_segments):
    """Uniform ψ₀ of the homogeneous state with the net polymer charge Σ_i z_i φ_i per segment.

    ψ₀ solves κ²ψ₀/E = q(ψ₀), the net charge per segment with the canonical point ions and the
    grand-canonical ones, e^(βμ_s) e^(-z_s ψ₀/N)/N (ions as in point_ions). q decreases with
    ψ₀, so the root is unique. Without screening and grand-canonical ions the system is neutral
    and ψ₀ = 0.
    """
    canonical = polymer_charge + sum(ion["valence"] * ion["concentration"] for ion in ions
                                     if ion["concentration"] is not None)
    grand = [ion for ion in ions if ion["concentration"] is None]
    if not grand and kappa_sq <= 0:
        return 0.0

    def excess(psi0):
        return kappa_sq * psi0 / coupling - canonical - sum(
            ion["valence"] * np.exp(ion["chemical_potential"] - ion["valence"] * psi0 / n_segments) / n_segments
            for ion in grand)
    low, high = -1.0, 1.0
    for _ in range(2000):
        if excess(low) <= 0:
            break
        low *= 2.0
    for _ in range(2000):
        if excess(high) >= 0:
            break
        high *= 2.0
    if not excess(low) <= 0 <= excess(high):
        raise ValidationError("No uniform potential neutralizes the homogeneous state")
    for _ in range(200):   # bisection to the precision of the doubles
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if excess(middle) > 0:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


class Electrostatics:
    """Electrostatic potential ψ, its Langevin step and its Hamiltonian term.

    Parameters
    ----------
    params : dict
        The `electrostatics` input: `charges` {type: z}, `bjerrum_length` (l_B/R₀) or `coupling`
        (E), and optionally `psi_dt_scaling` (mobility of ψ, default 1), `screening_length` (1/κ
        in units of R₀, the screening of a background electrolyte; none: unscreened) and `ions`
        {name: {valence, concentration, smearing_length}} (point ions).
    monomer_types : list of str
        Monomer types; types without a charge are neutral.
    smearing : Smearing
        Smearing of the simulation; supplies Γ_i(k) and k² of the grid.
    n_segments : float
        Segments N of the reference chain (1/ds of discrete chains, `n_segments` otherwise), or
        None when neither `bjerrum_length` nor point ions need it.
    langevin : dict
        Langevin settings with `nbar`, `dt` and the noise standard deviation `sigma`.
    bit_generator : numpy.random.PCG64
        Own noise stream of ψ, the jumped stream of the auxiliary fields
        (`random_bg.jumped()`): the noise of the auxiliary fields is the same as without
        electrostatics, and neighboring seeds do not share streams.
    fourier : FourierTransform
        PropagatorSolver.fourier of the simulation (complex fields; periodic or closed box).
    ion_concentrations : dict
        Concentration of each canonical point ion with the counterions (validation.point_ion_concentrations).
    open_charges : bool
        A grand-canonical species carries a charge (validation.open_charges): without screening the
        uniform mode of ψ then evolves.

    The optional `fixed_charge` of params, a charge density per segment of the reference density on
    the grid, adds to the charge density.
    """

    # Checkpoint data of ψ (checkpoint_data), and the pending sums of the charge structure
    # function that CLFTS keeps with Coulomb coupling
    CHECKPOINT_KEYS = ("psi_real", "psi_imag", "psi_k_real", "psi_k_imag", "psi_noise_prev", "psi_rng_state_json")
    CHARGE_SUMS = ("charge_mixed", "charge_difference", "charge_mixed_difference", "charge_difference_squared")

    def __init__(self, params, monomer_types, smearing, n_segments, langevin, bit_generator, *, fourier,
                 ion_concentrations=None, open_charges=False):
        self.charges = {name: float(params["charges"].get(name, 0.0)) for name in monomer_types}
        if "coupling" in params:
            self.coupling = float(params["coupling"])
        else:
            self.coupling = 4.0 * np.pi * float(params["bjerrum_length"]) * n_segments ** 2 * np.sqrt(langevin["nbar"])
        self.psi_dt_scaling = float(params.get("psi_dt_scaling", 1.0))
        self.nx = list(smearing.nx)
        self.k_sq = np.reshape(smearing.k_sq, self.nx)
        # k² + κ²: E divided by the (screened) Coulomb kernel
        self.screening_length = None if params.get("screening_length") is None else float(params["screening_length"])
        self.kappa_sq = 0.0 if self.screening_length is None else 1.0 / self.screening_length ** 2
        self.screened_k_sq = self.k_sq + self.kappa_sq
        # The uniform mode of ψ is zero for neutral canonical systems without screening
        self.zero_uniform_mode = self.screening_length is None and not open_charges
        self.sigma = langevin["sigma"]
        self._fourier = fourier
        self.n_segments = None if n_segments is None else float(n_segments)
        # Point ions: Γ_s(k) = exp(-a_s² k²/2), stored as multipliers of the transform
        self.ions = point_ions(params, ion_concentrations)
        for ion in self.ions:
            ion["gamma"] = np.exp(-ion["smearing_length"] ** 2 * self.k_sq / 2)
            ion["smear"] = fourier.set_multiplier(ion["gamma"])
        self._ion_state = None
        self.periodic = bool(fourier.periodic)
        self.fixed_charge = (None if params.get("fixed_charge") is None
                             else np.asarray(params["fixed_charge"], dtype=np.complex128).reshape(-1))
        self.psi = np.zeros(int(np.prod(self.nx)), dtype=np.complex128)

        self.random_bg = bit_generator
        self.random = np.random.Generator(self.random_bg)
        # Mean of the current and the previous noise (Leimkuhler-Matthews); CLFTS sets it
        # from the averaged_noise of its integrator
        self.averaged_noise = True
        self.previous_noise = np.zeros(self.psi.size)

        # Exact (Ornstein-Uhlenbeck) step of the linear part for each wavevector, rate
        # γ_k = s (k² + κ²)/E: decay a = exp(-γ_k δτN) and response (1 - a)/((k² + κ²)/E) to the
        # charge density. The noise amplitude sqrt((1 - a²)/(2 (k² + κ²)/E δτN)) of fresh noise
        # keeps the exact variance of the linear part; sqrt((1 - a)/(4 (k² + κ²)/E δτN)) on the
        # sum of two noises is that of a Leimkuhler-Matthews step of time (1 - a)/γ_k, exact also
        # with the explicit charge density. All reduce to the Euler (LM) step as γ_k → 0
        if self.coupling > 0:
            dt, s = float(langevin["dt"]), self.psi_dt_scaling
            rate = self.screened_k_sq / self.coupling
            with np.errstate(divide="ignore", invalid="ignore"):
                self._decay = np.exp(-rate * dt * s)
                self._response = np.where(rate > 0, -np.expm1(-rate * dt * s) / rate, dt * s)
                self._noise = np.where(rate > 0, np.sqrt(-np.expm1(-2 * rate * dt * s) / (2 * rate * dt)), np.sqrt(s))
                self._averaged_noise = np.where(rate > 0, np.sqrt(-np.expm1(-rate * dt * s) / (4 * rate * dt)),
                                                np.sqrt(s) / 2)

    @property
    def psi(self):
        """ψ on the grid (flattened)."""
        return self._psi

    @psi.setter
    def psi(self, value):
        self._psi = np.array(value, dtype=np.complex128).reshape(-1)
        self._psi_k = self._fourier.forward(self._psi)
        self._ion_state = None

    @property
    def psi_k(self):
        """ψ(k) on the grid shape (unnormalized FFT, read-only)."""
        view = self._psi_k.view()
        view.flags.writeable = False
        return view

    def psi_minus_k(self):
        """ψ(-k): the index-reversed ψ(k). ψ is complex, so this is not the conjugate of ψ(k)."""
        return np.roll(np.flip(self._psi_k), 1, axis=tuple(range(len(self.nx))))

    def add_potential(self, w_input):
        """Add z_i ψ to the monomer fields {type: w_i} of the charged types (in place)."""
        for name, z in self.charges.items():
            if z != 0.0:
                w_input[name] = w_input[name] + z * self.psi

    def charge_density(self, phi_smeared):
        """Smeared charge density c = Σ_i z_i (Γ_i * φ_i) + Σ_s z_s (Γ_s * c_s) from the smeared
        concentrations and the point ions of the current ψ, and the fixed charge density."""
        c = np.zeros_like(self.psi)
        for name, z in self.charges.items():
            if z != 0.0:
                c += z * phi_smeared[name]
        if self.ions:
            c += self.ion_charge()
        if self.fixed_charge is not None:
            c += self.fixed_charge
        return c

    def ion_densities(self):
        """Densities c_s(r) of the point ions and their partition functions Q_s for the current ψ."""
        if self._ion_state is None:
            densities, partitions = [], []
            for ion in self.ions:
                weight = np.exp(-ion["valence"] * self._fourier.backward(ion["gamma"] * self._psi_k) / self.n_segments)
                partition = np.mean(weight)
                if ion["concentration"] is None:   # grand canonical: e^(βμ_s) Q_s ions per reference chain
                    densities.append(np.exp(ion["chemical_potential"]) * weight / self.n_segments)
                else:
                    densities.append(ion["concentration"] * weight / partition)
                partitions.append(partition)
            self._ion_state = (densities, partitions)
        return self._ion_state

    def ion_charge(self):
        """Smeared charge density of the point ions, Σ_s z_s (Γ_s * c_s), for the current ψ."""
        densities, _ = self.ion_densities()
        return self._fourier.apply([(ion["smear"], ion["valence"] * c) for ion, c in zip(self.ions, densities)])

    def ion_chemical_potentials(self):
        """βμ_s = ln(N c_s) - ln Q_s of each point ion, per ion (the convention of the chains); the given
        value for grand-canonical ions."""
        _, partitions = self.ion_densities()
        with np.errstate(divide="ignore"):
            return [complex(ion["chemical_potential"]) if ion["concentration"] is None
                    else np.log(self.n_segments * ion["concentration"]) - np.log(complex(q))
                    for ion, q in zip(self.ions, partitions)]

    def ion_number_densities(self):
        """Ions per reference chain, N c_s, of each point ion: e^(βμ_s) Q_s for grand-canonical ions."""
        _, partitions = self.ion_densities()
        return [np.exp(ion["chemical_potential"]) * complex(q) if ion["concentration"] is None
                else complex(self.n_segments * ion["concentration"]) for ion, q in zip(self.ions, partitions)]

    def force(self, c):
        """δH/δψ = (∇² - κ²)ψ/E + c; zero without Coulomb coupling (ψ stays zero)."""
        if self.coupling <= 0:
            return np.zeros_like(self.psi)
        return self._fourier.backward(-self.screened_k_sq * self._psi_k) / self.coupling + c

    def step(self, c):
        """Advance ψ by one Langevin step with the charge density c of the same state."""
        if self.coupling <= 0:
            return
        xi = self.random.normal(0.0, self.sigma, self.psi.size)
        if self.averaged_noise:
            amplitude, noise = self._averaged_noise, xi + self.previous_noise
            self.previous_noise = xi
        else:
            amplitude, noise = self._noise, xi
        noise_k, c_k = self._fourier.forward(np.stack([noise, c]))
        psi_k = self._decay * self._psi_k + self._response * c_k + 1j * amplitude * noise_k
        self._set_psi_k(psi_k)

    def draw_noise(self):
        """Noise of ψ for one step, from its own stream, for an integrator that advances ψ
        together with the auxiliary fields (accept)."""
        return self.random.normal(0.0, self.sigma, self.psi.size)

    def accept(self, psi, noise):
        """Take ψ from such an integrator, with the noise of its step as the previous noise."""
        self._set_psi_k(self._fourier.forward(np.asarray(psi, dtype=np.complex128)))
        self.previous_noise = np.asarray(noise, dtype=np.float64).copy()

    def _set_psi_k(self, psi_k):
        # Without screening the uniform mode is zero (electroneutrality), unless charged
        # grand-canonical species need it as the Lagrange multiplier of neutrality
        if self.zero_uniform_mode:
            psi_k[(0,) * len(self.nx)] = 0.0
        self._psi_k = psi_k
        self._psi = self._fourier.backward(psi_k)
        self._ion_state = None

    @contextlib.contextmanager
    def trial(self, psi):
        """Set ψ for an intermediate evaluation and restore ψ and ψ(k) exactly afterwards."""
        saved = self._psi, self._psi_k, self._ion_state
        try:
            self.psi = psi
            yield
        finally:
            self._psi, self._psi_k, self._ion_state = saved

    def hamiltonian(self):
        """Electrostatic term (1/(2E V)) ∫ ψ (∇² - κ²) ψ dr of the current ψ, -Σ_s N c_s ln Q_s
        of the point ions and the coupling <ψ c_f> of the fixed charge density c_f (the chains and
        the ions couple to ψ through their partition functions).

        By Parseval's theorem, mean(ψ (∇² - κ²) ψ) = -Σ_k (k² + κ²) ψ(k) ψ(-k) / n² (unnormalized
        FFT, n grid points); with walls it is the grid mean of ψ times the backward cosine series of
        -(k² + κ²) ψ(k).
        """
        value = 0.0 if self.fixed_charge is None else np.mean(self._psi * self.fixed_charge)
        if self.ions:
            _, partitions = self.ion_densities()
            for ion, q in zip(self.ions, partitions):
                if ion["concentration"] is None:
                    value -= np.exp(ion["chemical_potential"]) * complex(q)
                else:
                    value -= self.n_segments * ion["concentration"] * np.log(complex(q))
        if self.coupling <= 0:
            return value
        if not self.periodic:
            return value - 0.5 * np.mean(self._psi * self._fourier.backward(self.screened_k_sq * self._psi_k)) / self.coupling
        return value + (-0.5 * np.sum(self.screened_k_sq * self._psi_k * self.psi_minus_k())
                        / self.psi.size ** 2 / self.coupling)

    def pressure(self, dim, n_chains):
        """Contribution of ψ to the osmotic pressure βΠ R₀³/C (clfts.py, pressure).

        At fixed grid values of ψ a dilation of the box by V changes k² as V^(-2/d), so the
        term V H_el changes by (1 - 2/d) H_k + H_κ, the parts of H_el from k² and from κ².
        The normalization of the Gaussian integral over ψ adds, for every mode it integrates
        (all modes with k² + κ² > 0), (1 - (2/d) k²/(k² + κ²))/2 per n_chains = C V/R₀³
        reference chains.

        A fixed charge density, held at its grid values, adds the field term -<ψ c_f>. Point ions add
        their ideal gas, Σ_s N <c_s> (e^(βμ_s) Q_s if grand canonical), and the dilation of
        their smearing functions,
        -Σ_s <c_s (∂Γ_s/∂ln V) * (z_s ψ)> with ∂Γ_s/∂ln V = (a_s² k²/d) Γ_s, as the monomer types
        do (clfts.py, pressure).
        """
        ions = 0.0 if self.fixed_charge is None else -np.mean(self._psi * self.fixed_charge)
        if self.ions:
            densities, _ = self.ion_densities()
            for ion, c in zip(self.ions, densities):
                dilation = (ion["smearing_length"] ** 2 * self.k_sq / dim) * ion["gamma"] * ion["valence"] * self._psi_k
                ions += self.n_segments * np.mean(c) - np.mean(c * self._fourier.backward(dilation))
        if self.coupling <= 0:
            return ions
        n = self.psi.size
        weight = self._psi_k * self.psi_minus_k() / n ** 2 / self.coupling
        h_k = -0.5 * np.sum(self.k_sq * weight)
        h_kappa = -0.5 * self.kappa_sq * np.sum(weight)
        modes = self.screened_k_sq > 0
        normalization = 0.5 * np.sum(1.0 - (2.0 / dim) * self.k_sq[modes] / self.screened_k_sq[modes]) / n_chains
        return ions - ((1.0 - 2.0 / dim) * h_k + h_kappa) + normalization

    def settings(self):
        """Settings that define a continuation (checkpoint hash)."""
        settings = {"charges": self.charges, "coupling": self.coupling, "psi_dt_scaling": self.psi_dt_scaling}
        if self.screening_length is not None:
            settings["screening_length"] = self.screening_length
        if self.ions:
            settings["ions"] = {ion["name"]: {key: ion[key] for key in ("valence", "concentration", "chemical_potential",
                                                                        "smearing_length") if ion[key] is not None}
                                for ion in self.ions}
            settings["n_segments"] = self.n_segments
        if self.fixed_charge is not None:
            settings["fixed_charge_sha256"] = hashlib.sha256(np.ascontiguousarray(np.real(self.fixed_charge)).tobytes()).hexdigest()
        return settings

    def checkpoint_data(self):
        # The order of the ions, which the record columns follow (settings() keeps them by name)
        return {"psi_real": np.real(self.psi), "psi_imag": np.imag(self.psi),
                "psi_k_real": np.real(self._psi_k), "psi_k_imag": np.imag(self._psi_k),
                "psi_rng_state_json": json.dumps(self.random_bg.state), "psi_noise_prev": self.previous_noise,
                "ion_names_json": json.dumps([ion["name"] for ion in self.ions])}

    def restore(self, data):
        """ψ, ψ(k), the previous noise and the noise stream from a checkpoint.

        ψ(k) is restored as stored, because transforming ψ back does not reproduce it bit
        for bit.
        """
        missing = [key for key in self.CHECKPOINT_KEYS if key not in data]
        if missing:
            raise ValidationError(f"The checkpoint has no electrostatic potential ({', '.join(missing)}); "
                                  "it was not written with 'electrostatics'")
        self.psi = np.asarray(data["psi_real"]) + 1j * np.asarray(data["psi_imag"])
        self._psi_k = np.reshape(np.asarray(data["psi_k_real"]) + 1j * np.asarray(data["psi_k_imag"]),
                                 self.nx).astype(np.complex128)
        self._ion_state = None
        self.random_bg.state = json.loads(str(data["psi_rng_state_json"]))
        self.previous_noise = np.asarray(data["psi_noise_prev"], dtype=np.float64).reshape(-1).copy()


class MeanFieldElectrostatics:
    """Electrostatic potential ψ at the saddle point (SCFT): the (screened) Poisson equation.

    The Hamiltonian is that of Electrostatics. At its saddle point ψ is real and
    (k² + κ²)ψ/E = c, where c is the smeared charge density of the chains and the point ions,
    and the free energy per reference chain is

        F = F_chains[w_i + z_i ψ] - (1/(2E)) <ψ (k² + κ²) ψ> + <ψ c_f> + Σ_s F_s,

    with F_chains the SCFT free energy of the chains in the fields they feel, the fixed charge
    density c_f (fixed_charge), and the point ions F_s = N c_s (ln(N c_s) - 1) - N c_s ln Q_s (canonical) or -e^(βμ_s) Q_s (grand
    canonical). F is stationary in w and ψ at the saddle point: δF/δψ = c - (k² + κ²)ψ/E.

    SCFT iterates ψ with the fields. The residual of ψ is the Poisson residual divided by its
    linear response, R(k) = [c(k) - (k² + κ²)ψ(k)/E] / [(k² + κ²)/E + D(k)], with the response
    D(k) = -∂c(k)/∂ψ(k) of the charges in the homogeneous state (set_response): R is the Newton
    step of the linearized equation, which keeps the iteration stable where E/k² is large.
    Without screening the uniform mode of ψ is zero in a neutral canonical system; charged
    grand-canonical ions make it the Donnan potential. Real fields in a periodic box or a closed
    box with walls on every side; the transforms are those of PropagatorSolver.fourier (Fourier
    modes, or the cosine modes of a closed box).

    Parameters
    ----------
    params : dict
        The `electrostatics` input: `charges`, `coupling` (E), and optionally `screening_length`,
        `ions`, `counterion` and `fixed_charge`.
    monomer_types : list of str
        Monomer types; types without a charge are neutral.
    n_segments : float or None
        Segments N of the reference chain, needed by point ions.
    fourier : FourierTransform
        PropagatorSolver.fourier of the simulation (real fields; periodic or closed box).
    ion_concentrations : dict
        Concentration of each canonical point ion with the counterions (validation.point_ion_concentrations).
    open_charges : bool
        A grand-canonical ion carries a charge (validation.open_charges).
    """

    def __init__(self, params, monomer_types, n_segments, fourier, *, ion_concentrations, open_charges):
        self.charges = {name: float(params["charges"].get(name, 0.0)) for name in monomer_types}
        self.coupling = float(params["coupling"])
        self.screening_length = None if params.get("screening_length") is None else float(params["screening_length"])
        self.kappa_sq = 0.0 if self.screening_length is None else 1.0 / self.screening_length ** 2
        self.zero_uniform_mode = self.screening_length is None and not open_charges
        self.n_segments = None if n_segments is None else float(n_segments)
        self.ions = point_ions(params, ion_concentrations)
        self.fixed_charge = (None if params.get("fixed_charge") is None
                             else np.asarray(params["fixed_charge"], dtype=np.float64).reshape(-1))
        self._fourier = fourier
        self.response = None
        self.set_box()

    def set_box(self):
        """Take k² of the current box (after the cell changes)."""
        self.k_sq = self._fourier.k_squared()
        self.screened_k_sq = self.k_sq + self.kappa_sq
        for ion in self.ions:
            ion["gamma"] = np.exp(-ion["smearing_length"] ** 2 * self.k_sq / 2)

    def set_response(self, response):
        """The response D(k) = -∂c(k)/∂ψ(k) of the homogeneous state (coefficient shape) for the residual."""
        self.response = np.maximum(np.real(response), 0.0)

    def add_potential(self, w_input, psi):
        """Add z_i ψ to the monomer fields {type: w_i} of the charged types."""
        for name, z in self.charges.items():
            if z != 0.0:
                w_input[name] = w_input[name] + z * psi

    def ion_densities(self, psi):
        """Densities c_s(r) of the point ions and their partition functions Q_s for ψ."""
        densities, partitions = [], []
        if self.ions:
            psi_k = self._fourier.forward(psi)
        for ion in self.ions:
            weight = np.exp(-ion["valence"] * self._fourier.backward(ion["gamma"] * psi_k) / self.n_segments)
            partition = float(np.mean(weight))
            if ion["concentration"] is None:
                densities.append(np.exp(ion["chemical_potential"]) * weight / self.n_segments)
            else:
                densities.append(ion["concentration"] * weight / partition)
            partitions.append(partition)
        return densities, partitions

    def charge_density(self, phi_smeared, psi):
        """Smeared charge density Σ_i z_i (Γ_i * φ_i) + Σ_s z_s (Γ_s * c_s) for ψ, and the fixed charge."""
        c = np.zeros_like(psi) if self.fixed_charge is None else self.fixed_charge.copy()
        for name, z in self.charges.items():
            if z != 0.0:
                c = c + z * phi_smeared[name]
        if self.ions:
            densities, _ = self.ion_densities(psi)
            charge_k = sum(ion["valence"] * ion["gamma"] * self._fourier.forward(density)
                           for ion, density in zip(self.ions, densities))
            c = c + self._fourier.backward(charge_k)
        return c

    def residual(self, c, psi):
        """Poisson residual of ψ divided by its linear response (the Newton step of ψ)."""
        rate = self.screened_k_sq / self.coupling + (0.0 if self.response is None else self.response)
        force_k = self._fourier.forward(c) - self.screened_k_sq * self._fourier.forward(psi) / self.coupling
        with np.errstate(divide="ignore", invalid="ignore"):
            step_k = np.where(rate > 0, force_k / rate, 0.0)
        if self.zero_uniform_mode:
            step_k[(0,) * step_k.ndim] = 0.0
        return self._fourier.backward(step_k)

    def poisson_error(self, phi_smeared, psi):
        """Root mean square of c - (k² + κ²)ψ/E relative to the mean density of charges,
        Σ_i |z_i| <φ_i> + Σ_s |z_s| <c_s> (the net charge can vanish)."""
        c = self.charge_density(phi_smeared, psi)
        force = c - self._fourier.backward(self.screened_k_sq * self._fourier.forward(psi)) / self.coupling
        if self.zero_uniform_mode:
            force = force - np.mean(force)
        densities, _ = self.ion_densities(psi)
        scale = sum(abs(z) * np.mean(phi_smeared[name]) for name, z in self.charges.items()) \
            + sum(abs(ion["valence"]) * np.mean(density) for ion, density in zip(self.ions, densities)) \
            + (0.0 if self.fixed_charge is None else float(np.mean(np.abs(self.fixed_charge))))
        return float(np.sqrt(np.mean(force ** 2)) / max(scale, 1e-300))

    def free_energy(self, psi):
        """-(1/(2E)) <ψ (k² + κ²) ψ>, the coupling <ψ c_f> of the fixed charge and the point ions
        (per reference chain)."""
        value = -0.5 * float(np.mean(psi * self._fourier.backward(self.screened_k_sq * self._fourier.forward(psi)))) \
            / self.coupling
        if self.fixed_charge is not None:
            value += float(np.mean(psi * self.fixed_charge))
        _, partitions = self.ion_densities(psi)
        for ion, q in zip(self.ions, partitions):
            if ion["concentration"] is None:
                value -= np.exp(ion["chemical_potential"]) * q
            elif ion["concentration"] > 0:
                number = self.n_segments * ion["concentration"]
                value += number * (np.log(number) - 1.0) - number * np.log(q)
        return value

    def ion_chemical_potentials(self, psi):
        """βμ_s = ln(N c_s) - ln Q_s of each point ion; the given value for grand-canonical ions."""
        _, partitions = self.ion_densities(psi)
        with np.errstate(divide="ignore"):
            return [ion["chemical_potential"] if ion["concentration"] is None
                    else float(np.log(self.n_segments * ion["concentration"]) - np.log(q))
                    for ion, q in zip(self.ions, partitions)]

    def ion_number_densities(self, psi):
        """Ions per reference chain, N c_s (e^(βμ_s) Q_s for grand-canonical ions)."""
        _, partitions = self.ion_densities(psi)
        return [np.exp(ion["chemical_potential"]) * q if ion["concentration"] is None
                else self.n_segments * ion["concentration"] for ion, q in zip(self.ions, partitions)]

    def homogeneous(self, polymer_charge):
        """Uniform ψ₀ (homogeneous_potential) and free energy of the homogeneous state with the net
        polymer charge Σ_i z_i φ_i.

        The free energy is -κ²ψ₀²/(2E) + ψ₀ (polymer and canonical ion charge) + Σ_s F_s: the
        uniform potential shifts the fields of the canonical species, and F_s of the
        grand-canonical ions is -N c_s.
        """
        psi0 = homogeneous_potential(polymer_charge, self.ions, self.coupling, self.kappa_sq, self.n_segments)
        canonical = polymer_charge + sum(ion["valence"] * ion["concentration"] for ion in self.ions
                                         if ion["concentration"] is not None)
        energy = -0.5 * self.kappa_sq * psi0 ** 2 / self.coupling + psi0 * canonical
        for ion in self.ions:
            if ion["concentration"] is None:
                energy -= np.exp(ion["chemical_potential"] - ion["valence"] * psi0 / self.n_segments)
            elif ion["concentration"] > 0:
                number = self.n_segments * ion["concentration"]
                energy += number * (np.log(number) - 1.0)
        return psi0, energy

    def stress(self, psi, coefficients, k_sq, dk_sq):
        """Lattice derivatives of F at fixed w and ψ from ψ and the point ions.

        -(1/(2E)) <ψ (k² + κ²) ψ> gives -(1/(2E)) Σ_k (∂k²/∂θ) |ψ(k)|²/n², and each ion
        <c_s z_s ((∂Γ_s/∂θ) * ψ)> with ∂Γ_s/∂θ = -(a_s²/2) Γ_s ∂k²/∂θ. coefficients gives the
        transform of a real field at the wavenumbers of k_sq and dk_sq ({stress slot: ∂k²/∂θ}),
        normalized like numpy.fft.fftn (Smearing.coefficients: Fourier modes, or the cosine modes
        of a closed box). The smearing of the monomer types is that of Smearing.stress with the
        fields w_i + z_i ψ.
        """
        result = np.zeros(6)
        n = psi.size
        psi_k = np.ravel(coefficients(psi))
        weight = -0.5 * np.abs(psi_k) ** 2 / n ** 2 / self.coupling
        if self.ions:
            densities, _ = self.ion_densities(psi)
            for ion, density in zip(self.ions, densities):
                density_k = np.ravel(coefficients(density))
                gamma = np.exp(-ion["smearing_length"] ** 2 * np.ravel(k_sq) / 2)
                weight = weight + ion["valence"] * np.real(np.conj(density_k) * psi_k) \
                    * (-0.5 * ion["smearing_length"] ** 2 * gamma) / n ** 2
        for slot, derivative in dk_sq.items():
            result[slot] = np.sum(weight * np.ravel(derivative))
        return result
