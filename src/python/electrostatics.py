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
and H is evaluated by Parseval's theorem.
Without screening the uniform mode of ψ is zero (electroneutrality); with screening it
fluctuates, and the system may carry a net charge. Periodic boxes only.

References: Riggleman, Kumar & Fredrickson, J. Chem. Phys. 136, 024903 (2012);
Delaney & Fredrickson, J. Phys. Chem. B 120, 7615 (2016).
"""

import contextlib
import json

import numpy as np

from .validation import ValidationError


class Electrostatics:
    """Electrostatic potential ψ, its Langevin step and its Hamiltonian term.

    Parameters
    ----------
    params : dict
        The `electrostatics` input: `charges` {type: z}, `bjerrum_length` (l_B/R₀, discrete
        chains) or `coupling` (E), and optionally `psi_dt_scaling` (mobility of ψ, default 1)
        and `screening_length` (1/κ in units of R₀, the screening of a background electrolyte;
        none: unscreened).
    monomer_types : list of str
        Monomer types; types without a charge are neutral.
    smearing : Smearing
        Smearing of the simulation; supplies Γ_i(k) and k² of the grid.
    n_segments : int
        Segments of the reference chain, N = 1/ds.
    langevin : dict
        Langevin settings with `nbar`, `dt` and the noise standard deviation `sigma`.
    bit_generator : numpy.random.PCG64
        Own noise stream of ψ, the jumped stream of the auxiliary fields
        (`random_bg.jumped()`): the noise of the auxiliary fields is the same as without
        electrostatics, and neighboring seeds do not share streams.
    """

    # Checkpoint data of ψ (checkpoint_data), and the pending sums of the charge structure
    # function that CLFTS keeps with Coulomb coupling
    CHECKPOINT_KEYS = ("psi_real", "psi_imag", "psi_k_real", "psi_k_imag", "psi_noise_prev", "psi_rng_state_json")
    CHARGE_SUMS = ("charge_mixed", "charge_difference", "charge_mixed_difference", "charge_difference_squared")

    def __init__(self, params, monomer_types, smearing, n_segments, langevin, bit_generator):
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
        self.sigma = langevin["sigma"]
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
        self._psi_k = np.fft.fftn(np.reshape(self._psi, self.nx))

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
        """Smeared charge density c = Σ_i z_i (Γ_i * φ_i) from the smeared concentrations."""
        c = np.zeros_like(self.psi)
        for name, z in self.charges.items():
            if z != 0.0:
                c += z * phi_smeared[name]
        return c

    def force(self, c):
        """δH/δψ = (∇² - κ²)ψ/E + c; zero without Coulomb coupling (ψ stays zero)."""
        if self.coupling <= 0:
            return np.zeros_like(self.psi)
        return np.reshape(np.fft.ifftn(-self.screened_k_sq * self._psi_k), -1) / self.coupling + c

    def step(self, c):
        """Advance ψ by one Langevin step with the charge density c of the same state."""
        if self.coupling <= 0:
            return
        xi = self.random.normal(0.0, self.sigma, self.psi.size)
        if self.averaged_noise:
            noise = self._averaged_noise * np.fft.fftn(np.reshape(xi + self.previous_noise, self.nx))
            self.previous_noise = xi
        else:
            noise = self._noise * np.fft.fftn(np.reshape(xi, self.nx))
        psi_k = (self._decay * self._psi_k
                 + self._response * np.fft.fftn(np.reshape(c, self.nx))
                 + 1j * noise)
        self._set_psi_k(psi_k)

    def draw_noise(self):
        """Noise of ψ for one step, from its own stream, for an integrator that advances ψ
        together with the auxiliary fields (accept)."""
        return self.random.normal(0.0, self.sigma, self.psi.size)

    def accept(self, psi, noise):
        """Take ψ from such an integrator, with the noise of its step as the previous noise."""
        self._set_psi_k(np.fft.fftn(np.reshape(np.asarray(psi, dtype=np.complex128), self.nx)))
        self.previous_noise = np.asarray(noise, dtype=np.float64).copy()

    def _set_psi_k(self, psi_k):
        # Without screening the uniform mode is zero (electroneutrality)
        if self.screening_length is None:
            psi_k[(0,) * len(self.nx)] = 0.0
        self._psi_k = psi_k
        self._psi = np.reshape(np.fft.ifftn(psi_k), -1)

    @contextlib.contextmanager
    def trial(self, psi):
        """Set ψ for an intermediate evaluation and restore ψ and ψ(k) exactly afterwards."""
        saved = self._psi, self._psi_k
        try:
            self.psi = psi
            yield
        finally:
            self._psi, self._psi_k = saved

    def hamiltonian(self):
        """Electrostatic term (1/(2E V)) ∫ ψ (∇² - κ²) ψ dr of the current ψ.

        By Parseval's theorem, mean(ψ (∇² - κ²) ψ) = -Σ_k (k² + κ²) ψ(k) ψ(-k) / n² (unnormalized
        FFT, n grid points).
        """
        if self.coupling <= 0:
            return 0.0
        return (-0.5 * np.sum(self.screened_k_sq * self._psi_k * self.psi_minus_k())
                / self.psi.size ** 2 / self.coupling)

    def pressure(self, dim, n_chains):
        """Contribution of ψ to the osmotic pressure βΠ R₀³/C (clfts.py, pressure).

        At fixed grid values of ψ a dilation of the box by V changes k² as V^(-2/d), so the
        term V H_el changes by (1 - 2/d) H_k + H_κ, the parts of H_el from k² and from κ².
        The normalization of the Gaussian integral over ψ adds, for every mode it integrates
        (all modes with k² + κ² > 0), (1 - (2/d) k²/(k² + κ²))/2 per n_chains = C V/R₀³
        reference chains.
        """
        if self.coupling <= 0:
            return 0.0
        n = self.psi.size
        weight = self._psi_k * self.psi_minus_k() / n ** 2 / self.coupling
        h_k = -0.5 * np.sum(self.k_sq * weight)
        h_kappa = -0.5 * self.kappa_sq * np.sum(weight)
        modes = self.screened_k_sq > 0
        normalization = 0.5 * np.sum(1.0 - (2.0 / dim) * self.k_sq[modes] / self.screened_k_sq[modes]) / n_chains
        return -((1.0 - 2.0 / dim) * h_k + h_kappa) + normalization

    def settings(self):
        """Settings that define a continuation (checkpoint hash)."""
        settings = {"charges": self.charges, "coupling": self.coupling, "psi_dt_scaling": self.psi_dt_scaling}
        if self.screening_length is not None:
            settings["screening_length"] = self.screening_length
        return settings

    def checkpoint_data(self):
        return {"psi_real": np.real(self.psi), "psi_imag": np.imag(self.psi),
                "psi_k_real": np.real(self._psi_k), "psi_k_imag": np.imag(self._psi_k),
                "psi_rng_state_json": json.dumps(self.random_bg.state), "psi_noise_prev": self.previous_noise}

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
        self.random_bg.state = json.loads(str(data["psi_rng_state_json"]))
        self.previous_noise = np.asarray(data["psi_noise_prev"], dtype=np.float64).reshape(-1).copy()
