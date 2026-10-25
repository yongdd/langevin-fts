"""Smeared Coulomb interactions for complex Langevin field-theoretic simulations.

Monomer type i carries the valence z_i per segment, and the charges interact through
the smeared charge density c(r) = Σ_i z_i (Γ_i * φ_i)(r) with the Coulomb kernel E/k²,
where Γ_i is the smearing function of the type (smearing.py). The interaction is
decoupled with an electrostatic potential ψ on the rotated contour: like the pressure
field it is an imaginary-type field, and type i feels z_i ψ, so the propagator input is
Γ_i * (w_i + z_i ψ). Per reference chain (as in polymer_field_theory.py),

    βH_el / (C V/R₀³) = (1/(2E V)) ∫ ψ ∇²ψ dr,   δH/δψ = ∇²ψ/E + c,

with the coupling E = 4π l_B ρ₀ N R₀² = 4π (l_B/R₀) N² N̄^(1/2). At the saddle point
-∇²ψ = E c (Poisson equation). ψ moves like an imaginary field (drift +δH/δψ, imaginary
noise). Its linear part ∇²ψ/E is stiff at large k and is integrated exactly for each
wavevector (exponential time differencing); the charge density stays explicit. ψ takes the
mean of the current and the previous noise, as the Leimkuhler-Matthews integrator of the
auxiliary fields does: the step is a Leimkuhler-Matthews step of ψ with the time step
(1 - exp(-γ δτN))/γ of each mode, γ = s k²/E, and the Gaussian statistics of ψ and the
auxiliary fields are exact for any time step. With an integrator that declares fresh noise
(averaged_noise = False, as Euler does), ψ takes fresh noise; the explicit charge density
then biases its variance at first order in γ δτN.
ψ is kept in Fourier space; its grid values follow from one inverse transform per step,
and H is evaluated by Parseval's theorem.
The uniform mode of ψ is zero (electroneutrality). Periodic boxes only.

References: Riggleman, Kumar & Fredrickson, J. Chem. Phys. 136, 024903 (2012);
Delaney & Fredrickson, J. Phys. Chem. B 120, 7615 (2016).
"""

import json

import numpy as np

from .validation import ValidationError


class Electrostatics:
    """Electrostatic potential ψ, its Langevin step and its Hamiltonian term.

    Parameters
    ----------
    params : dict
        The `electrostatics` input: `charges` {type: z}, `bjerrum_length` (l_B/R₀, discrete
        chains) or `coupling` (E), and optionally `psi_dt_scaling` (mobility of ψ, default 1).
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
        self.sigma = langevin["sigma"]
        self.psi = np.zeros(int(np.prod(self.nx)), dtype=np.complex128)

        self.random_bg = bit_generator
        self.random = np.random.Generator(self.random_bg)
        # Mean of the current and the previous noise (Leimkuhler-Matthews); CLFTS sets it
        # from the averaged_noise of its integrator
        self.averaged_noise = True
        self.previous_noise = np.zeros(self.psi.size)

        # Exact (Ornstein-Uhlenbeck) step of the linear part for each wavevector, rate
        # γ_k = s k²/E: decay a = exp(-γ_k δτN) and response (1 - a)/(k²/E) to the charge
        # density. The noise amplitude sqrt((1 - a²)/(2 k²/E δτN)) of fresh noise keeps the
        # exact variance of the linear part; sqrt((1 - a)/(4 k²/E δτN)) on the sum of two
        # noises is that of a Leimkuhler-Matthews step of time (1 - a)/γ_k, exact also with
        # the explicit charge density. All reduce to the Euler (LM) step as k → 0
        if self.coupling > 0:
            dt, s = float(langevin["dt"]), self.psi_dt_scaling
            rate = self.k_sq / self.coupling
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
        """δH/δψ = ∇²ψ/E + c; zero without Coulomb coupling (ψ stays zero)."""
        if self.coupling <= 0:
            return np.zeros_like(self.psi)
        return np.reshape(np.fft.ifftn(-self.k_sq * self._psi_k), -1) / self.coupling + c

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
        psi_k[(0,) * len(self.nx)] = 0.0
        self._psi_k = psi_k
        self._psi = np.reshape(np.fft.ifftn(psi_k), -1)

    def hamiltonian(self):
        """Electrostatic term (1/(2E V)) ∫ ψ ∇²ψ dr of the current ψ.

        By Parseval's theorem, mean(ψ ∇²ψ) = -Σ_k k² ψ(k) ψ(-k) / n² (unnormalized FFT, n grid
        points).
        """
        if self.coupling <= 0:
            return 0.0
        return -0.5 * np.sum(self.k_sq * self._psi_k * self.psi_minus_k()) / self.psi.size ** 2 / self.coupling

    def settings(self):
        """Settings that define a continuation (checkpoint hash)."""
        return {"charges": self.charges, "coupling": self.coupling, "psi_dt_scaling": self.psi_dt_scaling}

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
