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
wavevector (exponential time differencing), so that the stationary variance of the
linear modes does not depend on the time step; the charge density stays explicit.
The uniform mode of ψ is zero (electroneutrality). Periodic boxes only.

References: Riggleman, Kumar & Fredrickson, J. Chem. Phys. 136, 024903 (2012);
Delaney & Fredrickson, J. Phys. Chem. B 120, 7615 (2016).
"""

import json

import numpy as np


class Electrostatics:
    """Electrostatic potential ψ, its Langevin step and its Hamiltonian term.

    Parameters
    ----------
    params : dict
        The `electrostatics` input: `charges` {type: z}, `bjerrum_length` (l_B/R₀)
        or `coupling` (E), and optionally `psi_dt_scaling` (mobility of ψ, default 1).
    monomer_types : list of str
        Monomer types; types without a charge are neutral.
    smearing : Smearing
        Smearing of the simulation; supplies Γ_i(k) and k² of the grid.
    n_segments : int
        Segments of the reference chain, N = 1/ds.
    langevin : dict
        Langevin settings with `nbar`, `dt` and the noise standard deviation `sigma`.
    random_seed : int or None
        ψ uses its own PCG64 stream (seed + 1), so the noise of the auxiliary fields
        is the same as without electrostatics.
    """

    def __init__(self, params, monomer_types, smearing, n_segments, langevin, random_seed=None):
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

        self.random_bg = np.random.PCG64() if random_seed is None else np.random.PCG64(random_seed + 1)
        self.random = np.random.Generator(self.random_bg)

        # Exact (Ornstein-Uhlenbeck) step of the linear part for each wavevector, rate
        # γ_k = s k²/E: decay a = exp(-γ_k δτN), response (1 - a)/(k²/E) to the charge
        # density and noise amplitude sqrt((1 - a²)/(2 k²/E δτN)), which keeps the exact
        # stationary variance; all three reduce to the Euler step as k → 0
        if self.coupling > 0:
            dt, s = float(langevin["dt"]), self.psi_dt_scaling
            rate = self.k_sq / self.coupling
            with np.errstate(divide="ignore", invalid="ignore"):
                self._decay = np.exp(-rate * dt * s)
                self._response = np.where(rate > 0, -np.expm1(-rate * dt * s) / rate, dt * s)
                self._noise = np.where(rate > 0, np.sqrt(-np.expm1(-2 * rate * dt * s) / (2 * rate * dt)), np.sqrt(s))

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

    def _laplacian(self, field):
        return np.reshape(np.fft.ifftn(-self.k_sq * np.fft.fftn(np.reshape(field, self.nx))), -1)

    def force(self, c):
        """δH/δψ = ∇²ψ/E + c; zero without Coulomb coupling (ψ stays zero)."""
        if self.coupling <= 0:
            return np.zeros_like(self.psi)
        return self._laplacian(self.psi) / self.coupling + c

    def step(self, c):
        """Advance ψ by one Langevin step with the charge density c of the same state."""
        if self.coupling <= 0:
            return
        xi = self.random.normal(0.0, self.sigma, self.psi.size)
        psi_k = (self._decay * np.fft.fftn(np.reshape(self.psi, self.nx))
                 + self._response * np.fft.fftn(np.reshape(c, self.nx))
                 + 1j * self._noise * np.fft.fftn(np.reshape(xi, self.nx)))
        psi_k[(0,) * len(self.nx)] = 0.0
        self.psi = np.reshape(np.fft.ifftn(psi_k), -1)

    def hamiltonian(self):
        """Electrostatic term (1/(2E V)) ∫ ψ ∇²ψ dr of the current ψ."""
        if self.coupling <= 0:
            return 0.0
        return 0.5 * np.mean(self.psi * self._laplacian(self.psi)) / self.coupling

    def settings(self):
        """Settings that define a continuation (checkpoint hash)."""
        return {"charges": self.charges, "coupling": self.coupling, "psi_dt_scaling": self.psi_dt_scaling}

    def checkpoint_data(self):
        return {"psi_real": np.real(self.psi), "psi_imag": np.imag(self.psi),
                "psi_rng_state_json": json.dumps(self.random_bg.state)}

    def restore(self, data):
        """ψ and its noise stream from a checkpoint."""
        self.psi = (np.asarray(data["psi_real"]) + 1j * np.asarray(data["psi_imag"])).reshape(-1).astype(np.complex128)
        self.random_bg.state = json.loads(str(data["psi_rng_state_json"]))
