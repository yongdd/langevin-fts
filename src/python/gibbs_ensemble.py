"""Gibbs-ensemble complex Langevin simulation of two coexisting phases.

Two CL-FTS boxes, one per phase, exchange chains, point ions and volume while the totals stay
fixed (Riggleman & Fredrickson, J. Chem. Phys. 132, 024104 (2010)). Every box keeps the
reference density on average (the volume fractions of its polymers sum to 1), so its volume
follows from its chains: V_b = Σ_p α_p n_p^b, with the numbers of chains n_p^b in units of
C/R₀³ = ρ₀/N per unit volume (and of ions, n_s^b = N c_s^b V_b). Moving one chain of p from box
II to box I changes the free energy by μ̃_p^I - μ̃_p^II, with the exchange potential

    μ̃_p = βμ_p - α_p βΠR₀³/C,

since the chain takes its volume along; a point ion has no volume, μ̃_s = βμ_s. Both are
averages of the real parts of the CL-FTS operators (clfts.py, chemical_potentials and pressure)
over one cycle of Langevin steps of each box. After the cycle the amounts move down the
difference,

    Δn^I = -r M_P (μ̃^I - μ̃^II),   Δn^II = -Δn^I,

with the rate r (exchange_rate) and the mobility M = diag(n^I n^II/(n^I + n^II)), which makes
the steps relative to the amounts. A step moves at most half of the smaller amount of a
species. With charged species, no screening and only canonical ions each box stays neutral:
M_P = M - M z zᵀM/(zᵀMz) moves only neutral combinations, and the arbitrary uniform potential of
each box drops out of the step. Grand-canonical point ions have the same chemical potential in both
boxes and are not exchanged; if they are charged, the uniform potential of each box is its Donnan
potential against their reservoir, every species moves on its own and the boxes need not keep
their canonical charge. At coexistence μ̃ is the same in the boxes (in every neutral combination
when only those move); the pressures need not agree.

Each cycle builds the two boxes from the input (polymers, electrostatics, ...) with their
volume fractions, ion concentrations (counterion: None) and box lengths, which scale with
V_b^(1/d) from the given ones at the same grid. A box continues from the fields, ψ and the
previous noises of the Leimkuhler-Matthews step at the end of its previous cycle (its fields
file, at the same grid values) with a random seed of its own for each cycle. Every species needs
a positive amount in both boxes, since an absent species could not enter a box. The uniform modes of the fields move
with the homogeneous saddle point, w_i = Σ_j χ_ij N φ_j and ψ₀ (electrostatics.py,
homogeneous_potential), when the composition changes: in canonical CL-FTS the uniform mode of
the pressure field relaxes only at the rate 2/(χN + 2ζN) per unit time, and it shifts the
chemical potentials and the pressure. Without initial fields a box starts at its homogeneous
saddle point. The global step numbers continue, and
the CL-FTS output of each box goes to <dir>/box<b> (clfts.log holds its console output).
gibbs_history.json holds the settings and every cycle; continue_run() continues an interrupted
run from it as the uninterrupted run would.
"""

import contextlib
import copy
import json
import os

import numpy as np
from scipy.io import loadmat

from .clfts import CLFTS
from .electrostatics import homogeneous_potential, point_ions
from .utils import json_ready, input_hash
from .validation import (ValidationError, normalize_params, open_charges, parse_chi_n, point_ion_concentrations,
                         validate_gibbs_params, validate_lfts_params)


def cell_volume(lx, angles=None):
    """Volume of a cell with the lengths lx and the angles [alpha, beta, gamma] (or [gamma] in 2D) in degrees."""
    lx = [float(v) for v in lx]
    if not angles:
        return float(np.prod(lx))
    if len(lx) == 2:
        gamma = np.deg2rad(angles[0] if len(angles) == 1 else angles[2])
        return lx[0] * lx[1] * float(np.sin(gamma))
    cosines = np.cos(np.deg2rad(angles))
    factor = 1.0 - np.sum(cosines ** 2) + 2.0 * np.prod(cosines)
    return float(np.prod(lx) * np.sqrt(factor))


class GibbsEnsemble:
    """Two CL-FTS boxes that exchange chains, point ions and volume toward coexistence.

    Parameters
    ----------
    params : dict
        CL-FTS input with the section `gibbs_ensemble`: `boxes` (two dicts with `lx`,
        `volume_fractions` of the polymers and optionally `ion_concentrations` of the canonical
        point ions, by default those of `electrostatics` with the counterions), `cycles`,
        `cycle_steps` (Langevin steps of each box per cycle), `exchange_rate` (default 0.2) and
        `skip_steps` (steps at the start of each cycle left out of the averages, default 0).
        `recording.dir` holds the boxes and the history; `recording.sample_interval` sets the
        samples of the averages. The volume fractions of `polymers` are not used.
    random_seed : int, optional
        Seed of the noise of both boxes in all cycles; drawn and recorded if omitted.
    """

    HISTORY = "gibbs_history.json"

    def __init__(self, params, random_seed=None):
        params = normalize_params(copy.deepcopy(params))
        validate_gibbs_params(params)
        if random_seed is not None and (isinstance(random_seed, (bool, np.bool_))
                                        or not isinstance(random_seed, (int, np.integer)) or random_seed < 0):
            raise ValidationError(f"random_seed must be a non-negative integer, got {random_seed!r}")
        self.params = params
        gibbs = params["gibbs_ensemble"]
        self.cycles = int(gibbs["cycles"])
        self.cycle_steps = int(gibbs["cycle_steps"])
        self.exchange_rate = float(gibbs.get("exchange_rate", 0.2))
        self.skip_steps = int(gibbs.get("skip_steps", 0))
        self.directory = str(params["recording"]["dir"])
        self.sample_interval = int(params["recording"].get("sample_interval", 1))
        self.random_seed = int(np.random.SeedSequence().entropy % (2 ** 63)) if random_seed is None else int(random_seed)
        self.dim = len(params["nx"])
        self.angles = params.get("angles")

        # Species that move between the boxes: the polymers, then the canonical point ions
        self.alphas = np.array([sum(block["length"] for block in polymer["blocks"]) for polymer in params["polymers"]])
        n_polymers = len(self.alphas)
        es = params.get("electrostatics")
        ions = (es.get("ions") or {}) if es is not None else {}
        self.ion_names = [name for name, ion in ions.items() if "concentration" in ion]
        # Every point ion in the input order, which the chemical potentials of the boxes follow
        self.point_ion_names = list(ions)
        self.n_segments = None
        if self.ion_names:
            self.n_segments = (round(1.0 / params["ds"]) if str(params["chain_model"]).lower() == "discrete"
                               else float(es["n_segments"]))
        charges = (es or {}).get("charges", {})
        # Charge of a chain, N Σ_i z_i (length of type i), in the units of the ion valences
        n_segments = (round(1.0 / params["ds"]) if str(params["chain_model"]).lower() == "discrete"
                      else (es or {}).get("n_segments", 1.0))
        chain_charges = [n_segments * sum(charges.get(block["type"], 0.0) * block["length"] for block in polymer["blocks"])
                         for polymer in params["polymers"]]
        self.charges = np.array(chain_charges + [float(ions[name]["valence"]) for name in self.ion_names])
        # Charged grand-canonical ions fix the uniform potential of each box (Donnan): then all species move
        self.neutral = (es is not None and es.get("screening_length") is None and not open_charges(params)
                        and bool(np.any(self.charges != 0.0)))

        # Amounts of every species in each box from the given volume fractions and concentrations
        self.lx0 = [[float(v) for v in box["lx"]] for box in gibbs["boxes"]]
        self.volume0 = [cell_volume(lx, self.angles) for lx in self.lx0]
        amounts = []
        for b, box in enumerate(gibbs["boxes"]):
            volume = self.volume0[b]
            chains = [phi * volume / alpha for phi, alpha in zip(box["volume_fractions"], self.alphas)]
            concentrations = box.get("ion_concentrations")
            if concentrations is None and self.ion_names:
                template = copy.deepcopy(params)
                for polymer, phi in zip(template["polymers"], box["volume_fractions"]):
                    polymer["volume_fraction"] = phi
                concentrations = point_ion_concentrations(template)
            ions_amounts = [self.n_segments * concentrations[name] * volume for name in self.ion_names]
            amounts.append(chains + ions_amounts)
        self.amounts = np.array(amounts, dtype=np.float64)
        self.initial_amounts = self.amounts.copy()
        self.n_polymers = n_polymers
        names = [f"polymers[{p}]" for p in range(n_polymers)] + self.ion_names
        for b in range(2):
            absent = [name for name, amount in zip(names, self.amounts[b]) if not amount > 0]
            if absent:
                raise ValidationError(f"gibbs_ensemble.boxes[{b}] has none of {absent}: every species needs a positive "
                                      "amount in both boxes (leave ions that are absent out of 'electrostatics.ions')")
        if self.neutral:
            for b in range(2):
                if abs(self.amounts[b] @ self.charges) > 1e-10 * max(1.0, np.abs(self.amounts[b]) @ np.abs(self.charges)):
                    raise ValidationError(f"gibbs_ensemble.boxes[{b}] is not electroneutral; give its ion_concentrations")
        # Each box must be a valid CL-FTS input
        for b in range(2):
            validate_lfts_params(self.box_params(b, self.amounts[b], 0), complex_fields=True)
        self.history = None

    def settings(self):
        """Settings that a continuation must share (the number of cycles may grow)."""
        gibbs = {key: value for key, value in self.params["gibbs_ensemble"].items() if key != "cycles"}
        template = {key: value for key, value in self.params.items() if key != "gibbs_ensemble"}
        template["langevin"] = {key: value for key, value in template["langevin"].items() if key != "max_step"}
        # The amounts and the chemical potentials of the history follow the order of the point ions;
        # the hash of the input does not see that order
        return {"input_hash": input_hash(json_ready(template)), "gibbs_ensemble": json_ready(gibbs),
                "random_seed": self.random_seed, "ion_names": self.point_ion_names}

    def box_params(self, b, amounts, cycle):
        """CL-FTS input of box b with the given amounts for the given cycle."""
        params = copy.deepcopy(self.params)
        params.pop("gibbs_ensemble")
        chains = amounts[:self.n_polymers]
        volume = float(np.dot(self.alphas, chains))
        params["lx"] = [length * (volume / self.volume0[b]) ** (1.0 / self.dim) for length in self.lx0[b]]
        for p, polymer in enumerate(params["polymers"]):
            polymer.pop("chemical_potential", None)
            polymer["volume_fraction"] = float(self.alphas[p] * chains[p] / volume)
        if self.ion_names:
            es = params["electrostatics"]
            for name, amount in zip(self.ion_names, amounts[self.n_polymers:]):
                es["ions"][name]["concentration"] = float(amount / (self.n_segments * volume))
            es["counterion"] = None
        recording = params["recording"]
        recording["dir"] = os.path.join(self.directory, f"box{b}")
        recording["checkpoint_interval"] = self.cycle_steps
        recording["statistics_save_interval"] = self.cycle_steps
        recording["pressure"] = True
        params["langevin"]["max_step"] = (cycle + 1) * self.cycle_steps
        return params

    def _homogeneous_fields(self, params):
        """Monomer fields w_i = Σ_j χ_ij N φ_j and ψ₀ of the homogeneous saddle point of a box input
        (the volume fractions sum to 1, so the compressibility adds nothing)."""
        types = sorted(params["segment_lengths"])
        fractions = np.zeros(len(types))
        for polymer in params["polymers"]:
            length = sum(block["length"] for block in polymer["blocks"])
            for block in polymer["blocks"]:
                shares = block.get("monomer_fractions", {block["type"]: 1.0})
                for monomer, share in shares.items():
                    fractions[types.index(monomer)] += polymer["volume_fraction"] * block["length"] / length * share
        chi_n = parse_chi_n(params["chi_n"], params["segment_lengths"])
        matrix = np.zeros((len(types), len(types)))
        for key, value in chi_n.items():
            i, j = (types.index(name) for name in key.split(","))
            matrix[i, j] = matrix[j, i] = value
        w = matrix @ fractions
        es = params.get("electrostatics")
        if es is None:
            return w, 0.0
        n_segments = (round(1.0 / params["ds"]) if str(params["chain_model"]).lower() == "discrete" else es.get("n_segments"))
        coupling = (float(es["coupling"]) if "coupling" in es else
                    4.0 * np.pi * float(es["bjerrum_length"]) * n_segments ** 2 * np.sqrt(params["langevin"]["nbar"]))
        kappa_sq = 0.0 if es.get("screening_length") is None else 1.0 / float(es["screening_length"]) ** 2
        ions = point_ions(es, {name: ion.get("concentration") for name, ion in (es.get("ions") or {}).items()})
        polymer_charge = sum(float(es["charges"].get(name, 0.0)) * value for name, value in zip(types, fractions))
        psi0 = homogeneous_potential(polymer_charge, ions, coupling, kappa_sq, n_segments) if coupling > 0 else 0.0
        return w, psi0

    def _seed(self, cycle, b):
        return int(np.random.SeedSequence([self.random_seed, cycle, b]).generate_state(1, dtype=np.uint64)[0] % (2 ** 63))

    def _box_directory(self, b):
        return os.path.join(self.directory, f"box{b}")

    def _write_history(self):
        path = os.path.join(self.directory, self.HISTORY)
        text = json.dumps(json_ready(self.history), indent=1, allow_nan=False)
        with open(path + ".tmp", "w") as f:
            f.write(text)
        os.replace(path + ".tmp", path)

    def run(self, initial_fields=None, initial_psi=None):
        """Run all cycles from the given fields of the boxes.

        initial_fields : a dict {monomer type: field} for both boxes or a list of two such dicts;
        the homogeneous saddle point of each box if omitted. initial_psi : ψ for both boxes or a list
        of two (with electrostatics); the homogeneous ψ₀ if omitted.
        Returns the averages of the second half of the cycles (summary()).
        """
        os.makedirs(self.directory, exist_ok=True)
        self.amounts = self.initial_amounts.copy()
        self.history = {"settings": self.settings(), "cycles": []}
        # A history of an earlier run must not be continued with the boxes of this one
        self._write_history()
        fields = initial_fields if isinstance(initial_fields, (list, tuple)) else [initial_fields, initial_fields]
        psi = initial_psi if isinstance(initial_psi, (list, tuple)) else [initial_psi, initial_psi]
        return self._run_cycles(0, fields, psi)

    def continue_run(self):
        """Continue the cycles of gibbs_history.json in recording.dir up to `cycles`."""
        with open(os.path.join(self.directory, self.HISTORY)) as f:
            self.history = json.load(f)
        if not self.history["cycles"]:
            raise ValidationError("The Gibbs-ensemble history has no completed cycle; start again with run()")
        stored = dict(self.history["settings"])
        self.random_seed = int(stored["random_seed"])
        current = json.loads(json.dumps(json_ready(self.settings())))
        # Histories of earlier versions do not store the order of the ions
        ion_names = current.pop("ion_names")
        stored_names = stored.pop("ion_names", ion_names)
        if stored_names != ion_names and sorted(stored_names) == sorted(ion_names):
            raise ValidationError("'electrostatics.ions' lists the point ions in another order than the "
                                  "Gibbs-ensemble history; its amounts and chemical potentials follow that order")
        if stored != current or stored_names != ion_names:
            raise ValidationError("The input differs from that of the Gibbs-ensemble history (only 'cycles' may change)")
        if self.history["cycles"]:
            self.amounts = np.array(self.history["cycles"][-1]["amounts_after"], dtype=np.float64)
        return self._run_cycles(len(self.history["cycles"]), [None, None], [None, None])

    def _run_cycles(self, first_cycle, fields, psi):
        for cycle in range(first_cycle, self.cycles):
            boxes = [self._run_box(cycle, b, fields[b] if cycle == 0 else None, psi[b] if cycle == 0 else None,
                                   draw=cycle == first_cycle) for b in range(2)]
            amounts_before = self.amounts.copy()
            step, difference = self._exchange_step([box["exchange_potentials"] for box in boxes])
            self.amounts[0] += step
            self.amounts[1] -= step
            self.history["cycles"].append({"cycle": cycle, "last_step": (cycle + 1) * self.cycle_steps,
                                           "amounts_before": amounts_before.tolist(), "boxes": boxes,
                                           "amounts_after": self.amounts.tolist()})
            self._write_history()
            print("Gibbs cycle %d: volumes %s, volume fractions %s / %s, max |Δμ̃| %.3e"
                  % (cycle, [round(box["volume"], 6) for box in boxes],
                     np.round(boxes[0]["volume_fractions"], 5).tolist(),
                     np.round(boxes[1]["volume_fractions"], 5).tolist(), float(np.max(np.abs(difference)))))
        return self.summary()

    def _run_box(self, cycle, b, initial_fields, initial_psi, draw):
        """One cycle of box b; returns its state and the averages of its operators."""
        params = self.box_params(b, self.amounts[b], cycle)
        first_step = cycle * self.cycle_steps + 1
        directory = self._box_directory(b)
        os.makedirs(directory, exist_ok=True)
        monomer_types = sorted(params["segment_lengths"])
        w_saddle, psi_saddle = self._homogeneous_fields(params)
        noise = psi_noise = None
        if cycle > 0:
            # The fields, the previous noise and ψ at the end of the last cycle of this box, with the
            # uniform modes moved as the homogeneous saddle point moves with the composition
            data = loadmat(os.path.join(directory, "fields_%06d.mat" % (first_step - 1)), squeeze_me=True)
            previous = self.box_params(b, np.asarray(self.history["cycles"][-1]["amounts_before"][b]), cycle - 1)
            w_previous, psi_previous = self._homogeneous_fields(previous)
            initial_fields = {m: np.asarray(data["w_" + m + "_real"]) + 1j * np.asarray(data["w_" + m + "_imag"])
                              + (w_saddle[i] - w_previous[i]) for i, m in enumerate(monomer_types)}
            noise = data["normal_noise_prev"]
            if "psi_real" in data:
                initial_psi = (np.asarray(data["psi_real"]) + 1j * np.asarray(data["psi_imag"])
                               + (psi_saddle - psi_previous))
                psi_noise = data["psi_noise_prev"]
        with open(os.path.join(directory, "clfts.log"), "a") as log, contextlib.redirect_stdout(log):
            sim = CLFTS(params, random_seed=self._seed(cycle, b))
            sim.draw_architectures = draw
            n_grid = sim.prop_solver.n_grid
            if initial_fields is None:
                initial_fields = {m: np.full(n_grid, w_saddle[i]) for i, m in enumerate(monomer_types)}
            if initial_psi is None and sim.electrostatics is not None and sim.electrostatics.coupling > 0 \
                    and not sim.electrostatics.zero_uniform_mode:
                initial_psi = np.full(n_grid, psi_saddle)
            sim.run(initial_fields, normal_noise_prev=noise, start_langevin_step=first_step, initial_psi=initial_psi,
                    psi_noise_prev=psi_noise)
        # Averages of the cycle, without the samples of its first skip_steps steps
        data = loadmat(os.path.join(directory, "dH_%06d.mat" % ((cycle + 1) * self.cycle_steps)), squeeze_me=True)
        skip = self.skip_steps // self.sample_interval
        mu = np.reshape(np.real(data["chemical_potential_history_real"]), (-1, sim.n_chemical_potentials))[skip:]
        pressure = np.atleast_1d(np.real(data["pressure_history_real"]))[skip:]
        mu_mean, pressure_mean = np.mean(mu, axis=0), float(np.mean(pressure))
        ion_order = [ion["name"] for ion in sim.electrostatics.ions] if sim.electrostatics is not None else []
        mu_polymers = mu_mean[:self.n_polymers]
        mu_ions = [mu_mean[self.n_polymers + ion_order.index(name)] for name in self.ion_names]
        exchange = np.concatenate([mu_polymers - self.alphas * pressure_mean, mu_ions])
        volume = float(np.dot(self.alphas, self.amounts[b][:self.n_polymers]))
        return {"volume": volume, "lx": params["lx"],
                "volume_fractions": [polymer["volume_fraction"] for polymer in params["polymers"]],
                "ion_concentrations": {name: params["electrostatics"]["ions"][name]["concentration"]
                                       for name in self.ion_names},
                "chemical_potentials": mu_mean.tolist(), "pressure": pressure_mean,
                "exchange_potentials": exchange.tolist(), "samples": int(len(pressure))}

    def _exchange_step(self, exchange_potentials):
        """Δn of box I and the difference μ̃^I - μ̃^II that drives it, from the exchange potentials of
        the two boxes. In a neutral system the difference is taken without its component along the
        charges z (the arbitrary uniform potentials): Δn = -r M δ_P with δ_P = δ - z (zᵀMδ)/(zᵀMz),
        which is -r M_P δ."""
        n0, n1 = self.amounts
        mobility = self.exchange_rate * n0 * n1 / (n0 + n1)
        difference = np.asarray(exchange_potentials[0]) - np.asarray(exchange_potentials[1])
        if self.neutral:
            weighted = mobility * self.charges
            difference = difference - self.charges * float(weighted @ difference) / float(self.charges @ weighted)
        step = -mobility * difference
        # At most half of the smaller amount of a species per step
        ratio = float(np.max(np.abs(step) / np.minimum(n0, n1)))
        if ratio > 0.5:
            step = step * (0.5 / ratio)
        return step, difference

    def summary(self):
        """Averages over the second half of the cycles: volumes, volume fractions, ion concentrations,
        chemical potentials, pressures and exchange potentials of both boxes."""
        cycles = self.history["cycles"]
        kept = cycles[len(cycles) // 2:]
        result = {"cycles": len(cycles), "averaged_cycles": len(kept), "boxes": []}
        for b in range(2):
            boxes = [cycle["boxes"][b] for cycle in kept]
            result["boxes"].append({
                "volume": float(np.mean([box["volume"] for box in boxes])),
                "volume_fractions": np.mean([box["volume_fractions"] for box in boxes], axis=0).tolist(),
                "ion_concentrations": {name: float(np.mean([box["ion_concentrations"][name] for box in boxes]))
                                       for name in self.ion_names},
                "chemical_potentials": np.mean([box["chemical_potentials"] for box in boxes], axis=0).tolist(),
                "pressure": float(np.mean([box["pressure"] for box in boxes])),
                "exchange_potentials": np.mean([box["exchange_potentials"] for box in boxes], axis=0).tolist()})
        return result
