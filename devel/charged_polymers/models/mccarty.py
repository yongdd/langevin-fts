"""Sequence-resolved, salt-free McCarty bulk Gaussian theories.

Lengths are in b, monomer density in b^-3, and free-energy density in
kBT/b^3. No spatial grid, FFT, graft, ion reservoir, or CL sampler is used.

``electrostatic`` treats the density interaction at its saddle and adds a
Gaussian charge fluctuation determinant, using the complete ideal-chain
charge form factor. This is a PRL-style *bulk approximation*, not Duan SI
S13--S26's self-consistent conformational theory. ``paper-rpa`` implements
McCarty SI S12--S18, including density fluctuations. ``coupled-rpa`` also
retains the density/charge cross response omitted in those SI expressions.
All three keep ideal Gaussian conformations; none reproduces CL by definition.

Source: McCarty et al., JPCL 10, 1644 (2019), doi:10.1021/acs.jpclett.9b00099.
The five sequences were transcribed from Fig. 1, and independently checked
against its rounded sequence charge decoration (SCD) values.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, minimize_scalar


SEQUENCES = {
    'sv10': 'EKKKKKKEEKKKEEEEEKKKEEEKKKEKKEEKEKEEKEKKEKKEEKEEEE',
    'sv15': 'KKEKKEKKKEKKEKKEEEKEKEKKEKKKKEKEKKEEEEEEEEKEEKKEEE',
    'sv20': 'EEKEEEEEEKEEEKEEKKEEEKEKKEKKEKEEKKEKKKKKKKKKKKKEEE',
    'sv25': 'EEEEEEEEEEEKEEEEKEEKEEKEKKKKKKKKKKKKKKKKKKEEKKEEKE',
    'sv30': 'EEEEEEEEEEEEEEEEEEEEEEEEEKKKKKKKKKKKKKKKKKKKKKKKKK',
}
PAPER_SCD = dict(zip(SEQUENCES, [-2.10, -4.35, -7.37, -12.77, -27.84]))
CLOSURES = ('electrostatic', 'paper-rpa', 'coupled-rpa')


def charges(sequence):
    """Return site charges; a name or an explicit E/K sequence is accepted."""
    sequence = SEQUENCES.get(sequence, sequence)
    if not isinstance(sequence, str) or not sequence or set(sequence) - set('EK'):
        raise ValueError('Expected a nonempty E/K sequence or a known sv name')
    return np.array([1. if s == 'K' else -1. for s in sequence])


def scd(sequence):
    q = charges(sequence)
    return float(sum(np.dot(q[:-d], q[d:])*np.sqrt(d)
                     for d in range(1, len(q))) / len(q))


def sequence_blocks(sequence):
    """Existing PropagatorSolver block format, contour normalized to one.

    Use ds=1/N, discrete chains, bond_lengths={'E':1., 'K':1.}, and no
    grafting_points. The native solver's length unit is b*sqrt(N).
    """
    q = charges(sequence)
    ends = np.r_[0, np.flatnonzero(q[1:] != q[:-1])+1, len(q)]
    return [['K' if q[a] > 0 else 'E', (b-a)/len(q), j, j+1]
            for j, (a, b) in enumerate(zip(ends[:-1], ends[1:]))]


def form_factors(sequence, k):
    """Ideal discrete N-bead / N-1-bond responses g_nn, g_nq, g_qq.

    Each is N^-1 sum_ij c_i d_j exp(-k^2*|i-j|/6).
    expm1 preserves the neutral-chain k->0 cancellation.
    """
    q = charges(sequence)
    k = np.asarray(k, dtype=float)
    if np.any(~np.isfinite(k)) or np.any(k < 0):
        raise ValueError('Wave numbers must be finite and nonnegative')
    n = len(q)
    lag = np.arange(1, n)
    e = np.exp(-k[..., None]**2*lag/6)
    em = np.expm1(-k[..., None]**2*lag/6)
    cc = np.array([2*np.dot(q[:-d], q[d:]) for d in lag])
    nc = np.array([q[:-d].sum()+q[d:].sum() for d in lag])
    gnn = 1+2*np.sum(e*(n-lag), axis=-1)/n
    gnq = q.sum()+np.sum(em*nc, axis=-1)/n
    gqq = q.sum()**2/n+np.sum(em*cc, axis=-1)/n
    return gnn, gnq, gqq


@lru_cache(maxsize=16)
def _quadrature(points, cutoff):
    x, w = leggauss(points)
    k = (x+1)*cutoff/2
    return k, w*cutoff/2*k*k/(4*np.pi**2)


@dataclass(frozen=True)
class McCartyBulk:
    sequence: str = 'sv30'
    bjerrum: float = .33
    excluded_volume: float = .0068
    smear: float = 1/np.sqrt(6)
    closure: str = 'electrostatic'
    quadrature_points: int = 384
    cutoff_a: float = 12.

    def __post_init__(self):
        object.__setattr__(self, 'q', charges(self.sequence))
        object.__setattr__(self, 'n', len(self.q))
        if self.q.sum() != 0:
            raise ValueError('Salt-free bulk currently requires a net-neutral chain')
        if self.closure not in CLOSURES:
            raise ValueError(f'closure must be one of {CLOSURES}')
        if (not np.all(np.isfinite([self.bjerrum, self.excluded_volume, self.smear, self.cutoff_a]))
                or self.bjerrum < 0 or min(self.excluded_volume, self.smear, self.cutoff_a) <= 0):
            raise ValueError('Require lB>=0 and positive finite v, a, and cutoff')
        if not isinstance(self.quadrature_points, int) or self.quadrature_points < 32:
            raise ValueError('At least 32 integer quadrature points required')
        k, weights = _quadrature(self.quadrature_points, self.cutoff_a/self.smear)
        object.__setattr__(self, 'k', k)
        object.__setattr__(self, 'weights', weights)
        for name, value in zip(('gnn', 'gnq', 'gqq'), form_factors(self.sequence, k)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'modes', self._modes(self.bjerrum))
        # Frozen parameters prevent stale response caches. Use a new instance
        # (or dataclasses.replace) when changing the physical state.
        for value in (self.q, self.k, self.weights, self.gnn, self.gnq, self.gqq, self.modes):
            value.flags.writeable = False

    def _modes(self, lb):
        smear2 = np.exp(-(self.smear*self.k)**2)
        a = self.excluded_volume*smear2*self.gnn
        c = 4*np.pi*lb*smear2*self.gqq/self.k**2
        if self.closure == 'electrostatic':
            return c[:, None]
        if self.closure == 'paper-rpa':
            return np.stack([a, c], axis=-1)
        b = np.sqrt(self.excluded_volume*4*np.pi*lb)*smear2*self.gnq/self.k
        matrix = np.zeros((len(self.k), 2, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = a, c
        matrix[:, 0, 1] = matrix[:, 1, 0] = b
        eig = np.linalg.eigvalsh(matrix)
        if np.min(eig) < -1e-11:
            raise FloatingPointError('The ideal-chain response is not positive semidefinite')
        return np.maximum(eig, 0.)

    @staticmethod
    def _density(rho):
        rho = np.asarray(rho, dtype=float)
        if np.any(~np.isfinite(rho)) or np.any(rho <= 0):
            raise ValueError('Monomer density must be finite and strictly positive')
        return rho

    def _integral(self, value):
        return np.sum(np.sum(value, axis=-1)*self.weights, axis=-1)

    def free_energy(self, rho):
        rho = self._density(rho)
        x = rho[..., None, None]*self.modes
        return (rho/self.n*(np.log(rho/self.n)-1)
                + self.excluded_volume*rho*rho/2+self._integral(np.log1p(x)))

    def mu_segment(self, rho):
        """df/d rho; multiply by N for the paper's chain chemical potential."""
        rho = self._density(rho)
        x = rho[..., None, None]*self.modes
        return np.log(rho/self.n)/self.n+self.excluded_volume*rho+self._integral(self.modes/(1+x))

    def mu_chain(self, rho):
        return self.n*self.mu_segment(rho)

    def pressure(self, rho):
        rho = self._density(rho)
        x = rho[..., None, None]*self.modes
        value = x/(1+x)-np.log1p(x)
        # Stable pressure at very low density; linear terms cancel exactly.
        small = x < 1e-4
        value[small] = x[small]**2*(-.5+x[small]*(2/3+x[small]*(-.75+.8*x[small])))
        return rho/self.n+self.excluded_volume*rho*rho/2+self._integral(value)

    def curvature(self, rho, modes=None):
        """f''(rho); its sign determines homogeneous local stability."""
        rho = self._density(rho)
        modes = self.modes if modes is None else modes
        response = modes/(1+rho[..., None, None]*modes)
        return 1/(self.n*rho)+self.excluded_volume-self._integral(response**2)

    def third_derivative(self, rho, modes=None):
        rho = self._density(rho)
        modes = self.modes if modes is None else modes
        response = modes/(1+rho[..., None, None]*modes)
        return -1/(self.n*rho*rho)+2*self._integral(response**3)

    def structure_factors(self, rho, k):
        """Normalized number and charge factors; paper-rpa gives SI S17--S18.

        For the electrostatic closure, number response includes its coupling
        to the charge field but no fluctuating excluded-volume field.
        """
        rho = float(self._density(rho))
        k = np.asarray(k, dtype=float)
        if np.any(k <= 0):
            raise ValueError('Use strictly positive k for Coulomb structure factors')
        nn, nq, qq = form_factors(self.sequence, k)
        h2 = np.exp(-(self.smear*k)**2)
        un = self.excluded_volume*h2 if self.closure != 'electrostatic' else np.zeros_like(k)
        uq = 4*np.pi*self.bjerrum*h2/k**2
        if self.closure == 'paper-rpa':
            return nn/(1+rho*un*nn), qq/(1+rho*uq*qq)
        delta = nn*qq-nq*nq
        det = 1+rho*(un*nn+uq*qq)+rho*rho*un*uq*delta
        return (nn+rho*uq*delta)/det, (qq+rho*un*delta)/det

    def critical(self):
        """Find the first critical coupling by minimizing rho*f'' over rho."""
        log_grid = np.linspace(np.log(1e-9), np.log(1e4), 300)
        def minimum(lb, details=False):
            modes = self._modes(lb)
            density = np.exp(log_grid)
            values = density*self.curvature(density, modes)
            j = int(np.argmin(values))
            if j in (0, len(log_grid)-1):
                return (float(values[j]), float(density[j])) if details else float(values[j])
            opt = minimize_scalar(lambda t:float(np.exp(t)*self.curvature(np.exp(t), modes)),
                                  bounds=(log_grid[j-1], log_grid[j+1]), method='bounded',
                                  options={'xatol': 1e-12})
            return (float(opt.fun), float(np.exp(opt.x))) if details else float(opt.fun)
        if minimum(0.) <= 0:
            raise RuntimeError('Reference fluid already unstable at zero electrostatics')
        upper = .05
        while minimum(upper) > 0 and upper < 100:
            upper *= 2
        if minimum(upper) > 0:
            raise RuntimeError('No critical point bracketed below lB/b=100')
        lb = brentq(minimum, 0., upper, xtol=1e-12)
        residual, rho = minimum(lb, True)
        modes = self._modes(lb)
        return dict(bjerrum=lb, temperature=1/lb, rho=rho,
                    scaled_curvature=residual,
                    scaled_third_derivative=float(rho*rho*self.third_derivative(rho, modes)))

    def coexistence(self):
        """Two distinct stable roots with equal pressure and chain mu.

        Spinodal brackets prevent convergence to the trivial equal-density
        solution. A common-tangent test rejects metastable coexistence.
        """
        grid = np.geomspace(1e-12, 1e4, 800)
        values = self.curvature(grid)
        changes = np.flatnonzero(values[1:]*values[:-1] < 0)
        if len(changes) == 0:
            return dict(two_phase=False, reason='No spinodal in the searched density interval')
        if len(changes) != 2:
            raise RuntimeError('Multiple instability intervals require general phase selection')
        vapor_s, liquid_s = [brentq(self.curvature, grid[j], grid[j+1], xtol=1e-14) for j in changes]
        low, high = float(self.mu_segment(liquid_s)), float(self.mu_segment(vapor_s))
        def roots(mu):
            rv = np.exp(brentq(lambda t:float(self.mu_segment(np.exp(t))-mu),
                              -700., np.log(vapor_s), xtol=1e-12))
            rl = np.exp(brentq(lambda t:float(self.mu_segment(np.exp(t))-mu),
                              np.log(liquid_s), np.log(1e4), xtol=1e-12))
            return rv, rl
        eps = (high-low)*1e-10
        def delta_p(mu):
            rv, rl = roots(mu)
            return float(self.pressure(rl)-self.pressure(rv))
        mu = brentq(delta_p, low+eps, high-eps, xtol=1e-13)
        rv, rl = roots(mu)
        points = np.geomspace(rv, rl, 500)
        gap = self.free_energy(points)-self.free_energy(rv)-mu*(points-rv)
        dmu = float(abs(self.mu_chain(rv)-self.mu_chain(rl)))
        dp = float(abs(self.pressure(rv)-self.pressure(rl)))
        if (dmu > 1e-8 or dp > 1e-9 or gap.min() < -1e-9
                or min(self.curvature(rv), self.curvature(rl)) <= 0 or rv >= vapor_s or rl <= liquid_s):
            raise RuntimeError('Coexistence thermodynamic acceptance checks failed')
        return dict(two_phase=True, rho_dilute=float(rv), rho_dense=float(rl),
                    pressure=float(self.pressure(rv)), mu_chain=float(self.n*mu),
                    delta_pressure=dp, delta_mu_chain=dmu, tangent_min_gap=float(gap.min()),
                    spinodal_dilute=float(vapor_s), spinodal_dense=float(liquid_s))
