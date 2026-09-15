"""Independent thermodynamic and chain-response checks for the bulk model.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 workflow.py test mccarty
Native chain/FFT implementations and existing test parameters are unchanged.
"""
import unittest
from dataclasses import FrozenInstanceError, replace

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from models.mccarty import (McCartyBulk, SEQUENCES, PAPER_SCD, CLOSURES,
                            charges, scd, form_factors, sequence_blocks)


class McCartyTests(unittest.TestCase):
    def test_published_sequences(self):
        for name in SEQUENCES:
            q = charges(name)
            self.assertEqual(len(q), 50)
            self.assertEqual(sum(q), 0)
            self.assertEqual(round(scd(name), 2), PAPER_SCD[name])
            blocks = sequence_blocks(name)
            expanded = ''.join(t*round(length*50) for t, length, _, _ in blocks)
            self.assertEqual(expanded, SEQUENCES[name])

    def test_form_factors_direct_pairs(self):
        k = np.array([0., 1e-7, .07, .8, 4., 25.])
        for name in SEQUENCES:
            q = charges(name)
            dist = abs(np.arange(50)[:, None]-np.arange(50)[None, :])
            direct = np.exp(-k[:, None, None]**2*dist/6)
            expected = [direct.sum(axis=(1, 2))/50,
                        np.einsum('kij,j->k', direct, q)/50,
                        np.einsum('i,kij,j->k', q, direct, q)/50]
            assert_allclose(form_factors(name, k), expected, atol=1e-13, rtol=1e-11)
        nn, nq, qq = form_factors('sv30', [0., 50.])
        assert_allclose(nn, [50., 1.], atol=1e-14)
        assert_allclose(nq, [0., 0.], atol=1e-14)
        assert_allclose(qq, [0., 1.], atol=1e-14)

    def test_independent_adaptive_quadrature(self):
        m = McCartyBulk('sv20', bjerrum=.33, closure='paper-rpa')
        rho = .73
        q = charges('sv20')
        dist = abs(np.arange(50)[:, None]-np.arange(50)[None, :])
        def response(k):
            pair = np.exp(-k*k*dist/6)
            h2 = np.exp(-m.smear*m.smear*k*k)
            return np.array([m.excluded_volume*h2*pair.sum()/50,
                             4*np.pi*m.bjerrum*h2*(q@pair@q)/(50*k*k)])
        f = quad(lambda k:k*k*np.log1p(rho*response(k)).sum()/(4*np.pi**2),
                 1e-8, np.inf, epsabs=1e-11, epsrel=1e-10)[0]
        mu = quad(lambda k:k*k*(response(k)/(1+rho*response(k))).sum()/(4*np.pi**2),
                  1e-8, np.inf, epsabs=1e-11, epsrel=1e-10)[0]
        ideal = rho/50*(np.log(rho/50)-1)+m.excluded_volume*rho*rho/2
        self.assertAlmostEqual(m.free_energy(rho), ideal+f, delta=1e-10)
        self.assertAlmostEqual(m.mu_chain(rho), np.log(rho/50)+50*(m.excluded_volume*rho+mu), delta=1e-9)

    def test_thermodynamic_derivatives(self):
        for closure in CLOSURES:
            m = McCartyBulk('sv20', closure=closure)
            r, h = .43, 1e-4
            derivative = (m.free_energy(r-2*h)-8*m.free_energy(r-h)
                          +8*m.free_energy(r+h)-m.free_energy(r+2*h))/(12*h)
            self.assertAlmostEqual(derivative, m.mu_segment(r), delta=1e-10)
            dmu = (m.mu_segment(r+h)-m.mu_segment(r-h))/(2*h)
            self.assertAlmostEqual(dmu, m.curvature(r), delta=1e-8)
            rho = np.geomspace(1e-7, 20, 50)
            assert_allclose(m.pressure(rho), rho*m.mu_segment(rho)-m.free_energy(rho), atol=1e-13)

    def test_dilute_pair_integral(self):
        q = charges('sv20'); n = len(q)
        d = abs(np.arange(n)[:, None]-np.arange(n)[None, :])
        m = McCartyBulk('sv20', closure='paper-rpa')
        a2 = m.smear**2+d/6
        electro = m.bjerrum/(2*np.sqrt(np.pi))*np.sum(q[:, None]*q[None, :]/np.sqrt(a2))
        excluded = m.excluded_volume/(16*np.pi**1.5)*np.sum(a2**-1.5)
        rho = 1e-12
        self.assertAlmostEqual(m.mu_chain(rho)-np.log(rho/n), electro+excluded, delta=1e-7)

    def test_symmetries_and_cross_channel(self):
        seq = SEQUENCES['sv20']; rho = np.array([.01, .3, 3.])
        original = McCartyBulk(seq, closure='coupled-rpa')
        for other in [seq[::-1], seq.translate(str.maketrans('EK', 'KE'))]:
            assert_allclose(original.free_energy(rho), McCartyBulk(other, closure='coupled-rpa').free_energy(rho), atol=1e-13)
        # Antisymmetric diblock has exactly zero density-charge cross response.
        assert_allclose(McCartyBulk('sv30', closure='coupled-rpa').free_energy(rho),
                        McCartyBulk('sv30', closure='paper-rpa').free_energy(rho), atol=1e-13)
        self.assertGreater(np.max(abs(original.gnq)), .1)
        self.assertGreater(abs(original.free_energy(.3)-McCartyBulk(seq, closure='paper-rpa').free_energy(.3)), 1e-7)

    def test_no_electrostatic_limit(self):
        m = McCartyBulk('sv30', bjerrum=0.)
        r = np.geomspace(1e-9, 10, 20)
        assert_allclose(m.pressure(r), r/50+m.excluded_volume*r*r/2, rtol=1e-14)
        self.assertFalse(m.coexistence()['two_phase'])

    def test_critical_and_coexistence(self):
        critical = []
        for seq in SEQUENCES:
            m = McCartyBulk(seq)
            cp = m.critical()
            self.assertLess(abs(cp['scaled_curvature']), 1e-10)
            self.assertLess(abs(cp['scaled_third_derivative']), 1e-8)
            critical.append(cp['bjerrum'])
            c = McCartyBulk(seq, bjerrum=1.4*cp['bjerrum']).coexistence()
            self.assertTrue(c['two_phase'])
            self.assertLess(c['rho_dilute'], c['spinodal_dilute'])
            self.assertGreater(c['rho_dense'], c['spinodal_dense'])
            self.assertLess(c['delta_mu_chain'], 1e-8)
            self.assertLess(c['delta_pressure'], 1e-9)
        self.assertTrue(np.all(np.diff(critical) < 0))

    def test_quadrature_convergence(self):
        a = McCartyBulk('sv10', bjerrum=.4, closure='coupled-rpa', quadrature_points=384)
        b = McCartyBulk('sv10', bjerrum=.4, closure='coupled-rpa', quadrature_points=768, cutoff_a=16.)
        rho = np.geomspace(1e-7, 20, 30)
        assert_allclose(a.pressure(rho), b.pressure(rho), atol=1e-10, rtol=1e-9)
        assert_allclose(a.mu_chain(rho), b.mu_chain(rho), atol=1e-9, rtol=1e-9)
        ca, cb = a.coexistence(), b.coexistence()
        assert_allclose([ca['rho_dilute'], ca['rho_dense']], [cb['rho_dilute'], cb['rho_dense']], rtol=1e-8)

    def test_invalid_input(self):
        for kw in [dict(sequence='EEE'), dict(sequence='BAD'), dict(bjerrum=-1),
                   dict(smear=0), dict(excluded_volume=0), dict(closure='CL'),
                   dict(quadrature_points=8), dict(bjerrum=float('nan'))]:
            with self.assertRaises(ValueError):
                McCartyBulk(**kw)
        with self.assertRaises(ValueError):
            McCartyBulk().pressure(0.)
        m = McCartyBulk()
        with self.assertRaises(FrozenInstanceError):
            m.bjerrum = .1
        with self.assertRaises(ValueError):
            m.modes[:] = 0.
        self.assertNotEqual(m.pressure(1.), replace(m, bjerrum=.1).pressure(1.))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(McCartyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
