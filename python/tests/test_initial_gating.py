"""Run with python -m unittest discover -s tests in the compiled environment."""

import unittest

import numpy as np
import fastdyn_fic_dmf as dmf


class InitialGatingTests(unittest.TestCase):
    def setUp(self):
        self.params = dmf.default_params(
            C=np.array([[0.0, 0.2], [0.2, 0.0]]),
            dt=1.0, sigma=0.0, seed=1, batch_size=16,
            return_rate=True, return_bold=False, return_fic=True,
            with_plasticity=False, with_decay=False,
        )

    def test_first_rates_match_initial_state(self):
        # With dt=1 ms, the first saved rates precede the first gating update.
        for sn0, sg0 in ((0.05, 0.02), ([0.03, 0.08], [0.01, 0.04]), (0.0, 1.0)):
            with self.subTest(sn0=sn0, sg0=sg0):
                p = dict(self.params, sn0=sn0, sg0=sg0)
                sn = np.broadcast_to(sn0, (2,))
                sg = np.broadcast_to(sg0, (2,))
                xn = p['I0'] * p['Jexte'] + p['w'] * p['JN'] * sn + p['G'] * p['JN'] * (p['C'] @ sn) - p['J'] * sg
                xg = p['I0'] * p['Jexti'] + p['JN'] * sn - sg
                ye = p['ce'] * (xn - p['Ie'])
                yi = p['ci'] * (xg - p['Ii'])
                rates_e, rates_i, _, _ = dmf.run(p, 4)
                np.testing.assert_allclose(rates_e[:, 0], ye / (-np.expm1(-p['g_e'] * ye)))
                np.testing.assert_allclose(rates_i[:, 0], yi / (-np.expm1(-p['g_i'] * yi)))

    def test_defaults_and_repeated_runs(self):
        self.assertEqual(self.params['sn0'], 0.01)
        self.assertEqual(self.params['sg0'], 0.01)
        expected = dmf.run(self.params, 20)
        for omitted in (('sn0',), ('sg0',), ('sn0', 'sg0'), ()):
            p = self.params.copy()
            for name in omitted:
                del p[name]
            for actual, reference in zip(dmf.run(p, 20), expected):
                np.testing.assert_array_equal(actual, reference)

    def test_invalid_initial_states(self):
        for name in ('sn0', 'sg0'):
            for value in ([], [0.1, 0.2, 0.3], [[0.1, 0.2]], -0.1, 1.1, np.nan, np.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        dmf.run(dict(self.params, **{name: value}), 1)


if __name__ == '__main__':
    unittest.main()
