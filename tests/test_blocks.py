# -*- coding: utf-8 -*-
"""框图块测试：每个块的数值语义。"""
import math
import unittest

from mbd import blocks


class TestStateless(unittest.TestCase):

    def test_constant(self):
        self.assertEqual(blocks.Constant('c', value=3.5).output([], {}), 3.5)

    def test_inport_requires_drive(self):
        b = blocks.Inport('u', index=2)
        st = b.init_state()
        self.assertEqual(b.output([], st), 0.0)
        b.set_input(st, 7.25)
        self.assertEqual(b.output([], st), 7.25)
        self.assertEqual(b.index, 2)

    def test_gain(self):
        self.assertAlmostEqual(blocks.Gain('g', k=2.5).output([4.0], {}), 10.0)

    def test_sum_signs(self):
        self.assertAlmostEqual(blocks.Sum('s', signs='+-').output([5.0, 2.0], {}), 3.0)
        self.assertAlmostEqual(blocks.Sum('s', signs='++').output([5.0, 2.0], {}), 7.0)
        self.assertAlmostEqual(blocks.Sum('s', signs='+--').output([5.0, 2.0, 1.0], {}), 2.0)

    def test_sum_arity_follows_signs(self):
        self.assertEqual(blocks.Sum('s', signs='+++').n_in, 3)

    def test_product(self):
        self.assertAlmostEqual(blocks.Product('p').output([3.0, -4.0], {}), -12.0)

    def test_saturation(self):
        sat = blocks.Saturation('sat', lo=-1.0, hi=1.0)
        self.assertEqual(sat.output([0.5], {}), 0.5)
        self.assertEqual(sat.output([5.0], {}), 1.0)
        self.assertEqual(sat.output([-5.0], {}), -1.0)

    def test_saturation_bad_limits(self):
        with self.assertRaises(ValueError):
            blocks.Saturation('sat', lo=2.0, hi=1.0)

    def test_abs(self):
        self.assertEqual(blocks.Abs('a').output([-3.5], {}), 3.5)

    def test_minmax(self):
        self.assertEqual(blocks.MinMax('m', op='max').output([1.0, 2.0], {}), 2.0)
        self.assertEqual(blocks.MinMax('m', op='min').output([1.0, 2.0], {}), 1.0)
        with self.assertRaises(ValueError):
            blocks.MinMax('m', op='avg')


class TestLookup(unittest.TestCase):

    def setUp(self):
        self.lut = blocks.Lookup1D('lut', x=[0.0, 1.0, 2.0, 4.0], y=[0.0, 10.0, 20.0, 50.0])

    def test_exact_breakpoints(self):
        for x, y in ((0.0, 0.0), (1.0, 10.0), (2.0, 20.0), (4.0, 50.0)):
            self.assertAlmostEqual(self.lut.output([x], {}), y, places=12)

    def test_interpolation(self):
        self.assertAlmostEqual(self.lut.output([0.5], {}), 5.0, places=12)
        self.assertAlmostEqual(self.lut.output([3.0], {}), 35.0, places=12)

    def test_clamped_at_ends(self):
        self.assertAlmostEqual(self.lut.output([-100.0], {}), 0.0, places=12)
        self.assertAlmostEqual(self.lut.output([100.0], {}), 50.0, places=12)

    def test_uneven_spacing(self):
        """断点不等距时二分查找仍然正确 —— 这是最容易写错的一处。"""
        lut = blocks.Lookup1D('lut', x=[0.0, 0.01, 5.0, 5.001, 100.0], y=[0, 1, 2, 3, 4])
        self.assertAlmostEqual(lut.output([0.005], {}), 0.5, places=9)
        self.assertAlmostEqual(lut.output([5.0005], {}), 2.5, places=9)
        # 50 落在段 [5.001, 100] 上（不是 [5, 100]），按线性插值手算：
        expect = 3.0 + (50.0 - 5.001) / (100.0 - 5.001) * (4.0 - 3.0)
        self.assertAlmostEqual(lut.output([50.0], {}), expect, places=12)

    def test_validation(self):
        with self.assertRaises(ValueError):
            blocks.Lookup1D('l', x=[0.0], y=[1.0])
        with self.assertRaises(ValueError):
            blocks.Lookup1D('l', x=[0.0, 1.0], y=[1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            blocks.Lookup1D('l', x=[0.0, 0.0, 1.0], y=[1.0, 2.0, 3.0])


class TestStateful(unittest.TestCase):

    def test_integrator_forward_euler(self):
        b = blocks.Integrator('i', ic=1.0)
        st = b.init_state()
        self.assertEqual(b.output([5.0], st), 1.0)      # 输出只与状态有关
        b.update([5.0], 1.0, st, 0.1)
        self.assertAlmostEqual(st['y'], 1.5, places=12)  # 1 + 0.1*5

    def test_integrator_limits(self):
        b = blocks.Integrator('i', ic=0.0, lo=0.0, hi=1.0)
        st = b.init_state()
        for _ in range(100):
            b.update([10.0], 0.0, st, 0.1)
        self.assertAlmostEqual(st['y'], 1.0, places=12)
        for _ in range(100):
            b.update([-10.0], 0.0, st, 0.1)
        self.assertAlmostEqual(st['y'], 0.0, places=12)

    def test_unit_delay(self):
        b = blocks.UnitDelay('d', ic=2.0)
        st = b.init_state()
        self.assertEqual(b.output([9.0], st), 2.0)
        b.update([9.0], 2.0, st, 1.0)
        self.assertEqual(st['y'], 9.0)

    def test_rate_limiter(self):
        b = blocks.RateLimiter('r', rise=10.0, fall=10.0, ic=0.0)
        st = b.init_state()
        b.update([1.0], 0.0, st, 0.001)
        self.assertAlmostEqual(st['y'], 0.01, places=12)
        for _ in range(200):
            b.update([1.0], 0.0, st, 0.001)
        self.assertAlmostEqual(st['y'], 1.0, places=12)

    def test_rate_limiter_asymmetric(self):
        b = blocks.RateLimiter('r', rise=1.0, fall=10.0, ic=0.0)
        st = b.init_state()
        for _ in range(100):
            b.update([5.0], 0.0, st, 0.001)
        self.assertAlmostEqual(st['y'], 0.1, places=9)   # 上升慢
        for _ in range(100):
            b.update([-5.0], 0.0, st, 0.001)
        self.assertAlmostEqual(st['y'], -0.9, places=9)  # 下降快


class TestFeedthroughFlags(unittest.TestCase):
    """直通标志决定拓扑排序能不能打断代数环，所以它必须是对的。"""

    def test_stateful_blocks_are_not_feedthrough(self):
        for kind in (blocks.Integrator, blocks.UnitDelay, blocks.RateLimiter):
            self.assertFalse(kind.__dict__.get('direct_feedthrough', True),
                             '%s 应当是 direct_feedthrough = False' % kind.__name__)

    def test_pure_blocks_are_feedthrough(self):
        for kind in (blocks.Gain, blocks.Sum, blocks.Product, blocks.Saturation,
                     blocks.Abs, blocks.MinMax, blocks.Lookup1D):
            self.assertTrue(kind.direct_feedthrough, '%s 应当是直通块' % kind.__name__)


class TestRegistry(unittest.TestCase):

    def test_make(self):
        b = blocks.make('Gain', 'g', k=3.0)
        self.assertIsInstance(b, blocks.Gain)
        self.assertEqual(b.output([2.0], {}), 6.0)

    def test_make_unknown(self):
        with self.assertRaises(KeyError):
            blocks.make('NoSuchBlock', 'x')

    def test_arity_check(self):
        b = blocks.Gain('g')
        b.check(1, 1)
        with self.assertRaises(ValueError):
            b.check(2, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
