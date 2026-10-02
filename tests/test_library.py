# -*- coding: utf-8 -*-
"""预置示例模型的**行为**测试 —— 不只测"能跑"，要测"算得对"。

三个模型都取自 JD 的业务域（BMS / 电源控制），所以断言直接对着物理/工程含义写：
Buck 的稳态占空比应当等于 Vo/Vin；SOC 的库仑计数应当与理论安时数一致；
OCV 修正只能在静置时起作用。
"""
import math
import unittest

from mbd import library


class TestBuckVoltageLoop(unittest.TestCase):

    def setUp(self):
        self.m = library.get('buck_voltage_loop')      # Vin=24, vref=12, tau=10ms

    def _settle(self, vref, steps=4000):
        return self.m.simulate(steps, inputs={0: vref}, probe=['vout', 'duty', 'err'])

    def test_converges_to_reference(self):
        s = self._settle(12.0)
        self.assertAlmostEqual(s['vout'][-1], 12.0, places=6)

    def test_steady_state_duty_equals_vo_over_vin(self):
        """Buck 稳态占空比 d = Vo / Vin —— 这是最基本的物理关系，算错就说明模型不对。"""
        s = self._settle(12.0)
        self.assertAlmostEqual(s['duty'][-1], 12.0 / 24.0, places=6)

    def test_second_reference_also_tracks(self):
        s = self._settle(15.0)
        self.assertAlmostEqual(s['vout'][-1], 15.0, places=5)
        self.assertAlmostEqual(s['duty'][-1], 15.0 / 24.0, places=5)

    def test_duty_stays_within_limits(self):
        s = self._settle(12.0)
        for v in s['duty']:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_error_vanishes(self):
        s = self._settle(12.0)
        self.assertLess(abs(s['err'][-1]), 1e-6)

    def test_output_never_overshoots_wildly(self):
        """稳定裕度检查：超调不该超过给定值的 30%（参数选得太激进就会在这里挂掉）。"""
        s = self._settle(12.0)
        self.assertLess(max(s['vout']), 12.0 * 1.3)


class TestSocEstimator(unittest.TestCase):

    def setUp(self):
        self.m = library.get('soc_estimator')          # Q=50Ah, dt=1s, soc0=0.60

    def _run(self, current, volts, seconds):
        inp = {i: {0: current, 1: volts} for i in range(seconds)}
        return self.m.simulate(seconds, inputs=inp, probe=['soc', 'soc_rate', 'mode', 'ocv_soc'])

    def test_coulomb_counting_matches_theory(self):
        """2 A 放电 1800 s = 1 Ah；50 Ah 电池应当是 2% 的 SOC 下降。"""
        s = self._run(2.0, 3.85, 1800)
        drop = 0.60 - s['soc'][-1]
        self.assertAlmostEqual(drop, 1.0 / 50.0, delta=0.002)

    def test_charge_raises_soc(self):
        s = self._run(-3.0, 4.05, 1200)
        self.assertGreater(s['soc'][-1], 0.60)

    def test_state_machine_enters_rest(self):
        s = self._run(0.0, 3.88, 10)
        self.assertEqual(s['mode'][-1], 1.0)          # rest -> 状态码 1

    def test_state_machine_enters_discharge(self):
        s = self._run(2.0, 3.85, 10)                  # 正电流 = 放电
        self.assertEqual(s['mode'][-1], 0.0)

    def test_state_machine_distinguishes_charge_from_discharge(self):
        """三个状态必须给出**不同的状态码**，否则从输出上分不出充放电（这是踩过的坑）。"""
        dis = self._run(2.0, 3.85, 10)
        chg = self._run(-3.0, 4.05, 10)
        self.assertEqual(dis['mode'][-1], 0.0)
        self.assertEqual(chg['mode'][-1], 2.0)

    def test_hysteresis_holds_state_in_deadband(self):
        """迟滞：进入静置要 <=0.05 A，但已在静置时要 >0.10 A 才离开 ——
        0.08 A 落在死区里，状态应当保持不动。"""
        inp = {}
        for i in range(20):
            inp[i] = {0: 2.0, 1: 3.85}      # 先放电
        for i in range(20, 60):
            inp[i] = {0: 0.00, 1: 3.88}     # 进静置
        for i in range(60, 100):
            inp[i] = {0: 0.08, 1: 3.87}     # 死区内的小电流
        s = self.m.simulate(100, inputs=inp, probe=['mode'])
        self.assertEqual(s['mode'][60], 1.0)
        self.assertEqual(s['mode'][-1], 1.0, '0.08 A 在死区内，不该离开静置')
        # 超过 0.10 A 才离开
        inp2 = dict(inp)
        for i in range(60, 100):
            inp2[i] = {0: 0.20, 1: 3.87}
        s2 = self.m.simulate(100, inputs=inp2, probe=['mode'])
        self.assertEqual(s2['mode'][-1], 0.0)

    def test_rest_gate_only_opens_at_rest(self):
        """查表把状态码映射成修正增益：只有静置（1）非零。"""
        rest = self._run(0.0, 3.88, 50)
        loaded = self._run(2.0, 3.85, 50)
        self.assertGreater(abs(rest['soc_rate'][-1]), 0.0)
        self.assertAlmostEqual(loaded['soc_rate'][-1], -2.0 / (3600.0 * 50.0), places=12)

    def test_no_ocv_correction_while_loaded(self):
        """带载时不修正：库仑与修正速率之和应当只等于库仑速率本身。"""
        s = self._run(2.0, 3.85, 100)
        self.assertAlmostEqual(s['soc_rate'][-1], -2.0 / (3600.0 * 50.0), places=12)

    def test_ocv_correction_pulls_toward_table(self):
        """静置后修正应当把 SOC 往 OCV 查表值拉 —— 这是 SOX 算法的核心动作。"""
        s = self._run(0.0, 3.88, 3000)
        self.assertGreater(s['soc'][-1], s['soc'][0])
        self.assertLess(abs(s['soc'][-1] - s['ocv_soc'][-1]),
                        abs(s['soc'][0] - s['ocv_soc'][0]))

    def test_soc_stays_in_range(self):
        s = self._run(50.0, 3.85, 200)                # 大电流长时间放电
        for v in s['soc']:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_ocv_table_is_monotonic(self):
        v = library.OCV_SOC_V
        s = library.OCV_SOC_S
        self.assertEqual(len(v), len(s))
        for i in range(1, len(v)):
            self.assertGreater(v[i], v[i - 1])
            self.assertGreaterEqual(s[i], s[i - 1])


class TestRateLimitedTracker(unittest.TestCase):

    def test_step_response_reaches_target_in_expected_steps(self):
        m = library.get('rate_limited_tracker')       # rise=10/s, dt=1ms -> 0.01 每步
        s = m.simulate(200, inputs={0: 1.0}, probe=['ramp'])
        self.assertAlmostEqual(s['ramp'][100], 1.0, places=9)
        self.assertAlmostEqual(s['ramp'][99], 0.99, places=9)


class TestLibrary(unittest.TestCase):

    def test_all_models_validate_and_order(self):
        for name in sorted(library.LIBRARY):
            m = library.get(name)
            m.validate()
            self.assertEqual(len(m.order()), len(m.blocks))

    def test_all_models_have_at_least_one_stateful_block(self):
        for name in sorted(library.LIBRARY):
            m = library.get(name)
            self.assertGreater(m.summary()['stateful_blocks'], 0)

    def test_get_unknown(self):
        with self.assertRaises(KeyError):
            library.get('nope')

    def test_models_are_independent_instances(self):
        a = library.get('buck_voltage_loop')
        b = library.get('buck_voltage_loop')
        a.simulate(10, inputs={0: 1.0})
        b.simulate(10, inputs={0: 99.0})
        self.assertIsNot(a, b)

    def test_sample_time_is_positive(self):
        for name in sorted(library.LIBRARY):
            m = library.get(name)
            self.assertGreater(m.dt, 0.0)
            self.assertTrue(math.isfinite(m.dt))


if __name__ == '__main__':
    unittest.main(verbosity=2)
