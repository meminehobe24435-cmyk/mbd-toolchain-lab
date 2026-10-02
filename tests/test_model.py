# -*- coding: utf-8 -*-
"""模型与仿真测试：拓扑排序、代数环、状态更新顺序、输入激励。"""
import unittest

from mbd import Model, blocks
from mbd.model import AlgebraicLoopError
from mbd.statemachine import StateMachine


def _chain():
    """u -> gain(2) -> sum(+1) -> y"""
    m = Model('chain', dt=0.1)
    m.add(blocks.Inport('u', index=0))
    m.add(blocks.Gain('g', k=2.0), {0: 'u'})
    m.add(blocks.Constant('one', value=1.0))
    m.add(blocks.Sum('s', signs='++'), {0: 'g', 1: 'one'})
    return m


class TestTopology(unittest.TestCase):

    def test_order_respects_dependencies(self):
        order = _chain().order()
        self.assertLess(order.index('u'), order.index('g'))
        self.assertLess(order.index('g'), order.index('s'))

    def test_stateful_block_breaks_loop(self):
        """vout(积分器) -> err -> duty -> dv -> vout 是合法闭环，不该被判成代数环。"""
        m = Model('loop', dt=1e-3)
        m.add(blocks.Inport('r', index=0))
        m.add(blocks.Sum('e', signs='+-'), {0: 'r', 1: 'y'})
        m.add(blocks.Gain('k', k=1.0), {0: 'e'})
        m.add(blocks.Integrator('y'), {0: 'k'})
        self.assertEqual(len(m.order()), 4)

    def test_all_feedthrough_loop_is_algebraic(self):
        m = Model('bad', dt=1e-3)
        m.add(blocks.Gain('a', k=1.0), {0: 'b'})
        m.add(blocks.Gain('b', k=1.0), {0: 'a'})
        with self.assertRaises(AlgebraicLoopError):
            m.order()

    def test_unit_delay_makes_loop_legal(self):
        m = Model('ok', dt=1e-3)
        m.add(blocks.Gain('a', k=0.5), {0: 'd'})
        m.add(blocks.UnitDelay('d', ic=1.0), {0: 'a'})
        self.assertEqual(sorted(m.order()), ['a', 'd'])

    def test_validate_rejects_missing_source(self):
        m = Model('m', dt=1e-3)
        m.add(blocks.Gain('g'), {0: 'nope'})
        with self.assertRaises(ValueError):
            m.validate()

    def test_validate_rejects_wrong_arity(self):
        m = Model('m', dt=1e-3)
        m.add(blocks.Constant('c', value=1.0))
        m.add(blocks.Gain('g'), {0: 'c'})
        m.conn.append(('g', 1, 'c', 0))
        with self.assertRaises(ValueError):
            m.validate()

    def test_duplicate_name_rejected(self):
        m = Model('m', dt=1e-3)
        m.add(blocks.Constant('c', value=1.0))
        with self.assertRaises(ValueError):
            m.add(blocks.Constant('c', value=2.0))

    def test_forward_reference_allowed(self):
        m = Model('m', dt=1e-3)
        m.add(blocks.Sum('s', signs='+-'), {0: 'a', 1: 'b'})
        m.add(blocks.Inport('a', index=0))
        m.add(blocks.Inport('b', index=1))
        m.validate()
        self.assertEqual(len(m.order()), 3)


class TestSimulation(unittest.TestCase):

    def test_chain_values(self):
        m = _chain()
        s = m.simulate(3, inputs={0: 5.0}, probe=['s'])
        self.assertEqual(s['s'], [11.0, 11.0, 11.0])

    def test_integrator_accumulates(self):
        m = Model('int', dt=0.5)
        m.add(blocks.Constant('one', value=2.0))
        m.add(blocks.Integrator('i', ic=0.0), {0: 'one'})
        s = m.simulate(4, probe=['i'])
        # 输出是"更新前"的状态：0, 1, 2, 3
        self.assertEqual(s['i'], [0.0, 1.0, 2.0, 3.0])

    def test_state_update_after_all_outputs(self):
        """同一步里，后算的块必须看到前一个块**本步**的输出，而不是已被更新的状态。"""
        m = Model('m', dt=1.0)
        m.add(blocks.Constant('c', value=1.0))
        m.add(blocks.UnitDelay('d', ic=10.0), {0: 'c'})
        m.add(blocks.Sum('s', signs='++'), {0: 'd', 1: 'c'})
        s = m.simulate(2, probe=['d', 's'])
        self.assertEqual(s['d'], [10.0, 1.0])      # d 输出的是上一步的状态
        self.assertEqual(s['s'], [11.0, 2.0])      # 第二步 d=1 + c=1

    def test_input_forms(self):
        m = Model('m', dt=0.1)
        m.add(blocks.Inport('u', index=0))
        m.add(blocks.Gain('g', k=1.0), {0: 'u'})
        # 常数
        self.assertEqual(m.simulate(2, inputs={0: 3.0}, probe=['g'])['g'], [3.0, 3.0])
        # 序列
        self.assertEqual(m.simulate(3, inputs={0: [1.0, 2.0, 3.0]}, probe=['g'])['g'],
                         [1.0, 2.0, 3.0])
        # 按步给
        self.assertEqual(m.simulate(3, inputs={0: {0: 7.0}, 1: {0: 8.0}}, probe=['g'])['g'],
                         [7.0, 8.0, 0.0])

    def test_reset_clears_state(self):
        m = Model('m', dt=1.0)
        m.add(blocks.Constant('c', value=1.0))
        m.add(blocks.Integrator('i', ic=0.0), {0: 'c'})
        m.simulate(5, probe=['i'])
        m.reset()
        s = m.simulate(1, probe=['i'])
        self.assertEqual(s['i'], [0.0])

    def test_summary(self):
        s = _chain().summary()
        self.assertEqual(s['blocks'], 4)
        self.assertEqual(s['inports'], 1)
        self.assertEqual(s['stateful_blocks'], 0)


class TestStateMachine(unittest.TestCase):

    def _sm(self):
        return StateMachine(
            'mode', n_in=1,
            states={'off': 0.0, 'on': 1.0},
            initial='off',
            transitions=[('off', 'on', [(0, '>', 0.5)]),
                         ('on', 'off', [(0, '<', 0.5)])])

    def test_initial_state(self):
        sm = self._sm()
        st = sm.init_state()
        self.assertEqual(sm.output([0.0], st), 0.0)

    def test_transition_on_condition(self):
        sm = self._sm()
        st = sm.init_state()
        sm.update([1.0], 0.0, st, 1.0)
        self.assertEqual(st['state'], 'on')
        self.assertEqual(st['switches'], 1)

    def test_no_transition_when_condition_false(self):
        sm = self._sm()
        st = sm.init_state()
        sm.update([0.1], 0.0, st, 1.0)
        self.assertEqual(st['state'], 'off')

    def test_stays_until_next_condition(self):
        sm = self._sm()
        st = sm.init_state()
        sm.update([1.0], 0.0, st, 1.0)
        sm.update([1.0], 1.0, st, 1.0)     # 已在 on，on->off 条件不成立
        self.assertEqual(st['state'], 'on')
        self.assertEqual(st['switches'], 1)

    def test_first_matching_transition_wins(self):
        """转移按书写顺序求值，第一个成立的就走 —— 和 Stateflow 语义一致。"""
        sm = StateMachine('sm', n_in=1, states={'a': 0.0, 'b': 1.0, 'c': 2.0},
                          initial='a',
                          transitions=[('a', 'b', [(0, '>', 0.0)]),
                                       ('a', 'c', [(0, '>', 0.0)])])
        st = sm.init_state()
        sm.update([1.0], 0.0, st, 1.0)
        self.assertEqual(st['state'], 'b')

    def test_empty_condition_is_unconditional(self):
        sm = StateMachine('sm', n_in=1, states={'a': 0.0, 'b': 1.0}, initial='a',
                          transitions=[('a', 'b', [])])
        st = sm.init_state()
        sm.update([0.0], 0.0, st, 1.0)
        self.assertEqual(st['state'], 'b')

    def test_is_not_feedthrough(self):
        self.assertFalse(StateMachine.direct_feedthrough)

    def test_validation(self):
        with self.assertRaises(ValueError):
            StateMachine('sm', states={}, initial='x')
        with self.assertRaises(ValueError):
            StateMachine('sm', states={'a': 0.0}, initial='zzz')
        with self.assertRaises(ValueError):
            StateMachine('sm', states={'a': 0.0, 'b': 1.0}, initial='a',
                         transitions=[('a', 'q', [])])
        with self.assertRaises(ValueError):
            StateMachine('sm', n_in=1, states={'a': 0.0, 'b': 1.0}, initial='a',
                         transitions=[('a', 'b', [(5, '>', 0.0)])])
        with self.assertRaises(ValueError):
            StateMachine('sm', n_in=1, states={'a': 0.0, 'b': 1.0}, initial='a',
                         transitions=[('a', 'b', [(0, '~=', 0.0)])])


class TestFromDict(unittest.TestCase):

    def test_json_model(self):
        from mbd.model import from_dict
        spec = {
            'name': 'demo', 'dt': 0.01,
            'blocks': [
                {'kind': 'Inport', 'name': 'u', 'index': 0},
                {'kind': 'Gain', 'name': 'k', 'k': 3.0},
                {'kind': 'Integrator', 'name': 'i', 'ic': 0.0},
            ],
            'connections': [['k', 0, 'u', 0], ['i', 0, 'k', 0]],
        }
        m = from_dict(spec)
        m.validate()
        s = m.simulate(3, inputs={0: 1.0}, probe=['i'])
        self.assertEqual(s['i'], [0.0, 0.03, 0.06])

    def test_bad_connection(self):
        from mbd.model import from_dict
        with self.assertRaises(ValueError):
            from_dict({'blocks': [{'kind': 'Gain', 'name': 'g'}],
                       'connections': [['g', 0, 'ghost', 0]]})


if __name__ == '__main__':
    unittest.main(verbosity=2)
