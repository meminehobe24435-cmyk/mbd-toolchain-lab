# -*- coding: utf-8 -*-
"""状态机图（对应 MBD 里 Stateflow 图的最小实现）。

为什么需要它
------------
框图适合表达连续量的运算，但**模式切换、启停、故障处理**这类逻辑用框图表达会很别扭。
所以 MBD 里通常把这类逻辑放进状态机图。本项目实现一个够用的子集：

* **状态**：每个状态带一个输出值；
* **转移**：``(源状态, 目标状态, 条件列表)``，条件之间是**与**关系；
* **条件**：``(输入端口号, 比较符, 阈值)``，比较符支持 ``> >= < <= == !=``；
* **默认转移**：按转移列表顺序求值，**第一个成立的就走**（和 Stateflow 的语义一致，
  所以**转移的书写顺序是有意义的**）。

刻意不做的：层次状态、并行状态、after/on 事件、历史节点、内部转移与 entry/during/exit 动作。
这些在真实项目里很常用，但实现它们需要事件队列与状态层次树，超出"最小可用"的范围 ——
与其做一个半吊子，不如把边界写清楚（见 README 边界一节）。

输出只与当前状态有关（``direct_feedthrough = False``），因此**和积分器一样能打断反馈环**。
"""
from .blocks import Block, REGISTRY

_OPS = {
    '>':  lambda a, b: a > b,
    '>=': lambda a, b: a >= b,
    '<':  lambda a, b: a < b,
    '<=': lambda a, b: a <= b,
    '==': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
}


class StateMachine(Block):
    kind = 'StateMachine'
    direct_feedthrough = False
    stateful = True

    def __init__(self, name, states=None, transitions=None, initial=None, n_in=1, **params):
        Block.__init__(self, name, **params)
        if not states:
            raise ValueError('%s：状态机至少要有一个状态' % name)
        self.states = {str(k): float(v) for k, v in states.items()}
        self.n_in = int(n_in)
        self.initial = str(initial) if initial is not None else sorted(self.states)[0]
        if self.initial not in self.states:
            raise ValueError('%s：初始状态 %s 不在状态表里' % (name, self.initial))

        self.transitions = []
        for t in (transitions or []):
            src, dst, conds = t[0], t[1], (t[2] if len(t) > 2 else [])
            if src not in self.states:
                raise ValueError('%s：转移的源状态 %s 不存在' % (name, src))
            if dst not in self.states:
                raise ValueError('%s：转移的目标状态 %s 不存在' % (name, dst))
            checked = []
            for (port, op, threshold) in conds:
                if op not in _OPS:
                    raise ValueError('%s：不支持的比较符 %r；可用 %s'
                                     % (name, op, ', '.join(sorted(_OPS))))
                port = int(port)
                if port < 0 or port >= self.n_in:
                    raise ValueError('%s：条件引用了端口 %d，但该状态机只有 %d 个输入'
                                     % (name, port, self.n_in))
                checked.append((port, op, float(threshold)))
            self.transitions.append((src, dst, checked))

    def init_state(self):
        return {'state': self.initial, 'switches': 0}

    def output(self, u, st):
        return self.states[st['state']]

    def update(self, u, y, st, dt):
        cur = st['state']
        for (src, dst, conds) in self.transitions:
            if src != cur:
                continue
            if all(_OPS[op](u[port], thr) for (port, op, thr) in conds):
                st['state'] = dst
                st['switches'] += 1
                return

    def describe(self):
        return {'initial': self.initial, 'states': self.states,
                'transitions': [(s, d, list(c)) for (s, d, c) in self.transitions]}


REGISTRY['StateMachine'] = StateMachine
