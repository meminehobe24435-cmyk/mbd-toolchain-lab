# -*- coding: utf-8 -*-
"""框图块定义：MBD 里"模型"的最小单元。

设计口径（全项目唯一）
----------------------
* 每个块有固定数量的输入 / 输出端口（本项目不做可变端口）。
* 块的输出由 :meth:`Block.output` 计算；带状态的块（积分器、单位延时）另有
  :meth:`Block.update`，在**每一步末尾**更新状态 —— 顺序与 Simulink 的
  "输出 → 状态更新"一致，这一点决定了仿真结果能不能和代码生成对上。
* ``direct_feedthrough``：输出是否**直接依赖本步输入**。
  积分器与单位延时的输出只取决于状态，所以是 ``False`` ——
  这正是它们能**打断代数环**的原因，也是拓扑排序的依据。

所有数值一律用 Python ``float``（双精度），与生成的 C 代码里的 ``double`` 对应，
这样"模型仿真"与"生成代码"才能逐位比对。
"""
import math


class Block(object):
    """所有块的基类。"""

    kind = 'block'
    n_in = 1
    n_out = 1
    direct_feedthrough = True      # 输出是否直接依赖本步输入
    stateful = False               # 是否有跨步状态

    def __init__(self, name, **params):
        self.name = name
        self.params = params

    # ---- 需要子类实现 ----
    def output(self, u, st):
        raise NotImplementedError

    def init_state(self):
        return {}

    def update(self, u, y, st, dt):
        """默认无状态：什么都不做。"""
        return None

    # ---- 通用 ----
    def check(self, n_inputs, n_outputs):
        if n_inputs != self.n_in:
            raise ValueError('%s（%s）需要 %d 个输入，实际接了 %d 个'
                             % (self.name, self.kind, self.n_in, n_inputs))
        if n_outputs != self.n_out:
            raise ValueError('%s（%s）有 %d 个输出端口，被引用了 %d 个'
                             % (self.name, self.kind, self.n_out, n_outputs))

    def __repr__(self):
        return '<%s %s %s>' % (self.kind, self.name, self.params or '')


# ---------------------------------------------------------------- 无状态块

class Constant(Block):
    kind = 'Constant'
    n_in = 0

    def output(self, u, st):
        return float(self.params.get('value', 0.0))


class Inport(Block):
    """外部输入端口。仿真时由仿真器逐步写入，代码生成时对应 step 函数的入参。"""

    kind = 'Inport'
    n_in = 0

    def __init__(self, name, index=0, **params):
        Block.__init__(self, name, **params)
        self.index = int(index)

    def init_state(self):
        return {'u': 0.0}

    def output(self, u, st):
        return st['u']

    def set_input(self, st, value):
        st['u'] = float(value)


class Gain(Block):
    kind = 'Gain'
    n_in = 1

    def output(self, u, st):
        return self.params.get('k', 1.0) * u[0]


class Sum(Block):
    """按 ``signs`` 逐项加权求和，如 ``signs='+-'`` 表示 ``u0 - u1``。"""
    kind = 'Sum'
    direct_feedthrough = True

    def __init__(self, name, signs='++', **params):
        Block.__init__(self, name, **params)
        self.signs = signs
        self.n_in = len(signs)

    def output(self, u, st):
        acc = 0.0
        for sign, value in zip(self.signs, u):
            acc += value if sign == '+' else -value
        return acc


class Product(Block):
    kind = 'Product'
    n_in = 2

    def output(self, u, st):
        return u[0] * u[1]


class Saturation(Block):
    kind = 'Saturation'
    n_in = 1

    def __init__(self, name, lo=-1.0, hi=1.0, **params):
        Block.__init__(self, name, **params)
        self.lo = float(lo)
        self.hi = float(hi)
        if self.lo > self.hi:
            raise ValueError('%s：下限 %.6g 大于上限 %.6g' % (name, self.lo, self.hi))

    def output(self, u, st):
        return min(self.hi, max(self.lo, u[0]))


class Abs(Block):
    kind = 'Abs'
    n_in = 1

    def output(self, u, st):
        return abs(u[0])


class MinMax(Block):
    kind = 'MinMax'
    n_in = 2

    def __init__(self, name, op='max', **params):
        Block.__init__(self, name, **params)
        if op not in ('min', 'max'):
            raise ValueError('%s：op 只能是 min 或 max' % name)
        self.op = op

    def output(self, u, st):
        return max(u[0], u[1]) if self.op == 'max' else min(u[0], u[1])


class Lookup1D(Block):
    """一维查表 + 线性插值；超出端点按端点**钳位**（不外推）。"""

    kind = 'Lookup1D'
    n_in = 1

    def __init__(self, name, x=None, y=None, **params):
        Block.__init__(self, name, **params)
        self.x = [float(v) for v in (x or [])]
        self.y = [float(v) for v in (y or [])]
        if len(self.x) != len(self.y) or len(self.x) < 2:
            raise ValueError('%s：查表至少 2 点，且 x / y 长度必须一致' % name)
        for i in range(1, len(self.x)):
            if self.x[i] <= self.x[i - 1]:
                raise ValueError('%s：查表断点必须严格递增（第 %d 点不满足）' % (name, i + 1))

    def output(self, u, st):
        v = u[0]
        if v <= self.x[0]:
            return self.y[0]
        if v >= self.x[-1]:
            return self.y[-1]
        lo, hi = 0, len(self.x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.x[mid] <= v:
                lo = mid
            else:
                hi = mid
        span = self.x[hi] - self.x[lo]
        t = (v - self.x[lo]) / span
        return self.y[lo] + t * (self.y[hi] - self.y[lo])


# ---------------------------------------------------------------- 带状态块

class Integrator(Block):
    """前向欧拉积分器：``y = state``，每步末 ``state += dt * u``。

    输出只与状态有关（``direct_feedthrough = False``），因此**可以打断反馈环**。
    """

    kind = 'Integrator'
    n_in = 1
    direct_feedthrough = False
    stateful = True

    def __init__(self, name, ic=0.0, lo=None, hi=None, **params):
        Block.__init__(self, name, **params)
        self.ic = float(ic)
        self.lo = None if lo is None else float(lo)
        self.hi = None if hi is None else float(hi)

    def init_state(self):
        return {'y': self.ic}

    def output(self, u, st):
        return st['y']

    def update(self, u, y, st, dt):
        nxt = st['y'] + dt * u[0]
        if self.lo is not None:
            nxt = max(self.lo, nxt)
        if self.hi is not None:
            nxt = min(self.hi, nxt)
        st['y'] = nxt


class UnitDelay(Block):
    """单位延时：``y = state``，每步末 ``state = u``。

    同样 ``direct_feedthrough = False``，是 MBD 里做反馈环最常用的手段
    （物理上对应"这一拍采到的值下一拍才用"）。
    """

    kind = 'UnitDelay'
    n_in = 1
    direct_feedthrough = False
    stateful = True

    def __init__(self, name, ic=0.0, **params):
        Block.__init__(self, name, **params)
        self.ic = float(ic)

    def init_state(self):
        return {'y': self.ic}

    def output(self, u, st):
        return st['y']

    def update(self, u, y, st, dt):
        st['y'] = u[0]


class RateLimiter(Block):
    """变化率限制：每步增量被限制在 ``[-fall*dt, rise*dt]`` 内。"""

    kind = 'RateLimiter'
    n_in = 1
    direct_feedthrough = False
    stateful = True

    def __init__(self, name, rise=math.inf, fall=math.inf, ic=0.0, **params):
        Block.__init__(self, name, **params)
        self.rise = float(rise)
        self.fall = float(fall)
        self.ic = float(ic)

    def init_state(self):
        return {'y': self.ic}

    def output(self, u, st):
        return st['y']

    def update(self, u, y, st, dt):
        delta = u[0] - st['y']
        lim_up = self.rise * dt
        lim_dn = self.fall * dt
        st['y'] += min(lim_up, max(-lim_dn, delta))


#: 供 :func:`make` 与代码生成器使用的注册表
REGISTRY = {
    'Constant': Constant,
    'Inport': Inport,
    'Gain': Gain,
    'Sum': Sum,
    'Product': Product,
    'Saturation': Saturation,
    'Abs': Abs,
    'MinMax': MinMax,
    'Lookup1D': Lookup1D,
    'Integrator': Integrator,
    'UnitDelay': UnitDelay,
    'RateLimiter': RateLimiter,
}


def make(kind, name, **params):
    """按类型名构造块，供 JSON 模型文件使用。"""
    if kind not in REGISTRY:
        raise KeyError('未知块类型 %r；可用：%s' % (kind, ', '.join(sorted(REGISTRY))))
    return REGISTRY[kind](name, **params)
