# -*- coding: utf-8 -*-
"""模型、拓扑排序与定步长仿真。

执行顺序（与 Simulink 的语义一致，**这一点决定了仿真能否和生成代码逐位对上**）
----------------------------------------------------------------------------
每一步按拓扑序做两件事：

1. **算输出**：按拓扑序遍历，用「本步输入 + 上一步的状态」算出每个块的输出；
2. **更新状态**：全部输出算完之后，再统一更新带状态块的状态。

之所以要分两步、且状态更新放在最后，是为了避免"同一步里先更新的块影响后算的块"——
那会让结果依赖块的书写顺序，也让代码生成无法复现。

代数环
------
拓扑排序只把 ``direct_feedthrough = True`` 的块算作依赖边；
积分器 / 单位延时 / 变化率限制的输出只取决于状态，因此能**打断反馈环**。
如果一个环里全是直通块，就构成**代数环**，本实现会**显式报错**而不是给一个看似能跑的结果。
"""
from . import statemachine  # noqa: F401  （导入即完成 StateMachine 的注册）
from .blocks import Block, Inport, make, REGISTRY


class AlgebraicLoopError(ValueError):
    """环内全是直通块，无法确定求值顺序。"""


class Model(object):
    """一个框图模型。"""

    def __init__(self, name='model', dt=1e-3):
        if dt <= 0:
            raise ValueError('采样步长 dt 必须为正')
        self.name = name
        self.dt = float(dt)
        self.blocks = {}
        self.conn = []          # [(dst, dst_port, src, src_port)]
        self._ordered = None
        self._states = {}

    # ------------------------------------------------------------ 建模
    def add(self, block, inputs=None):
        """加入一个块。``inputs`` 形如 ``{端口号: 源块名}`` 或 ``{端口号: (源块名, 源端口)}``。

        **允许前向引用**（可以先写 ``{'vout'}`` 再加 vout 块）—— 反馈回路里这是必需的，
        真正的存在性检查放在 :meth:`validate`。
        """
        if not isinstance(block, Block):
            raise TypeError('add() 需要 Block 实例，收到 %r' % type(block))
        if block.name in self.blocks:
            raise ValueError('块名重复：%s' % block.name)
        self.blocks[block.name] = block
        for port, src in (inputs or {}).items():
            if isinstance(src, tuple):
                src_name, src_port = src
            else:
                src_name, src_port = src, 0
            self.conn.append((block.name, int(port), str(src_name), int(src_port)))
        self._ordered = None
        return block

    # ------------------------------------------------------------ 校验与排序
    def validate(self):
        """检查块名引用、端口数量与来源端口是否合法。"""
        for (dst, _dp, src, _sp) in self.conn:
            if src not in self.blocks:
                raise ValueError('%s 的连接引用了不存在的块 %s' % (dst, src))
        for blk in self.blocks.values():
            n_in = sum(1 for (d, _p, _s, _sp) in self.conn if d == blk.name)
            blk.check(n_in, 1)
        for (dst, _dp, src, sport) in self.conn:
            if sport != 0:
                raise ValueError('%s 只提供 1 个输出端口，被引用了端口 %d'
                                 % (src, sport))
            if self.blocks[src].n_out <= sport:
                raise ValueError('%s 没有端口 %d' % (src, sport))
        return self

    def order(self):
        """拓扑排序，返回块名列表；存在代数环则抛 :class:`AlgebraicLoopError`。"""
        if self._ordered is not None:
            return self._ordered

        self.validate()
        deps = {name: set() for name in self.blocks}
        for (dst, _dp, src, _sp) in self.conn:
            if self.blocks[dst].direct_feedthrough:
                deps[dst].add(src)

        ready = sorted(n for n, d in deps.items() if not d)
        out = []
        while ready:
            name = ready.pop(0)
            out.append(name)
            for other in sorted(deps):
                if name in deps[other]:
                    deps[other].discard(name)
                    if not deps[other] and other not in out and other not in ready:
                        ready.append(other)
                        ready.sort()

        if len(out) != len(self.blocks):
            stuck = sorted(set(self.blocks) - set(out))
            raise AlgebraicLoopError(
                '存在代数环（环内全是直通块，无状态块打断）：%s；'
                '请在环里插入 UnitDelay 或 Integrator' % ', '.join(stuck))

        self._ordered = out
        return out

    # ------------------------------------------------------------ 仿真
    def reset(self, inputs=None):
        """把状态复位到初值。``inputs`` 可给各 Inport 的初值 ``{index: value}``。"""
        self._states = {n: b.init_state() for n, b in self.blocks.items()}
        for blk in self.blocks.values():
            if isinstance(blk, Inport):
                blk.set_input(self._states[blk.name], (inputs or {}).get(blk.index, 0.0))
        return self

    def step(self, inputs=None):
        """推进一个采样步，返回 ``{块名: 输出值}``。"""
        if not self._states:
            self.reset(inputs)

        for blk in self.blocks.values():
            if isinstance(blk, Inport) and inputs is not None:
                blk.set_input(self._states[blk.name], inputs.get(blk.index, 0.0))

        incoming = {}
        for (dst, dport, src, _sp) in self.conn:
            incoming.setdefault(dst, {})[dport] = src

        def gather(name):
            srcs = incoming.get(name, {})
            return [y_of[srcs[p]] for p in sorted(srcs)]

        # 1) 算输出：直通块用本步输入，非直通块（积分器 / 单位延时 / 状态机）只用状态。
        #    这里**不能**给非直通块取输入 —— 它们的输入来源此刻可能还没算出来（那正是打断代数环的原因）。
        y_of = {}
        for name in self.order():
            blk = self.blocks[name]
            y_of[name] = (blk.output(gather(name), self._states[name])
                          if blk.direct_feedthrough
                          else blk.output(None, self._states[name]))

        # 2) 更新状态：此时所有输出都已就绪，可以安全地取输入。
        for name in self.order():
            blk = self.blocks[name]
            if blk.stateful:
                blk.update(gather(name), y_of[name], self._states[name], self.dt)

        return y_of

    def simulate(self, steps, inputs=None, probe=None):
        """跑 ``steps`` 步，返回 ``probe`` 指定块的输出序列（默认取所有块）。

        ``inputs`` 可为：

        * ``None``：全部 Inport 保持 0；
        * ``{step_index: {inport_index: value}}``：按步给激励；
        * ``{inport_index: 序列}``：按步取序列。
        """
        self.reset()
        probe = list(probe) if probe else sorted(self.blocks)
        series = {name: [] for name in probe}
        for i in range(steps):
            series['__t'] = series.get('__t', [])
            series['__t'].append(i * self.dt)
            y = self.step(_pick_inputs(inputs, i))
            for name in probe:
                series[name].append(y[name])
        return series

    # ------------------------------------------------------------ 报告
    def summary(self):
        kinds = {}
        for blk in self.blocks.values():
            kinds[blk.kind] = kinds.get(blk.kind, 0) + 1
        n_state = sum(1 for b in self.blocks.values() if b.stateful)
        return {
            'name': self.name,
            'dt': self.dt,
            'blocks': len(self.blocks),
            'connections': len(self.conn),
            'stateful_blocks': n_state,
            'inports': sum(1 for b in self.blocks.values() if isinstance(b, Inport)),
            'kinds': kinds,
        }

    def state_size(self):
        """状态变量个数（供代码生成报告）。"""
        return sum(len(b.init_state()) for b in self.blocks.values())


def _pick_inputs(inputs, i):
    """把 ``inputs`` 规范成本步的 ``{inport_index: value}``。

    支持三种写法（按值的类型自动区分，避免歧义）：

    * ``{0: 12.0}`` —— 端口 0 恒为常数；
    * ``{0: [.., ..]}`` —— 端口 0 取序列的第 i 个；
    * ``{0: {0: 2.0, 1: 3.85}, 1: {...}}`` —— 按**步号**给整组输入。
    """
    if not inputs:
        return None
    if all(isinstance(v, dict) for v in inputs.values()):
        return inputs.get(i, {})
    out = {}
    for k, v in inputs.items():
        if isinstance(v, (list, tuple)):
            out[k] = v[i] if i < len(v) else 0.0
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- JSON 载入

def from_dict(spec):
    """由字典构造模型，便于把模型存成 JSON（对应 MBD 里"模型文件"的角色）。

    ::

        {
          "name": "demo", "dt": 0.001,
          "blocks": [
            {"kind": "Inport", "name": "r", "index": 0},
            {"kind": "Sum",    "name": "e", "signs": "+-"},
            {"kind": "Gain",   "name": "Kp", "k": 2.0}
          ],
          "connections": [["e", 0, "r", 0], ["e", 1, "y", 0], ["Kp", 0, "e", 0]]
        }
    """
    model = Model(spec.get('name', 'model'), spec.get('dt', 1e-3))
    for item in spec.get('blocks', []):
        item = dict(item)
        kind = item.pop('kind')
        name = item.pop('name')
        name = name if isinstance(name, str) else str(name)
        model.add(make(kind, str(name), **item))
    for conn in spec.get('connections', []):
        dst, dport, src, sport = conn
        if dst not in model.blocks:
            raise ValueError('连接引用了不存在的目标块 %s' % dst)
        if src not in model.blocks:
            raise ValueError('连接引用了不存在的源块 %s' % src)
        model.conn.append((str(dst), int(dport), str(src), int(sport)))
    model._ordered = None
    return model
