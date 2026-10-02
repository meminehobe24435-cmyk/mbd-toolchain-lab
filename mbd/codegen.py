# -*- coding: utf-8 -*-
"""C99 代码生成：把框图模型转成可编译的 C 代码。

生成什么
--------
两个文件（默认 ``mbd_model.h`` / ``mbd_model.c``）：

* **状态结构体** —— 每个带状态块一个字段（积分器的 ``y``、状态机的当前状态等）；
* ``<model>_init(MbdState *s)`` —— 状态复位到模型里写的初值；
* ``<model>_step(MbdState *s, const double *u, double *y)`` —— 推进一个采样步；
* ``<model>_block_names[]`` —— 各输出槽位对应的块名，便于比对时定位。

**``y[i]`` 的下标与 ``Model.order()`` 的拓扑序一致**，所以生成代码的每一个中间量
都能和 Python 仿真逐位对账 —— 这既是调试手段，也是本项目验证代码生成正确性的方式。

逐位一致是怎么保证的
--------------------
1. 两边都用双精度（Python ``float`` / C ``double``）；
2. **运算顺序完全一致**：表达式按拓扑序逐个赋值到临时变量，
   不靠编译器重排（这也意味着生成代码**不应该加 ``-ffast-math``**）；
3. 常数用 ``repr()`` 输出（Python 的最短往返表示），C 的 ``strtod`` 能精确还原。

⚠️ **边界**：本项目生成的是"逐块直译"代码，**不做定点化、不做表达式融合、不做查表优化**，
也不生成符合 MISRA / AUTOSAR 的代码。真实量产工具链（如 Embedded Coder）会做这些，
本项目只覆盖"模型 → 可编译 C → 与模型一致"这条主链路。
"""
import math

from .blocks import (
    Abs, Constant, Gain, Inport, Integrator, Lookup1D, MinMax, Product,
    RateLimiter, Saturation, Sum, UnitDelay,
)
from .statemachine import StateMachine


class CodegenError(ValueError):
    pass


def _num(x):
    """把浮点数写成 C 字面量，保证往返精确。"""
    x = float(x)
    if math.isnan(x):
        return 'NAN'
    if math.isinf(x):
        return 'INFINITY' if x > 0 else '(-INFINITY)'
    return repr(x)


def _ident(name):
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch == '_') else '_')
    s = ''.join(out)
    if not s or s[0].isdigit():
        s = 'v_' + s
    return s


class Generated(object):
    """代码生成结果。"""

    def __init__(self, header, source, order, names, state_fields):
        self.header = header
        self.source = source
        self.order = order
        self.names = names
        self.state_fields = state_fields


def generate(model, model_name=None, guard=None):
    """生成 C 代码。返回 :class:`Generated`。"""
    model.validate()
    order = model.order()
    name = _ident(model_name or model.name)
    guard = _ident((guard or name).upper()) + '_H'

    inports = sorted((b for b in model.blocks.values() if isinstance(b, Inport)),
                     key=lambda b: b.index)
    n_in = len(inports)
    idx_seen = [b.index for b in inports]
    if idx_seen != list(range(n_in)):
        raise CodegenError('Inport 的 index 必须是 0..%d 连续编号，当前为 %s'
                           % (n_in - 1, idx_seen))

    incoming = {}
    for (dst, dport, src, _sp) in model.conn:
        incoming.setdefault(dst, {})[dport] = src

    # ---------------- 状态字段 ----------------
    state_fields = []          # [(块名, 字段名, C 类型, 初值表达式)]
    for blk in model.blocks.values():
        if not blk.stateful:
            continue
        fid = _ident(blk.name)
        if isinstance(blk, (Integrator, UnitDelay)):
            state_fields.append((blk.name, '%s_y' % fid, 'double', _num(blk.ic)))
        elif isinstance(blk, RateLimiter):
            state_fields.append((blk.name, '%s_y' % fid, 'double', _num(blk.ic)))
        elif isinstance(blk, StateMachine):
            states = sorted(blk.states)
            state_fields.append((blk.name, '%s_st' % fid, 'int',
                                 str(states.index(blk.initial))))
        else:
            raise CodegenError('%s：暂不支持为 %s 生成状态' % (blk.name, blk.kind))

    # ---------------- 头文件 ----------------
    h = []
    h.append('/* 由 mbd-toolchain-lab 自动生成，请勿手工修改 */')
    h.append('#ifndef %s' % guard)
    h.append('#define %s' % guard)
    h.append('')
    h.append('#include <stddef.h>')
    h.append('')
    h.append('#define %s_N_IN   %d' % (guard[:-2], n_in))
    h.append('#define %s_N_OUT  %d' % (guard[:-2], len(order)))
    h.append('')
    h.append('typedef struct {')
    if state_fields:
        for (_blk, field, ctype, _init) in state_fields:
            h.append('    %-7s %s;' % (ctype, field))
    else:
        h.append('    int _unused;   /* 该模型没有带状态的块 */')
    h.append('} %sState;' % name)
    h.append('')
    h.append('void %s_init(%sState *s);' % (name, name))
    h.append('void %s_step(%sState *s, const double *u, double *y);' % (name, name))
    h.append('')
    h.append('extern const char *const %s_block_names[%d];' % (name, len(order)))
    h.append('')
    h.append('#endif /* %s */' % guard)
    h.append('')

    # ---------------- 源文件 ----------------
    c = []
    c.append('/* 由 mbd-toolchain-lab 自动生成，请勿手工修改 */')
    c.append('#include "%s.h"' % name)
    c.append('#include <math.h>')
    c.append('')

    # 查表断点表 + 查表函数（二分 + 线性插值，端点钳位 —— 与 Python 端逐位一致）
    for blk in model.blocks.values():
        if isinstance(blk, Lookup1D):
            fid = _ident(blk.name)
            c.append('static const double %s_x[%d] = {%s};'
                     % (fid, len(blk.x), ', '.join(_num(v) for v in blk.x)))
            c.append('static const double %s_y[%d] = {%s};'
                     % (fid, len(blk.y), ', '.join(_num(v) for v in blk.y)))
            c.append('')
            c.append('static double %s_lookup(double v)' % fid)
            c.append('{')
            c.append('    int lo = 0, hi = %d, mid;' % (len(blk.x) - 1))
            c.append('    double span, t;')
            c.append('    if (v <= %s_x[0]) { return %s_y[0]; }' % (fid, fid))
            c.append('    if (v >= %s_x[%d]) { return %s_y[%d]; }'
                     % (fid, len(blk.x) - 1, fid, len(blk.y) - 1))
            c.append('    while (hi - lo > 1) {')
            c.append('        mid = (lo + hi) / 2;')
            c.append('        if (%s_x[mid] <= v) { lo = mid; } else { hi = mid; }' % fid)
            c.append('    }')
            c.append('    span = %s_x[hi] - %s_x[lo];' % (fid, fid))
            c.append('    t = (v - %s_x[lo]) / span;' % fid)
            c.append('    return %s_y[lo] + t * (%s_y[hi] - %s_y[lo]);' % (fid, fid, fid))
            c.append('}')
            c.append('')

    # 状态机输出表与转移
    for blk in model.blocks.values():
        if isinstance(blk, StateMachine):
            fid = _ident(blk.name)
            keys = sorted(blk.states)
            c.append('static const double %s_out[%d] = {%s};'
                     % (fid, len(keys), ', '.join(_num(blk.states[k]) for k in keys)))
            c.append('')

    c.append('const char *const %s_block_names[%d] = {%s};'
             % (name, len(order), ', '.join('"%s"' % b for b in order)))
    c.append('')

    # init
    c.append('void %s_init(%sState *s)' % (name, name))
    c.append('{')
    for (_blk, field, ctype, init) in state_fields:
        c.append('    s->%s = (%s)%s;' % (field, ctype, init))
    if not state_fields:
        c.append('    (void)s;')
    c.append('}')
    c.append('')

    # step
    c.append('void %s_step(%sState *s, const double *u, double *y)' % (name, name))
    c.append('{')
    temps = {}
    for blk_name in order:
        temps[blk_name] = 'v_%s' % _ident(blk_name)
    c.append('    double %s;' % ', '.join(temps[b] for b in order))
    c.append('')
    c.append('    /* ---- 第 1 步：按拓扑序算输出 ---- */')
    for blk_name in order:
        blk = model.blocks[blk_name]
        c.append('    %s = %s;' % (temps[blk_name], _expr(blk, incoming.get(blk_name, {}), temps)))
    c.append('')
    c.append('    /* ---- 第 2 步：全部输出算完后统一更新状态 ---- */')
    updates = [_update(blk, incoming.get(blk.name, {}), temps, model.dt)
               for blk in model.blocks.values() if blk.stateful]
    order_of = {n: i for i, n in enumerate(order)}
    updates = [u for u in updates if u is not None]
    updates.sort(key=lambda t: order_of.get(t[0], 0))
    for (_n, line) in updates:
        c.append('    ' + line)
    if not updates:
        c.append('    (void)s;   /* 无状态块 */')
    c.append('')
    c.append('    /* ---- 输出 ---- */')
    for i, blk_name in enumerate(order):
        c.append('    y[%d] = %s;' % (i, temps[blk_name]))
    c.append('}')
    c.append('')
    c.append('/* 编译建议：cc -std=c99 -Wall -Wextra -Werror -c %s.c */' % name)
    c.append('')

    return Generated('\n'.join(h), '\n'.join(c), order, list(order), state_fields)


def _src(incoming, port, temps):
    if port not in incoming:
        raise CodegenError('端口 %d 没有连接来源' % port)
    return temps[incoming[port]]


def _inputs(blk, incoming, temps):
    ports = sorted(incoming)
    return [_src(incoming, p, temps) for p in ports]


def _expr(blk, incoming, temps):
    """生成某个块的输出表达式。"""
    if isinstance(blk, Constant):
        return '(%s)' % _num(blk.params.get('value', 0.0))

    if isinstance(blk, Inport):
        return 'u[%d]' % blk.index

    u = _inputs(blk, incoming, temps)

    if isinstance(blk, Gain):
        return '(%s) * (%s)' % (_num(blk.params.get('k', 1.0)), u[0])

    if isinstance(blk, Sum):
        acc = u[0]
        for sign, value in zip(blk.signs[1:], u[1:]):
            acc = '(%s) %s (%s)' % (acc, '+' if sign == '+' else '-', value)
        return acc

    if isinstance(blk, Product):
        return '(%s) * (%s)' % (u[0], u[1])

    if isinstance(blk, Saturation):
        return '((%s) < (%s) ? (%s) : ((%s) > (%s) ? (%s) : (%s)))' % (
            u[0], _num(blk.lo), _num(blk.lo), u[0], _num(blk.hi), _num(blk.hi), u[0])

    if isinstance(blk, Abs):
        return 'fabs(%s)' % u[0]

    if isinstance(blk, MinMax):
        fn = 'fmax' if blk.op == 'max' else 'fmin'
        return '%s(%s, %s)' % (fn, u[0], u[1])

    if isinstance(blk, Lookup1D):
        fid = _ident(blk.name)
        return '%s_lookup(%s)' % (fid, u[0])

    if isinstance(blk, (Integrator, UnitDelay, RateLimiter)):
        return 's->%s_y' % _ident(blk.name)

    if isinstance(blk, StateMachine):
        # ⚠️ 输出表是**文件作用域的静态数组**（不是状态结构体的成员），所以不带 s-> 前缀；
        #    当前状态 st 才是结构体成员。
        fid = _ident(blk.name)
        return '%s_out[s->%s_st]' % (fid, fid)

    raise CodegenError('%s：暂不支持为 %s 生成代码' % (blk.name, blk.kind))


def _update(blk, incoming, temps, dt):
    """生成状态更新语句，返回 ``(块名, C 语句)``。"""
    if not blk.stateful:
        return None

    if isinstance(blk, Integrator):
        fid = _ident(blk.name)
        src = _src(incoming, 0, temps)
        parts = ['s->%s_y = s->%s_y + (%s) * (%s);' % (fid, fid, _num(dt), src)]
        if blk.lo is not None:
            parts.append('if (s->%s_y < (%s)) { s->%s_y = (%s); }' % (fid, _num(blk.lo), fid, _num(blk.lo)))
        if blk.hi is not None:
            parts.append('if (s->%s_y > (%s)) { s->%s_y = (%s); }' % (fid, _num(blk.hi), fid, _num(blk.hi)))
        return (blk.name, ' '.join(parts))

    if isinstance(blk, UnitDelay):
        fid = _ident(blk.name)
        return (blk.name, 's->%s_y = %s;' % (fid, _src(incoming, 0, temps)))

    if isinstance(blk, RateLimiter):
        fid = _ident(blk.name)
        src = _src(incoming, 0, temps)
        return (blk.name,
                's->%s_y = s->%s_y + fmin((%s) * (%s), fmax(-(%s) * (%s), (%s) - s->%s_y));'
                % (fid, fid, _num(blk.rise), _num(dt), _num(blk.fall), _num(dt), src, fid))

    if isinstance(blk, StateMachine):
        fid = _ident(blk.name)
        keys = sorted(blk.states)
        lines = []
        for (src_state, dst_state, conds) in blk.transitions:
            cond = ' && '.join('(%s) %s (%s)' % (_src(incoming, p, temps), op, _num(thr))
                               for (p, op, thr) in conds) or '1'
            lines.append('if (s->%s_st == %d && %s) { s->%s_st = %d; }'
                         % (fid, keys.index(src_state), cond, fid, keys.index(dst_state)))
        return (blk.name, ' '.join(lines) if lines else '(void)0;')

    raise CodegenError('%s：暂不支持为 %s 生成状态更新' % (blk.name, blk.kind))


def write_files(model, outdir, model_name=None):
    """把生成结果写到目录，返回 ``(头文件路径, 源文件路径)``。"""
    import os
    gen = generate(model, model_name=model_name)
    name = _ident(model_name or model.name)
    os.makedirs(outdir, exist_ok=True)
    hp = os.path.join(outdir, '%s.h' % name)
    cp = os.path.join(outdir, '%s.c' % name)
    with open(hp, 'w', encoding='utf-8', newline='\n') as f:
        f.write(gen.header)
    with open(cp, 'w', encoding='utf-8', newline='\n') as f:
        f.write(gen.source)
    return hp, cp
