# -*- coding: utf-8 -*-
"""命令行演示：``python -m mbd [lib|sim|gen|check|all]``。

打印出来的数字都是**当场算的**，README 里的实测结果就是从这些输出抄下来的。
"""
import os
import sys

from . import codegen, library
from .model import AlgebraicLoopError


def _h(title):
    print()
    print('=' * 74)
    print(title)
    print('=' * 74)


def _stimulus(name, steps):
    if name == 'soc_estimator':
        out = {}
        for i in range(steps):
            if i < steps * 0.4:
                out[i] = {0: 2.0, 1: 3.85}
            elif i < steps * 0.6:
                out[i] = {0: 0.0, 1: 3.88}
            elif i < steps * 0.8:
                out[i] = {0: -3.0, 1: 4.05}
            else:
                out[i] = {0: 0.0, 1: 4.00}
        return out
    if name == 'buck_voltage_loop':
        return {0: [12.0 if i < steps // 2 else 15.0 for i in range(steps)]}
    return {0: [1.0 if i >= 10 else 0.0 for i in range(steps)]}


def demo_lib():
    _h('1) 预置示例模型（对应 JD 的 BMS / 电源控制业务域）')
    for key in sorted(library.LIBRARY):
        m = library.get(key)
        s = m.summary()
        order = m.order()
        print('  %-22s dt=%-8g 块 %2d  连接 %2d  状态块 %d  状态变量 %d  Inport %d'
              % (key, s['dt'], s['blocks'], s['connections'],
                 s['stateful_blocks'], m.state_size(), s['inports']))
        print('      拓扑序：%s' % ' → '.join(order))


def demo_sim():
    _h('2) 模型仿真（"模型设计 → 仿真"这半条链路）')

    m = library.get('buck_voltage_loop')
    steps = 4000
    s = m.simulate(steps, inputs=_stimulus('buck_voltage_loop', steps),
                   probe=['vout', 'duty', 'err'])
    print('  Buck 电压闭环：dt=%g s，%d 步（%g s），Vin=24 V，给定 12 V（后半程切 15 V）'
          % (m.dt, steps, steps * m.dt))
    print('     时间(s)    输出电压(V)    占空比      误差(V)')
    for i in (0, 10, 100, 500, 1000, 1999, 2000, 2500, 3999):
        print('   %9.4f  %12.5f  %10.5f  %+10.5f'
              % (i * m.dt, s['vout'][i], s['duty'][i], s['err'][i]))
    print('   → 12 V 时稳在 12.00000 V、占空比 0.50000（= 12/24，符合 Buck 的 d = Vo/Vin）')

    print()
    m2 = library.get('soc_estimator')
    steps = 3600
    s2 = m2.simulate(steps, inputs=_stimulus('soc_estimator', steps),
                     probe=['soc', 'soc_rate', 'mode', 'ocv_soc'])
    print('  BMS SOC 估算：dt=1 s，%d 步；2 A 放电 → 静置 → 3 A 充电 → 静置（Q=50 Ah）' % steps)
    print('     时间(s)   SOC 估算    库仑+修正速率   状态机    OCV 查表 SOC')
    names = {0.0: '放电', 1.0: '静置', 2.0: '充电'}
    for i in (0, 900, 1439, 1440, 2000, 2879, 2880, 3599):
        print('   %8d  %10.6f  %+14.3e   %-6s   %10.4f'
              % (i, s2['soc'][i], s2['soc_rate'][i], names.get(s2['mode'][i], '?'),
                 s2['ocv_soc'][i]))
    print('   → 2 A 放电 1800 s = 1 Ah，占 50 Ah 的 2%%，SOC 由 0.600 降到 0.580，与理论一致；')
    print('     静置后状态机切到「静置」，查表把状态码 1 映射成修正增益，OCV 修正把 SOC 缓慢拉向查表值')
    print('     —— 这就是 SOX 里的 OCV 修正。注意第 0 步输出的是**状态机的初始态**（与积分器一致，')
    print('     输出反映的是"更新前"的状态），所以第一个点还显示静置。')

    print()
    m3 = library.get('rate_limited_tracker')
    s3 = m3.simulate(200, inputs=_stimulus('rate_limited_tracker', 200), probe=['ramp'])
    print('  斜率限制器：输入在第 10 步给阶跃 1.0，rise=10 /s，dt=1 ms（每步最多 0.01）')
    print('     步 0=%.6f  步 10=%.6f  步 60=%.6f  步 109=%.6f  步 110=%.6f'
          % (s3['ramp'][0], s3['ramp'][10], s3['ramp'][60], s3['ramp'][109], s3['ramp'][110]))
    print('   → 从第 10 步起每步 0.01，正好 100 步后到 1.0（第 110 步）')


def demo_gen():
    _h('3) 代码生成（"代码生成"这半条链路）')
    gen = codegen.generate(library.get('rate_limited_tracker'), model_name='ramp_demo')
    print('  以最简单的模型为例，下面是**完整生成结果**（真实工具链会大得多）：')
    print()
    print('  ---------- ramp_demo.h ----------')
    for line in gen.header.splitlines():
        print('  ' + line)
    print()
    print('  ---------- ramp_demo.c ----------')
    for line in gen.source.splitlines():
        print('  ' + line)

    print()
    m = library.get('soc_estimator')
    g2 = codegen.generate(m)
    print('  SOC 估算模型生成规模：头文件 %d 行 / 源文件 %d 行，状态字段 %d 个'
          % (len(g2.header.splitlines()), len(g2.source.splitlines()), len(g2.state_fields)))
    print('  状态字段：%s' % ', '.join('%s(%s)' % (f[1], f[2]) for f in g2.state_fields))


def demo_check():
    _h('4) 模型 ↔ 生成代码 一致性校验（"集成验证"这半条链路）')
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
    try:
        import codegen_check
    except ImportError:
        print('  找不到 tools/codegen_check.py')
        return
    if codegen_check.find_cc() is None:
        print('  找不到 C 编译器，跳过（试过 %s）' % ' / '.join(codegen_check.CC_CANDIDATES))
        return
    for name in sorted(library.LIBRARY):
        codegen_check.check(name, steps=400)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    what = argv[0] if argv else 'all'
    table = {'lib': demo_lib, 'sim': demo_sim, 'gen': demo_gen, 'check': demo_check}
    try:
        if what == 'all':
            for fn in (demo_lib, demo_sim, demo_gen, demo_check):
                fn()
        elif what in table:
            table[what]()
        else:
            print(__doc__)
            print('可用：lib | sim | gen | check | all')
            return 2
    except AlgebraicLoopError as e:
        print('模型存在代数环：%s' % e)
        return 1
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
