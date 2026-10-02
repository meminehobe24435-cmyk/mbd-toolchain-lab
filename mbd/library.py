# -*- coding: utf-8 -*-
"""预置示例模型：都取自这份 JD 的业务域（BMS / 电源控制）。

为什么示例模型选这两个
----------------------
JD 的工作职责是「基于模型开发 **EMS/BMS/LC/SMU** 软件功能」，任职资格第 4 条又要求
「课题研究方向为 **BMS 或者 EMS** 软件方向，如 **SOX 相关算法**或者**电网功率调度**」。
所以示例不写玩具模型，直接写这两个方向里最典型的结构：

* :func:`soc_estimator` —— **BMS 的 SOC 估算**：库仑计数 + OCV 查表修正，
  并用**状态机**判断充/放/静置（只有静置时才用 OCV 修正，这是工程上的常规做法）；
* :func:`buck_voltage_loop` —— **数字电源电压闭环**：PI 控制器 + 占空比饱和 + 一阶被控对象，
  是 EMS/电源类控制器最基础的回路。
"""
from .blocks import (
    Abs, Constant, Gain, Inport, Integrator, Lookup1D, Product, RateLimiter,
    Saturation, Sum,
)
from .model import Model
from .statemachine import StateMachine

#: 一段典型的锂离子 OCV–SOC 曲线（用于示例，非实测电芯数据）
OCV_SOC_V = [3.00, 3.30, 3.50, 3.60, 3.70, 3.80, 3.90, 4.00, 4.10, 4.20]
OCV_SOC_S = [0.00, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.93, 0.97, 1.00]


def soc_estimator(name='soc_estimator', dt=1.0, capacity_ah=50.0,
                  rest_gain=0.002, rest_enter=0.05, rest_exit=0.10, soc0=0.60):
    """BMS 的 SOC 估算模型。

    输入：``u[0]`` = 电流（A，**正为放电**）；``u[1]`` = 端电压（V）。
    输出里可用 ``soc`` 取估算的 SOC（0~1），``mode`` 取状态码（0=放电 / 1=静置 / 2=充电）。

    结构：

    1. **库仑计数**：``d(soc)/dt = -I / (3600 * Q)``（Q 为安时容量；电流为正表示放电，SOC 下降）；
    2. **OCV 查表**：由端电压查 OCV–SOC 曲线得到参考 SOC；
    3. **状态机**：按**带符号**电流判充 / 放 / 静置，用 ``rest_enter`` / ``rest_exit`` 两个阈值
       做**迟滞**（否则电流在零附近抖动时状态会来回跳）；
    4. **修正**：**只有静置时**才把 ``K * (SOC_ocv - SOC)`` 加进积分器 ——
       放电时端电压被内阻拉低，直接用 OCV 查表会明显偏低，所以工程上不在带载时修正。

    ⚠️ 两个踩过的坑（见 README「开发中踩到的问题」）：
      ① 状态机一度接的是**绝对值**电流，于是放电也被判成充电；
      ② 三个状态里有**两个输出值相同**，导致从输出上分不出充放电 —— 现在三个状态码互不相同。
    """
    m = Model(name, dt=dt)

    m.add(Inport('i_pack', index=0))
    m.add(Inport('v_pack', index=1))

    # 库仑计数速率：d(soc)/dt = -I / (3600*Q)
    m.add(Gain('coulomb_rate', k=-1.0 / (3600.0 * capacity_ah)), {0: 'i_pack'})
    m.add(Lookup1D('ocv_soc', x=OCV_SOC_V, y=OCV_SOC_S), {0: 'v_pack'})

    m.add(StateMachine(
        'mode',
        n_in=1,
        # 三个状态给**不同的输出值**，这样它同时是"状态码"（0=放电 / 1=静置 / 2=充电）；
        # 再把状态码用查表映射成"静置修正增益"（只有静置才非零）。
        states={'discharge': 0.0, 'rest': 1.0, 'charge': 2.0},
        initial='rest',
        transitions=[
            ('rest', 'discharge', [(0, '>', rest_exit)]),      # 放到阈值以上才算放电
            ('rest', 'charge', [(0, '<', -rest_exit)]),
            ('discharge', 'rest', [(0, '<', rest_enter)]),     # 回落到小电流才算静置（迟滞）
            ('charge', 'rest', [(0, '>', -rest_enter)]),
        ]), {0: 'i_pack'})
    m.add(Lookup1D('rest_gate', x=[0.0, 1.0, 2.0], y=[0.0, rest_gain, 0.0]), {0: 'mode'})

    m.add(Integrator('soc', ic=soc0, lo=0.0, hi=1.0))
    m.add(Sum('ocv_err', signs='+-'), {0: 'ocv_soc', 1: 'soc'})
    m.add(Product('corr'), {0: 'rest_gate', 1: 'ocv_err'})
    m.add(Sum('soc_rate', signs='++'), {0: 'coulomb_rate', 1: 'corr'})
    m.conn.append(('soc', 0, 'soc_rate', 0))

    m._ordered = None
    return m


def buck_voltage_loop(name='buck_voltage_loop', dt=1e-4, vin=24.0, tau=0.01,
                      kp=0.05, ki=20.0, vref=12.0):
    """Buck 变换器电压闭环（PI + 占空比饱和 + 一阶被控对象）。

    输入：``u[0]`` = 电压给定（V）。

    结构：``e = vref - vout`` → PI（比例直通 + 积分）→ 占空比 ``d`` 限幅到 [0, 1]
    → 一阶对象 ``dvout/dt = (vin*d - vout)/tau``。

    这是一个**带反馈的闭环**，能跑起来正好说明拓扑排序与状态更新顺序是对的
    （被控对象的积分器打断了整个环）。
    """
    m = Model(name, dt=dt)

    m.add(Inport('vref', index=0))
    m.add(Sum('err', signs='+-'), {0: 'vref', 1: 'vout'})
    m.add(Gain('kp_term', k=kp), {0: 'err'})
    m.add(Gain('ki_term', k=ki), {0: 'err'})
    m.add(Integrator('integ', ic=0.0, lo=0.0, hi=1.0), {0: 'ki_term'})
    m.add(Sum('duty_raw', signs='++'), {0: 'kp_term', 1: 'integ'})
    m.add(Saturation('duty', lo=0.0, hi=1.0), {0: 'duty_raw'})
    m.add(Gain('drive', k=vin / tau), {0: 'duty'})
    m.add(Gain('drop', k=1.0 / tau), {0: 'vout'})
    m.add(Sum('dv', signs='+-'), {0: 'drive', 1: 'drop'})
    m.add(Integrator('vout', ic=0.0), {0: 'dv'})

    m._ordered = None
    return m


def rate_limited_tracker(name='rate_limited_tracker', dt=1e-3, rise=10.0, fall=10.0):
    """斜率限制跟踪器 —— 用来演示 :class:`~mbd.blocks.RateLimiter` 这个带状态的块。"""
    m = Model(name, dt=dt)
    m.add(Inport('cmd', index=0))
    m.add(RateLimiter('ramp', rise=rise, fall=fall, ic=0.0), {0: 'cmd'})
    return m


#: 名字 -> 构造函数，供 CLI 与测试遍历
LIBRARY = {
    'soc_estimator': soc_estimator,
    'buck_voltage_loop': buck_voltage_loop,
    'rate_limited_tracker': rate_limited_tracker,
}


def get(name):
    if name not in LIBRARY:
        raise KeyError('未知示例模型 %r；可用：%s' % (name, ', '.join(sorted(LIBRARY))))
    return LIBRARY[name]()
