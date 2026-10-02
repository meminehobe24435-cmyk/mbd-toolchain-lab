# -*- coding: utf-8 -*-
"""mbd-toolchain-lab —— 一个最小可用的 MBD（基于模型的设计）工具链。

模块划分::

    blocks      框图块（Constant / Gain / Sum / Integrator / UnitDelay / Saturation / Lookup1D ...）
    statemachine 状态机图（Stateflow 的最小实现）
    model       模型、拓扑排序、代数环检测、定步长仿真
    codegen     模型 -> C99 代码生成（含查表与状态机）
    library     预置示例模型（BMS 的 SOC 估算 / Buck 电压闭环）

对应 MBD 的四个环节：**模型设计 → 仿真 → 代码生成 → 集成验证**。

纯 Python 标准库，零第三方依赖。
"""

__version__ = '1.0.0'

from . import blocks, codegen, library, model, statemachine  # noqa: F401
from .blocks import REGISTRY, make  # noqa: F401
from .codegen import generate, write_files  # noqa: F401
from .model import AlgebraicLoopError, Model  # noqa: F401
from .statemachine import StateMachine  # noqa: F401

__all__ = ['blocks', 'statemachine', 'model', 'codegen', 'library',
           'Model', 'StateMachine', 'AlgebraicLoopError',
           'make', 'REGISTRY', 'generate', 'write_files']
