# -*- coding: utf-8 -*-
"""模型 ↔ 生成代码一致性校验（本项目的核心验证手段）。

做法
----
1. 在 Python 里跑一遍模型，记下**每个块每一步**的输出；
2. 用 :mod:`mbd.codegen` 生成 C 代码，编一个 driver 跑**完全相同的激励**；
3. 逐块、逐步比对两边结果，报最大绝对误差。

**判据是"逐位一致"（最大误差 0.0），不是"差不多"** —— 因为两边都是双精度、
运算顺序也刻意保持一致，任何非零偏差都说明代码生成器写错了。

用法::

    python tools/codegen_check.py            # 跑全部示例模型
    python tools/codegen_check.py soc_estimator
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbd import codegen, library  # noqa: E402

#: 依次尝试的 C 编译器
CC_CANDIDATES = ('gcc', 'cc', 'clang')


def find_cc():
    for name in CC_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def stimulus(name, steps):
    """给每个示例模型造一段激励，返回 ``{步号: {端口: 值}}`` 或 ``{端口: 序列}``。"""
    if name == 'soc_estimator':
        out = {}
        for i in range(steps):
            if i < steps * 0.4:                 # 前 40%：2A 放电
                out[i] = {0: 2.0, 1: 3.85}
            elif i < steps * 0.6:               # 中间 20%：静置（触发 OCV 修正）
                out[i] = {0: 0.0, 1: 3.88}
            elif i < steps * 0.8:               # 再 20%：3A 充电
                out[i] = {0: -3.0, 1: 4.05}
            else:                               # 收尾：静置
                out[i] = {0: 0.0, 1: 4.00}
        return out
    if name == 'buck_voltage_loop':
        # 给定阶跃后，中途改一次给定，看闭环能不能跟过去
        return {0: [12.0 if i < steps // 2 else 15.0 for i in range(steps)]}
    return {0: [1.0 if i >= 10 else 0.0 for i in range(steps)]}


DRIVER = r'''#include <stdio.h>
#include <stdlib.h>
#include "%(name)s.h"

int main(int argc, char **argv)
{
    %(name)sState s;
    double u[%(upper)s_N_IN];
    double y[%(upper)s_N_OUT];
    char line[8192];
    FILE *f;
    int i;

    if (argc < 2) { fprintf(stderr, "usage: %%s <stimulus.csv>\n", argv[0]); return 2; }
    f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 2; }

    %(name)s_init(&s);
    while (fgets(line, sizeof line, f)) {
        char *p = line;
        for (i = 0; i < %(upper)s_N_IN; i++) {
            u[i] = strtod(p, &p);
            if (*p == ',' ) { p++; }
        }
        %(name)s_step(&s, u, y);
        for (i = 0; i < %(upper)s_N_OUT; i++) {
            printf("%%.17g%%c", y[i], (i + 1 == %(upper)s_N_OUT) ? '\n' : ',');
        }
    }
    fclose(f);
    return 0;
}
'''


def _py_series(model, steps, inputs):
    """跑 Python 侧仿真，返回 ``{块名: [值...]}``。"""
    return model.simulate(steps, inputs=inputs)


def check(model_name, steps=400, keep=False, verbose=True):
    """校验单个模型，返回 ``(ok, report_dict)``。"""
    cc = find_cc()
    if cc is None:
        raise RuntimeError('找不到 C 编译器（试过 %s）' % ' / '.join(CC_CANDIDATES))

    model = library.get(model_name)
    inputs = stimulus(model_name, steps)
    py = _py_series(model, steps, inputs)

    work = tempfile.mkdtemp(prefix='mbdchk_')
    try:
        name = model_name
        upper = name.upper()
        hp, cp = codegen.write_files(model, work, model_name=name)
        driver = os.path.join(work, 'driver.c')
        with io.open(driver, 'w', encoding='utf-8', newline='\n') as f:
            f.write(DRIVER % {'name': name, 'upper': upper})

        # 激励写成 CSV
        csv_path = os.path.join(work, 'stim.csv')
        with io.open(csv_path, 'w', encoding='utf-8', newline='\n') as f:
            for i in range(steps):
                row = inputs.get(i, {}) if isinstance(next(iter(inputs.values()), None), dict) \
                    else {k: (v[i] if isinstance(v, (list, tuple)) else v) for k, v in inputs.items()}
                vals = [repr(float(row.get(k, 0.0))) for k in range(model.summary()['inports'])]
                f.write(','.join(vals) + '\n')

        exe = os.path.join(work, 'runner.exe' if os.name == 'nt' else 'runner')
        cmd = [cc, '-std=c99', '-Wall', '-Wextra', '-Werror', '-O0',
               driver, cp, '-o', exe, '-lm']
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
        if p.returncode != 0:
            raise RuntimeError('编译失败：\n%s\n%s' % (p.stdout, p.stderr))

        p = subprocess.run([exe, csv_path], capture_output=True, text=True, cwd=work)
        if p.returncode != 0:
            raise RuntimeError('运行失败：\n%s\n%s' % (p.stdout, p.stderr))

        rows = [ln for ln in p.stdout.splitlines() if ln.strip()]
        if len(rows) != steps:
            raise RuntimeError('生成代码跑了 %d 步，模型跑了 %d 步' % (len(rows), steps))

        order = model.order()
        worst = 0.0
        worst_at = None
        per_block = {}
        for i, row in enumerate(rows):
            vals = [float(v) for v in row.split(',')]
            if len(vals) != len(order):
                raise RuntimeError('第 %d 步输出个数 %d，期望 %d' % (i, len(vals), len(order)))
            for j, blk in enumerate(order):
                d = abs(vals[j] - py[blk][i])
                if d > per_block.get(blk, -1.0):
                    per_block[blk] = d
                if d > worst:
                    worst, worst_at = d, (i, blk)

        ok = worst == 0.0
        report = {
            'model': model_name,
            'steps': steps,
            'blocks': len(order),
            'compiler': os.path.basename(cc),
            'max_abs_diff': worst,
            'max_at': worst_at,
            'per_block': per_block,
            'bit_exact': ok,
        }
        if verbose:
            _print(report, model)
        return ok, report
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print('  工作目录保留在：%s' % work)


def _print(report, model):
    print('  模型 %s：%d 个块，%d 步，编译器 %s'
          % (report['model'], report['blocks'], report['steps'], report['compiler']))
    if report['bit_exact']:
        print('    模型 ↔ 生成代码 逐位一致（最大绝对误差 0.0）')
        return
    i, blk = report['max_at']
    print('    ✗ 不一致：最大绝对误差 %.3e（第 %d 步，块 %s）'
          % (report['max_abs_diff'], i, blk))
    bad = sorted((v, k) for k, v in report['per_block'].items() if v != 0.0)
    for v, k in bad[:10]:
        print('       %-20s %.3e' % (k, v))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    names = argv or sorted(library.LIBRARY)
    print('=' * 74)
    print('模型 ↔ 生成代码 一致性校验（判据：逐位一致，最大误差 0.0）')
    print('=' * 74)
    if find_cc() is None:
        print('找不到 C 编译器，跳过（试过 %s）' % ' / '.join(CC_CANDIDATES))
        return 0
    all_ok = True
    for name in names:
        ok, _ = check(name)
        all_ok = all_ok and ok
    print()
    print('结论：%s' % ('全部逐位一致' if all_ok else '存在不一致，见上'))
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
