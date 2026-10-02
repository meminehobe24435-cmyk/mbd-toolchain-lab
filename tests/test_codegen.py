# -*- coding: utf-8 -*-
"""代码生成测试。

分两层：

* **结构层**（不依赖编译器）—— 生成的 C 代码里该有的声明、表达式、拓扑顺序；
* **一致性层**（需要 C 编译器）—— 真的编译、跑同一激励、与模型逐位比对。
  没有编译器时这一层会 **skip 而不是失败**，但会在输出里说明跳过了。
"""
import os
import sys
import unittest

from mbd import Model, blocks, codegen, library
from mbd.codegen import CodegenError

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
try:
    import codegen_check
    _HAS_TOOLS = True
except ImportError:                                   # pragma: no cover
    _HAS_TOOLS = False


def _simple():
    m = Model('demo', dt=0.01)
    m.add(blocks.Inport('u', index=0))
    m.add(blocks.Gain('g', k=2.5), {0: 'u'})
    m.add(blocks.Saturation('sat', lo=-1.0, hi=1.0), {0: 'g'})
    m.add(blocks.Integrator('i', ic=0.0), {0: 'sat'})
    return m


class TestStructure(unittest.TestCase):

    def setUp(self):
        self.gen = codegen.generate(_simple(), model_name='demo')

    def test_header_guard(self):
        self.assertIn('#ifndef DEMO_H', self.gen.header)
        self.assertIn('#define DEMO_H', self.gen.header)
        self.assertIn('#endif', self.gen.header)

    def test_prototypes(self):
        self.assertIn('void demo_init(demoState *s);', self.gen.header)
        self.assertIn('void demo_step(demoState *s, const double *u, double *y);', self.gen.header)

    def test_state_struct_has_one_field_per_stateful_block(self):
        self.assertEqual(len(self.gen.state_fields), 1)
        self.assertTrue(self.gen.state_fields[0][1].endswith('_y'))
        # 生成时字段名按 %-7s 对齐，名字与类型之间有多个空格，断言只查字段名
        self.assertIn('i_y;', self.gen.header)
        self.assertIn('} demoState;', self.gen.header)

    def test_block_name_table_matches_order(self):
        self.assertIn('demo_block_names[%d]' % len(self.gen.order), self.gen.source)
        for name in self.gen.order:
            self.assertIn('"%s"' % name, self.gen.source)

    def test_state_update_comes_after_outputs(self):
        src = self.gen.source
        self.assertLess(src.index('第 1 步：按拓扑序算输出'),
                        src.index('第 2 步：全部输出算完后统一更新状态'))

    def test_expression_shapes(self):
        src = self.gen.source
        self.assertIn('2.5) * (v_u)', src)                       # Gain
        self.assertIn('fabs', codegen.generate(
            _abs_model(), model_name='d').source)                # Abs
        self.assertIn('?', src)                                  # Saturation 三元式

    def test_outputs_written_in_order(self):
        for i, name in enumerate(self.gen.order):
            self.assertIn('y[%d] = v_%s;' % (i, name), self.gen.source)

    def test_no_fast_math_note(self):
        self.assertIn('不应该加', codegen.__doc__ or '')


def _abs_model():
    m = Model('d', dt=0.01)
    m.add(blocks.Inport('u', index=0))
    m.add(blocks.Abs('a'), {0: 'u'})
    return m


class TestCodegenErrors(unittest.TestCase):

    def test_inport_index_must_be_contiguous(self):
        m = Model('m', dt=0.01)
        m.add(blocks.Inport('a', index=0))
        m.add(blocks.Inport('b', index=2))
        with self.assertRaises(CodegenError):
            codegen.generate(m)

    def test_algebraic_loop_surfaces(self):
        from mbd.model import AlgebraicLoopError
        m = Model('m', dt=0.01)
        m.add(blocks.Gain('a', k=1.0), {0: 'b'})
        m.add(blocks.Gain('b', k=1.0), {0: 'a'})
        with self.assertRaises(AlgebraicLoopError):
            codegen.generate(m)


class TestWriteFiles(unittest.TestCase):

    def test_files_written_with_lf(self):
        import tempfile
        d = tempfile.mkdtemp()
        hp, cp = codegen.write_files(_simple(), d, model_name='demo')
        self.assertTrue(os.path.exists(hp))
        self.assertTrue(os.path.exists(cp))
        with open(cp, 'rb') as f:
            data = f.read()
        self.assertNotIn(b'\r\n', data, '生成文件应当是 LF 换行')

    def test_ident_sanitising(self):
        m = Model('weird name-1', dt=0.01)
        m.add(blocks.Constant('a b', value=1.0))
        gen = codegen.generate(m)
        self.assertIn('weird_name_1', gen.header)


@unittest.skipUnless(_HAS_TOOLS, '找不到 tools/codegen_check.py')
class TestBitExact(unittest.TestCase):
    """真的编译并逐位比对 —— 本项目最核心的一组断言。"""

    def setUp(self):
        if codegen_check.find_cc() is None:
            self.skipTest('本机没有 C 编译器（试过 %s）'
                          % ' / '.join(codegen_check.CC_CANDIDATES))

    def test_all_library_models_bit_exact(self):
        for name in sorted(library.LIBRARY):
            ok, rep = codegen_check.check(name, steps=200, verbose=False)
            self.assertTrue(ok, '%s 模型与生成代码不一致，最大误差 %.3e（块 %s）'
                            % (name, rep['max_abs_diff'], rep['max_at']))
            self.assertEqual(rep['max_abs_diff'], 0.0)

    def test_custom_model_with_every_block_kind(self):
        m = Model('allblocks', dt=0.005)
        m.add(blocks.Inport('u', index=0))
        m.add(blocks.Constant('c', value=1.25))
        m.add(blocks.Gain('g', k=-0.75), {0: 'u'})
        m.add(blocks.Sum('s', signs='+-'), {0: 'g', 1: 'c'})
        m.add(blocks.Product('p'), {0: 's', 1: 'c'})
        m.add(blocks.Saturation('sat', lo=-0.5, hi=0.5), {0: 'p'})
        m.add(blocks.Abs('ab'), {0: 'sat'})
        m.add(blocks.MinMax('mx', op='max'), {0: 'ab', 1: 'c'})
        m.add(blocks.Lookup1D('lut', x=[0.0, 1.0, 2.0], y=[0.0, 5.0, 5.0]), {0: 'mx'})
        m.add(blocks.UnitDelay('d', ic=0.1), {0: 'lut'})
        m.add(blocks.RateLimiter('rl', rise=2.0, fall=2.0), {0: 'd'})
        ok, rep = _check_custom(m)
        self.assertTrue(ok, '全块类型模型不一致：%.3e @ %s' % (rep['max_abs_diff'], rep['max_at']))


def _check_custom(model, steps=200):
    """对任意模型跑一次一致性校验（复用 tools 里的实现）。"""
    import tempfile
    cc = codegen_check.find_cc()
    work = tempfile.mkdtemp(prefix='mbdcustom_')
    try:
        name = 'allblocks'
        upper = name.upper()
        hp, cp = codegen.write_files(model, work, model_name=name)
        driver = os.path.join(work, 'driver.c')
        with open(driver, 'w', encoding='utf-8', newline='\n') as f:
            f.write(codegen_check.DRIVER % {'name': name, 'upper': upper})
        csv_path = os.path.join(work, 'stim.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='\n') as f:
            for i in range(steps):
                f.write('%r\n' % (0.3 * ((i % 20) - 10)))
        exe = os.path.join(work, 'runner.exe' if os.name == 'nt' else 'runner')
        import subprocess
        # -ffp-contract=off：见 tools/codegen_check.py 里的说明（clang/ARM64 会做 FMA 收缩）
        p = subprocess.run([cc, '-std=c99', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror', '-O0',
                            driver, cp, '-o', exe, '-lm'],
                           capture_output=True, text=True, cwd=work)
        if p.returncode != 0:
            raise AssertionError('编译失败：\n%s' % p.stderr)
        p = subprocess.run([exe, csv_path], capture_output=True, text=True, cwd=work)
        rows = [r for r in p.stdout.splitlines() if r.strip()]
        py = model.simulate(steps, inputs={0: [0.3 * ((i % 20) - 10) for i in range(steps)]})
        order = model.order()
        worst, at = 0.0, None
        for i, row in enumerate(rows):
            vals = [float(v) for v in row.split(',')]
            for j, blk in enumerate(order):
                d = abs(vals[j] - py[blk][i])
                if d > worst:
                    worst, at = d, (i, blk)
        return worst == 0.0, {'max_abs_diff': worst, 'max_at': at}
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
