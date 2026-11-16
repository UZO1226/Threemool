"""고정 여유 대조의 공기량 평균 판정 회귀 검사."""
import unittest
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from compare import decision


class MeanAirGuard(unittest.TestCase):
    def test_mean_ratio_guard_blocks_false_improvement(self):
        fixed=[dict(air_m3=a,nh4_cost=10) for a in [100,10,10]]
        adaptive=[dict(air_m3=a,nh4_cost=9) for a in [110,9,9]]
        v=decision(fixed,adaptive)
        self.assertTrue(v['original_report_H1'])
        self.assertFalse(v['H1'])
        self.assertAlmostEqual(v['mean_air_change_pct'],100*8/120)

    def test_e53_report_mean_ratio_guard(self):
        p=Path(__file__).resolve().parents[2]/'e53/report.py'
        spec=importlib.util.spec_from_file_location('e53_report',p)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def load(root,e52,method,condition,seed):
            adaptive=condition=='k10'
            v=dict(air_m3=([110,9,9] if adaptive else [100,10,10])[seed],
                   nh4_cost=9 if adaptive else 10,do_cost=1,nh4_excess_mg_h_L=2,worst_day_nh4_cost=20)
            return v,dict(v,nh4_cost=3)
        module.load=load
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            module.main(Path(tmp),Path(tmp),Path(tmp))
            v=json.loads((Path(tmp)/'summary.json').read_text())['verdict']['fcsrl']
        self.assertFalse(v['H1'])
        self.assertAlmostEqual(v['mean_air_change_pct'],100*8/120)

    def test_true_air_improvement_is_retained(self):
        fixed=[dict(air_m3=a,nh4_cost=10) for a in [100,10,10]]
        adaptive=[dict(air_m3=a*.95,nh4_cost=10) for a in [100,10,10]]
        v=decision(fixed,adaptive)
        self.assertTrue(v['air_route'])
        self.assertTrue(v['H1'])


if __name__=='__main__':unittest.main()
