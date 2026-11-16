"""평균 공기량 증가를 감소로 잘못 판정했던 반례를 두 보고서에 적용한다."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

EXP = Path(__file__).resolve().parents[2]


def module(exp):
    spec=importlib.util.spec_from_file_location(exp,EXP/exp/'report.py')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReportRegression(unittest.TestCase):
    def test_e52_mean_air_increases_despite_mean_percentage_decrease(self):
        mod=module('e52')
        def load(root,method,k,seed):
            v=dict(air_m3=([110,9,9] if k else [100,10,10])[seed],nh4_cost=9 if k else 10,
                   do_cost=1 if k else 2,nh4_excess_mg_h_L=2,worst_day_nh4_cost=30)
            return dict(val=v,train=dict(v,nh4_cost=3),source='synthetic fixture')
        mod.load=load
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            mod.main(Path(tmp),Path(tmp))
            result=json.loads((Path(tmp)/'summary.json').read_text())['verdict']['fcsrl']
        self.assertLess(result['mean_paired_air_change_pct'],0)
        self.assertAlmostEqual(result['mean_air_change_pct'],100*8/120)
        self.assertTrue(result['air_up_over_5pct'])
        self.assertFalse(result['all_three_lower_mean'])

    def test_e50_air_guard_uses_physical_mean(self):
        mod=module('e50')
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            (root/'manifest.json').write_text(json.dumps(dict(seeds=list(range(5)),steps_per_run=1)))
            for m in mod.METHODS:
                for s in range(5):
                    v=dict(air_m3=([120,9,9,9,9] if m=='fcsrl_actor' else [100,10,10,10,10])[s],
                           cost=9 if m=='fcsrl_actor' else 10,nh4_viol_pct=1,do_viol_pct=1,days_within_limit_pct=100)
                    p=root/'runs'/m/f'seed_{s}'
                    p.mkdir(parents=True)
                    (p/'metrics.json').write_text(json.dumps(dict(steps=1,val_final=v)))
            mod.main(root,root)
            h=json.loads((root/'summary.json').read_text())['H1_actor_vs_fcsrl']
        self.assertLess(h['mean_paired_air_diff_pct'],0)
        self.assertGreater(h['mean_air_diff_pct'],2)
        self.assertFalse(h['supported'])


if __name__=='__main__':
    unittest.main()
