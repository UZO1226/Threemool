"""탐지/FPR 상충을 동시 개선으로 잘못 통과시키지 않는지 확인한다."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('uncertainty_gate',Path(__file__).with_name('evaluate.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class GateTests(unittest.TestCase):
    def test_tradeoff_fails(self):
        r=dict(fpr_pct=10,recall_pct=60)
        a=dict(fpr_pct=11,recall_pct=70)
        self.assertFalse(m.screen(a,r,[5.62,5.62],5.62)['exploratory_pass'])
        a=dict(fpr_pct=9,recall_pct=59)
        self.assertFalse(m.screen(a,r,[5.62,5.62],5.62)['exploratory_pass'])

    def test_improvement_needs_matching_and_nonidentity(self):
        r=dict(fpr_pct=10,recall_pct=60);a=dict(fpr_pct=9,recall_pct=61)
        self.assertTrue(m.screen(a,r,[5.62,5.62],5.62)['exploratory_pass'])
        self.assertFalse(m.screen(a,r,[7.,5.62],5.62)['exploratory_pass'])
        self.assertFalse(m.screen(r,r,[5.62,5.62],5.62)['exploratory_pass'])

    def test_paired_identical_bootstrap_zero(self):
        c=np.array([[3,2,5,10],[4,1,6,9],[2,4,7,8],[1,2,3,4]],float)
        self.assertEqual(m.block_intervals(c,c),dict(recall_diff_pp=[0.,0.],fpr_diff_pp=[0.,0.]))

    def test_ensemble_members_do_not_become_days(self):
        truth=np.zeros((2,5,4),bool);truth[:,:,0]=True
        flags=truth.copy();flags[0,:,1]=True
        c=m.counts(flags,truth)
        self.assertEqual(c.shape,(2,4))
        r=m.rates(c)
        self.assertEqual(r['recall_pct'],100)
        self.assertAlmostEqual(r['fpr_pct'],100*5/30)


if __name__=='__main__':unittest.main()
