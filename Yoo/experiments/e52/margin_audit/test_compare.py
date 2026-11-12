"""고정 여유 대조의 공기량 평균 판정 회귀 검사."""
import unittest
from compare import decision


class MeanAirGuard(unittest.TestCase):
    def test_mean_ratio_guard_blocks_false_improvement(self):
        fixed=[dict(air_m3=a,nh4_cost=10) for a in [100,10,10]]
        adaptive=[dict(air_m3=a,nh4_cost=9) for a in [110,9,9]]
        v=decision(fixed,adaptive)
        self.assertTrue(v['original_report_H1'])
        self.assertFalse(v['H1'])
        self.assertAlmostEqual(v['mean_air_change_pct'],100*8/120)

    def test_true_air_improvement_is_retained(self):
        fixed=[dict(air_m3=a,nh4_cost=10) for a in [100,10,10]]
        adaptive=[dict(air_m3=a*.95,nh4_cost=10) for a in [100,10,10]]
        v=decision(fixed,adaptive)
        self.assertTrue(v['air_route'])
        self.assertTrue(v['H1'])


if __name__=='__main__':unittest.main()
