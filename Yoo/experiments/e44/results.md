# e44 결과

개발 val40일, 환경 구성원5개. 수치는 모델 내부 결과이며 실제 공정 성과가 아니다.

| 방법 | 입력 | seed | 공기 m³/23.75h | NH4 비용 | DO 비용 | NH4 초과 적분 | 두 평균6/6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ppo | padded | 0 | 84737.04 | 6.360 | 1.570 | 3.831 | False |
| ppo | padded | 1 | 86127.85 | 7.110 | 1.425 | 3.959 | False |
| ppo | previous | 0 | 82336.96 | 8.025 | 1.845 | 4.153 | False |
| ppo | previous | 1 | 84013.96 | 6.010 | 1.505 | 3.923 | False |
| fcsrl | padded | 0 | 84237.44 | 6.305 | 1.020 | 5.057 | False |
| fcsrl | padded | 1 | 80739.63 | 28.150 | 0.970 | 8.101 | False |
| fcsrl | previous | 0 | 79177.70 | 15.580 | 0.850 | 7.378 | False |
| fcsrl | previous | 1 | 83307.89 | 8.130 | 0.745 | 5.378 | False |
| fcsrl_actor | padded | 0 | 81481.89 | 8.140 | 1.115 | 4.845 | False |
| fcsrl_actor | padded | 1 | 79502.65 | 13.505 | 0.790 | 10.825 | False |
| fcsrl_actor | previous | 0 | 77930.35 | 19.805 | 0.825 | 14.882 | False |
| fcsrl_actor | previous | 1 | 78272.19 | 11.495 | 0.770 | 8.203 | False |

## 사전 지정 대조

차이는 후보에서 기준을 뺀 값이다. 감소가 좋다. 2seed 탐색이며 통계적 확증이나 채택 결정이 아니다.

| 후보 / 기준 | seed | 공기 차이 | NH4 차이 | DO 차이 | 탐색 기준 |
| --- | --- | --- | --- | --- | --- |
| ppo/previous / ppo/padded | 0 | -2400.08 | 1.665 | 0.275 | False |
| ppo/previous / ppo/padded | 1 | -2113.89 | -1.100 | 0.080 | False |
| fcsrl/previous / fcsrl/padded | 0 | -5059.74 | 9.275 | -0.170 | False |
| fcsrl/previous / fcsrl/padded | 1 | 2568.26 | -20.020 | -0.225 | False |
| fcsrl_actor/previous / fcsrl_actor/padded | 0 | -3551.54 | 11.665 | -0.290 | False |
| fcsrl_actor/previous / fcsrl_actor/padded | 1 | -1230.46 | -2.010 | -0.020 | False |
| fcsrl_actor/padded / fcsrl/padded | 0 | -2755.56 | 1.835 | 0.095 | False |
| fcsrl_actor/padded / fcsrl/padded | 1 | -1236.98 | -14.645 | -0.180 | False |
| fcsrl_actor/previous / fcsrl/previous | 0 | -1247.35 | 4.225 | -0.025 | False |
| fcsrl_actor/previous / fcsrl/previous | 1 | -5035.70 | 3.365 | 0.025 | False |

탐색 기준은 세 주 지표 모두 감소, 초과 적분/최악 날짜 두 비용 악화 없음, 후보 평균6/6 만족이다. 두 seed 모두 통과해도 추가 검증 후보일 뿐이다. 모델 오차와 반복 사용한 개발 날짜의 한계는 남는다.
