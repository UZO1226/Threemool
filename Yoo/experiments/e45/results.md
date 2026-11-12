# e45 동일 조건 평가 결과

전체 train188일과 개발 val40일, 모델5개, 고정 정책12개. 확률 행동은3회 평균이며 독립 정책 seed가 아니다.
새 학습과 test 평가는 없다. 수치는 대리 환경 내부 결과다.

## NH4 비용

| 정책 | train 결정적 | train 확률적 | val 결정적 | val 확률적 |
| --- | --- | --- | --- | --- |
| ppo_padded_s0 | 4.207 | 5.497 | 6.360 | 7.558 |
| ppo_padded_s1 | 4.662 | 5.811 | 7.110 | 7.873 |
| ppo_previous_s0 | 4.784 | 5.836 | 8.025 | 9.362 |
| ppo_previous_s1 | 3.848 | 5.278 | 6.010 | 6.943 |
| fcsrl_padded_s0 | 4.409 | 5.388 | 6.305 | 7.068 |
| fcsrl_padded_s1 | 4.999 | 5.156 | 28.150 | 28.732 |
| fcsrl_previous_s0 | 4.139 | 4.872 | 15.580 | 15.272 |
| fcsrl_previous_s1 | 5.569 | 6.302 | 8.130 | 8.600 |
| fcsrl_actor_padded_s0 | 4.460 | 5.043 | 8.140 | 8.912 |
| fcsrl_actor_padded_s1 | 4.267 | 5.066 | 13.505 | 15.268 |
| fcsrl_actor_previous_s0 | 4.371 | 5.103 | 19.805 | 20.035 |
| fcsrl_actor_previous_s1 | 4.735 | 5.005 | 11.495 | 12.448 |

## 세 주 지표와 초과 적분

| 정책 | 날짜 | 행동 | 공기 m³/23.75h | NH4 비용 | DO 비용 | NH4 초과 적분 | DO 초과 적분 | 두 평균6/6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_padded_s0 | val | deterministic | 84737.0 | 6.360 | 1.570 | 3.831 | 0.152 | False |
| ppo_padded_s0 | val | stochastic | 85060.9 | 7.558 | 1.633 | 4.038 | 0.137 | False |
| ppo_padded_s0 | train | deterministic | 61726.7 | 4.207 | 0.915 | 1.549 | 0.056 | True |
| ppo_padded_s0 | train | stochastic | 63765.6 | 5.497 | 0.921 | 1.799 | 0.054 | True |
| ppo_padded_s1 | val | deterministic | 86127.9 | 7.110 | 1.425 | 3.959 | 0.101 | False |
| ppo_padded_s1 | val | stochastic | 85920.5 | 7.873 | 1.403 | 4.214 | 0.098 | False |
| ppo_padded_s1 | train | deterministic | 62512.3 | 4.662 | 0.990 | 2.225 | 0.048 | True |
| ppo_padded_s1 | train | stochastic | 63707.3 | 5.811 | 1.062 | 2.380 | 0.051 | True |
| ppo_previous_s0 | val | deterministic | 82337.0 | 8.025 | 1.845 | 4.153 | 0.181 | False |
| ppo_previous_s0 | val | stochastic | 82267.9 | 9.362 | 1.750 | 4.341 | 0.163 | False |
| ppo_previous_s0 | train | deterministic | 61198.3 | 4.784 | 1.003 | 1.627 | 0.066 | True |
| ppo_previous_s0 | train | stochastic | 62430.5 | 5.836 | 1.033 | 1.884 | 0.066 | True |
| ppo_previous_s1 | val | deterministic | 84014.0 | 6.010 | 1.505 | 3.923 | 0.110 | False |
| ppo_previous_s1 | val | stochastic | 83846.2 | 6.943 | 1.482 | 4.089 | 0.117 | False |
| ppo_previous_s1 | train | deterministic | 60575.9 | 3.848 | 1.023 | 1.884 | 0.048 | True |
| ppo_previous_s1 | train | stochastic | 62308.8 | 5.278 | 1.071 | 2.227 | 0.054 | True |
| fcsrl_padded_s0 | val | deterministic | 84237.4 | 6.305 | 1.020 | 5.057 | 0.063 | False |
| fcsrl_padded_s0 | val | stochastic | 83634.5 | 7.068 | 1.030 | 5.004 | 0.056 | False |
| fcsrl_padded_s0 | train | deterministic | 59607.3 | 4.409 | 0.812 | 2.006 | 0.040 | True |
| fcsrl_padded_s0 | train | stochastic | 60675.1 | 5.388 | 0.937 | 2.218 | 0.047 | True |
| fcsrl_padded_s1 | val | deterministic | 80739.6 | 28.150 | 0.970 | 8.101 | 0.110 | False |
| fcsrl_padded_s1 | val | stochastic | 80837.8 | 28.732 | 1.032 | 8.579 | 0.094 | False |
| fcsrl_padded_s1 | train | deterministic | 61436.4 | 4.999 | 1.080 | 1.684 | 0.056 | True |
| fcsrl_padded_s1 | train | stochastic | 63035.0 | 5.156 | 1.085 | 1.926 | 0.055 | True |
| fcsrl_previous_s0 | val | deterministic | 79177.7 | 15.580 | 0.850 | 7.378 | 0.038 | False |
| fcsrl_previous_s0 | val | stochastic | 80047.7 | 15.272 | 0.863 | 6.928 | 0.028 | False |
| fcsrl_previous_s0 | train | deterministic | 60400.5 | 4.139 | 0.780 | 1.334 | 0.036 | True |
| fcsrl_previous_s0 | train | stochastic | 62217.9 | 4.872 | 0.837 | 1.483 | 0.042 | True |
| fcsrl_previous_s1 | val | deterministic | 83307.9 | 8.130 | 0.745 | 5.378 | 0.047 | False |
| fcsrl_previous_s1 | val | stochastic | 84297.6 | 8.600 | 0.828 | 5.693 | 0.052 | False |
| fcsrl_previous_s1 | train | deterministic | 62747.2 | 5.569 | 1.352 | 1.878 | 0.103 | True |
| fcsrl_previous_s1 | train | stochastic | 64100.8 | 6.302 | 1.375 | 1.977 | 0.097 | False |
| fcsrl_actor_padded_s0 | val | deterministic | 81481.9 | 8.140 | 1.115 | 4.845 | 0.063 | False |
| fcsrl_actor_padded_s0 | val | stochastic | 82361.1 | 8.912 | 1.105 | 5.124 | 0.064 | False |
| fcsrl_actor_padded_s0 | train | deterministic | 58292.6 | 4.460 | 0.809 | 1.578 | 0.035 | True |
| fcsrl_actor_padded_s0 | train | stochastic | 61404.3 | 5.043 | 0.845 | 1.727 | 0.038 | True |
| fcsrl_actor_padded_s1 | val | deterministic | 79502.7 | 13.505 | 0.790 | 10.825 | 0.039 | False |
| fcsrl_actor_padded_s1 | val | stochastic | 80367.2 | 15.268 | 0.767 | 11.382 | 0.031 | False |
| fcsrl_actor_padded_s1 | train | deterministic | 57343.4 | 4.267 | 1.154 | 2.520 | 0.063 | True |
| fcsrl_actor_padded_s1 | train | stochastic | 60387.3 | 5.066 | 1.198 | 2.799 | 0.068 | True |
| fcsrl_actor_previous_s0 | val | deterministic | 77930.3 | 19.805 | 0.825 | 14.882 | 0.044 | False |
| fcsrl_actor_previous_s0 | val | stochastic | 78766.4 | 20.035 | 0.767 | 13.868 | 0.025 | False |
| fcsrl_actor_previous_s0 | train | deterministic | 57478.7 | 4.371 | 0.970 | 1.900 | 0.040 | True |
| fcsrl_actor_previous_s0 | train | stochastic | 60086.7 | 5.103 | 0.994 | 2.015 | 0.042 | True |
| fcsrl_actor_previous_s1 | val | deterministic | 78272.2 | 11.495 | 0.770 | 8.203 | 0.029 | False |
| fcsrl_actor_previous_s1 | val | stochastic | 78437.0 | 12.448 | 0.752 | 9.083 | 0.027 | False |
| fcsrl_actor_previous_s1 | train | deterministic | 57629.9 | 4.735 | 1.422 | 3.383 | 0.072 | True |
| fcsrl_actor_previous_s1 | train | stochastic | 59024.9 | 5.005 | 1.344 | 3.253 | 0.068 | True |

네 조건 모두 NH4 평균6을 초과한 정책은 0/12개다. 이는 표현 head 분리가 해결책이라는 증명은 아니다.
기존 e44 결정적 val 재현 12/12개, 독립 수치 검사 3552개 통과.
report.json에 반복별 범위, 모델별 결과, 반복별 최악 날짜의 평균과 조건 간 차이를 보관한다. 최악 날짜 평균은 모든 반복 중 최댓값과 다르다.
날짜 분할은 무작위 배정이 아니므로 분포 변화의 인과 효과로 단정하지 않는다. 개발 자료는 반복 사용했고 환경 오차는 남는다.
