# e42 정상/위반 예측기 분리 결과

실행 `521b55caf08dfde8924b652a3e068af8c45cc8b5`, 총 232.00초. NH4 학습기 25개, 새 RL/test 없음.

control/augmented는 e41 동결 기준, direct는 다음 절대 농도 단일 회귀 T, capacity는 같은 단일 회귀 3T, split은 정상/위반/혼합기 각각 T다. 주 비교는 split 대 direct/capacity다.

## 검증 40일 95전이

| 군 | NH4 MAE | NH4 recall % | NH4 FPR % | DO1 MAE | DO2 MAE | DO3 MAE | DO recall % | DO FPR % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 1.8839 | 39.088 | 13.698 | 0.2534 | 0.2784 | 0.1419 | 0.000 | 1.544 |
| augmented | 1.7374 | 21.824 | 6.461 | 0.2737 | 0.2849 | 0.1284 | 0.000 | 1.480 |
| direct | 1.9866 | 26.535 | 11.127 | 0.2721 | 0.2798 | 0.1307 | 0.000 | 1.411 |
| capacity | 2.6426 | 34.620 | 22.941 | 0.2741 | 0.2872 | 0.1493 | 0.455 | 1.400 |
| split | 3.2942 | 46.717 | 34.564 | 0.2654 | 0.2834 | 0.1586 | 0.000 | 1.246 |

## holdout 48일

| 군 | NH4 MAE | NH4 recall % | NH4 FPR % | DO1 MAE | DO2 MAE | DO3 MAE | DO recall % | DO FPR % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 1.7137 | 62.896 | 9.852 | 0.1485 | 0.2793 | 0.6800 | 9.070 | 2.454 |
| augmented | 1.6926 | 73.535 | 13.624 | 0.1524 | 0.2879 | 0.6732 | 9.577 | 2.283 |
| direct | 2.0985 | 83.165 | 18.250 | 0.1528 | 0.2863 | 0.6786 | 9.408 | 2.098 |
| capacity | 2.7570 | 81.549 | 28.154 | 0.1610 | 0.3004 | 0.6908 | 10.310 | 2.083 |
| split | 2.5124 | 65.253 | 22.149 | 0.1570 | 0.2930 | 0.6772 | 10.254 | 2.221 |

## 검증 첫 1시간 실측 뒤 91전이

| 군 | NH4 MAE | NH4 recall % | NH4 FPR % | DO1 MAE | DO2 MAE | DO3 MAE | DO recall % | DO FPR % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 1.7858 | 35.717 | 11.375 | 0.2473 | 0.2747 | 0.1351 | 2.273 | 1.463 |
| augmented | 1.7470 | 18.905 | 5.618 | 0.2680 | 0.2858 | 0.1271 | 2.273 | 1.479 |
| direct | 1.9612 | 25.217 | 10.467 | 0.2668 | 0.2787 | 0.1277 | 2.273 | 1.413 |
| capacity | 2.4943 | 31.916 | 19.497 | 0.2682 | 0.2879 | 0.1463 | 2.273 | 1.435 |
| split | 2.8470 | 34.010 | 25.088 | 0.2606 | 0.2837 | 0.1461 | 2.273 | 1.513 |

## 검증 한 스텝 실측 입력

| 군 | NH4 MAE | NH4 recall % | NH4 FPR % | DO1 MAE | DO2 MAE | DO3 MAE | DO recall % | DO FPR % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 0.2354 | 92.979 | 2.260 | 0.1026 | 0.1289 | 0.0607 | 17.273 | 0.165 |
| augmented | 0.2683 | 89.514 | 2.336 | 0.1026 | 0.1289 | 0.0607 | 17.273 | 0.165 |
| direct | 0.2851 | 91.398 | 3.074 | 0.1026 | 0.1289 | 0.0607 | 17.273 | 0.165 |
| capacity | 0.3226 | 88.815 | 2.884 | 0.1026 | 0.1289 | 0.0607 | 17.273 | 0.165 |
| split | 0.3566 | 95.015 | 5.302 | 0.1026 | 0.1289 | 0.0607 | 17.273 | 0.165 |

## 검증 정상/위반 구간과 초과 집계

| 군 | 정상 MAE | 위반 MAE | 정상 편향 | 위반 편향 | NH4 초과 수 평균 | 초과 수 MAE | 초과 강도 MAE mg h/L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 1.7884 | 2.3400 | +1.2642 | -1.0082 | 17.190 | 17.900 | 15.570 |
| augmented | 1.4245 | 3.2317 | +0.2287 | -2.8707 | 8.665 | 13.275 | 5.542 |
| direct | 1.7309 | 3.2076 | +0.5724 | -2.4102 | 13.105 | 16.525 | 12.035 |
| capacity | 2.4326 | 3.6455 | +1.2643 | -2.1625 | 23.715 | 21.715 | 22.058 |
| split | 3.2862 | 3.3325 | +2.3301 | -0.8084 | 34.835 | 32.265 | 39.343 |

실측 NH4 초과 수 평균은 16.45다. 실제 정상/위반 구분은 평가 집계에만 쓰며 추론 전문가 선택에는 쓰지 않는다. MAE와 편향 단위는 mg/L다.

## 혼합기의 진단

| 구간 | AUC | Brier | 실제 정상 평균 g | 실제 위반 평균 g | 실제 위반에서 mu0 | 실제 위반에서 mu1 |
| --- | --- | --- | --- | --- | --- | --- |
| holdout | 0.7620 | 0.2206 | 0.2226 | 0.6532 | 2.6760 | 7.7730 |
| cold | 0.5623 | 0.3647 | 0.3433 | 0.4577 | 2.3341 | 6.9638 |
| warm | 0.5661 | 0.3086 | 0.2491 | 0.3371 | 2.0895 | 6.5483 |
| one_step | 0.9887 | 0.0329 | 0.0588 | 0.8494 | 3.3122 | 6.6813 |

혼합기 확률은 별도 calibration으로 보정하지 않았다. g>0.5 진단은 JSON에만 있으며 주 비용/판정을 대체하지 않는다.

## 공기 변화 뒤 반응

| 군 | 구간 | 분 | 사건 수 | NH4 MAE | 실제 변화 | 예측 변화 | recall % | FPR % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | increase | 15 | 67 | 0.2958 | -0.0910 | +0.0253 | 94.286 | 9.375 |
| control | decrease | 15 | 65 | 0.2024 | -0.1303 | -0.0081 | 100.000 | 0.312 |
| control | decrease_rising | 15 | 10 | 0.2396 | +0.3095 | +0.0949 | 정의 불가 | 0.000 |
| control | increase | 60 | 67 | 1.1853 | -1.4680 | -0.4906 | 94.444 | 25.714 |
| control | decrease | 60 | 65 | 0.7700 | +0.2362 | +0.1514 | 33.333 | 0.323 |
| control | decrease_rising | 60 | 37 | 0.7891 | +1.0541 | +0.4055 | 0.000 | 0.571 |
| control | increase | 180 | 67 | 2.1714 | -3.1415 | -1.3211 | 92.000 | 23.871 |
| control | decrease | 180 | 65 | 1.9010 | +2.1139 | +0.9103 | 11.200 | 9.000 |
| control | decrease_rising | 180 | 55 | 1.9683 | +2.7181 | +1.0899 | 11.200 | 5.333 |
| augmented | increase | 15 | 67 | 0.4071 | -0.0910 | -0.0386 | 89.143 | 10.000 |
| augmented | decrease | 15 | 65 | 0.1799 | -0.1303 | -0.0917 | 100.000 | 0.312 |
| augmented | decrease_rising | 15 | 10 | 0.3014 | +0.3095 | +0.0438 | 정의 불가 | 0.000 |
| augmented | increase | 60 | 67 | 1.3940 | -1.4680 | -0.7720 | 76.667 | 24.490 |
| augmented | decrease | 60 | 65 | 0.7876 | +0.2362 | -0.0356 | 66.667 | 2.258 |
| augmented | decrease_rising | 60 | 37 | 0.9220 | +1.0541 | +0.3065 | 50.000 | 2.857 |
| augmented | increase | 180 | 67 | 2.1741 | -3.1415 | -1.8865 | 40.000 | 24.194 |
| augmented | decrease | 180 | 65 | 2.3822 | +2.1139 | +0.6548 | 6.400 | 11.000 |
| augmented | decrease_rising | 180 | 55 | 2.4003 | +2.7181 | +0.7684 | 6.400 | 5.333 |
| direct | increase | 15 | 67 | 0.5087 | -0.0910 | +0.1140 | 93.143 | 16.250 |
| direct | decrease | 15 | 65 | 0.1631 | -0.1303 | -0.1001 | 100.000 | 0.000 |
| direct | decrease_rising | 15 | 10 | 0.3007 | +0.3095 | +0.0692 | 정의 불가 | 0.000 |
| direct | increase | 60 | 67 | 1.7088 | -1.4680 | -0.3044 | 90.000 | 28.571 |
| direct | decrease | 60 | 65 | 0.7814 | +0.2362 | -0.0348 | 53.333 | 2.258 |
| direct | decrease_rising | 60 | 37 | 0.8860 | +1.0541 | +0.2869 | 30.000 | 2.286 |
| direct | increase | 180 | 67 | 2.8578 | -3.1415 | -1.0753 | 72.000 | 32.581 |
| direct | decrease | 180 | 65 | 2.4679 | +2.1139 | +0.7778 | 12.000 | 9.500 |
| direct | decrease_rising | 180 | 55 | 2.4937 | +2.7181 | +0.9075 | 12.000 | 5.333 |
| capacity | increase | 15 | 67 | 0.5310 | -0.0910 | +0.0423 | 90.857 | 13.125 |
| capacity | decrease | 15 | 65 | 0.2106 | -0.1303 | -0.1378 | 100.000 | 0.000 |
| capacity | decrease_rising | 15 | 10 | 0.3589 | +0.3095 | +0.0209 | 정의 불가 | 0.000 |
| capacity | increase | 60 | 67 | 1.7519 | -1.4680 | -0.6300 | 83.333 | 27.347 |
| capacity | decrease | 60 | 65 | 0.9124 | +0.2362 | -0.1373 | 66.667 | 2.258 |
| capacity | decrease_rising | 60 | 37 | 1.0509 | +1.0541 | +0.1551 | 50.000 | 2.286 |
| capacity | increase | 180 | 67 | 2.8890 | -3.1415 | -1.1258 | 64.000 | 32.258 |
| capacity | decrease | 180 | 65 | 2.7314 | +2.1139 | +0.5978 | 6.400 | 11.500 |
| capacity | decrease_rising | 180 | 55 | 2.7521 | +2.7181 | +0.6620 | 6.400 | 7.333 |
| split | increase | 15 | 67 | 0.6419 | -0.0910 | +0.3329 | 97.143 | 21.250 |
| split | decrease | 15 | 65 | 0.1849 | -0.1303 | -0.1168 | 100.000 | 0.000 |
| split | decrease_rising | 15 | 10 | 0.3436 | +0.3095 | +0.1084 | 정의 불가 | 0.000 |
| split | increase | 60 | 67 | 2.6202 | -1.4680 | +0.7567 | 91.111 | 46.122 |
| split | decrease | 60 | 65 | 0.8824 | +0.2362 | -0.0472 | 73.333 | 2.258 |
| split | decrease_rising | 60 | 37 | 1.0509 | +1.0541 | +0.3131 | 60.000 | 2.857 |
| split | increase | 180 | 67 | 4.5915 | -3.1415 | +0.8632 | 80.000 | 54.516 |
| split | decrease | 180 | 65 | 2.7336 | +2.1139 | +0.7805 | 7.200 | 13.500 |
| split | decrease_rising | 180 | 55 | 2.6690 | +2.7181 | +0.8136 | 7.200 | 8.667 |

decrease_rising은 실측 NH4가 0.1보다 상승한 공기 감소 구간이다. 관측 재생의 부분집합이며 인과 효과가 아니다.

## split에서 비교군을 뺀 변화

| 비교군 | 지표 | 차이 | 날짜 7일 블록 95% 구간 |
| --- | --- | --- | --- |
| control | nh4 MAE | +1.4103 | [+0.8896, +2.0596] |
| control | no3 MAE | -0.0011 | [-0.0134, +0.0115] |
| control | do1 MAE | +0.0120 | [+0.0061, +0.0186] |
| control | do2 MAE | +0.0050 | [-0.0027, +0.0111] |
| control | do3 MAE | +0.0166 | [-0.0015, +0.0365] |
| control | NH4 recall_pct pp | +7.629 | [-6.940, +25.835] |
| control | NH4 fpr_pct pp | +20.866 | [+7.761, +34.534] |
| control | DO recall_pct pp | +0.000 | [+0.000, +0.000] |
| control | DO fpr_pct pp | -0.298 | [-0.878, +0.011] |
| direct | nh4 MAE | +1.3076 | [+0.8934, +1.8632] |
| direct | no3 MAE | -0.0121 | [-0.0239, -0.0007] |
| direct | do1 MAE | -0.0067 | [-0.0148, +0.0017] |
| direct | do2 MAE | +0.0036 | [-0.0020, +0.0112] |
| direct | do3 MAE | +0.0279 | [+0.0176, +0.0376] |
| direct | NH4 recall_pct pp | +20.182 | [+13.479, +28.019] |
| direct | NH4 fpr_pct pp | +23.437 | [+16.517, +31.781] |
| direct | DO recall_pct pp | +0.000 | [+0.000, +0.000] |
| direct | DO fpr_pct pp | -0.165 | [-0.379, +0.000] |
| capacity | nh4 MAE | +0.6516 | [+0.3788, +0.9513] |
| capacity | no3 MAE | -0.0016 | [-0.0143, +0.0120] |
| capacity | do1 MAE | -0.0087 | [-0.0170, -0.0010] |
| capacity | do2 MAE | -0.0038 | [-0.0130, +0.0043] |
| capacity | do3 MAE | +0.0093 | [+0.0024, +0.0165] |
| capacity | NH4 recall_pct pp | +12.097 | [+5.163, +18.876] |
| capacity | NH4 fpr_pct pp | +11.623 | [+6.336, +17.674] |
| capacity | DO recall_pct pp | -0.455 | [-1.667, +0.000] |
| capacity | DO fpr_pct pp | -0.154 | [-0.386, +0.037] |

2000회, seed4100. 반복 개발 검증의 탐색적 구간이며 독립 확증이나 비열등성 검정이 아니다.

## 학습량과 표본

| 군 | 구성원 | 트리 수 | 총 행 수 | 행-트리 수 | 초 |
| --- | --- | --- | --- | --- | --- |
| direct | 0 | 300 | 21280 | 6384000 | 4.160 |
| capacity | 0 | 900 | 21280 | 19152000 | 11.155 |
| split | 0 | 900 | 21280 | 12768000 | 9.347 |
| direct | 1 | 256 | 21280 | 5447680 | 3.072 |
| capacity | 1 | 768 | 21280 | 16343040 | 6.981 |
| split | 1 | 768 | 21280 | 10895360 | 8.108 |
| direct | 2 | 151 | 21280 | 3213280 | 2.103 |
| capacity | 2 | 453 | 21280 | 9639840 | 4.491 |
| split | 2 | 453 | 21280 | 6426560 | 4.642 |
| direct | 3 | 193 | 21280 | 4107040 | 2.702 |
| capacity | 3 | 579 | 21280 | 12321120 | 7.547 |
| split | 3 | 579 | 21280 | 8214080 | 4.459 |
| direct | 4 | 169 | 21280 | 3596320 | 1.224 |
| capacity | 4 | 507 | 21280 | 10788960 | 2.684 |
| split | 4 | 507 | 21280 | 7192640 | 2.950 |

capacity와 split은 합계 트리 수가 같지만 학습할 행 수와 벽시계 시간은 다르다. 클래스별 학습 행/날짜 수는 comparison.json에 기록한다.

## 사전 선별

| 비교군 | 항목 | 통과 |
| --- | --- | --- |
| control | nh4_mae_lower | 아니오 |
| control | nh4_count_mae_not_higher | 아니오 |
| control | nh4_excess_mae_not_higher | 아니오 |
| control | NH4_recall_not_lower | 예 |
| control | NH4_fpr_not_higher | 아니오 |
| control | DO_recall_not_lower | 예 |
| control | DO_fpr_not_higher | 예 |
| control | do1_mae_not_higher | 아니오 |
| control | do2_mae_not_higher | 아니오 |
| control | do3_mae_not_higher | 아니오 |
| control | increase_15min_mae_not_higher | 아니오 |
| control | increase_60min_mae_not_higher | 아니오 |
| control | increase_180min_mae_not_higher | 아니오 |
| control | decrease_15min_mae_not_higher | 예 |
| control | decrease_60min_mae_not_higher | 아니오 |
| control | decrease_180min_mae_not_higher | 아니오 |
| direct | nh4_mae_lower | 아니오 |
| direct | nh4_count_mae_not_higher | 아니오 |
| direct | nh4_excess_mae_not_higher | 아니오 |
| direct | NH4_recall_not_lower | 예 |
| direct | NH4_fpr_not_higher | 아니오 |
| direct | DO_recall_not_lower | 예 |
| direct | DO_fpr_not_higher | 예 |
| direct | do1_mae_not_higher | 예 |
| direct | do2_mae_not_higher | 아니오 |
| direct | do3_mae_not_higher | 아니오 |
| direct | increase_15min_mae_not_higher | 아니오 |
| direct | increase_60min_mae_not_higher | 아니오 |
| direct | increase_180min_mae_not_higher | 아니오 |
| direct | decrease_15min_mae_not_higher | 아니오 |
| direct | decrease_60min_mae_not_higher | 아니오 |
| direct | decrease_180min_mae_not_higher | 아니오 |
| capacity | nh4_mae_lower | 아니오 |
| capacity | nh4_count_mae_not_higher | 아니오 |
| capacity | nh4_excess_mae_not_higher | 아니오 |
| capacity | NH4_recall_not_lower | 예 |
| capacity | NH4_fpr_not_higher | 아니오 |
| capacity | DO_recall_not_lower | 아니오 |
| capacity | DO_fpr_not_higher | 예 |
| capacity | do1_mae_not_higher | 예 |
| capacity | do2_mae_not_higher | 예 |
| capacity | do3_mae_not_higher | 아니오 |
| capacity | increase_15min_mae_not_higher | 아니오 |
| capacity | increase_60min_mae_not_higher | 아니오 |
| capacity | increase_180min_mae_not_higher | 아니오 |
| capacity | decrease_15min_mae_not_higher | 예 |
| capacity | decrease_60min_mae_not_higher | 예 |
| capacity | decrease_180min_mae_not_higher | 아니오 |
| 기존 장점 보존 | normal_mae_not_above_augmented | 아니오 |
| 기존 장점 보존 | fpr_not_above_augmented | 아니오 |
| 기존 장점 보존 | violation_mae_not_above_control | 아니오 |

전체 통과: False. 자동 채택 없음. 상세 해석은 README.md에 기록한다.
