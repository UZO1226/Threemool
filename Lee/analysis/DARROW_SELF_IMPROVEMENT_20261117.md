# e63 이후의 자체 개선 목표

목표는 인증된 우리 e63보다 같은 실제 공기량에서 더 적은 수질 위반을
보이는 확장이다. 기본 FCSRL은 보조 비교로 남긴다. e64가 정책 seed0~4의
체크포인트, 위험 모델, 고정 환경, seed별 공기량 보정값과 기존 독립 검사,
최종 보고서 재계산을 인증했다. 기존 목표의 완료 기록은 보존한다.

새 성공 기준은 seed별 공기량 일치를 모두 만족하고, e63 대비 비용과 NH4
차이의 paired policy-seed bootstrap95% 구간 상한이0보다 작으며 DO 상한은
0 이하인 것이다. 기존10000회 재표집과 RNG2026을 유지한다. 반복 사용한
validation40일은 탐색 결과다. test41일은 추가로 사용하지 않는다.

## 완료한 실험과 다음 가설

| 실험 | 사전 고정 가설 | 현재 확인한 근거 |
| --- | --- | --- |
| e65 | 같은 위험 모델과 반경0.25에서 행동 후보를5개에서17개로 세분화 | 5개 seed의 공기량 맞춤과 독립 검사 완료. 비용 구간만 통과하고 NH4, DO 구간은 실패. 채택하지 않음 |
| e66 | 현재 유입량, 수온, sin, cos로4단계 공동 잡음 블록의32개 이웃 선택 | 독립 경로, 전체 하루 재계산과 사건 평가 완료. 채택 기준7개 모두 실패. 기존 환경 유지 |
| e67 | 같은 위험 모델에서 행동 반경0.5와9개 후보 사용 | 5개 seed 비교와 독립 검사 완료. DO만 개선하고 비용, NH4의 공동 기준 실패. 실패 결과와 모델, 로그 보존 완료 |
| e68 | 새 정책 seed5~9의 기준선 준비 | 모든 seed의401280단계 학습, 원본과 관측 기록 버전 일치 검사, 위험 모델 준비, 실제 e63 보정 정책 재구성과 독립 인증 완료. 후보의 성능 평가 아님 |
| e69 | 학습 상태에서 e63과 평균 행동을 맞추는 공기 예산 보정 | 전체5개 seed 비교와 독립 검사 완료. DO만 개선해 공동 기준 실패. e67 보조 비교의 비용, NH4 개선을 e63 대비 성공으로 해석하지 않음 |
| e70 | 기록된 상태의 평균 제약과 실제 운전 공기량의 차이 분리 | 실패한 e69와 기존 e63의 native 재현과 진단 완료. 같은 상태의 행동 변화는 공기량을 늘리고 방문 상태 변화는 일부 상쇄. 새 기법의 성능 결과 아님 |
| e71 | 확장 예측 이득이 있는 상태에서만 공기 예산 보정 | 기존 seed0~4 통과했으나 새 seed5~9는 DO만 개선하고 비용, NH4 공동 기준 실패. 모든 독립 검사와 자료 보존 완료. 목표는 계속 진행 |
| e72 | e71의 NH4, DO 예측 위험 교환 진단 | 기존과 노출된 새 seed10개의 분리 구간 진단 완료. 성분별 위험 교환과 양쪽 비증가 후보가 확인됐으나 새 기법의 수질 개선 증거는 아님 |
| e73 | 성분별 예측 위험을 높이는 후보를 제외한 공기 예산 보정 | 기존 seed5개의 학습 내부 보정과 분리 구간 검사 완료. 실행 재현 후 독립 검사 중이며 실제 공기량을 맞춘 수질 개선은 아직 미검증 |

e65는 환경을 수정하지 않는다. e66은 e40 양자 회귀 모델과 alpha0.5를
유지하고 잡음 분포만 바꾸며, 공기량이나 농도로 잡음을 선택하지 않는다.
동일한 잡음을 사용한 공기량 반응 검사를 별도로 요구한다. 어떤 후보도
기존 e65, e67, e69는 새 개선 기준을 통과하지 않았다. e71은 기존 seed0~4에서
공동 기준과 독립 검사를 통과했으나, 별도 새 seed5~9에서 공동 기준을 통과하지 못했다.
기존 e42 환경은 교체하지 않았다.

e66의 날짜를 하나씩 제외한 calibration 진단에서 조건부 방식은 NH4
예측 빈도4.990351%, 관측5.508772%다. DO 예측2.211404%, 관측10.736842%로
큰 과소 예측이 남는다. 전체 후보 블록 대조군은 NH4 7.738596%, DO
1.397368%다. 조건부 DO Brier .111508은 대조군 .107569보다 높다.
이는 학습 내 진단이며 검증 개선이나 독립 확인을 뜻하지 않는다.

## 재현

workspace에서 threemool Python 환경을 사용한다. OMP, OPENBLAS, MKL,
NUMBA 스레드는 각각1이며 DARROW_FAST_DYNAMICS=1이다. 원본 DARROW pickle,
기존 e42, e55, e56, e63 모델과 로그를 기존 체크섬에 맞게 복원해야 한다.
모든 출력 경로는 기존 경로를 덮어쓰지 않는다. 재현할 때 출력 위치를
새로 지정하고 runner의 고정 작업 경로도 분리된 workspace에서 사용한다.

e65의 실제 실행 순서는 preflight.py, compare.py의 native-only,
run_study.py다. runner는 native 독립 검사와 항등 대조군을 먼저 통과해야
5개 seed 비교, 각각의 native와 matched 독립 검사, 기존 e44 보고서
생성을 진행한다. baseline_native 필드는 인증된 matched e63을 뜻한다.
원래 e63 native나 원래 NN에 맞추는 방식으로 바꾸지 않는다.

```powershell
python Threemool/Lee/experiments/e65/run_study.py --workspace C:/Users/user/Downloads/IoT_project2
```

e66 사전 검사는 validation과 test를 평가하지 않는다.

```powershell
python Threemool/Lee/experiments/e66/check_blocks.py --workspace C:/Users/user/Downloads/IoT_project2 --freeze Threemool/Lee/experiments/e42/environment_freeze.json --out results/e66_block_preflight.json
python Threemool/Lee/experiments/e66/evaluate_blocks.py --workspace C:/Users/user/Downloads/IoT_project2 --freeze Threemool/Lee/experiments/e42/environment_freeze.json --preflight results/e66_block_preflight.json --out results/e66_exogenous_blocks
python Threemool/Lee/experiments/e66/audit_blocks.py --workspace C:/Users/user/Downloads/IoT_project2 --freeze Threemool/Lee/experiments/e42/environment_freeze.json --preflight results/e66_block_preflight.json --runs results/e66_exogenous_blocks --out results/e66_exogenous_blocks/audit.json
```

e66은 원래 point gate와 e40 대비 NH4, DO Brier 비열등성, 유지 예측 대비
DO Brier 비열등성, DO AP와 AUC, 새로운 DO 위반 발생 시점 탐지를 모두
요구한다. 날짜 단위5일 원형 블록 bootstrap을 기존과 동일하게 사용한다.
어느 실패도 숨기거나 다른 지표로 대신하지 않는다. 채택하지 않은 환경에서
에이전트 성능을 e42 기준 성능과 직접 비교해 개선이라고 주장하지 않는다.

실행 상태는 결과 파일만으로 판단하지 않는다. 프로세스와 tool session을
확인하고, 종료 전에는 동일한 실험을 다시 시작하지 않는다. 새 실험에는
새로운 가중치 학습이 없으며 기존 전체401280단계 학습 seed5개를 사용한다.
초기 후보가 통과하면 가능한 새 정책 학습 seed로 추가 확인이 필요하다.

## e71 초기 개선과 별도 확인

e71은 확장 후보가 기존 후보보다 예측 위험을 낮출 가능성이 있는 상태에서만
공기 예산 보정을 적용한다. 그 밖의 상태에서는 e63 행동을 그대로 유지한다.
기존 seed0~4에서 같은 공기량의 평균 비용20.089750, NH4 위반20.900132%,
DO 위반.246974%로 개선했고, 사전 지정 구간과 모든 독립 검사를 통과했다.
[전체 초기 결과](../experiments/e71/results.md)와 실패 기록을 함께 보존한다.
실제 새 seed5~9의 e63 기준선 인증과 e71 학습 내부 검사는 완료했다.
기법, 판정 기준과 정책 코어를 고정한 별도 비교와 독립 검사는 완료했다.
비용 차이의95% 구간[-1.142875,.460375], NH4 차이[-1.095921,.793553]이0을
포함해 공동 기준을 통과하지 못했다. DO는[-.416316,-.044474]로 개선했다.
[재현 실패 결과](../experiments/e71/fresh_results.md)와 모델, 로그를 모두 보존한다.
기존 구간과 합쳐 실패를 숨기지 않으며, 이후 기법에서 이 새 seed들은 이미
개발 결과에 노출된 구간이라고 명시한다. e72는 성분별 예측 위험 교환을 진단한다.
반복 사용한 validation40에서의 대리 환경 결과이며 목표는 아직 진행 중이다.
