# e76: 같은 공기 사용량에서 FCSRL 표현 학습 기여 확인

2026-11-22. 실행 준비 완료. [설계와 문헌](REFERENCES_AND_DESIGN.md).

e74에서 FCSRL의 공기는 적고 수질 비용은 높았다. 두 방법을 같은 공기 사용량에 맞췄을 때도 수질 차이가 남는지 확인한다. 새 학습 없이 e74의 FCSRL과 같은 망의 고정 표현 PPO 각각 seed 0~9를 평가한다. FCSRL 구조, 환경, 비용, 관측은 유지한다. 미래 정보와 차단은 사용하지 않는다.

dev에서 e74 ABAC의 실제 공기량을 목표로 총공기 명령 배율을 고르고 val에 그대로 적용한다. 두 방법에 같은 후보와 선택 기회를 주며 수질 비용으로 배율을 고르지 않는다. 표본 행동 평가 난수3개를 유지하고 배율1에서 원 e74 결과부터 재현한다. 맞춤 허용 범위와 실패 처리, 공동 개선 판정은 설계 문서에 고정했다. 배율을 바꾸면 정책도 바뀌므로 원 정책의 결과를 함께 보존한다.

## 준비물과 실행

- 원 정책: `/home/jio/fcsrl_src/results/e74/main/runs/`
- 원 대리 환경: `/home/jio/fcsrl_src/results/e63/prepared/`
- 기준 ABAC: `/home/jio/fcsrl_src/results/e74/baselines/baselines.json`
- 출력 예정: `/home/jio/fcsrl_src/results/e76/main/`
- Python: 기존 `/home/jio/fcsrl_src/.venv-skl191/bin/python`. 패키지 변경 없음.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/jio/fcsrl_src/.venv-skl191/bin/python Yoo/experiments/e76/run76.py \
  --out /home/jio/fcsrl_src/results/e76/main --jobs 8
```

GBM 전이와 원 e74 CPU 난수를 그대로 사용한다. 새 GPU 학습이 아니다. 8개 작업 동시 실행, 작업별 로그와 후보 수는 main/logs 및 main/jobs의 progress.json에 기록한다. 실행 당시 커밋과 준비물/코드/정책 해시는 manifest.json에 기록한다.

## 검산과 한계

dev 2일 x 구성원5 x 표본3 스모크에서 총공기 변환 경계, 실제 구간 공기 합, 비용 재계산을 통과했다. 본 실행은 각 정책의 전체 dev/val 배율1 원 집계와 첫 표본 궤적을 재현해야 다음 후보로 진행한다. 선택된 배율의 dev 재생, 정상화기 불변, 입력 해시 불변 및 최종 80개 궤적 파일의 독립 재계산을 완료 조건으로 둔다.

dev/val은 반복 사용한 개발 자료다. holdout/test rollout을 하지 않는다. 공기 맞춤 실패를 열등성이나 동등성으로 해석하지 않는다. 같은 공기량에서도 FCSRL이 낫다면 제한된 정책/배율 계열에서 표현 학습의 효용을 지지하지만 feasibility와 dynamics 손실의 기여를 분리한 것은 아니다. 현장 성능이나 에너지 kWh 절감의 증거가 아니다.
