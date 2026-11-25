# e134 같은 공기량에서 Actor 보정의 수질 비교

e133의 기본 FCSRL, 표현 갱신 후 Actor 보정, Actor 추가 PPO를 같은 공기 사용량에서 비교한다. 원 정책에서 보정은 기본보다 평균 공기를 10.56% 덜 썼고 NH4 위반이 102.09분 늘었다. 이 진단에서는 세 방법의 결정적 정책을 고정한 채 학습 기록의 공기량에 맞춘다. [원 정책 결과](../e133/README.md), [평가 전 판정 기준](REFERENCES_AND_DESIGN.md)

## 재현

사전 설계 e947356, 평가 코드 dd317d3. Kaggle DARROW 공통 v1.1, 학습 188일, 겨울 개발 40일, 시드 10~14. 원 정책은 서버 `/home/jio/fcsrl_src/results/e133/main`, e50 준비물은 `/home/jio/fcsrl_src/results/e50/prepared`에 있다. [입력 및 정책 지문, 실행 환경](../e133/README.md)

저장된 마지막 정책 15개를 e133과 같은 CPU 환경에서 사용한다. 11개 관측에 대해 총공기 명령을 내고 기록 비율에 따라 5개 구역에 배분한다. 총 95개 전이에서 매 15분 실제 공급률을 적분한다. 시드별 하루 수치는 40일과 GBM 구성원 5개에 같은 가중치를 준 평균이다. 날짜별 표에서는 같은 날의 구성원 5개를 먼저 평균한다.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
export PYTHONPATH=/home/jio/fcsrl_src/results/e113/accel-deps
py=/home/jio/fcsrl_src/.venv-skl191/bin/python
exp=Yoo/experiments/e134
out=/home/jio/fcsrl_src/results/e134/replay
"$py" "$exp/run.py" "$out" --jobs 6
"$py" "$exp/report.py" "$out" "$out/tables"
```

원 실행 자료는 `/home/jio/fcsrl_src/results/e134/main`에 있다. `selection.json`에는 모든 학습 후보의 공기량과 고정한 배율, 각 조건의 NPZ에는 겨울 200개 경로의 공기 및 상태가 있다.
