# Lee/code/darrow

Kaggle 대회 "WWTP model: mechanistic, data-driven, or hybrid?" (DARROW, Tilburg 처리장) 데이터로 폭기 제어
Safe RL 환경을 만들고 Yoo 의 PPO-Lag, FCSRL + PPO-Lag 구현을 그대로 쓰기 위한 코드다.

## 데이터

| 항목 | 내용 |
| --- | --- |
| 출처 | Kaggle 대회 `dynamic-modeling-of-wastewater-treatment-process` ("WWTP model: mechanistic, data-driven, or hybrid?", DARROW 프로젝트, IWA 하이브리드 모델링 그룹) |
| 받은 파일 | `dynamic-modeling-of-wastewater-treatment-process.zip` (사용자가 2026-11-04 에 내려받음), sha256 `cdbbc915c783919fc7306797f2293b8ac30d7a13d921aa3ce61b6fb8330cf3b7` |
| 쓰는 파일 | `train_val_test_online_dataset.csv` (15분 간격, 2021-05-05 에서 2022-01-30, 25,921 행, 결측과 공백 없음), `train_val_test_dynamic_influent.csv` (ASM1 유입 성분) |
| 쓰지 않는 파일 | `evaluation_*` (대회 제출용이라 DO, NH4, NO3 정답이 없음), `*_lab_dataset.xlsx` |
| 이용 조건 | 대회 약관을 확인하지 못했다. 수업 과제의 연구 목적이며 재배포하지 않는다. 약관 확인이 필요하다 |

유입 파일의 유량 Q_in(m3/d)과 온라인 자료의 Q_inf(m3/h)는 시차 0 에서 상관 0.9984 로 시간이 맞는 것을 확인했고,
14.4분 간격인 유입 성분을 15분 격자에 선형 보간해 입력으로 쓴다.

## Yoo 에게서 복사한 파일 (수정 없음)

기준 커밋 `d1ea573db7225b38cb25fe02bcbfd19a39d25f9b` (main). `ppo_config.py`, `ppo_lag.py`, `ppo_fcsrl.py`,
`ppo_tail.py`, `env.py`, `surrogate.py`, `FCSRL_LICENSE` 는 `Yoo/code/wwtp/` 의 같은 이름 파일이고 sha256 은
`Lee/code/avedore/README.md` 와 같다. `env.py` 와 `surrogate.py` 는 이 실험에서 쓰지 않지만 `ppo_lag.py` 가
가져올 때 필요해서 함께 둔다. 원본 FCSRL(czp16/FCSRL, d882b91, Apache-2.0)은 수정하지 않는다.

## Lee 가 쓴 파일

| 파일 | 내용 |
| --- | --- |
| `prep_darrow.py` | 온라인 자료와 유입 성분을 하루(96행) 단위로 정리하고 날짜 순서로 70%, 15%, 15% 분할 |
| `surrogate_dw.py` | 상태 전이 대리 모델. MLP 앙상블과 단조 제약 GBM 앙상블을 학습하고 비교 |
| `env_dw.py` | 벡터화한 폭기 제어 환경과 일괄 평가 |
| `calibrate_dw.py` | 실제 운전을 대리 모델에서 재생해 학습 구간 기준 비용 한도를 정함 |
| `train_dw.py` | Yoo/e06 `train_ppo.py` 의 수집, 갱신, 선택 흐름을 이 환경으로 옮긴 학습 스크립트 |
| `feasibility_policy.py`, `fcsrl_compat.py` | Lee/e06 에서 쓴 것과 같다 |
