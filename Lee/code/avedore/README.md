# Lee/code/avedore

계획서의 Kaggle 폐수처리 데이터(Avedøre)에서 Yoo 가 만든 PPO-Lag 와 FCSRL + PPO-Lag 파이프라인을
그대로 쓰기 위한 코드다. 대부분 Yoo 파일의 수정 없는 복사본이고 Lee 가 쓴 파일은 셋이다.

기준 커밋은 `d1ea573db7225b38cb25fe02bcbfd19a39d25f9b` (main). sha256 앞 16자리는 다음과 같다.

| 파일 | 출처 | sha256 |
| --- | --- | --- |
| `ppo_config.py` | `Yoo/code/wwtp/ppo_config.py` | `7236f11934ee8bca` |
| `ppo_lag.py` | `Yoo/code/wwtp/ppo_lag.py` | `b55da69b3b04746b` |
| `ppo_fcsrl.py` | `Yoo/code/wwtp/ppo_fcsrl.py` | `3bb56d34b3e47e12` |
| `ppo_tail.py` | `Yoo/code/wwtp/ppo_tail.py` | `9a1452736d03ecaf` |
| `env.py` | `Yoo/code/wwtp/env.py` | `c4bc095d4a9ee941` |
| `surrogate.py` | `Yoo/code/wwtp/surrogate.py` | `552a491fec8f4c5f` |
| `evaluate.py` | `Yoo/code/wwtp/evaluate.py` | `87ecbfb5420275cc` |
| `prep.py` | `Yoo/code/wwtp/prep.py` | `24c0b3cb2934bcf3` |
| `FCSRL_LICENSE` | `Yoo/code/wwtp/FCSRL_LICENSE` | `c71d239df91726fc` |
| `yoo_e06_train_ppo.py` | `Yoo/experiments/e06/train_ppo.py` | `5007604d55a8522b` |

대리 모델 가중치 `Yoo/experiments/e02/surrogate.json` (sha256 `d25dbe3ea1a8e33b...`)은 저장소의 파일을
그대로 쓰며 이 폴더에 복사하지 않았다.

## Lee 가 쓴 파일

| 파일 | 내용 |
| --- | --- |
| `train_av.py` | `yoo_e06_train_ppo.py` 에 방법 `ppolag_fcsrl_fpen` 과 인자 `--fpen-weight`, `--fpen-relative` 를 더한 학습 스크립트. 나머지는 Yoo 코드를 따른다 |
| `feasibility_policy.py` | feasibility 점수를 정책 이점의 벌점으로 쓰는 확장 (Lee/e05 와 같은 아이디어에 크기를 이점의 weight 배로 맞추는 `relative` 옵션 추가) |
| `fcsrl_compat.py` | Windows 에서 `safety_gymnasium` 없이 원본 FCSRL 을 가져오기 위한 자리표시자 |

원본 FCSRL(czp16/FCSRL, d882b91, Apache-2.0)은 수정하지 않는다.
