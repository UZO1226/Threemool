# Lee/code/bsm

BSM1 물리 환경에서 안전 비용 정의를 바꿔 비교하기 위한 코드다. 대부분은 Yoo 가 이미
만든 파일을 그대로 복사했고, Lee 가 새로 쓴 파일은 두 개다 (BRANCH_RULES 4절, 다른
사람 폴더의 코드가 필요하면 자기 폴더로 복사하고 출처와 커밋을 적는다).

## Yoo 에게서 복사한 파일 (내용 수정 없음)

복사 기준 커밋은 `ac0e9bf50a939dc05cd3e385122d0ec70477906b` (main) 이다.

| 파일 | 출처 경로 | sha256 앞 16자리 |
| --- | --- | --- |
| `ppo_config.py` | `Yoo/code/wwtp/ppo_config.py` | `9e11cf4eec433601` |
| `ppo_lag.py` | `Yoo/code/wwtp/ppo_lag.py` | `79e9fb579d986659` |
| `ppo_fcsrl.py` | `Yoo/code/wwtp/ppo_fcsrl.py` | `56020b2dd8c5477b` |
| `ppo_tail.py` | `Yoo/code/wwtp/ppo_tail.py` | `1a865fe264831f6f` |
| `env.py` | `Yoo/code/wwtp/env.py` | `53adb88d86de302d` |
| `surrogate.py` | `Yoo/code/wwtp/surrogate.py` | `02bc4716bb328678` |
| `FCSRL_LICENSE` | `Yoo/code/wwtp/FCSRL_LICENSE` | `1eb85fc97224598d` |
| `bsm_env.py` | `Yoo/experiments/e16/bsm_env.py` | `a5758c914464ab8c` |
| `BSM_LICENSE` | `Yoo/experiments/e16/BSM_LICENSE` | `979dfed37e932576` |
| `bsm_mean.py` | `Yoo/experiments/e17/bsm_mean.py` | `5dcf36a55a7420bb` |

`env.py` 와 `surrogate.py` 는 이 실험에서 쓰지 않는다. `ppo_lag.py` 가 모듈을 가져올 때
`from env import AerationEnv` 를 실행하므로 가져오기를 성립시키려고 함께 복사했다.
`Lee/experiments/e01/verify.py` 가 복사본과 Yoo 원본의 sha256 일치를 검사한다.

## Lee 가 새로 쓴 파일

| 파일 | 내용 |
| --- | --- |
| `bsm_binary.py` | 이번 실험의 변경 한 가지. 순간 위반 지시 비용(0 또는 1)과 한도, 평가 지표 |
| `fcsrl_compat.py` | Windows 에서 원본 FCSRL 을 가져오기 위한 호환 자리표시자 |

## 원본 구현체

원본 FCSRL 은 `https://github.com/czp16/FCSRL` 커밋
`d882b91054d52c7a63a18a21f7db67bbbed82b63` 를 수정 없이 사용한다 (Apache-2.0,
`FCSRL_LICENSE`). Yoo/e01 과 e17 이 고정한 커밋과 같다. 저장소에 포함하지 않고
설치해서 쓴다. 설치 방법은 `Lee/experiments/e01/README.md` 에 있다.

BSM1 모델과 유입 자료의 출처, 라이선스, 체크섬은 `BSM_LICENSE` 와
`Yoo/experiments/e16/README.md` 를 따른다.
