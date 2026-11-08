"""Windows 에서 원본 FCSRL 패키지를 가져오기 위한 호환 모듈 (Lee 작성).

원본 czp16/FCSRL 은 `fcsrl/__init__.py` 에서 패키지 전체를 가져오고, 그 안의
`fcsrl/env/gym_utils.py` 가 safety_gymnasium 을 요구한다. safety_gymnasium 은
Windows 에서 pygame 빌드가 실패해 설치할 수 없다. 이 실험은 BSM1 물리 환경만
사용하므로 vision wrapper 와 Safety Gymnasium 환경을 전혀 쓰지 않는다.

그래서 원본 코드를 고치는 대신, 가져오기 전에 사용하지 않는 이름 하나
(VisionEnvConf)만 빈 자리표시자로 등록한다. 원본 저장소 파일은 수정하지 않는다
(커밋 d882b91054d52c7a63a18a21f7db67bbbed82b63 그대로).

`import fcsrl_compat` 를 fcsrl 관련 import 보다 먼저 실행해야 한다.
Safety Gymnasium 과제를 돌리려면 이 모듈을 쓰지 말고 Linux 에서 원본 환경을
설치해야 한다 (Yoo/e01 재현 환경).
"""
import sys
import types


def install_stub():
    if 'safety_gymnasium' in sys.modules:
        return False
    package = types.ModuleType('safety_gymnasium')
    bases = types.ModuleType('safety_gymnasium.bases')
    underlying = types.ModuleType('safety_gymnasium.bases.underlying')

    class VisionEnvConf:
        """원본 vision wrapper 전용 자리표시자. 이 실험에서는 호출하지 않는다."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError('Safety Gymnasium vision config is not available in this environment')

    underlying.VisionEnvConf = VisionEnvConf
    bases.underlying = underlying
    package.bases = bases
    package.__fcsrl_compat_stub__ = True
    sys.modules.update({'safety_gymnasium': package, 'safety_gymnasium.bases': bases,
                        'safety_gymnasium.bases.underlying': underlying})
    return True


installed = install_stub()
