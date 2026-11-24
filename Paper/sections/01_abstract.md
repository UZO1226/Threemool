<!-- owner: none -->
## Abstract

We evaluated safe reinforcement learning for aeration control using records from the Tilburg wastewater treatment plant. Across five training seeds, correcting the FCSRL policy output reduced mean air use by 5.73%, but lowered violation cost in only two paired runs. On a separate fixed surrogate, a risk-based correction reduced cost from 22.46 to 21.06 at matched air use. The corrected policy still exceeded its cost limit. Errors in winter ammonium predictions limit the transfer of these simulation results to plant operation.
<!-- exp: Yoo/e50 results.md, Lee/e63 results.md, Yoo/e104 README.md, snapshot 2f34c13 -->

## 요약

하수처리장의 폭기 제어에서는 공급 공기를 줄이는 동안 암모늄 농도를 관리해야 한다. 본 연구는 공개 안전 강화학습 기법 FCSRL을 재현한 뒤, 네덜란드 Tilburg 처리장 운전 기록에 기반한 대리 환경에서 정책 보정의 효과를 평가하였다. 같은 조건에서 학습한 다섯 시드의 평균을 비교하면 정책 출력 보정은 기본 FCSRL보다 공기를 5.73% 적게 사용했지만, 동일 시드에서 수질 위반 비용이 낮아진 경우는 둘이었다. 별도의 고정 환경에서 후보 행동의 위험을 평가해 명령을 보정하자, 공기량을 맞춘 뒤에도 평균 비용이 22.46에서 21.06으로 감소했다. 이때도 목표 한도 12는 넘었다. 동일한 차단을 적용한 단기 예측 제어는 학습 정책보다 낮은 비용을 보였다. 겨울철 폭기 증가 뒤 관측된 암모늄 하락을 대리 모델이 재현하지 못한 결과는 현장 적용에 앞서 공정 반응과 정책 성능을 각각 검증할 필요가 있음을 보여준다.
<!-- exp: Yoo/e01 README.md, Yoo/e50 results.md, Lee/e63 results.md, Park/e66 results.md, Yoo/e104 README.md, snapshot 2f34c13 -->
