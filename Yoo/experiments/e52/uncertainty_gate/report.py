"""불확실성 신호 선별 결과를 표와 그림으로 저장한다."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
LABEL={'plain':'여유 없음','fixed':'고정 여유 0.45','original':'기존 불확실성 (kappa 10)','candidate':'날짜 분할 불확실성'}


def main(a):
    result=json.loads((a.root/'results.json').read_text())
    manifest=json.loads((a.root/'manifest.json').read_text())
    a.out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.root/'results.json',a.out/'results.json')
    d=result['decision']
    md=['# e54 불확실성 신호의 탐지와 거짓양성 선별','',
        '동일한 기록 공기량과 e35 전이를 사용한 사후 위험 판단 대조다. 정책 제어 성과를 평가한 것은 아니다. '
        '표의 cost는 95개 전이 중 NH4 위반으로 판정한 수이며, 여유를 적용한 학습 비용 진단값이다. '
        '환경 상태, 원 NH4 기준 4 mg/L, 행동과 원본 데이터는 유지했다.', '',
        f"후보 kappa={result['kappa_candidate']:g}. 후보 build가 train 날짜만으로 정한 값을 그대로 사용했다. 로컬 실행 중 kappa나 경보 문턱을 탐색하지 않았다.", '']
    for split in ['train','val']:
        z=result['splits'][split]
        md += [f'## {split}', '',f"관측 기록 NH4 비용: {z['observed_cost']:.4f}.", '',
               '| 판정 | 비용 | FPR % | 탐지율 % | 추가 경보의 실제 위반 비율 % | 추가 진양성 수 | 추가 거짓양성 수 |',
               '| --- | --- | --- | --- | --- | --- | --- |']
        for name in LABEL:
            r=z[name]
            md.append(f"| {LABEL[name]} | {r['cost']:.4f} | {r['fpr_pct']:.4f} | {r['recall_pct']:.4f} | {r['added_precision_pct']:.3f} | {r['added_true']} | {r['added_false']} |")
        md+=['', '| 불확실성 | SD 대 절대 오차 상관 | SD 대 과소예측량 상관 |', '| --- | --- | --- |']
        for name in ['original','candidate']:
            r=z[name];md.append(f"| {LABEL[name]} | {r['sd_absolute_error_spearman']:.4f} | {r['sd_underprediction_spearman']:.4f} |")
    ci=d['paired_block_ci95']
    md += ['', '## 고정한 선별 기준의 결과','',
           f"학습 날짜 비용 맞춤: {d['train_costs_matched']}. 탐지/FPR 동시 개선 선별: {d['exploratory_pass']}. 신뢰구간 방향 일치: {d['reliability_flag']}.",
           f"검증 탐지율 차이 {d['recall_diff_pp']:+.4f}%p, 7일 날짜 블록 탐색 95% 구간 [{ci['recall_diff_pp'][0]:+.4f}, {ci['recall_diff_pp'][1]:+.4f}].",
           f"검증 FPR 차이 {d['fpr_diff_pp']:+.4f}%p, 같은 구간 [{ci['fpr_diff_pp'][0]:+.4f}, {ci['fpr_diff_pp'][1]:+.4f}].", '',
           '검증 40일을 재사용한 탐색이며 독립 확증이 아니다. 블록 길이 7, 2000회, 고정 RNG 20261112이며 모델 구성원을 독립 표본으로 세지 않았다. '
           '점추정치 선별과 구간 판정을 분리했다. 선별 미통과가 정책 학습의 실패를 증명하지는 않는다. '
           '다만 불확실성 신호의 품질이 좋아졌다는 이유로 후보를 채택하거나 확대할 근거는 되지 못한다.', '',
           '## 재현과 범위','',
           f"실행 커밋 `{manifest['commit']}`, 소요 {result['elapsed_seconds']:.2f}초. 원 기록 재생의 상태와 기존 SD는 e53 감사 저장값과 배열 단위로 정확히 일치했다. 입력과 소스 해시 불변, test 평가 없음, 정책 가중치 갱신 없음.",
           f"전체 입력 해시와 원 시계열: `{a.root}`.", '',
           '![위험 탐지와 거짓양성 비교](tradeoff.png)', '']
    (a.out/'results.md').write_text('\n'.join(md))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,['train','val']):
        for name,label in [('plain','No margin'),('fixed','Fixed margin'),('original','Original SD'),('candidate','Day-block SD')]:
            r=result['splits'][split][name]
            ax.scatter(r['fpr_pct'],r['recall_pct'],s=55,label=label)
        ax.set_title(split);ax.set_xlabel('False positive rate (%)');ax.set_ylabel('Recall (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.margins(x=.15,y=.25)
    fig.suptitle('Recorded-action recursive trajectories; frozen margin scales')
    fig.tight_layout();fig.savefig(a.out/'tradeoff.png',dpi=170);plt.close(fig)
    print(d)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,default=HERE)
    main(p.parse_args())
