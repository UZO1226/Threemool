"""원 probe 출력에서 작은 공개 요약과 seed별 표를 만든다."""
import argparse
import json
from pathlib import Path
import statistics


def report(source, out):
    z = json.loads((source / 'summary.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    compact = {k: v for k, v in z.items() if k != 'seeds'}
    compact['seeds'] = []
    for row in z['seeds']:
        r = json.loads(json.dumps(row))
        for target in r['targets'].values():
            for model in target.values():
                for score in model['scores'].values():
                    score.pop('daily_mae')
        compact['seeds'].append(r)
    text = ['# e57 고정 표현 진단 결과', '',
            '실제 기록 행동 아래의 관측 예측이다. 정책 제어 성능이나 새 행동의 인과 효과가 아니다.', '',
            'MAE는 작을수록 좋다. 5개는 고정 정책 Encoder seed이며 같은 날짜와 정답을 공유한다. 아래 평균을 독립 데이터 5회 검증으로 해석하지 않는다.', '']
    for target, unit in [('current_nh4', '현재 NH4, mg/L'), ('future_excess', '다음 한 시간 초과 적분, mg h/L')]:
        text += [f'## {unit}', '', '| 읽는 입력과 예측기 | train 내 probe holdout MAE | 개발 val MAE |', '| --- | --- | --- |']
        names = z['seeds'][0]['targets'][target]
        for name in names:
            v = {split: statistics.mean(r['targets'][target][name]['scores'][split]['mae'] for r in z['seeds']) for split in ['holdout', 'val']}
            text.append(f"| {name} | {v['holdout']:.6f} | {v['val']:.6f} |")
    text += ['', '## 미래 초과 적분 seed별 MAE', '',
             '| seed | raw holdout | raw + trained z holdout | raw + initial z holdout | raw + trained z val |', '| --- | --- | --- | --- | --- |']
    for r in z['seeds']:
        t = r['targets']['future_excess']
        vals = [t[k]['scores'][s]['mae'] for k, s in [('raw', 'holdout'), ('raw_plus_trained_z', 'holdout'), ('raw_plus_initial_z', 'holdout'), ('raw_plus_trained_z', 'val')]]
        text.append('| ' + str(r['seed']) + ' | ' + ' | '.join(f'{v:.6f}' for v in vals) + ' |')
    text += ['', f"선형 probe 사이에서 정한 학습 표현 이점 조건은 {z['trained_z_advantage_seeds']}/5 seed가 만족했다. 그러나 비선형 raw 대조와 지속성 기준이 더 낮은 오차를 보여, 이 조건을 네트워크 확장의 제어 성능 근거로 사용하지 않는다.", '',
             'NH4 정규화 잘림은 모든 분할에서 0개다. 현재 NH4 역변환 최대 오차는 약 4.8e-7 mg/L로 원 자료의 float32 변환 오차 범위다. 원 관측 prefix와 정책 입력의 일치, 모든 정책 가중치 및 입력 해시 불변, 날짜 분리와 미래 창 인덱스를 확인했다.', '',
             '정책 Encoder는 train188일 전체를 이미 학습했다. holdout은 probe만 학습하지 않은 38일이며 Encoder 미학습 날짜 일반화 검사가 아니다. val은 여러 연구에서 반복 사용한 개발 자료다. 미래 4개 관측으로 라벨을 만들었지만 입력에는 현재와 과거 관측, 현재 기록 행동만 쓴다. test 행은 추출하거나 평가하지 않았다.', '']
    (out / 'results.md').write_text('\n'.join(text))
    (out / 'summary.json').write_text(json.dumps(compact, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report(a.source, a.out)
