"""실행 상태와 실제 갱신 속도 기반 ETA를 표시한다."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path


def main(root):
    status = json.loads((root / 'status.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    total = len(manifest['methods'])*len(manifest['seeds'])
    done = sum(r['exit_code'] == 0 for r in status['completed'])
    rates = []
    remaining = []
    print(f"{status['phase']}: 완료 {done}/{total}, 실행 {len(status['running'])}, 대기 {len(status['pending'])}")
    for run in status['running']:
        path = root / 'runs' / run['method'] / f"seed_{run['seed']}" / 'progress.json'
        if not path.exists():
            print(run['method'], run['seed'], '초기화 중')
            continue
        p = json.loads(path.read_text())
        if p['phase'] == 'training':
            elapsed, u = p['elapsed_seconds'], p['update']
            rate = elapsed/u
            rates.append(rate)
            remaining.append(rate*(p['updates']-u))
            width = 24
            filled = int(width*u/p['updates'])
            bar = '#' * filled + '.'*(width-filled)
            print(f"{run['method']} seed{run['seed']} [{bar}] {u}/{p['updates']} ({100*u/p['updates']:.1f}%), 남은 {p['estimated_remaining_seconds']/60:.1f}분")
        else:
            print(run['method'], run['seed'], p['phase'])
    if rates:
        work = sum(remaining) + len(status['pending'])*sum(rates)/len(rates)*manifest['updates']
        eta = datetime.now().astimezone() + timedelta(seconds=work/manifest['parallel_jobs'])
        print('전체 예상 종료', eta.strftime('%Y-%m-%d %H:%M %Z'), '(현재 속도 추정, 최종 평가/집계 시간 제외)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
