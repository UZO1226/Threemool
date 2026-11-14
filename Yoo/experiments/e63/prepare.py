"""e63 준비물: e50 준비물에서 대리 모델만 e51 DO 대응 수정 모델로 바꾼 사본."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

E50 = Path('/home/jio/fcsrl_src/results/e50/prepared')
E51 = Path('/home/jio/fcsrl_src/results/e51/main/corrected_surrogate.pkl')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    for name in ('data.pkl', 'normalizer.json', 'prior.pt', 'data_manifest.json', 'preparation.json'):
        shutil.copy2(E50 / name, a.out / name)
    shutil.copy2(E51, a.out / 'surrogate.pkl')
    manifest = {name: dict(source=str((E51 if name == 'surrogate.pkl' else E50 / name)), sha256=sha(a.out / name))
                for name in ('data.pkl', 'normalizer.json', 'prior.pt', 'surrogate.pkl')}
    assert all(sha(E50 / n) == manifest[n]['sha256'] for n in ('data.pkl', 'normalizer.json', 'prior.pt'))
    assert sha(E51) == manifest['surrogate.pkl']['sha256']
    (a.out / 'e63_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
