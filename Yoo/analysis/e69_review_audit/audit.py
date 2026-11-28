"""Read existing logs only; do not evaluate or alter running experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def compare(old, new):
    n = min(len(old), len(new), 250)
    differences = []
    for i in range(n):
        keys = (set(old[i]) | set(new[i])) - {'seconds'}
        for key in sorted(keys):
            if key not in old[i] or key not in new[i] or old[i][key] != new[i][key]:
                differences.append([i + 1, key])
    return dict(compared_updates=n, mismatched_fields=differences,
                complete_prefix_match=n == 250 and not differences)


def audit(root, previous):
    manifest = json.loads((root / 'manifest.json').read_text())
    changed = [p for p, h in manifest['hashes'].items()
               if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    commands = dict(manifest['commands'])
    expected = {f'{method}_s{seed}' for method in ('fcsrl', 'ppolag') for seed in range(10)}
    wrong_commands = sorted(expected ^ set(commands))
    for name, cmd in commands.items():
        if '--updates' not in cmd or cmd[cmd.index('--updates') + 1] != '500':
            wrong_commands.append(name)
    rows = []
    for name in sorted(expected):
        p = root / 'runs' / name / 'train_log.json'
        if not p.exists():
            rows.append(dict(run=name, status='not_logged'))
            continue
        try:
            new = json.loads(p.read_text())
        except json.JSONDecodeError:
            rows.append(dict(run=name, status='write_in_progress'))
            continue
        old = json.loads((previous / 'runs' / name / 'train_log.json').read_text())
        rows.append(dict(run=name, status='checked', logged_updates=len(new),
                         metrics_exist=(p.parent / 'metrics.json').exists(), **compare(old, new)))
    bad = bool(changed or wrong_commands or any(r.get('mismatched_fields') for r in rows))
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), code_commit=manifest['code_commit'],
                root=str(root), previous=str(previous), changed_manifest_files=changed,
                invalid_commands=wrong_commands, full_prefix_matches=sum(r.get('complete_prefix_match', False) for r in rows),
                total_expected=20, audit_failed=bad, runs=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/jio/fcsrl_src/results/e69/main'))
    parser.add_argument('--previous', type=Path, default=Path('/home/jio/fcsrl_src/results/e68/main'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.previous)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, ensure_ascii=False))
    raise SystemExit(1 if result['audit_failed'] else 0)
