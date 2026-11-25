#!/usr/bin/env bash
set -eu -o pipefail
out=${1:?supply an unused output directory}
jobs=${2:-6}
py=/home/jio/fcsrl_src/.venv-skl191/bin/python
exp=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=/home/jio/fcsrl_src/results/e113/accel-deps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
mkdir -p "$(dirname "$out")"
mkdir "$out"
"$py" "$exp/trial.py" "$out" --check > "$out/preflight.log" 2>&1
git -C "$exp" rev-parse HEAD > "$out/commit"
for seed in 10 11 12 13 14; do
    for method in fcsrl fcsrl_actor actor_extra_ppo; do
        printf '%s %s\n' "$method" "$seed"
    done
done > "$out/jobs.txt"
export out py exp
xargs -P "$jobs" -n 2 bash -c '
    "$py" -u "$exp/trial.py" "$out" --method "$1" --seed "$2" > "$out/$1-$2.log" 2>&1
' worker < "$out/jobs.txt"
"$py" "$exp/report.py" "$out" --tables "$out/tables"
